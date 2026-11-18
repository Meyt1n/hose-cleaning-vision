"""Existing R19 detector + R18 Hand5, used as independent 2D evidence only.

No inferred anatomical side, fabricated depth, connection state or hose identity.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def tensor_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        a = state[k].detach().cpu().contiguous().numpy()
        h.update(str(k).encode()); h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _checked_tensor_file(base, entry):
    path = (base/entry['file']).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    if h.hexdigest() != entry['sha256']:
        raise ValueError('glove model SHA mismatch: ' + path.name)
    # Full mutable states are our own frozen, SHA-verified training artifacts.
    trusted = entry.get('format') == 'trusted_full_mutable_checkpoint'
    return torch.load(path, map_location='cpu', weights_only=not trusted)


def map_crop_points(points, crop_xyxy, offset_xy=(0, 0)):
    roi = np.asarray(crop_xyxy, dtype=np.float32)
    return np.asarray(points, dtype=np.float32)/256*(roi[2:]-roi[:2])+roi[:2]+offset_xy


def mediapipe_hand_evidence(features, mode='full'):
    hands = []
    start = 99 if mode == 'full' else 57
    for index in range(2):
        pts = np.asarray(features[start+index*63:start+(index+1)*63]).reshape(-1, 3)
        if pts.shape == (21, 3) and np.isfinite(pts).all() and np.count_nonzero(pts) > 15:
            hands.append({'xy_normalized': pts[:, :2].tolist(), 'source': 'original_mediapipe'})
    return {'present': bool(hands), 'fresh': True, 'observation_state': 'observed',
            'hands': hands, 'source': 'original_mediapipe', 'actual_connection': None}


class GloveHandEvidence:
    def __init__(self, manifest_path, device='cpu', score_threshold=.5, roi_normalized=None):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from common.native_hand5_model import Hand5Pose
        manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        base = manifest_path.parent
        self.device, self.threshold, self.roi = device, float(score_threshold), roi_normalized
        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=640, max_size=896,
            rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=256,
            box_nms_thresh=.5, box_detections_per_img=6)
        self.detector.load_state_dict(_checked_tensor_file(base, self.manifest['coco_parent']), strict=True)
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(self.detector.roi_heads.box_predictor.cls_score.in_features, 2)
        delta = _checked_tensor_file(base, self.manifest['detector'])
        if 'model' in delta:
            delta = delta['model']
        state = self.detector.state_dict()
        if set(delta) != {k for k in state if not k.startswith('backbone.body.')}:
            raise ValueError('R19 mutable key contract mismatch')
        state.update(delta)
        self.detector.load_state_dict(state, strict=True)
        self.pose = Hand5Pose()
        if 'pose' in self.manifest:
            state = _checked_tensor_file(base, self.manifest['pose'])
            self.pose.load_state_dict(state.get('model', state), strict=True)
        else:
            state = _checked_tensor_file(base, self.manifest['pose_parent'])
            self.pose.load_state_dict(state, strict=True)
            delta = _checked_tensor_file(base, self.manifest['pose_delta'])['delta']
            state.update(delta)
            self.pose.load_state_dict(state, strict=True)
        expected = self.manifest.get('canonical_pose_tensor_digest')
        if expected and tensor_digest(self.pose.state_dict()) != expected:
            raise ValueError('canonical pose tensor digest mismatch')
        for model in (self.detector, self.pose):
            model.requires_grad_(False).eval().to(device)
        self.detector_forward_calls = self.pose_forward_calls = self.pose_hand_crops = 0

    @torch.inference_mode()
    def predict(self, bgr, include_action_features=False, rgb=None):
        started = time.perf_counter()
        height, width = bgr.shape[:2]
        origin = np.array([0, 0], dtype=np.float32)
        image = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]) if rgb is None else rgb)
        if image.size != (width, height):
            raise ValueError('native RGB and display BGR geometry mismatch')
        if self.roi:
            x, y, w, h = self.roi  # explicit camera region, not a hose orientation prior
            box = [round(x*width), round(y*height), round((x+w)*width), round((y+h)*height)]
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError('invalid normalized camera ROI')
            image = image.crop(box)
            origin = np.array(box[:2], dtype=np.float32)
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().to(self.device)/255
        out = self.detector([x])[0]
        self.detector_forward_calls += 1
        keep = out['scores'] >= self.threshold
        boxes = out['boxes'][keep].cpu().numpy()
        scores = out['scores'][keep].cpu().numpy()
        rois, crops = [], []
        for box in boxes:
            center = (box[:2]+box[2:])/2
            side = max(box[2:]-box[:2])*1.25
            roi = np.r_[center-side/2, center+side/2].round().astype(int)
            if np.any(roi[2:] <= roi[:2]):
                continue
            rois.append(roi)
            crops.append(np.array(image.crop(tuple(roi)).resize((256, 256), Image.Resampling.BILINEAR)))
        hands = []
        action_features = np.zeros((6, 816), np.float32)
        action_valid = np.zeros(6, bool)
        if crops:
            x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.device)
            x = (x-x.new_tensor([123.675, 116.28, 103.53])[None, :, None, None])/x.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
            feat = self.pose.backbone(x)
            sx, sy = self.pose.head(feat)
            self.pose_forward_calls += 1
            self.pose_hand_crops += len(crops)
            pooled = feat[-1].mean((2, 3)).cpu().numpy() if include_action_features else None
            xy = torch.stack((sx.argmax(-1), sy.argmax(-1)), -1).float().cpu().numpy()/2
            for i, roi in enumerate(rois):
                points = map_crop_points(xy[i], roi, origin)
                source_box = boxes[i]+np.tile(origin, 2)
                if include_action_features:
                    geometry = np.r_[(xy[i]/256-.5).reshape(-1),
                                     source_box/[width, height, width, height], scores[i], 1.]
                    action_features[i] = np.r_[pooled[i], geometry]
                    action_valid[i] = True
                hands.append({'candidate_index': i, 'box_xyxy': (boxes[i]+np.tile(origin, 2)).tolist(),
                              'detection_score': float(scores[i]), 'xy_source': points.tolist(),
                              'xy_normalized': (points/[width, height]).tolist(),
                              'anatomical_side': None, 'visibility': None, 'z': None,
                              'source': 'R19_R18'})
        result = {'present': bool(hands), 'fresh': True, 'observation_state': 'observed',
                'hands': hands, 'source': 'R19_detector_R18_Hand5',
                'elapsed_ms': (time.perf_counter()-started)*1000,
                'actual_connection': None, 'physical_hose_id': None}
        if include_action_features:
            result.update(action_features=action_features, action_valid=action_valid)
        return result


def fuse_hand_evidence(original, glove):
    """Presence OR; absence from a detector is not proof of an empty station."""
    if glove is None:
        return original
    return {'present': bool(original['present'] or glove.get('present') is True),
            'fresh': True, 'observation_state': 'observed' if original['present'] or glove.get('observation_state') == 'observed' else 'unknown',
            'hands': original['hands']+glove.get('hands', []),
            'source': 'original_mediapipe+R19_R18', 'glove_backend': glove,
            'actual_connection': None, 'physical_hose_id': None}
