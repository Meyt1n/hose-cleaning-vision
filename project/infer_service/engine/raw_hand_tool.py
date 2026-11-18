"""Raw-frame adapter for the trained, frozen-encoder hand/tool action backend.

The feature recipe is versioned with the checkpoint. No cached features, old
predictions, body landmarks, inferred depth or hose identities are consumed.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
import torch
from common.frame_timing import TimedFrame
from common.glove_evidence import GloveHandEvidence
from common.portable_hand_evidence import PortableHandEvidence
from common.label_catalog import index_of, NAME_ZH, COLOR_HEX
from infer_service.engine.local_action_backend import LocalActionBackend
from infer_service.engine.result import FrameResult


def _resolve_checked(base, entry):
    path = (base/entry['file']).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    if h.hexdigest() != entry['sha256']:
        raise ValueError('raw action dependency SHA mismatch: ' + path.name)
    return path


class NativeDualROI:
    """The training FFmpeg filters, applied to native decoded YUV frames."""
    def __init__(self):
        self.graph = self.source = self.sink = None
        self.calls = 0

    def _init(self, frame):
        import av
        graph = av.filter.Graph()
        source = graph.add_buffer(template=frame)
        crop = graph.add('crop', '2200:2300:1000:200')
        scale = graph.add('scale', '1600:1672:flags=bicubic')
        split = graph.add('split', '2')
        a = graph.add('crop', '727:727:582:218')
        b = graph.add('crop', '1600:1527:0:145')
        sa = graph.add('scale', '224:224:flags=bicubic')
        sb = graph.add('scale', '224:224:flags=bicubic')
        stack = graph.add('hstack', 'inputs=2')
        fmt = graph.add('format', 'pix_fmts=rgb24')
        sink = graph.add('buffersink')
        source.link_to(crop); crop.link_to(scale); scale.link_to(split)
        split.link_to(a, output_idx=0); split.link_to(b, output_idx=1)
        a.link_to(sa); b.link_to(sb)
        sa.link_to(stack, input_idx=0); sb.link_to(stack, input_idx=1)
        stack.link_to(fmt); fmt.link_to(sink)
        graph.configure()
        self.graph, self.source, self.sink = graph, source, sink

    def __call__(self, frame):
        if frame is None or (frame.width, frame.height) != (4000, 3000):
            raise ValueError('raw backend requires native 4000x3000 camera frames')
        if self.graph is None:
            self._init(frame)
        original_pts = frame.pts
        self.source.push(frame)
        out = self.sink.pull()
        if out.pts != original_pts:
            raise ValueError('native ROI filter did not preserve source PTS')
        rgb = out.to_ndarray(format='rgb24')
        if rgb.shape != (224, 448, 3):
            raise ValueError('frozen dual ROI output geometry changed')
        self.calls += 1
        return rgb


class RawHandToolDetector:
    def __init__(self, manifest_path, device='cpu', max_gap_seconds=.12):
        from torchvision.models import resnet18
        manifest_path = Path(manifest_path).resolve()
        cfg = json.loads(manifest_path.read_text(encoding='utf-8'))
        if cfg['input_contract'] != 'R25_native_dualROI1024_hand6x816_v1':
            raise ValueError('unsupported raw feature recipe')
        base = manifest_path.parent
        self.device, self.cfg = device, cfg
        checkpoint = _resolve_checked(base, cfg['action'])
        rgb_weight = _resolve_checked(base, cfg['rgb_parent'])
        glove_manifest = _resolve_checked(base, cfg['glove_manifest'])
        hand_config = dict(device=device, score_threshold=.5, roi_normalized=[.375, .1, .375, 1400/3000])
        fallback = cfg.get('fallback_detector')
        if fallback is None:
            self.hand_provider = GloveHandEvidence(glove_manifest, **hand_config)
        else:
            self.hand_provider = PortableHandEvidence(
                glove_manifest, base / fallback['file'], fallback['sha256'], **hand_config)
        self.rgb_encoder = resnet18(weights=None)
        self.rgb_encoder.load_state_dict(torch.load(rgb_weight, map_location='cpu', weights_only=True), strict=True)
        self.rgb_encoder.fc = torch.nn.Identity()
        self.rgb_encoder.requires_grad_(False).eval().to(device)
        self.backend = LocalActionBackend(checkpoint, cfg['action']['sha256'], device,
            cfg['positive_threshold'], cfg['negative_threshold'], max_gap_seconds)
        self.native_roi = NativeDualROI()
        self.seq_len, self.extractor, self._gpu_enabled = 32, self, False
        self.rgb_forward_calls = self.action_forward_calls = 0
        self.last_details = None
        label_map = {index_of(c):dict(id=c, name=NAME_ZH[c], color=COLOR_HEX[c])
                     for c in ('CLEANING_OPERATION', 'OTHER_OPERATION')}
        self.loaded = SimpleNamespace(path=str(checkpoint), mode='raw_hand_tool',
            seq_len=32, feature_dim=1024+6*816, aug_version='R25_frozen_raw_v1',
            sample_rate=2, train_fps=25., label_map=label_map,
            info=dict(model_type='R26 causal action; raw RGB+gloved hands',
                      checkpoint_sha256=cfg['action']['sha256'],
                      semantic_scope='cleaning_operation_not_physical_connection'))

    @torch.inference_mode()
    def process(self, packet):
        if not isinstance(packet, TimedFrame) or packet.source_time is None:
            raise ValueError('raw action pipeline requires source PTS')
        native = packet.native_video_frame
        if native is None or native.pts != packet.source_pts or str(native.time_base) != packet.source_time_base:
            raise ValueError('original decoded frame and packet PTS must match')
        started = time.perf_counter()
        pair = self.native_roi(native)
        images = np.stack([pair[:, :224], pair[:, 224:]])
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)/255
        x = (x-x.new_tensor([.485, .456, .406])[None, :, None, None])/x.new_tensor([.229, .224, .225])[None, :, None, None]
        context = self.rgb_encoder(x).reshape(1024).cpu().numpy()
        self.rgb_forward_calls += 1
        evidence = self.hand_provider.predict(packet.image, include_action_features=True,
                                             rgb=native.to_ndarray(format='rgb24'))
        hands, valid = evidence.pop('action_features'), evidence.pop('action_valid')
        # The pipeline owns counting. This backend only supplies causal logits.
        detail = self.backend.process_features(context, hands, valid, True,
            packet.source_pts, packet.source_time_base, packet.epoch, score_enabled=False)
        self.action_forward_calls += 1
        detail['reason'] = ('missing_current_hand_or_context' if not valid.any() else
                            'confidence_unknown' if not detail['accepted'] else 'current_raw_action_observation')
        detail['source_frame_index'] = packet.frame_index
        detail['feature_origin'] = 'current_raw_native_frame'
        detail['score_enabled'] = packet.score_enabled
        code, p = detail['action_code'], detail['probability']
        raw_code = 'CLEANING_OPERATION' if p >= .5 else 'OTHER_OPERATION'
        result = FrameResult(index_of(code), NAME_ZH.get(code, '待判定'),
            p if code == 'CLEANING_OPERATION' else 1-p if code == 'OTHER_OPERATION' else max(p, 1-p),
            detail['history_samples'], action_code=code, results=evidence,
            skipped=not detail['accepted'] or not packet.score_enabled,
            elapsed_ms=(time.perf_counter()-started)*1000,
            raw_action_id=index_of(raw_code), raw_confidence=max(p, 1-p))
        result.hand_evidence = evidence
        result.model_details = detail
        result.observation_state, result.decision_reason = detail['observation_state'], detail['reason']
        result.source_pts, result.source_time_base = packet.source_pts, packet.source_time_base
        result.epoch, result.time_basis = packet.epoch, packet.time_basis
        result.score_enabled = packet.score_enabled
        self.last_details = detail
        return result

    def reset(self):
        self.backend.reset()

    def stats(self):
        provider = self.hand_provider
        return dict(backend='raw_hand_tool', buffered=len(self.backend.history),
            native_roi_frames=self.native_roi.calls, rgb_forward_calls=self.rgb_forward_calls,
            hand_detector_forward_calls=provider.detector_forward_calls,
            R19_primary_forward_calls=getattr(provider, 'primary_calls', provider.detector_forward_calls),
            R29_fallback_attempts=getattr(provider, 'fallback_attempts', 0),
            R29_fallback_successes=getattr(provider, 'fallback_successes', 0),
            hand_pose_forward_calls=provider.pose_forward_calls,
            hand_pose_crops=provider.pose_hand_crops, action_forward_calls=self.action_forward_calls,
            cached_feature_inputs=0, actual_connection_verified=False, physical_hose_identity=None)

    def draw_landmarks(self, frame, evidence):
        import cv2
        height, width = frame.shape[:2]
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                 (13,17),(0,17),(17,18),(18,19),(19,20)]
        for i, hand in enumerate(evidence.get('hands', [])):
            xy = np.rint(np.asarray(hand['xy_normalized'])*[width, height]).astype(int)
            color = (0, 230, 255) if i % 2 == 0 else (255, 180, 40)
            for a,b in edges:
                cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 1, cv2.LINE_AA)
            for point in xy:
                cv2.circle(frame, tuple(point), 1, color, -1, cv2.LINE_AA)
            box = np.asarray(hand['box_xyxy'])/[4000, 3000, 4000, 3000]*[width, height, width, height]
            box = np.rint(box).astype(int)
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
