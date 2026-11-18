# -*- coding: utf-8 -*-
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build(cfg):
    torch.manual_seed(cfg['seed'])
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,
        min_size=cfg['min_size'],max_size=cfg['max_size'],
        rpn_pre_nms_top_n_train=1000,rpn_post_nms_top_n_train=512,
        rpn_pre_nms_top_n_test=1000,rpn_post_nms_top_n_test=256,
        box_batch_size_per_image=256,box_score_thresh=.05,box_nms_thresh=cfg['NMS_IoU'],box_detections_per_img=cfg['max_hands'])
    state=torch.load(cfg['coco_parent'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    for p in model.backbone.body.parameters():p.requires_grad_(False)
    return model


def training_mode(model):
    model.train()
    model.backbone.body.eval()


def mutable_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('backbone.body.')}


def load_mutable(model,state):
    expected={k for k in model.state_dict() if not k.startswith('backbone.body.')}
    if expected!=set(state):raise ValueError('MUTABLE_MODEL_KEYS_CHANGED')
    merged=model.state_dict()
    for k,v in state.items():
        if merged[k].shape!=v.shape:raise ValueError('MUTABLE_TENSOR_SHAPE_CHANGED')
        merged[k]=v
    model.load_state_dict(merged,strict=True)
