"""Body-independent local RGB/hand action heads.

LocalRGBActionHead matches the already-trained research heads. HandToolTemporalHead
is an untrained architecture until a separately validated checkpoint is loaded.
Neither hand detection nor action logits establish physical hose connection.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class LocalRGBActionHead(nn.Module):
    def __init__(self, kind='lstm'):
        super().__init__()
        self.kind = kind
        self.adapter = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024,128), nn.ReLU(), nn.Dropout(.1))
        if kind == 'lstm':
            self.temporal = nn.LSTM(128,128,num_layers=2,batch_first=True,dropout=.1)
            self.head = nn.Linear(128,2)
        elif kind == 'frame':
            self.head = nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2))
        else:
            raise ValueError('unsupported local action head')

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 1024:
            raise ValueError('B,T,1024 local RGB feature sequence required')
        if self.kind == 'frame':
            return self.head(self.adapter(x[:,-1]))
        z,_ = self.temporal(self.adapter(x))
        return self.head(z[:,-1])


class _CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = 2*dilation
        self.conv1 = nn.Conv1d(channels,channels,3,dilation=dilation)
        self.conv2 = nn.Conv1d(channels,channels,3,dilation=dilation)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        z = self.drop(F.gelu(self.conv1(F.pad(x,(self.pad,0)))))
        z = self.drop(F.gelu(self.conv2(F.pad(z,(self.pad,0)))))
        return x+z


class HandToolTemporalHead(nn.Module):
    """Unordered observed hands + local interaction/context + causal history.

hand_features: B,T,K,816 = frozen Hand5 pooled768 + localXY42 + box4 +
detector_score1 + provided_observation1. Provided does not mean finger visibility.
context: B,T,1024 = frozen two-ROI RGB. No full-body skeleton or fake depth.
Outputs all time steps; any missing current evidence remains explicitly missing.
"""
    def __init__(self, channels=96):
        super().__init__()
        self.hand = nn.Sequential(nn.LayerNorm(816),nn.Linear(816,channels),nn.GELU())
        self.context = nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,channels),nn.GELU())
        self.fusion = nn.Linear(channels*3+3,channels)
        self.temporal = nn.Sequential(*[_CausalBlock(channels,d) for d in (1,2,4)])
        self.output = nn.Linear(channels,2)
        self.receptive_field_samples = 29

    def forward(self, context, hands, hand_valid, context_valid, delta_pts):
        if context.ndim != 3 or context.shape[-1] != 1024:
            raise ValueError('context shape B,T,1024 required')
        if hands.ndim != 4 or hands.shape[:2] != context.shape[:2] or hands.shape[-1] != 816:
            raise ValueError('hand shape B,T,K,816 required')
        if hand_valid.shape != hands.shape[:-1] or context_valid.shape != context.shape[:-1] or delta_pts.shape != context_valid.shape:
            raise ValueError('observation/time mask shape mismatch')
        valid = hand_valid.bool()
        clean_hands = torch.where(valid[...,None],hands,torch.zeros_like(hands))
        clean_context = torch.where(context_valid[...,None].bool(),context,torch.zeros_like(context))
        if not torch.isfinite(clean_hands).all() or not torch.isfinite(clean_context).all() or not torch.isfinite(delta_pts).all():
            raise ValueError('nonfinite observed feature/time')
        if (delta_pts < 0).any() or (delta_pts > .120001).any():
            raise ValueError('history must reset before PTS gap/reversal')
        h = self.hand(clean_hands)
        n = valid.sum(-1,keepdim=True)
        mean = (h*valid[...,None]).sum(-2)/n.clamp_min(1)
        maximum = h.masked_fill(~valid[...,None],torch.finfo(h.dtype).min).amax(-2)
        maximum = torch.where(n > 0,maximum,torch.zeros_like(maximum))
        c = self.context(clean_context)
        dt = delta_pts.clamp(0,.12)/.08
        z = torch.cat((mean,maximum,c,dt[...,None],(n>0).float(),context_valid[...,None].float()),-1)
        z = self.fusion(z).transpose(1,2)
        shared = self.temporal(z).transpose(1,2)
        logits = self.output(shared)
        return {'logits':logits,'shared':shared,'current_evidence_observed':valid.any(-1)&context_valid.bool(),
                'actual_connection_verified':False,'physical_hose_identity':None}


def load_trained_hand_tool_head(checkpoint_path, device='cpu'):
    """Never silently use a randomly initialized action head for inference."""
    if checkpoint_path is None:
        raise ValueError('trained hand/tool action checkpoint required; architecture alone has no accuracy')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (state.get('kind') != 'hand_tool_causal_action_v1'
            or int(state.get('optimizer_steps',0)) <= 0
            or state.get('semantic_scope') != 'cleaning_operation_not_physical_connection'):
        raise ValueError('wrong/untrained action checkpoint or semantic scope')
    model = HandToolTemporalHead(channels=int(state.get('channels',96)))
    model.load_state_dict(state['model'],strict=True)
    return model.requires_grad_(False).eval().to(device)
