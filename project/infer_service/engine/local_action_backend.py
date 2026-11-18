"""Opt-in offline hand/tool action backend for frozen, source-PTS-aligned features.

This is a separate binary action contract. It neither supplies fake body/depth
features to the281D LSTM nor invents WATER/GAS or physical hose identities.
Camera/ROI feature extraction must obey the checkpoint's frozen input manifest.
"""
from __future__ import annotations
from collections import deque
from fractions import Fraction
from pathlib import Path
import hashlib
import numpy as np
import torch
from common.local_action_model import load_trained_hand_tool_head
from infer_service.engine.continuity import ContinuousEvidenceTimer


class LocalActionBackend:
    def __init__(self, checkpoint_path, checkpoint_sha256, device='cpu',
                 positive_threshold=.8, negative_threshold=.2,
                 max_gap_seconds=.12, threshold_seconds=3):
        path=Path(checkpoint_path)
        if not checkpoint_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()!=checkpoint_sha256:
            raise ValueError('checkpoint SHA256 mismatch')
        if not 0<=negative_threshold<positive_threshold<=1:
            raise ValueError('ordered negative/positive confidence thresholds required')
        self.model=load_trained_hand_tool_head(path,device)
        self.device=device
        self.checkpoint_sha256=checkpoint_sha256
        self.positive_threshold=positive_threshold
        self.negative_threshold=negative_threshold
        self.max_gap=Fraction(str(max_gap_seconds))
        self.timer=ContinuousEvidenceTimer(threshold_seconds,max_gap_seconds,'cleaning_action')
        self.history=deque(maxlen=32)
        self.last_timestamp=None
        self.epoch=None

    def reset(self):
        """Break continuity and history while preserving the session action count."""
        self.history.clear()
        self.last_timestamp=None
        self.epoch=None
        self.timer.reset()

    def process_features(self, context, hands, hand_valid, context_valid,
                         source_pts, time_base, epoch=0, score_enabled=True):
        """Consume one current observation plus past state; never consume future frames."""
        try:
            base=Fraction(str(time_base))
            if base<=0 or source_pts is None:raise ValueError('invalid source timestamp')
            timestamp=Fraction(int(source_pts))*base
        except (TypeError,ValueError,ZeroDivisionError,OverflowError):
            self.reset()
            return self._result(None,False,None,'invalid_source_timestamp',None)
        reset_reason=None
        if self.last_timestamp is not None and (epoch!=self.epoch or timestamp<=self.last_timestamp or timestamp-self.last_timestamp>self.max_gap):
            self.reset();reset_reason='PTS_gap_reversal_or_epoch_change'
        dt=0 if self.last_timestamp is None else float(timestamp-self.last_timestamp)
        context=np.asarray(context,dtype=np.float32)
        hands=np.asarray(hands,dtype=np.float32)
        valid=np.asarray(hand_valid,dtype=bool)
        if context.shape!=(1024,) or hands.shape!=(6,816) or valid.shape!=(6,):
            self.reset();raise ValueError('frozen R25 feature shapes1024 and6x816 required')
        observed=bool(context_valid and valid.any())
        context=np.where(bool(context_valid),context,0)
        hands=np.where(valid[:,None],hands,0)
        if not np.isfinite(context).all() or not np.isfinite(hands).all():
            self.reset();raise ValueError('nonfinite observed features')
        self.history.append((context.copy(),hands.copy(),valid.copy(),bool(context_valid),dt))
        self.last_timestamp=timestamp;self.epoch=epoch
        batch={'context':np.zeros((1,32,1024),np.float32),
               'hands':np.zeros((1,32,6,816),np.float32),
               'hand_valid':np.zeros((1,32,6),bool),
               'context_valid':np.zeros((1,32),bool),
               'delta_pts':np.zeros((1,32),np.float32)}
        offset=32-len(self.history)
        for j,row in enumerate(self.history,offset):
            for key,value in zip(batch,row):batch[key][0,j]=value
        batch={k:torch.from_numpy(x).to(self.device) for k,x in batch.items()}
        with torch.inference_mode():
            p=float(self.model(**batch)['logits'][0,-1].softmax(-1)[1].cpu())
        accepted=observed and (p<=self.negative_threshold or p>=self.positive_threshold)
        positive=accepted and p>=self.positive_threshold
        if score_enabled:
            event=self.timer.update(bool(positive),int(source_pts),str(base),epoch=epoch,
                                    observation_state='observed' if accepted else 'unknown')
        else:
            self.timer.reset();event=None
        reason=('causal_warmup_unscored' if not score_enabled else
                'missing_current_hand_or_context' if not observed else
                'confidence_unknown' if not accepted else 'current_action_observation')
        return self._result(p,accepted,event,reason,reset_reason)

    def _result(self,p,accepted,event,reason,reset_reason):
        positive=accepted and p is not None and p>=self.positive_threshold
        return dict(action_code='CLEANING_OPERATION' if positive else 'OTHER_OPERATION' if accepted else 'UNKNOWN',
                    probability=p,raw_positive=p is not None and p>=.5,
                    observation_state='observed' if accepted else 'unknown',
                    accepted=accepted,accepted_positive=bool(positive),reason=reason,
                    history_reset_reason=reset_reason,history_samples=len(self.history),
                    cleaning_action_count=self.timer.count,cleaning_action_duration_event=event,
                    actual_connection_verified=False,physical_hose_identity=None,
                    per_hose_completion=False,semantic_scope='cleaning_operation_not_physical_connection')
