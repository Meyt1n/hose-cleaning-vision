"""Shared frame result without importing a particular pose runtime."""
from __future__ import annotations


class FrameResult:
    __slots__ = ('action_id', 'action_name', 'action_code', 'confidence',
                 'buffered', 'rule_hit', 'features', 'results', 'skipped',
                 'elapsed_ms', 'raw_action_id', 'raw_confidence',
                 'hand_evidence', 'observation_state', 'decision_reason',
                 'source_pts', 'source_time_base', 'epoch', 'time_basis',
                 'model_details', 'score_enabled')

    def __init__(self, action_id, action_name, confidence, buffered,
                 action_code=None, rule_hit=None, features=None, results=None,
                 skipped=False, elapsed_ms=0.0,
                 raw_action_id=None, raw_confidence=None):
        self.action_id, self.action_name = action_id, action_name
        self.action_code = action_code or action_name
        self.confidence, self.buffered = confidence, buffered
        self.rule_hit, self.features, self.results = rule_hit, features, results
        self.skipped, self.elapsed_ms = skipped, elapsed_ms
        self.raw_action_id, self.raw_confidence = raw_action_id, raw_confidence
        self.hand_evidence = self.model_details = None
        self.observation_state, self.decision_reason = 'unknown', 'not_observed'
        self.source_pts = self.source_time_base = None
        self.epoch, self.time_basis = 0, 'unverified'
        self.score_enabled = True
