"""Strict observation continuity, separate from SOP display smoothing.

Gesture duration events do not establish physical connection or unique hose ID.
"""
from __future__ import annotations
from fractions import Fraction

class ContinuousEvidenceTimer:
    def __init__(self, threshold_seconds=3, max_gap_seconds=.12, scope='cleaning_action'):
        self.threshold = Fraction(str(threshold_seconds))
        self.max_gap = Fraction(str(max_gap_seconds))
        if self.threshold <= 0 or self.max_gap <= 0:
            raise ValueError('positive threshold and maximum gap required')
        if scope not in ('cleaning_action', 'actual_connection'):
            raise ValueError('unsupported evidence scope')
        self.scope = scope
        self.count = 0
        self.reset()

    def reset(self):
        self.start = self.last = None
        self.epoch = self.identity = None
        self.emitted = False
        self.duration = Fraction(0)

    def update(self, positive, source_pts=None, time_base=None, epoch=0,
               observation_state='observed', identity=None):
        if (positive is not True or observation_state != 'observed'
                or source_pts is None or not time_base
                or (self.scope == 'actual_connection' and identity is None)):
            self.reset()
            return None
        try:
            base = Fraction(time_base)
            timestamp = Fraction(int(source_pts)) * base
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            self.reset()
            return None
        if base <= 0:
            self.reset()
            return None
        changed = self.epoch != epoch or self.identity != identity
        broken = self.last is not None and (timestamp <= self.last or timestamp-self.last > self.max_gap)
        if self.start is None or changed or broken:
            self.reset()
            self.start = timestamp
            self.epoch, self.identity = epoch, identity
        self.last = timestamp
        self.duration = timestamp-self.start
        if self.duration < self.threshold or self.emitted:
            return None
        self.emitted = True
        self.count += 1
        return {
            'type': 'cleaning_action_duration_met' if self.scope == 'cleaning_action' else 'connection_duration_met',
            'evidence_scope': self.scope,
            'duration': float(self.duration),
            'threshold_seconds': float(self.threshold),
            'start_source_seconds': float(self.start),
            'end_source_seconds': float(timestamp),
            'source_pts': int(source_pts),
            'source_time_base': str(base),
            'time_basis': 'source_pts',
            'event_count': self.count,
            'hose_id': identity if self.scope == 'actual_connection' else None,
            'actual_connection_verified': self.scope == 'actual_connection',
            'per_hose_completion': self.scope == 'actual_connection',
        }
