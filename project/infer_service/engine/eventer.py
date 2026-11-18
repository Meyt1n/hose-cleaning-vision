"""Observed action segments. Unknown/gaps never bridge cleaning episodes.

Duration uses source PTS when available. No physical connection/unique hose claim.
"""
from __future__ import annotations
import time
from datetime import datetime


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


class ActionSegmenter:
    def __init__(self, station_no=None, batch_no=None, fps=25., merge_tolerance=10,
                 min_action_duration=0., sample_rate=1, merge_tolerance_frames=None,
                 max_observation_gap_seconds=.12):
        self.station_no, self.batch_no = station_no, batch_no
        self.fps, self.sample_rate = max(1., float(fps)), max(1, sample_rate)
        self.merge_tolerance = max(1, int(merge_tolerance))
        self.min_action_duration = max(0., float(min_action_duration))
        self.merge_tolerance_frames = merge_tolerance_frames or {}
        self.max_gap = float(max_observation_gap_seconds)
        self.segments = []
        self._gap_seconds = 0.
        self._epoch = None
        self.reset()

    def set_batch(self, station_no, batch_no):
        self.station_no, self.batch_no = station_no, batch_no

    def end_batch(self, reason='NORMAL'):
        self.flush()
        return {'type': 'batch_ended', 'station_no': self.station_no,
                'batch_no': self.batch_no, 'reason': reason, 'ts': _now()}

    def current_duration(self):
        return 0. if self.current_action is None else max(0., self._last-self._start)

    def update(self, action_id, action_type, confidence, timestamp_s=None,
               observation_state='observed', time_basis=None, epoch=0):
        basis = time_basis or ('source_pts' if timestamp_s is not None else 'capture_clock_unverified')
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        events = []
        if (self._last_input is not None and (timestamp <= self._last_input
                or timestamp-self._last_input > self.max_gap or self._epoch != epoch
                or self._basis != basis)):
            events.extend(self.flush())
        self._last_input, self._epoch, self._basis = timestamp, epoch, basis
        if action_id < 0 or action_type == 'UNKNOWN' or observation_state != 'observed':
            return events+self.flush()
        item = (action_id, action_type, float(confidence), timestamp, _now())
        if self.current_action is None:
            self._start_segment(item)
            return events
        if action_id == self.current_action:
            self._pending = []
            self._last = timestamp
            self._count += 1
            self._confidence_sum += float(confidence)
            self._end_wall = item[4]
            return events
        # Cleaning boundaries are never debounced/merged; short disconnections matter.
        cleaning_codes = ('WATER', 'GAS', 'CLEANING_OPERATION')
        if self.current_name in cleaning_codes or action_type in cleaning_codes:
            events.extend(self.flush())
            self._start_segment(item)
            return events
        if not self._pending or self._pending[-1][0] != action_id:
            self._pending = [item]
        else:
            self._pending.append(item)
        tolerance = max(1, int(self.merge_tolerance_frames.get(action_type, self.merge_tolerance)))
        if len(self._pending) >= tolerance:
            pending = self._pending[:]
            events.extend(self.flush())
            self._start_segment(pending[0])
            self._last, self._end_wall = pending[-1][3], pending[-1][4]
            self._count = len(pending)
            self._confidence_sum = sum(x[2] for x in pending)
        return events

    def notify_gap(self, seconds):
        self._gap_seconds += max(0., float(seconds))
        return self.flush()

    def on_stream_resume(self):
        events = self.flush()
        self._last_input = None
        return events

    def flush(self):
        if self.current_action is None:
            return []
        duration = self.current_duration()
        event = None
        if duration >= self.min_action_duration:
            event = {'type': 'action_segment', 'station_no': self.station_no,
                     'batch_no': self.batch_no, 'action_type': self.current_name,
                     'start_time': self._start_wall, 'end_time': self._end_wall,
                     'duration': round(duration, 6),
                     'confidence': round(self._confidence_sum/max(1, self._count), 4),
                     'frames': self._count, 'ts': _now(), 'time_basis': self._basis,
                     'start_source_seconds': self._start if self._basis == 'source_pts' else None,
                     'end_source_seconds': self._last if self._basis == 'source_pts' else None,
                     'evidence_scope': 'action_recognition', 'actual_connection_verified': False,
                     'per_hose_completion': False}
            self.segments.append({'action': self.current_name, 'action_id': self.current_action,
                                  'duration': event['duration'], 'frames': self._count,
                                  'end_ts': event['end_time'], 'time_basis': self._basis})
            self.segments = self.segments[-200:]
        self.current_action = self.current_name = None
        self._pending = []
        return [] if event is None else [event]

    def reset(self):
        self.current_action = self.current_name = None
        self._pending = []
        self._start = self._last = 0.
        self._count = 0
        self._confidence_sum = 0.
        self._start_wall = self._end_wall = None
        self._last_input = None
        self._basis = 'unverified'

    def reset_batch_state(self):
        self.reset()
        self.segments = []
        self._gap_seconds = 0.

    def _start_segment(self, item):
        self.current_action, self.current_name, confidence, timestamp, wall = item
        self._start = self._last = timestamp
        self._start_wall = self._end_wall = wall
        self._confidence_sum, self._count = confidence, 1
        self._pending = []

    def stats(self):
        return {'current_action': self.current_name, 'current_duration': round(self.current_duration(), 6),
                'segments': len(self.segments), 'gap_seconds': round(self._gap_seconds, 2),
                'time_basis': self._basis, 'per_hose_completion': False}
