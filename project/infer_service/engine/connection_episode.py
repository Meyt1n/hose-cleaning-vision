"""Observed connection event lifecycle, independent of action/display smoothing.

The caller must supply actual-connection evidence, not cleaning-action logits.
Unknown breaks duration proof but never proves a detach or a new physical event.
No physical hose identity is inferred. Ambiguous gaps require review.
"""
from fractions import Fraction
from numbers import Integral


class ConnectionEpisodeCounter:
    def __init__(self, stream_id, threshold_seconds=3, max_gap_seconds=.12):
        if not stream_id:raise ValueError('a single tool/interface stream ID required')
        self.stream_id=stream_id
        self.threshold=Fraction(str(threshold_seconds));self.max_gap=Fraction(str(max_gap_seconds))
        if self.threshold<=0 or self.max_gap<=0:raise ValueError('positive temporal thresholds required')
        self.confirmed_count=0;self.next_episode=1;self.active=None
        self.run_start=None;self.last_timestamp=None;self.epoch=None
        self.review_reasons=set();self.ambiguous_repeat_candidates=0

    def _unknown(self, reason):
        self.run_start=None
        if self.active is not None:
            self.active['gap_unknown']=True
            self.review_reasons.add(reason)

    def update(self, connection_state, source_pts, time_base, epoch=0,
               evidence_scope='actual_connection'):
        if evidence_scope!='actual_connection':
            self._unknown('wrong_evidence_scope')
            raise ValueError('cleaning action evidence cannot prove actual connection')
        if connection_state not in ('connected','disconnected','unknown'):
            self._unknown('invalid_state');raise ValueError('unsupported connection state')
        try:
            if not isinstance(source_pts,Integral):raise ValueError('integer source PTS required')
            base=Fraction(str(time_base));timestamp=int(source_pts)*base
            if base<=0:raise ValueError('positive source time base required')
        except (TypeError,ValueError,ZeroDivisionError,OverflowError):
            self._unknown('invalid_PTS');self.last_timestamp=None
            return self._result(None,0,'unknown','invalid_PTS')
        if self.last_timestamp is not None and (epoch!=self.epoch or timestamp<=self.last_timestamp):
            self._unknown('epoch_change_or_PTS_reversal');self.last_timestamp=timestamp;self.epoch=epoch
            return self._result(None,0,'unknown','epoch_change_or_PTS_reversal')
        if self.last_timestamp is not None and timestamp-self.last_timestamp>self.max_gap:
            self._unknown('PTS_gap')
        self.last_timestamp=timestamp;self.epoch=epoch;event=None
        if connection_state=='unknown':
            self._unknown('unobservable_connection')
            return self._result(None,0,'unknown','unobservable_connection')
        if connection_state=='disconnected':
            self.active=None;self.run_start=None
            return self._result(None,0,'disconnected','observed_detach')
        if self.active is None:
            self.active=dict(id=self.next_episode,qualified=False,gap_unknown=False,repeat_reviewed=False)
            self.next_episode+=1
        if self.run_start is None:self.run_start=timestamp
        duration=timestamp-self.run_start
        if duration>=self.threshold:
            if not self.active['qualified']:
                self.active['qualified']=True;self.confirmed_count+=1
                event=dict(type='connection_duration_met',stream_id=self.stream_id,episode_id=self.active['id'],duration=float(duration),source_pts=int(source_pts),source_time_base=str(base),start_source_seconds=float(self.run_start),end_source_seconds=float(timestamp),confirmed_count=self.confirmed_count,physical_hose_id=None,per_hose_completion=False,actual_connection_verified=True,evidence_contract='provided actual-connection observations; not action logits')
            elif self.active['gap_unknown'] and not self.active['repeat_reviewed']:
                self.active['repeat_reviewed']=True;self.ambiguous_repeat_candidates+=1
                self.review_reasons.add('same_or_new_connection_after_unknown')
        return self._result(event,float(duration),'connected','current_connection_observation')

    def _result(self,event,duration,state,reason):
        return dict(connection_state=state,reason=reason,continuous_seconds=duration,
                    confirmed_count=self.confirmed_count,ambiguous_repeat_candidates=self.ambiguous_repeat_candidates,
                    review_required=bool(self.review_reasons),review_reasons=sorted(self.review_reasons),
                    count_status='minimum_confirmed_with_unresolved_gaps' if self.review_reasons else 'confirmed_under_provided_state_evidence',
                    episode_id=self.active['id'] if self.active else None,event=event,
                    physical_hose_id=None,per_hose_completion=False)
