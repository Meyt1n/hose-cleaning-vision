"""Frame observation timestamps; source PTS are never reconstructed from FPS."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

@dataclass
class TimedFrame:
    image: Any
    source_pts: int | None = None
    source_time_base: str | None = None
    captured_monotonic: float | None = None
    epoch: int = 0
    frame_index: int | None = None
    native_video_frame: Any = None
    score_enabled: bool = True

    @property
    def source_time(self):
        if self.source_pts is None or not self.source_time_base:
            return None
        base = Fraction(self.source_time_base)
        return Fraction(int(self.source_pts)) * base if base > 0 else None

    @property
    def time_basis(self):
        return 'source_pts' if self.source_time is not None else 'capture_clock_unverified'
