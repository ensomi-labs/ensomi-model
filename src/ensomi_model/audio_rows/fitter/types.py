from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimingSegment:
    offset_ms: float
    beat_length_ms: float

    @property
    def local_bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class _SegmentFit:
    start_frame: int
    end_frame: int
    score: float
    beat_length_ms: float
    offset_ms: float
    raw_bpm: float
    tempo_multiplier: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class _SplitCandidate:
    frame: int
    score: float


@dataclass(frozen=True)
class _ChangeTimeCandidate:
    time_ms: float
    score: float


@dataclass(frozen=True)
class _EvaluatedSplit:
    segment_index: int
    candidate: _SplitCandidate
    left_fit: _SegmentFit
    right_fit: _SegmentFit
    improvement: float


@dataclass(frozen=True)
class _GridCandidate:
    score: float
    bpm: float
    beat_length_ms: float
    offset_ms: float
