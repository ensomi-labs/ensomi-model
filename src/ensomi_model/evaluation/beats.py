"""Beat coordinates for chart evaluation.

Every event time is placed on a beat grid: the chart's own red timing points, or
the grid a generation condition supplied. Positions are expressed in canonical
beats. One factor ``2**fold`` per grid puts the dominant notated BPM into
[80, 160), and that factor rescales every position, subdivision and bar, not only
the BPM label: a 1/4 at a notated 240 BPM is a 1/8 at the canonical 120. The same
chart notated at half or double BPM therefore gets identical canonical
coordinates. A bar holds ``meter`` canonical beats.

The fold is chosen once per grid, from the segment that covers the longest part
of the chart, so relative tempo inside a chart is kept and a tempo change across
160 BPM does not fold one segment and not the next.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ..osu_core.timing import RedTimingPoint, validate_red_timing_point

CANONICAL_BPM_MIN = 80.0
CANONICAL_BPM_MAX = 160.0
SNAP_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
COMMON_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16)
SNAP_TOLERANCE_MS = 2.0

TAP, LN_HEAD, LN_RELEASE = 1, 2, 3


@dataclass(frozen=True)
class Segment:
    """A red timing point as notated: offset, notated beat length and meter."""
    offset_ms: float
    beat_length_ms: float
    meter: int


def fold_for_bpm(bpm: float) -> int:
    """Power of two that moves ``bpm`` into [80, 160): canonical = bpm * 2**fold."""
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f'BPM must be positive and finite, got {bpm!r}')
    fold = 0
    while bpm * 2.0 ** fold < CANONICAL_BPM_MIN:
        fold += 1
    while bpm * 2.0 ** fold >= CANONICAL_BPM_MAX:
        fold -= 1
    return fold


@dataclass(frozen=True)
class BeatGrid:
    """Red timing segments plus one fold factor shared by all of them."""
    segments: tuple[Segment, ...]
    fold: int
    dominant: int

    @classmethod
    def from_timing_points(cls, points: Sequence[RedTimingPoint], span_ms: tuple[float, float]) -> 'BeatGrid':
        """Build a grid; ``span_ms`` is the chart's first and last event time.

        Points sharing an offset keep the last one, as osu! applies them. The
        dominant segment is the one covering the longest part of the span; ties
        go to the earliest segment.
        """
        if not points:
            raise ValueError('A beat grid needs at least one red timing point')
        for point in points:
            validate_red_timing_point(point)
        by_offset: dict[float, RedTimingPoint] = {}
        for point in points:
            by_offset[point.offset_ms] = point
        segments = tuple(Segment(float(p.offset_ms), float(p.beat_length_ms), int(p.meter))
                         for _, p in sorted(by_offset.items()))
        start, end = (float(span_ms[0]), float(span_ms[1]))
        if not (math.isfinite(start) and math.isfinite(end) and start <= end):
            raise ValueError(f'Span must be finite and ordered, got {span_ms!r}')
        covered = []
        for i, segment in enumerate(segments):
            lo = start if i == 0 else max(start, segment.offset_ms)
            hi = end if i + 1 == len(segments) else min(end, segments[i + 1].offset_ms)
            covered.append(max(0.0, hi - lo))
        dominant = int(np.argmax(covered)) if any(covered) else 0
        return cls(segments, fold_for_bpm(60000.0 / segments[dominant].beat_length_ms), dominant)

    @property
    def scale(self) -> float:
        return 2.0 ** self.fold

    @property
    def dominant_bpm(self) -> float:
        return 60000.0 / self.segments[self.dominant].beat_length_ms

    @property
    def canonical_bpm(self) -> float:
        return self.dominant_bpm * self.scale

    def canonical_beat_lengths(self) -> np.ndarray:
        return np.array([s.beat_length_ms for s in self.segments]) / self.scale

    def _origins(self) -> tuple[np.ndarray, np.ndarray]:
        """Global canonical beat and global bar index at each segment's offset.

        A segment ending inside a bar counts that partial bar unless the
        remainder is within ``SNAP_TOLERANCE_MS``; a new red line always starts a
        new bar, as in osu!.
        """
        lengths = self.canonical_beat_lengths()
        beats, bars = np.zeros(len(self.segments)), np.zeros(len(self.segments), dtype=np.int64)
        for i in range(1, len(self.segments)):
            span = (self.segments[i].offset_ms - self.segments[i - 1].offset_ms) / lengths[i - 1]
            beats[i] = beats[i - 1] + span
            bar_ms = self.segments[i - 1].meter * lengths[i - 1]
            whole = math.floor(span / self.segments[i - 1].meter)
            remainder_ms = span * lengths[i - 1] - whole * bar_ms
            bars[i] = bars[i - 1] + whole + (remainder_ms > SNAP_TOLERANCE_MS)
        return beats, bars

    def locate(self, times_ms) -> dict[str, np.ndarray]:
        """Canonical coordinates of each time.

        Returns arrays: ``segment``; ``beat`` (canonical beats from the segment's
        offset); ``global_beat``; ``bar`` (global bar index); ``bar_beat``
        (canonical beats into the bar); ``snap`` (smallest canonical denominator
        in ``SNAP_DENOMINATORS`` within ``SNAP_TOLERANCE_MS``, 0 when off-grid);
        ``residual_ms`` (signed error to that snap position; for off-grid times,
        to the nearest position of a denominator in ``COMMON_DENOMINATORS``).
        Times before the first red line use the first segment, as osu! does.
        """
        times = np.asarray(times_ms, dtype=np.float64)
        offsets = np.array([s.offset_ms for s in self.segments])
        meters = np.array([s.meter for s in self.segments], dtype=np.float64)
        lengths = self.canonical_beat_lengths()
        segment = np.clip(np.searchsorted(offsets, times, side='right') - 1, 0, len(offsets) - 1)
        beat = (times - offsets[segment]) / lengths[segment]
        beat_origin, bar_origin = self._origins()
        snap = np.zeros(times.shape, dtype=np.int16)
        residual = np.full(times.shape, np.nan)
        for d in SNAP_DENOMINATORS:
            error = (beat - np.round(beat * d) / d) * lengths[segment]
            hit = (snap == 0) & (np.abs(error) <= SNAP_TOLERANCE_MS)
            snap[hit], residual[hit] = d, error[hit]
        off = snap == 0
        if off.any():
            best = np.full(times.shape, np.inf)
            for d in COMMON_DENOMINATORS:
                error = (beat - np.round(beat * d) / d) * lengths[segment]
                closer = off & (np.abs(error) < np.abs(best))
                best[closer] = error[closer]
            residual[off] = best[off]
        placed = np.where(off, beat, beat - residual / lengths[segment])
        local_bar = np.floor(placed / meters[segment] + 1e-9)
        return dict(segment=segment, beat=beat, global_beat=beat_origin[segment] + beat,
                    bar=bar_origin[segment] + local_bar.astype(np.int64),
                    bar_beat=placed - local_bar * meters[segment], snap=snap, residual_ms=residual)

    def time_of(self, segment, beat) -> np.ndarray:
        """Inverse of ``locate``: milliseconds of a canonical beat in a segment."""
        segment = np.asarray(segment, dtype=np.int64)
        offsets = np.array([s.offset_ms for s in self.segments])
        return offsets[segment] + np.asarray(beat, dtype=np.float64) * self.canonical_beat_lengths()[segment]


@dataclass(frozen=True)
class ChartEvents:
    """Columnar events of one chart, ordered by time, then lane."""
    time_ms: np.ndarray
    lane: np.ndarray
    kind: np.ndarray
    coords: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.time_ms)

    def take(self, mask) -> 'ChartEvents':
        mask = np.asarray(mask)
        return ChartEvents(self.time_ms[mask], self.lane[mask], self.kind[mask],
                           {k: v[mask] for k, v in self.coords.items()})


def renotate(points: Sequence[RedTimingPoint], factor: int) -> list[RedTimingPoint]:
    """The same timing notated at ``2**factor`` times the BPM, meter unchanged.

    A chart and its renotation describe the same music; their canonical
    coordinates must agree, which is what this helper lets callers check.
    """
    scale = 2.0 ** factor
    return [RedTimingPoint(p.offset_ms, p.beat_length_ms / scale, p.meter) for p in points]


def chart_events(objects: Sequence[ManiaHitObject], grid: BeatGrid) -> ChartEvents:
    """Taps, long-note heads and long-note releases placed on ``grid``."""
    rows = []
    for o in objects:
        if o.kind is ManiaHitObjectKind.HOLD:
            rows.append((o.start_time_ms, o.lane, LN_HEAD))
            rows.append((o.end_time_ms, o.lane, LN_RELEASE))
        else:
            rows.append((o.start_time_ms, o.lane, TAP))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    time = np.array([r[0] for r in rows], dtype=np.float64)
    return ChartEvents(time, np.array([r[1] for r in rows], dtype=np.int8),
                       np.array([r[2] for r in rows], dtype=np.int8), grid.locate(time))


def in_spans(times_ms, spans: Sequence[tuple[float, float]]) -> np.ndarray:
    """Mask of times inside any half-open span ``[start, end)``.

    Spans select either part of a scope: the given part a condition fixes, or
    the part that is scored. Both may be any union of intervals.
    """
    times = np.asarray(times_ms, dtype=np.float64)
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in spans:
        mask |= (times >= start) & (times < end)
    return mask
