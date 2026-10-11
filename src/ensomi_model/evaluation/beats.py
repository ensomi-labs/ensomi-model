"""Beat coordinates for chart evaluation.

Every event time is placed on a beat grid: the musical red lines of a chart (see
``redlines``), or the grid a generation condition supplied. Positions are in
canonical beats. Each segment's notated BPM is moved into [80, 160) by its own
factor ``2**fold``, and that factor rescales the segment's positions,
subdivisions and bars, not only the BPM label: a 1/4 at a notated 240 BPM is a
1/8 at the canonical 120. A chart, or any of its segments, notated at half or
double the BPM therefore gets identical canonical coordinates.

Beats restart at every segment's offset. Bars are kept apart from beats: a bar
starts at each bar start the grid lists and holds ``meter`` canonical beats.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ..osu_core.timing import RedTimingPoint

CANONICAL_BPM_MIN = 80.0
CANONICAL_BPM_MAX = 160.0
SNAP_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
COMMON_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16)
SNAP_TOLERANCE_MS = 2.0

TAP, LN_HEAD, LN_RELEASE = 1, 2, 3


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


def canonical_beat_length(beat_length_ms: float) -> float:
    return beat_length_ms / 2.0 ** fold_for_bpm(60000.0 / beat_length_ms)


@dataclass(frozen=True)
class Segment:
    """A musical red line: offset, notated beat length and meter."""
    offset_ms: float
    beat_length_ms: float
    meter: int

    @property
    def fold(self) -> int:
        return fold_for_bpm(60000.0 / self.beat_length_ms)

    @property
    def canonical_beat_length_ms(self) -> float:
        return canonical_beat_length(self.beat_length_ms)

    @property
    def canonical_bpm(self) -> float:
        return 60000.0 / self.canonical_beat_length_ms


@dataclass(frozen=True)
class BarStart:
    time_ms: float
    meter: int


@dataclass(frozen=True)
class BeatGrid:
    """Musical segments, each folded on its own, and the bar starts on them."""
    segments: tuple[Segment, ...]
    bar_starts: tuple[BarStart, ...]

    def __post_init__(self):
        if not self.segments:
            raise ValueError('A beat grid needs at least one segment')
        for s in self.segments:
            # In canonical terms, on the fold's own quantity: a renotation is valid exactly when the original is.
            if not (math.isfinite(s.offset_ms) and math.isfinite(s.beat_length_ms) and s.beat_length_ms > 0):
                raise ValueError(f'A segment needs a finite offset and a finite positive beat length: {s}')
            if s.meter < 1:
                raise ValueError(f'A segment needs a meter of at least 1: {s}')
            bpm = 60000.0 / s.beat_length_ms
            if not CANONICAL_BPM_MIN <= bpm * 2.0 ** fold_for_bpm(bpm) < CANONICAL_BPM_MAX:
                raise ValueError(f'A segment must fold into [{CANONICAL_BPM_MIN:g}, {CANONICAL_BPM_MAX:g}) BPM: {s}')
        if any(b.offset_ms <= a.offset_ms for a, b in zip(self.segments, self.segments[1:])):
            raise ValueError('Segment offsets must increase')
        if not self.bar_starts or any(b.time_ms <= a.time_ms for a, b in zip(self.bar_starts, self.bar_starts[1:])):
            raise ValueError('A beat grid needs increasing bar starts, at least one')

    @classmethod
    def from_timing_points(cls, points: Sequence[RedTimingPoint]) -> 'BeatGrid':
        """A grid whose every point is musical and starts a bar, as a supplied grid is.

        Points sharing an offset keep the last one, as osu! applies them.
        """
        by_offset = {float(p.offset_ms): p for p in points}
        segments = tuple(Segment(o, float(p.beat_length_ms), int(p.meter)) for o, p in sorted(by_offset.items()))
        return cls(segments, tuple(BarStart(s.offset_ms, s.meter) for s in segments))

    def canonical_beat_lengths(self) -> np.ndarray:
        return np.array([s.canonical_beat_length_ms for s in self.segments])

    def dominant(self, span_ms: tuple[float, float]) -> int:
        """Index of the segment covering the longest part of ``span_ms``; ties go to the earliest."""
        start, end = float(span_ms[0]), float(span_ms[1])
        covered = []
        for i, s in enumerate(self.segments):
            lo = start if i == 0 else max(start, s.offset_ms)
            hi = end if i + 1 == len(self.segments) else min(end, self.segments[i + 1].offset_ms)
            covered.append(max(0.0, hi - lo))
        return int(np.argmax(covered))

    def _segment_of(self, times: np.ndarray) -> np.ndarray:
        offsets = np.array([s.offset_ms for s in self.segments])
        return np.clip(np.searchsorted(offsets, times, side='right') - 1, 0, len(offsets) - 1)

    def _beat_origins(self) -> np.ndarray:
        """Global canonical beat at each segment's offset."""
        lengths = self.canonical_beat_lengths()
        spans = np.diff([s.offset_ms for s in self.segments]) / lengths[:-1]
        return np.concatenate([[0.0], np.cumsum(spans)])

    def global_beat(self, times_ms) -> np.ndarray:
        times = np.asarray(times_ms, dtype=np.float64)
        segment = self._segment_of(times)
        offsets = np.array([s.offset_ms for s in self.segments])
        return self._beat_origins()[segment] + (times - offsets[segment]) / self.canonical_beat_lengths()[segment]

    def _bar_origins(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Global beat, meter and global bar index of each bar start.

        A bar start within ``SNAP_TOLERANCE_MS`` of a whole canonical beat is
        put on it. A bar cut short by the next bar start counts as a bar unless
        the remainder is within the tolerance.
        """
        times = np.array([b.time_ms for b in self.bar_starts])
        segment = self._segment_of(times)
        lengths = self.canonical_beat_lengths()[segment]
        local = (times - self._segment_start(times)) / lengths
        whole = np.round(local)
        local = np.where(np.abs(local - whole) * lengths <= SNAP_TOLERANCE_MS, whole, local)
        beats = self._beat_origins()[segment] + local
        meters = np.array([b.meter for b in self.bar_starts], dtype=np.float64)
        bars = np.zeros(len(times), dtype=np.int64)
        for i in range(1, len(times)):
            span = beats[i] - beats[i - 1]
            full = math.floor(span / meters[i - 1] + 1e-9)
            remainder_ms = (span - full * meters[i - 1]) * lengths[i]
            bars[i] = bars[i - 1] + full + (remainder_ms > SNAP_TOLERANCE_MS)
        return beats, meters, bars

    def _segment_start(self, times: np.ndarray) -> np.ndarray:
        return np.array([s.offset_ms for s in self.segments])[self._segment_of(times)]

    def locate(self, times_ms) -> dict[str, np.ndarray]:
        """Canonical coordinates of each time.

        Returns arrays: ``segment``; ``beat`` (canonical beats from the segment's
        offset); ``global_beat``; ``bar`` (global bar index); ``bar_beat``
        (canonical beats into the bar); ``snap`` (smallest canonical denominator
        in ``SNAP_DENOMINATORS`` within ``SNAP_TOLERANCE_MS``, 0 when off-grid);
        ``residual_ms`` (signed error to that snap position; for off-grid times,
        to the nearest position of a denominator in ``COMMON_DENOMINATORS``).
        Times before the first segment use the first segment, as osu! does.
        """
        times = np.asarray(times_ms, dtype=np.float64)
        segment = self._segment_of(times)
        lengths = self.canonical_beat_lengths()[segment]
        beat = (times - self._segment_start(times)) / lengths
        snap = np.zeros(times.shape, dtype=np.int16)
        residual = np.full(times.shape, np.nan)
        for d in SNAP_DENOMINATORS:
            error = (beat - np.round(beat * d) / d) * lengths
            hit = (snap == 0) & (np.abs(error) <= SNAP_TOLERANCE_MS)
            snap[hit], residual[hit] = d, error[hit]
        off = snap == 0
        if off.any():
            best = np.full(times.shape, np.inf)
            for d in COMMON_DENOMINATORS:
                error = (beat - np.round(beat * d) / d) * lengths
                closer = off & (np.abs(error) < np.abs(best))
                best[closer] = error[closer]
            residual[off] = best[off]
        global_beat = self._beat_origins()[segment] + beat
        placed = np.where(off, global_beat, global_beat - residual / lengths)
        bar_beats, meters, bar_index = self._bar_origins()
        which = np.clip(np.searchsorted(bar_beats, placed + 1e-9, side='right') - 1, 0, len(bar_beats) - 1)
        into = placed - bar_beats[which]
        local_bar = np.floor(into / meters[which] + 1e-9)
        return dict(segment=segment, beat=beat, global_beat=global_beat,
                    bar=bar_index[which] + local_bar.astype(np.int64),
                    bar_beat=into - local_bar * meters[which], snap=snap, residual_ms=residual)

    def time_of(self, segment, beat) -> np.ndarray:
        """Inverse of ``locate``: milliseconds of a canonical beat in a segment."""
        segment = np.asarray(segment, dtype=np.int64)
        offsets = np.array([s.offset_ms for s in self.segments])
        return offsets[segment] + np.asarray(beat, dtype=np.float64) * self.canonical_beat_lengths()[segment]

    def time_of_bar(self, bar) -> np.ndarray:
        """Milliseconds at which each global bar index starts (extrapolated before the first bar)."""
        bar = np.asarray(bar, dtype=np.float64)
        bar_beats, meters, bar_index = self._bar_origins()
        which = np.clip(np.searchsorted(bar_index, bar, side='right') - 1, 0, len(bar_index) - 1)
        beats = bar_beats[which] + (bar - bar_index[which]) * meters[which]
        origins = self._beat_origins()
        segment = np.clip(np.searchsorted(origins, beats + 1e-9, side='right') - 1, 0, len(origins) - 1)
        return self.time_of(segment, beats - origins[segment])


@dataclass(frozen=True)
class ChartEvents:
    """Columnar events of one chart, ordered by time, then lane.

    ``head_ms`` is the head time of each event's object: a release belongs
    where its long note starts.
    """
    time_ms: np.ndarray
    lane: np.ndarray
    kind: np.ndarray
    head_ms: np.ndarray
    coords: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.time_ms)

    def take(self, mask) -> 'ChartEvents':
        mask = np.asarray(mask)
        return ChartEvents(self.time_ms[mask], self.lane[mask], self.kind[mask], self.head_ms[mask],
                           {k: v[mask] for k, v in self.coords.items()})


def renotate(points: Sequence[RedTimingPoint], factor: int | Sequence[int]) -> list[RedTimingPoint]:
    """The same timing notated at ``2**factor`` times the BPM, meter unchanged.

    ``factor`` is one power for every point or one per point. A chart and its
    renotation describe the same music; their canonical coordinates must agree.
    """
    factors = [factor] * len(points) if isinstance(factor, int) else list(factor)
    return [RedTimingPoint(p.offset_ms, p.beat_length_ms / 2.0 ** f, p.meter) for p, f in zip(points, factors)]


def event_times(objects: Sequence[ManiaHitObject]) -> np.ndarray:
    """Head of every object and release of every long note, in ms."""
    return np.array([o.start_time_ms for o in objects]
                    + [o.end_time_ms for o in objects if o.kind is ManiaHitObjectKind.HOLD], dtype=np.float64)


def chart_events(objects: Sequence[ManiaHitObject], grid: BeatGrid) -> ChartEvents:
    """Taps, long-note heads and long-note releases placed on ``grid``."""
    rows = []
    for o in objects:
        if o.kind is ManiaHitObjectKind.HOLD:
            rows.append((o.start_time_ms, o.lane, LN_HEAD, o.start_time_ms))
            rows.append((o.end_time_ms, o.lane, LN_RELEASE, o.start_time_ms))
        else:
            rows.append((o.start_time_ms, o.lane, TAP, o.start_time_ms))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    time = np.array([r[0] for r in rows], dtype=np.float64)
    return ChartEvents(time, np.array([r[1] for r in rows], dtype=np.int8),
                       np.array([r[2] for r in rows], dtype=np.int8),
                       np.array([r[3] for r in rows], dtype=np.float64), grid.locate(time))
