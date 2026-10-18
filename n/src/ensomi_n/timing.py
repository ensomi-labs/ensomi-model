"""Notated beats, segments and canonical beat lines.

A `Timing` is a list of notated beats: the beats as a chart or a model writes
them, at its own tempo level. Each beat carries:

- ``downbeat``: whether it starts a bar;
- ``seg_start``: whether it starts a segment (a red line);
- ``step_ms``: the notated beat length in force at it. When segments are
  known this is the segment's beat length, exactly; for a bare beat list it is
  the median of the five beat intervals around the beat, so that tracker
  jitter does not flip the fold.

`Timing.canonical` gives the canonical beat lines. Each beat's notated tempo
is moved into [80, 160) BPM by a power of two (its fold), so that timings
notated at half or double the tempo give the same lines:

- a beat slower than 80 BPM keeps its line and gains ``2**fold - 1`` lines
  after it, one canonical beat apart, up to the next beat (with exact steps)
  or evenly between it and the next beat (for a bare beat list);
- a beat at 160 BPM or faster is a line only when its count since the last
  anchor is a multiple of ``2**-fold``. Anchors are the first beat, every
  segment start, every change of fold, and every downbeat whose bar holds at
  least ``2**-fold`` notated beats.

So a downbeat is always a line, unless its bar is shorter than one canonical
beat (a meter set for visual effect). 7/8 notated at 240 BPM with meter 7 gives
the same lines as 7/8 at 120 BPM with a reset on every bar. Where neither
segment starts nor downbeats are given, the first beat fixes the phase.

Times are milliseconds on one clock. Nothing here rounds a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

CANONICAL_BPM_MIN = 80.0
CANONICAL_BPM_MAX = 160.0
MIN_BPM = 20.0
MAX_BPM = 1000.0
MIN_BEAT_MS = 60000.0 / MAX_BPM
MAX_BEAT_MS = 60000.0 / MIN_BPM
#: Two beats or lines closer than this are one; a segment start this close
#: after a beat of the previous segment replaces that beat.
MERGE_MS = 2.0


def fold_of(step_ms) -> np.ndarray:
    """Power of two moving each notated beat length into [80, 160) BPM.

    ``canonical BPM = 60000 / step_ms * 2**fold``. Non-finite or non-positive
    steps get fold 0.
    """
    step = np.atleast_1d(np.asarray(step_ms, dtype=np.float64))
    ok = np.isfinite(step) & (step > 0)
    bpm = np.where(ok, 60000.0 / np.where(ok, step, 1.0), CANONICAL_BPM_MIN)
    fold = np.ceil(np.log2(CANONICAL_BPM_MIN / bpm)).astype(np.int64)
    fold[bpm * np.exp2(fold) >= CANONICAL_BPM_MAX] -= 1
    fold[bpm * np.exp2(fold) < CANONICAL_BPM_MIN] += 1
    return np.where(ok, fold, 0)


def canonical_beat_ms(step_ms) -> np.ndarray:
    step = np.asarray(step_ms, dtype=np.float64)
    return step / np.exp2(fold_of(step).reshape(step.shape))


@dataclass(frozen=True)
class Segment:
    """A red line: the notated beat length and meter from ``start_ms`` on.

    Bars restart at the start. ``bar_at_start`` is False for osu!'s
    omit-first-barline flag: the start is then no downbeat, and the first bar
    line falls ``meter`` beats later, as osu! draws it.
    """
    start_ms: float
    beat_ms: float
    meter: int = 4
    bar_at_start: bool = True

    @property
    def plausible(self) -> bool:
        """Finite, with a tempo in [20, 1000] BPM. Implausible lines carry no beats."""
        return (math.isfinite(self.start_ms) and math.isfinite(self.beat_ms)
                and MIN_BEAT_MS <= self.beat_ms <= MAX_BEAT_MS)

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_ms


@dataclass(frozen=True)
class Lines:
    """Canonical beat lines.

    ``beat_ms`` is the canonical beat length at each line. ``src_lo`` and
    ``src_hi`` index the notated beats a line comes from: the same beat for a
    kept beat, the beats on either side for an inserted line.
    """
    t: np.ndarray
    beat_ms: np.ndarray
    src_lo: np.ndarray
    src_hi: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    def take(self, sl) -> 'Lines':
        return Lines(self.t[sl], self.beat_ms[sl], self.src_lo[sl], self.src_hi[sl])


def _running_median_step(beats: np.ndarray) -> np.ndarray:
    n = len(beats)
    if n < 2:
        return np.full(n, np.nan)
    d = np.diff(beats)
    padded = np.concatenate([np.full(2, np.nan), d, np.full(3, np.nan)])
    # Beat i's window holds intervals d[i-2 .. i+2]; padded index = d index + 2.
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5)
    return np.nanmedian(windows[:n], axis=1)


@dataclass(frozen=True)
class Timing:
    """Notated beats with their bar starts, segment starts and steps.

    Build one with `from_segments` or `from_beats`. ``exact_steps`` is True
    when steps come from segments; inserted canonical lines then sit one
    canonical beat apart from the beat before them, as a red line places them.
    ``has_bars`` is False when the source gave no bar information.
    """
    beats: np.ndarray
    downbeat: np.ndarray
    seg_start: np.ndarray
    step_ms: np.ndarray
    exact_steps: bool = True
    has_bars: bool = True
    _lines: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        b = self.beats
        if b.ndim != 1 or not np.all(np.isfinite(b)):
            raise ValueError('Beat times must be a finite 1-D array')
        if len(b) > 1 and not np.all(np.diff(b) > 0):
            i = int(np.flatnonzero(np.diff(b) <= 0)[0])
            raise ValueError(f'Beat times must increase strictly: beat {i + 1} at {b[i + 1]!r} '
                             f'follows {b[i]!r}')
        for name in ('downbeat', 'seg_start', 'step_ms'):
            if len(getattr(self, name)) != len(b):
                raise ValueError(f'{name} needs one entry per beat')

    def __len__(self) -> int:
        return len(self.beats)

    @classmethod
    def empty(cls) -> 'Timing':
        z = np.zeros(0)
        return cls(z, z.astype(bool), z.astype(bool), z, True, False)

    @classmethod
    def from_segments(cls, segments: Sequence[Segment], lo_ms: float, hi_ms: float) -> 'Timing':
        """The beats red lines place from ``lo_ms`` to ``hi_ms``, with two beats' margin.

        ``segments`` must have strictly increasing starts (`labels` resolves
        lines sharing an offset). The first segment also governs times before
        its start, as osu! applies it; its backward beats begin on an anchor
        (a multiple of ``lcm(meter, 8)`` beats), so the range never moves the
        canonical phase. A segment's beats stop more than ``MERGE_MS`` before
        the next segment's start. Implausible segments place no beats.
        """
        starts = [s.start_ms for s in segments]
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError('Segment starts must increase strictly')
        parts = []
        for k, s in enumerate(segments):
            if not s.plausible:
                continue
            step = s.beat_ms
            end = segments[k + 1].start_ms if k + 1 < len(segments) else math.inf
            limit = min(end - MERGE_MS, hi_ms + 2.0 * step)
            if limit <= s.start_ms:
                continue
            j_lo = 0
            if k == 0 and lo_ms - 2.0 * step < s.start_ms:
                anchor = math.lcm(max(int(s.meter), 1), 8)
                j_lo = (math.floor((lo_ms - s.start_ms) / step) - 2) // anchor * anchor
            j_hi = math.ceil((limit - s.start_ms) / step) - 1
            j = np.arange(j_lo, j_hi + 1, dtype=np.int64)
            meter = max(int(s.meter), 1)
            down = (j % meter == 0) & ~((j == 0) & (not s.bar_at_start))
            parts.append((s.start_ms + j * step, down, j == 0, np.full(len(j), step)))
        if not parts:
            return cls.empty()
        cat = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        return cls(cat[0].astype(np.float64), cat[1], cat[2], cat[3], True, True)

    @classmethod
    def from_beats(cls, beats, downbeats=None, segments: Sequence[Segment] | None = None) -> 'Timing':
        """A timing from a beat list, as a model writes it.

        ``downbeats`` is an index list or a boolean mask. Without it, bar
        starts come from ``segments`` (beats within ``MERGE_MS`` of a segment's
        bar grid), and without either the timing has no bars. With
        ``segments``, each beat takes the beat length of the segment holding
        it (a beat within ``MERGE_MS`` before a start belongs to the new
        segment) and steps are exact; without, steps are running medians.
        """
        b = np.asarray(beats, dtype=np.float64).reshape(-1)
        n = len(b)
        seg_start = np.zeros(n, dtype=bool)
        has_bars = downbeats is not None
        down = np.zeros(n, dtype=bool)
        if downbeats is not None:
            d = np.asarray(downbeats)
            if d.dtype == bool:
                if len(d) != n:
                    raise ValueError('A downbeat mask needs one entry per beat')
                down = d.copy()
            elif len(d):
                idx = d.astype(np.int64)
                if np.any((idx < 0) | (idx >= n)):
                    raise ValueError('Downbeat indices must index the beats')
                down[idx] = True
        segs = [s for s in (segments or ()) if s.plausible]
        if segs:
            starts = np.array([s.start_ms for s in segs])
            which = np.clip(np.searchsorted(starts, b + MERGE_MS, side='right') - 1, 0, len(segs) - 1)
            step = np.array([segs[k].beat_ms for k in which], dtype=np.float64)
            if n:
                near = np.searchsorted(b, starts - MERGE_MS, side='left')
                for k, i in enumerate(near):
                    if i < n and abs(b[i] - starts[k]) <= MERGE_MS:
                        seg_start[i] = True
            if downbeats is None:
                has_bars = True
                for k, s in enumerate(segs):
                    end = starts[k + 1] if k + 1 < len(segs) else math.inf
                    inside = (which == k) & (b < end - MERGE_MS)
                    pos = (b[inside] - s.start_ms) / s.beat_ms
                    j = np.round(pos)
                    on = (np.abs(pos - j) * s.beat_ms <= MERGE_MS) & (j % max(int(s.meter), 1) == 0)
                    on &= ~((j == 0) & (not s.bar_at_start)) & (j >= 0 if k > 0 else True)
                    down[np.flatnonzero(inside)[on]] = True
            exact = True
        else:
            step = _running_median_step(b)
            exact = False
        return cls(b, down, seg_start, step, exact, has_bars)

    def scaled(self, factor: float, shift_ms: float = 0.0) -> 'Timing':
        """The same timing played at ``1 / factor`` the speed: times and steps times ``factor``."""
        return Timing(self.beats * factor + shift_ms, self.downbeat, self.seg_start, self.step_ms * factor,
                      self.exact_steps, self.has_bars)

    def shifted(self, shift_ms: float) -> 'Timing':
        return self.scaled(1.0, shift_ms)

    def bar_starts(self) -> np.ndarray:
        return self.beats[self.downbeat] if self.has_bars else np.zeros(0)

    def step_at(self, times_ms) -> np.ndarray:
        """Notated beat length in force at each time; NaN outside the beats."""
        t = np.atleast_1d(np.asarray(times_ms, dtype=np.float64))
        out = np.full(t.shape, np.nan)
        if not len(self.beats):
            return out
        i = np.searchsorted(self.beats, t + MERGE_MS, side='right') - 1
        ok = (i >= 0) & (t <= self.beats[-1] + np.nan_to_num(self.step_ms[-1], nan=0.0) + MERGE_MS)
        out[ok] = self.step_ms[i[ok]]
        return out

    def folds(self) -> np.ndarray:
        return fold_of(self.step_ms) if len(self.beats) else np.zeros(0, dtype=np.int64)

    def anchors(self) -> np.ndarray:
        """Boolean mask of the beats that restart the fast-beat count (module docstring)."""
        n = len(self.beats)
        if not n:
            return np.zeros(0, dtype=bool)
        f = self.folds()
        idx = np.arange(n)
        restart = (idx == 0) | self.seg_start
        restart[1:] |= f[1:] != f[:-1]
        if self.has_bars and self.downbeat.any():
            bounds = np.flatnonzero(self.downbeat | self.seg_start)
            nxt = bounds[np.minimum(np.searchsorted(bounds, idx, side='right'), len(bounds) - 1)]
            has_next = np.searchsorted(bounds, idx, side='right') < len(bounds)
            bar_beats = np.where(has_next & (nxt > idx), nxt - idx, np.iinfo(np.int64).max)
            restart |= self.downbeat & ((f >= 0) | (bar_beats >= np.exp2(-np.minimum(f, 0))))
        return restart

    def canonical(self) -> Lines:
        """The canonical beat lines (module docstring). Cached."""
        if self._lines:
            return self._lines[0]
        n = len(self.beats)
        if not n:
            z = np.zeros(0)
            lines = Lines(z, z, z.astype(np.int64), z.astype(np.int64))
            self._lines.append(lines)
            return lines
        f = self.folds()
        idx = np.arange(n)
        last_anchor = np.maximum.accumulate(np.where(self.anchors(), idx, 0))
        count = idx - last_anchor
        period = np.exp2(-np.minimum(f, 0)).astype(np.int64)
        keep = (f >= 0) | (count % period == 0)
        step = np.where(np.isfinite(self.step_ms), self.step_ms, 0.0)
        t = [self.beats[keep]]
        length = [np.where(np.isfinite(self.step_ms[keep]), step[keep] / np.exp2(f[keep]), np.nan)]
        lo = [idx[keep]]
        hi = [idx[keep]]
        for fold in np.unique(f[f > 0]):
            m = int(2 ** fold)
            i = idx[(f == fold) & (idx < n - 1)]
            if not len(i):
                continue
            k = np.arange(1, m)
            if self.exact_steps:
                pos = self.beats[i, None] + k[None, :] * (step[i, None] / m)
                ok = pos < self.beats[i + 1, None] - MERGE_MS
                beat = np.broadcast_to((step[i] / m)[:, None], pos.shape)
            else:
                gap = self.beats[i + 1] - self.beats[i]
                pos = self.beats[i, None] + k[None, :] * (gap[:, None] / m)
                ok = np.ones(pos.shape, dtype=bool)
                beat = np.broadcast_to((gap / m)[:, None], pos.shape)
            t.append(pos[ok])
            length.append(beat[ok])
            lo.append(np.broadcast_to(i[:, None], pos.shape)[ok])
            hi.append(np.broadcast_to(i[:, None] + 1, pos.shape)[ok])
        t_all = np.concatenate(t)
        order = np.argsort(t_all, kind='stable')
        lines = Lines(t_all[order], np.concatenate(length)[order], np.concatenate(lo)[order].astype(np.int64),
                      np.concatenate(hi)[order].astype(np.int64))
        self._lines.append(lines)
        return lines


@dataclass(frozen=True)
class Reference:
    """What a prediction is scored against: a timing, the heads that weight it, and the scored range.

    ``lo_ms`` and ``hi_ms`` bound the scored range, normally the first and
    last head. The reference lines run from the last line at or before
    ``lo_ms`` to the first line at or after ``hi_ms``, so every head lies
    between two of them. ``heads`` of None weighs every line 1. Heads inside
    an ``unscorable`` span (an implausible red line's) weigh nothing.
    ``segments`` are the red lines when the source has them; the level,
    reset, re-time and redo terms read them.
    """
    timing: Timing
    lo_ms: float
    hi_ms: float
    heads: np.ndarray | None = None
    segments: tuple[Segment, ...] | None = None
    unscorable: tuple[tuple[float, float], ...] = ()
    key: str = ''
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_segments(cls, segments: Sequence[Segment], heads=None, *, lo_ms: float | None = None,
                      hi_ms: float | None = None, key: str = '', meta: dict | None = None) -> 'Reference':
        """A reference from red lines. The range defaults to the first and last head."""
        segments = tuple(segments)
        h = None if heads is None else np.unique(np.asarray(heads, dtype=np.float64))
        if lo_ms is None or hi_ms is None:
            if h is None or not len(h):
                raise ValueError('A reference from segments needs heads or an explicit range')
            lo_ms = float(h[0]) if lo_ms is None else lo_ms
            hi_ms = float(h[-1]) if hi_ms is None else hi_ms
        unscorable = []
        for k, s in enumerate(segments):
            if not s.plausible:
                end = segments[k + 1].start_ms if k + 1 < len(segments) else math.inf
                unscorable.append((s.start_ms, end))
        # One bar past the range, so that the bar holding the last head has its end.
        margin = max((min(max(s.meter, 1), 64) * s.beat_ms for s in segments if s.plausible), default=0.0)
        return cls(Timing.from_segments(segments, lo_ms, hi_ms + margin), float(lo_ms), float(hi_ms), h,
                   segments, tuple(unscorable), key, dict(meta or {}))

    @classmethod
    def from_beats(cls, beats, heads=None, *, downbeats=None, segments: Sequence[Segment] | None = None,
                   lo_ms: float | None = None, hi_ms: float | None = None, key: str = '') -> 'Reference':
        """A reference from a beat list (synthetic ground truth, or labels already read).

        The range defaults to the heads' span, or to the beats' span without heads.
        """
        timing = Timing.from_beats(beats, downbeats, segments)
        h = None if heads is None else np.unique(np.asarray(heads, dtype=np.float64))
        if lo_ms is None:
            lo_ms = float(h[0]) if h is not None and len(h) else float(timing.beats[0])
        if hi_ms is None:
            hi_ms = float(h[-1]) if h is not None and len(h) else float(timing.beats[-1])
        return cls(timing, float(lo_ms), float(hi_ms), h, tuple(segments) if segments else None, (), key)

    def lines(self) -> Lines:
        """The scored reference lines (see the class docstring)."""
        all_lines = self.timing.canonical()
        t = all_lines.t
        a = max(int(np.searchsorted(t, self.lo_ms + 1e-9, side='right')) - 1, 0)
        b = min(int(np.searchsorted(t, self.hi_ms - 1e-9, side='left')), len(t) - 1)
        return all_lines.take(slice(a, b + 1))

    def scorable(self, times) -> np.ndarray:
        t = np.asarray(times, dtype=np.float64)
        ok = np.ones(t.shape, dtype=bool)
        for a, b in self.unscorable:
            ok &= ~((t >= a) & (t < b))
        return ok

    def weights(self, lines: Lines | None = None) -> np.ndarray:
        """Head weight of each scored line.

        Each head splits its weight between the two lines around it, in
        proportion to its closeness, as its position is interpolated from
        them; a head on a line gives that line all of it.
        """
        lines = self.lines() if lines is None else lines
        t = lines.t
        if self.heads is None:
            return np.ones(len(t))
        w = np.zeros(len(t))
        h = self.heads[self.scorable(self.heads)]
        if not len(t) or not len(h):
            return w
        k = np.clip(np.searchsorted(t, h, side='right') - 1, 0, len(t) - 1)
        nxt = np.minimum(k + 1, len(t) - 1)
        gap = t[nxt] - t[k]
        frac = np.where(gap > 0, np.clip((h - t[k]) / np.where(gap > 0, gap, 1.0), 0.0, 1.0), 0.0)
        np.add.at(w, k, 1.0 - frac)
        np.add.at(w, nxt, frac)
        return w
