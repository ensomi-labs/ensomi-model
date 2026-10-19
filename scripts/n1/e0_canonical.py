"""E0's canonical lines, frozen from ``n1/eval-harness`` at ``be69bf9``, for E3's before-and-after count.

The reader fix (``cb41e0f``) replaced E0's canonical phase rule, and ``3821eb9``
stopped barline art from starting bars. E3 compares both readings with R's on
identical strata in one pass, so E0's reading is kept here as it was, outside
the ``ensomi_n`` package: `Timing.from_segments` (beats, downbeats, segment
starts and steps), `Timing.anchors` and the line placement of
`Timing.canonical`, with `Reference.lines`'s scored range. Beats are placed
exactly as ``ensomi_n.timing`` places them; only the anchors and the downbeats
they read differ. Do not change this file to follow ``ensomi_n``.

E0's rule: a beat at 160 BPM or more is a line when its count since the last
anchor is a multiple of p = ``2**-fold``; anchors are the first beat, every
segment start, every change of fold, and every downbeat whose bar (to the next
downbeat or segment start) holds at least p notated beats.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ensomi_n.timing import MERGE_MS, Segment, fold_of


def _place(segments: Sequence[Segment], lo_ms: float, hi_ms: float):
    """``be69bf9``'s `Timing.from_segments`: beats, downbeats, segment starts and steps."""
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
        z = np.zeros(0)
        return z, z.astype(bool), z.astype(bool), z
    cat = [np.concatenate([p[i] for p in parts]) for i in range(4)]
    return cat[0].astype(np.float64), cat[1], cat[2], cat[3]


def _anchors(beats, downbeat, seg_start, f) -> np.ndarray:
    n = len(beats)
    idx = np.arange(n)
    restart = (idx == 0) | seg_start
    restart[1:] |= f[1:] != f[:-1]
    if downbeat.any():
        bounds = np.flatnonzero(downbeat | seg_start)
        nxt = bounds[np.minimum(np.searchsorted(bounds, idx, side='right'), len(bounds) - 1)]
        has_next = np.searchsorted(bounds, idx, side='right') < len(bounds)
        bar_beats = np.where(has_next & (nxt > idx), nxt - idx, np.iinfo(np.int64).max)
        restart |= downbeat & ((f >= 0) | (bar_beats >= np.exp2(-np.minimum(f, 0))))
    return restart


def canonical(segments: Sequence[Segment], lo_ms: float, hi_ms: float) -> np.ndarray:
    """E0's canonical line times for the beats ``segments`` place over ``[lo_ms, hi_ms]`` (with margin)."""
    beats, downbeat, seg_start, step_ms = _place(segments, lo_ms, hi_ms)
    n = len(beats)
    if not n:
        return np.zeros(0)
    f = fold_of(step_ms)
    idx = np.arange(n)
    last_anchor = np.maximum.accumulate(np.where(_anchors(beats, downbeat, seg_start, f), idx, 0))
    count = idx - last_anchor
    period = np.exp2(-np.minimum(f, 0)).astype(np.int64)
    keep = (f >= 0) | (count % period == 0)
    step = np.where(np.isfinite(step_ms), step_ms, 0.0)
    t = [beats[keep]]
    for fold in np.unique(f[f > 0]):
        m = int(2 ** fold)
        i = idx[(f == fold) & (idx < n - 1)]
        if not len(i):
            continue
        k = np.arange(1, m)
        pos = beats[i, None] + k[None, :] * (step[i, None] / m)
        ok = pos < beats[i + 1, None] - MERGE_MS
        t.append(pos[ok])
    return np.sort(np.concatenate(t), kind='stable')


def reference_lines(ref) -> np.ndarray:
    """E0's scored lines of an ``ensomi_n`` `Reference` built from segments, as ``be69bf9``'s `Reference.lines`."""
    segs = tuple(ref.segments or ())
    margin = max((min(max(s.meter, 1), 64) * s.beat_ms for s in segs if s.plausible), default=0.0)
    t = canonical(segs, ref.lo_ms, ref.hi_ms + margin)
    if not len(t):
        return t
    a = max(int(np.searchsorted(t, ref.lo_ms + 1e-9, side='right')) - 1, 0)
    b = min(int(np.searchsorted(t, ref.hi_ms - 1e-9, side='left')), len(t) - 1)
    return t[a:b + 1]
