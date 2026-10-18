"""Label checks of design §3.2, per set.

- `on_grid`: the share of heads within 2 ms of their own notated beat's
  subdivisions (1/1-1/16, 1/3, 1/6, 1/12, 1/24). The expectation is 0.99 or
  more per set; sets below are excluded and listed.
- `expressive`: counts of red lines with an implausible tempo (scroll-speed
  art) and of segments shorter than one notated beat that hold no heads
  (barline art). The reader keeps both; `on_grid` shows whether heads stay on
  the musical grid with them kept.
- `onset_residual`: the median distance from each head-holding canonical line
  to the nearest log-Mel onset within 50 ms. A set whose median is more than
  3 x 7.41 ms from the population median is flagged (weighted 0.5 in training).
- `second_set_bl20`: BL-20 of one set's lines against another set's reference
  for recordings with differing sets (characterises).
"""
from __future__ import annotations

import numpy as np

from ..timing import MERGE_MS

SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
ON_GRID_MIN = 0.99
OFFSET_FLAG_MS = 3 * 7.41


def on_grid(ref) -> float:
    b, step = ref.timing.beats, ref.timing.step_ms
    h = ref.heads if ref.heads is not None else np.zeros(0)
    h = h[ref.scorable(h)] if len(h) else h
    if not len(h) or not len(b):
        return float('nan')
    i = np.searchsorted(b, h + MERGE_MS, side='right') - 1
    ok = i >= 0
    i = np.maximum(i, 0)
    s = step[i]
    x = (h - b[i]) / np.where(np.isfinite(s) & (s > 0), s, 1.0)
    hit = np.zeros(len(h), dtype=bool)
    for d in SUBDIVISIONS:
        hit |= np.abs(x - np.round(x * d) / d) * s <= MERGE_MS
    hit &= ok & np.isfinite(s)
    return float(hit.mean())


def expressive(ref) -> dict:
    segs = ref.segments or ()
    h = ref.heads if ref.heads is not None else np.zeros(0)
    implausible = sum(1 for s in segs if not s.plausible)
    tiny = 0
    for k, s in enumerate(segs[:-1]):
        end = segs[k + 1].start_ms
        if s.plausible and end - s.start_ms < s.beat_ms and not np.any((h >= s.start_ms) & (h < end)):
            tiny += 1
    return dict(implausible=implausible, tiny=tiny)


def onset_times(mel: np.ndarray, frame_ms: float = 10.0) -> np.ndarray:
    """Local maxima of the log-Mel flux above its 70th percentile, ms (parabolic sub-frame)."""
    from ..features.mel import onset_strength
    f = onset_strength(mel).astype(np.float64)
    if len(f) < 3:
        return np.zeros(0)
    thr = np.percentile(f, 70)
    i = np.flatnonzero((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]) & (f[1:-1] > thr)) + 1
    y0, y1, y2 = f[i - 1], f[i], f[i + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    return (i + frac) * frame_ms


def onset_residual(lab, onsets: np.ndarray, window_ms: float = 50.0) -> float:
    """Median (onset - line) over head-holding canonical lines with an onset within ``window_ms``."""
    lines = lab.lines[(lab.lines >= lab.lo_ms - 1) & (lab.lines <= lab.hi_ms + 1)]
    if not len(lines) or not len(onsets):
        return float('nan')
    held = np.isin(np.round(lines, 1), np.round(lab.heads, 1)) | (
        np.abs(lab.heads[np.clip(np.searchsorted(lab.heads, lines), 0, len(lab.heads) - 1)] - lines) <= MERGE_MS)
    lines = lines[held]
    if not len(lines):
        return float('nan')
    j = np.clip(np.searchsorted(onsets, lines), 1, max(len(onsets) - 1, 1))
    cand = np.stack([onsets[j - 1], onsets[np.minimum(j, len(onsets) - 1)]])
    d = cand - lines[None, :]
    best = d[np.argmin(np.abs(d), axis=0), np.arange(len(lines))]
    best = best[np.abs(best) <= window_ms]
    return float(np.median(best)) if len(best) else float('nan')


def second_set_bl20(ref_a, ref_b) -> float:
    """BL-20 of set B's timing scored against set A's reference."""
    from ..eval import score
    return float(score(ref_a, ref_b.timing).bl20)
