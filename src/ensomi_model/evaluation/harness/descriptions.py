"""Targeted descriptions: one scalar per defect family, for the corpus-plausible dose.

These are harness measurements, not evaluator descriptions. Each says how far an
injection moved the thing it targets, so that a dose can be placed inside or
outside the corpus distribution at the chart's key. Where a family's dose is
defined by a bound (holds of at most 1/8 canonical beat or 40 ms, releases 40 ms
or less before a head, heads under 20 ms apart), its description counts the
same bound; that bound belongs to the injection, not to any evaluator. Residuals
are continuous distances in ms to the nearest canonical position with a
denominator in ``COMMON_DENOMINATORS``; no snap tolerance is used. Unit-level
descriptions (LN against density, spread of head counts) use the harness's
placement units (``arrays.unit_layout``).
"""
from __future__ import annotations

import math

import numpy as np

from ..beats import COMMON_DENOMINATORS, BeatGrid
from ..case import Chart
from .arrays import Objects, local_beat, object_units, rows

SHORT_HOLD_BEATS = 1.0 / 8
SHORT_HOLD_MS = 40.0
CROWD_MS = 40.0
NEAR_DUPLICATE_MS = 20.0

NAMES = ('run_length', 'short_hold_beats', 'short_hold_ms', 'crowd_same_hand', 'crowd_other_hand',
         'ln_density_corr', 'head_residual_ms', 'adjacent_step_share', 'jack_share', 'triplet_share',
         'density', 'max_lane_share', 'mean_chord', 'bar_repeat_share', 'unit_count_cv', 'ln_share',
         'release_residual_ms', 'near_duplicate_share')


def residual_ms(grid: BeatGrid, times) -> np.ndarray:
    """Distance in ms to the nearest canonical position of any denominator in ``COMMON_DENOMINATORS``."""
    _, beat, length = local_beat(grid, times)
    best = np.full(len(beat), np.inf)
    for d in COMMON_DENOMINATORS:
        best = np.minimum(best, np.abs(beat - np.round(beat * d) / d) * length)
    return best


def _share(mask) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if len(mask) else math.nan


def _runs(present: np.ndarray) -> np.ndarray:
    """Lengths of runs of True."""
    padded = np.concatenate([[False], present, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return np.flatnonzero(diff == -1) - np.flatnonzero(diff == 1)


def run_lengths(o: Objects, keys: int = 4) -> np.ndarray:
    """Same-lane runs: maximal runs of consecutive rows that all hold a head on one lane."""
    times, row = rows(o.start)
    out = []
    for lane in range(keys):
        present = np.zeros(len(times), dtype=bool)
        present[row[o.lane == lane]] = True
        out.append(_runs(present))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def next_heads(o: Objects, lane: int, relation: str, times, *, inclusive: bool = False) -> np.ndarray:
    """First head on ``lane``'s partner lanes after each time (at or after when ``inclusive``); inf if none."""
    heads = np.sort(o.start[np.isin(o.lane, partner_lanes(lane, relation))])
    times = np.asarray(times, dtype=np.float64)
    if not len(heads):
        return np.full(len(times), math.inf)
    idx = np.searchsorted(heads, times, side='left' if inclusive else 'right')
    return np.where(idx < len(heads), heads[np.minimum(idx, len(heads) - 1)], math.inf)


def crowding(o: Objects, relation: str) -> np.ndarray:
    """Gap in ms from each release to the next head on the partner lanes (same or other hand)."""
    idx = np.flatnonzero(o.hold)
    gaps = np.full(len(idx), math.inf)
    for lane in range(4):
        sel = o.lane[idx] == lane
        if sel.any():
            release = o.end[idx[sel]]
            gaps[sel] = next_heads(o, lane, relation, release) - release
    return gaps


def partner_lanes(lane: int, relation: str) -> tuple[int, ...]:
    """Lanes 0-1 are one hand, 2-3 the other."""
    if relation == 'same':
        return (lane ^ 1,)
    return (2, 3) if lane < 2 else (0, 1)


def bar_patterns(o: Objects, grid: BeatGrid) -> list[frozenset]:
    """Head pattern of each non-empty bar from the first to the last: (48th of a canonical beat, lane)."""
    if not len(o):
        return []
    coords = grid.locate(o.start)
    slot = np.round(coords['bar_beat'] * 48).astype(np.int64)
    out = []
    for b in np.unique(coords['bar']):
        m = coords['bar'] == b
        out.append(frozenset(zip(slot[m].tolist(), o.lane[m].tolist())))
    return out


def describe(chart: Chart, grid: BeatGrid) -> dict[str, float]:
    o = Objects.from_chart(chart)
    n = len(o)
    out = {name: math.nan for name in NAMES}
    if not n:
        return out
    times, row = rows(o.start)
    chord = np.bincount(row)
    runs = run_lengths(o)
    out['run_length'] = float((runs ** 2).sum() / runs.sum()) if runs.sum() else math.nan
    length = o.end - o.start
    _, _, beat_ms = local_beat(grid, o.start)
    out['short_hold_beats'] = _share(o.hold & (length <= SHORT_HOLD_BEATS * beat_ms + 1e-9))
    out['short_hold_ms'] = _share(o.hold & (length <= SHORT_HOLD_MS))
    if o.hold.any():
        out['crowd_same_hand'] = _share((lambda g: (g > 0) & (g <= CROWD_MS))(crowding(o, 'same')))
        out['crowd_other_hand'] = _share((lambda g: (g > 0) & (g <= CROWD_MS))(crowding(o, 'other')))
        out['release_residual_ms'] = float(np.median(residual_ms(grid, o.end[o.hold])))
    unit, _ = object_units(o, grid)
    inside = unit >= 0
    if inside.any():
        p = unit[inside]
        heads = np.bincount(p).astype(np.float64)
        lns = np.bincount(p, weights=o.hold[inside].astype(np.float64), minlength=len(heads))
        used = heads > 0
        if used.sum() >= 3:
            share = lns[used] / heads[used]
            if heads[used].std() > 0 and share.std() > 0:
                out['ln_density_corr'] = float(np.corrcoef(heads[used], share)[0, 1])
        if heads.mean() > 0:
            out['unit_count_cv'] = float(heads.std() / heads.mean())
    out['head_residual_ms'] = float(np.median(residual_ms(grid, o.start)))
    single = chord == 1
    lane_of_row = np.zeros(len(times), dtype=np.int64)
    lane_of_row[row[single[row]]] = o.lane[single[row]]
    both = single[:-1] & single[1:]
    out['adjacent_step_share'] = _share(np.abs(np.diff(lane_of_row))[both] == 1)
    masks = np.zeros(len(times), dtype=np.int64)
    np.bitwise_or.at(masks, row, 1 << o.lane)
    out['jack_share'] = _share((masks[:-1] & masks[1:]) > 0)
    _, beat, _ = local_beat(grid, o.start)
    k = np.round((beat - np.floor(beat)) * 48).astype(np.int64)
    out['triplet_share'] = _share(k % 3 != 0)
    seconds = (o.start.max() - o.start.min()) / 1000.0
    out['density'] = float(n / seconds) if seconds > 0 else math.nan
    out['max_lane_share'] = float(np.bincount(o.lane, minlength=4).max() / n)
    out['mean_chord'] = float(n / len(times))
    patterns = bar_patterns(o, grid)
    seen, repeated = set(), 0
    for pattern in patterns:
        repeated += pattern in seen
        seen.add(pattern)
    out['bar_repeat_share'] = repeated / len(patterns) if patterns else math.nan
    out['ln_share'] = float(o.hold.mean())
    out['near_duplicate_share'] = _share(np.diff(times) < NEAR_DUPLICATE_MS) if len(times) > 1 else math.nan
    return out


def density_log_ratio(chart: Chart, grid: BeatGrid | None, t_ms: float) -> float:
    """log of heads per second after ``t_ms`` over heads per second before it (``grid`` unused)."""
    o = Objects.from_chart(chart)
    if not len(o):
        return math.nan
    first, last = o.start.min(), o.start.max()
    if not first < t_ms <= last:
        return math.nan
    given, scored = (o.start < t_ms).sum(), (o.start >= t_ms).sum()
    if given == 0 or scored == 0:
        return math.nan
    return float(math.log((scored / (last - t_ms)) / (given / (t_ms - first)))) if last > t_ms else math.nan
