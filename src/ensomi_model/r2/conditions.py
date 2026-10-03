"""Interval condition tracks: training draws, dropout and the runtime edit operation.

A track is a tuple of ``Interval(kind, a, b, value)`` with half-open [a, b);
kind 0 is LN share, kind 1 is tiled star. Intervals of one kind never overlap.
An empty track is natural mode. Training draws (spec section 5): LN intervals
partition the song from time zero into 8/16/32/64-beat pieces and keep 1-4
nonempty ones (consecutive with p 0.5); star uses the cached whole-song label
with p 0.10, else 1-3 cached cells of one length; then dropout 0.20 of all
tracks, 0.25 per kind and 0.20 per interval.
"""
from __future__ import annotations

import math

import numpy as np

from .common import ContractError
from .features import Chart, Interval

LN_BEATS = (8, 16, 32, 64)
P_WHOLE_SONG = 0.10
P_CONSECUTIVE = 0.5
DROP_ALL, DROP_KIND, DROP_INTERVAL = 0.20, 0.25, 0.20


def _choose(rng, items, count):
    if rng.random() < P_CONSECUTIVE:
        s = int(rng.integers(0, len(items) - count + 1))
        return items[s:s + count]
    idx = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[i] for i in idx]


def ln_intervals(chart: Chart, rng) -> list[Interval]:
    g = chart.grid
    beat = float(g.beat(0.0))
    end_beat = float(g.beat(chart.song_ms))
    bounds = [0.0]
    while beat < end_beat:
        beat += LN_BEATS[int(rng.integers(0, len(LN_BEATS)))]
        bounds.append(min(float(g.time_of_beat(beat)), chart.song_ms))
    d = chart.derived()
    heads = d.attack.sum(1)[:chart.K]
    lns = d.ln_head.sum(1)[:chart.K]
    cand = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        lo, hi = np.searchsorted(chart.head_ms, [a, b], side='left')
        n = int(heads[lo:hi].sum())
        if n:
            cand.append(Interval(0, a, b, float(lns[lo:hi].sum()) / n))
    if not cand:
        return []
    count = int(rng.integers(1, min(4, len(cand)) + 1))
    return _choose(rng, cand, count)


def star_intervals(chart: Chart, labels, rng) -> list[Interval]:
    if not labels:
        return []
    whole = [(a, b, v) for a, b, v in labels if a == 0.0 and b == chart.song_ms]
    local = {}
    for a, b, v in labels:
        if not (a == 0.0 and b == chart.song_ms):
            local.setdefault(round(b - a), []).append((a, b, v))
    if whole and (rng.random() < P_WHOLE_SONG or not local):
        a, b, v = whole[0]
        return [Interval(1, a, b, v)]
    if not local:
        return []
    lengths = sorted(local)
    cells = sorted(local[lengths[int(rng.integers(0, len(lengths)))]])
    count = int(rng.integers(1, min(3, len(cells)) + 1))
    return [Interval(1, a, b, v) for a, b, v in _choose(rng, cells, count)]


def draw_track(chart: Chart, star_labels, rng, *, star_conditions: bool):
    """A training condition track after dropout; returns (track, dropout record)."""
    ln = ln_intervals(chart, rng)
    star = star_intervals(chart, star_labels, rng) if star_conditions else []
    record = dict(drawn_ln=len(ln), drawn_star=len(star), drop_all=False)
    if rng.random() < DROP_ALL:
        record['drop_all'] = True
        return (), record
    kept = []
    for kind, ivs in ((0, ln), (1, star)):
        if rng.random() < DROP_KIND:
            continue
        kept += [iv for iv in ivs if rng.random() >= DROP_INTERVAL]
    return tuple(kept), record


def validate_track(track):
    for kind in (0, 1):
        ivs = sorted((iv for iv in track if iv.kind == kind), key=lambda iv: iv.a)
        for iv in ivs:
            if not (math.isfinite(iv.a) and math.isfinite(iv.b) and iv.b > iv.a):
                raise ContractError('Interval bounds must be finite with positive length')
            if kind == 0 and not 0.0 <= iv.value <= 1.0 or kind == 1 and not iv.value >= 0.0:
                raise ContractError('LN share must be in [0,1] and star nonnegative')
        if any(x.b > y.a for x, y in zip(ivs, ivs[1:])):
            raise ContractError('Intervals of one kind may not overlap')


def replace_interval(track, kind: int, a: float, b: float, value: float, frontier: float | None):
    """Replace [a, b) of one kind's track from the next decision on; returns (track, (a_eff, b))."""
    if frontier is not None:
        a = max(a, float(np.nextafter(frontier, math.inf)))
    if not b > a:
        raise ContractError('Edit span is empty after clipping to the uncommitted suffix')
    out = [iv for iv in track if iv.kind != kind]
    same = []
    for iv in track:
        if iv.kind != kind:
            continue
        if iv.b <= a or iv.a >= b:
            same.append(iv)
            continue
        if iv.a < a:
            same.append(Interval(kind, iv.a, a, iv.value))
        if iv.b > b:
            same.append(Interval(kind, b, iv.b, iv.value))
    same.append(Interval(kind, a, b, value))
    same.sort(key=lambda iv: iv.a)
    merged = []
    for iv in same:
        if merged and merged[-1].b == iv.a and merged[-1].value == iv.value:
            merged[-1] = Interval(kind, merged[-1].a, iv.b, iv.value)
        else:
            merged.append(iv)
    new = tuple(out + merged)
    validate_track(new)
    return new, (a, b)
