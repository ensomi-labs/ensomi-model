"""Legality of a 4K chart: what any generator's output must satisfy before it is judged.

A chart is legal when every object time is finite and inside the song, every
lane is in ``0..keys-1``, every long note's release is strictly after its head,
every tap ends where it starts, and no two objects overlap in one lane. A long
note occupies its lane from head to release, both included, so a head at
another object's release time on the same lane overlaps it; a tap occupies one
instant, so two taps at one time on one lane overlap.

This is not R1's row contract (``research/bounded_typed_continuation``), which
is tied to R1's timing and row interface; it holds for any ``.osu`` chart.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind

NONFINITE, LANE, RELEASE, TAP_LENGTH, OVERLAP = 'nonfinite', 'lane', 'release_not_after_head', 'tap_length', 'overlap'
BEFORE_SONG, AFTER_SONG, EMPTY = 'before_song', 'after_song', 'empty'


def violations(objects: Sequence[ManiaHitObject], *, keys: int = 4,
               song_span: tuple[float, float] | None = None) -> dict[str, int]:
    """Count of each rule an object list breaks; empty when the chart is legal.

    ``song_span`` is ``(start_ms, end_ms)``: every head and release must lie in
    it, both ends included. A chart without objects breaks ``empty``.
    """
    if not objects:
        return {EMPTY: 1}
    return violations_arrays(np.array([o.start_time_ms for o in objects], dtype=np.float64),
                             np.array([o.end_time_ms for o in objects], dtype=np.float64),
                             np.array([o.lane for o in objects], dtype=np.int64),
                             np.array([o.kind is ManiaHitObjectKind.HOLD for o in objects]),
                             keys=keys, song_span=song_span)


def violations_arrays(start, end, lane, hold, *, keys: int = 4,
                      song_span: tuple[float, float] | None = None) -> dict[str, int]:
    """``violations`` on columns: head and release times, lane and long-note flag of each object."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    lane, hold = np.asarray(lane, dtype=np.int64), np.asarray(hold, dtype=bool)
    if not len(start):
        return {EMPTY: 1}
    out: Counter[str] = Counter()
    finite = np.isfinite(start) & np.isfinite(end)
    out[NONFINITE] = int((~finite).sum())
    out[LANE] = int(((lane < 0) | (lane >= keys)).sum())
    out[RELEASE] = int((hold & finite & ~(end > start)).sum())
    out[TAP_LENGTH] = int((~hold & finite & (end != start)).sum())
    if song_span is not None:
        lo, hi = float(song_span[0]), float(song_span[1])
        out[BEFORE_SONG] = int((finite & (start < lo)).sum())
        out[AFTER_SONG] = int((finite & (np.maximum(start, end) > hi)).sum())
    ok = finite & (lane >= 0) & (lane < keys)
    for k in range(keys):
        mask = ok & (lane == k)
        if mask.sum() < 2:
            continue
        s, e = start[mask], np.maximum(end[mask], start[mask])
        order = np.lexsort((e, s))
        s, e = s[order], e[order]
        covered = np.maximum.accumulate(e)[:-1]
        out[OVERLAP] += int((s[1:] <= covered).sum())
    return {k: v for k, v in sorted(out.items()) if v}


def is_legal(objects: Sequence[ManiaHitObject], *, keys: int = 4,
             song_span: tuple[float, float] | None = None) -> bool:
    return not violations(objects, keys=keys, song_span=song_span)
