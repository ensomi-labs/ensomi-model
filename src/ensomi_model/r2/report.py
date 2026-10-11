"""Per-checkpoint free-run summary of one generated chart (spec section 7), with its source for scale."""
from __future__ import annotations

import numpy as np

from ..evaluation.legality import violations
from ..osu_core.hitobjects import ManiaHitObjectKind
from .candidates import build_candidates
from .common import GridArrays

HOLD = ManiaHitObjectKind.HOLD


def chart_summary(objects, head_ms, song_ms: float, grid: GridArrays) -> dict:
    head_ms = np.asarray(head_ms)
    starts = np.array([o.start_time_ms for o in objects])
    lanes = np.array([o.lane for o in objects])
    holds = [o for o in objects if o.kind is HOLD]
    n, n_ln = len(objects), len(holds)
    rows, counts = np.unique(starts, return_counts=True)
    lengths = np.array([o.end_time_ms - o.start_time_ms for o in holds])
    cats = dict(on_row=0, midpoint=0, eighth=0, quarter=0, other=0)
    near_head = 0
    head_set = set(head_ms.tolist())
    for o in holds:
        r = o.end_time_ms
        later = (starts >= r + 1) & (starts <= r + 40) & (lanes != o.lane)
        near_head += bool(later.any())
        if r in head_set:
            cats['on_row'] += 1
            continue
        q = int(np.searchsorted(head_ms, r))
        cands = build_candidates(head_ms[q - 1], head_ms[q] if q < len(head_ms) else song_ms, grid)
        hit = np.flatnonzero(cands.times == r)
        flags = cands.flags[hit[0]] if len(hit) else np.zeros(4, dtype=bool)
        cats['midpoint' if flags[0] else 'eighth' if flags[1] else 'quarter' if flags[2] else 'other'] += 1
    # open-hold occupancy at head rows: holds with start < t < end
    occupancy = 0.0
    if holds and len(head_ms):
        hs = np.array([o.start_time_ms for o in holds])
        he = np.array([o.end_time_ms for o in holds])
        occupancy = float(np.mean([((hs < t) & (he > t)).sum() for t in head_ms]))
    span = (head_ms[0], head_ms[-1]) if len(head_ms) else (0.0, 0.0)
    third = (span[1] - span[0]) / 3.0

    def share(mask):
        m = int(mask.sum())
        return None if m == 0 else float(sum(o.kind is HOLD for o, keep in zip(objects, mask) if keep)) / m

    chord = np.bincount(np.minimum(counts, 4), minlength=5)[1:]
    return dict(
        violations=violations(objects, song_span=(0.0, float(song_ms))),
        heads_present=bool(len(rows) == len(head_ms) and np.array_equal(rows, head_ms)),
        objects=n, ln=n_ln, ln_share=(n_ln / n) if n else None,
        hold_le40=float((lengths <= 40).mean()) if n_ln else None,
        hold_le60=float((lengths <= 60).mean()) if n_ln else None,
        release_1_40_before_other_head=(near_head / n_ln) if n_ln else None,
        release_categories={k: (v / n_ln if n_ln else None) for k, v in cats.items()},
        chord_hist=(chord / max(1, chord.sum())).tolist(),
        open_hold_occupancy=occupancy,
        ln_share_first_third=share(starts < span[0] + third),
        ln_share_last_third=share(starts >= span[1] - third),
    )
