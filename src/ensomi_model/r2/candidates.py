"""Target-independent release candidates for one gap (version ``cand-v1``).

The support of a gap release in (a, b) is the strictly interior union of the
rounded 1/32 and 1/24 canonical grid positions of every intersected segment,
the rounded midpoint, and the rounded positions 1/8 and 1/4 global canonical
beat before b. Rounding is floor(t + 0.5). Nothing here reads a source release.
With every gap at least 2 ms the rounded midpoint is always interior, so the
set is never empty.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import CANDIDATE_VERSION, ContractError, GridArrays, psi_pair

GRID_DENOMINATORS = (32, 24)
SNAP_ONE_HOT = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SNAP_TOLERANCE_MS = 1.0
STATIC_DIM = 8 + 1 + 4 + len(SNAP_ONE_HOT) + 1 + 6     # 30, lane-free
CANDIDATE_DIM = STATIC_DIM + 4                          # 34 with u - h views
PHASE_PERIODS = (1.0, 4.0, 16.0)

__all__ = ['GapCandidates', 'build_candidates', 'snap_label', 'phase', 'CANDIDATE_DIM', 'STATIC_DIM',
           'CANDIDATE_VERSION']


def _round(x):
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def phase(beats):
    beats = np.asarray(beats, dtype=np.float64)
    parts = []
    for p in PHASE_PERIODS:
        angle = 2.0 * math.pi * beats / p
        parts += [np.sin(angle), np.cos(angle)]
    return np.stack(parts, -1)


@dataclass(frozen=True)
class GapCandidates:
    a: float
    b: float
    times: np.ndarray       # [C] float64, strictly increasing, a < u < b
    flags: np.ndarray       # [C,4] midpoint, eighth-before, quarter-before, grid
    static: np.ndarray      # [C,30] lane-free features (views to b and a, fraction, flags, snap, bpm, phase)
    beats: np.ndarray       # [C] global canonical beat of each candidate

    def lane_features(self, held_start_ms: float, held_start_beat: float) -> np.ndarray:
        """[C,34]: static features plus the u - h views in ms and beats."""
        hv = psi_pair(self.times - held_start_ms, self.beats - held_start_beat)
        return np.concatenate((self.static[:, :8], hv, self.static[:, 8:]), -1).astype(np.float32)


def build_candidates(a: float, b: float, grid: GridArrays) -> GapCandidates:
    a, b = float(a), float(b)
    if not b - a >= 2.0:
        raise ContractError(f'Gap ({a}, {b}) is shorter than 2 ms')
    pieces, grid_parts = [], []
    s_lo, s_hi = int(grid.segment(a)), int(grid.segment(b))
    S = len(grid.offsets)
    for s in range(s_lo, s_hi + 1):
        lo = -math.inf if s == 0 else grid.offsets[s]
        hi = math.inf if s == S - 1 else grid.offsets[s + 1]
        left, right = max(a - 1.0, lo), min(b + 1.0, hi)
        if right < left:
            continue
        off, length = grid.offsets[s], grid.lengths[s]
        for d in GRID_DENOMINATORS:
            step = length / d
            n0, n1 = math.ceil((left - off) / step), math.floor((right - off) / step)
            if n1 < n0:
                continue
            pos = off + np.arange(n0, n1 + 1, dtype=np.float64) * step
            pos = pos[(pos >= lo) & (pos < hi)]
            grid_parts.append(_round(pos))
    grid_times = np.concatenate(grid_parts) if grid_parts else np.empty(0)
    grid_times = grid_times[(grid_times > a) & (grid_times < b)]
    bb = float(grid.beat(b))
    mid = float(_round((a + b) / 2.0))
    eighth = float(_round(grid.time_of_beat(bb - 0.125)))
    quarter = float(_round(grid.time_of_beat(bb - 0.25)))
    anchors = [x for x in (mid, eighth, quarter) if a < x < b]
    times = np.unique(np.concatenate((grid_times, np.array(anchors, dtype=np.float64))))
    if not len(times):
        raise ContractError(f'Empty candidate set for gap ({a}, {b})')
    flags = np.stack((times == mid, times == eighth, times == quarter, np.isin(times, grid_times)), -1)
    flags[:, 0] &= a < mid < b
    flags[:, 1] &= a < eighth < b
    flags[:, 2] &= a < quarter < b
    beats = grid.beat(times)
    local, seg = grid.local_beat(times)
    snap = np.zeros((len(times), len(SNAP_ONE_HOT)), dtype=np.float64)
    found = np.zeros(len(times), dtype=bool)
    lengths = grid.lengths[seg]
    for i, d in enumerate(SNAP_ONE_HOT):
        err = np.abs(local * d - np.round(local * d)) / d * lengths
        hit = ~found & (err <= SNAP_TOLERANCE_MS)
        snap[hit, i] = 1.0
        found |= hit
    ba = float(grid.beat(a))
    static = np.concatenate((
        psi_pair(times - b, beats - bb), psi_pair(times - a, beats - ba),
        ((times - a) / (b - a))[:, None], flags.astype(np.float64), snap,
        (grid.bpm(times) / 120.0)[:, None], phase(beats)), -1)
    return GapCandidates(a, b, times, flags, static, beats)


def snap_label(release_ms: float, cands: GapCandidates) -> tuple[int, float]:
    """Nearest candidate in the same gap, ties to the earlier one: (index, snapped - original)."""
    if not cands.a < release_ms < cands.b:
        raise ContractError('A gap release must lie strictly inside its gap')
    dist = np.abs(cands.times - release_ms)
    i = int(np.flatnonzero(dist == dist.min())[0])
    return i, float(cands.times[i] - release_ms)
