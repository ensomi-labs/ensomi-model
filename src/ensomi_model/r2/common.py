"""Shared constants, the scalar transform and a fast array view of a heads-only beat grid.

Lanes are 0..3; the mirror M maps lane l to 3 - l. The left hand frame lists
lanes (0, 1, 2, 3) and the right hand frame (3, 2, 1, 0), so mirroring a state
swaps the two hand frames exactly.

Decision codes (per lane, meaning depends on whether the lane is held before
the decision):

    code  free before        held before
    0     nothing            keep held
    1     tap                release exactly at t_k
    2     LN head            release strictly inside (t_{k-1}, t_k)
    3     invalid            gap release, then tap at t_k
    4     invalid            gap release, then LN head at t_k

Decision k in 0..K-1 is the head row at t_k; decision K is EOS at the song end
T with gap (t_{K-1}, T), where held lanes use code 2 and free lanes code 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from ..evaluation.beats import BarStart, BeatGrid, Segment
from ..research.chart.dataset import ContractError

__all__ = ['ContractError', 'ACTIONS', 'ACTION_INDEX', 'MIRROR_ACTION', 'HAND_LANES', 'MS_SCALE',
           'psi', 'psi_pair', 'GridArrays', 'grid_to_arrays', 'grid_from_arrays', 'CACHE_VERSION',
           'CANDIDATE_VERSION', 'FEATURE_VERSION', 'MIN_GAP_MS', 'band_of']

CACHE_VERSION = 'r2-cache-v1'
CANDIDATE_VERSION = 'cand-v1'
FEATURE_VERSION = 'r2-feat-v1'
MIN_GAP_MS = 2.0
MS_SCALE = 1000.0

ACTIONS = np.array(tuple(product(range(5), repeat=4)), dtype=np.int64)  # [625,4]
ACTIONS.flags.writeable = False
ACTION_INDEX = {tuple(int(x) for x in a): i for i, a in enumerate(ACTIONS)}
MIRROR_ACTION = np.array([ACTION_INDEX[tuple(int(x) for x in a[::-1])] for a in ACTIONS], dtype=np.int64)
MIRROR_ACTION.flags.writeable = False
HAND_LANES = ((0, 1, 2, 3), (3, 2, 1, 0))


def action_index(codes) -> int:
    a = [int(c) for c in codes]
    return ((a[0] * 5 + a[1]) * 5 + a[2]) * 5 + a[3]


def psi(d, scale):
    """psi_s(d) = [clip(d/s, -32, 32), sign(d) log1p(|d|/s)], last axis 2."""
    d = np.asarray(d, dtype=np.float64) / scale
    return np.stack((np.clip(d, -32.0, 32.0), np.sign(d) * np.log1p(np.abs(d))), -1)


def psi_pair(ms, beats):
    """Milliseconds through psi_1000 and beats through psi_1: last axis 4."""
    return np.concatenate((psi(ms, MS_SCALE), psi(beats, 1.0)), -1)


def band_of(star: float) -> int:
    return int(min(5, max(2, math.floor(star))))


@dataclass(frozen=True)
class GridArrays:
    """Vectorised canonical coordinates of a BeatGrid (segments only; bars are not used)."""
    offsets: np.ndarray        # [S]
    lengths: np.ndarray        # canonical beat length per segment, ms
    origins: np.ndarray        # global beat at each segment offset
    meters: np.ndarray         # [S]

    @classmethod
    def from_grid(cls, grid: BeatGrid) -> 'GridArrays':
        offsets = np.array([s.offset_ms for s in grid.segments], dtype=np.float64)
        lengths = np.array([s.canonical_beat_length_ms for s in grid.segments], dtype=np.float64)
        origins = np.concatenate([[0.0], np.cumsum(np.diff(offsets) / lengths[:-1])])
        meters = np.array([s.meter for s in grid.segments], dtype=np.float64)
        return cls(offsets, lengths, origins, meters)

    def segment(self, times) -> np.ndarray:
        return np.clip(np.searchsorted(self.offsets, times, side='right') - 1, 0, len(self.offsets) - 1)

    def beat(self, times) -> np.ndarray:
        """Global canonical beat B(t); same as BeatGrid.global_beat."""
        times = np.asarray(times, dtype=np.float64)
        s = self.segment(times)
        return self.origins[s] + (times - self.offsets[s]) / self.lengths[s]

    def time_of_beat(self, beats) -> np.ndarray:
        beats = np.asarray(beats, dtype=np.float64)
        s = np.clip(np.searchsorted(self.origins, beats, side='right') - 1, 0, len(self.origins) - 1)
        return self.offsets[s] + (beats - self.origins[s]) * self.lengths[s]

    def bpm(self, times) -> np.ndarray:
        return 60000.0 / self.lengths[self.segment(times)]

    def meter(self, times) -> np.ndarray:
        return self.meters[self.segment(times)]

    def local_beat(self, times):
        times = np.asarray(times, dtype=np.float64)
        s = self.segment(times)
        return (times - self.offsets[s]) / self.lengths[s], s

    def mirror_invariant_key(self):
        return (tuple(self.offsets), tuple(self.lengths), tuple(self.meters))


def grid_to_arrays(grid: BeatGrid):
    seg = np.array([(s.offset_ms, s.beat_length_ms, s.meter) for s in grid.segments], dtype=np.float64)
    bars = np.array([(b.time_ms, b.meter) for b in grid.bar_starts], dtype=np.float64)
    return seg, bars


def grid_from_arrays(seg, bars) -> BeatGrid:
    return BeatGrid(tuple(Segment(float(o), float(l), int(m)) for o, l, m in np.asarray(seg)),
                    tuple(BarStart(float(t), int(m)) for t, m in np.asarray(bars)))
