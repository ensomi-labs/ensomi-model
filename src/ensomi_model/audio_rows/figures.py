"""Beat figures: the set of head positions inside one canonical beat, and the figure vocabulary.

A position is a multiple of 1/48 beat on the union of the duple 1/16 and the triple 1/12 lattices:
24 positions, multiples of 3 (duple) or 4 (triple). A figure is a 24-bit mask over them, so
{0} and {1/2} are the same figure on either lattice. Human heads are snapped one lattice per beat
by ``lattice.snap``; a figure therefore stays on one lattice, apart from the shared positions.

The vocabulary is the list of figures the generator can emit, index 0 the empty beat.
"""
from __future__ import annotations

import numpy as np

from .lattice import DUPLE, TRIPLE

POSITIONS = np.array(sorted(set(range(0, 48, 3)) | set(range(0, 48, 4))))     # in 1/48 beat
NPOS = len(POSITIONS)                                                          # 24
_INDEX = {int(p): i for i, p in enumerate(POSITIONS)}
DUPLE_BITS = np.array([1 << _INDEX[3 * s] for s in range(DUPLE)], dtype=np.int64)
TRIPLE_BITS = np.array([1 << _INDEX[4 * s] for s in range(TRIPLE)], dtype=np.int64)


def beat_figures(lattice, mask) -> np.ndarray:
    """[nb] int64 figure masks from per-beat lattice [nb] (0 duple, 1 triple) and slot mask [nb, 16]."""
    mask = np.asarray(mask, dtype=bool)
    duple = (mask[:, :DUPLE] * DUPLE_BITS).sum(1)
    triple = (mask[:, :TRIPLE] * TRIPLE_BITS).sum(1)
    return np.where(np.asarray(lattice) == 1, triple, duple).astype(np.int64)


def members(figures) -> np.ndarray:
    """[n, 24] bool occupied positions of figure masks [n]."""
    return ((np.asarray(figures, dtype=np.int64)[:, None] >> np.arange(NPOS)) & 1).astype(bool)


def counts(figures) -> np.ndarray:
    return members(figures).sum(1)


def head_beats(b0: int, figures) -> np.ndarray:
    """Sorted head positions in global canonical beats from per-beat figures starting at beat ``b0``."""
    m = members(figures)
    beat, pos = np.nonzero(m)
    return b0 + beat + POSITIONS[pos] / 48.0


def head_times(grid, b0: int, figures) -> np.ndarray:
    """Sorted head times in ms on ``grid`` (``GridArrays``)."""
    return np.round(grid.time_of_beat(head_beats(b0, figures)), 3)


class Vocab:
    """Figure masks [V] (index 0 empty) with lookup; unknown figures encode to -1."""

    def __init__(self, figures):
        self.figures = np.asarray(figures, dtype=np.int64)
        self.index = {int(f): i for i, f in enumerate(self.figures)}
        self.members = members(self.figures)
        self.counts = self.members.sum(1)

    def __len__(self):
        return len(self.figures)

    def encode(self, figures) -> np.ndarray:
        return np.array([self.index.get(int(f), -1) for f in figures], dtype=np.int64)

    @classmethod
    def fit(cls, figures, min_count: int = 3):
        """Figures seen at least ``min_count`` times, empty first, then by frequency."""
        values, n = np.unique(np.asarray(figures, dtype=np.int64), return_counts=True)
        keep = values[(n >= min_count) & (values != 0)]
        order = np.argsort(-n[np.isin(values, keep)], kind='stable')
        return cls(np.concatenate(([0], keep[order])))
