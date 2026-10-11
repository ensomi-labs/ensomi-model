"""Canonical-beat lattice: per-beat targets (lattice, count, slot mask) from head times, and back.

A beat is one canonical beat of a ``GridArrays`` (BPM folded into [80, 160)). Inside a beat a head
sits on the duple lattice (16 slots at 1/16) or the triple lattice (12 slots at 1/12); one lattice
per beat, chosen by the smaller total snap error of the beat's heads (ties go to duple). Slot 0 is
shared by both lattices. A head that rounds to the next beat's slot 0 belongs to that beat, and
heads that snap to one slot merge.
"""
from __future__ import annotations

import numpy as np

DUPLE, TRIPLE = 16, 12
SLOTS = (DUPLE, TRIPLE)     # slots per beat, indexed by lattice 0 (duple) or 1 (triple)
MAX_SLOTS = 16
MAX_COUNT = 16              # count classes 0..16


def beat_range(grid, song_ms: float):
    """First beat index and the number of beats covering [0, song_ms]."""
    b0 = int(np.floor(grid.beat(0.0)))
    return b0, max(int(np.ceil(grid.beat(song_ms))) - b0, 1)


def slot_times(grid, b0: int, nb: int) -> np.ndarray:
    """[nb, 2, 16] slot times in ms; triple slots 12..15 are NaN."""
    beats = b0 + np.arange(nb)[:, None]
    out = np.full((nb, 2, MAX_SLOTS), np.nan)
    out[:, 0] = grid.time_of_beat(beats + np.arange(DUPLE)[None] / DUPLE)
    out[:, 1, :TRIPLE] = grid.time_of_beat(beats + np.arange(TRIPLE)[None] / TRIPLE)
    return out


def snap(head_ms, grid, song_ms: float) -> dict:
    """Per-beat targets: b0, nb, lattice [nb], count [nb], mask [nb, 16], err_ms [K], merged."""
    head_ms = np.asarray(head_ms, dtype=np.float64)
    b0, nb = beat_range(grid, song_ms)
    hb = grid.beat(head_ms)
    base = np.floor(hb)
    bk = base.astype(int) - b0
    frac = hb - base
    cost = np.zeros((nb, 2))
    slot = np.zeros((2, len(hb)), dtype=int)
    err = np.zeros((2, len(hb)))
    for l, n in enumerate(SLOTS):
        s = np.round(frac * n).astype(int)          # n means the next beat's slot 0
        slot[l] = s
        err[l] = grid.time_of_beat(base + s / n) - head_ms
        np.add.at(cost[:, l], np.clip(bk, 0, nb - 1), np.abs(err[l]))
    lattice = (cost[:, 1] < cost[:, 0]).astype(np.int64)
    mask = np.zeros((nb, MAX_SLOTS), dtype=bool)
    chosen = np.zeros(len(hb))
    merged = 0
    for k in range(len(hb)):
        l = lattice[min(max(bk[k], 0), nb - 1)]
        s, b = slot[l, k], bk[k]
        if s == SLOTS[l]:
            s, b = 0, b + 1
        if 0 <= b < nb:
            merged += int(mask[b, s])
            mask[b, s] = True
        chosen[k] = err[l, k]
    return dict(b0=b0, nb=nb, lattice=lattice, count=mask.sum(1), mask=mask, err_ms=chosen, merged=merged)


def heads_from_targets(grid, b0: int, lattice, mask) -> np.ndarray:
    """Sorted distinct head times in ms from per-beat lattice and mask."""
    lattice = np.asarray(lattice, dtype=np.int64)
    times = slot_times(grid, b0, len(lattice))[np.arange(len(lattice)), lattice]
    return np.unique(np.round(times[np.asarray(mask, dtype=bool)], 3))
