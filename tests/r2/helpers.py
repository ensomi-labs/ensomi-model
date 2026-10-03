"""Synthetic R2 fixtures: heads-only grids, legal random decision sequences and small models."""
from __future__ import annotations

import numpy as np
import torch

from ensomi_model.evaluation.redlines import RedLine
from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ensomi_model.r2.cache import decisions_from_objects
from ensomi_model.r2.common import ACTIONS, GridArrays, grid_from_arrays
from ensomi_model.r2.features import Chart, Interval
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.state import action_support_mask

TAP, HOLD = ManiaHitObjectKind.TAP, ManiaHitObjectKind.HOLD
LINES = [RedLine(0.0, 500.0, 4), RedLine(6000.0, 400.0, 3)]


def tap(t, lane):
    return ManiaHitObject(float(t), float(t), lane, TAP)


def hold(t, e, lane):
    return ManiaHitObject(float(t), float(e), lane, HOLD)


def chart_from_objects(objects, song_ms, lines=LINES):
    dec = decisions_from_objects(objects, lines, song_ms)
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    return Chart(dec.head_ms, dec.song_ms, grid, dec.actions.astype(np.int64), dec.gap_release_ms), dec


def random_skeleton(rng, K, lo=25, hi=400):
    gaps = rng.integers(lo, hi, size=K)
    head = 500.0 + np.cumsum(gaps).astype(np.float64)
    return head, float(head[-1] + rng.integers(30, 600))


def random_decisions(rng, chart_inputs, n=None, p_hold=0.35):
    """A legal random decision sequence over (head_ms, song_ms, grid) through EOS (or n decisions)."""
    head, song, grid = chart_inputs
    K = len(head)
    n = K + 1 if n is None else n
    acts = np.zeros((K + 1, 4), dtype=np.int64)
    gap = np.full((K + 1, 4), np.nan)
    chart = Chart(head, song, grid, acts[:0], gap[:0])
    for k in range(n):
        chart = chart.with_decisions(acts[:k], gap[:k])
        held = chart.derived().held[k]
        mask = action_support_mask(held, k == K)
        choices = np.flatnonzero(mask)
        weights = np.array([_weight(ACTIONS[a], held, p_hold) for a in choices])
        a = ACTIONS[rng.choice(choices, p=weights / weights.sum())]
        acts[k] = a
        if k > 0:
            cands = chart.candidates(k).times
            for lane in range(4):
                if held[lane] and a[lane] in (2, 3, 4):
                    gap[k, lane] = cands[rng.integers(0, len(cands))]
    return Chart(head, song, grid, acts[:n], gap[:n])


def _weight(codes, held, p_hold):
    w = 1.0
    for c, h in zip(codes, held):
        if h:
            w *= (0.5, 0.15, 0.15, 0.1, 0.1)[c]
        else:
            w *= (0.6, 0.4 * (1 - p_hold), 0.4 * p_hold)[c]
    return w


def two_segment_grid():
    from ensomi_model.evaluation.beats import BarStart, BeatGrid, Segment
    grid = BeatGrid((Segment(0.0, 500.0, 4), Segment(6000.0, 400.0, 3)), (BarStart(0.0, 4), BarStart(6000.0, 3)))
    return GridArrays.from_grid(grid)


def sample_track(rng, chart):
    T = chart.song_ms
    a1 = float(rng.uniform(0, T / 2))
    b1 = float(rng.uniform(a1 + 100, T))
    a2 = float(rng.uniform(0, T / 3))
    return (Interval(0, a1, b1, float(rng.uniform(0, 1))), Interval(1, a2, T, float(rng.uniform(1, 6))))


def tiny_model(dtype=torch.float64, conditioner='film', seed=0, **kw):
    torch.manual_seed(seed)
    model = R2Model(R2Config(conditioner=conditioner, **kw))
    with torch.no_grad():  # give zero-initialised outputs some weight so every path matters
        for name, p in model.named_parameters():
            if p.ndim >= 1 and float(p.abs().sum()) == 0.0 and 'bias' not in name:
                p.normal_(0, 0.05)
    return model.to(dtype)
