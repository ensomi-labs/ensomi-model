"""Small fixtures for the system tests: a tiny R2, a skeleton on a one-segment grid, a demo calibration."""
from __future__ import annotations

import numpy as np
import torch

from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.r2.model import R2Config, R2Model


def tiny_r2(seed=0):
    """Phase N's form (memory 'none') at a tiny width, zero-initialised outputs given small random weights."""
    torch.manual_seed(seed)
    model = R2Model(R2Config(hidden=16, levels=2, expansion=1, rank=4, memory='none'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 1 and float(p.abs().sum()) == 0.0 and 'bias' not in name:
                p.normal_(0, 0.05)
    return model.eval()


def grid(beat_ms=500.0, offset=0.0):
    return GridArrays.from_grid(grid_from_arrays(np.array([[offset, beat_ms, 4]]), np.array([[offset, 4]])))


def skeleton(rng, K, beat_ms=500.0, start=1000.0):
    """K heads on 1/4-beat positions of a 120 BPM grid, gaps of 1-4 quarter beats."""
    steps = rng.integers(1, 5, size=K)
    head = start + np.cumsum(steps) * beat_ms / 4
    return head.astype(np.float64), float(head[-1] + 1500.0)


def calibration():
    masks = dict(reattack_floor_ms=105.0, quad_run_cap=0, gap_tercile_edges_ms=[100.0, 163.0], jack_run_caps=[2, 4, 6])
    rng = np.random.default_rng(0)
    fit = dict(alpha=0.04, beta=0.79, gamma=0.88, seed_sd=0.03)
    return dict(masks={str(b): masks for b in range(2, 6)}, stream_eligibility_gap_ms=187.0,
                difficulty={str(b): dict(values=list(1.2 + 0.3 * rng.random(200))) for b in range(2, 6)},
                difficulty_fit=dict(accepted=True, domain=[0.8, 1.3], by_band={str(b): fit for b in range(2, 6)}),
                ln_source_levels={str(b): list(rng.random(100) * 0.3) for b in range(2, 6)},
                quad_p95={str(b): 0.05 for b in range(2, 6)}, scope_table='unused.parquet')
