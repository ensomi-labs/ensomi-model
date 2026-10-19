"""The phase heads' log I0 (an elementwise approximation that runs on MPS) against numpy's I0."""
from __future__ import annotations

import numpy as np

from ensomi_n.models.heads import _log_i0


def test_log_i0_matches_numpy_across_the_branch_point():
    k = np.concatenate([np.linspace(1e-3, 10.0, 2001), np.linspace(10.0, 600.0, 500)])
    assert np.max(np.abs(_log_i0(k, np) - np.log(np.i0(k)))) < 1e-6
    edge = _log_i0(np.array([3.75 - 1e-9, 3.75 + 1e-9]), np)
    assert abs(edge[1] - edge[0]) < 1e-6
