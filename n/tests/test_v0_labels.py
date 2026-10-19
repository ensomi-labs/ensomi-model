"""N1 v0 labels and batches: boundaries, detour weights, the literal interim rule, the sampler's clocks."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ensomi_n.timing import Reference, Segment
from ensomi_n.v0.labels import V0Labels, dense_v0, literal_flag


def _ref(segments, lo=0.0, hi=60000.0, step=125.0):
    heads = np.arange(lo, hi + 1e-9, step)
    return Reference.from_segments(segments, heads)


def test_boundaries_from_tempo_and_reset_lines():
    # 120 BPM, a reset 0.37 beat off the grid at 20 s, 150 BPM from 40.185 s
    segs = [Segment(0.0, 500.0), Segment(20185.0, 500.0), Segment(40185.0, 400.0)]
    lab = V0Labels.from_reference(_ref(segs))
    assert np.allclose(lab.bounds, [20185.0, 40185.0])
    assert lab.bound_kind.tolist() == [1, 0]
    back = V0Labels.from_json(json.loads(json.dumps(lab.to_json())))
    assert np.allclose(back.bounds, lab.bounds) and np.allclose(back.lines, lab.lines)
    tg = dense_v0(lab, 20000.0, 400)
    j = int(round((20185.0 - 20000.0) / 10.0))
    assert tg['bound'][j] == 1.0 and tg['bound'][j + 1] == 1.0 and tg['bound'][j + 3] == 0.5
    assert tg['bound'][j + 4] == 0.0 and tg['bound'][:j - 3].sum() == 0.0


def test_redundant_line_is_no_boundary_and_stretch_maps_bounds():
    segs = [Segment(0.0, 500.0), Segment(10000.0, 500.0), Segment(30000.0, 600.0)]   # 10 s: redundant
    lab = V0Labels.from_reference(_ref(segs))
    assert np.allclose(lab.bounds, [30000.0])
    tg = dense_v0(lab, 32000.0, 600, rate=1.1)       # 30 s * 1.1 = 33 s -> frame 100
    assert tg['bound'][100] == 1.0 and tg['bound'][96] == 0.0


def test_detour_runs_weigh_half():
    # a 2-beat detour at another tempo that returns to the 120 BPM grid
    segs = [Segment(0.0, 500.0), Segment(10000.0, 300.0), Segment(11000.0, 500.0)]
    ref = Reference.from_segments(segs, np.arange(0.0, 30000.0, 250.0))
    lab = V0Labels.from_reference(ref)
    assert lab.detours.shape == (1, 2) and np.allclose(lab.detours[0], [10000.0, 11000.0])
    tg = dense_v0(lab, 9000.0, 300)
    assert np.isclose(tg['weight'][150], 0.5) and np.isclose(tg['weight'][50], 1.0)
    assert len(lab.bounds) == 0


def test_literal_rule():
    assert not literal_flag(_ref([Segment(0.0, 500.0)]))
    assert literal_flag(_ref([Segment(0.0, 375.0)]))                         # 160 BPM
    assert literal_flag(_ref([Segment(0.0, 500.0), Segment(8000.0, 500.0)]))  # redundant
    assert not literal_flag(_ref([Segment(0.0, 500.0), Segment(8100.0, 500.0)]))  # a reset is not redundant
