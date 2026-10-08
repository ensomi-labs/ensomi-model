import numpy as np

from ensomi_model.audio_rows import lattice as L
from ensomi_model.r2.common import GridArrays, grid_from_arrays


def grid(offset_ms=0.0, beat_ms=500.0):
    return GridArrays.from_grid(grid_from_arrays(np.array([[offset_ms, beat_ms, 4]]), np.array([[offset_ms, 4]])))


def test_snap_and_decode_round_trip_duple_and_triple_beats_before_the_first_segment():
    g = grid(offset_ms=1_614.0)
    beats = np.array([-3.0, -2.75, -2.5, 2.0, 2 + 1 / 3, 2 + 2 / 3, 5 + 3 / 16, 5 + 15 / 16])
    heads = g.time_of_beat(beats)
    t = L.snap(heads, g, 10_000.0)
    assert t['b0'] == -4 and t['merged'] == 0 and np.abs(t['err_ms']).max() < 1e-6
    assert [t['lattice'][b - t['b0']] for b in (-3, 2, 5)] == [0, 1, 0]
    assert [t['count'][b - t['b0']] for b in (-3, 2, 5)] == [3, 3, 2]
    assert np.allclose(L.heads_from_targets(g, t['b0'], t['lattice'], t['mask']), heads, atol=1e-3)


def test_head_rounding_to_the_next_beat_merges_with_its_downbeat():
    g = grid()
    t = L.snap(g.time_of_beat(np.array([3.99, 4.0])), g, 5_000.0)
    assert t['count'][3] == 0 and t['count'][4] == 1 and t['merged'] == 1
