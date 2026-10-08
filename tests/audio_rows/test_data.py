import numpy as np

from ensomi_model.audio_rows.data import chart_targets, section_label, sections
from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.r2.strain import StrainTrace


def grid(offset_ms, beat_ms):
    return GridArrays.from_grid(grid_from_arrays(np.array([[offset_ms, beat_ms, 4]]), np.array([[offset_ms, 4]])))


def reference_label(trace, a_ms, b_ms):
    return np.log(max(trace.reference_rows(*trace.row_bounds(a_ms, b_ms)) / ((b_ms - a_ms) / 1000), 1e-3))


def test_section_labels_equal_the_strain_reference_over_sections_clipped_to_the_song():
    head_ms, song_ms = np.arange(1_000.0, 20_000.0, 250.0), 25_000.0
    t = chart_targets(head_ms, song_ms, grid(100.0, 500.0))
    nb = len(t['count'])
    assert t['beat_ms'][0] == 0.0 and t['beat_ms'][-1] == song_ms
    trace = StrainTrace(head_ms, song_ms)
    for a, b in [*sections(nb), (0, nb)]:
        expected = reference_label(trace, t['beat_ms'][a], t['beat_ms'][b])
        assert np.isclose(section_label(t['wh_beat'], t['beat_ms'], a, b), expected)


def test_a_head_on_a_beat_edge_counts_in_the_beat_it_starts():
    # floor(grid.beat(182211.0)) falls one beat early in floating point at this beat length.
    head_ms, song_ms = np.array([182_211.0, 185_211.0]), 190_000.0
    t = chart_targets(head_ms, song_ms, grid(179_211.0, 428.571428571429))
    edge = int(np.searchsorted(t['beat_ms'], 182_211.0))
    trace = StrainTrace(head_ms, song_ms)
    for a, b in ((edge - 1, edge), (edge, edge + 1)):
        expected = reference_label(trace, t['beat_ms'][a], t['beat_ms'][b])
        assert np.isclose(section_label(t['wh_beat'], t['beat_ms'], a, b), expected)
