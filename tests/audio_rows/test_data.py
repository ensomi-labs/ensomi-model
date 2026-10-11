import numpy as np

from ensomi_model.audio_rows.data import audio_subset, chart_targets, section_label, sections
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


def test_audio_subset_preserves_all_charts_and_split_membership():
    rows = [dict(key=f'{role}-{audio}', role=role, sha=f'{role}-{audio}-{chart}')
            for role in ('fit_train', 'fit_dev') for audio in range(5) for chart in range(3)]
    chosen = audio_subset(rows, 2, 1, seed=8)
    assert len(chosen) == 9
    assert {r['sha'] for r in chosen} == {r['sha'] for r in audio_subset(rows[::-1], 2, 1, seed=8)}
    assert all(r in chosen for r in rows if r['key'] in {c['key'] for c in chosen})
    assert audio_subset(rows) == rows
