import numpy as np

from ensomi_model.audio_rows.audio import FEATURE_DIM
from ensomi_model.audio_rows.measures import chart_measures, correlation, lattice_measures
from ensomi_model.audio_rows.data import chart_targets
from .test_data import grid


def test_measures_recognise_rests_in_quiet_audio_and_onset_aligned_heads():
    g = grid(0, 500)
    features = np.zeros((4000, FEATURE_DIM), dtype=np.float32)
    features[800:2400, :128] = 2.0
    heads = np.arange(8000.0, 24000.0, 500.0)
    for h in heads:
        features[int(h / 10):int(h / 10) + 3, :128] += 2.0
    m = chart_measures(heads, 40000, g, features)
    assert m['intensity_workload_rho'] > 0.8
    assert m['low_intensity_rest'] == 1.0
    assert m['low_intensity_wh_s'] == 0.0 < m['high_intensity_wh_s']
    assert m['onset_rank'] > m['shifted_onset_rank']


def test_lattice_measure_counts_shared_slots_separately():
    t = chart_targets([0, 500 / 3, 1000 / 3], 2000, grid(0, 500))
    m = lattice_measures(t)
    assert m['heads'] == m['triple_heads'] == 3
    assert m['triple_only_heads'] == 2
    assert m['triple_occupied_beats'] == 1 and m['occupied_beats'] == 1
    assert correlation([0, 0, 0], [1, 2, 3], rank=True) is None


def test_onset_rank_reports_a_head_with_no_slot_in_its_partial_boundary_beat():
    features = np.zeros((200, FEATURE_DIM), dtype=np.float32)
    m = chart_measures([1, 531, 1031], 2000, grid(31, 500), features)
    assert m['onset_unscored_heads'] == 1 and m['onset_heads'] == 2
    assert m['onset_rank'] == m['shifted_onset_rank'] == 0.5
