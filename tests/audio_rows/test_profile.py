import json

import numpy as np

from ensomi_model.audio_rows.data import Dataset
from ensomi_model.audio_rows.generate import requests
from ensomi_model.audio_rows.profile import FEATURE_NAMES, WorkloadProfile, distribute, fit

from .helpers import tiny_build
from .test_data import grid


def test_profile_preserves_linear_budget_with_partial_sections_and_a_silence_floor():
    duration = np.array([.1, 8, 8, 3.7])
    score = np.array([-100, -1, 1, 0.5])
    values = distribute(score, duration, np.log(6))
    assert np.isclose(np.average(np.exp(values), weights=duration), 6)
    assert np.exp(values).min() >= 1e-3
    assert np.allclose(values, distribute(score + 99, duration, np.log(6)))
    assert np.allclose(distribute(np.zeros(4), duration, np.log(6)), np.log(6))


def test_absolute_section_override_wins_over_profile_without_mutating_it():
    g = grid(0, 500)
    base = np.linspace(-2, 3, 20)
    original = base.copy()
    got = requests(g, 10000, 3, {3: 2.3}, [dict(start_ms=2000, end_ms=4000, log_wh_per_s=1.5)], base=base)
    assert np.array_equal(base, original)
    assert np.all(got[4:8] == 1.5)
    assert np.array_equal(got[:4], base[:4]) and np.array_equal(got[8:], base[8:])


def test_profile_fit_ignores_development_labels_and_roundtrips(tmp_path):
    data = Dataset(tiny_build(tmp_path / 'data'))
    profile, record = fit(data)
    for row in data.dev:
        data.charts[row['sha']]['wh_beat'][:] = 1e8
    changed, _ = fit(data)
    assert changed == profile
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(record))
    assert WorkloadProfile.load(path) == profile
    g = grid(0, 500)
    wanted = profile.requests(data.features('a'), g, 40000, 2.3)
    assert len(wanted) == 80 and np.isclose(np.exp(wanted).mean(), np.exp(2.3))
    assert len(profile.coefficient) == len(FEATURE_NAMES)
