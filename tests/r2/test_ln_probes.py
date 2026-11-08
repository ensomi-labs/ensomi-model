"""Measurement invariants on synthetic charts; no corpus or weights required."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ensomi_model.r2.features import Chart
from ensomi_model.r2.common import GridArrays
from .helpers import tiny_model as tiny_r2


def grid():
    return GridArrays(np.array([0.]), np.array([500.]), np.array([0.]), np.array([4.]))

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'r2'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


separation = load_script('separation')
probe = load_script('ln_probe')
analysis = load_script('ln_analysis')


def chord_chart():
    head = np.array([1000., 1500., 2000., 2500.])
    actions = np.array([[2, 2, 1, 0], [0, 0, 1, 0], [4, 2, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
    gap = np.full((5, 4), np.nan)
    gap[2, :2] = 1875.
    return Chart(head, 3000., grid(), actions, gap)


def test_likelihood_probe_matches_model_and_same_birth_hook():
    chart, model = chord_chart(), tiny_r2()
    with torch.inference_mode():
        records, hazards, events = probe.tf_chart(model, chart, np.arange(5), lesions=True)
        expected = model.window(chart, 0, 5)
        np.testing.assert_allclose([r['code_nll'] for r in records], -expected.action.numpy(), atol=1e-6)
        np.testing.assert_allclose([r['pointer_nll'] for r in records], -expected.release.numpy(), atol=1e-6)
        with probe.same_birth_lesion(model, True):
            zero = model.window(chart, 0, 5)
        np.testing.assert_allclose([r['same_birth_pointer_nll'] for r in records], -zero.release.numpy(), atol=1e-6)
    assert sum(r['gap_releases'] for r in records) == 2
    assert len(hazards) == 5
    assert any(e['event'] == 'together' and e['observed'] == 1 for e in events)


def test_row_head_lesion_preserves_release_type_and_source():
    chart = chord_chart()
    factors = probe.factors_for(chart, np.array([2]))
    original = [f.codes.copy() for f in factors]
    _, altered = probe.lesion_inputs(torch.zeros(1, 2, 16), factors, chart, 'row_heads')
    for old, new, source in zip(factors, altered, original):
        physical = new.codes[::-1] if new.orientation else new.codes
        assert np.array_equal(physical, [2, 2, 0, 0])
        assert np.array_equal(old.codes, source)
        assert np.array_equal(old.times, new.times)


def test_class_and_gap_denominator_exclude_terminal_lane_hold():
    objects = [ManiaHitObject(i * 500., i * 500. + 450, 0, ManiaHitObjectKind.HOLD) for i in range(40)]
    all_stats = separation.ln_statistics(objects, grid())
    assert all_stats['ln_class'] == 'inverse'
    assert all_stats['inverse_gap_n'] == 39
    assert all_stats['inverse_gap_share'] == 1
    suffix = separation.ln_statistics(objects, grid(), start_ms=5000)
    assert suffix['objects'] == 30 and suffix['inverse_gap_n'] == 29
    # The minimum denominator is part of the class definition, including on suffixes.
    assert suffix['ln_class'] == 'ln_heavy'


def test_code_counts_use_state_dependent_meanings_and_separate_eos():
    counts = separation.code_counts(chord_chart().actions, stop=4)
    assert counts['held_4'] == 1
    assert counts['held_2'] == 1
    assert counts['free_2'] == 2
    assert sum(counts.values()) == 16
    assert len(probe.panel_indices(10)) == 11


def test_joint_pointer_sampling_uses_candidates_and_both_coherent_orders():
    chart, model = chord_chart(), tiny_r2()
    with torch.inference_mode():
        z = model.window_hands(chart, [2])
        draws = probe.joint_draws(model, chart, 2, z, 810, samples=256)
    candidates = chart.candidates(2).times
    assert np.isin(draws[:, :2], candidates).all()
    assert np.isnan(draws[:, 2:]).all()
    assert len(np.unique(draws[:, 0])) > 1
    events = probe.mc_events(chart, 2, draws)
    assert 0 <= events['together'] <= 1
    assert probe.mc_events(chart, 2, chart.derived().release[2:3])['together'] == 1


def test_inference_floor_uses_paired_song_differences_not_pooled_mean_change():
    records = []
    for song, sign in [('a', 1), ('b', -1)]:
        for checkpoint, offset in [(48, -2), (56, 0), (64, 2)]:
            for seed, noise in [(0, -1), (1, 0), (2, 1)]:
                records.append(dict(sha=song, arm='baseline', checkpoint=checkpoint, seed=seed,
                                    metric=sign * (offset + noise)))
    floors = analysis.fr_floors(pd.DataFrame(records), 'metric')
    # Opposite changes cancel in the pooled mean but remain in the paired noise floor.
    assert floors['sampling_rms'] == pytest.approx(np.sqrt(2))
    assert floors['checkpoint_rms'] == pytest.approx(np.sqrt(8))
    assert floors['sampling_pairs'] == floors['checkpoint_pairs'] == 6


def test_a_single_song_cannot_get_an_interval_verdict():
    result = analysis.scaled(analysis.interval([1.]), dict(sampling_rms=0., checkpoint_rms=0.))
    assert np.isinf(result['effect_over_floor'])
    assert result['verdict'] == 'insufficient_songs'
