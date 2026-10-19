"""The rung-1 probe measures on fixtures with fixed expected values: identity 1, a 25 ms shift 0, chance 40 ms / L."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n import probe
from ensomi_n.labels import read_set


def _ref(osu_writer, red, heads, name='a.osu'):
    return read_set([osu_writer(name, red, heads)], key='k')


@pytest.fixture
def two_tempo(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(40)] + [21000.0 + 375.0 * k for k in range(60)]
    return _ref(osu_writer, [(1000.0, 500.0), (21000.0, 375.0)], heads)


def test_identity_scores_one(two_tempo):
    wins = probe.beat_windows(two_tempo)
    act = probe.Activation.impulses(two_tempo.lines().t, 10.0)
    assert probe.m1(act, wins) == 1.0
    assert all(len(w['t']) <= 8 for w in wins)
    # no window crosses the tempo change
    assert all((w['t'] < 21000.0 - 1).all() or (w['t'] >= 21000.0 - 1).all() for w in wins)


def test_shift_of_25ms_scores_zero(two_tempo):
    wins = probe.beat_windows(two_tempo)
    act = probe.Activation.impulses(two_tempo.lines().t + 25.0, 10.0)
    assert probe.m1(act, wins) == 0.0
    act = probe.Activation.impulses(two_tempo.lines().t + 15.0, 10.0)
    assert probe.m1(act, wins) == 1.0


def test_chance_is_40ms_over_beat(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(400)]
    ref = _ref(osu_writer, [(1000.0, 500.0)], heads)
    wins = probe.beat_windows(ref)
    rng = np.random.default_rng(1)
    vals = [probe.m1(probe.Activation(rng.random(25000), 0.0, 10.0), wins) for _ in range(12)]
    assert probe.chance(wins) == pytest.approx(0.08)
    assert np.mean(vals) == pytest.approx(0.08, abs=0.03)


def test_zero_activation_fails_and_bar_windows(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(64)]
    ref = _ref(osu_writer, [(1000.0, 500.0, 4)], heads)
    wins = probe.bar_windows(ref)
    assert wins and all(w['beat'] == pytest.approx(2000.0) for w in wins)
    assert probe.m1(probe.Activation(np.zeros(5000), 0.0, 10.0), probe.beat_windows(ref)) == 0.0
    bars = ref.timing.bar_starts()
    assert probe.m1(probe.Activation.impulses(bars, 10.0), wins) == 1.0
    assert probe.m1(probe.Activation.impulses(bars + 500.0, 10.0), wins) == 0.0   # a beat off


def test_m2_classes_and_m3(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(64)]
    ref = _ref(osu_writer, [(1000.0, 500.0)], heads)
    wins = probe.beat_windows(ref)
    lines = ref.lines().t
    assert probe.m2(probe.Activation.impulses(lines, 10.0), wins)['tempo'] == pytest.approx(1.0)
    half = np.sort(np.concatenate([lines, lines + 250.0]))
    assert probe.m2(probe.Activation.impulses(half, 10.0), wins)['octave'] == pytest.approx(1.0)
    assert probe.m3(lines, ref) == pytest.approx(1.0)
    assert probe.m3(lines + 30.0, ref) == 0.0
    acts = probe.Activation.impulses(lines + 3.0, 10.0)
    assert probe.m3(probe.peaks(acts, 0.4), ref) == pytest.approx(1.0)


def test_bootstrap_clusters():
    v = np.array([1.0, 1.0, 0.0, 0.0])
    mean, lo, hi = probe.bootstrap_mean(v, ['a', 'a', 'b', 'b'], n=500)
    assert mean == 0.5 and lo == 0.0 and hi == 1.0
