import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes.common import auc, interval, paired, song_auc
from n1_redesign.probes.report import compare, merge_stats


def summary(values):
    return dict(**interval(list(values.values())), recordings=values)


def test_auc_ties_are_half_credit():
    assert auc([0, 1], [.5, .5]) == .5
    assert auc([0, 1, 0, 1], [0, 1, 0, 1]) == 1
    assert auc([], []) is None
    assert auc([1, 1], [.2, .4]) is None


def test_pooled_auc_uses_only_eligible_recordings():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0, 0, 1, 1, 2, -1])
    rec = np.array(['a'] * 4 + ['b'] * 2)
    result = song_auc(y, p, rec, minimum=2)
    assert result['n'] == 1
    assert result['pooled_auc'] == 1


def test_contrast_uses_shared_recordings_only():
    d = paired({'a': 1., 'b': 0., 'c': 999.}, {'a': .5, 'b': -.5, 'z': -999.})
    assert d['n'] == 2 and d['mean'] == .5
    assert d['ci95'] == [.5, .5]


def test_missing_seed_or_refit_noise_cannot_pass():
    a = summary({str(i): .9 for i in range(50)})
    assert not compare({0: a, 1: None}, None, threshold=.7)['holds_both_seeds']
    assert not compare({0: a, 1: a}, None, threshold=.7, require_refits=True)['holds_both_seeds']


def test_power_floor_and_refit_margin():
    a = summary({str(i): .9 for i in range(39)})
    assert not compare({0: a, 1: a}, None, threshold=.7)['holds_both_seeds']
    a = summary({str(i): .9 for i in range(40)})
    assert compare({0: a, 1: a}, None, threshold=.7)['holds_both_seeds']
    noisy = {0: [.6, .8, .9, 1., .7], 1: [.6, .8, .9, 1., .7]}
    assert not compare({0: a, 1: a}, None, threshold=.7, refits=noisy)['holds_both_seeds']


def test_joined_read_cohorts_count_each_recording_once():
    result = merge_stats([summary({'a': .2, 'shared': .4}), None,
                          summary({'b': .8, 'shared': .6})])
    assert result['n'] == 3
    assert result['recordings'] == {'a': .2, 'b': .8, 'shared': .5}


def test_curve_follows_adjacent_gains(monkeypatch):
    from n1_redesign.probes import report
    data = {'curves': {f'graph:{g}': {'M4_away': summary({'a': x}),
                                    'M3i': summary({'a': y})}
                       for g, x, y in [(0, 0, 0), (1, .3, .9), (2, .1, .1)]}}
    monkeypatch.setattr(report, 'read', lambda _: data)
    value = report.curve_read(0, 'graph', .2)
    assert value['gains'] == [0, 1]
    assert abs(value['mean'] - .6) < 1e-6
