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


def test_identical_zero_recovery_is_not_an_improvement():
    zero = summary({str(i): 0. for i in range(40)})
    assert not compare({0: zero, 1: zero}, {0: zero, 1: zero}, threshold=0,
                       episode_spreads={0: 0., 1: 0.})['holds_both_seeds']


def test_disjoint_strata_use_independent_recording_bootstrap():
    a = summary({f'a{i}': .1 for i in range(40)})
    b = summary({f'b{i}': .4 for i in range(40)})
    assert not compare({0: a, 1: a}, {0: b, 1: b}, threshold=0, direction='below')['holds_both_seeds']
    result = compare({0: a, 1: a}, {0: b, 1: b}, threshold=0, direction='below', paired_recordings=False)
    assert result['holds_both_seeds']
    assert result['seeds']['0']['contrast']['n_left'] == 40


def test_missing_experiments_do_not_qualify_packet(monkeypatch):
    from n1_redesign.probes import report
    monkeypatch.setattr(report, 'read', lambda _: None)
    assert report.probe_verdicts()['P1']['verdict'] == 'inconclusive'
    value = report.propagation_verdict()
    assert value['packet_eligible_arms'] == []
    assert value['verdict'] == 'inconclusive'
    monkeypatch.setattr(report, 'read', lambda name: dict(complete=False, breaches=1) if 'invariants' in name else None)
    value = report.propagation_verdict()
    assert value['packet_eligible_arms'] == []
    assert value['verdict'] == 'stopped: invariant breach'


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


def test_matched_curve_contrast_bootstraps_crossing(monkeypatch):
    from n1_redesign.probes import report
    data = {'curves': {}}
    for arm, scale in [('graph', .2), ('learned0', .8)]:
        for gain in (0, 1):
            data['curves'][f'{arm}:{gain}'] = {
                'M4_away': summary({str(i): gain * .4 for i in range(40)}),
                'M3i': summary({str(i): gain * scale for i in range(40)})}
    monkeypatch.setattr(report, 'read', lambda _: data)
    a = {s: report.curve_read(s, 'learned0', .2) for s in (0, 1)}
    b = {s: report.curve_read(s, 'graph', .2) for s in (0, 1)}
    result = compare(a, b, threshold=.05)
    assert result['holds_both_seeds']
    assert result['seeds']['0']['contrast']['bootstrap_crossings'] == 1000
    assert abs(result['seeds']['0']['contrast']['mean'] - .3) < 1e-6


def test_bootstrap_scaler_matches_physical_recording_resample(monkeypatch):
    from n1_redesign.probes import models
    monkeypatch.setattr(models, 'prereg_check', lambda: None)
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y, rec = np.tile([0, 1], 6), np.repeat(np.arange(4), 3)
    sampled = np.random.default_rng(6100).integers(4, size=4)
    indices = np.concatenate([np.flatnonzero(rec == r) for r in sampled])
    a = models.fit(x, y, rec, x, y, kind='logistic', seed=1, bootstrap=0, epochs=0)
    b = models.fit(x[indices], y[indices], np.repeat(np.arange(4), 3), x, y,
                   kind='logistic', seed=1, epochs=0)
    np.testing.assert_array_equal(a['mean'], b['mean'])
    np.testing.assert_array_equal(a['sd'], b['sd'])


def test_attack_buffer_preserves_cohorts_rows_and_recording_ids(tmp_path, monkeypatch):
    from n1_redesign.probes import attacks
    members = [dict(id='b-fit', rec='b', cohort='probe-fit'),
               dict(id='c-dev', rec='c', cohort='probe-dev'),
               dict(id='a-fit', rec='a', cohort='probe-fit'),
               dict(id='z-panel', rec='z', cohort='panel')]
    monkeypatch.setattr(attacks, 'units', lambda: members)
    monkeypatch.setattr(attacks, 'OUT', tmp_path)
    folder = tmp_path / 'frames' / 's0' / 'real'
    folder.mkdir(parents=True)
    for identity, f, pos, y in [('b-fit', [7], [9], [1]),
                               ('c-dev', [5], [6], [0]),
                               ('a-fit', [1, 2], [3, 4], [0, 1])]:
        np.savez(folder / (identity + '.npz'), F=np.array(f, np.float16)[:, None],
                 Pos=np.array(pos, np.float16)[:, None], y=np.array(y, np.int8))
    (x, y, rec), (xd, yd, rd) = attacks.training(0, 'F+Pos')
    np.testing.assert_array_equal(x, [[7, 9], [1, 3], [2, 4]])
    np.testing.assert_array_equal(y, [1, 0, 1])
    np.testing.assert_array_equal(rec, [1, 0, 0])
    np.testing.assert_array_equal(xd, [[5, 6]])
    np.testing.assert_array_equal(yd, [0])
    np.testing.assert_array_equal(rd, [2])
    assert x.dtype == xd.dtype == np.float16
    assert y.dtype == yd.dtype == np.int8
