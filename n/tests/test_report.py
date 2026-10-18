"""Aggregation: the recording is the unit, strata tables, bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.eval import aggregate, build_strata, read_strata, score
from ensomi_n.eval.report import Entry
from ensomi_n.mapformat import MusicMap
from ensomi_n.timing import Reference, Segment

B = 500.0


def ref(key='r') -> Reference:
    return Reference.from_segments([Segment(0.0, B, 4)], np.arange(32) * B, key=key)


def shifted(ms: float) -> MusicMap:
    return MusicMap(segments=(Segment(ms, B, 4),))


def test_each_recording_weighs_one():
    good = [score(ref(f'a{i}'), shifted(0.0)) for i in range(3)]
    bad = score(ref('b'), shifted(B / 2))
    out = aggregate(good + [bad], ['A', 'A', 'A', 'B'], bootstrap=0)
    assert out['n_recordings'] == 2 and out['n_sets'] == 4
    assert out['bl20']['value'] == 0.5
    assert out['s_bl20']['value'] == 0.5
    per_set = aggregate(good + [bad], bootstrap=0)
    assert per_set['bl20']['value'] == 0.75


def test_identical_recordings_give_a_point_interval():
    scores = [score(ref(f'k{i}'), shifted(0.0)) for i in range(5)]
    out = aggregate(scores, bootstrap=200, seed=3)
    assert out['bl20'] == {'value': 1.0, 'lo': 1.0, 'hi': 1.0, 'n': 5}


def test_bootstrap_is_seeded_and_brackets_the_value():
    scores = [score(ref(f'k{i}'), shifted(s)) for i, s in enumerate([0, 0, 0, 30, 0, 250, 0, 15])]
    a = aggregate(scores, bootstrap=500, seed=1)
    b = aggregate(scores, bootstrap=500, seed=1)
    assert a == b
    v = a['bl20']
    assert v['value'] == pytest.approx(6 / 8) and v['lo'] <= v['value'] <= v['hi'] and v['lo'] < v['hi']


def test_offset_distribution_over_recordings():
    scores = [score(ref(f'o{i}'), shifted(s)) for i, s in enumerate([0.0, 5.0, -15.0, 30.0])]
    out = aggregate(scores, bootstrap=0)
    assert out['offset_median']['value'] == pytest.approx(2.5)
    assert out['offset_within_5']['value'] == 0.5
    assert out['offset_within_10']['value'] == 0.5
    assert out['offset_within_20']['value'] == 0.75
    # Robust SD: 1.4826 x median |o - 2.5| over {2.5, 2.5, 17.5, 27.5} = 1.4826 x 10.
    assert out['offset_robust_sd']['value'] == pytest.approx(14.826)


def test_bar_and_reset_terms_pool_without_the_offset():
    heads = [k * 1750.0 + j * 250.0 for k in range(8) for j in range(7)]
    seven_eight = Reference.from_segments([Segment(k * 1750.0, 500.0, 4) for k in range(8)], heads)

    def resets(ms):
        return MusicMap(segments=tuple(Segment(k * 1750.0 + ms, 500.0, 4) for k in range(8)))
    exact, late = score(seven_eight, resets(0.0)), score(seven_eight, resets(30.0))
    out = aggregate([exact, late, late], ['A', 'A', 'B'], bootstrap=0)
    # A's resets are hit by its exact set (the union), B's by none: 7 of 14.
    assert out['reset_recall']['value'] == 0.5 and out['reset_recall_o']['value'] == 1.0
    # Bar starts: A's two sets average 0.5, B scores 0.
    assert out['bars_start']['value'] == 0.25 and out['bars_start_o']['value'] == 1.0


def test_pooled_calibration_matches_one_set():
    beats = np.concatenate([np.arange(16) * B, np.arange(16, 32) * B + 100.0])
    m = MusicMap(beats=beats, confidence=np.r_[np.ones(16), np.zeros(16)])
    out = aggregate([score(ref(), m)], bootstrap=0)
    assert out['ece']['value'] == 0.0 and out['auc']['value'] == 1.0


def test_strata_table(tmp_path):
    path = tmp_path / 'strata.csv'
    path.write_text('key,recording,stratum,source,start_ms,end_ms,reference\n'
                    'a,rec1,S2;S4,ranked,,,1\n'
                    'b,rec1,S2,ranked,,,0\n'
                    'c,rec2,S3,loved,1000,5000,\n'
                    'c,rec2,S3,loved,8000,9000,\n'
                    'd,rec3,S4,,,,\n'
                    'd,rec3,S4,,100,200,\n', encoding='utf-8')
    table = read_strata(path)
    assert table['a'] == {'recording': 'rec1', 'source': 'ranked', 'reference': True,
                          'strata': {'S2': None, 'S4': None}}
    assert table['b']['reference'] is False
    assert table['c']['strata'] == {'S3': [(1000.0, 5000.0), (8000.0, 9000.0)]}
    assert table['d']['strata'] == {'S4': None}


def test_build_strata_splits_and_respects_the_reference_mark():
    r = ref('a')
    entries = [
        Entry(score(r, shifted(0.0)), 'rec1', 'ranked', True, {'S2': None}),
        Entry(score(ref('b'), shifted(B / 2)), 'rec1', 'ranked', False, {'S2': None}),
        Entry(score(ref('c'), shifted(B / 2)), 'rec2', 'loved', None, {'S3': [(0.0, 4000.0)]}),
    ]
    spans = {('c', 'S3'): score(ref('c'), shifted(0.0), spans=[(0.0, 4000.0)])}
    out = build_strata(entries, spans, bootstrap=0)
    assert list(out)[0] == 'all'
    assert set(out) == {'all', 'all|ranked', 'all|loved', 'S2', 'S2|ranked', 'S3', 'S3|loved'}
    assert out['all']['bl20']['value'] == 0.5 and out['all']['n_sets'] == 2
    assert out['S2']['bl20']['value'] == 1.0
    assert out['S3']['bl20']['value'] == 1.0
