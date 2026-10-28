import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes import episodes as ep
from n1_redesign.probes import p7m


def fixture_song():
    bars = np.repeat(np.arange(12), 4)
    z = dict(beats=np.arange(48) * 500., length=np.full(48, 500.),
             red=np.array([[0., 500., 4.]]), bars=bars, bib=np.tile(np.arange(4), 12),
             bar_ids=np.arange(12), nontrivial=np.ones(12, bool), y=np.zeros(48, int))
    c = np.eye(48, dtype=np.float32)
    p = np.full(48, .3, np.float32)
    similarity = np.eye(48, dtype=np.float32)
    similarity[1, [0, 2, 10, 20, 21, 33, 37]] = [.9, .9, .8, .7, .8, .9, .95]
    exact = np.eye(12, dtype=bool)
    exact[0, [7, 8, 9, 10]] = True
    z['nontrivial'][10] = False
    return ep.Song({}, z, p, c, c.copy(), similarity, similarity.copy(),
                   np.eye(12), exact, [], ep.pulse_hash(z))


def test_oracle_requires_exact_nontrivial_distant_bar_and_same_position():
    s = fixture_song()
    matrix = p7m.oracle_similarity(s)
    assert np.flatnonzero(matrix[1]).tolist() == [1, 33, 37]
    assert np.diag(matrix).all()
    s.z['y'][:] = 1
    s.p[:] = .9
    np.testing.assert_array_equal(matrix, p7m.oracle_similarity(s))


@pytest.mark.parametrize('parent', ['mel', 'oracle'])
def test_random_preserves_each_parent_degree_weights_and_local_run(parent):
    s = fixture_song()
    config = dict(local=.85, theta=.6, K=2, gain=.5)
    mark = [dict(beat=1, L=1., kind=0)]
    real, local, _ = p7m.update(s, mark, parent + '-graph', config)
    random, random_local, _ = p7m.update(s, mark, parent + '-random', config, 91)
    np.testing.assert_array_equal(local, random_local)
    np.testing.assert_array_equal(real[local], random[local])
    a = logit(real) - logit(s.p)
    b = logit(random) - logit(s.p)
    np.testing.assert_allclose(np.sort(a[~local]), np.sort(b[~local]), atol=1e-7)
    assert np.count_nonzero(a[~local]) == np.count_nonzero(b[~local])


def test_mel_update_matches_original_secondary_arm_exactly():
    s = fixture_song()
    marks = [dict(beat=1, L=2., kind=0), dict(beat=47, L=-1., kind=0)]
    config = dict(local=.85, theta=.6, K=2, gain=.5)
    expected = ep.update(s, marks, 'mel', config)
    actual = p7m.update(s, marks, 'mel-graph', config)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_new_arm_fixtures_include_segment_resets_last_beat_and_no_marks():
    result = p7m.fixtures()
    assert result['passed'] and result['updates'] == 12
    assert result['original']['updates'] == 33


def test_unknown_graph_is_not_silently_treated_as_feature_graph():
    with pytest.raises(ValueError, match='Unknown graph'):
        p7m.update(fixture_song(), [], 'invalid', {})


def test_missing_seed_or_invariant_breach_prevents_followup_verdict():
    from n1_redesign.probes.p7m_report import verdicts
    missing = verdicts({0: None, 1: None}, {0: None, 1: None})
    assert missing['verdict'] == 'incomplete'
    assert not missing['prediction_holds_both'] and not missing['retire_holds_both']
    breached = verdicts({0: None, 1: None}, {0: dict(complete=False, breaches=1), 1: None})
    assert breached['verdict'] == 'stopped: invariant breach'


def test_curve_does_not_bridge_an_undefined_harm_point():
    from n1_redesign.probes.common import interval
    from n1_redesign.probes.p7m_report import curve_at
    def summary(value):
        return dict(**interval([value]), recordings={'r': value})
    data = dict(curves={
        'mel-graph:0.0': dict(M4_away=summary(.1), M3i=summary(.1)),
        'mel-graph:0.0625': dict(M4_away=interval([]), M3i=summary(.2)),
        'mel-graph:0.125': dict(M4_away=summary(.3), M3i=summary(.3))})
    assert curve_at(data, 'mel-graph', .2) is None


def test_zero_recovery_can_retire_only_with_power_and_known_margin():
    from n1_redesign.probes.p7m_report import within_margin
    comparison = dict(seeds={str(s): dict(power=True, margin=.01,
                      contrast=dict(mean=0.), status='does_not_hold') for s in (0, 1)})
    assert within_margin(comparison)['holds_both_seeds']
    comparison['seeds']['1']['margin'] = None
    assert not within_margin(comparison)['holds_both_seeds']


def test_resource_receipt_includes_report_process_without_changing_measurements(tmp_path):
    import json
    from n1_redesign.probes.p7m_guard import finalize_resources
    original = dict(prereg={'sha256': 'fixture'}, verdict='inconclusive', seeds={'0': {'M3i': .2}})
    (tmp_path / 'results.json').write_text(json.dumps(original))
    (tmp_path / 'report.md').write_text('Measurements\n\nResource records: `[]`.\n\nUnfinished: none\n')
    rows = [dict(command=['seed'], peak_process_tree_rss_bytes=100, stopped=None),
            dict(command=['report'], peak_process_tree_rss_bytes=200, stopped=None)]
    (tmp_path / 'resources.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    finalize_resources(tmp_path)
    actual = json.loads((tmp_path / 'results.json').read_text())
    assert all(actual[k] == v for k, v in original.items())
    assert actual['resources'] == rows
    assert actual['resource_totals']['peak_process_tree_rss_bytes'] == 200
    assert 'Measurements' in (tmp_path / 'report.md').read_text()
    assert '"report"' in (tmp_path / 'report.md').read_text()
