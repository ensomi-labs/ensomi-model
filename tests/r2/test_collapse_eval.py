"""Known-pattern statistics, paired aggregation, export guards and scratch parity."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from ensomi_model.r2 import collapse_eval as ce
from ensomi_model.r2.features import Chart

from .helpers import two_segment_grid


def tap_chart(patterns, *, K=None):
    patterns = np.asarray(patterns)
    if K is not None:
        patterns = np.resize(patterns, K)
    head = 100 + np.arange(len(patterns)) * 80.
    actions = ((patterns[:, None] & (1 << np.arange(4))) != 0).astype(np.int64)
    actions = np.vstack([actions, np.zeros(4, dtype=np.int64)])
    return Chart(head, float(head[-1] + 100), two_segment_grid(), actions, np.full(actions.shape, np.nan))


def wide_reference():
    return dict(complete=True, bands={str(b): {k: [-1., 10000.] for k in ce.STATISTICS} for b in ce.BANDS})


def test_constant_pattern_window_and_global_lock_match_known_values():
    chart = tap_chart([1], K=128)
    rows, whole = ce.seq_stats(chart.head_ms, chart.actions)
    first, second = rows
    assert first['pent'] == 0
    assert first['ng4'] == 1 / 61
    assert first['rep1'] == first['loop'] == 1
    assert first['jack'] == first['fjack'] == 63 / 64
    assert second['jack'] == 1
    assert first['nh'] == first['single'] == first['lmax'] == first['hmax'] == 1
    assert first['held'] == first['c3'] == first['c4'] == first['bus4'] == 0
    assert first['hlock'] == 64 and second['hlock'] == 128
    assert whole == dict(K=128, maxlock=128, maxhlock=128, empty=0)


def test_alternating_lanes_and_busy_hold_are_distinguished():
    chart = tap_chart([1, 2], K=64)
    stats = ce.window_statistics(chart)
    assert stats['pent'] == 1
    assert stats['ng4'] == 2 / 61
    assert stats['rep1'] == stats['jack'] == 0
    assert stats['loop'] == 1 and stats['hlock'] == 1
    chart.actions[:, 2] = 0
    chart.actions[0, 2] = 2
    chart.actions[-1, 2] = 2
    chart.gap[-1, 2] = chart.head_ms[-1] + 50
    held, heads, ln = ce.decode(chart.actions)
    assert held[1:, 2].all() and not held[0, 2]
    assert heads[:, 2].sum() == ln[:, 2].sum() == 1
    stats = ce.window_statistics(chart)
    assert stats['held'] == 63 / 256
    assert stats['lock'] == 64 and stats['hlock'] == 1


def test_envelopes_directional_exits_and_distances():
    chart = tap_chart(np.arange(1, 16), K=640)
    rows, _ = ce.seq_stats(chart.head_ms, chart.actions)
    windows = [dict(row, band=2) for row in rows]
    reference = dict(bands=ce.fit_envelope(windows))
    assert reference['bands']['2']['nh'] == pytest.approx(np.quantile([r['nh'] for r in rows], [.01, .99]))
    stats = {key: float(np.mean(reference['bands']['2'][key])) for key in ce.STATISTICS}
    assert not any(ce.exits(stats, 2, reference).values())
    stats['rep1'] = reference['bands']['2']['rep1'][1] + .4
    stats['pent'] = reference['bands']['2']['pent'][0] - .2
    flags = ce.exits(stats, 2, reference)
    assert flags['rep1'] and flags['pent']
    width = {key: (reference['bands']['2'][key][1] - reference['bands']['2'][key][0]) or 1.0 for key in ('rep1', 'pent')}
    assert ce.envelope_score(stats, 2, reference, ['rep1', 'pent']) == pytest.approx(-(.4 / width['rep1'] + .2 / width['pent']))


def test_shuffle_correction_detects_order_but_not_constant_marginals():
    sha = '12ab34cd' + '0' * 56
    chart = tap_chart([1, 2], K=256)
    measured = ce.adjacency(chart.head_ms, chart.actions, sha)
    assert measured == ce.adjacency(chart.head_ms, chart.actions, sha)
    assert measured['all']['adjacent_MI_minus_shuffle'] > .9
    constant = tap_chart([1], K=256)
    assert ce.adjacency(constant.head_ms, constant.actions, sha)['all']['adjacent_MI_minus_shuffle'] == 0
    assert ce.adjacency(chart.head_ms[:100], chart.actions[:100], sha)['all']['adjacent_MI_minus_shuffle'] is None


def summary(x, *, scale=1., degenerate=.1, adjacency=.2):
    return dict(means={k: x * scale for k in ce.STATISTICS + ('lnlen',)},
                thirds=[{k: x * scale * (third + 1) for k in ce.COHERENCE} for third in range(3)],
                drift={k: {'1': x, '8': 8 * x} for k in ('nh', 'held')},
                degenerate=degenerate, degenerate_n=degenerate * 10, windows=10,
                entry=1., entry_n=4., stay=3., stay_n=6.,
                adjacency={'all': {'adjacent_MI_minus_shuffle': adjacency}},
                tail_exits=.1, maxhlock=2)


def records(*, scale=1., degenerate=.1, adjacency=.2, seeds=(954, 955)):
    return [dict(sha256=f'{band}{i:063x}', band=band, seed=seed, mode='bos', start=0, stop=1024,
                 source=summary(i + 1), sample=summary(i + 1, scale=scale, degenerate=degenerate, adjacency=adjacency),
                 legal={'pass_': True})
            for band in ce.BANDS for i in range(4) for seed in seeds]


def test_all_seven_measures_known_correlations_variances_spreads_and_gaps():
    result = ce.band_metrics([r for r in records(scale=2., degenerate=.3, adjacency=.45) if r['band'] == 2])
    assert result['metrics'] == pytest.approx(dict(m1=.3, m2=.5, m3=0., m4=64., m5=1., m6=2.5, m7=.25))
    assert result['entry_rate'] == .25
    assert result['source_minus_run_adjacent_MI'] == pytest.approx(-.25)
    assert result['charts'] == 4


def test_spread_counts_each_run_not_the_chart_average():
    rows = [r for r in records() if r['band'] == 2]
    for row in rows:
        for key in ce.SPREAD:
            row['sample']['means'][key] += 1. if row['seed'] == 954 else -1.
    assert ce.band_metrics(rows)['metrics']['m5'] > .1


def test_missing_ln_lengths_use_paired_remaining_charts():
    rows = [r for r in records(scale=2.) if r['band'] == 2]
    rows[0]['sample']['means']['lnlen'] = None
    rows[1]['sample']['means']['lnlen'] = None
    assert ce.band_metrics(rows)['sd_ratios']['lnlen'] == 2
    for row in rows:
        row['source']['means']['lnlen'] = 1.
    assert ce.band_metrics(rows)['metrics']['m5'] is None


def test_paired_chart_bootstrap_preserves_seed_clusters_and_common_baseline():
    base = records(degenerate=.5)
    left, right = records(degenerate=.3), records(degenerate=.2)
    # Opposite seed perturbations disappear within each chart, including bootstrap draws.
    for row in left:
        row['sample']['degenerate'] += .08 if row['seed'] == 954 else -.08
        row['sample']['degenerate_n'] = row['sample']['degenerate'] * row['sample']['windows']
    result = ce.bootstrap_difference(left, right, base, measures=['m1'], bootstrap=100, seed=31)
    assert result['difference'] == pytest.approx(.25)
    assert result['ci90'] == pytest.approx([.25, .25])
    assert result['valid_resamples'] == 100
    assert result == ce.bootstrap_difference(left, right, base, measures=['m1'], bootstrap=100, seed=31)
    assert ce.bootstrap_difference(left, left, base, measures=['m1'], bootstrap=20)['ci90'] == [0., 0.]


def test_source_and_base_use_same_chart_resampling_as_both_systems():
    base, left, right = records(), records(), records()
    for baseline, a, b in zip(base, left, right):
        x = int(baseline['sha256'][-1], 16) + 1
        for row, gap in ((baseline, .1), (a, .05), (b, .02)):
            row['source']['degenerate'] = .05 * x
            row['sample']['degenerate'] = (.05 + gap) * x
            row['source']['degenerate_n'] = row['source']['degenerate'] * row['source']['windows']
            row['sample']['degenerate_n'] = row['sample']['degenerate'] * row['sample']['windows']
    result = ce.bootstrap_difference(left, right, base, measures=['m1'], bootstrap=100)
    assert result['difference'] == pytest.approx(.3)
    assert result['ci90'] == pytest.approx([.3, .3])


def test_degenerate_rate_pools_windows_while_resampling_charts():
    rows = [r for r in records(seeds=(954,)) if r['band'] == 2][:2]
    rows[0]['sample'].update(windows=2, degenerate_n=2, degenerate=1.)
    rows[1]['sample'].update(windows=8, degenerate_n=0, degenerate=0.)
    assert ce.band_metrics(rows)['metrics']['m1'] == .2
    ids = [rows[0]['sha256'], rows[0]['sha256'], rows[1]['sha256']]
    assert ce.band_metrics(rows, chart_ids=ids)['metrics']['m1'] == pytest.approx(4 / 12)


def test_missing_band_never_becomes_four_band_G():
    base = records(degenerate=.5)
    partial = [r for r in records(degenerate=.3) if r['band'] == 2]
    score = ce.gap_score(ce.aggregate(partial), ce.aggregate(partial, field='source'), ce.aggregate(base), ['m1'])
    assert score['G'] is None
    assert score['gaps']['2']['m1'] == pytest.approx(.5)
    assert score['gaps']['3']['m1'] is None
    result = ce.bootstrap_difference(partial, partial, base, measures=['m1'], bootstrap=20)
    assert result['ci90'] is None and result['valid_resamples'] == 0


def test_guard_export_closes_only_temporary_copy_and_rejects_illegal_prefix(tmp_path, monkeypatch):
    from ensomi_model.osu_core.hitobjects import parse_mania_hit_objects
    from ensomi_model.r2 import export
    original_export = export.export_chart
    parsed_exports = []
    def checked_export(*args):
        objects, text = original_export(*args)
        path = tmp_path / 'guard.osu'
        path.write_text(text)
        parsed = parse_mania_hit_objects(path)
        key = lambda o: (o.start_time_ms, o.lane, o.end_time_ms)
        assert sorted(map(key, parsed)) == sorted(map(key, objects))
        parsed_exports.append(parsed)
        return objects, text
    monkeypatch.setattr(export, 'export_chart', checked_export)
    chart = tap_chart([1, 2], K=128)
    chart.actions[0, 2] = 2
    chart.actions[-1, 2] = 2
    chart.gap[-1, 2] = chart.head_ms[-1] + 50
    prefix = chart.with_decisions(chart.actions[:80], chart.gap[:80])
    before = prefix.actions.copy(), prefix.gap.copy()
    result = ce.legal_export(prefix)
    assert result == dict(pass_=True, partial=True, closure='midpoint-to-next-head')
    np.testing.assert_array_equal(prefix.actions, before[0])
    np.testing.assert_array_equal(prefix.gap, before[1])
    assert ce.legal_export(chart) == dict(pass_=True, partial=False, closure=None)
    assert len(parsed_exports) == 2
    assert next(o for o in parsed_exports[0] if o.lane == 2).end_time_ms == (chart.head_ms[79] + chart.head_ms[80]) / 2
    broken = prefix.with_decisions(prefix.actions.copy(), prefix.gap.copy())
    broken.actions[0, 0] = 4
    assert ce.legal_export(broken)['pass_'] is False


def test_tail_guard_covers_short_remainder_with_matched_source_windows():
    chart = tap_chart([1, 2], K=200)
    ref = wide_reference()
    ref['bands']['2']['loop'] = [-1, .5]
    result = ce.summarize_sequence(chart, '0' * 64, 2, ref)
    assert result['tail_rows'] == 20
    assert result['tail_exits'] == result['degenerate'] == 1
    assert result['stay'] == result['stay_n'] == 2
    assert result['drift']['nh']['8'] is None
    short = ce.summarize_sequence(tap_chart([1, 2], K=20), '0' * 64, 2, ref)
    assert short['tail_exits'] is None and short['degenerate'] is None


def test_guards_enforce_lock_nll_tail_and_diversity_thresholds():
    rows = records()
    ref = wide_reference()
    passed = ce.guards(rows, ref, nll=1.02, baseline_nll=1.)
    assert passed['legal_export'] and passed['no_head_lock_30'] and passed['nll_pass']
    assert all(band['tail_pass'] for band in passed['bands'].values())
    rows[0]['sample']['maxhlock'] = 30
    rows[0]['legal']['pass_'] = False
    for row in rows:
        row['sample']['tail_exits'] = .151
    ref['bands']['2']['single'] = [0., .5]
    ref['bands']['2']['ng4'] = [0., .5]
    failed = ce.guards(rows, ref, nll=1.0201, baseline_nll=1.)
    assert not failed['legal_export'] and not failed['no_head_lock_30'] and not failed['nll_pass']
    assert not any(band['tail_pass'] for band in failed['bands'].values())
    assert failed['bands']['2']['envelope'] == {'single': False, 'ng4': False}


def test_comparison_writes_json_markdown_and_nll_guard(tmp_path):
    paths = []
    for name, rate, nll in [('b0', .5, 1.), ('b1', .3, 1.03)]:
        rows = records(degenerate=rate)
        result = dict(system=name, records=rows, nll={'nll_per_decision': nll},
                      guards=ce.guards(rows, wide_reference(), nll=nll))
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(result))
        paths.append(path)
    out = tmp_path / 'compare.md'
    result = ce.compare_systems(paths, out, measures=['m1'], bootstrap=20)
    assert result['scores']['b0']['G'] == 1
    assert result['scores']['b1']['G'] == pytest.approx(.5)
    assert result['guards']['b1']['nll_pass'] is False
    assert 'b1' in out.read_text()
    assert result['ln_share']['b1']['2']['source_held_iqr'] == [1.75, 3.25]
    assert result['ln_share']['b1']['2']['lock'] == 2.5
    assert json.loads(out.with_suffix('.json').read_text()) == result
    assert json.loads(paths[1].read_text())['guards']['nll_pass'] is False
    altered = json.loads(paths[1].read_text())
    altered['nll']['requested_windows'] = 2
    paths[1].write_text(json.dumps(altered))
    assert ce.compare_systems(paths, out, measures=['m1'], bootstrap=0)['guards']['b1']['nll_pass'] is None


def test_capped_source_does_not_read_future_release_or_eos(tmp_path, monkeypatch):
    from ensomi_model.r2 import data
    chart = tap_chart([2], K=20)
    chart.actions[0, 0] = 2
    for k in range(2, 20, 2):
        chart.actions[k, 0] = 4
        chart.gap[k, 0] = (chart.head_ms[k - 1] + chart.head_ms[k]) / 2
    chart.actions[-1, 0] = 2
    chart.gap[-1, 0] = (chart.head_ms[-1] + chart.song_ms) / 2
    assert ce._ln_length(chart, 0, chart.K) is not None
    assert ce._ln_length(chart, 0, 18) is None
    prefix = chart.with_decisions(chart.actions[:chart.K], chart.gap[:chart.K])
    assert ce._ln_length(prefix, 0, chart.K) is None
    monkeypatch.setattr(data, 'Corpus', lambda *a, **kw: SimpleNamespace(chart=lambda sha: chart))
    panel = tmp_path / 'panel.json'
    panel.write_text('{}')
    ref = tmp_path / 'reference.json'
    ref.write_text(json.dumps(wide_reference()))
    runs = tmp_path / 'runs'
    runs.mkdir()
    np.savez(runs / 'run.npz', sha256='0' * 64, head_ms=chart.head_ms, song_ms=chart.song_ms,
             actions=prefix.actions, gap_release_ms=prefix.gap, start=0, stop=20, seed=954, band=2, mode='bos')
    result = ce.evaluate_system(panel, runs, 'b0', tmp_path, tmp_path / 'b0.json', reference_path=ref)
    assert result['records'][0]['source']['means']['lnlen'] is None
    assert result['records'][0]['sample']['means']['lnlen'] is None
    assert result['guards']['legal_export'] is True


def test_vectorized_resamples_equal_individual_chart_draws():
    rows = [r for r in records(scale=2.) if r['band'] == 2]
    rows[0]['sample']['means']['lnlen'] = None
    rows[1]['sample']['means']['lnlen'] = None
    keys, runs, sample = ce._band_data(rows, 'sample')
    _, _, source = ce._band_data(rows, 'source')
    indices = np.random.default_rng(71).integers(len(keys), size=(20, len(keys)))
    run_indices = runs[indices].reshape(20, -1)
    batch = ce._metric_arrays({k: v[run_indices] for k, v in sample.items()},
                              {k: v[run_indices] for k, v in source.items()})[0]
    for draw, expected in zip(indices, batch):
        actual = ce.band_metrics(rows, chart_ids=[keys[i] for i in draw])['metrics']
        np.testing.assert_allclose(np.array(list(actual.values()), dtype=float), expected, equal_nan=True)


def _scratch_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_actual_source_matches_scratch_windows_and_shuffle_diagnostic():
    root = Path(__file__).resolve().parents[2]
    scratch = root.parent / '.sync/cp/scratch/r2-collapse/opus/collapse.py'
    helper = root.parent / '.sync/cp/scratch/r2-ln-level/lnlib.py'
    adjacency = root / 'artifacts/r2-collapse-20261007/astra/shuffle_diagnostic.py'
    probe = adjacency.with_name('temperature_probe.py')
    cache = root / 'artifacts/r2-cache/v1'
    if not all(p.exists() for p in (scratch, helper, adjacency, probe, cache / 'index.parquet')):
        pytest.skip('Named scratch evidence and source cache are local research inputs')
    from ensomi_model.r2.data import Corpus
    corpus = Corpus(cache, 'fit_dev', star_conditions=False, capacity=1)
    sha = corpus.table[corpus.table.K >= 640].sort_values('sha256').iloc[0].sha256
    chart = corpus.chart(sha)
    expected = _scratch_functions(helper, ['decode'], {'np': np})
    _scratch_functions(scratch, ['longest_lock', 'seq_stats'], expected)
    expected['W'] = 64
    # Entropy marginals are omitted in both outputs; only the required collapse measures are compared.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        original, whole = expected['seq_stats'](chart.head_ms, chart.actions)
    actual, actual_whole = ce.seq_stats(chart.head_ms, chart.actions)
    assert actual_whole == whole
    for a, b in zip(actual, original):
        assert {k: a[k] for k in ce.STATISTICS if k != 'single'} == pytest.approx(
            {k: b[k] for k in ce.STATISTICS if k != 'single'}, abs=1e-14)
    assert len(actual) == len(original)
    p = _scratch_functions(probe, ['decode', 'entropy_counts'], {'np': np})
    expected_adj = _scratch_functions(adjacency, ['joint_metrics', 'measure'],
                                      {'np': np, 'p': SimpleNamespace(**p)})
    expected_value = expected_adj['measure'](chart, chart.actions, sha)
    actual_value = ce.adjacency(chart.head_ms, chart.actions, sha)
    for part in expected_value:
        assert actual_value[part] == pytest.approx(expected_value[part], abs=1e-14)
