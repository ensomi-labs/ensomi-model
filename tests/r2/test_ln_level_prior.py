"""Whole-song source semantics, empirical supports, and sampling stream isolation."""
from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError, band_of
from ensomi_model.r2.data import Corpus
from ensomi_model.r2.ln_level import (EmpiricalLNPrior, PRIOR_FILE, build_prior, checked_level,
                                     resolve_ln_level, skeleton_density, whole_ln_level)
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.properties import ln_share, prefix_objects
from ensomi_model.r2.sampling import continue_chart

from .helpers import chart_from_objects, hold, tap, tiny_model


def source_chart():
    return chart_from_objects([hold(100, 700, 0), tap(200, 1), hold(300, 600, 2), tap(400, 3),
                               tap(500, 1), tap(800, 2), tap(1000, 0), tap(1200, 3)], 1500)[0]


def prior_records():
    return [dict(sha256=f'{band}-{i}', role='fit_train', star=band + .1, K=i + 1,
                 song_ms=1000., n_objects=20, n_ln=i // 2)
            for band in range(2, 6) for i in range(9)]


@pytest.fixture
def prior():
    return EmpiricalLNPrior.fit(prior_records())


def test_whole_level_matches_object_property_and_survives_row_merge():
    objects = [hold(100, 600, 0), tap(101, 1), tap(400, 2), hold(700, 900, 3)]
    chart, dec = chart_from_objects(objects, 1000)
    assert dec.stats['merged_rows'] == 1
    assert whole_ln_level(chart) == ln_share(objects, 0, 1000, 1000) == .5
    assert whole_ln_level(chart) == dec.stats['n_ln'] / dec.stats['n_objects']
    no_eos = chart.with_decisions(chart.actions[:-1], chart.gap[:-1])
    assert whole_ln_level(no_eos) == .5
    with pytest.raises(ContractError, match='every source head row'):
        whole_ln_level(chart.with_decisions(chart.actions[:2], chart.gap[:2]))


def test_prior_support_uses_all_training_sources_only(prior):
    records = prior_records()
    dev = dict(records[0], sha256='dev', role='fit_dev', n_ln=20)
    again = EmpiricalLNPrior.fit([dev, *records[::-1]])
    assert again.data == prior.data
    assert prior.data['count'] == 36
    for band in range(2, 6):
        part = prior.data['bands'][str(band)]
        assert part['density_thresholds'] == pytest.approx([11 / 3, 19 / 3])
        assert [c['count'] for c in part['cells']] == [3, 3, 3]
        assert [v for c in part['cells'] for v in c['shares']] == [i // 2 / 20 for i in range(9)]
        assert [s for c in part['cells'] for s in c['source_sha256']] == [f'{band}-{i}' for i in range(9)]
    assert skeleton_density([1, 2], 4000) == .5
    assert band_of(6.9) == 5


def test_prior_sampling_threshold_ties_support_and_seed(prior):
    for band in range(2, 6):
        part = prior.data['bands'][str(band)]
        for tercile, density in enumerate((1., *part['density_thresholds'])):
            first = prior.sample(star=band + .2, head_ms=[1], song_ms=1000 / density, seed=17)
            second = prior.sample(star=band + .2, head_ms=[1], song_ms=1000 / density, seed=17)
            assert first == second
            value, info = first
            assert info['density_tercile'] == tercile
            assert info['band'] == band
            assert value in part['cells'][tercile]['shares']
            assert value == part['cells'][tercile]['shares'][info['support_index']]
    data = json.loads(json.dumps(prior.data))
    empty = data['bands']['2']['cells'][0]
    data['bands']['2']['count'] -= empty['count']
    data['count'] -= empty['count']
    empty.update(count=0, shares=[], source_sha256=[])
    with pytest.raises(ContractError, match='no source observations'):
        EmpiricalLNPrior(data).sample(star=2., head_ms=[1], song_ms=1000., seed=1)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), True, '0.4', None])
def test_invalid_numeric_levels_fail(value):
    with pytest.raises(ContractError, match='finite number'):
        checked_level(value)


def test_modes_are_explicit_and_do_not_read_source_for_unknown(prior):
    assert resolve_ln_level('unknown', source_ln_level=.8) == (None, dict(mode='unknown', known=False, value=None))
    assert resolve_ln_level(None, source_ln_level=.8)[0] is None
    assert resolve_ln_level('oracle', source_ln_level=.8) == (.8, dict(mode='oracle', known=True, value=.8))
    for fixed in (0, .3, 1):
        assert resolve_ln_level(fixed)[0] == fixed
    value, info = resolve_ln_level('prior', ln_prior=prior, star=3.4, head_ms=[1], song_ms=1000, seed=17)
    assert info['known'] and info['mode'] == 'prior'
    assert value in prior.data['bands']['3']['cells'][0]['shares']
    for mode, kwargs in [('oracle', {}), ('prior', {}), ('fixed', {}), (float('nan'), {})]:
        with pytest.raises(ContractError):
            resolve_ln_level(mode, **kwargs)


@pytest.mark.parametrize('prefix', [0, 3])
def test_off_sampler_matches_frozen_original_and_ignores_level(prefix):
    from .test_min_hold import ORIGINAL_SAMPLING_SOURCE
    original = types.ModuleType('ensomi_model.r2._sampling_before_ln_level')
    exec(compile(ORIGINAL_SAMPLING_SOURCE, '<sampling-before-ln-level>', 'exec'), original.__dict__)
    chart = source_chart()
    model = tiny_model(seed=19)
    kwargs = dict(prefix_actions=chart.actions[:prefix], prefix_gap=chart.gap[:prefix], seed=954)
    expected = original.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs)
    for mode in ('unknown', 'oracle', 'prior', 0., 1.):
        stats = {}
        actual = continue_chart(model, chart.head_ms, chart.song_ms, chart.grid,
                                **kwargs, ln_level=mode, ln_level_stats=stats)
        assert stats == {}
        for before, after in zip(expected, actual):
            assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize('prefix', [0, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_reader_preserves_actions_and_pointer_draws(prior, prefix, dtype):
    chart = source_chart()
    off = tiny_model(seed=391, dtype=dtype)
    on = R2Model(R2Config(ln_level='on')).to(dtype)
    state = dict(off.state_dict(), **{'ln_level_reader.weight': on.ln_level_reader.weight.detach().clone()})
    on.load_state_dict(state, strict=True)
    assert torch.count_nonzero(on.ln_level_reader.weight) == 0
    kwargs = dict(prefix_actions=chart.actions[:prefix], prefix_gap=chart.gap[:prefix], seed=17, min_hold_ms=60)
    expected = continue_chart(off, chart.head_ms, chart.song_ms, chart.grid, **kwargs)
    for mode in ('unknown', 'oracle', 'prior', 0., 1.):
        stats, levels = {}, []
        query = on.query_features

        def capture(chart, ks, ln_level=None):
            levels.append(ln_level)
            return query(chart, ks, ln_level=ln_level)

        on.query_features = capture
        actual = continue_chart(on, chart.head_ms, chart.song_ms, chart.grid, **kwargs, ln_level=mode,
                                source_ln_level=whole_ln_level(chart), ln_prior=prior, star=3.4,
                                ln_level_stats=stats)
        on.query_features = query
        assert levels == [stats['value']] * (chart.K + 1 - prefix)
        assert stats['known'] == (mode != 'unknown')
        for before, after in zip(expected, actual):
            assert before.tobytes() == after.tobytes()


def test_prior_does_not_advance_global_random_streams(prior):
    np.random.seed(771)
    torch.manual_seed(891)
    np_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
    prior.sample(star=3.4, head_ms=[1], song_ms=1000, seed=17)
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


def test_prior_seed_varies_by_song_within_one_cell(prior):
    draws = []
    for offset in range(64):
        kwargs = dict(star=3.4, head_ms=[100. + offset], song_ms=1000., seed=17)
        value, record = prior.sample(**kwargs)
        assert (value, record) == prior.sample(**kwargs)
        assert record['density_tercile'] == 0
        draws.append((value, record))
    assert len({r['skeleton_sha256'] for _, r in draws}) == 64
    assert len({r['support_index'] for _, r in draws}) == 3
    assert len({value for value, _ in draws}) > 1


def test_cached_prior_counts_match_decision_heads(tmp_path):
    root = Path(__file__).resolve().parents[2] / 'artifacts/r2-cache/v1'
    if not (root / 'index.parquet').exists():
        pytest.skip('Cached prior provenance check requires the R2 cache')
    corpus = Corpus(root, 'fit_train', star_conditions=False)
    prior = build_prior(root, out=tmp_path / PRIOR_FILE)
    assert prior.data['count'] == len(corpus.table)
    shas = {sha for p in prior.data['bands'].values() for c in p['cells'] for sha in c['source_sha256']}
    assert shas == set(corpus.table.sha256)
    chosen = set(corpus.table.sort_values(['merged_rows', 'sha256'], ascending=[False, True]).head(3).sha256)
    for field in ('n_ln', 'n_objects', 'song_ms'):
        chosen.update(corpus.table.nsmallest(1, field).sha256)
        chosen.update(corpus.table.nlargest(1, field).sha256)
    for row in corpus.table[corpus.table.sha256.isin(chosen)].itertuples():
        chart = corpus.chart(row.sha256)
        objects = prefix_objects(chart.head_ms, chart.actions, chart.gap)
        assert corpus.source_ln_level(row.sha256) == row.n_ln / row.n_objects
        assert whole_ln_level(chart) == ln_share(objects, 0, chart.song_ms, chart.song_ms)
        assert len(objects) == row.n_objects
        assert sum(o.hold for o in objects) == row.n_ln
    assert EmpiricalLNPrior.load(tmp_path / PRIOR_FILE).data == prior.data
