"""Closure ownership, strict duration boundaries and the binding guard-v2 selection rules."""
import json

import numpy as np
import pytest

from ensomi_model.r2.defect_references import defect_summary, dense_births, model_defects
from ensomi_model.r2.generate import defects
from ensomi_model.r2.properties import PObject, prefix_objects
from ensomi_model.r2.select import guard_iv_v1, guards, select


def decisions():
    # At decision 1 lane 0 closes at exactly 60 ms. Lane 1 was opened by the seed,
    # and closes inside decision 2's gap, so decision 2 still owns that hold.
    head = np.array([0., 60., 100., 150.])
    acts = np.array([[2, 0, 0, 0], [1, 2, 1, 0], [0, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    gap = np.full(acts.shape, np.nan)
    gap[2, 1] = 99.
    return head, acts, gap


def test_strict_short_and_closing_decision_ownership():
    head, acts, gap = decisions()
    old = defects(prefix_objects(head, acts, gap), head)
    assert old == dict(holds=2, holds_le60=2, release_1_40_before_other_head=2)
    assert model_defects(head, acts, gap) == dict(holds=2, holds_lt60=1, release_1_40=2)
    assert model_defects(head, acts, gap, first_generated=2) == dict(holds=1, holds_lt60=1, release_1_40=1)
    assert model_defects(head, acts, gap, first_generated=3) == dict(holds=0, holds_lt60=0, release_1_40=0)
    assert defects(prefix_objects(head, acts, gap), head) == old


def test_near_head_uses_other_lanes_and_inclusive_one_to_forty():
    head = np.array([0., 100.])
    acts = np.array([[2, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]])
    gap = np.full(acts.shape, np.nan)
    for end in (60., 99.):
        gap[1, 0] = end
        assert model_defects(head, acts, gap)['release_1_40'] == 0  # same-lane head only
        acts[1, 1] = 1
        assert model_defects(head, acts, gap)['release_1_40'] == 1
        acts[1, 1] = 0
    for end in (59.999, 99.001):
        gap[1, 0] = end
        acts[1, 1] = 1
        assert model_defects(head, acts, gap)['release_1_40'] == 0


def test_eos_closure_and_release_rebirth_each_own_their_hold():
    head = np.array([0., 100.])
    acts = np.array([[2, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0]])
    gap = np.full(acts.shape, np.nan)
    gap[1, 0], gap[2, 0] = 50., 130.
    assert model_defects(head, acts, gap, first_generated=1) == dict(holds=2, holds_lt60=2, release_1_40=0)
    assert model_defects(head, acts, gap, first_generated=2) == dict(holds=1, holds_lt60=1, release_1_40=0)


def reference(short=.1, near=.2):
    return dict(version='defect-references-v1', bands={str(b): dict(holds_lt60_rate=short * (b - 1),
                  release_1_40_rate=near * (b - 1)) for b in (2, 3, 4, 5)})


def test_expected_counts_weight_generated_holds_by_band_and_bind_overall():
    rows = [dict(band=2, holds=10, holds_lt60=2, release_1_40=2),
            dict(band=3, holds=20, holds_lt60=3, release_1_40=8)]
    result = defect_summary(rows, reference())
    assert result['overall']['holds_lt60']['expected'] == pytest.approx(5.)
    assert result['overall']['release_1_40']['expected'] == pytest.approx(10.)
    assert result['guard_iv']
    assert not result['by_band']['2']['guard_iv']  # per-band ratios are diagnostic
    rows[0]['holds_lt60'] = 4
    assert not defect_summary(rows, reference())['guard_iv']


def test_zero_expected_counts_are_decided_without_ratio_division():
    rows = [dict(band=2, holds=10, holds_lt60=0, release_1_40=0)]
    good = defect_summary(rows, reference(0., 0.))
    assert good['guard_iv'] and good['overall']['holds_lt60']['ratio'] == 0.
    rows[0]['holds_lt60'] = 1
    bad = defect_summary(rows, reference(0., 0.))
    assert not bad['guard_iv'] and bad['overall']['holds_lt60']['ratio'] is None
    assert not defect_summary([], reference())['guard_iv']


def test_dense_birth_denominator_is_heads_on_dense_rows():
    objects = [PObject(0., 20., 0, True), PObject(0., 0., 1, False),
               PObject(60., 110., 2, True), PObject(121., 140., 3, True)]
    assert dense_births(objects, np.array([0., 60., 121.])) == dict(heads=2, ln_heads=1, rate=.5)


def record(exposures=30, drift=.05):
    return dict(exposures=exposures, checkpoint=f'ckpt-{exposures}.pt', safe=False,
                natural_manifest=dict(windows=[dict(nll=1., decisions=1, group_id='a')]),
                panels=dict(legality=dict(illegal=0, heads_missing=0), prefix_natural=dict(guard_i=True),
                            calibration=dict(guard_iii=True), defects_v2=dict(guard_iv=True),
                            natural_bos=dict(drift=dict(mean=drift)),
                            defects=dict(holds_le60_rate=.5, release_1_40_rate=.3),
                            source_defects=dict(release_1_40_rate=.2)))


def test_v2_guards_and_nonbinding_old_guard():
    rec = record()
    assert all(guards(rec, 'natural').values())
    assert not guard_iv_v1(rec['panels'])
    rec['panels']['natural_bos']['drift']['mean'] = -.050001
    assert not guards(rec, 'natural')['v']
    del rec['panels']['defects_v2']
    assert not guards(rec, 'natural')['iv']


def test_select_earliest_plateau_candidate_ignores_iv_v1(tmp_path):
    (tmp_path / 'evals.jsonl').write_text('\n'.join(json.dumps(record(e)) for e in (10, 20, 30, 40)))
    result, selected = select(tmp_path, 0, 'natural')
    assert result['selected'] == 'ckpt-30.pt'
    assert selected['exposures'] == 30
    assert result['candidates'][0]['nonbinding']['iv_v1'] is False
    assert all(c['within_two_se'] and c['se_to_min'] == 0. for c in result['candidates'])


def test_evaluator_enabled_modes_and_prior_selection(tmp_path, monkeypatch):
    import types
    import pandas as pd
    import torch
    from ensomi_model.r2 import evaluate
    from ensomi_model.r2.ln_level import EmpiricalLNPrior, whole_ln_level
    from ensomi_model.r2.select import window_table
    from .helpers import chart_from_objects, hold, tap, tiny_model

    torch.set_num_threads(1)
    chart = chart_from_objects([hold(100, 400, 0), tap(200, 1), tap(300, 2), tap(500, 3)], 700)[0]
    model = tiny_model(torch.float32, ln_level='on')
    ev = object.__new__(evaluate.Evaluator)
    ev.model = ev._cpu = model
    ev.cfg = types.SimpleNamespace(cache=str(tmp_path), seed_validation=954)
    ev.conditions, ev.g3c, ev.min_hold_ms, ev.baseline, ev.code = False, False, None, None, None
    ev.write_dir, ev.ln_mode = tmp_path, 'unknown'
    ev.dev = types.SimpleNamespace(chart=lambda sha: chart, band={'a': 2, 'b': 2},
                                    table=pd.DataFrame([dict(sha256=s, star=2.) for s in ('a', 'b')]))
    prior_rows = [dict(sha256=f'{b}-{i}', role='fit_train', star=float(b), K=i + 1,
                       song_ms=1000., n_objects=10, n_ln=i) for b in (2, 3, 4, 5) for i in range(3)]
    ev._ln_prior = EmpiricalLNPrior.fit(prior_rows)
    ev.m = dict(natural=dict(windows=[dict(sha256='a', group_id='g', start=0, stop=chart.K + 1)]))
    monkeypatch.setattr(evaluate, 'load_reference', lambda cache: reference())
    with torch.no_grad():
        manifests = ev.natural_manifests()
        result = ev.panels(panel_charts=dict(prefix=['a', 'b'], onset=[]), natural_only=True)
    assert manifests['natural_manifest_selection_mode'] == 'prior'
    assert set(manifests['natural_manifest_modes']) == {'oracle', 'unknown', 'prior'}
    oracle = manifests['natural_manifest_modes']['oracle']['ln_levels']['a']
    assert oracle['value'] == whole_ln_level(chart)
    assert manifests['natural_manifest_modes']['unknown']['ln_levels']['a']['value'] is None
    assert manifests['natural_manifest'] is manifests['natural_manifest_modes']['prior']
    assert np.array_equal(window_table(manifests)[0], np.array([manifests['natural_manifest']['windows'][0]['nll']]))
    p = result['panels']
    assert p['ln_level_selection_mode'] == 'prior'
    assert set(p['ln_level_modes']) == {'oracle', 'unknown', 'prior'}
    assert p['defects_v2'] == p['ln_level_modes']['prior']['defects_v2']
    assert ev.ln_mode == 'unknown'
    records = [json.loads(line) for line in (tmp_path / 'records.jsonl').read_text().splitlines()]
    assert len(records) == 36
    prior_records = [r for r in records if r['ln_level']['mode'] == 'prior']
    assert len(prior_records) == 12
    for row in prior_records:
        paired = [r['ln_level'] for r in prior_records if r['sha256'] == row['sha256'] and
                  r['random_seed'] == row['random_seed']]
        assert paired[0] == paired[1]  # same per-song draw for BOS and prefix modes
    nested = record()
    nested['panels']['ln_level_modes'] = dict(prior=record()['panels'])
    nested['panels']['calibration']['guard_iii'] = False
    assert guards(nested, 'natural')['iii']

    forecasts, samples = [], []
    original_forecast, original_continue = evaluate._forecast, evaluate.continue_chart

    def forecast(*args, **kwargs):
        forecasts.append(kwargs['ln_level'])
        return original_forecast(*args, **kwargs)

    def continuation(*args, **kwargs):
        samples.append(kwargs['ln_level'])
        return original_continue(*args, **kwargs)

    monkeypatch.setattr(evaluate, '_forecast', forecast)
    monkeypatch.setattr(evaluate, 'continue_chart', continuation)
    for mode in ('oracle', 'unknown', 'prior'):
        forecasts.clear()
        samples.clear()
        ev.ln_mode = mode
        calibrated = ev.calibration(['a'], states=2, horizon=2)
        levels = [r['value'] for r in calibrated['ln_levels']]
        assert len(levels) == 2
        assert forecasts == [levels[0], levels[0], levels[1], levels[1]]
        assert samples == levels
        assert all(r['mode'] == mode for r in calibrated['ln_levels'])
