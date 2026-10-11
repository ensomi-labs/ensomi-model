"""Plan v5 training phases: the identity gate, the trainable sets of phase N and phase C, the KL term,
the refusal of undecided keys and checkpoints across phases.

The synthetic tests use the locality fixture with tiny CPU float64 models; the trainer tests need the
R2 cache. Engineering tests, not evidence about learning."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.loss import LossConfig, decision_kl, window_kl, window_loss, window_terms
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.sampling import continue_chart
from ensomi_model.r2.train_ce import (TrainConfig, check_config, make_optimizer, trainable_parameters)

from .helpers import tiny_model
from .locality_fixture import EXPECTED_V, I1, I2, STAR, fixture_chart

CACHE = Path('artifacts/r2-cache/v1')
CONFIGS = Path('src/ensomi_model/r2/configs')
TRACK = (I1, I2, STAR)
needs_cache = pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')


@pytest.fixture(scope='module')
def chart():
    return fixture_chart()


def phase_models(seed=0, **kw):
    """(phase-N model, phase-C model): the same natural parameters, the phase-C conditioning path trained
    (here: drawn at random), as after a frozen-base phase C."""
    natural = tiny_model(torch.float64, seed=seed, **kw)
    conditioned = copy.deepcopy(natural)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in conditioned.film.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1)
    return natural, conditioned


def outputs(model, chart, track, start=0, stop=None):
    stop = chart.K + 1 if stop is None else stop
    with torch.no_grad():
        out = model.window(chart, start, stop, track)
    return out.action, out.release, out.logp


def visible_rows(out):
    return out.visible.any(1)


# ---- the identity gate --------------------------------------------------------------------------------

def test_gate_conditioned_model_equals_phase_n_on_an_empty_track(chart):
    natural, conditioned = phase_models()
    for a, b in zip(outputs(natural, chart, ()), outputs(conditioned, chart, ())):
        assert torch.equal(a, b)
    rows_n = continue_chart(natural, chart.head_ms, chart.song_ms, chart.grid, seed=954)
    rows_c = continue_chart(conditioned, chart.head_ms, chart.song_ms, chart.grid, seed=954)
    assert np.array_equal(rows_n[0], rows_c[0]) and np.array_equal(rows_n[1], rows_c[1], equal_nan=True)


def test_gate_decisions_outside_v_equal_phase_n_and_inside_v_differ(chart):
    natural, conditioned = phase_models()
    with torch.no_grad():
        out = conditioned.window(chart, 0, chart.K + 1, TRACK)
    act_n, rel_n, logp_n = outputs(natural, chart, ())
    outside = ~visible_rows(out)
    assert outside.sum() > 20 and (~outside).sum() > 20
    o = torch.as_tensor(outside)
    assert torch.equal(out.action[o], act_n[o]) and torch.equal(out.release[o], rel_n[o])
    assert torch.equal(out.logp[o], logp_n[o])
    inside = np.flatnonzero(~outside)
    assert all(int(k) in inside for k in EXPECTED_V['I1'] + EXPECTED_V['I2'])
    assert all(float(out.action[i]) != float(act_n[i]) for i in inside)        # power: every decision in V moves


def test_gate_power_without_it_training_film_moves_natural_decisions(chart):
    natural, conditioned = phase_models(identity_gate=False)
    act_n, _, _ = outputs(natural, chart, ())
    act_c, _, _ = outputs(conditioned, chart, ())
    assert not torch.equal(act_n, act_c)


def test_film_capacity_is_configurable_and_starts_at_identity(chart):
    small = R2Model(R2Config())
    assert small.parameter_counts()['film'] == 42_112                  # the v2 default, unchanged
    torch.manual_seed(0)
    wide = R2Model(R2Config(film_width=256, film_layers=2)).double()
    assert wide.parameter_counts()['film'] == (68 * 256 + 256) + (256 * 256 + 256) + (256 * 256 + 256) + 256
    conditioning, natural = wide.parameter_split()
    assert {n.split('.')[0] for n, _ in conditioning} == {'film'}
    assert sum(p.numel() for _, p in conditioning) == wide.parameter_counts()['film']
    assert not any(n.startswith(('film.', 'tokens.')) for n, _ in natural)
    plain = copy.deepcopy(wide)
    with torch.no_grad():                                               # a zero output layer is the identity
        a = wide.window(chart, 0, chart.K + 1, TRACK)
        b = plain.window(chart, 0, chart.K + 1, ())
    assert torch.equal(a.action, b.action) and torch.equal(a.release, b.release)


# ---- trainable sets -----------------------------------------------------------------------------------

def phase_config(phase, base_mode=None):
    return TrainConfig(phase=phase, base_mode=base_mode, lr=1e-2, weight_decay=0.01)


def step(model, cfg, chart, track, natural_ce):
    named = trainable_parameters(model, cfg)
    opt = make_optimizer(named, cfg)
    out = model.window(chart, 0, chart.K + 1, track)
    loss, _ = window_loss(window_terms(out, 0.7), LossConfig(200.0, 60.0, 60.0, 1.0, 1.0, natural_ce=natural_ce))
    loss.backward()
    opt.step()
    return named


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_frozen_phase_c_trains_exactly_the_conditioning_path(chart):
    model = tiny_model(torch.float64, film_width=64, film_layers=2)
    before = snapshot(model)
    named = step(model, phase_config('conditions', 'frozen'), chart, TRACK, natural_ce=False)
    film = {n for n, _ in model.named_parameters() if n.startswith('film.')}
    assert {n for n, _ in named} == film
    assert {n for n, p in model.named_parameters() if p.requires_grad} == film
    after = snapshot(model)
    for n in before:
        if n in film:
            continue
        assert torch.equal(before[n], after[n]), n                       # bit-identical natural parameters
    assert any(not torch.equal(before[n], after[n]) for n in film)


def test_phase_n_leaves_the_conditioning_path_and_kl_mode_trains_both(chart):
    model = tiny_model(torch.float64)
    before = snapshot(model)
    named = step(model, phase_config('natural'), chart, (), natural_ce=True)
    names = {n for n, _ in named}
    assert not any(n.startswith(('film.', 'tokens.')) for n in names)
    after = snapshot(model)
    assert all(torch.equal(before[n], after[n]) for n in before if n.startswith(('film.', 'tokens.')))
    assert not torch.equal(before['exact.0.weight'], after['exact.0.weight'])
    kl_named = trainable_parameters(tiny_model(torch.float64), phase_config('conditions', 'kl'))
    conditioning, natural = tiny_model(torch.float64).parameter_split()
    assert {n for n, _ in kl_named} == {n for n, _ in conditioning} | {n for n, _ in natural}


def test_a_frozen_window_without_visible_decisions_has_no_gradient(chart):
    model = tiny_model(torch.float64)
    trainable_parameters(model, phase_config('conditions', 'frozen'))
    out = model.window(chart, 60, 80, ())
    loss, _ = window_loss(window_terms(out), LossConfig(200.0, 60.0, 60.0, 1.0, 1.0, natural_ce=False))
    assert float(loss) == 0.0 and not loss.requires_grad


# ---- the KL term --------------------------------------------------------------------------------------

def fresh_phase_c(seed=0):
    """A phase-C model at its start: phase-N natural parameters, conditioning output layer zero."""
    model = tiny_model(torch.float64, seed=seed)
    with torch.no_grad():
        model.film.mlp[-1].weight.zero_()
        model.film.mlp[-1].bias.zero_()
    return model


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
@pytest.mark.parametrize('decisions', ['natural', 'all'])
def test_kl_is_zero_when_the_model_equals_its_reference(chart, direction, decisions):
    model = fresh_phase_c()
    reference = copy.deepcopy(model)
    out = model.window(chart, 0, chart.K + 1, TRACK, pairs=True)
    with torch.no_grad():
        ref = reference.window(chart, 0, chart.K + 1, (), pairs=True)
    assert len(out.pair_rows) > 100
    assert torch.equal(decision_kl(out, ref, direction), torch.zeros(chart.K + 1, dtype=torch.float64))
    assert float(window_kl(out, ref, 0.7, decisions, direction).detach()) == 0.0


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_kl_scope_power_and_value(chart, direction):
    reference = fresh_phase_c()
    model = copy.deepcopy(reference)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.film.parameters():                                 # a trained conditioning path
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1)
    out = model.window(chart, 0, chart.K + 1, TRACK, pairs=True)
    with torch.no_grad():
        ref = reference.window(chart, 0, chart.K + 1, (), pairs=True)
    assert float(window_kl(out, ref, 0.7, 'natural', direction).detach()) == 0.0  # natural decisions are gated
    assert float(window_kl(out, ref, 0.7, 'all', direction).detach()) > 0.0
    with torch.no_grad():
        model.exact[0].weight.mul_(1.1)                                   # a moved natural parameter
    out = model.window(chart, 0, chart.K + 1, TRACK, pairs=True)
    kl = decision_kl(out, ref, direction)
    natural = torch.as_tensor(~visible_rows(out))
    assert (kl[natural] > 0).all()
    total = window_kl(out, ref, 0.7, 'natural', direction)
    assert torch.allclose(total, 0.7 * kl[natural].sum(), rtol=1e-12, atol=0)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    i = 5                                                                  # the action part of one decision
    p_ref, p_mod = ref.logp[i].exp().numpy(), out.logp[i].detach().exp().numpy()
    legal = p_ref > 0
    a, b = (p_ref, p_mod) if direction == 'forward' else (p_mod, p_ref)
    expect = float((a[legal] * (np.log(a[legal]) - np.log(b[legal]))).sum())
    assert len(out.pair_rows) and i not in set(out.pair_rows.tolist())    # decision 5 has no release factor
    assert abs(float(kl[i].detach()) - expect) < 1e-10


# ---- undecided keys -----------------------------------------------------------------------------------

DECIDED = dict(total_exposures=1000, checkpoint_every=500, g3c_exposures=[], lr=1e-3, lr_min=1e-4,
               warmup_exposures=10)
PHASE_C = dict(init_from='x.pt', lambda_ln=1.0, lambda_star=1.0, mu_star=0.0)
KL = dict(kl_weight=1.0, kl_direction='forward', kl_decisions='natural', natural_ce=False)


@pytest.mark.parametrize('name,fill', [('ce_v2_n.json', {}), ('ce_v2_c_frozen.json', PHASE_C),
                                       ('ce_v2_c_kl.json', dict(PHASE_C, **KL))])
def test_trainer_refuses_null_required_keys(name, fill):
    cfg = replace(TrainConfig.load(CONFIGS / name), **{key: None for key in DECIDED})
    with pytest.raises(ContractError, match='still null'):
        check_config(cfg)
    from ensomi_model.r2.train_ce import Trainer
    with pytest.raises(ContractError):
        Trainer(cfg, write=False)                                         # before it touches the cache
    full = replace(cfg, **DECIDED, **fill)
    check_config(full)
    for key in list(DECIDED) + list(fill):
        if key == 'lr' or key == 'lr_min' or key == 'warmup_exposures':
            continue                                                      # replaceable by lr_schedule
        with pytest.raises(ContractError, match=key):
            check_config(replace(full, **{key: None}))
    with pytest.raises(ContractError, match='lr'):
        check_config(replace(full, lr=None))
    check_config(replace(full, lr=None, lr_min=None, warmup_exposures=None,
                         lr_schedule=[dict(until=1000, shape='constant', lr_start=1e-3)]))


def test_inapplicable_keys_are_refused():
    n = replace(TrainConfig.load(CONFIGS / 'ce_v2_n.json'), **DECIDED)
    check_config(n)
    for key, value in (('base_mode', 'frozen'), ('lambda_ln', 1.0), ('kl_weight', 0.5), ('init_from', 'x.pt')):
        with pytest.raises(ContractError, match=key):
            check_config(replace(n, **{key: value}))
    frozen = replace(TrainConfig.load(CONFIGS / 'ce_v2_c_frozen.json'), **DECIDED, **PHASE_C)
    check_config(frozen)
    with pytest.raises(ContractError, match='kl_weight'):
        check_config(replace(frozen, kl_weight=1.0))
    with pytest.raises(ContractError):
        check_config(replace(frozen, draw=dict(selection='natural', p_align=0.0, ipw=False)))
    with pytest.raises(ContractError):
        check_config(replace(n, draw=dict(selection='per_window', p_align=0.6, ipw=True)))
    with pytest.raises(ContractError):
        check_config(replace(frozen, base_mode='soft'))


def test_phase_configs_differ_only_where_the_plan_says():
    n, f, k = (json.loads((CONFIGS / name).read_text())
               for name in ('ce_v2_n.json', 'ce_v2_c_frozen.json', 'ce_v2_c_kl.json'))
    assert set(k) - set(f) == {'kl_weight', 'kl_direction', 'kl_decisions', 'natural_ce'} and set(f) <= set(k)
    assert {key for key in f if f[key] != k[key]} == {'base_mode'}
    assert set(f) - set(n) == {'init_from', 'base_mode', 'lambda_ln', 'lambda_star', 'mu_star'} and set(n) <= set(f)
    decided_n = {'total_exposures', 'warmup_exposures', 'lr', 'lr_min', 'checkpoint_every', 'g3c_exposures',
                 'n_bar', 'n_bar_ln', 'n_bar_star', 'n_bar_key'}         # phase N's accepted values; phase C's open
    assert {key for key in n if n[key] != f[key]} == {'phase', 'star_conditions', 'star_value', 'draw'} | decided_n
    assert all(f[key] is None for key in decided_n)
    assert n['memory'] == f['memory'] == k['memory'] == 'none'
    assert not (CONFIGS / 'ce_v2_a0.json').exists() and not (CONFIGS / 'ce_v2_a1.json').exists()


def test_accepted_phase_n_config_passes():
    check_config(TrainConfig.load(CONFIGS / 'ce_v2_n.json'))


# ---- checkpoints across phases (cache) ----------------------------------------------------------------

def run_config(tmp_path, name, **kw):
    from .test_train_step import with_n_bar
    base = dict(run_dir=str(tmp_path / name), cache=str(CACHE), device='cpu', threads=4, accumulate=2, window=64,
                total_exposures=100_000, checkpoint_every=50_000, g3c_exposures=[], warmup_exposures=100, lr=1e-3,
                lr_min=1e-4, freerun=False)
    base.update(kw)
    return with_n_bar(TrainConfig(**base))


def natural_config(tmp_path):
    return run_config(tmp_path, 'n', phase='natural', star_conditions='off',
                      draw=dict(selection='natural', p_align=0.0, ipw=False))


def conditions_config(tmp_path, init, base_mode, name, **kw):
    kl = dict(KL) if base_mode == 'kl' else {}
    kl.update(kw)
    return run_config(tmp_path, name, phase='conditions', init_from=str(init), base_mode=base_mode,
                      star_conditions='auto', star_value='absolute', lambda_ln=1.0, lambda_star=1.0, mu_star=0.0,
                      **kl)


def state_of(model, prefix_excluded=('film.', 'tokens.')):
    return {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(prefix_excluded)}


@needs_cache
def test_checkpoint_round_trip_across_phases(tmp_path):
    from ensomi_model.r2.train_ce import Trainer
    n = Trainer(natural_config(tmp_path), write=False)
    n.step(n.next_batch())
    assert all(p.grad is None for p in n.model.film.parameters())
    phase_n = tmp_path / 'phase-n.pt'
    torch.save(n.payload(), phase_n)
    natural_state = state_of(n.model)

    c = Trainer(conditions_config(tmp_path, phase_n, 'frozen', 'cf'), write=False)
    for k, v in state_of(c.model).items():
        assert torch.equal(v, natural_state[k]), k                     # phase C starts from phase N's natural model
    assert float(c.model.film.mlp[-1].weight.detach().abs().sum()) == 0.0 and c.init['sha256']
    c.step(c.next_batch())
    after_one = tmp_path / 'c-one.pt'
    torch.save(c.payload(), after_one)
    c.step(c.next_batch())
    for k, v in state_of(c.model).items():
        assert torch.equal(v, natural_state[k]), k                     # frozen: still bit-identical
    uninterrupted = {k: v.clone() for k, v in c.model.state_dict().items()}

    resumed = Trainer(conditions_config(tmp_path, phase_n, 'frozen', 'cf'), write=False)
    resumed.load(after_one)
    resumed.step(resumed.next_batch())
    for k, v in resumed.model.state_dict().items():
        assert torch.equal(v, uninterrupted[k]), k                     # exact resume within phase C
    with pytest.raises(ContractError, match='another phase'):
        resumed.load(phase_n)                                          # a phase-N checkpoint is not a phase-C resume
    with pytest.raises(ContractError, match='phase-N checkpoint'):
        Trainer(conditions_config(tmp_path, after_one, 'frozen', 'cf2'), write=False)

    kl = Trainer(conditions_config(tmp_path, phase_n, 'kl', 'ck'), write=False)
    for k, v in state_of(kl.reference).items():
        assert torch.equal(v, natural_state[k]), k
    stats_before = kl.stats
    info = kl.step(kl.next_batch(), stats_before)
    assert np.isfinite(info['loss'])
    assert abs(stats_before.d['loss_kl']) < 1e-6                        # the first step starts at the reference
    assert {n.split('.')[0] for n, p in kl.model.named_parameters() if p.requires_grad} >= {'film', 'temporal'}
    assert not any(p.requires_grad for p in kl.reference.parameters())
