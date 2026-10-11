"""Pre-run test 7: real windows train, gradients reach every natural block in phase N, and resume is exact.

The token conditioner is a lead-in form and raises under the default eta, so the trainer is
exercised with FiLM only (the token path keeps its own tests behind ``token_lead_in``). Phase C is
tested in ``test_phases.py``."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.train_ce import Trainer, TrainConfig, build_corpus

from .helpers import tiny_model
from .locality_fixture import fixture_chart

CACHE = Path('artifacts/r2-cache/v1')


def with_n_bar(cfg, draws=48):
    """A quick N_bar estimate for tests (training uses the 2,000-draw draw_sim value)."""
    from ensomi_model.r2.draw_sim import n_bar_key, simulate
    corpus, star, dcfg = build_corpus(cfg, 'fit_train')
    values, _ = simulate(corpus, dcfg, draws, cfg.accumulate, cfg.window, seed=5)
    key = n_bar_key(dcfg, star_conditions=star, window=cfg.window, accumulate=cfg.accumulate,
                    label_sha256=corpus.label_sha256)
    return replace(cfg, n_bar=values['N_bar'], n_bar_ln=max(values['N_bar_ln'], 1.0),
                   n_bar_star=max(values['N_bar_star'], 1.0), n_bar_key=key)


NATURAL = dict(phase='natural', star_conditions='off', draw=dict(selection='natural', p_align=0.0, ipw=False),
               lr=3e-4, lr_min=3e-5, checkpoint_every=250_000, g3c_exposures=[])


def config(tmp_path, **kw):
    """A phase-N configuration (test values for the undecided keys)."""
    cfg = TrainConfig(run_dir=str(tmp_path / 'film'), cache=str(CACHE), device='cpu', threads=4, conditioner='film',
                      accumulate=8, total_exposures=1_000_000, warmup_exposures=1000, freerun=False,
                      **dict(NATURAL, **kw))
    return with_n_bar(cfg)


def grads_reach(model):
    def nonzero(module):
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

    parts = dict(tcn=model.temporal, landmark_read=model.lm_out, exact=model.exact, joint=model.joint,
                 pointer_query=model.pointer_in, pointer_relation=model.pointer_relation,
                 pointer_candidate=model.candidate)
    return {name: nonzero(m) for name, m in parts.items()}


needs_cache = pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')


@needs_cache
def test_train_step_gradients_and_exact_resume(tmp_path):
    cfg = config(tmp_path)
    a = Trainer(cfg, write=False)
    batch = a.next_batch()
    assert len(batch) == 8
    info = a.step(batch)
    assert np.isfinite(info['loss']) and info['heads'] > 0
    reached = grads_reach(a.model)
    assert all(reached.values()), reached
    assert all(p.grad is None and not p.requires_grad for p in a.model.film.parameters())   # phase N: no conditioning
    path = tmp_path / 'after-one.pt'
    torch.save(a.payload(), path)
    a.step(a.next_batch())
    uninterrupted = {k: v.clone() for k, v in a.model.state_dict().items()}

    b = Trainer(cfg, write=False)
    b.load(path)
    b.step(b.next_batch())
    for k, v in b.model.state_dict().items():
        assert torch.equal(v, uninterrupted[k]), k
    assert b.state['exposures'] == a.state['exposures'] and b.state['windows'] == a.state['windows']


@needs_cache
def test_trainer_refuses_tokens_and_foreign_n_bar(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ContractError):
        Trainer(replace(cfg, conditioner='tokens'), write=False)
    with pytest.raises(ContractError):
        Trainer(replace(cfg, n_bar_key='0' * 64), write=False)
    with pytest.raises(ContractError):                               # the draw changed: N_bar must be re-measured
        Trainer(replace(cfg, draw=dict(selection='natural', p_align=0.0, ipw=False, ln_beats=[16])), write=False)


@needs_cache
def test_nonfinite_loss_reloads_skips_halves_and_stops_after_three(tmp_path):
    import json
    cfg = with_n_bar(TrainConfig(run_dir=str(tmp_path / 'nan'), cache=str(CACHE), device='cpu', threads=4,
                                 accumulate=2, window=64, total_exposures=4000, warmup_exposures=100, log_every=1000,
                                 freerun=False, **dict(NATURAL, checkpoint_every=10_000_000)))
    trainer = Trainer(cfg)
    real = trainer.model.window
    calls = {'n': 0}

    def poisoned(*args, **kwargs):
        calls['n'] += 1
        out = real(*args, **kwargs)
        if calls['n'] in (3, 9, 15):
            out.action = out.action * float('nan')
        return out

    trainer.model.window = poisoned
    rc = trainer.train(resume=False)
    assert rc == 3
    events = [json.loads(l) for l in (tmp_path / 'nan' / 'events.jsonl').read_text().splitlines()]
    bad = [e for e in events if e['event'] == 'nonfinite']
    assert [e['action'] for e in bad] == ['reloaded', 'reloaded', 'stop']
    assert [e['lr_mult'] for e in bad] == [0.5, 0.25, 0.125]
    assert trainer.state['lr_mult'] == 0.25 and trainer.state['nan_events'] == 2
    assert all(i in trainer.state['skip'] for e in bad[:2] for i in e['windows'])


def test_checkpointed_temporal_blocks_change_memory_only():
    """Recomputing the TCN blocks in backward gives the same log-probabilities and the same gradient
    for every parameter as keeping their activations."""
    chart = fixture_chart()
    runs = []
    for flag in (False, True):
        model = tiny_model(checkpoint_temporal=flag)
        out = model.window(chart, 0, chart.K + 1)
        (-(out.action.sum() + out.release.sum())).backward()
        runs.append((out, {name: p.grad for name, p in model.named_parameters()}))
    (plain, plain_grads), (recomputed, recomputed_grads) = runs
    assert torch.equal(plain.action, recomputed.action) and torch.equal(plain.release, recomputed.release)
    assert plain_grads.keys() == recomputed_grads.keys()
    for name, grad in plain_grads.items():
        other = recomputed_grads[name]
        assert (grad is None) == (other is None), name
        assert grad is None or torch.equal(grad, other), name
    assert any(float(g.abs().sum()) > 0 for name, g in recomputed_grads.items()
               if name.startswith('temporal.blocks') and g is not None)
