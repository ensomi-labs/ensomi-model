"""Pre-run test 7: real windows train, gradients reach every block, and resume is exact."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ensomi_model.r2.train_ce import Trainer, TrainConfig

CACHE = Path('artifacts/r2-cache/v1')


def config(tmp_path, conditioner):
    return TrainConfig(run_dir=str(tmp_path / conditioner), cache=str(CACHE), device='cpu', threads=4,
                       conditioner=conditioner, star_conditions='auto', accumulate=8, total_exposures=1_000_000,
                       warmup_exposures=1000, freerun=False)


def grads_reach(model, conditioner):
    def nonzero(module):
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())

    parts = dict(tcn=model.temporal, landmark_read=model.lm_out, exact=model.exact, joint=model.joint,
                 pointer_query=model.pointer_in, pointer_relation=model.pointer_relation,
                 pointer_candidate=model.candidate, conditioner=model.film if conditioner == 'film' else model.tokens)
    return {name: nonzero(m) for name, m in parts.items()}


@pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')
@pytest.mark.parametrize('conditioner', ['film', 'tokens'])
def test_train_step_gradients_and_exact_resume(tmp_path, conditioner):
    cfg = config(tmp_path, conditioner)
    a = Trainer(cfg, write=False)
    batch = a.next_batch()
    assert len(batch) == 8
    info = a.step(batch)
    assert np.isfinite(info['loss']) and info['heads'] > 0
    reached = grads_reach(a.model, conditioner)
    assert all(reached.values()), reached
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


@pytest.mark.skipif(not (CACHE / 'index.parquet').exists(), reason='cache not built here')
def test_nonfinite_loss_reloads_skips_halves_and_stops_after_three(tmp_path):
    import json
    cfg = TrainConfig(run_dir=str(tmp_path / 'nan'), cache=str(CACHE), device='cpu', threads=4, accumulate=2,
                      window=64, total_exposures=4000, warmup_exposures=100, log_every=1000,
                      checkpoint_every=10_000_000, freerun=False)
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
