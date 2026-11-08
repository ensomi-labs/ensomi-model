"""Clean-stop and failed-step recovery without corpus data or a model fit."""
from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('release_train',
    Path(__file__).resolve().parents[2] / 'scripts/r2/release_train.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@dataclass
class Config:
    run_dir: str
    total_exposures: int = 4
    log_every: int = 1
    rss_limit_gib: int = 100


@pytest.mark.parametrize('interruption', ['signal', 'failed_step'])
def test_resume_restores_draw_order_and_retains_only_final(tmp_path, monkeypatch, interruption):
    config = Config(str(tmp_path))
    pending = [True]
    class FakeTrainer:
        def __init__(self):
            self.rng = np.random.default_rng(471)
            self.state = dict(exposures=0, windows=0, steps=0, wall_s=0.)
            self.weight = 0.
        def next_batch(self):
            self.state['windows'] += 1
            return float(self.rng.random())
        def step(self, value):
            if interruption == 'failed_step' and pending[0] and self.state['windows'] == 3:
                pending[0] = False
                raise RuntimeError('synthetic failure after a draw, before update')
            self.weight += value
            self.state['exposures'] += 1
            self.state['steps'] += 1
            if interruption == 'signal' and pending[0]:
                pending[0] = False
                os.kill(os.getpid(), signal.SIGTERM)
            return dict(loss=self.weight)
        def payload(self):
            return dict(state=self.state, draw_rng=self.rng.bit_generator.state,
                        weight=self.weight, config=asdict(config))
        def load(self, path):
            d = torch.load(path, weights_only=False)
            self.state, self.weight = d['state'], d['weight']
            self.rng.bit_generator.state = d['draw_rng']
    monkeypatch.setattr(runner, 'construct', lambda *a, **kw: FakeTrainer())
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=110_000_000_000))
    args = SimpleNamespace(representation='r2', resume=False, recovery_every=2)
    assert runner.train(args, config) == 75
    recovery = torch.load(tmp_path / 'resume.pt', weights_only=False)
    assert recovery['state']['windows'] == (1 if interruption == 'signal' else 2)
    args.resume = True
    assert runner.train(args, config) == 0
    final = torch.load(tmp_path / 'final.pt', weights_only=False)
    assert final['weight'] == pytest.approx(np.random.default_rng(471).random(4).sum())
    assert final['state']['windows'] == 4
    assert not (tmp_path / 'resume.pt').exists()
