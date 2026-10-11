"""Whole-song LN-level input, strict warm loading, and independent dropout resume."""
from dataclasses import asdict, replace
import importlib
import sys
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.features import LN_LEVEL_EPS, QUERY_DIM, ln_level_features, query_features
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.train_ce import TrainConfig, check_config, load_warm_start, lr_at, model_config

from .helpers import random_decisions, random_skeleton, two_segment_grid


CONFIGS = Path('src/ensomi_model/r2/configs')
BASELINE = Path('artifacts/r2-lnlevel-20261007/baseline/src/ensomi_model/r2')
OLD_CHECKPOINT = Path('artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0048000198.pt')
SMALL = dict(hidden=16, levels=2, expansion=2, rank=4, film_width=8)


def chart(seed=173, K=19):
    rng = np.random.default_rng(seed)
    head, song = random_skeleton(rng, K)
    return random_decisions(rng, (head, song, two_segment_grid()))


def save_source(tmp_path, model, **updates):
    data = dict(config=dict(phase='natural'), model_config=asdict(model.config),
                model=model.state_dict(), state=dict(exposures=48_000_198))
    data.update(updates)
    path = tmp_path / 'source.pt'
    torch.save(data, path)
    return path


def frozen_modules():
    if not BASELINE.exists():
        pytest.skip('frozen pre-change R2 source is not available')
    # A distinct package keeps every relative R2 import on the frozen source tree.
    name = 'ensomi_model._ln_level_baseline'
    package = types.ModuleType(name)
    package.__path__ = [str(BASELINE.resolve())]
    sys.modules[name] = package
    return importlib.import_module(name + '.features'), importlib.import_module(name + '.model')


@pytest.mark.parametrize('level', [None, 0.0, 0.2, 0.5, 1.0])
def test_feature_channels(level):
    source = chart()
    ks = np.arange(source.K + 1)
    old = query_features(source, ks)
    assert QUERY_DIM == 150 and old.shape == (len(ks), 2, 150)
    assert old.tobytes() == query_features(source, ks, ln_level='off', level=level).tobytes()
    result = query_features(source, ks, ln_level='on', level=level)
    assert result.shape == (len(ks), 2, 153)
    assert old.tobytes() == result[..., :QUERY_DIM].tobytes()
    expected = ln_level_features(level)
    assert np.array_equal(result[..., QUERY_DIM:], np.broadcast_to(expected, (len(ks), 2, 3)))
    if level is None:
        assert np.array_equal(expected, [0.0, 0.0, 0.0])
    else:
        clipped = np.clip(level, LN_LEVEL_EPS, 1 - LN_LEVEL_EPS)
        assert expected[0] == 1.0 and expected[1] == np.float32(level)
        assert expected[2] == np.float32(np.log(clipped / (1 - clipped)))


@pytest.mark.parametrize('value', [True, '0.4', [], -0.1, 1.1, np.nan, np.inf])
def test_invalid_level_fails(value):
    with pytest.raises(ContractError, match='ln_level'):
        ln_level_features(value)


@pytest.mark.parametrize('seed', [1, 171, 954])
def test_off_matches_frozen_code_bytes(seed):
    features, models = frozen_modules()
    source = chart(seed=seed)
    ks = np.arange(source.K + 1)
    assert query_features(source, ks).tobytes() == features.query_features(source, ks).tobytes()
    torch.manual_seed(seed)
    before = models.R2Model(models.R2Config(**SMALL)).eval()
    old_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    after = R2Model(R2Config(**SMALL)).eval()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert before.state_dict().keys() == after.state_dict().keys()
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key]), key
    with torch.no_grad():
        a = before.window(source, 0, source.K + 1)
        b = after.window(source, 0, source.K + 1, ln_level=0.8)
    for field in ('logp', 'action', 'release'):
        assert torch.equal(getattr(a, field), getattr(b, field)), field


@pytest.mark.parametrize('seed', [2, 171, 955])
def test_zero_reader_preserves_random_state_action_probabilities(tmp_path, seed):
    torch.manual_seed(seed)
    source_model = R2Model(R2Config(**SMALL)).eval()
    target = R2Model(R2Config(**SMALL, ln_level='on')).eval()
    record = load_warm_start(save_source(tmp_path, source_model), target)
    assert record['missing_keys'] == ['ln_level_reader.weight']
    assert not torch.count_nonzero(target.ln_level_reader.weight)
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, target.state_dict()[key]), key
    source = chart(seed)
    with torch.no_grad():
        old = source_model.window(source, 0, source.K + 1)
        for level in (None, 0.0, 0.3, 1.0):
            new = target.window(source, 0, source.K + 1, ln_level=level)
            assert torch.equal(torch.isfinite(old.logp), torch.isfinite(new.logp))
            torch.testing.assert_close(old.logp, new.logp, atol=1e-6, rtol=0)
            torch.testing.assert_close(old.release, new.release, atol=1e-6, rtol=0)


def test_actual_checkpoint_matches_frozen_off_and_zero_reader():
    if not OLD_CHECKPOINT.exists():
        pytest.skip('selected phase-N checkpoint is not available')
    _, models = frozen_modules()
    data = torch.load(OLD_CHECKPOINT, map_location='cpu', weights_only=False)
    old = models.R2Model(models.R2Config(**data['model_config'])).eval()
    old.load_state_dict(data['model'], strict=True)
    off = R2Model(R2Config(**data['model_config'])).eval()
    off.load_state_dict(data['model'], strict=True)
    on = R2Model(replace(off.config, ln_level='on')).eval()
    receipt = load_warm_start(OLD_CHECKPOINT, on)
    assert receipt['loaded_parameters'] == len(data['model'])
    assert receipt['missing_keys'] == ['ln_level_reader.weight']
    with torch.no_grad():
        for seed in (10, 21, 173):
            source = chart(seed)
            a = old.window(source, 0, source.K + 1)
            b = off.window(source, 0, source.K + 1)
            assert torch.equal(a.logp, b.logp)
            assert torch.equal(a.release, b.release)
            for level in (None, 0.0, 0.5, 1.0):
                c = on.window(source, 0, source.K + 1, ln_level=level)
                assert torch.equal(torch.isfinite(a.logp), torch.isfinite(c.logp))
                torch.testing.assert_close(a.logp, c.logp, atol=1e-6, rtol=0)


@pytest.mark.parametrize('mutation', ['missing', 'unexpected', 'shape', 'config', 'phase', 'nonzero'])
def test_warm_start_rejects_incomplete_or_mismatched_source(tmp_path, mutation):
    old = R2Model(R2Config(**SMALL))
    path = save_source(tmp_path, old)
    data = torch.load(path, map_location='cpu', weights_only=False)
    new = R2Model(R2Config(**SMALL, ln_level='on'))
    if mutation == 'missing':
        del data['model']['film.norm.weight']
    elif mutation == 'unexpected':
        data['model']['foreign'] = torch.zeros(1)
    elif mutation == 'shape':
        data['model']['exact.0.weight'] = data['model']['exact.0.weight'][:1]
    elif mutation == 'config':
        data['model_config']['memory'] = 'none'
    elif mutation == 'phase':
        data['config']['phase'] = 'conditions'
    else:
        with torch.no_grad():
            new.ln_level_reader.weight.fill_(1)
    torch.save(data, path)
    with pytest.raises(ContractError, match='warm_start'):
        load_warm_start(path, new)


def test_reader_receives_gradient_and_unknown_is_zero():
    model = R2Model(R2Config(**SMALL, ln_level='on'))
    source = chart()
    out = model.window(source, 0, source.K + 1, ln_level=0.25)
    (-out.total.sum()).backward()
    assert torch.count_nonzero(model.ln_level_reader.weight.grad)
    model.zero_grad(set_to_none=True)
    out = model.window(source, 0, source.K + 1, ln_level=None)
    (-out.total.sum()).backward()
    assert not torch.count_nonzero(model.ln_level_reader.weight.grad)


def test_likelihood_entrypoints_forward_the_level():
    model = R2Model(R2Config(**SMALL, ln_level='on')).eval()
    with torch.no_grad():
        model.ln_level_reader.weight.normal_(0, 0.5)
        source = chart(K=7)
        out = model.window(source, 0, source.K + 1, ln_level=0.25)
        assert torch.equal(model.sequence_log_prob(source, 0, source.K + 1, ln_level=0.25), out.total.sum())
        single = model.window(source, 2, 3, ln_level=0.25)
        assert torch.equal(model.decision_log_prob(source, 2, ln_level=0.25), single.total[0])
        assert not torch.equal(out.logp, model.window(source, 0, source.K + 1, ln_level=None).logp)


def test_config_and_schedule():
    cfg = TrainConfig.load(CONFIGS / 'ce_v2_n_lnlevel_ft.json')
    check_config(cfg)
    assert model_config(cfg).ln_level == 'on'
    assert cfg.ln_level_dropout == 0.3 and cfg.seed_ln_level == 1471
    assert cfg.total_exposures == 12_000_000 and cfg.checkpoint_every == 4_000_000
    assert cfg.full_eval_every == 1 and cfg.threads == 4
    assert cfg.warmup_exposures == 200_000
    assert lr_at(cfg, 200_000) == 1e-4
    assert lr_at(cfg, 12_000_000) == 3e-5
    for kwargs in (dict(ln_level='bad'), dict(ln_level_dropout=-0.1), dict(ln_level_dropout=float('nan')),
                   dict(seed_ln_level=-1), dict(warm_start='')):
        with pytest.raises(ContractError):
            check_config(replace(cfg, **kwargs))
    with pytest.raises(ContractError):
        R2Config(ln_level='yes')


def synthetic_trainer(monkeypatch, tmp_path, **overrides):
    from ensomi_model.r2 import train_ce
    from ensomi_model.r2.data import Draw
    from ensomi_model.r2.draw_sim import n_bar_key

    class Corpus:
        baseline = label_sha256 = None

        def __init__(self):
            self.source = chart(K=11)

        def draw(self, rng, window):
            start = int(rng.integers(0, 3))
            return Draw('synthetic', start, self.source.K + 1, (), {})

        def chart(self, sha):
            return self.source

        def source_ln_level(self, sha):
            return 0.25

    cfg = replace(TrainConfig.load(CONFIGS / 'ce_v2_n.json'), **SMALL, run_dir=str(tmp_path),
                  cache=str(tmp_path), threads=1, accumulate=2, window=11, **overrides)
    draw = train_ce.draw_config(cfg)
    cfg.n_bar_key = n_bar_key(draw, star_conditions=False, window=cfg.window, accumulate=cfg.accumulate,
                             label_sha256=None)
    monkeypatch.setattr(train_ce, 'build_corpus', lambda cfg, role: (Corpus(), False, draw))
    monkeypatch.setattr(train_ce, 'cache_hashes', lambda root: {})
    return train_ce.Trainer(cfg, write=False)


def test_dropout_does_not_shift_draws_and_resume_is_exact(monkeypatch, tmp_path):
    off = synthetic_trainer(monkeypatch, tmp_path)
    on = synthetic_trainer(monkeypatch, tmp_path, ln_level='on')
    known = []
    for _ in range(8):
        a, b = off.next_batch(), on.next_batch()
        assert [(d.sha, d.start, d.stop) for _, d in a] == [(d.sha, d.start, d.stop) for _, d in b]
        assert off.rng.bit_generator.state == on.rng.bit_generator.state
        known.extend(d.ln_level is not None for _, d in b)
    assert any(known) and not all(known)
    on.step(on.next_batch())
    path = tmp_path / 'resume.pt'
    torch.save(on.payload(), path)
    expected_batch = on.next_batch()
    on.step(expected_batch)
    restored = synthetic_trainer(monkeypatch, tmp_path, ln_level='on')
    restored.load(path)
    resumed_batch = restored.next_batch()
    assert [(i, d.start, d.ln_level) for i, d in expected_batch] == [
        (i, d.start, d.ln_level) for i, d in resumed_batch]
    restored.step(resumed_batch)
    assert restored.state == on.state
    for name, value in on.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name]), name
    assert on.ln_level_rng.bit_generator.state == restored.ln_level_rng.bit_generator.state
    assert on.rng.bit_generator.state == restored.rng.bit_generator.state


def test_dropout_frequency_and_skipped_windows_keep_seed_alignment(monkeypatch, tmp_path):
    trainer = synthetic_trainer(monkeypatch, tmp_path, ln_level='on')
    trainer.state['skip'] = [0, 3, 10]
    expected = np.random.default_rng(trainer.cfg.seed_ln_level).random(5003) >= 0.3
    observed = []
    for _ in range(2500):
        for index, draw in trainer.next_batch():
            known = draw.ln_level is not None
            assert known == expected[index]
            observed.append(known)
    assert trainer.state['windows'] == 5003
    assert 0.28 < 1 - np.mean(observed) < 0.32


def test_on_initialization_preserves_legacy_weights_and_torch_rng():
    torch.manual_seed(171)
    off = R2Model(R2Config(**SMALL))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(171)
    on = R2Model(R2Config(**SMALL, ln_level='on'))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in off.state_dict().items():
        assert torch.equal(value, on.state_dict()[name]), name


def test_warm_trainer_starts_fresh_and_resume_checks_dropout(monkeypatch, tmp_path):
    old = synthetic_trainer(monkeypatch, tmp_path)
    old.step(old.next_batch())
    source = tmp_path / 'warm.pt'
    torch.save(old.payload(), source)
    new = synthetic_trainer(monkeypatch, tmp_path, warm_start=str(source), ln_level='on')
    assert new.state['exposures'] == new.state['windows'] == new.state['steps'] == 0
    assert new.opt.state_dict()['state'] == {}
    assert new.warm_start['exposures'] == old.state['exposures']
    for key, value in old.model.state_dict().items():
        assert torch.equal(value, new.model.state_dict()[key]), key
    path = tmp_path / 'on.pt'
    torch.save(new.payload(), path)
    other = synthetic_trainer(monkeypatch, tmp_path, warm_start=str(source), ln_level='on', ln_level_dropout=0.4)
    with pytest.raises(ContractError, match='dropout configuration'):
        other.load(path)
    data = torch.load(path, map_location='cpu', weights_only=False)
    del data['ln_level_rng']
    torch.save(data, path)
    with pytest.raises(ContractError, match='dropout RNG state'):
        new.load(path)
