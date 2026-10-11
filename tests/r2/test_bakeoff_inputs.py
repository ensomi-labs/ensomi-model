from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ensomi_model.r2.common import MIRROR_ACTION
from ensomi_model.r2.features import Chart, anchor_features, history_tokens, query_features, theta_features
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.theta import (COORDINATES, WHOLE, ThetaTable, chart_coordinates, resolve_theta,
                                   skeleton_features)
from ensomi_model.r2.train_ce import TrainConfig, check_config, load_warm_start

from .helpers import random_decisions, random_skeleton, two_segment_grid
from .test_ln_level_model import SMALL, synthetic_trainer
from .test_mirror import mirrored

CHECKPOINT = Path('artifacts/r2-runs/r2-phaseN-20261006/checkpoints/ckpt-0056000574.pt')


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def source(K=31):
    rng = np.random.default_rng(171)
    head, song = random_skeleton(rng, K)
    return random_decisions(rng, (head, song, two_segment_grid()))


def table_fixture():
    rng = np.random.default_rng(182)
    table = pd.DataFrame(rng.normal(size=(42, len(WHOLE))), columns=WHOLE)
    table['sha256'] = ['synthetic'] + [f'{i:064x}' for i in range(1, len(table))]
    table['role'] = ['fit_train'] * 40 + ['fit_dev'] * 2
    for coordinate in COORDINATES:
        table[f'raw_{coordinate}'] = rng.normal(size=len(table))
        table[f'half_{coordinate}'] = rng.normal(size=len(table))
    table['raw_hand'] = rng.uniform(.2, .8, len(table))
    return table


def test_anchor_prefix_only_and_hand_equivariance():
    chart = source()
    ks = np.array([0, 1, 4, 17, chart.K])
    value = anchor_features(chart, ks)
    assert value.shape == (5, 2, 13)
    assert not value[0].any()
    for i, k in enumerate(ks):
        prefix = chart.with_decisions(chart.actions[:k], chart.gap[:k])
        np.testing.assert_array_equal(value[i], anchor_features(prefix, [k])[0])
    np.testing.assert_array_equal(value[:, ::-1], anchor_features(mirrored(chart), ks))
    d = chart.derived()
    assert value[-1, 0, 0] == np.float32(d.attack[:chart.K].sum() / chart.K)
    assert value[-1, 0, 5] == np.float32(d.ln_head.sum() / d.attack.sum())
    assert value[-1, :, -1].sum() == pytest.approx(1)
    counts = np.bincount((d.attack[:chart.K] * (1 << np.arange(4))).sum(1), minlength=16)[1:]
    p = counts[counts > 0] / counts.sum()
    assert value[-1, 0, 7] == pytest.approx(-(p * np.log2(p)).sum(), rel=1e-6)


@pytest.mark.parametrize('theta', ['off', 'on'])
def test_zero_readers_preserve_initialization_rng(theta):
    torch.manual_seed(171)
    old = R2Model(R2Config(**SMALL))
    old_rng = torch.get_rng_state()
    torch.manual_seed(171)
    new = R2Model(R2Config(**SMALL, anchor='on', theta=theta))
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    chart = source(9)
    with torch.no_grad():
        a = old.window(chart, 0, chart.n)
        b = new.window(chart, 0, chart.n, theta=np.arange(10))
    for name in ('logp', 'action', 'release'):
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='requires real 56M checkpoint')
@pytest.mark.parametrize('theta', ['off', 'on'])
def test_zero_readers_equal_real_56m(theta):
    data = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    config = R2Config(**data['model_config'])
    old = R2Model(config).eval()
    old.load_state_dict(data['model'])
    new = R2Model(replace(config, anchor='on', theta=theta)).eval()
    record = load_warm_start(CHECKPOINT, new)
    assert record['exposures'] == 56_000_574
    chart = source(19)
    with torch.no_grad():
        a = old.window(chart, 0, chart.n)
        b = new.window(chart, 0, chart.n, theta=np.linspace(-2, 2, 10))
    for name in ('logp', 'action', 'release'):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_nonzero_readers_keep_mirror_equivariance():
    chart = source(21)
    model = R2Model(R2Config(**SMALL, memory='none', anchor='on', theta='on')).double().eval()
    with torch.no_grad():
        model.anchor_reader.weight.normal_(0, .1)
        model.theta_reader.weight.normal_(0, .1)
        vector = np.arange(10) / 7
        reverse = vector.copy()
        reverse[-1] *= -1
        a = model.window(chart, 0, chart.n, theta=vector)
        b = model.window(mirrored(chart), 0, chart.n, theta=reverse)
    torch.testing.assert_close(a.logp, b.logp[:, MIRROR_ACTION.copy()], rtol=0, atol=1e-12)
    torch.testing.assert_close(a.total, b.total, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(theta_features(vector)[::-1], theta_features(reverse))


@pytest.mark.parametrize('history_start', [0, 13])
def test_truncated_history_matches_cached_temporal(history_start):
    chart = source(31)
    model = R2Model(R2Config(**SMALL, memory='none', anchor='on')).eval()
    with torch.no_grad():
        model.temporal.boundary[1].fill_(.4)
        cache = model.temporal.empty_cache(truncated_start=history_start > 0)
        raw = model._t(history_tokens(chart, chart.K))
        before = []
        for k in range(history_start, chart.K + 1):
            before.append(model.temporal.read(cache))
            if k < chart.K:
                cache = model.temporal.append(cache, raw[k])
        ks = np.arange(history_start, chart.K + 1)
        expected = model.hands(torch.stack(before), model.query_features(chart, ks), None, None,
                               model.row_condition(chart, (), ks))
        actual = model.window_hands(chart, ks, history_start=history_start)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_theta_fit_train_only_noise_and_deterministic_donor(tmp_path):
    raw = table_fixture()
    first = ThetaTable.fit(raw)
    changed = raw.copy()
    changed.loc[changed.role == 'fit_dev', [f'raw_{c}' for c in COORDINATES] + WHOLE] = 1e8
    second = ThetaTable.fit(changed)
    assert first.metadata == second.metadata
    train = raw.role == 'fit_train'
    expected = raw.loc[train, [f'half_{c}' for c in COORDINATES]].to_numpy() / first.sd
    np.testing.assert_allclose(first.noise_sd, expected.std(0) / np.sqrt(2))
    assert first.mean[-1] == .5
    path = tmp_path / 'theta.parquet'
    first.save(path)
    loaded = ThetaTable.load(path)
    chart = source(9)
    kwargs = dict(head_ms=chart.head_ms, song_ms=chart.song_ms, grid=chart.grid, seed=954,
                  source_sha256=raw.sha256.iloc[-1])
    a, info = first.sample(**kwargs)
    b, info2 = loaded.sample(**kwargs)
    np.testing.assert_array_equal(a, b)
    assert info == info2
    assert info['donor_sha256'] in raw.loc[train, 'sha256'].tolist()
    assert resolve_theta('unknown')[0] is None
    oracle, _ = resolve_theta('oracle', source_sha256='synthetic', table=loaded)
    np.testing.assert_array_equal(oracle, loaded.vector('synthetic'))


def test_theta_measures_and_skeleton_use_no_decisions():
    chart = source(75)
    raw, diff = chart_coordinates(chart.head_ms, chart.song_ms, chart.grid, chart.actions, chart.gap)
    reverse = mirrored(chart)
    other, _ = chart_coordinates(reverse.head_ms, reverse.song_ms, reverse.grid, reverse.actions, reverse.gap)
    np.testing.assert_allclose(raw[:-1], other[:-1], rtol=0, atol=1e-14)
    assert raw[-1] + other[-1] == pytest.approx(1)
    assert raw[5] == pytest.approx(chart.derived().ln_head.sum() / chart.derived().attack.sum())
    assert diff.shape == (10,)
    skel = skeleton_features(chart.head_ms, chart.song_ms, chart.grid)
    assert skel.shape == (48,) and np.isfinite(skel).all()


def test_generation_records_theta_used():
    from ensomi_model.r2.generate import generate
    from ensomi_model.r2.request_set import RequestSet
    chart = source(9)
    model = R2Model(R2Config(**SMALL, memory='none', anchor='on', theta='on')).eval()
    vector = np.linspace(-1, 1, 10)
    _, _, record = generate(model, chart.head_ms, chart.song_ms, chart.grid, RequestSet(chart.song_ms),
                            theta=vector, stop=4)
    assert record['theta'] == dict(mode='vector', value=vector.tolist())


def test_cached_theta_matches_x3_and_skeleton_columns():
    theta_path = Path('artifacts/r2-theta/theta-v1.parquet')
    x3_path = Path('artifacts/r2-collapse-20261007/phase0-opus/x3/theta.parquet')
    if not theta_path.exists() or not x3_path.exists():
        pytest.skip('requires theta and X3 data tables')
    table = ThetaTable.load(theta_path)
    x3 = pd.read_parquet(x3_path).set_index('sha256')
    ours = table.table.set_index('sha256').loc[x3.index]
    for name in ('nh', 'c3', 'c4', 'jack', 'held', 'pent', 'loop'):
        np.testing.assert_allclose(ours[f'raw_{name}'], x3[name], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.maximum(ours.raw_hand, 1 - ours.raw_hand), x3.hbal, rtol=0, atol=1e-12)
    known = x3.lnlen_b.notna()
    np.testing.assert_allclose(ours.raw_ln_length[known], x3.lnlen_b[known] / np.log(2), rtol=0, atol=1e-12)
    from ensomi_model.r2.data import Corpus
    corpus = Corpus('artifacts/r2-cache/v1', 'fit_dev', star_conditions=False)
    for sha in table.table.loc[table.table.role == 'fit_dev', 'sha256'].iloc[:3]:
        chart = corpus.chart(sha)
        actual = skeleton_features(chart.head_ms, chart.song_ms, chart.grid)
        np.testing.assert_allclose(actual, ours.loc[sha, WHOLE].to_numpy(float), rtol=0, atol=1e-12)


def test_training_dropout_draws_and_resume(monkeypatch, tmp_path):
    table = ThetaTable.fit(table_fixture())
    monkeypatch.setattr(ThetaTable, 'load', lambda path: table)
    trainer = synthetic_trainer(monkeypatch, tmp_path, anchor='on', theta='on', history_dropout=.25)
    from ensomi_model.r2.data import Draw
    trainer.corpus.draw = lambda rng, window: Draw('synthetic', 150, 160, (), {})
    draws = [d for _ in range(500) for _, d in trainer.next_batch()]
    short = [d for d in draws if d.history_start]
    assert .20 < len(short) / len(draws) < .30
    assert all(64 <= d.start - d.history_start <= 128 for d in short)
    assert .25 < sum(d.theta is None for d in draws) / len(draws) < .35
    path = tmp_path / 'resume.pt'
    torch.save(trainer.payload(), path)
    resumed = synthetic_trainer(monkeypatch, tmp_path, anchor='on', theta='on', history_dropout=.25)
    resumed.corpus.draw = trainer.corpus.draw
    resumed.load(path)
    for (_, a), (_, b) in zip(trainer.next_batch(), resumed.next_batch()):
        assert a.history_start == b.history_start
        np.testing.assert_array_equal(a.theta, b.theta)


@pytest.mark.parametrize('arm', [1, 2, 3])
def test_bakeoff_configs(arm):
    cfg = TrainConfig.load(f'src/ensomi_model/r2/configs/ce_v2_bo_b{arm}.json')
    check_config(cfg)
    assert cfg.ln_level == cfg.ln_length == 'off'
    assert cfg.anchor == ('on' if arm >= 2 else 'off')
    assert cfg.theta == ('on' if arm == 3 else 'off')
    assert cfg.history_dropout == (.25 if arm >= 2 else 0)
    assert not cfg.freerun and cfg.full_eval_every > 1000
    assert cfg.threads == 4 and cfg.warm_start == str(CHECKPOINT)
