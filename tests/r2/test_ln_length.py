"""Paired chart-level LN inputs: source timing, strict upgrades, and shared dropout."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError, GridArrays
from ensomi_model.r2.data import Corpus
from ensomi_model.r2.features import ln_length_features, query_features
from ensomi_model.r2.ln_level import (EmpiricalLNPrior, JOINT_PRIOR_VERSION, LENGTH_BEAT_FLOOR,
                                     build_prior, resolve_ln_controls, whole_ln_length)
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.sampling import continue_chart
from ensomi_model.r2.train_ce import TrainConfig, check_config, load_warm_start, model_config

from .helpers import chart_from_objects, hold, tap
from .test_ln_level_model import CONFIGS, SMALL, chart, save_source, synthetic_trainer
from .test_ln_level_prior import prior_records


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def length_chart(n=12):
    objects = [hold(100 + 700 * i, 100 + 700 * i + (150, 350, 600)[i % 3], i % 4) for i in range(n)]
    objects += [tap(300, 3), tap(600, 2)]
    return chart_from_objects(objects, 700. * n + 500)[0]


def joint_records():
    rows = prior_records()
    for i, row in enumerate(rows):
        row.update(n_objects=30, n_ln=9 if i % 3 == 0 else 10 + i % 9,
                   ln_length=None if i % 3 == 0 else -2. + i / 10)
    return rows


@pytest.fixture
def joint_prior():
    return EmpiricalLNPrior.fit(joint_records(), with_length=True)


def frozen_share_modules():
    root = Path('artifacts/r2-lnlevel2-20261007/baseline')
    if not root.exists():
        pytest.skip('Frozen share-only source is unavailable')
    name = 'ensomi_model._ln_length_baseline'
    package = types.ModuleType(name)
    package.__path__ = [str(root.resolve()), str(Path('src/ensomi_model/r2').resolve())]
    sys.modules[name] = package
    return [importlib.import_module(name + '.' + module) for module in ('features', 'model', 'ln_level', 'sampling')]


@pytest.mark.parametrize('share', ['off', 'on'])
@pytest.mark.parametrize('seed', [171, 954])
def test_length_off_preserves_share_code_bytes(share, seed):
    features, models, priors, sampling = frozen_share_modules()
    source = chart(seed, K=9)
    ks = np.arange(source.K + 1)
    expected = features.query_features(source, ks, ln_level=share, level=.3)
    actual = query_features(source, ks, ln_level=share, level=.3, ln_length='off', length=123)
    assert expected.tobytes() == actual.tobytes()
    torch.manual_seed(seed)
    old = models.R2Model(models.R2Config(**SMALL, ln_level=share)).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    new = R2Model(R2Config(**SMALL, ln_level=share)).eval()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    if share == 'on':
        with torch.no_grad():
            old.ln_level_reader.weight.fill_(.13)
            new.ln_level_reader.weight.copy_(old.ln_level_reader.weight)
    with torch.no_grad():
        before = old.window(source, 0, source.K + 1, ln_level=.3)
        after = new.window(source, 0, source.K + 1, ln_level=.3, ln_length=123)
    for field in ('logp', 'action', 'release'):
        assert torch.equal(getattr(before, field), getattr(after, field)), field
    previous = priors.EmpiricalLNPrior.fit(prior_records())
    current = EmpiricalLNPrior.fit(prior_records())
    assert previous.data == current.data
    for run_seed in (17, 954, 955):
        kwargs = dict(star=3.2, head_ms=source.head_ms, song_ms=source.song_ms, seed=run_seed)
        assert previous.sample(**kwargs) == current.sample(**kwargs)
    kwargs = dict(ln_level=.3, seed=seed, stop=5)
    before = sampling.continue_chart(old, source.head_ms, source.song_ms, source.grid, **kwargs)
    after = continue_chart(new, source.head_ms, source.song_ms, source.grid, ln_length=123, **kwargs)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(before, after))


@pytest.mark.parametrize('length', [None, -3., 0., 2.5])
def test_length_channels(length):
    source = length_chart()
    ks = [0, 3, source.K]
    old = query_features(source, ks, ln_level='on', level=.5)
    value = query_features(source, ks, ln_level='on', level=.5, ln_length='on', length=length)
    assert value.shape == (3, 2, 155)
    assert value[..., :153].tobytes() == old.tobytes()
    expected = [0, 0] if length is None else [1, length]
    np.testing.assert_array_equal(ln_length_features(length), expected)
    np.testing.assert_array_equal(value[..., 153:], np.broadcast_to(expected, (3, 2, 2)))


@pytest.mark.parametrize('value', [True, '2', [], float('nan'), float('inf'), 1e100])
def test_invalid_length_fails(value):
    with pytest.raises(ContractError, match='LN length'):
        ln_length_features(value)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_initialization_identities_and_rng(tmp_path, dtype):
    torch.manual_seed(171)
    base = R2Model(R2Config(**SMALL)).to(dtype).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(171)
    both = R2Model(R2Config(**SMALL, ln_level='on', ln_length='on')).to(dtype).eval()
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in base.state_dict().items():
        assert torch.equal(value, both.state_dict()[name]), name
    receipt = load_warm_start(save_source(tmp_path, base), both)
    assert receipt['missing_keys'] == ['ln_length_reader.weight', 'ln_level_reader.weight']
    source = chart(K=9)
    with torch.no_grad():
        expected = base.window(source, 0, source.K + 1)
        for level, length in [(None, None), (.4, None), (.4, -2), (.4, 3)]:
            out = both.window(source, 0, source.K + 1, ln_level=level, ln_length=length)
            assert torch.equal(expected.logp, out.logp)
            assert torch.equal(expected.release, out.release)
    share = R2Model(R2Config(**SMALL, ln_level='on')).to(dtype).eval()
    with torch.no_grad():
        share.ln_level_reader.weight.normal_(0, .3)
    load_warm_start(save_source(tmp_path, share), both)
    with torch.no_grad():
        both.ln_length_reader.weight.normal_(0, .3)
        expected = share.window(source, 0, source.K + 1, ln_level=.4)
        actual = both.window(source, 0, source.K + 1, ln_level=.4, ln_length=None)
    assert torch.equal(expected.logp, actual.logp)
    assert torch.equal(expected.release, actual.release)


@pytest.mark.parametrize('mutation', ['legacy_missing', 'length_missing', 'unexpected', 'nonzero', 'config', 'shape'])
def test_strict_length_warm_start_rejects(tmp_path, mutation):
    old = R2Model(R2Config(**SMALL, ln_level='on'))
    path = save_source(tmp_path, old)
    data = torch.load(path, weights_only=False)
    target = R2Model(replace(old.config, ln_length='on'))
    if mutation == 'legacy_missing':
        del data['model']['ln_level_reader.weight']
    elif mutation == 'length_missing':
        data['model_config']['ln_length'] = 'on'
    elif mutation == 'unexpected':
        data['model']['foreign'] = torch.zeros(1)
    elif mutation == 'nonzero':
        with torch.no_grad():
            target.ln_length_reader.weight.fill_(1.)
    elif mutation == 'config':
        data['model_config']['levels'] += 1
    else:
        data['model']['exact.0.weight'] = data['model']['exact.0.weight'][:1]
    torch.save(data, path)
    with pytest.raises(ContractError, match='warm_start'):
        load_warm_start(path, target)


def test_source_length_uses_snapped_releases_all_holds_and_grid_segments():
    source = length_chart()
    d = source.derived()
    closes = d.held[:-1] & (source.actions != 0)
    start, end = d.start[:-1][closes], d.release[closes]
    beats = source.grid.beat(end) - source.grid.beat(start)
    assert len(beats) == 12 and closes[-1].any()
    expected = np.median(np.log(np.maximum(beats, 1e-4))) / np.log(2)
    assert whole_ln_length(source) == pytest.approx(expected, abs=1e-14)
    assert whole_ln_length(length_chart(9)) is None
    assert whole_ln_length(length_chart(10)) is not None
    for n in (2, source.K):
        with pytest.raises(ContractError, match='through EOS'):
            whole_ln_length(source.with_decisions(source.actions[:n], source.gap[:n]))
    changed = source.gap.copy()
    eos_lane = int(np.flatnonzero(closes[-1])[0])
    changed[-1, eos_lane] = np.nan
    with pytest.raises(ContractError, match='finite positive'):
        whole_ln_length(source.with_decisions(source.actions, changed))
    changed[-1, eos_lane] = source.song_ms + 1
    with pytest.raises(ContractError, match='outside its decision gap'):
        whole_ln_length(source.with_decisions(source.actions, changed))


def test_length_floor_preserves_invalid_grid_errors():
    source = length_chart(10)
    grid = GridArrays(np.array([0.]), np.array([1e10]), np.array([0.]), np.array([4]))
    source.grid = grid
    assert whole_ln_length(source) == pytest.approx(np.log2(LENGTH_BEAT_FLOOR))
    source.grid = replace(grid, lengths=np.array([-500.]))
    with pytest.raises(ContractError, match='positive grid-beat'):
        whole_ln_length(source)


def test_corpus_caches_whole_source_length():
    corpus = object.__new__(Corpus)
    corpus._source_ln_lengths = {}
    calls = []
    source = length_chart()
    corpus.chart = lambda sha: calls.append(sha) or source
    expected = whole_ln_length(source)
    assert corpus.source_ln_length('source') == expected
    assert corpus.source_ln_length('source') == expected
    assert calls == ['source']


def test_joint_seeded_draw_matches_one_observation_and_ignores_other_roles(joint_prior):
    rows = joint_records()
    dev = dict(rows[0], role='fit_dev', sha256='dev', ln_length=123)
    assert EmpiricalLNPrior.fit([dev, *rows[::-1]], with_length=True).data == joint_prior.data
    assert joint_prior.version == JOINT_PRIOR_VERSION
    for seed in range(20):
        kwargs = dict(star=3.2, head_ms=[100. + seed], song_ms=1000, seed=seed)
        (share, length), record = joint_prior.sample_joint(**kwargs)
        assert ((share, length), record) == joint_prior.sample_joint(**kwargs)
        source = next(row for row in rows if row['sha256'] == record['sampled_source_sha256'])
        assert share == source['n_ln'] / source['n_objects']
        assert length == source['ln_length']
        cell = joint_prior.data['bands']['3']['cells'][record['density_tercile']]
        assert cell['observations'][record['support_index']] == dict(share=share, length=length,
                                                                    source_sha256=source['sha256'])
    data = json.loads(json.dumps(joint_prior.data))
    empty = data['bands']['3']['cells'][0]
    data['count'] -= empty['count']
    data['bands']['3']['count'] -= empty['count']
    empty.update(count=0, observations=[])
    with pytest.raises(ContractError, match='no source observations'):
        EmpiricalLNPrior(data).sample_joint(star=3.2, head_ms=[100], song_ms=1000, seed=1)


def test_joint_prior_rejects_missing_lengths_and_v1():
    rows = joint_records()
    rows[1]['ln_length'] = None
    with pytest.raises(ContractError, match='hold count'):
        EmpiricalLNPrior.fit(rows, with_length=True)
    with pytest.raises(ContractError, match='joint v2'):
        EmpiricalLNPrior.fit(prior_records()).sample_joint(star=3., head_ms=[1], song_ms=1000, seed=1)
    with pytest.raises(ContractError, match='explicit output'):
        build_prior('missing-cache', with_length=True)


def test_pair_resolution_is_explicit(joint_prior):
    assert resolve_ln_controls('unknown', source_ln_level=.8, source_ln_length=5)[:2] == (None, None)
    assert resolve_ln_controls('oracle', source_ln_level=.8, source_ln_length=5)[:2] == (.8, 5.)
    assert resolve_ln_controls('oracle', source_ln_level=.8, source_ln_length=None)[:2] == (.8, None)
    assert resolve_ln_controls(.2)[:2] == (.2, None)
    assert resolve_ln_controls(.2, ln_length=-1)[:2] == (.2, -1.)
    kwargs = dict(ln_prior=joint_prior, star=3., head_ms=[1], song_ms=1000, seed=17)
    share, length, record = resolve_ln_controls('prior', source_ln_level=.999, source_ln_length=100, **kwargs)
    pair, provenance = joint_prior.sample_joint(**{k: v for k, v in kwargs.items() if k != 'ln_prior'})
    assert (share, length) == pair
    assert record['sampled_source_sha256'] == provenance['sampled_source_sha256']
    assert record['length'] == dict(known=length is not None, value=length, units='log2 beats')
    with pytest.raises(ContractError, match='fixed numeric'):
        resolve_ln_controls('prior', ln_length=1., **kwargs)


@pytest.mark.parametrize('prefix', [0, 3])
def test_sampler_holds_both_inputs_through_eos(joint_prior, prefix):
    source = length_chart()
    model = R2Model(R2Config(**SMALL, ln_level='on', ln_length='on')).eval()
    calls, stats = [], {}
    query = model.query_features

    def capture(chart, ks, ln_level=None, ln_length=None):
        calls.append((ln_level, ln_length))
        return query(chart, ks, ln_level, ln_length)

    model.query_features = capture
    continue_chart(model, source.head_ms, source.song_ms, source.grid,
                   source.actions[:prefix], source.gap[:prefix], ln_level='prior', ln_prior=joint_prior,
                   star=3., seed=17, ln_level_stats=stats)
    assert calls == [(stats['value'], stats['length']['value'])] * (source.K + 1 - prefix)
    expected = joint_prior.sample_joint(star=3., head_ms=source.head_ms, song_ms=source.song_ms, seed=17)
    assert calls[0] == expected[0]


def test_length_gradient_and_likelihood_entrypoints():
    model = R2Model(R2Config(**SMALL, ln_level='on', ln_length='on'))
    source = chart(K=7)
    out = model.window(source, 0, source.K + 1, ln_level=.4, ln_length=-1.)
    (-out.total.sum()).backward()
    assert torch.count_nonzero(model.ln_length_reader.weight.grad)
    model.zero_grad(set_to_none=True)
    out = model.window(source, 0, source.K + 1, ln_level=.4, ln_length=None)
    (-out.total.sum()).backward()
    assert not torch.count_nonzero(model.ln_length_reader.weight.grad)
    with torch.no_grad():
        model.ln_length_reader.weight.normal_(0, .5)
        out = model.window(source, 0, source.K + 1, ln_level=.4, ln_length=-1.)
        assert torch.equal(model.sequence_log_prob(source, 0, source.K + 1, ln_level=.4, ln_length=-1.), out.total.sum())
        assert torch.equal(model.decision_log_prob(source, 0, ln_level=.4, ln_length=-1.),
                           model.window(source, 0, 1, ln_level=.4, ln_length=-1.).total[0])


def test_shared_dropout_keeps_old_rng_draws_and_resume(monkeypatch, tmp_path):
    share = synthetic_trainer(monkeypatch, tmp_path, ln_level='on')
    both = synthetic_trainer(monkeypatch, tmp_path, ln_level='on', ln_length='on')
    both.corpus.source_ln_length = lambda sha: -1.5
    for _ in range(8):
        a, b = share.next_batch(), both.next_batch()
        assert [(i, d.sha, d.start, d.ln_level) for i, d in a] == [(i, d.sha, d.start, d.ln_level) for i, d in b]
        assert all(d.ln_length == (-1.5 if d.ln_level is not None else None) for _, d in b)
        assert share.rng.bit_generator.state == both.rng.bit_generator.state
        assert share.ln_level_rng.bit_generator.state == both.ln_level_rng.bit_generator.state
    both.step(both.next_batch())
    path = tmp_path / 'pair-resume.pt'
    torch.save(both.payload(), path)
    resumed = synthetic_trainer(monkeypatch, tmp_path, ln_level='on', ln_length='on')
    resumed.corpus.source_ln_length = lambda sha: -1.5
    resumed.load(path)
    first, second = both.next_batch(), resumed.next_batch()
    assert [(i, d.start, d.ln_level, d.ln_length) for i, d in first] == [
        (i, d.start, d.ln_level, d.ln_length) for i, d in second]
    both.step(first)
    resumed.step(second)
    for name, value in both.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    both.corpus.source_ln_length = lambda sha: None
    draws = [d for _ in range(10) for _, d in both.next_batch()]
    assert any(d.ln_level is not None for d in draws)
    assert all(d.ln_length is None for d in draws)


def test_config_changes_exactly_three_values_and_preserves_null():
    old = json.loads((CONFIGS / 'ce_v2_n_lnlevel_ft.json').read_text())
    new = json.loads((CONFIGS / 'ce_v2_n_lnlevel2_ft.json').read_text())
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    assert changed == {'ln_length', 'warm_start', 'full_eval_every'}
    assert new['warm_start'].endswith('ckpt-0056000574.pt')
    assert new['lr_schedule'] is None
    cfg = TrainConfig.load(CONFIGS / 'ce_v2_n_lnlevel2_ft.json')
    check_config(cfg)
    assert model_config(cfg).ln_length == 'on'
    assert cfg.full_eval_every == 4
    for values in (dict(ln_level='off'), dict(ln_length='bad')):
        with pytest.raises(ContractError, match='ln_length'):
            check_config(replace(cfg, **values))


def test_checkpoint_schedule_has_three_teacher_forced_evaluations(monkeypatch, tmp_path):
    trainer = synthetic_trainer(monkeypatch, tmp_path, ln_level='on', ln_length='on', full_eval_every=4)
    calls = []
    trainer.save = lambda: (tmp_path / trainer.ckpt_name(), {'checkpoint_bytes': 0})
    trainer.eval_and_log = lambda path, size=None, full=True: calls.append((trainer.state['exposures'], full))
    for exposures in (4_000_000, 8_000_000, 12_000_000):
        trainer.state['exposures'] = exposures
        trainer.checkpoint_and_eval()
    assert calls == [(4_000_000, False), (8_000_000, False), (12_000_000, False)]


@pytest.mark.parametrize('every,count,expected', [(4, 3, False), (1, 3, True), (4, 0, False)])
def test_resume_missing_eval_obeys_checkpoint_schedule(monkeypatch, tmp_path, every, count, expected):
    from ensomi_model.r2 import train_ce
    # Scheduling does not depend on system swap counters, which the macOS sandbox blocks.
    monkeypatch.setattr(train_ce, 'ResourceGuard', lambda *args, **kwargs: None)
    trainer = synthetic_trainer(monkeypatch, tmp_path, ln_level='on', ln_length='on',
                                full_eval_every=every, total_exposures=12_000_000)
    trainer.state.update(exposures=12_000_000, checkpoints=count)
    trainer.latest = lambda: tmp_path / trainer.ckpt_name()
    trainer.load = lambda path: None
    trainer.evaluated = lambda: set()
    trainer.receipt = lambda mode: {}
    calls = []
    trainer.eval_and_log = lambda path, size=None, full=True: calls.append(full)
    assert trainer.train(resume=True) == 0
    assert calls == [expected]


def test_generate_records_joint_prior_and_fixed_length(joint_prior):
    from ensomi_model.r2.generate import generate
    from ensomi_model.r2.request_set import RequestSet
    source = length_chart()
    model = R2Model(R2Config(**SMALL, ln_level='on', ln_length='on')).eval()
    kwargs = dict(ln_level='prior', ln_prior=joint_prior, star=3., random_seed=17, stop=2)
    _, _, record = generate(model, source.head_ms, source.song_ms, source.grid, RequestSet(source.song_ms), **kwargs)
    expected, provenance = joint_prior.sample_joint(star=3., head_ms=source.head_ms, song_ms=source.song_ms, seed=17)
    assert (record['ln_level']['value'], record['ln_level']['length']['value']) == expected
    assert record['ln_level']['sampled_source_sha256'] == provenance['sampled_source_sha256']
    _, _, fixed = generate(model, source.head_ms, source.song_ms, source.grid, RequestSet(source.song_ms),
                            ln_level=.4, ln_length=-2., stop=2)
    assert fixed['ln_level']['mode'] == 'fixed'
    assert fixed['ln_level']['length'] == dict(known=True, value=-2., units='log2 beats')


def test_evaluator_fixed_length_and_explicit_prior_path(tmp_path, joint_prior):
    import pandas as pd
    from ensomi_model.r2.evaluate import Evaluator
    source = length_chart()
    path = tmp_path / 'joint-prior.json'
    path.write_text(json.dumps(joint_prior.data))
    ev = object.__new__(Evaluator)
    ev.model = R2Model(R2Config(**SMALL, ln_level='on', ln_length='on')).eval()
    ev.cfg = types.SimpleNamespace(cache='missing-cache', ln_prior=str(path))
    ev.dev = types.SimpleNamespace(table=pd.DataFrame([dict(sha256='chart', star=3.)]))
    ev.ln_mode, ev.ln_length, ev._ln_prior = np.float32(.5), -2., None
    level, record = ev._level(source, 'chart', 17)
    assert level == .5 and record['length']['value'] == -2.
    prior_level, prior_record = ev._level(source, 'chart', 17, mode='prior')
    pair, _ = joint_prior.sample_joint(star=3., head_ms=source.head_ms, song_ms=source.song_ms, seed=17)
    assert (prior_level, prior_record['length']['value']) == pair
    assert ev._ln_prior.sha256 is not None


def test_actual_56m_warm_start_adds_only_zero_ln_readers(tmp_path):
    cfg = TrainConfig.load(CONFIGS / 'ce_v2_n_lnlevel2_ft.json')
    path = Path(cfg.warm_start)
    if not path.exists():
        pytest.skip('The strict 56M warm-load check requires the phase-N checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    target = R2Model(model_config(cfg)).eval()
    receipt = load_warm_start(path, target)
    additions = {'ln_level_reader.weight', 'ln_length_reader.weight'}
    assert receipt['exposures'] == 56_000_574
    assert set(receipt['missing_keys']) == additions
    assert set(target.state_dict()) - set(saved['model']) == additions
    assert receipt['loaded_parameters'] == len(saved['model'])
    for name, value in saved['model'].items():
        assert torch.equal(value, target.state_dict()[name]), name
    for name in additions:
        assert torch.count_nonzero(target.state_dict()[name]) == 0
    (tmp_path / 'warm-load-receipt.json').write_text(json.dumps(receipt, indent=2))


@pytest.mark.parametrize('case,sha', [
    ('zero', '1f8e646ae6b1fe59d88822f7be7aaea90d866fb523936d52498acf6f777abc73'),
    ('few', '2a286f1b38226f8b297a485dc1072af0bfcfafc946d89ac1f888cb70c77a698c'),
    ('snapped', '0de90d74df826eb0bcbc67dca054e96c08b88a5d3a3ba48c1674a060a1ce9c10'),
    ('eos', '3dde5e08d6c1b8946d0c7933b0c916f1a22ae824c9b034ded3204d4d7544a152'),
    ('segments', '0904ea85140704ccf4511e81a5c9907b014d424268d9e76d80ade0b6de455913'),
])
def test_cached_length_matches_lnlength_reference(case, sha, monkeypatch, tmp_path):
    import importlib.util
    from ensomi_model.r2.train_ce import file_sha256

    root = Path(__file__).resolve().parents[2]
    cache = root / 'artifacts/r2-cache/v1'
    reference = root.parent / '.sync/cp/scratch/r2-ln-level/lnlength.py'
    if not (cache / 'index.parquet').exists() or not reference.exists():
        pytest.skip('Cached length comparison requires the R2 cache and lnlength.py reference')
    monkeypatch.syspath_prepend(str(reference.parent))
    spec = importlib.util.spec_from_file_location('r2_lnlength_reference', reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = Corpus(cache, 'fit_train', star_conditions=False)
    source = corpus.chart(sha)
    row = corpus.table.set_index('sha256').loc[sha]
    holds = module.holds_list(source.head_ms, source.actions, source.gap)
    summary = module.summarise(module.hold_table(source.head_ms, source.actions, source.gap, source.grid))
    actual = whole_ln_length(source)
    assert source.n == source.K + 1
    assert len(holds) == summary['holds'] == row.n_ln
    if case == 'zero':
        assert len(holds) == 0
    elif case == 'few':
        assert 0 < len(holds) < module.MINH
    elif case == 'snapped':
        assert row.snap_max > 1 and row.snap_n > 0
    elif case == 'eos':
        assert (holds[:, 4] == source.K).any()
    else:
        start, end = holds[:, 0], holds[:, 1]
        integrated = source.grid.beat(end) - source.grid.beat(start)
        head_bpm_only = (end - start) / source.grid.lengths[source.grid.segment(start)]
        assert not np.allclose(integrated, head_bpm_only)
    expected = summary['med_lb'] / np.log(2) if len(holds) >= module.MINH else None
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, abs=1e-13)
    assert corpus.source_ln_length(sha) == actual
    (tmp_path / 'cached-length-receipt.json').write_text(json.dumps(dict(
        case=case, source_sha256=sha, cached_file_sha256=row.file_sha256,
        reference_sha256=file_sha256(reference), lnlib_sha256=file_sha256(reference.with_name('lnlib.py')),
        holds=len(holds), actual_log2_beats=actual, reference_log2_beats=expected), indent=2))
