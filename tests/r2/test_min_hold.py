"""Minimum-hold decoding: strict support, fallback, composition, and disabled identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from ensomi_model.r2 import sampling
from ensomi_model.r2.common import ACTIONS
from ensomi_model.r2.data import Corpus
from ensomi_model.r2.features import Chart
from ensomi_model.r2.generate import defects
from ensomi_model.r2.properties import prefix_objects
from ensomi_model.r2.state import replay_decisions
from ensomi_model.r2.strain import action_head_masks
from .helpers import chart_from_objects, hold, tap, tiny_model, two_segment_grid


@pytest.fixture(scope='module')
def original_sampler():
    assert hashlib.sha256(ORIGINAL_SAMPLING_SOURCE.encode()).hexdigest() == ORIGINAL_SAMPLING_SHA256
    module = types.ModuleType('ensomi_model.r2._sampling_before_min_hold')
    exec(compile(ORIGINAL_SAMPLING_SOURCE, '<sampling-before-min-hold>', 'exec'), module.__dict__)
    return module.continue_chart


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('prefix', [0, 3])
@pytest.mark.parametrize('biased', [False, True])
def test_none_preserves_original_bytes(original_sampler, dtype, prefix, biased):
    chart = chart_from_objects([hold(100, 700, 0), tap(200, 1), hold(300, 600, 2), tap(400, 3),
                                tap(500, 1), tap(800, 2), tap(1000, 0), tap(1200, 3)], 1500)[0]
    model = tiny_model(dtype=dtype, seed=751)
    bias = (lambda *args: np.arange(15)) if biased else None
    for seed in (17, 954):
        kwargs = dict(prefix_actions=chart.actions[:prefix], prefix_gap=chart.gap[:prefix],
                      seed=seed, head_mask_bias=bias)
        expected = original_sampler(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs)
        stats = {}
        for extra in ({}, dict(min_hold_ms=None, min_hold_stats=stats)):
            actual = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs, **extra)
            for before, after in zip(expected, actual):
                assert before.dtype == after.dtype
                assert before.tobytes() == after.tobytes()
        assert stats == dict(decisions=chart.K + 1 - prefix, fallbacks=0)


@pytest.mark.parametrize('eos', [False, True])
def test_empty_support_restores_whole_decision(original_sampler, eos):
    heads = np.array([100.0] if eos else [100.0, 160.0])
    song = 160.0 if eos else 500.0
    grid = two_segment_grid()
    prefix = np.full((1, 4), 2, dtype=np.int64)
    gap = np.full((1, 4), np.nan)
    model = tiny_model(dtype=torch.float32, seed=81)
    kwargs = dict(prefix_actions=prefix, prefix_gap=gap, seed=17, stop=2)
    expected = original_sampler(model, heads, song, grid, **kwargs)
    stats = {}
    actual = sampling.continue_chart(model, heads, song, grid, min_hold_ms=60, min_hold_stats=stats, **kwargs)
    assert stats == dict(decisions=1, fallbacks=1)
    for before, after in zip(expected, actual):
        assert before.tobytes() == after.tobytes()
    replay_decisions(heads, song, *actual)
    assert defects(prefix_objects(heads, *actual), heads)['holds_le60'] > 0


@pytest.mark.parametrize('next_head', [160.0, 161.0, 170.0])
def test_support_precedes_bias_and_keeps_row_pointer_consistent(monkeypatch, next_head):
    heads = np.array([100.0, next_head, 300.0])
    prefix = np.array([[2, 0, 0, 0]])
    gap = np.full((1, 4), np.nan)
    chart = Chart(heads, 500.0, two_segment_grid(), prefix, gap)
    model = tiny_model(dtype=torch.float64, seed=87)
    bias = np.arange(15, dtype=float) * 10
    row_logits = []
    pointer_logits = []
    gumbel = sampling._gumbel_argmax

    def hook(k, committed, held, logp):
        assert k == 1
        torch.testing.assert_close(logp.logsumexp(0), logp.new_zeros(()), atol=1e-12, rtol=0)
        allowed = np.isfinite(logp.numpy())
        assert allowed[ACTIONS[:, 0] == 0].any()
        assert allowed[ACTIONS[:, 0] == 1].any() == (next_head - 100 > 60)
        surviving = committed.candidates(k).times - 100 > 60
        for code in (2, 3, 4):
            assert allowed[ACTIONS[:, 0] == code].any() == surviving.any()
        ids = action_head_masks(held)
        row_logits.append(logp + torch.as_tensor(np.r_[0.0, bias][ids]))
        return bias

    def capture(logits, generator):
        if len(logits) == 625:
            torch.testing.assert_close(logits, row_logits[-1], rtol=0, atol=0)
        else:
            pointer_logits.append(logits)
            keep = chart.candidates(1).times - 100 > 60
            np.testing.assert_array_equal(torch.isfinite(logits).numpy(), keep)
            torch.testing.assert_close(logits.logsumexp(0), logits.new_zeros(()), atol=1e-12, rtol=0)
        return gumbel(logits, generator)

    monkeypatch.setattr(sampling, '_gumbel_argmax', capture)
    stats = {}
    acts, gaps = sampling.continue_chart(model, heads, chart.song_ms, chart.grid, prefix, gap, stop=2,
                                        seed=22, min_hold_ms=60, min_hold_stats=stats, head_mask_bias=hook)
    assert stats == dict(decisions=1, fallbacks=0)
    replay_decisions(heads, chart.song_ms, acts, gaps)
    if acts[1, 0] >= 2:
        assert pointer_logits
        assert gaps[1, 0] - 100 > 60
    if next_head == 170:
        assert pointer_logits  # the large bias forces a head in the held lane


@pytest.mark.parametrize('sha', [
    '18960b87ba606487cd77c89b4e18be580410557bbe9148f8b6c1d908624dd96e',
    '266d68cb93bfd3216c33bce09c0cb725b7a01ac050ca417bdd07cbc1b6f4056c',
    '3a1295925f50c3c424216a55dac42b6212585e387e129b03b65af24f0256dad0',
])
def test_fit_dev_full_runs_have_no_short_holds(sha):
    root = Path(__file__).resolve().parents[2] / 'artifacts/r2-cache/v1'
    if not (root / 'index.parquet').exists():
        pytest.skip('Full fit_dev runs require the R2 cache')
    corpus = Corpus(root, 'fit_dev', star_conditions=False)
    chart = corpus.chart(sha)
    model = tiny_model(dtype=torch.float32, seed=871)
    stats = {}
    acts, gap = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, seed=954,
                                       min_hold_ms=60, min_hold_stats=stats)
    replay_decisions(chart.head_ms, chart.song_ms, acts, gap)
    d = defects(prefix_objects(chart.head_ms, acts, gap), chart.head_ms)
    assert len(acts) == chart.K + 1
    assert stats == dict(decisions=chart.K + 1, fallbacks=0)
    assert d['holds'] > 0
    assert d['holds_le60'] == 0


def test_evaluator_plumbs_only_natural_panels():
    from ensomi_model.r2.evaluate import Evaluator
    from ensomi_model.r2.request_set import RequestSet

    chart = chart_from_objects([hold(100, 700, 0), tap(200, 1), tap(500, 2), tap(800, 3)], 1000)[0]
    ev = object.__new__(Evaluator)
    ev._cpu = tiny_model(dtype=torch.float32, seed=715)
    ev.dev = types.SimpleNamespace(chart=lambda sha: chart)
    ev.min_hold_ms, ev.baseline, ev.code, ev.write_dir = 60, None, None, None
    for panel in ('natural_bos', 'prefix_natural'):
        _, acts, gap, record = ev.run('test', RequestSet(chart.song_ms), 1, 954, panel=panel)
        assert record['min_hold'] == dict(ms=60, decisions=chart.K, fallbacks=0)
        replay_decisions(chart.head_ms, chart.song_ms, acts, gap)
    _, _, _, record = ev.run('test', RequestSet(chart.song_ms), 1, 954, panel='g3_c')
    assert 'min_hold' not in record


# Frozen independently of the implementation under test, including the head-bias hook.
ORIGINAL_SAMPLING_SHA256 = 'de12a4dfd5f8dcf1e85a0e60032613a4935d3248f8a5c1427f18b2334bcf31e1'
ORIGINAL_SAMPLING_SOURCE = '''"""Free-running generation: sample a whole continuation decision by decision.

Each decision: mask the support, sample the joint action (CPU float64 Gumbel
maximum), draw a fair orientation, then the gap releases one lane at a time in
that orientation's order from the directed pointer, and commit. The EOS
decision closes every hold inside (t_last, T). One ``torch.Generator`` per
continuation, seeded from an integer that is logged by the caller. The learned
history is extended with the TCN's online cache (fixed weights only).

``track`` is the effective track (``request_set.effective_track``); the model applies
rule L per decision. Requests, their validation and the generation record live in
``generate.py``. The ``baseline`` slot (the formulation's baseline style rho) admits
only ``None``: R2 carries identity through committed history alone.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ACTIONS, ContractError, GridArrays
from .features import (Chart, gap_release_lanes, history_tokens, lane_query, make_factor, query_features,
                       row_placements)
from .state import action_support_mask, replay_decisions
from .strain import action_head_masks


def _gumbel_argmax(logits: torch.Tensor, generator) -> int:
    logits = logits.detach().cpu().to(torch.float64)
    u = torch.rand(len(logits), generator=generator, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return int((logits - (-u.log()).log()).argmax())


BASELINE_NONE = 'none (not implemented; identity carried by committed history only)'


def require_no_baseline(baseline):
    if baseline is not None:
        raise ContractError('The baseline style rho is not implemented in R2; the baseline slot admits only None')


def _scope_closed(chart: Chart, k: int, a: float, b: float) -> bool:
    """After decision k: no lane still holds an LN headed in [a, b)."""
    starts = chart.derived().start[k + 1]
    return not any(np.isfinite(x) and a <= x < b for x in starts)


@torch.no_grad()
def continue_chart(model, head_ms, song_ms: float, grid: GridArrays, prefix_actions=None, prefix_gap=None,
                   track=(), seed: int = 954, stop: int | None = None, *, baseline=None, close_scope=None,
                   head_mask_bias=None):
    """Return (actions [n,4], gap [n,4]) for decisions 0..n-1, n = stop (default through EOS).

    ``close_scope=(a, b)`` ends generation early, at the first decision at or after the exit
    decision of [a, b) after which no LN headed in [a, b) is still held (realised-response runs).

    ``head_mask_bias(k, chart, held, logp)`` optionally returns 15 finite additive
    biases, ordered by masks 1..15 (bit l denotes lane l). Inputs are read-only:
    chart contains every committed decision < k, including the source prefix;
    held is its replayed held-before state; logp is the unmodified normalized
    [625] action log probability tensor. The next call exposes the last sampled
    action and gaps; a call also occurs at EOS. Biases apply only to the action
    Gumbel maximum, with zero for the empty mask/EOS. The release pointer is
    unchanged. None and all-zero biases preserve the original sampling bits.
    A caller collecting a trace must consume returned arrays after early stop.
    """
    require_no_baseline(baseline)
    was_training = model.training
    model.eval()
    K = len(head_ms)
    stop = K + 1 if stop is None else stop
    s = 0 if prefix_actions is None else len(prefix_actions)
    if s > K:
        raise ContractError('Prefix longer than the head skeleton')
    actions = np.zeros((K + 1, 4), dtype=np.int64)
    gap = np.full((K + 1, 4), np.nan)
    if s:
        actions[:s], gap[:s] = prefix_actions, prefix_gap
        replay_decisions(head_ms, song_ms, actions[:s], gap[:s])  # raises if the prefix is not legal
    generator = torch.Generator().manual_seed(int(seed))
    temporal = model.temporal
    cache = temporal.empty_cache()
    marks = []
    stride = model.config.stride
    if s:
        chart = Chart(head_ms, song_ms, grid, actions[:s], gap[:s])
        toks = model._t(history_tokens(chart, min(s, K)))
        for i in range(len(toks)):
            cache = temporal.append(cache, toks[i])
            if i % stride == 0:
                marks.append(temporal.read(cache))
    base = Chart(head_ms, song_ms, grid, actions[:0], gap[:0])
    exit_k = None if close_scope is None else int(np.searchsorted(head_ms, close_scope[1], side='left'))
    n = stop
    for k in range(s, stop):
        chart = base.with_decisions(actions[:k], gap[:k])
        held = chart.derived().held[k]
        before = temporal.read(cache).to(model.dtype)[None]
        mk = torch.stack(marks) if marks else None
        vis = torch.ones(1, len(marks), dtype=torch.bool, device=model.device) if marks else None
        qf = model._t(query_features(chart, [k]))
        z = model.hands(before, qf, mk, vis, model.row_condition(chart, track, [k]))
        mask = action_support_mask(held, k == K)
        logp = model.action_log_probs(z, mask[None])[0]
        if head_mask_bias is not None:
            bias = torch.as_tensor(head_mask_bias(k, chart, held, logp), dtype=torch.float64, device='cpu')
            if bias.shape != (15,) or not torch.isfinite(bias).all():
                raise ContractError('Head-mask bias must contain 15 finite values')
            if torch.any(bias != 0):
                ids = torch.as_tensor(action_head_masks(held, eos=k == K))
                action_bias = torch.cat((bias.new_zeros(1), bias))[ids]
                logp = logp.detach().cpu().to(torch.float64) + action_bias
        codes = ACTIONS[_gumbel_argmax(logp, generator)]
        orientation = int(torch.randint(2, (), generator=generator))
        lanes = gap_release_lanes(held, codes, orientation)
        rel = np.full(4, np.nan)
        if lanes:
            placed = row_placements(chart, k, held, codes)
            lq = lane_query(chart, [k]).astype(np.float32)[0]
            for lane in lanes:
                f = make_factor(chart, k, 0, codes, lane, orientation, dict(placed), lq)
                ctx = model.factor_contexts(z, [f])
                scores, _, _ = model.all_pair_scores(ctx, [f], chart, track)
                u = float(f.times[_gumbel_argmax(scores, generator)])
                placed[lane] = (u, False)
                rel[lane] = u
        actions[k], gap[k] = codes, rel
        base = chart
        if exit_k is not None and k >= exit_k:
            done = base.with_decisions(actions[:k + 1], gap[:k + 1])
            if _scope_closed(done, k, *close_scope):
                n = k + 1
                break
        if k < K:
            nxt = base.with_decisions(actions[:k + 1], gap[:k + 1])
            tok = model._t(history_tokens(nxt, k + 1)[k])
            cache = temporal.append(cache, tok)
            if k % stride == 0:
                marks.append(temporal.read(cache))
    if was_training:
        model.train()
    return actions[:n], gap[:n]
'''


def test_allocation_n_uses_current_masked_expectation_and_realized_budget():
    from ensomi_model.r2.c0 import AllocationBias, tilted_moments
    from ensomi_model.r2.state import action_support_mask
    from ensomi_model.r2.strain import head_mask_log_probs

    chart = chart_from_objects([tap(t, i % 4) for i, t in enumerate(range(100, 1000, 100))], 1200)[0]
    bias = AllocationBias(chart.head_ms, chart.song_ms, 100, 900, rule='N', target=1.8,
                          kappa=.25, rho0=1.4, stats={})
    for k in (0, 1):
        history = chart.with_decisions(chart.actions[:k], chart.gap[:k])
        held = history.derived().held[k]
        logits = torch.arange(625, dtype=torch.float64) / 200
        mask = torch.as_tensor(action_support_mask(held, False))
        # Remove one entire head-mask class to exercise zero-probability support.
        mask &= torch.as_tensor(action_head_masks(held) != 15)
        logp = logits.masked_fill(~mask, -torch.inf).log_softmax(0)
        bias(k, history, held, logp)
        row = bias.rows[-1]
        costs = np.array(row['mask_costs'])
        lm = head_mask_log_probs(logp, held).numpy()
        e = tilted_moments(lm, costs, bias.scale, 0)[0]
        prior_e = sum(r['natural_expected_charge'] for r in bias.rows)
        rho2 = (prior_e + 8 * bias.scale * 1.4 ** 2) / (bias.trace.reference_rows(0, k + 1) + 8 * bias.scale)
        forecast = e + rho2 * bias.trace.reference_rows(k + 1, bias.stop)
        budget = 1.8 ** 2 * bias.reference - bias.trace.workload_rows(0, k)
        assert row['natural_expected_charge'] == pytest.approx(e)
        assert row['rho2'] == pytest.approx(rho2)
        assert row['forecast_remaining_workload'] == pytest.approx(forecast)
        assert row['desired_charge'] == pytest.approx(e * budget / forecast)
        assert row['natural_mask_log_probs'][14] is None
        assert row['kl'] <= .25 + 1e-12
        assert abs(row['eta']) <= 8
    assert bias.rows[1]['committed_workload'] == pytest.approx(bias.trace.charges[0])


def test_allocation_nb_exact_zero_updates_natural_estimate():
    from ensomi_model.r2.c0 import AllocationBias
    from ensomi_model.r2.state import action_support_mask

    chart = chart_from_objects([tap(t, i % 4) for i, t in enumerate(range(100, 1000, 100))], 1200)[0]
    reference = AllocationBias(chart.head_ms, chart.song_ms, 100, 900, rule='reference', target=None,
                               kappa=None, rho0=1.4, stats={})
    first = chart.with_decisions(chart.actions[:0], chart.gap[:0])
    held = first.derived().held[0]
    logp = torch.zeros(625, dtype=torch.float64).masked_fill(
        ~torch.as_tensor(action_support_mask(held, False)), -torch.inf).log_softmax(0)
    reference(0, first, held, logp)
    projected = reference.rows[0]['projected_r']
    neutral = AllocationBias(chart.head_ms, chart.song_ms, 100, 900, rule='NB',
                             target=projected * np.exp(.019), kappa=1, rho0=1.4, stats={})
    np.testing.assert_array_equal(neutral(0, first, held, logp), np.zeros(15))
    assert neutral.rows[0]['eta'] == 0.0
    assert neutral.rows[0]['kl'] == 0.0
    first_sum = neutral.natural_sum
    second = chart.with_decisions(chart.actions[:1], chart.gap[:1])
    neutral(1, second, second.derived().held[1], logp)
    assert neutral.natural_sum == pytest.approx(first_sum + neutral.rows[1]['natural_expected_charge'])
    outside = AllocationBias(chart.head_ms, chart.song_ms, 100, 900, rule='NB',
                             target=projected * np.exp(.021), kappa=1, rho0=1.4, stats={})
    outside(0, first, held, logp)
    assert not outside.rows[0]['neutral_band']
    assert outside.rows[0]['eta'] != 0.0


def test_allocation_r_matches_existing_controller():
    from ensomi_model.r2.c0 import AllocationBias, MeasurementBias
    from ensomi_model.r2.state import action_support_mask

    chart = chart_from_objects([tap(t, i % 4) for i, t in enumerate(range(100, 1000, 100))], 1200)[0]
    kwargs = dict(target=1.8, kappa=.25)
    old = MeasurementBias(chart.head_ms, chart.song_ms, 100, 900, **kwargs)
    new = AllocationBias(chart.head_ms, chart.song_ms, 100, 900, rule='R', rho0=1.4, stats={}, **kwargs)
    for k in (0, 1):
        history = chart.with_decisions(chart.actions[:k], chart.gap[:k])
        held = history.derived().held[k]
        logp = torch.arange(625, dtype=torch.float64).masked_fill(
            ~torch.as_tensor(action_support_mask(held, False)), -torch.inf).log_softmax(0)
        np.testing.assert_array_equal(new(k, history, held, logp), old(k, history, held, logp))
        for key in ('desired_charge', 'eta', 'kl', 'expected_charge'):
            assert new.rows[-1][key] == old.rows[-1][key]


def test_allocation_r_sampling_matches_c0_with_minimum_hold_mask():
    from ensomi_model.r2.c0 import AllocationBias, MeasurementBias

    chart = chart_from_objects([hold(100, 700, 0)] + [tap(t, 1 + i % 3) for i, t in
                                enumerate((140, 160, 200, 240, 300, 400, 600, 800, 1000))], 1200)[0]
    model = tiny_model(dtype=torch.float32, seed=181)
    original = MeasurementBias(chart.head_ms, chart.song_ms, 140, 1000, target=1.5, kappa=.25)
    stats = {}
    allocation = AllocationBias(chart.head_ms, chart.song_ms, 140, 1000, rule='R', target=1.5,
                                kappa=.25, rho0=1.4, stats=stats)
    kwargs = dict(prefix_actions=chart.actions[:1], prefix_gap=chart.gap[:1], seed=515,
                  stop=int(np.searchsorted(chart.head_ms, 1000)), min_hold_ms=60)
    expected = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid,
                                       head_mask_bias=original, **kwargs)
    actual = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid,
                                     head_mask_bias=allocation, min_hold_stats=stats, **kwargs)
    for a, b in zip(expected, actual):
        assert a.tobytes() == b.tobytes()
    for a, b in zip(original.rows, allocation.rows):
        assert a['eta'] == b['eta']
        assert a['natural_mask_log_probs'] == b['natural_mask_log_probs']
    assert allocation.rows[0]['natural_mask_log_probs'][14] is None
    assert stats['fallbacks'] == 0
