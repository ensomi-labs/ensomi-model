"""ras-v1 invariants and sampling compatibility, including real cached charts."""
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from ensomi_model.r2 import sampling
from ensomi_model.r2.cache import load_chart, objects_from_decisions
from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.r2.state import R2State, advance, action_support_mask, decision_at
from ensomi_model.r2.strain import (HEAD_MASKS, RAS_VERSION, REFERENCE_LANES, TAU_G, TAU_I,
                                   WEIGHT_G, WEIGHT_I, FixedEtaBias, StrainState, StrainTrace,
                                   action_head_masks, head_mask_log_probs, heads_from_codes)
from ensomi_model.research.chart.actions import LN_START, TAP

from .helpers import chart_from_objects, hold, tap, tiny_model


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    torch.set_num_threads(1)


def tap_trace(times, masks):
    times = np.asarray(times, dtype=np.float64)
    actions = np.asarray(masks, dtype=np.int64).reshape(-1, 4)
    song_ms = float(times[-1] + 1000) if len(times) else 1000.0
    return StrainTrace(times, song_ms).extend(actions, np.full(actions.shape, np.nan))


def test_screen_export_keeps_prefix_heads_and_closes_holds_at_scope_end():
    from ensomi_model.r2.c0 import close_screen_scope
    from ensomi_model.r2.export import decisions_to_objects

    chart, _ = chart_from_objects([tap(100, 0), hold(500, 800, 1), tap(1000, 2)], 2000)
    partial = chart.with_decisions(chart.actions[:2], chart.gap[:2])
    complete = close_screen_scope(partial, 400, 900)
    objects, _ = decisions_to_objects(complete.head_ms, complete.song_ms, complete.actions, complete.gap)
    assert [(o.start_time_ms, o.end_time_ms, o.lane) for o in objects] == [(100, 100, 0), (500, 900, 1)]
    assert np.array_equal(complete.actions[:2], partial.actions)
    assert np.array_equal(complete.gap[:2], partial.gap, equal_nan=True)
    before = StrainTrace(chart.head_ms, chart.song_ms).extend(partial.actions, partial.gap).scope(400, 900)
    after = StrainTrace(complete.head_ms, complete.song_ms).extend(complete.actions, complete.gap).scope(400, 900)
    assert after == before
    with pytest.raises(ValueError, match='b < song_ms'):
        close_screen_scope(partial, 400, 2000)


def direct_charges(times, masks):
    # Independent scalar recurrence pins the constants and seconds conversion.
    x, g, previous, charges = [0.0] * 4, 0.0, 0.0, []
    for t, q in zip(times, masks):
        dt = (float(t) - previous) / 1000.0
        x = [v * 0.125 ** dt for v in x]
        g *= 0.30 ** dt
        m = sum(q)
        charges.append((-1 / math.log(0.125)) * (sum(v * h for v, h in zip(x, q)) + m / 2)
                       + 0.25 * (-1 / math.log(0.30)) * (g * m + m * m / 2))
        x = [v + h for v, h in zip(x, q)]
        g += m
        previous = float(t)
    return np.asarray(charges)


@pytest.fixture(scope='module')
def real_charts():
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = Path(__file__).resolve().parents[2] / 'artifacts/r2-cache/v1'
    if not (root / 'index.parquet').exists():
        pytest.skip('Real ras-v1 cache checks require artifacts/r2-cache/v1')
    groups = json.loads((root / 'splits.json').read_text())['groups']
    columns = ['sha256', 'group_id', 'role', 'file', 'K', 'n_ln', 'n_row_release']
    records = pq.read_table(root / 'index.parquet', columns=columns, use_threads=False).to_pylist()
    selected = []
    for role in ('fit_train', 'fit_dev'):
        eligible = sorted((r for r in records if r['role'] == role and 32 <= r['K'] <= 3000
                           and r['n_ln'] and r['n_row_release']), key=lambda r: r['sha256'])
        assert len(eligible) >= 3
        for record in eligible[:3]:
            assert groups[record['group_id']] == role
            dec = load_chart(root / 'charts' / record['file'])
            selected.append((record['sha256'], dec))
    return selected


def test_ras_v1_frozen_definition():
    assert RAS_VERSION == 'ras-v1'
    assert (TAU_I, TAU_G, WEIGHT_I, WEIGHT_G, REFERENCE_LANES) == (
        -1 / math.log(0.125), -1 / math.log(0.30), 1.0, 0.25, (0, 2, 1, 3))
    masks = HEAD_MASKS[[0, 2, 8, 14, 7]]
    times = [0, 40, 500, 1000, 8000]
    trace = tap_trace(times, masks)
    np.testing.assert_allclose(trace.charges, direct_charges(times, masks), rtol=2e-15)


def test_adjacent_scopes_recombine_workload_reference_and_weighted_ratio_squared():
    trace = tap_trace([100, 200, 300, 400, 500, 600], HEAD_MASKS[[0, 6, 7, 14, 1, 8]])
    left, right, whole = (trace.scope(a, b) for a, b in [(0, 350), (350, 1000), (0, 1000)])
    assert left.workload + right.workload == pytest.approx(whole.workload)
    assert left.reference + right.reference == pytest.approx(whole.reference)
    recombined = (left.reference * left.ratio ** 2 + right.reference * right.ratio ** 2) / whole.reference
    assert recombined == pytest.approx(whole.ratio ** 2)
    assert whole.intensity == pytest.approx(whole.workload)


def test_request_partition_independence():
    times = np.arange(30) * 70.0
    actions = HEAD_MASKS[np.arange(30) % 15].astype(np.int64)
    gap = np.full(actions.shape, np.nan)
    whole = StrainTrace(times, 3000).extend(actions, gap)
    partitioned = StrainTrace(times, 3000)
    for n in (3, 11, 12, 21, 30):
        partitioned.extend(actions[:n], gap[:n])
        partitioned.scope(0, float(times[n - 1] + 1))
    np.testing.assert_array_equal(partitioned.workload_prefix, whole.workload_prefix)
    np.testing.assert_array_equal(partitioned.reference_prefix, whole.reference_prefix)
    scopes = [(0, 201), (201, 203), (203, 799), (799, 1800), (1800, 3000)]
    for a, b in reversed(scopes):
        assert partitioned.scope(a, b) == whole.scope(a, b)
    assert sum(partitioned.scope(a, b).workload for a, b in scopes) == pytest.approx(whole.scope(0, 3000).workload)


def test_dense_prefix_carry_in_raises_scope_workload():
    times = np.arange(12) * 100.0
    masks = np.tile([1, 0, 0, 0], (12, 1))
    dense = tap_trace(times, masks)
    rested = tap_trace(times[-2:], masks[-2:])
    assert dense.scope(1000, 1200).workload > rested.scope(1000, 1200).workload


def test_exact_half_open_endpoints():
    trace = tap_trace([100, 200, 300], [[1, 0, 0, 0]] * 3)
    assert trace.scope(100, 200).workload == trace.charges[0]
    assert trace.scope(200, 300).workload == pytest.approx(trace.charges[1])
    assert trace.scope(100, np.nextafter(200.0, np.inf)).head_rows == 2
    assert trace.scope(200, 200).workload == 0
    assert trace.scope(201, 300).head_rows == 0


def test_head_one_ms_before_end_pays_full_charge():
    trace = tap_trace([100, 999], [[1, 0, 0, 0]] * 2)
    assert trace.scope(999, 1000).workload == pytest.approx(trace.charges[-1])
    assert trace.scope(999, 1200).workload == trace.scope(999, 1000).workload


def test_lane_mirror_invariance():
    rng = np.random.default_rng(84)
    times = np.cumsum(rng.uniform(2, 500, 100))
    masks = HEAD_MASKS[rng.integers(0, 15, len(times))]
    original, mirrored = tap_trace(times, masks), tap_trace(times, masks[:, ::-1])
    np.testing.assert_allclose(original.charges, mirrored.charges, rtol=3e-16)
    for a, b in ((0, 5000), (1234, 9999), (0, float(times[-1] + 1))):
        assert original.scope(a, b).ratio == pytest.approx(mirrored.scope(a, b).ratio)


def test_adding_head_increases_workload_same_lane_repeat_and_chord_cost_more():
    sparse = tap_trace([100, 200, 300], [[1, 0, 0, 0]] * 3)
    added = tap_trace([100, 200, 300], [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    assert added.scope(0, 1000).workload > sparse.scope(0, 1000).workload
    state, _ = StrainState().append(100, [1, 0, 0, 0])
    _, repeat = state.append(200, [1, 0, 0, 0])
    _, rested_lane = state.append(200, [0, 1, 0, 0])
    _, rested_chart = StrainState().append(200, [1, 0, 0, 0])
    _, chord = state.append(200, [1, 1, 0, 0])
    assert chord > repeat > rested_lane > rested_chart


def test_nonnegative_charges_and_zero_empty_scope():
    trace = tap_trace(np.arange(15) * 100.0, HEAD_MASKS)
    assert np.all(np.asarray(trace.charges) >= 0)
    empty = trace.scope(1, 99)
    assert empty.workload == empty.reference == 0
    assert empty.ratio is empty.intensity is None
    assert StrainState().append(10, [0, 0, 0, 0])[1] == 0
    no_heads = tap_trace([], [])
    assert no_heads.scope(0, 1000).ratio is None


def test_prefix_sums_match_direct_random_scopes_on_real_cached_charts(real_charts):
    rng = np.random.default_rng(871)
    for _, dec in real_charts:
        trace = StrainTrace(dec.head_ms, dec.song_ms).extend(dec.actions, dec.gap_release_ms)
        objects = objects_from_decisions(dec.head_ms, dec.actions, dec.gap_release_ms, dec.song_ms)
        heads_by_time = {}
        for obj in objects:
            heads_by_time.setdefault(obj.start_time_ms, [0] * 4)[obj.lane] = 1
        q = [heads_by_time[float(t)] for t in dec.head_ms]
        direct = direct_charges(dec.head_ms, q)
        reference_q = np.eye(4)[np.asarray(REFERENCE_LANES)[np.arange(dec.K) % 4]]
        direct_reference = direct_charges(dec.head_ms, reference_q)
        for _ in range(40):
            a, b = sorted(rng.uniform(0, dec.song_ms, 2))
            selected = (a <= dec.head_ms) & (dec.head_ms < b)
            measured = trace.scope(a, b)
            assert measured.workload == pytest.approx(direct[selected].sum(), rel=2e-12, abs=1e-10)
            assert measured.reference == pytest.approx(direct_reference[selected].sum(), rel=2e-12, abs=1e-10)


def test_all_fifteen_costs_equal_append_trace_charge():
    rng = np.random.default_rng(287)
    state = StrainState()
    for t in np.cumsum(rng.uniform(2, 900, 50)):
        costs = state.mask_costs(t)
        for mask, cost in zip(HEAD_MASKS, costs):
            _, charge = state.append(t, mask)
            assert cost == pytest.approx(charge, rel=1e-15)
        state, _ = state.append(t, HEAD_MASKS[rng.integers(15)])


def test_held_code_heads_match_exact_replay_on_real_cache(real_charts):
    seen_held_codes = set()
    for _, dec in real_charts:
        state = R2State()
        for k in range(dec.K + 1):
            decision = decision_at(dec.actions, dec.gap_release_ms, k)
            q = heads_from_codes(decision.codes, state.held, eos=k == dec.K)
            seen_held_codes.update(c for c, h in zip(decision.codes, state.held) if h)
            state, rows = advance(state, decision, dec.head_ms, dec.song_ms)
            actual = [sum(row.actions[lane] in (TAP, LN_START) for row in rows) for lane in range(4)]
            np.testing.assert_array_equal(q, actual)
        assert state.finalized
    assert {1, 2, 3, 4} <= seen_held_codes


def test_exact_remaining_budget_uses_committed_scope_and_full_reference():
    times = np.arange(20) * 100.0
    actions = HEAD_MASKS[np.arange(20) % 15].astype(np.int64)
    gap = np.full(actions.shape, np.nan)
    trace = StrainTrace(times, 3000)
    start, stop = trace.row_bounds(500, 1700)
    for n in (0, 3, 8, 19, 20):
        trace.extend(actions[:n], gap[:n])
        expected = 0.5 ** 2 * trace.reference_rows(start, stop) - sum(trace.charges[start:min(stop, n)])
        assert trace.remaining_budget(0.5, 500, 1700) == pytest.approx(expected)
    assert trace.remaining_budget(0.5, 500, 1700) < 0


def test_head_mask_probability_masses():
    generator = torch.Generator().manual_seed(951)
    for bits in range(16):
        held = np.array([bool(bits & (1 << lane)) for lane in range(4)])
        logp = torch.randn(625, generator=generator, dtype=torch.float64)
        logp[~torch.as_tensor(action_support_mask(held, False))] = -torch.inf
        logp = logp.log_softmax(0)
        mass = head_mask_log_probs(logp, held).exp()
        ids = action_head_masks(held)
        expected = torch.stack([logp[ids == m].exp().sum() for m in range(1, 16)])
        torch.testing.assert_close(mass, expected, rtol=1e-14, atol=1e-16)
        assert float(mass.sum()) == pytest.approx(1.0)


@pytest.fixture(scope='module')
def short_chart():
    return chart_from_objects([hold(100, 700, 0), tap(200, 1), hold(300, 600, 2), tap(400, 3),
                               tap(500, 1), tap(800, 2), tap(1000, 0), tap(1200, 3)], 1500)[0]


def original_sampler():
    assert hashlib.sha256(ORIGINAL_SAMPLING_SOURCE.encode()).hexdigest() == (
        '32caf167154480645924a3cd38fca52934a0b24972775a9906785b1225bb8d0a')
    module = types.ModuleType('ensomi_model.r2._sampling_before_strain')
    exec(compile(ORIGINAL_SAMPLING_SOURCE, '<original-r2-sampling>', 'exec'), module.__dict__)
    return module.continue_chart


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('prefix', [0, 3])
def test_no_bias_actions_and_gaps_bit_identical_to_original(short_chart, dtype, prefix):
    model = tiny_model(dtype=dtype, seed=751)
    chart = short_chart
    original = original_sampler()
    for seed in (17, 954):
        kwargs = dict(prefix_actions=chart.actions[:prefix], prefix_gap=chart.gap[:prefix], seed=seed)
        expected = original(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs)
        absent = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, **kwargs)
        zero = FixedEtaBias(chart.head_ms, chart.song_ms, 0, chart.song_ms, 0.0)
        actual_zero = sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid,
                                               head_mask_bias=zero, **kwargs)
        for actual in (absent, actual_zero):
            for before, after in zip(expected, actual):
                assert before.dtype == after.dtype
                assert before.tobytes() == after.tobytes()


def test_within_mask_conditional_log_probs_unchanged_in_sampler(short_chart, monkeypatch):
    model = tiny_model(dtype=torch.float64, seed=453)
    chart = short_chart
    tilt = FixedEtaBias(chart.head_ms, chart.song_ms, 300, 1000, 0.8)
    pending, observed = [], []
    real_argmax = sampling._gumbel_argmax

    def hook(k, committed, held, logp):
        bias = tilt(k, committed, held, logp)
        pending.append((k, held.copy(), logp.clone(), bias))
        return bias

    def capture(logits, generator):
        if pending:
            k, held, original, bias = pending.pop()
            ids = action_head_masks(held, eos=k == chart.K)
            expected = original.cpu().double() + torch.as_tensor(np.r_[0.0, bias][ids])
            torch.testing.assert_close(logits.cpu().double(), expected, rtol=0, atol=0)
            if k < chart.K:
                for m in range(1, 16):
                    selected = (ids == m) & np.isfinite(original.cpu().numpy())
                    before = original[selected].log_softmax(0)
                    after = logits[selected].log_softmax(0)
                    torch.testing.assert_close(before, after, rtol=0, atol=4e-15)
            observed.append(k)
        return real_argmax(logits, generator)

    monkeypatch.setattr(sampling, '_gumbel_argmax', capture)
    sampling.continue_chart(model, chart.head_ms, chart.song_ms, chart.grid, seed=11, head_mask_bias=hook)
    assert observed == list(range(chart.K + 1))


def test_fixed_eta_is_zero_outside_half_open_scope_and_at_eos(short_chart):
    chart = short_chart
    tilt = FixedEtaBias(chart.head_ms, chart.song_ms, 300, 800, -0.4)
    for k in range(chart.K + 1):
        prefix = chart.with_decisions(chart.actions[:k], chart.gap[:k])
        held = prefix.derived().held[k]
        bias = tilt(k, prefix, held, torch.zeros(625))
        if k == chart.K or not 300 <= chart.time(k) < 800:
            np.testing.assert_array_equal(bias, np.zeros(15))
        else:
            assert np.all(bias < 0)
            np.testing.assert_allclose(bias, -0.4 * tilt.trace.state.mask_costs(chart.time(k)) / tilt.scale)


def test_cached_source_prefix_then_generated_continuation_hook(real_charts):
    _, dec = real_charts[0]
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    model = tiny_model(dtype=torch.float32, seed=989)
    prefix, stop = min(20, dec.K - 10), min(28, dec.K)
    tilt = FixedEtaBias(dec.head_ms, dec.song_ms, float(dec.head_ms[prefix]), float(dec.head_ms[stop]), 0.3)
    visits = []

    def hook(k, chart, held, logp):
        visits.append((k, chart.actions.copy(), chart.gap.copy()))
        return tilt(k, chart, held, logp)

    actions, gap = sampling.continue_chart(model, dec.head_ms, dec.song_ms, grid,
                                           dec.actions[:prefix], dec.gap_release_ms[:prefix],
                                           seed=47, stop=stop, head_mask_bias=hook)
    np.testing.assert_array_equal(visits[0][1], dec.actions[:prefix])
    for k, seen_actions, seen_gap in visits:
        np.testing.assert_array_equal(seen_actions, actions[:k])
        np.testing.assert_array_equal(seen_gap, gap[:k])
    tilt.trace.extend(actions, gap)
    offline = StrainTrace(dec.head_ms, dec.song_ms).extend(actions, gap)
    np.testing.assert_array_equal(tilt.trace.workload_prefix, offline.workload_prefix)
    assert tilt.trace.replay == offline.replay
    assert visits[-1][0] == stop - 1


# Original source captured before the hook was added. Keep this independent of
# the current sampler so absent-vs-zero agreement cannot mask a shared regression.
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
                   track=(), seed: int = 954, stop: int | None = None, *, baseline=None, close_scope=None):
    """Return (actions [n,4], gap [n,4]) for decisions 0..n-1, n = stop (default through EOS).

    ``close_scope=(a, b)`` ends generation early, at the first decision at or after the exit
    decision of [a, b) after which no LN headed in [a, b) is still held (realised-response runs).
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
