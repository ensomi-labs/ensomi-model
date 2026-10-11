"""Pre-run test 2: p(Md | Ms) = p(d | s) for whole decisions, support masks permute, advance commutes."""
import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ACTIONS, MIRROR_ACTION
from ensomi_model.r2.features import Chart
from ensomi_model.r2.state import (R2State, action_support_mask, advance, decision_at, mirror_decision,
                                   mirror_state, replay_decisions)

from .helpers import random_decisions, random_skeleton, sample_track, tiny_model, two_segment_grid


def mirrored(chart: Chart) -> Chart:
    return Chart(chart.head_ms, chart.song_ms, chart.grid, chart.actions[:, ::-1].copy(), chart.gap[:, ::-1].copy())


def states_and_decisions(n_states=32, per_state=5, seed=20261004):
    rng = np.random.default_rng(seed)
    grid = two_segment_grid()
    out = []
    while len(out) < n_states:
        K = int(rng.integers(3, 160))
        head, song = random_skeleton(rng, K, lo=20, hi=260)
        base = random_decisions(rng, (head, song, grid))
        k = int(rng.integers(0, K + 1)) if len(out) % 4 else K  # include EOS states
        variants = []
        for _ in range(per_state):
            acts, gap = base.actions.copy(), base.gap.copy()
            # same prefix, a different legal decision at k drawn by re-sampling from the prefix state
            prefix = base.with_decisions(acts[:k], gap[:k])
            held = prefix.derived().held[k]
            mask = action_support_mask(held, k == K)
            a = ACTIONS[rng.choice(np.flatnonzero(mask))]
            acts[k] = a
            gap[k] = np.nan
            if k > 0:
                cands = prefix.candidates(k).times
                for lane in range(4):
                    if held[lane] and a[lane] in (2, 3, 4):
                        gap[k, lane] = cands[rng.integers(0, len(cands))]
            variants.append(Chart(head, song, grid, acts[:k + 1], gap[:k + 1]))
        out.append((variants, k, sample_track(rng, base) if len(out) % 2 else ()))
    return out


@pytest.mark.parametrize('conditioner', ['film', 'tokens'])
def test_mirror_identity_cpu_float64(conditioner):
    model = tiny_model(torch.float64, conditioner)
    worst = 0.0
    with torch.no_grad():
        for variants, k, track in states_and_decisions():
            for chart in variants:
                lp = float(model.decision_log_prob(chart, k, track))
                lm = float(model.decision_log_prob(mirrored(chart), k, track))
                assert np.isfinite(lp)
                worst = max(worst, abs(lp - lm))
    assert worst <= 1e-10, worst


def test_support_masks_and_advance_commute_with_mirror():
    rng = np.random.default_rng(3)
    grid = two_segment_grid()
    for _ in range(32):
        K = int(rng.integers(2, 80))
        head, song = random_skeleton(rng, K)
        chart = random_decisions(rng, (head, song, grid))
        k = int(rng.integers(0, K + 1))
        state, _ = replay_decisions(head, song, chart.actions, chart.gap, upto=k)
        mstate = mirror_state(state)
        for eos in (False, True):
            m = action_support_mask(state.held, eos)
            mm = action_support_mask(mstate.held, eos)
            assert np.array_equal(mm[MIRROR_ACTION], m)
        d = decision_at(chart.actions, chart.gap, k)
        s1, rows = advance(state, d, head, song)
        s2, mrows = advance(mstate, mirror_decision(d), head, song)
        assert s2 == mirror_state(s1)
        assert [(r.time_ms, r.actions[::-1]) for r in rows] == [(r.time_ms, r.actions) for r in mrows]


def test_asymmetric_prefix_is_not_trivially_symmetric():
    """p(Md | s) differs from p(d | s) on an asymmetric history, so the identity is not a row-only average."""
    model = tiny_model(torch.float64)
    rng = np.random.default_rng(11)
    grid = two_segment_grid()
    head, song = random_skeleton(rng, 40)
    chart = random_decisions(rng, (head, song, grid))
    free = [k for k in range(5, 40) if not chart.derived().held[k].any()]
    assert free, 'fixture needs a held-free state'
    k = free[0]
    acts = chart.actions[:k + 1].copy()
    acts[k] = (1, 0, 0, 0)
    alt = acts.copy()
    alt[k] = (0, 0, 0, 1)
    with torch.no_grad():
        a = float(model.decision_log_prob(Chart(head, song, grid, acts, chart.gap[:k + 1]), k))
        b = float(model.decision_log_prob(Chart(head, song, grid, alt, chart.gap[:k + 1]), k))
    assert abs(a - b) > 1e-6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS not available')
def test_mirror_identity_mps_float32():
    model = tiny_model(torch.float32).to('mps')
    worst = 0.0
    with torch.no_grad():
        for variants, k, track in states_and_decisions(n_states=16, per_state=3):
            for chart in variants:
                lp = float(model.decision_log_prob(chart, k, track))
                lm = float(model.decision_log_prob(mirrored(chart), k, track))
                worst = max(worst, abs(lp - lm))
    assert worst <= 3e-5, worst


def test_directed_pointer_alone_breaks_the_identity(monkeypatch):
    """Control: scoring only the forward orientation must violate the identity on some decision."""
    model = tiny_model(torch.float64)
    original = model.release_from_factors

    def forward_only(z, factors, chart, track, m):
        kept = [f for f in factors if f.orientation == 0]
        if not kept:
            return z.new_zeros(m)
        lp = model.factor_log_probs(z, kept, chart, track)
        slot = torch.as_tensor([f.owner for f in kept])
        return z.new_zeros(m).index_add(0, slot, lp)

    worst_mixture, worst_directed = 0.0, 0.0
    with torch.no_grad():
        cases = states_and_decisions(n_states=12, per_state=4)
        for variants, k, track in cases:
            for chart in variants:
                worst_mixture = max(worst_mixture, abs(float(model.decision_log_prob(chart, k, track)) -
                                                       float(model.decision_log_prob(mirrored(chart), k, track))))
        monkeypatch.setattr(model, 'release_from_factors', forward_only)
        for variants, k, track in cases:
            for chart in variants:
                worst_directed = max(worst_directed, abs(float(model.decision_log_prob(chart, k, track)) -
                                                         float(model.decision_log_prob(mirrored(chart), k, track))))
    assert worst_mixture <= 1e-10
    assert worst_directed > 1e-6
    del original
