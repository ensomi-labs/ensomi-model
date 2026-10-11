"""Each named model input changes the decision likelihood when it changes (none is silently ignored),
and ``sequence_log_prob`` is the differentiable sum of per-decision log-probabilities."""
import numpy as np
import torch

from ensomi_model.r2.features import Chart, Interval

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid


def setup(seed=13, K=200):
    rng = np.random.default_rng(seed)
    grid = two_segment_grid()
    head, song = random_skeleton(rng, K, lo=30, hi=300)
    return random_decisions(rng, (head, song, grid), p_hold=0.5), grid


def lp(model, chart, k, track=()):
    with torch.no_grad():
        return float(model.decision_log_prob(chart.with_decisions(chart.actions[:k + 1], chart.gap[:k + 1]), k, track))


def first_with_release(chart, lo):
    d = chart.derived()
    for k in range(lo, chart.K):
        if any(d.held[k][l] and chart.actions[k][l] in (2, 3, 4) for l in range(4)):
            return k
    raise AssertionError('no gap release in fixture')


def test_history_lookahead_landmarks_and_conditions_are_used():
    chart, grid = setup()
    model = tiny_model(torch.float64)
    k = first_with_release(chart, 140)
    base = lp(model, chart, k)
    # committed history: change an earlier decision's release time
    d = chart.derived()
    j = next(j for j in range(k - 1, 0, -1) if np.isfinite(chart.gap[j]).any())
    gap = chart.gap.copy()
    lane = int(np.flatnonzero(np.isfinite(gap[j]))[0])
    others = [u for u in chart.candidates(j).times if u != gap[j, lane]]
    gap[j, lane] = others[0]
    assert lp(model, Chart(chart.head_ms, chart.song_ms, grid, chart.actions, gap), k) != base
    # look-ahead: move a future head row (inputs after k only)
    head = chart.head_ms.copy()
    head[k + 3] = (head[k + 2] + head[k + 3]) / 2.0 + 1.0
    if head[k + 3] - head[k + 2] >= 2.0:
        assert lp(model, Chart(head, chart.song_ms, grid, chart.actions, chart.gap), k) != base
    # landmarks: the read contributes for k > 64
    marks = model.read_landmarks
    model.read_landmarks = lambda h, m, v: torch.zeros_like(h)
    try:
        assert lp(model, chart, k) != base
    finally:
        model.read_landmarks = marks
    # conditions: an active LN interval and an active star interval each change the likelihood
    t = chart.time(k)
    assert lp(model, chart, k, (Interval(0, t - 500.0, t + 500.0, 0.8),)) != base
    assert lp(model, chart, k, (Interval(1, t - 500.0, t + 500.0, 4.0),)) != base
    del d


def test_sequence_log_prob_sums_decisions_and_has_gradients():
    chart, grid = setup(seed=21, K=60)
    model = tiny_model(torch.float64)
    j, stop = 10, 40
    total = model.sequence_log_prob(chart, j, stop)
    parts = [lp(model, chart, k) for k in range(j, stop)]
    assert abs(float(total.detach()) - sum(parts)) < 1e-9
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.candidate.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.temporal.parameters())
    # including EOS
    full = model.sequence_log_prob(chart, chart.K - 5, chart.K + 1)
    assert torch.isfinite(full)
