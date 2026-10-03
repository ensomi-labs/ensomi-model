"""Condition inputs: FiLM sees active frames only, tokens see the whole track; edits; absent vs zero."""
import numpy as np
import pytest
import torch

from ensomi_model.r2.conditions import draw_track, replace_interval
from ensomi_model.r2.features import Interval, frames, tokens

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid


@pytest.fixture(scope='module')
def chart():
    rng = np.random.default_rng(2)
    head, song = random_skeleton(rng, 80)
    return random_decisions(rng, (head, song, two_segment_grid()))


def test_absent_and_explicit_zero_differ(chart):
    t = [chart.time(10)]
    zero = (Interval(0, 0.0, chart.song_ms, 0.0),)
    assert not np.array_equal(frames(chart, (), t, 10), frames(chart, zero, t, 10))


def test_film_frames_hide_future_values_tokens_show_them(chart):
    k = 10
    t = chart.time(k)
    future = lambda v: (Interval(0, t + 1000.0, t + 5000.0, v),)  # noqa: E731
    assert np.array_equal(frames(chart, future(0.1), [t], k), frames(chart, future(0.9), [t], k))
    assert not np.array_equal(tokens(chart, future(0.1), [t], k), tokens(chart, future(0.9), [t], k))
    model = tiny_model(torch.float64, 'tokens')
    with torch.no_grad():
        a = model.window_hands(chart, np.array([k]), future(0.1))
        b = model.window_hands(chart, np.array([k]), future(0.9))
    assert not torch.allclose(a, b)


def test_token_conditioner_is_identity_at_init_and_handles_natural_mode():
    from ensomi_model.r2.model import R2Config, R2Model
    model = R2Model(R2Config(conditioner='tokens')).double()
    z = torch.randn(5, 2, 128, dtype=torch.float64)
    assert torch.equal(model.condition(z, torch.zeros(5, 0, 18, dtype=torch.float64)), z)
    assert torch.equal(model.condition(z, torch.randn(5, 4, 18, dtype=torch.float64)), z)


def test_condition_counts_use_committed_heads_only(chart):
    k = 30
    iv = (Interval(0, 0.0, chart.song_ms, 0.5), Interval(1, 0.0, chart.song_ms, 2.0))
    full = frames(chart, iv, [chart.time(k)], k)
    prefix = chart.with_decisions(chart.actions[:k], chart.gap[:k])
    assert np.array_equal(full, frames(prefix, iv, [chart.time(k)], k))
    assert np.array_equal(tokens(chart, iv, [chart.time(k)], k), tokens(prefix, iv, [chart.time(k)], k))


def test_replace_interval_clips_splits_and_coalesces():
    track = (Interval(0, 0.0, 1000.0, 0.2), Interval(1, 0.0, 5000.0, 3.0))
    new, span = replace_interval(track, 0, 200.0, 600.0, 0.5, frontier=None)
    lns = sorted((iv.a, iv.b, iv.value) for iv in new if iv.kind == 0)
    assert lns == [(0.0, 200.0, 0.2), (200.0, 600.0, 0.5), (600.0, 1000.0, 0.2)]
    new2, span2 = replace_interval(new, 0, 100.0, 700.0, 0.2, frontier=300.0)
    assert span2[0] > 300.0
    lns2 = sorted((iv.a, iv.b, iv.value) for iv in new2 if iv.kind == 0)
    assert lns2[0][0] == 0.0 and lns2[0][2] == 0.2 and lns2[-1][1] == 1000.0
    with pytest.raises(Exception):
        replace_interval(track, 0, 0.0, 100.0, 0.5, frontier=200.0)


def test_training_draws_are_valid(chart):
    rng = np.random.default_rng(0)
    labels = [(0.0, chart.song_ms, 3.1)]
    for _ in range(50):
        track, record = draw_track(chart, labels, rng, star_conditions=True)
        for kind in (0, 1):
            ivs = sorted((iv for iv in track if iv.kind == kind), key=lambda iv: iv.a)
            assert all(x.b <= y.a for x, y in zip(ivs, ivs[1:]))
