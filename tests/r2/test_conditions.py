"""Condition inputs: FiLM sees active frames only, tokens (a lead-in form) see the whole track; absent vs zero;
training draws are effective tracks. ``replace_interval`` is retired: requests are validated at addition
(tests/r2/test_requests.py, T-A and T-C)."""
import numpy as np
import pytest
import torch

from ensomi_model.r2.common import ContractError
from ensomi_model.r2.conditions import DrawConfig, draw_window, validate_track
from ensomi_model.r2.features import FRAME_DIM, Interval, frames, tokens
from ensomi_model.r2.model import R2Config
from ensomi_model.r2.request_set import effective_track, from_track

from .helpers import random_decisions, random_skeleton, tiny_model, two_segment_grid


@pytest.fixture(scope='module')
def chart():
    rng = np.random.default_rng(2)
    head, song = random_skeleton(rng, 80)
    return random_decisions(rng, (head, song, two_segment_grid()))


@pytest.fixture(scope='module')
def long_chart():
    rng = np.random.default_rng(3)
    head, song = random_skeleton(rng, 400, lo=150, hi=450)    # about two minutes: every cell length exists
    return random_decisions(rng, (head, song, two_segment_grid()))


def star_cells(chart, value=3.1):
    T = chart.song_ms
    cells = {'whole': [(0.0, T, value)]}
    for L in (30, 60, 120):
        for p in (0, 10, 20):
            a, out = 1000.0 * p, []
            while a + 1000.0 * L <= T:
                out.append((a, a + 1000.0 * L, value))
                a += 1000.0 * L
            cells[(L, p)] = out
    return cells


def test_absent_and_explicit_zero_differ(chart):
    t = [chart.time(10)]
    zero = (Interval(0, 0.0, chart.song_ms, 0.0),)
    assert not np.array_equal(frames(chart, (), t, 10), frames(chart, zero, t, 10))


def test_frame_layout_and_reserved_channel(chart):
    t = [chart.time(10)]
    track = (Interval(0, 0.0, chart.song_ms, 0.4), Interval(1, 0.0, chart.song_ms, 3.0))
    f = frames(chart, track, t, 10)
    assert f.shape == (1, 2, FRAME_DIM) == (1, 2, 17)
    assert (f[..., -1] == 0).all()                       # the lead-in channel is zero under presence 'none'
    g = frames(chart, (), t, 10, presence='anywhere', full_track=track)
    assert (g[..., -1] == 1).all() and (g[..., :-1] == 0).all()
    with pytest.raises(ContractError):
        frames(chart, track, t, 10, presence='lead-in')
    with pytest.raises(ContractError):
        R2Config(presence='lead-in')


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


def test_token_conditioner_raises_under_the_default_eta():
    with pytest.raises(ContractError):
        R2Config(conditioner='tokens')


def test_token_conditioner_is_identity_at_init_and_handles_natural_mode():
    from ensomi_model.r2.model import R2Model
    model = R2Model(R2Config(conditioner='tokens', token_lead_in=True)).double()
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


@pytest.mark.parametrize('selection', ['per_window', 'per_song'])
def test_training_draws_are_effective_tracks(long_chart, selection):
    """Every drawn track passes validation and maps to itself through the request set (T-R, first clause)."""
    rng = np.random.default_rng(0)
    cfg = DrawConfig(selection=selection, **({} if selection == 'per_window' else dict(p_align=0.0, ipw=False)))
    seen = 0
    chart = long_chart
    for _ in range(80):
        w = draw_window(chart, star_cells(chart), rng, cfg, star_conditions=True)
        validate_track(w.track, song_ms=chart.song_ms)
        if not w.track:
            continue
        seen += 1
        rs = from_track(w.track, chart.song_ms)
        eff = effective_track(rs, chart.head_ms, chart.grid)
        assert sorted(eff.track, key=lambda iv: (iv.kind, iv.a)) == sorted(w.track, key=lambda iv: (iv.kind, iv.a))
        assert 0 <= w.start < w.stop <= chart.K + 1 and w.weight > 0
        if selection == 'per_song':
            assert w.weight == 1.0
    assert seen > 20
