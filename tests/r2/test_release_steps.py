"""Common-space likelihood and future-feature boundaries for release hazards."""
import numpy as np
from itertools import product
import pytest
import torch

from ensomi_model.r2.common import ACTIONS, GridArrays
from ensomi_model.r2.features import Chart
from ensomi_model.r2.model import R2Config, R2Model
from ensomi_model.r2.release_steps import (ReleaseStepModel, lane_state_features, point_candidates,
                                          release_queries, continue_release_steps, make_step_factor)
from ensomi_model.r2.state import action_support_mask, replay_decisions
from .helpers import random_decisions


def grid():
    return GridArrays(np.array([0., 1001.25]), np.array([500., 333.333]),
                      np.array([0., 1001.25 / 500]), np.array([4., 3.]))


def model(blind=False):
    torch.manual_seed(171)
    return ReleaseStepModel(R2Config(hidden=12, levels=1, expansion=2, rank=3, memory='none'), blind=blind)


def test_initial_parameters_are_exactly_matched_to_r2():
    steps = model()
    torch.manual_seed(171)
    baseline = R2Model(steps.config)
    assert baseline.state_dict().keys() == steps.state_dict().keys()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, steps.state_dict()[name]), name


def test_r2_loader_rejects_shape_compatible_release_step_weights(tmp_path):
    from ensomi_model.r2.bakeoff_panel import load_model
    path = tmp_path / 'steps.pt'
    torch.save(dict(representation='visible'), path)
    with pytest.raises(ValueError, match='own likelihood'):
        load_model(path)


@pytest.mark.parametrize('held,delta', [([True, False, False, False], 2.), ([True] * 4, 2.),
                                      ([True, False, False, False], 17.)])
def test_all_legal_gap_outcomes_have_total_probability_one(held, delta):
    m = model()
    head = np.array([1000., 1000. + delta])
    first = np.where(held, 2, 0)
    legal = ACTIONS[action_support_mask(np.array(held), False)]
    times = Chart(head, head[-1] + 2, grid(), first[None], np.full((1, 4), np.nan)).candidates(1).times
    probs = []
    with torch.inference_mode():
        for a in legal:
            lanes = np.array(held) & (a >= 2)
            for releases in product(times, repeat=int(lanes.sum())):
                gap = np.full((2, 4), np.nan)
                gap[1, lanes] = releases
                c = Chart(head, head[-1] + 2, grid(), np.stack([first, a]), gap)
                probs.append(float(m.window(c, 1, 2).total.exp()))
    assert sum(probs) == pytest.approx(1., abs=2e-6)


def test_eos_likelihood_is_normalized_over_all_release_pairs():
    m = model()
    c = Chart(np.array([1000.]), 1017., grid(), np.array([[2, 2, 0, 0], [2, 2, 0, 0]]),
              np.full((2, 4), np.nan))
    probability = 0.
    with torch.inference_mode():
        for ends in product(c.candidates(1).times, repeat=2):
            gap = c.gap.copy(); gap[1, :2] = ends
            probability += float(m.window(c.with_decisions(c.actions, gap), 1, 2).total.exp())
    assert probability == pytest.approx(1., abs=2e-6)


@pytest.mark.parametrize('blind', [False, True])
def test_mirror_likelihood_and_finite_backward(blind):
    m = model(blind)
    c = random_decisions(np.random.default_rng(181),
        (np.array([999., 1004., 1010., 1020., 1100.]), 1130., grid()), p_hold=.8)
    mirrored = c.with_decisions(c.actions[:, ::-1].copy(), c.gap[:, ::-1].copy())
    a = m.window(c, 0, c.n).total
    b = m.window(mirrored, 0, c.n).total
    torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-6)
    (-a.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_blind_point_features_do_not_read_next_time_or_anchor_flags():
    g = grid()
    a = point_candidates(1000., 1020., [1004., 1010.], g, blind=True)
    b = point_candidates(1000., 4000., [1004., 1010.], g, blind=True)
    assert np.array_equal(a.static, b.static)
    assert np.isnan(a.b)
    assert not a.static[:, :4].any()
    assert not a.static[:, 8:12].any()
    visible = point_candidates(1000., 1020., [1004., 1010.], g)
    assert not np.array_equal(a.static, visible.static)


def test_point_features_match_cand_v1_across_a_fractional_tempo_boundary():
    from ensomi_model.r2.candidates import build_candidates
    candidates = build_candidates(900., 1100., grid())
    points = point_candidates(900., 1100., candidates.times, grid())
    np.testing.assert_allclose(points.static, candidates.static, atol=1e-12, rtol=0.)
    assert np.array_equal(points.flags, candidates.flags)


def test_blind_release_query_accepts_only_past_rows_and_current_clock():
    g = grid()
    lanes = lane_state_features([1010.], g, np.ones((1, 4), bool), np.full((1, 4), 1000.),
                                np.full((1, 4), 1000.), np.full((1, 4), np.nan))
    q = release_queries([1010.], g, lanes, 1000., 3000., np.array([500., 1000.]))
    assert q.shape == (1, 2, 150)
    assert not q[:, :, 80:].any()  # density and lookahead
    assert not q[:, :, 69].any()   # no index/total-row-count shortcut
    changed_end = release_queries([1010.], g, lanes, 1000., 9000., np.array([500., 1000.]))
    assert np.array_equal(q, changed_end)
    visible = release_queries([1010.], g, lanes, 1000., 3000., np.array([500., 1000.]),
                               np.array([1020., 1040., 1500.]))
    assert not np.array_equal(q, visible)


def test_release_likelihood_does_not_read_the_later_head_allocation():
    m = model()
    h = np.array([1000., 1002.])
    g = np.full((2, 4), np.nan)
    g[1, 0] = 1001.
    a = Chart(h, 1004., grid(), np.array([[2, 0, 0, 0], [3, 1, 0, 0]]), g)
    b = a.with_decisions(np.array([[2, 0, 0, 0], [4, 2, 1, 0]]), g)
    with torch.inference_mode():
        aa, bb = m.window(a, 1, 2), m.window(b, 1, 2)
    torch.testing.assert_close(aa.release, bb.release, atol=0., rtol=0.)


def test_blind_raw_release_logit_is_invariant_to_future_rows_and_song_end():
    m = model(blind=True)
    g = GridArrays(np.array([0.]), np.array([500.]), np.array([0.]), np.array([4.]))
    scores = []
    with torch.inference_mode():
        for future, end in [([1020., 1040.], 2000.), ([1100., 1500., 1800.], 3000.)]:
            c = Chart(np.r_[1000., future], end, g, np.array([[2, 2, 0, 0]]), np.full((1, 4), np.nan))
            d = c.derived()
            lanes = lane_state_features([1016.], g, d.held[1:2], d.start[1:2],
                                        d.last_attack[1:2], d.last_release[1:2])
            q = release_queries([1016.], g, lanes, 1000., end, c.head_ms[:1])
            z = m.step_hands(m.before_rows(c, [1]), q)
            point = point_candidates(1000., future[0], [1016.], g, blind=True)
            factor = make_step_factor(c, 1, 0, point, lanes[0], {}, 0, 0)
            score, _, _ = m.all_pair_scores(m.factor_contexts(z, [factor]), [factor], c, ())
            scores.append(score)
    torch.testing.assert_close(scores[0], scores[1], atol=0., rtol=0.)


def test_scheduler_is_not_blind_even_when_numeric_features_are():
    # This pins the limitation rather than pretending an input-zeroing test
    # proves a causal generator. cand-v1 includes the next-head midpoint.
    from ensomi_model.r2.candidates import build_candidates
    a = build_candidates(1000., 1020., grid()).times
    b = build_candidates(1000., 1040., grid()).times
    assert not np.array_equal(a, b[b < 1020.])


@pytest.mark.parametrize('blind', [False, True])
def test_sampler_keeps_prefix_and_closes_holds_even_with_dense_full_occupancy(blind):
    m = model(blind)
    head = np.array([1000., 1002., 1004., 1006., 1010.])
    prefix = np.array([[2, 2, 2, 2]])
    masks = dict(reattack_floor_ms=145., quad_run_cap=0,
                 gap_tercile_edges_ms=[153., 197.], jack_run_caps=[2, 3, 5])
    for seed in range(3):
        a, g, trace = continue_release_steps(m, head, 1012., grid(), seed=seed,
            prefix_actions=prefix, prefix_gap=np.full((1, 4), np.nan), masks=masks,
            target=[.5, 2., .5], kappa=3)
        final, _ = replay_decisions(head, 1012., a, g)
        assert final.finalized and not np.any(final.held)
        assert np.array_equal(a[:1], prefix)
        assert trace['fallback_count'] > 0
        with torch.inference_mode():
            lp = m.window(Chart(head, 1012., grid(), a, g), 1, len(a)).total
        assert torch.isfinite(lp).all()
