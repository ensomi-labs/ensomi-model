import numpy as np
import pytest
import torch

from ensomi_model.r2.sampling import continue_chart
from ensomi_model.r2.state import action_support_mask
from ensomi_model.r2.strain import action_head_masks
from ensomi_model.r2.theta import decode
from ensomi_model.system.arrange import arrange, mask_log_probs
from ensomi_model.system.controls import validate_requests

from .helpers import calibration, grid, skeleton, tiny_r2


def test_without_force_masks_or_copy_it_is_r2s_own_sampler():
    rng = np.random.default_rng(5)
    head, song = skeleton(rng, 50)
    model = tiny_r2()
    expected = continue_chart(model, head, song, grid(), seed=954)
    actions, gap, _ = arrange(model, head, song, grid(), seed=954, target=[0, 1, 0], kappa=0)
    assert actions.tobytes() == expected[0].tobytes()
    assert np.array_equal(gap, expected[1], equal_nan=True)


def test_masks_off_preserves_grammar_and_matches_unmasked_sampler():
    from ensomi_model.r2.state import replay_decisions
    head = np.arange(30, dtype=float) * 20 + 1000
    model = tiny_r2()
    kwargs = dict(seed=954, target=[.3, 2., .2], kappa=3)
    expected, eg, _ = arrange(model, head, 1700., grid(), **kwargs)
    actual, ag, trace = arrange(model, head, 1700., grid(), masks=calibration()['masks']['2'],
                                masks_off=True, **kwargs)
    assert np.array_equal(actual, expected)
    assert np.array_equal(ag, eg, equal_nan=True)
    assert trace['fallback_count'] == 0
    final, _ = replay_decisions(head, 1700., actual, ag)
    assert final.finalized and not np.any(final.held)


def test_prefix_continuation_matches_natural_sampler_and_keeps_open_holds():
    head, song = skeleton(np.random.default_rng(12), 80)
    model = tiny_r2()
    original, releases = continue_chart(model, head, song, grid(), seed=9)
    held, _, _ = decode(original)
    cut = next(k for k in range(10, 60) if held[k].any())
    kwargs = dict(prefix_actions=original[:cut], prefix_gap=releases[:cut], seed=954)
    expected = continue_chart(model, head, song, grid(), **kwargs)
    actions, gap, trace = arrange(model, head, song, grid(), target=[0, 1, 0], kappa=0, **kwargs)
    assert np.array_equal(actions, expected[0])
    assert np.array_equal(gap, expected[1], equal_nan=True)
    assert np.array_equal(actions[:cut], original[:cut])
    assert np.isnan(trace['action_nll'][:cut]).all()


def test_prefix_populates_controller_and_playability_history():
    from ensomi_model.r2.state import replay_decisions
    head = np.arange(20, dtype=float) * 125 + 1000
    model = tiny_r2()
    prefix = np.zeros((8, 4), dtype=int)
    prefix[:, 0] = 1
    a, g, trace = arrange(model, head, 4000., grid(), seed=5, target=[0, 1.5, 0], kappa=3,
                          masks=calibration()['masks']['3'], prefix_actions=prefix,
                          prefix_gap=np.full((8, 4), np.nan))
    assert trace['heads_force_running'][8] == 1
    assert trace['heads_window_rows'][8] == 8
    assert trace['mask_firings'][8, 2] > 0
    assert replay_decisions(head, 4000., a, g)[0].finalized


def test_fixed_pointer_orientation_is_legal_and_validated():
    from ensomi_model.r2.state import replay_decisions
    head, song = skeleton(np.random.default_rng(13), 50)
    model = tiny_r2()
    for orientation in (0, 1):
        a, g, _ = arrange(model, head, song, grid(), seed=4, target=[0, 1, 0], kappa=0,
                          pointer_orientation=orientation)
        assert replay_decisions(head, song, a, g)[0].finalized
    with pytest.raises(ValueError, match='orientation'):
        arrange(model, head, song, grid(), seed=4, target=[0, 1, 0], kappa=0, pointer_orientation=2)
    with pytest.raises(ValueError, match='together'):
        arrange(model, head, song, grid(), seed=4, target=[0, 1, 0], kappa=0,
                prefix_actions=np.zeros((1, 4)))


def test_a_zero_copy_bias_changes_nothing_and_a_strong_one_steers_the_heads():
    rng = np.random.default_rng(6)
    head, song = skeleton(rng, 50)
    model = tiny_r2()
    base, _, _ = arrange(model, head, song, grid(), seed=954, target=[0, 1, 0], kappa=0)
    zero, _, tr = arrange(model, head, song, grid(), seed=954, target=[0, 1, 0], kappa=0,
                          copy=lambda k, hist, lp15: np.zeros(15))
    assert zero.tobytes() == base.tobytes() and tr['copy_abs'].max() == 0
    seen = []

    def lane0(k, hist, lp15):
        seen.append(hist[:k].copy())
        b = np.zeros(15)
        b[0] = 30.0                     # mask 1: lane 0 only
        return b

    steered, _, _ = arrange(model, head, song, grid(), seed=954, target=[0, 1, 0], kappa=0, copy=lane0)
    _, heads, _ = decode(steered[:len(head)])
    assert (heads == [True, False, False, False]).all(1).mean() > 0.9
    masks = heads.astype(int) @ (1 << np.arange(4))
    assert all(np.array_equal(h, masks[:k]) for k, h in enumerate(seen))      # the bias sees committed masks only


def test_mask_log_probs_sums_actions_by_head_mask():
    held = np.array([True, False, False, False])
    support = action_support_mask(held, False)
    logp = np.where(support, -np.log(support.sum()), -np.inf)
    ids = action_head_masks(held)
    lp = mask_log_probs(logp, held)
    expected = [np.log(((ids == m) & support).sum() / support.sum()) for m in range(1, 16)]
    assert np.allclose(lp, expected)


def test_controls_run_on_rows_with_masks_force_and_a_scope():
    rng = np.random.default_rng(7)
    head, song = skeleton(rng, 120)
    cal = calibration()
    a, b = float(head[30]), float(head[30] + 9000)
    request = dict(concept='jack', q=0.9, a_ms=a, b_ms=b, target=0.1, band=[-0.4, 0.2])
    validate_requests([request], song)
    actions, gap, tr = arrange(tiny_r2(), head, song, grid(), seed=3, target=[0.05, 1.4, 0.02], kappa=3,
                               ln_fraction=0.0, requests=[request], masks=cal['masks']['3'],
                               copy=lambda k, hist, lp15: np.full(15, 0.1))
    held, heads, ln = decode(actions[:len(head)])
    assert ln.sum() == 0                          # tap only: no LN heads
    assert len(tr['requests']) == 1 and tr['requests'][0]['band_status'] == 'in band'
    assert tr['requests'][0]['transitions'] > 0
    assert heads.any(1).all()


def test_copy_bias_through_r2s_own_hook_matches_arrange():
    from ensomi_model.system.copy import CopyBias, CopyConfig, CopyPath
    rng = np.random.default_rng(8)
    head, song = skeleton(rng, 60)
    g = grid()
    model = tiny_r2()
    torch.manual_seed(1)
    path = CopyPath(CopyConfig())
    with torch.no_grad():
        for p in path.parameters():
            p.normal_(0, 0.3)
    mel = rng.normal(size=(int(song / 10) + 10, 130)).astype(np.float16)
    a, _, _ = arrange(model, head, song, g, seed=954, target=[0, 1, 0], kappa=0,
                      copy=CopyBias(path.eval(), head, g, mel, strength=0.5))
    b, _ = continue_chart(model, head, song, g, seed=954, head_mask_bias=CopyBias(path, head, g, mel, strength=0.5).hook)
    plain, _ = continue_chart(model, head, song, g, seed=954)
    assert a.tobytes() == b.tobytes() and a.tobytes() != plain.tobytes()
