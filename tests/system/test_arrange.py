import numpy as np
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
