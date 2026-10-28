import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes.episodes import Reach, fixtures, sparsemax, contiguous, kernel
from n1_redesign.probes import episodes


def test_all_arms_preserve_literal_pulse_and_no_marks():
    r = fixtures()
    assert r['passed'] and r['updates'] == 33


def test_sparsemax_has_a_zero_null_value_and_sparse_support():
    scores = torch.tensor([[2., 0., 0.], [.3, .3, .3]], requires_grad=True)
    weights = sparsemax(scores)
    np.testing.assert_allclose(weights.detach().numpy()[0], [1, 0, 0])
    np.testing.assert_allclose(weights.detach().numpy().sum(axis=1), 1)
    (weights[:, 0].sum()).backward()
    assert torch.isfinite(scores.grad).all()


def test_local_run_cannot_jump_a_gap():
    s = np.array([.9, .8, .1, 1., .8, .7])
    np.testing.assert_array_equal(contiguous(s, 3, .6), [False, False, False, True, True, True])
    np.testing.assert_allclose(kernel(np.array([.2, .6, 1]), .6), [0, 0, 1])


def test_learned_zero_marks_is_exact_and_trainable_mark_path():
    torch.manual_seed(10)
    m = Reach()
    c, base = torch.randn(7, 192), torch.randn(7)
    assert torch.equal(m(c, base, []), torch.zeros(7))
    delta = m(c, base, [dict(beat=6, L=2., kind=0)])
    delta.square().sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in m.parameters())
    assert 5000 <= sum(p.numel() for p in m.parameters()) <= 6500


def test_training_draw_is_uniform_over_eligible_heads_not_third_bins(monkeypatch):
    monkeypatch.setattr(episodes, 'calibration', lambda: dict(
        offset_mean_ms=0., offset_sd_ms=1., residual_robust_sd_ms=1.))
    z = dict(beats=np.array([0.]), length=np.array([600.]), heads=np.array([200.]),
             divisor=np.array([3]), head_beat=np.array([0]))
    for position in range(3):
        marks = episodes.simulate(z, np.array([.1]), np.array([True]), np.ones((1, 192)),
            simulator=0, position=position, run_length=1, confirming=0, identity='fixture', training=True)
        assert len(marks) == 1 and marks[0]['beat'] == 0
