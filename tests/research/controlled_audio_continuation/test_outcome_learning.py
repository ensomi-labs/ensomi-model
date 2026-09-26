"""Exact finite-policy checks for coupled draws and state-dependent KL credit."""
from itertools import product

import numpy as np
import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.outcome_learning import (
    paired_gap_coefficients, trajectory_kl_coefficients,
)


@pytest.mark.parametrize('probabilities', [(0.31, 0.68), (0.77, 0.23)])
@pytest.mark.parametrize('count', [2, 3])
def test_gap_gradient_with_common_random_numbers(probabilities, count):
    p = np.array(probabilities)
    boundaries = sorted({0., *p, 1.})
    coupled = [((np.array([(a+b)/2]*2) < p).astype(float), b-a)
               for a, b in zip(boundaries[:-1], boundaries[1:])]
    gradient = np.zeros(2)
    offsets, scales, target_gap = np.array([2., 2.5]), np.array([1.2, .8]), .9
    for selection in product(range(len(coupled)), repeat=count):
        actions = np.stack([coupled[i][0] for i in selection], -1)
        probability = np.prod([coupled[i][1] for i in selection])
        outcomes = offsets[:, None]+scales[:, None]*actions
        coefficients = paired_gap_coefficients(outcomes, target_gap)
        gradient += probability*(coefficients*(actions-p[:, None])).sum(-1)
    mean = offsets+scales*p
    error = mean[1]-mean[0]-target_gap
    exact = 2*error*np.array([-1., 1.])*scales*p*(1-p)
    np.testing.assert_allclose(gradient, exact, atol=1e-12)


@pytest.mark.parametrize('baseline', [0., .63])
def test_kl_credit_includes_changes_to_later_visited_states(baseline):
    logits = torch.tensor([-.2, .5, -.6], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([.58, .19, .71], dtype=torch.float64, requires_grad=True)
    p = logits.sigmoid()
    exact = logits.new_zeros(())
    surrogate = logits.new_zeros(())
    for a, b in product((0, 1), repeat=2):
        # The first action selects a different second-step conditional policy.
        # Treating per-row KL as a direct loss alone misses this occupancy effect.
        indices, actions = [0, 1+a], logits.new_tensor([a, b])
        selected = p[indices]
        selected_ref = reference[indices]
        chosen = (actions*selected+(1-actions)*(1-selected)).log()
        ref = (actions*selected_ref+(1-actions)*(1-selected_ref)).log()
        probability = chosen.sum().exp()
        exact = exact+probability*(chosen-ref.detach()).sum()
        coefficient = trajectory_kl_coefficients(chosen, ref, baseline)
        assert not coefficient.requires_grad
        surrogate = surrogate+probability.detach()*(coefficient*chosen).sum()
    expected = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    actual = torch.autograd.grad(surrogate, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert reference.grad is None
