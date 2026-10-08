"""Pre-run test 1: complete decision probabilities sum to one, both orientations included."""
from itertools import product

import numpy as np
import torch

from ensomi_model.r2.common import ACTIONS
from ensomi_model.r2.features import decision_factors, lane_query
from ensomi_model.r2.state import action_support_mask

from .helpers import chart_from_objects, hold, sample_track, tap, tiny_model

ROWS = [1000.0, 1030.0, 1062.0, 1072.0, 1135.0, 1170.0]  # gap (1062, 1072) has 2 candidates, EOS gap 5


def fixture(held_lanes, eos=False):
    """Objects whose LNs in ``held_lanes`` are open before the enumerated decision (3, or EOS)."""
    K = len(ROWS)
    end = ROWS[-1] + 15.0 if eos else ROWS[3] + 15.0
    free = [l for l in range(4) if l not in held_lanes]
    objects = []
    if free:
        objects += [hold(ROWS[0], end, l) for l in held_lanes] + [tap(ROWS[0], l) for l in free]
        objects += [tap(t, free[0]) for t in ROWS[1:]]
    else:  # all four held before decision 3: lane 0 holds from row 2 into the gap and taps row 3
        objects += [hold(ROWS[0], end, l) for l in (1, 2, 3)]
        objects += [tap(ROWS[0], 0), tap(ROWS[1], 0), hold(ROWS[2], ROWS[2] + 4.0, 0)]
        objects += [tap(t, 0) for t in ROWS[3:]]
    chart, _ = chart_from_objects(objects, ROWS[-1] + 40.0)
    return chart, (K if eos else 3)


def total_mass(model, chart, k, track):
    prefix = chart.with_decisions(chart.actions[:k], chart.gap[:k])
    d = prefix.derived()
    held = d.held[k]
    mask = action_support_mask(held, k == prefix.K)
    z = model.window_hands(prefix, np.array([k]), track)
    alp = model.action_log_probs(z, mask[None])[0].double()
    lanes_q = lane_query(prefix, [k]).astype(np.float64)[0]
    cands = prefix.candidates(k).times if k > 0 else np.empty(0)
    total = 0.0
    sizes = []
    for a in np.flatnonzero(mask):
        codes = ACTIONS[a]
        lanes = [l for l in range(4) if held[l] and codes[l] in (2, 3, 4)]
        if not lanes:
            total += float(alp[a].exp())
            continue
        combos = list(product(cands, repeat=len(lanes)))
        factors = []
        for i, combo in enumerate(combos):
            rel = np.full(4, np.nan)
            rel[lanes] = combo
            for o in (0, 1):
                factors += decision_factors(prefix, k, i, codes, rel, lanes_q, o)
        zz = z.expand(len(combos), 2, z.shape[-1])
        rlp = model.release_from_factors(zz, factors, prefix, track, len(combos))
        total += float(alp[a].exp() * rlp.double().exp().sum())
        sizes.append(len(cands))
    return total, len(cands)


def test_decision_probabilities_sum_to_one():
    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    cases = [((), False), ((1,), False), ((0, 3), False), ((0, 1, 2), False), ((0, 1, 2, 3), False),
             ((2,), True), ((0, 3), True), ((1, 2, 3), True)]
    for conditioner in ('film', 'tokens'):
        model = tiny_model(torch.float64, conditioner)
        for i, (held, eos) in enumerate(cases):
            chart, k = fixture(set(held), eos)
            track = sample_track(rng, chart) if i % 2 else ()
            with torch.no_grad():
                mass, n_cand = total_mass(model, chart, k, track)
            assert 2 <= n_cand <= 5, n_cand
            assert abs(mass - 1.0) < 1e-8, (conditioner, held, eos, mass)


def test_decision_log_prob_matches_enumeration_terms():
    """The public decision_log_prob equals action + release terms computed above for one decision."""
    model = tiny_model(torch.float64)
    chart, k = fixture({0, 3})
    with torch.no_grad():
        full = float(model.decision_log_prob(chart.with_decisions(chart.actions[:k + 1], chart.gap[:k + 1]), k))
        out = model.window(chart, k, k + 1)
    assert abs(full - float(out.total[0])) < 1e-12
    assert np.isfinite(full)
