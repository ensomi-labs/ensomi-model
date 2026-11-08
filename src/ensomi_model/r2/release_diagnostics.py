"""Future-dependent opportunity and completion counts on fixed head rows.

The opportunity proxy extends each gap by 25% (at least 2 ms), then compares
candidate prefixes before the original next head. It measures available
schedule information, not whether a model uses it. Completion replay averages
the two fixed lane orders; it is counterfactual for human charts and R2.
"""
from __future__ import annotations

import numpy as np

from .candidates import build_candidates


def opportunity_profile(head_ms, grid):
    """Return per-head-gap changed-prefix flags and final interior candidates.

    Index zero is BOS and excluded from every count. The hypothetical later
    next head need not precede the song's subsequent heads: this is a local
    opportunity-prefix probe, not a legal full-song timing intervention.
    """
    head = np.asarray(head_ms)
    changed = np.zeros(len(head), bool)
    last = np.full(len(head), np.nan)
    for k in range(1, len(head)):
        a, b = head[k - 1:k + 1]
        delta = max(2., np.floor((b - a) / 4. + .5))
        original = build_candidates(a, b, grid).times
        other = build_candidates(a, b + delta, grid).times
        changed[k] = not np.array_equal(original, other[other < b])
        last[k] = original[-1]
    return dict(changed=changed, last=last)


def leak_counts(chart, profile, start=0):
    """Count scored head gaps, excluding EOS, from start through chart.K-1.

    A full-occupancy force in a replayed order occurs if only its final lane
    releases at the final interior candidate, with all four held until then.
    A0 does not execute this force; its count is a common-chart diagnostic.
    """
    ks = np.arange(max(1, start), chart.K)
    d = chart.derived()
    held = d.held[ks]
    full = held.all(1)
    any_held = held.any(1)
    times = np.where(np.isfinite(d.release[ks]), d.release[ks], np.inf)
    last = profile['last'][ks]
    pressure = full & (times.min(1) == last)
    forced = np.zeros(len(ks), int)
    for lane in (3, 0):
        others = [j for j in range(4) if j != lane]
        forced += full & (times[:, lane] == last) & (times[:, others] > last[:, None]).all(1)
    changed = profile['changed'][ks]
    return dict(head_gaps=len(ks), changed_prefix_gaps=int(changed.sum()),
        held_gaps=int(any_held.sum()), changed_prefix_held_gaps=int((changed & any_held).sum()),
        full_occupancy_gaps=int(full.sum()), completion_pressure_gaps=int(pressure.sum()),
        forced_events_two_orders=int(forced.sum()), replay_orders=2)
