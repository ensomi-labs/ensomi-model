import numpy as np

from ensomi_model.r2.candidates import GridArrays
from ensomi_model.r2.features import Chart
from ensomi_model.r2.release_diagnostics import opportunity_profile, leak_counts


def chart(release=1016., second_lane=False):
    grid = GridArrays(np.array([0.]), np.array([500.]), np.array([0.]), np.array([4.]))
    actions = np.array([[2, 2, 2, 2], [4, 0, 0, 2 if second_lane else 0]])
    gap = np.full((2, 4), np.nan)
    gap[1, 0] = release
    if second_lane:
        gap[1, 3] = release
    return Chart(np.array([1000., 1020.]), 1100., grid, actions, gap)


def test_local_schedule_proxy_ignores_bos_and_clips_counterfactual_prefix():
    c = chart()
    profile = opportunity_profile(c.head_ms, c.grid)
    assert profile['changed'].tolist() == [False, True]
    assert profile['last'][1] == 1016.
    assert leak_counts(c, profile, start=2)['head_gaps'] == 0


def test_forced_replay_distinguishes_full_occupancy_from_a_forced_lane():
    c = chart(); profile = opportunity_profile(c.head_ms, c.grid)
    counts = leak_counts(c, profile)
    assert counts['full_occupancy_gaps'] == counts['completion_pressure_gaps'] == 1
    assert counts['forced_events_two_orders'] == 1  # lane zero is last only in the reverse order
    earlier = leak_counts(chart(release=1010.), profile)
    assert earlier['full_occupancy_gaps'] == 1
    assert earlier['completion_pressure_gaps'] == earlier['forced_events_two_orders'] == 0
    simultaneous = leak_counts(chart(second_lane=True), profile)
    assert simultaneous['completion_pressure_gaps'] == 1
    assert simultaneous['forced_events_two_orders'] == 0
