from __future__ import annotations

import unittest

from pulsefield_model.evals.mapper_v3_conditioned_event_distribution_short_rollout import (
    decision_from_aggregate,
    gate_results_from_aggregate,
    select_short_rollout_cases,
)


class MapperV3ConditionedEventDistributionShortRolloutTests(unittest.TestCase):
    def test_select_short_rollout_cases_keeps_starved_and_controls(self) -> None:
        rows = [
            _row(0, "safe", starved=False, rigid=0.5, ratio=1.0),
            _row(1, "starved-a", starved=True, rigid=0.9, ratio=0.5),
            _row(2, "starved-b", starved=True, rigid=0.9, ratio=0.6),
            _row(3, "over", starved=False, rigid=0.5, ratio=1.8),
            _row(4, "rigid", starved=False, rigid=0.99, ratio=1.0),
        ]

        selected = select_short_rollout_cases({"runs": rows})

        self.assertEqual([row["case_id"] for row in selected], ["safe", "starved-a", "starved-b", "over", "rigid"])
        reasons = {row["case_id"]: row["selection_reasons"] for row in selected}
        self.assertEqual(reasons["starved-a"], ["ce_starved"])
        self.assertIn("pass_like_control", reasons["safe"])
        self.assertIn("overproduction_risk_control", reasons["over"])
        self.assertIn("rigid_risk_control", reasons["rigid"])

    def test_gate_results_pass_when_candidate_beats_starvation_without_flooding(self) -> None:
        aggregate = _aggregate(starved=4, rigid=6, median_ratio=1.05, starved_delta=-1)

        guards = gate_results_from_aggregate(aggregate)
        decision = decision_from_aggregate(aggregate, guard_results=guards)

        self.assertTrue(all(guards.values()))
        self.assertEqual(decision["route"], "TEST_FULL32_CONDITIONED_EVENT_DISTRIBUTION")

    def test_decision_kills_event_flooding(self) -> None:
        aggregate = _aggregate(starved=4, rigid=6, median_ratio=1.7, starved_delta=-1)

        guards = gate_results_from_aggregate(aggregate)
        decision = decision_from_aggregate(aggregate, guard_results=guards)

        self.assertFalse(guards["median_event_count_ratio_in_range"])
        self.assertEqual(decision["route"], "KILL")

    def test_decision_mutates_when_starvation_does_not_improve(self) -> None:
        aggregate = _aggregate(starved=5, rigid=6, median_ratio=1.05, starved_delta=0)

        guards = gate_results_from_aggregate(aggregate)
        decision = decision_from_aggregate(aggregate, guard_results=guards)

        self.assertFalse(guards["starved_below_ce_baseline_5"])
        self.assertEqual(decision["route"], "MUTATE_TO_TARGET_GRAMMAR_REPAIR")


def _row(index: int, case_id: str, *, starved: bool, rigid: float, ratio: float) -> dict[str, object]:
    return {
        "case_index": index,
        "case_id": case_id,
        "starved": starved,
        "dominant_spacing_ratio": rigid,
        "event_count_ratio": ratio,
    }


def _aggregate(*, starved: int, rigid: int, median_ratio: float, starved_delta: int) -> dict[str, object]:
    return {
        "dead_end_count": 0,
        "max_token_count": 0,
        "starved_case_delta": starved_delta,
        "candidate": {
            "case_count": 8,
            "all_legal": True,
            "starved_count": starved,
            "rigid_case_count": rigid,
            "median_event_count_ratio": median_ratio,
        },
    }


if __name__ == "__main__":
    unittest.main()
