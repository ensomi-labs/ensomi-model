from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_conditioned_event_distribution_gate import (
    run_conditioned_event_distribution_gate,
    synthesize_decision,
)


class MapperV3ConditionedEventDistributionGateTests(unittest.TestCase):
    def test_gate_routes_to_short_rollout_when_synthetic_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = run_conditioned_event_distribution_gate(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
            )

        self.assertEqual(summary["decision"]["route"], "TEST_SHORT_ROLLOUT_GATE")
        self.assertEqual(
            summary["decision"]["next_card"],
            "target_grammar_v3_conditioned_event_distribution_short_rollout_gate",
        )
        self.assertGreater(summary["checks"]["high_difficulty_overproduction_ratio"], 1.5)
        self.assertGreater(summary["checks"]["underproduction_gap"], 0.0)
        self.assertTrue(summary["checks"]["disabled_default_ok"])

    def test_gate_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "summary.json"
            report_path = root / "report.md"

            run_conditioned_event_distribution_gate(
                summary_output_path=summary_path,
                report_output_path=report_path,
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("What Remains Unproven", report_path.read_text(encoding="utf-8"))

    def test_decision_kills_failed_synthetic_conditioning(self) -> None:
        decision = synthesize_decision(
            {
                "high_difficulty_overproduction_finite": True,
                "high_difficulty_overproduction_conditioned": False,
                "underproduction_finite": True,
                "underproduction_penalized": True,
                "disabled_default_ok": True,
                "enabled_metric_positive": True,
                "config_fields_present": True,
            }
        )

        self.assertEqual(decision["route"], "KILL_CONDITIONED_EVENT_DISTRIBUTION_PLUMBING")
        self.assertIn("high_difficulty_overproduction_conditioned", decision["reason"])


if __name__ == "__main__":
    unittest.main()
