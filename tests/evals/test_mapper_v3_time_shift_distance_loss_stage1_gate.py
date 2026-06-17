from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.mapper_v3_time_shift_distance_loss_stage1_gate import (
    run_time_shift_distance_loss_stage1_gate,
    synthesize_decision,
)


class MapperV3TimeShiftDistanceLossStage1GateTests(unittest.TestCase):
    def test_gate_routes_to_tiny_training_when_synthetic_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = run_time_shift_distance_loss_stage1_gate(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
            )

        self.assertEqual(summary["decision"]["route"], "TEST_TIME_SHIFT_DISTANCE_TINY_TRAINING_GATE")
        self.assertEqual(
            summary["decision"]["next_card"],
            "target_grammar_v3_time_shift_distance_tiny_training_gate",
        )
        self.assertGreater(summary["probes"]["matching_vs_rigid"]["gap"], 0.0)
        self.assertTrue(summary["checks"]["gradient_nonzero"])
        self.assertTrue(summary["checks"]["disabled_default_ok"])

    def test_gate_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "summary.json"
            report_path = root / "report.md"

            run_time_shift_distance_loss_stage1_gate(
                summary_output_path=summary_path,
                report_output_path=report_path,
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("What Remains Unproven", report_path.read_text(encoding="utf-8"))

    def test_decision_kills_failed_separation(self) -> None:
        decision = synthesize_decision(
            {
                "matching_loss_finite": True,
                "rigid_loss_finite": True,
                "rigid_loss_greater_than_matching": False,
                "gradient_finite": True,
                "gradient_nonzero": True,
                "non_time_shift_rows_ignored": True,
                "disabled_default_ok": True,
                "enabled_metric_positive": True,
                "config_fields_present": True,
            }
        )

        self.assertEqual(decision["route"], "KILL_TIME_SHIFT_DISTANCE_PLUMBING")
        self.assertIn("rigid_loss_greater_than_matching", decision["reason"])


if __name__ == "__main__":
    unittest.main()
