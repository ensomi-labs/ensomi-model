from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_continuation_cluster_diagnostic import (
    parse_named_summary,
    run_continuation_cluster_diagnostic,
)


class MapperV3ContinuationClusterDiagnosticTests(unittest.TestCase):
    def test_parse_named_summary_requires_name(self) -> None:
        name, path = parse_named_summary("baseline=report.json")

        self.assertEqual(name, "baseline")
        self.assertEqual(path, Path("report.json"))

    def test_diagnostic_classifies_boundary_and_ranking_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            latest = root / "latest.json"
            baseline.write_text(json.dumps(_summary("case_a", starved=True, boundary=True)), encoding="utf-8")
            latest.write_text(json.dumps(_summary("case_a", starved=True, boundary=True)), encoding="utf-8")

            report = run_continuation_cluster_diagnostic(
                summaries={"baseline": baseline, "latest": latest},
                baseline_name="baseline",
            )

            self.assertEqual(report["matched_case_count"], 1)
            self.assertEqual(report["variants"]["baseline"]["failure_counts"]["starved"], 1)
            self.assertEqual(report["variants"]["baseline"]["failure_counts"]["boundary_clamped"], 1)
            self.assertEqual(report["decision"]["route"], "TEST_DECODE_TIMING_CALIBRATION")


def _summary(case_id: str, *, starved: bool, boundary: bool) -> dict[str, object]:
    last_times = [6860, 7260, 7660, 7860, 7960, 7990] if boundary else [9000, 9400, 9800]
    return {
        "experiment": "test",
        "aggregate": {"all_legal": True, "median_event_count_ratio": 0.5, "mean_f1_100ms": 0.4},
        "runs": [
            {
                "case_id": case_id,
                "legal": True,
                "completed": True,
                "dead_end": False,
                "max_tokens_exceeded": False,
                "generated_event_count": 4,
                "reference_event_count": 20,
                "event_count_ratio": 0.2,
                "second_window_event_share": 0.0 if starved else 0.5,
                "reference_second_window_event_share": 0.7,
                "dominant_spacing_ms": 400,
                "dominant_spacing_ratio": 0.95,
                "timing_match_100ms": 0.1,
                "first_12_generated_times": [410, 820, 1230],
                "last_12_generated_times": last_times,
                "logit_step_count": 10,
                "event_valid_step_count": 9,
                "event_top1_step_count": 2,
                "event_topk_step_count": 7,
                "best_event_rank_median": 2.0,
                "best_event_margin_median": -0.1,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
