from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_post_teacher_forced_exposure_route_synthesis import (
    SourceSpec,
    run_post_teacher_forced_exposure_route_synthesis,
)


class MapperV3PostTeacherForcedExposureRouteSynthesisTests(unittest.TestCase):
    def test_routes_to_generated_prefix_state_trace_when_shortcuts_are_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root)

            summary = run_post_teacher_forced_exposure_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "TEST_GENERATED_PREFIX_STATE_TRACE")
        self.assertTrue(summary["checks"]["teacher_forced_healthy"])
        self.assertTrue(summary["checks"]["exposure_trace_justified"])

    def test_reports_missing_required_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = SourceSpec("teacher_forced_time_shift", "teacher_forced", root / "missing.json")

            summary = run_post_teacher_forced_exposure_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=(missing,),
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_SYNTHESIS_INPUTS")
        self.assertEqual(summary["missing_required_count"], 1)

    def test_routes_to_timing_objective_when_teacher_forced_logits_are_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root, teacher_recall_at_5=0.4)

            summary = run_post_teacher_forced_exposure_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_TIMING_OBJECTIVE")
        self.assertFalse(summary["checks"]["teacher_forced_healthy"])


def _write_sources(root: Path, *, teacher_recall_at_5: float = 0.92) -> tuple[SourceSpec, ...]:
    payloads = {
        "teacher_forced_time_shift": {
            "decision": {"route": "TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC", "reason": "healthy"},
            "checks": {"faithful_global_context": True},
            "metrics": {
                "time_shift_row_count": 100,
                "recall_at_k": {"5": teacher_recall_at_5},
                "rank": {"median": 1.0},
                "argmax_top_shift_share": 0.32,
            },
        },
        "generated_continuation": {
            "decision": {"route": "TEST_DECODE_TIMING_CALIBRATION", "reason": "generated collapse"},
            "baseline_snapshot": {
                "failure_counts": {"starved": 11, "rigid": 12, "boundary_clamped": 10},
                "event_rank": {"mean_event_valid_step_ratio": 0.99},
            },
        },
        "decode_policy_sweep": {
            "decision": {"route": "KILL", "reason": "no passing decode policy"},
            "policy_summaries": [{"passes_positive_signal": False}],
        },
        "event_margin_viability": {
            "decision": {
                "route": "KILL_GLOBAL_EVENT_BONUS",
                "reason": "global bonus unsafe",
                "starved_gt4_share": 0.72,
            },
        },
        "conditioned_short_rollout": {
            "decision": {"route": "KILL", "reason": "event ratio guard failed"},
            "guard_results": {"median_event_count_ratio_in_range": False},
            "aggregate": {"starved_case_delta": -2, "mean_f1_delta": 0.05},
        },
        "post_time_shift_full32": {
            "decision": {"route": "MUTATE_TIME_SHIFT_DISTANCE_AND_TEST_C3_TARGET_SIDE_STREAM", "reason": "failed full32"},
        },
        "c3_diminishing_returns": {
            "decision": {"route": "MUTATE_C3_MAPPER_PATH_TEST_V3_TIMING_LOGIT_AUDIT", "reason": "diminishing"},
        },
    }
    sources: list[SourceSpec] = []
    families = {
        "teacher_forced_time_shift": "teacher_forced",
        "generated_continuation": "generated_prefix",
        "decode_policy_sweep": "decode_policy",
        "event_margin_viability": "decode_policy",
        "conditioned_short_rollout": "training_objective",
        "post_time_shift_full32": "training_objective",
        "c3_diminishing_returns": "c3_route",
    }
    for key, payload in payloads.items():
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sources.append(SourceSpec(key, families[key], path))
    return tuple(sources)


if __name__ == "__main__":
    unittest.main()
