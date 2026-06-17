from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_ce_density_trace_diagnostic import (
    analyze_density_case,
    run_ce_density_trace_diagnostic,
)


class MapperV3CEDensityTraceDiagnosticTests(unittest.TestCase):
    def test_analyze_density_case_counts_sparse_context_opportunities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trace_path = root / "trace.json"
            _write_trace_summary(
                trace_path,
                examples=[
                    _example(step=0, current_ms=8100, rank=2, margin=-0.5),
                    _example(step=1, current_ms=8200, rank=2, margin=-0.5),
                    _example(step=2, current_ms=9000, rank=2, margin=-0.5),
                    _example(step=3, current_ms=9100, argmax_kind="event", rank=1, margin=0.0),
                ],
                timepoints=[7900, 8950, 8970],
            )

            case = analyze_density_case(
                trace_oracle_result=_oracle_case("case_a", trace_path),
                positive_opportunity_count=2,
            )

        self.assertEqual(case["second_window_opportunity_count"], 3)
        self.assertEqual(case["sparse_context_opportunity_count"], 2)
        self.assertEqual(case["dense_context_opportunity_count"], 1)
        self.assertTrue(case["positive_density_signal"])

    def test_diagnostic_routes_positive_when_three_starved_cases_pass_and_control_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trace_paths = [root / f"case_{index}.json" for index in range(4)]
            for trace_path in trace_paths[:3]:
                _write_trace_summary(trace_path, examples=_opportunity_examples(count=10), timepoints=[])
            _write_trace_summary(trace_paths[3], examples=_opportunity_examples(count=2), timepoints=[])
            oracle_path = root / "oracle.json"
            oracle_path.write_text(
                json.dumps(
                    {
                        "case_results": [
                            _oracle_case("starved_a", trace_paths[0], role="low_bias_starved"),
                            _oracle_case("starved_b", trace_paths[1], role="low_bias_starved"),
                            _oracle_case("starved_c", trace_paths[2], role="low_bias_starved"),
                            _oracle_case("control", trace_paths[3], role="pass_like_control"),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_ce_density_trace_diagnostic(
                trace_oracle_summary_path=oracle_path,
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
            )

        self.assertEqual(summary["decision"]["route"], "TEST_DENSITY_AWARE_SELECTOR_OR_OBJECTIVE")
        self.assertEqual(summary["aggregate"]["positive_low_bias_starved_case_count"], 3)

    def test_diagnostic_routes_mutate_when_coverage_is_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trace_paths = [root / f"case_{index}.json" for index in range(5)]
            _write_trace_summary(trace_paths[0], examples=_opportunity_examples(count=10), timepoints=[])
            for trace_path in trace_paths[1:]:
                _write_trace_summary(trace_path, examples=_opportunity_examples(count=2), timepoints=[])
            oracle_path = root / "oracle.json"
            oracle_path.write_text(
                json.dumps(
                    {
                        "case_results": [
                            _oracle_case(f"starved_{index}", trace_paths[index], role="low_bias_starved")
                            for index in range(5)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_ce_density_trace_diagnostic(
                trace_oracle_summary_path=oracle_path,
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR")
        self.assertEqual(summary["aggregate"]["positive_low_bias_starved_case_count"], 1)


def _oracle_case(case_id: str, trace_path: Path, *, role: str = "low_bias_starved") -> dict[str, object]:
    return {
        "case": {
            "case_id": case_id,
            "role": role,
            "required_event_bias": 1.0,
            "classes": ["starved"] if role == "low_bias_starved" else ["pass_like"],
        },
        "trace_summary_path": trace_path.as_posix(),
    }


def _write_trace_summary(path: Path, *, examples: list[dict[str, object]], timepoints: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "rollout": {
                    "completed": True,
                    "dead_end": False,
                    "max_tokens_exceeded": False,
                    "timepoints": [{"time_ms": value, "lane_actions": ["TAP", "NONE", "NONE", "NONE"]} for value in timepoints],
                },
                "logit_diagnostics": {
                    "step_count": len(examples),
                    "examples": examples,
                },
            }
        ),
        encoding="utf-8",
    )


def _example(
    *,
    step: int,
    current_ms: int,
    argmax_kind: str = "time_shift",
    rank: int = 2,
    margin: float = -0.5,
) -> dict[str, object]:
    return {
        "step": step,
        "current_ms": current_ms,
        "argmax_kind": argmax_kind,
        "argmax_token": "TS_100" if argmax_kind != "event" else "EV_0001",
        "best_event_token": "EV_0001",
        "best_event_rank": rank,
        "best_event_margin_vs_argmax": margin,
    }


def _opportunity_examples(*, count: int) -> list[dict[str, object]]:
    return [_example(step=index, current_ms=8100 + 100 * index) for index in range(count)]


if __name__ == "__main__":
    unittest.main()
