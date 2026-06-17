from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_low_bias_trace_oracle import (
    analyze_trace_summary,
    run_low_bias_trace_oracle,
    select_trace_cases,
    summarize_trace_oracle,
)


class MapperV3LowBiasTraceOracleTests(unittest.TestCase):
    def test_select_trace_cases_picks_low_bias_and_controls(self) -> None:
        margin_summary = {
            "baseline": "baseline",
            "variants": {
                "baseline": {
                    "case_rows": [
                        _case_row("low_a", starved=True, pass_like=False, required_bias=1.0, second_share=0.05),
                        _case_row("low_b", starved=True, pass_like=False, required_bias=1.5, second_share=0.06),
                        _case_row("high", starved=True, pass_like=False, required_bias=7.0, second_share=0.04),
                        _case_row("pass", starved=False, pass_like=True, required_bias=3.0, second_share=0.50),
                    ]
                }
            },
        }
        manifest = [{"case_id": case_id, "summary_path": f"{case_id}.json"} for case_id in ("low_a", "low_b", "high", "pass")]

        selected = select_trace_cases(margin_summary=margin_summary, manifest=manifest, low_bias_limit=2)

        self.assertEqual([row["case_id"] for row in selected], ["low_a", "low_b", "high", "pass"])
        self.assertEqual([row["role"] for row in selected], [
            "low_bias_starved",
            "low_bias_starved",
            "high_bias_starved_control",
            "pass_like_control",
        ])

    def test_select_trace_cases_accepts_custom_threshold_without_duplicate_control(self) -> None:
        margin_summary = {
            "baseline": "baseline",
            "variants": {
                "baseline": {
                    "case_rows": [
                        _case_row("low", starved=True, pass_like=False, required_bias=1.0, second_share=0.05),
                        _case_row("medium", starved=True, pass_like=False, required_bias=3.5, second_share=0.04),
                        _case_row("pass", starved=False, pass_like=True, required_bias=1.0, second_share=0.50),
                    ]
                }
            },
        }
        manifest = [{"case_id": case_id, "summary_path": f"{case_id}.json"} for case_id in ("low", "medium", "pass")]

        selected = select_trace_cases(
            margin_summary=margin_summary,
            manifest=manifest,
            low_bias_limit=5,
            low_bias_threshold=4.0,
            high_bias_threshold=3.0,
        )

        self.assertEqual([row["case_id"] for row in selected], ["low", "medium", "pass"])
        self.assertEqual([row["role"] for row in selected], ["low_bias_starved", "low_bias_starved", "pass_like_control"])

    def test_run_oracle_accepts_runs_manifest_and_records_thresholds(self) -> None:
        margin_summary = {
            "baseline": "baseline",
            "variants": {
                "baseline": {
                    "case_rows": [
                        _case_row("medium", starved=True, pass_like=False, required_bias=3.5, second_share=0.04),
                        _case_row("pass", starved=False, pass_like=True, required_bias=1.0, second_share=0.50),
                    ]
                }
            },
        }
        manifest_summary = {
            "runs": [
                {"case_id": "medium", "summary_path": "medium.json"},
                {"case_id": "pass", "summary_path": "pass.json"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            margin_path = tmp / "margin.json"
            manifest_path = tmp / "manifest_summary.json"
            summary_path = tmp / "summary.json"
            report_path = tmp / "report.md"
            margin_path.write_text(json.dumps(margin_summary), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest_summary), encoding="utf-8")

            summary = run_low_bias_trace_oracle(
                margin_summary_path=margin_path,
                manifest_path=manifest_path,
                summary_output_path=summary_path,
                report_output_path=report_path,
                work_dir=tmp / "work",
                low_bias_limit=5,
                low_bias_threshold=4.0,
                dry_run=True,
            )

        self.assertEqual(summary["thresholds"]["low_bias"], 4.0)
        self.assertEqual(summary["selection"]["case_ids"], ["medium", "pass"])

    def test_analyze_trace_summary_counts_second_window_opportunities(self) -> None:
        summary = {
            "rollout": {"completed": True, "dead_end": False, "max_tokens_exceeded": False, "timepoint_count": 10},
            "logit_diagnostics": {
                "step_count": 3,
                "examples": [
                    _example(step=0, current_ms=400, argmax_kind="time_shift", rank=2, margin=-1.0),
                    _example(step=1, current_ms=8200, argmax_kind="time_shift", rank=4, margin=-1.5),
                    _example(step=2, current_ms=8400, argmax_kind="event", rank=1, margin=0.0),
                ],
            },
        }

        analysis = analyze_trace_summary(summary)

        self.assertTrue(analysis["full_trace"])
        self.assertEqual(analysis["first_window_opportunity_count"], 1)
        self.assertEqual(analysis["second_window_opportunity_count"], 1)
        self.assertEqual(analysis["windows"]["second"]["step_count"], 2)
        self.assertFalse(analysis["positive_signal"])

    def test_summarize_trace_oracle_routes_positive(self) -> None:
        cases = [
            {"case_id": "a", "role": "low_bias_starved"},
            {"case_id": "b", "role": "low_bias_starved"},
            {"case_id": "c", "role": "low_bias_starved"},
        ]
        case_results = [
            _case_result("a", second_opportunities=10, positive=True),
            _case_result("b", second_opportunities=12, positive=True),
            _case_result("c", second_opportunities=0, positive=False),
        ]

        summary = summarize_trace_oracle(
            cases=cases,
            case_results=case_results,
            margin_summary_path=Path("margin.json"),
            manifest_path=Path("manifest.json"),
            work_dir=Path("work"),
            elapsed_s=0.1,
            dry_run=False,
        )

        self.assertEqual(summary["decision"]["route"], "TEST_SELECTIVE_EVENT_GATE")


def _case_row(
    case_id: str,
    *,
    starved: bool,
    pass_like: bool,
    required_bias: float,
    second_share: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "case_index": 0,
        "starved": starved,
        "pass_like": pass_like,
        "required_event_bias": required_bias,
        "best_event_margin_median": -required_bias,
        "best_event_rank_median": 2.0,
        "event_count_ratio": 1.0,
        "second_window_event_share": second_share,
        "dominant_spacing_ratio": 0.5,
        "classes": ["pass_like"] if pass_like else ["starved"],
    }


def _example(
    *,
    step: int,
    current_ms: int,
    argmax_kind: str,
    rank: int,
    margin: float,
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


def _case_result(case_id: str, *, second_opportunities: int, positive: bool) -> dict[str, object]:
    return {
        "case": {"case_id": case_id, "role": "low_bias_starved"},
        "analysis": {
            "full_trace": True,
            "second_window_opportunity_count": second_opportunities,
            "positive_signal": positive,
            "dead_end": False,
            "max_tokens_exceeded": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
