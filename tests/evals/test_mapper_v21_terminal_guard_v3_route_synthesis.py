from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v21_terminal_guard_v3_route_synthesis import (
    SourceSpec,
    run_terminal_guard_v3_route_synthesis,
)


class MapperV21TerminalGuardV3RouteSynthesisTests(unittest.TestCase):
    def test_routes_to_v3_state_repair_with_v21_guard_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root)

            summary = run_terminal_guard_v3_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(
            summary["decision"]["route"],
            "TEST_V3_BOUNDARY_OR_SPACING_STATE_REPAIR_KEEP_V21_GUARD_OPT_IN",
        )
        self.assertTrue(summary["checks"]["terminal_guard"]["full32_passed"])
        self.assertTrue(summary["checks"]["same_slice"]["comparison_available"])
        self.assertEqual(summary["decision"]["next_card"], "target_grammar_v3_boundary_or_spacing_state_repair")

    def test_reports_missing_required_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = SourceSpec("v21_terminal_guard_full32", root / "missing.json")

            summary = run_terminal_guard_v3_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=(missing,),
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_SYNTHESIS_INPUTS")
        self.assertEqual(summary["missing_required_count"], 1)

    def test_kills_when_terminal_guard_full32_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root, terminal_all_legal=False)

            summary = run_terminal_guard_v3_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "KILL_V21_TERMINAL_GUARD_ROUTE")
        self.assertFalse(summary["checks"]["terminal_guard"]["full32_passed"])

    def test_mutates_when_v3_representation_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root, v3_reconstruction_mismatches=1)

            summary = run_terminal_guard_v3_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_V3_REPRESENTATION_AUDIT")
        self.assertFalse(summary["checks"]["v3_representation"]["ready"])


def _write_sources(
    root: Path,
    *,
    terminal_all_legal: bool = True,
    v3_reconstruction_mismatches: int = 0,
) -> tuple[SourceSpec, ...]:
    payloads = {
        "v21_terminal_guard_full32": _terminal_guard_payload(all_legal=terminal_all_legal),
        "v21_v3_fixed_slice": _fixed_slice_payload(),
        "v3_full_dataset": _v3_full_dataset_payload(reconstruction_mismatches=v3_reconstruction_mismatches),
        "v3_full_pipeline_smoke": {"decision": {"route": "TEST_NEXT"}},
        "v3_generated_prefix_trace": {
            "decision": {"route": "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR"},
            "aggregate": {
                "failure_class_counts": {"time_shift_repetition": 2},
                "mean_dominant_spacing_ratio": 0.95,
                "mean_event_count_ratio": 0.55,
                "mean_second_window_event_share": 0.04,
            },
        },
        "v3_post_teacher_forced_exposure": {
            "decision": {
                "route": "TEST_GENERATED_PREFIX_STATE_TRACE",
                "next_card": "target_grammar_v3_generated_prefix_state_trace_audit",
            }
        },
        "c3_v3_target_complexity": {
            "decision": {"route": "MUTATE_TO_V3_GRAMMAR_REPAIR"},
            "guard_results": {
                "c3_codec_reconstruction_zero": True,
                "c3_production_input_legal": False,
                "v3_no_future_lookup": True,
            },
            "c3": {
                "production_input_legal": False,
                "combined_to_baseline_token_ratio": 1.5,
                "ordered_to_exact_token_ratio": 3.1,
                "reference_cross_mapper_window_span_rate": 0.1,
                "test_delta_bits_per_event": -0.38,
            },
            "v3": {
                "decision_route": "TEST_NEXT",
                "full_dataset_reconstruction_mismatches": v3_reconstruction_mismatches,
                "eval_token_reduction_ratio": 0.2,
                "eval_total_bit_reduction_ratio": 0.05,
                "requires_c3_style_backreference_replay": False,
            },
        },
    }
    sources: list[SourceSpec] = []
    for key, payload in payloads.items():
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sources.append(SourceSpec(key, path))
    return tuple(sources)


def _terminal_guard_payload(*, all_legal: bool) -> dict[str, object]:
    return {
        "decision": {
            "route": "TEST_NEXT",
            "full32": True,
            "primary_case04_baseline_legal": all_legal,
            "primary_case05_anti_rigid_legal": all_legal,
        },
        "aggregate": {
            "run_count": 4,
            "case_count": 2,
            "all_candidate_legal": all_legal,
            "max_token_count": 0,
            "new_starved_count": 0,
            "mean_f1_delta": 0.01,
            "total_anti_rigid_blocked_count": 8,
            "baseline_guard": {"starved_count": 0},
            "anti_rigid_guard": {"starved_count": 0},
        },
        "case_results": [
            _terminal_case("a", "baseline_guard", legal=all_legal, f1=0.8, rigid=0.9, event_ratio=1.0),
            _terminal_case("a", "anti_rigid_guard", legal=all_legal, f1=0.7, rigid=0.7, event_ratio=1.1),
            _terminal_case("b", "baseline_guard", legal=all_legal, f1=0.6, rigid=1.0, event_ratio=1.2),
            _terminal_case("b", "anti_rigid_guard", legal=all_legal, f1=0.65, rigid=0.8, event_ratio=1.0),
        ],
    }


def _fixed_slice_payload() -> dict[str, object]:
    return {
        "aggregate": {"v3": {"legal_count": 2, "starved_count": 1, "mean_f1_100ms": 0.5}},
        "case_results": [
            {"case_id": "a", "v3_metrics": _metrics(f1=0.5, rigid=0.8, event_ratio=0.9, starved=False)},
            {"case_id": "b", "v3_metrics": _metrics(f1=0.4, rigid=0.7, event_ratio=0.8, starved=True)},
        ],
    }


def _v3_full_dataset_payload(*, reconstruction_mismatches: int) -> dict[str, object]:
    return {
        "decision": {"route": "TEST_NEXT"},
        "comparison": {
            "reconstruction_mismatches": reconstruction_mismatches,
            "token_reduction_ratio": 0.2,
            "total_bit_reduction_ratio": 0.05,
        },
    }


def _terminal_case(
    case_id: str,
    mode: str,
    *,
    legal: bool,
    f1: float,
    rigid: float,
    event_ratio: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "candidate_legal": legal,
        "candidate_metrics": _metrics(f1=f1, rigid=rigid, event_ratio=event_ratio, starved=False),
    }


def _metrics(*, f1: float, rigid: float, event_ratio: float, starved: bool) -> dict[str, object]:
    return {
        "dominant_spacing_ratio": rigid,
        "event_count_ratio": event_ratio,
        "starved": starved,
        "timing_match_100ms": {"f1": f1},
    }


if __name__ == "__main__":
    unittest.main()
