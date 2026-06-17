from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_post_ce_route_synthesis import (
    SourceSpec,
    run_post_ce_route_synthesis,
)


class MapperV3PostCERouteSynthesisTests(unittest.TestCase):
    def test_synthesis_selects_conditioned_event_distribution_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root, tap_only_route="KILL")

            summary = run_post_ce_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE")
        self.assertTrue(summary["checks"]["simple_decode_family_killed"])
        self.assertEqual(summary["decision"]["next_card"], "target_grammar_v3_conditioned_event_distribution_objective")

    def test_synthesis_reports_missing_required_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = SourceSpec("v3_full_dataset", "representation", root / "missing.json")

            summary = run_post_ce_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=(missing,),
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE_SYNTHESIS_INPUTS")
        self.assertEqual(summary["missing_required_count"], 1)

    def test_synthesis_routes_open_decode_branch_when_tap_only_not_killed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sources = _write_sources(root, tap_only_route="TEST_NEXT")

            summary = run_post_ce_route_synthesis(
                summary_output_path=root / "summary.json",
                report_output_path=root / "report.md",
                sources=sources,
            )

        self.assertEqual(summary["decision"]["route"], "TEST_OPEN_DECODE_BRANCH")
        self.assertFalse(summary["checks"]["antirigid_decode_killed"])


def _write_sources(root: Path, *, tap_only_route: str) -> tuple[SourceSpec, ...]:
    payloads = {
        "v3_full_dataset": {
            "decision": {"route": "TEST_NEXT", "positive_gate_passed": True},
            "positive_gate_passed": True,
        },
        "v3_real_config": {"decision": "TEST_NEXT"},
        "v3_real_audio": {"decision": "TEST_NEXT", "rollout": {"timepoint_count": 0, "completed": True}},
        "density_loss": {"decision": {"route": "TEST_NEXT"}},
        "event_budget_0_05": {
            "decision": {"route": "MUTATE"},
            "aggregate": {"max_token_hit_case_count": 4, "median_event_count_ratio": 0.7},
        },
        "ce_weight": {"decision": {"route": "MUTATE"}, "aggregate": {"rigid_case_count": 11}},
        "ce_residual_cluster": {"decision": {"route": "TEST_EVENT_RANKING_CALIBRATION"}},
        "ce_margin": {"decision": {"route": "TEST_SELECTIVE_TRACE_ORACLE"}},
        "ce_selective_trace": {"decision": {"route": "TEST_DENSITY_AWARE_TRACE"}},
        "ce_density_trace": {
            "decision": {"route": "MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR"},
            "aggregate": {"positive_low_bias_starved_case_count": 1, "low_bias_starved_case_count": 5},
        },
        "ce_antirigid_full32": {
            "decision": {"route": "KILL"},
            "aggregate": {"candidate": {"median_event_count_ratio": 1.41}},
        },
        "ce_antirigid_soft_full32": {
            "decision": {"route": "KILL"},
            "aggregate": {"candidate": {"median_event_count_ratio": 1.41}},
        },
        "ce_tap_only_antirigid_full32": {
            "decision": {"route": tap_only_route},
            "aggregate": {"dead_end_count": 1, "candidate": {"median_event_count_ratio": 1.39}},
        },
        "c3_v3_complexity": {"decision": {"route": "MUTATE_TO_V3_GRAMMAR_REPAIR"}},
    }
    sources: list[SourceSpec] = []
    for key, payload in payloads.items():
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sources.append(SourceSpec(key, "test", path))
    return tuple(sources)


if __name__ == "__main__":
    unittest.main()
