from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.c3_v3_target_complexity_comparison import (
    run_c3_v3_target_complexity_comparison,
    write_report,
)


class C3V3TargetComplexityComparisonTests(unittest.TestCase):
    def test_comparison_mutates_to_v3_when_c3_target_costs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_inputs(root)

            summary = run_c3_v3_target_complexity_comparison(
                c3_lz_hardening_path=paths["c3_lz"],
                c3_side_stream_path=paths["c3_side"],
                c3_exact_sidecar_path=paths["c3_exact"],
                c3_ordered_raw_path=paths["c3_ordered"],
                v3_full_dataset_path=paths["v3"],
            )

            self.assertEqual(summary["decision"]["route"], "MUTATE_TO_V3_GRAMMAR_REPAIR")
            self.assertTrue(summary["guard_results"]["v3_reconstruction_zero"])
            self.assertTrue(summary["guard_results"]["c3_codec_reconstruction_zero"])
            self.assertFalse(summary["guard_results"]["c3_combined_sequence_competitive"])
            self.assertFalse(summary["guard_results"]["c3_ordered_sequence_bounded"])
            self.assertFalse(summary["guard_results"]["c3_production_input_legal"])
            self.assertAlmostEqual(summary["c3"]["combined_to_baseline_token_ratio"], 1.5)
            self.assertAlmostEqual(summary["v3"]["full_dataset_token_reduction_ratio"], 0.2)

    def test_write_report_contains_decision_and_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_inputs(root)
            summary = run_c3_v3_target_complexity_comparison(
                c3_lz_hardening_path=paths["c3_lz"],
                c3_side_stream_path=paths["c3_side"],
                c3_exact_sidecar_path=paths["c3_exact"],
                c3_ordered_raw_path=paths["c3_ordered"],
                v3_full_dataset_path=paths["v3"],
            )
            report_path = root / "report.md"

            write_report(summary, report_path)

            text = report_path.read_text(encoding="utf-8")
            self.assertIn("MUTATE_TO_V3_GRAMMAR_REPAIR", text)
            self.assertIn("c3_combined_sequence_competitive", text)
            self.assertIn("production input legal", text)


def _write_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "c3_lz": root / "c3_lz.json",
        "c3_side": root / "c3_side.json",
        "c3_exact": root / "c3_exact.json",
        "c3_ordered": root / "c3_ordered.json",
        "v3": root / "v3.json",
    }
    _write_json(
        paths["c3_lz"],
        {
            "pass_criteria": {
                "selected_variant": "a2",
                "test_delta_bits_per_event_with_dictionary": -0.38,
                "same_song_filtered_delta_bits_per_event": -0.37,
                "bootstrap_95pct_below_zero_pass": True,
            },
            "reconstruction_guard": {"pass": True},
        },
    )
    _write_json(
        paths["c3_side"],
        {
            "reconstruction_guard": {"pass": True},
            "token_stats": {
                "combined_main_side_token_count": 1500,
                "baseline_encoded_token_count": 1000,
                "combined_to_baseline_token_ratio": 1.5,
                "main_stream_token_count": 700,
                "fallback_placeholder_count": 300,
                "side_stream_token_count": 800,
            },
            "span_stats": {
                "span_count": 100,
                "noncontiguous_main_stream_span_count": 40,
                "target_cross_chunk_span_count": 30,
            },
        },
    )
    _write_json(
        paths["c3_exact"],
        {
            "pass_criteria": {
                "p5_exact_sidecar_generation_pass": True,
                "loader_guard_pass": True,
            },
            "sidecar_stats": {
                "sidecar_token_count": 800,
                "window_count": 50,
                "window_with_tokens_rate": 0.9,
                "token_vocab_size": 200,
                "cap_sweep": {
                    "256": {
                        "truncated_window_rate": 0.0,
                        "overflow_token_rate": 0.0,
                    }
                },
            },
            "span_window_stats": {
                "target_cross_mapper_window_span_rate": 0.1,
                "reference_cross_mapper_window_span_rate": 0.1,
                "target_reference_same_window_span_rate": 0.3,
            },
        },
    )
    _write_json(
        paths["c3_ordered"],
        {
            "pass_criteria": {
                "reconstruction_pass": True,
                "raw_bits_reduction_pass": False,
                "sequence_expansion_bounded_pass": False,
            },
            "dataset": {
                "ordered_symbol_count": 3000,
                "token_count": 1000,
            },
            "sequence": {"ordered_to_exact_token_ratio": 3.0},
            "bits": {
                "exact_c3": {"bits_per_item": 10.0, "item_count": 1000},
                "ordered_field": {"total_bits": 20000.0},
                "raw_exact": {"bits_per_item": 8.0, "item_count": 300},
                "raw_ordered_field": {"total_bits": 9000.0},
            },
        },
    )
    _write_json(
        paths["v3"],
        {
            "audit": {
                "total": {
                    "window_count": 100,
                    "candidate_token_count": 800,
                    "baseline_token_count": 1000,
                    "token_reduction_ratio": 0.2,
                    "reconstruction_mismatches": 0,
                    "cross_window_ln_window_count": 20,
                },
                "eval": {
                    "baseline_token_count": 100,
                },
            },
            "comparison": {
                "eval_token_count": 80,
                "token_reduction_ratio": 0.2,
                "total_bit_reduction_ratio": 0.05,
                "total_bit_reduction": 50.0,
            },
            "vocab": {
                "v2_1_vocab_size": 37,
                "v3_vocab_size": 280,
                "v3_event_token_count": 255,
            },
            "decision": {"route": "TEST_NEXT"},
        },
    )
    return paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
