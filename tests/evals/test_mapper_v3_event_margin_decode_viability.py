from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_event_margin_decode_viability import (
    parse_named_summary,
    run_event_margin_decode_viability,
)


class MapperV3EventMarginDecodeViabilityTests(unittest.TestCase):
    def test_parse_named_summary_requires_name(self) -> None:
        name, path = parse_named_summary("baseline=summary.json")

        self.assertEqual(name, "baseline")
        self.assertEqual(path, Path("summary.json"))

    def test_kills_global_bonus_when_starved_cases_need_large_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment": "test",
                        "aggregate": {"all_legal": True},
                        "runs": [
                            _run("case_a", margin=-6.0, second_share=0.0),
                            _run("case_b", margin=-5.0, second_share=0.0),
                            _run("case_c", margin=-1.0, second_share=0.5),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_event_margin_decode_viability(summaries={"baseline": path})

            self.assertEqual(report["variants"]["baseline"]["failure_counts"]["starved"], 2)
            self.assertEqual(report["variants"]["baseline"]["class_bias"]["starved"]["gt_4"], 2)
            self.assertEqual(report["decision"]["route"], "KILL_GLOBAL_EVENT_BONUS")

    def test_selects_simple_bonus_when_starved_cases_are_low_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment": "test",
                        "aggregate": {"all_legal": True},
                        "runs": [
                            _run("case_a", margin=-1.0, second_share=0.0),
                            _run("case_b", margin=-2.0, second_share=0.0),
                            _run("case_c", margin=-5.0, second_share=0.5),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_event_margin_decode_viability(summaries={"baseline": path})

            self.assertEqual(report["variants"]["baseline"]["class_bias"]["starved"]["le_4"], 2)
            self.assertEqual(report["decision"]["route"], "TEST_SIMPLE_EVENT_BONUS")


def _run(case_id: str, *, margin: float, second_share: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "legal": True,
        "completed": True,
        "dead_end": False,
        "max_tokens_exceeded": False,
        "event_count_ratio": 1.0,
        "second_window_event_share": second_share,
        "reference_second_window_event_share": 0.7,
        "dominant_spacing_ratio": 0.4,
        "best_event_rank_median": 2.0,
        "best_event_margin_median": margin,
        "logit_step_count": 10,
        "event_valid_step_count": 10,
        "event_top1_step_count": 3,
        "event_topk_step_count": 6,
    }


if __name__ == "__main__":
    unittest.main()
