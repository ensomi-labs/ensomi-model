from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.evals.mapper_v3_training_comparison import (
    EXPECTED_V21_CONTRACT,
    EXPECTED_V3_CONTRACT,
    compare_mapper_training_reports,
    load_training_report_summary,
    write_comparison_report,
    write_summary_json,
)


class MapperV3TrainingComparisonTests(unittest.TestCase):
    def test_compare_reports_passes_when_contracts_and_v3_token_count_improve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v21_path = root / "v21.json"
            v3_path = root / "v3.json"
            _write_report(v21_path, contract=EXPECTED_V21_CONTRACT, eval_loss=5.0, valid_tokens=200)
            _write_report(v3_path, contract=EXPECTED_V3_CONTRACT, eval_loss=5.5, valid_tokens=125)

            v21 = load_training_report_summary(v21_path, label="v2.1")
            v3 = load_training_report_summary(v3_path, label="v3")
            summary = compare_mapper_training_reports(v21=v21, v3=v3)

        self.assertEqual(summary["decision"]["route"], "TEST_NEXT")
        self.assertAlmostEqual(summary["metrics"]["v3_eval_valid_token_ratio"], 0.625)
        self.assertAlmostEqual(summary["metrics"]["v3_eval_valid_token_reduction_ratio"], 0.375)

    def test_compare_reports_mutates_when_v3_token_count_does_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v21_path = root / "v21.json"
            v3_path = root / "v3.json"
            _write_report(v21_path, contract=EXPECTED_V21_CONTRACT, eval_loss=5.0, valid_tokens=100)
            _write_report(v3_path, contract=EXPECTED_V3_CONTRACT, eval_loss=4.0, valid_tokens=100)

            summary = compare_mapper_training_reports(
                v21=load_training_report_summary(v21_path, label="v2.1"),
                v3=load_training_report_summary(v3_path, label="v3"),
            )

        self.assertEqual(summary["decision"]["route"], "MUTATE")
        self.assertFalse(summary["checks"]["v3_eval_valid_tokens_lower"])

    def test_loader_rejects_missing_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            _write_report(path, contract=EXPECTED_V3_CONTRACT, eval_loss=5.0, valid_tokens=10)
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["final_eval_metrics"]["loss/total"]
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required metric"):
                load_training_report_summary(path, label="v3")

    def test_writers_emit_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v21_path = root / "v21.json"
            v3_path = root / "v3.json"
            summary_path = root / "summary.json"
            report_path = root / "report.md"
            _write_report(v21_path, contract=EXPECTED_V21_CONTRACT, eval_loss=5.0, valid_tokens=200)
            _write_report(v3_path, contract=EXPECTED_V3_CONTRACT, eval_loss=5.5, valid_tokens=125)
            summary = compare_mapper_training_reports(
                v21=load_training_report_summary(v21_path, label="v2.1"),
                v3=load_training_report_summary(v3_path, label="v3"),
            )

            write_summary_json(summary, summary_path)
            write_comparison_report(summary, report_path)

            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(saved["decision"]["route"], "TEST_NEXT")
        self.assertIn("Decision: `TEST_NEXT`", markdown)
        self.assertIn("v3 valid-token ratio", markdown)


def _write_report(path: Path, *, contract: str, eval_loss: float, valid_tokens: int) -> None:
    payload = {
        "run_name": path.stem,
        "completed_steps": 1,
        "is_complete": True,
        "parameter_count": 123,
        "training_config": {
            "mapper_token_contract": contract,
            "dataset": {
                "mapper_token_contract": contract,
                "train_window_count": 1,
                "eval_window_count": 1,
                "source_window_count": 2,
            },
        },
        "final_eval_metrics": {
            "loss/total": float(eval_loss),
            "token/valid_count": float(valid_tokens),
        },
        "final_train_metrics": {
            "loss/total": float(eval_loss) + 0.1,
            "token/valid_count": float(valid_tokens),
        },
        "last_train_metrics": {
            "loss/total": float(eval_loss) + 0.2,
            "token/valid_count": float(valid_tokens),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
