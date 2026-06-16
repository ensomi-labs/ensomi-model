import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.multiscale_context_audit import audit_multiscale_context


class MultiscaleContextAuditTests(unittest.TestCase):
    def test_boundary_signal_can_defer_shifted_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "forensic.json"
            tables_path = root / "tables.csv"
            out_path = root / "report.json"
            log_path = root / "log.md"
            decision_path = root / "decision.csv"
            report_path.write_text(
                json.dumps({"forensic_summary": {"test": {"boundary_near_fallback_share": 0.2}}}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    _row("test", "overall", "all", 1000, 200, 0.2),
                    _row("test", "boundary_bucket", "exact_start", 100, 5, 0.05),
                    _row("test", "boundary_bucket", "near_start_le12", 100, 20, 0.2),
                    _row("test", "boundary_bucket", "middle", 700, 160, 0.2286),
                    _row("test", "boundary_bucket", "near_end_le12", 100, 15, 0.15),
                    _row("valid", "overall", "all", 1000, 200, 0.2),
                    _row("valid", "boundary_bucket", "middle", 700, 160, 0.2286),
                ]
            ).to_csv(tables_path, index=False)

            report = audit_multiscale_context(
                forensic_report_path=report_path,
                forensic_tables_path=tables_path,
                report_path=out_path,
                result_log_path=log_path,
                decision_table_path=decision_path,
            )

            self.assertFalse(report["pass_criteria"]["research_pass"])
            self.assertFalse(report["pass_criteria"]["m1_shifted_window_recommended"])
            self.assertTrue(out_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(decision_path.exists())


def _row(split: str, table: str, bucket: str, groups: int, fallback: int, rate: float) -> dict[str, object]:
    return {
        "split": split,
        "table": table,
        "bucket": bucket,
        "group_count": groups,
        "event_count": groups,
        "fallback_group_count": fallback,
        "fallback_group_rate": rate,
        "fallback_event_count": fallback,
        "fallback_event_rate": rate,
        "lane_erased_recovered_fallback_count": 0,
        "lane_erased_recovered_fallback_rate": 0.0,
        "skeleton_recovered_fallback_count": 0,
        "skeleton_recovered_fallback_rate": 0.0,
        "mapset_count": 1,
        "examples": "[]",
    }


if __name__ == "__main__":
    unittest.main()
