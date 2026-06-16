import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_representation_audit import audit_beat_representation_index


def _write_osu(
    path: Path,
    *,
    timing_lines: list[str],
    hitobject_lines: list[str],
    circle_size: int = 4,
    mode: int = 3,
) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                f"Mode: {mode}",
                "",
                "[Difficulty]",
                f"CircleSize:{circle_size}",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        )
        + "\n",
        encoding="utf-8",
    )


class BeatRepresentationAuditTests(unittest.TestCase):
    def test_index_audit_writes_reusable_cache_and_report_without_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_set = dataset_root / "0" / "1001"
            beatmap_set.mkdir(parents=True)
            _write_osu(
                beatmap_set / "good.osu",
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,500,1,0,0:0:0:0:",
                    "192,192,1000,128,0,1500:0:0:0:0:",
                ],
            )
            index_path = root / "index.parquet"
            cache_path = root / "cache.parquet"
            report_path = root / "report.json"
            pd.DataFrame.from_records(
                [
                    _index_row("1001/good.osu", beatmap_id=1),
                    _index_row("1001/missing.osu", beatmap_id=2),
                    _index_row("../escape.osu", beatmap_id=3),
                ],
            ).to_parquet(index_path, index=False)

            report = audit_beat_representation_index(
                source_index_path=index_path,
                dataset_root=dataset_root,
                cache_path=cache_path,
                report_path=report_path,
            )
            cache_df = pd.read_parquet(cache_path)
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["source_row_count"], 3)
        self.assertEqual(report["processed_row_count"], 3)
        self.assertEqual(report["counts"]["conversion_success_count"], 1)
        self.assertEqual(report["counts"]["conversion_failure_count"], 2)
        self.assertEqual(report["failure_counts_by_error_type"], {"path_error": 2})
        self.assertEqual(report_payload["cache_path"], cache_path.as_posix())
        self.assertEqual(len(cache_df), 3)

        good = cache_df.loc[cache_df["beatmap_path"] == "1001/good.osu"].iloc[0]
        self.assertTrue(bool(good["ok"]))
        self.assertEqual(int(good["tap_count"]), 1)
        self.assertEqual(int(good["hold_count"]), 1)
        self.assertEqual(int(good["event_count"]), 3)
        self.assertTrue(bool(good["event_count_match"]))
        self.assertEqual(int(good["snap_error_bound_violation_count"]), 0)

        bad = cache_df.loc[cache_df["beatmap_path"] == "../escape.osu"].iloc[0]
        self.assertFalse(bool(bad["ok"]))
        self.assertEqual(bad["error_type"], "path_error")

    def test_audit_records_semantic_guard_failures_separately_from_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_set = dataset_root / "0" / "1001"
            beatmap_set.mkdir(parents=True)
            _write_osu(
                beatmap_set / "negative_hold.osu",
                timing_lines=[
                    "0,500,4,2,0,80,1,0",
                    "0,250,4,2,0,80,1,0",
                ],
                hitobject_lines=["192,192,1000,128,0,900:0:0:0:0:"],
            )
            index_path = root / "index.parquet"
            cache_path = root / "cache.parquet"
            pd.DataFrame.from_records([_index_row("1001/negative_hold.osu", beatmap_id=1)]).to_parquet(
                index_path,
                index=False,
            )

            report = audit_beat_representation_index(
                source_index_path=index_path,
                dataset_root=dataset_root,
                cache_path=cache_path,
                report_path=None,
            )
            cache_df = pd.read_parquet(cache_path)

        row = cache_df.iloc[0]
        self.assertTrue(bool(row["conversion_ok"]))
        self.assertFalse(bool(row["semantic_guard_ok"]))
        self.assertFalse(bool(row["ok"]))
        self.assertEqual(int(row["negative_hold_duration_count"]), 1)
        self.assertEqual(int(row["duplicate_red_timing_offset_count"]), 1)
        self.assertEqual(int(row["conflicting_duplicate_red_timing_offset_count"]), 1)
        self.assertEqual(report["counts"]["conversion_success_count"], 1)
        self.assertEqual(report["counts"]["semantic_guard_failure_count"], 1)
        self.assertEqual(report["counts"]["negative_hold_duration_row_count"], 1)


def _index_row(beatmap_path: str, *, beatmap_id: int) -> dict[str, object]:
    return {
        "shard": "0",
        "beatmap_set_id": 1001,
        "beatmap_path": beatmap_path,
        "beatmap_id": beatmap_id,
        "title": "Fixture",
        "artist": "Tests",
        "creator": "Unit",
        "version": Path(beatmap_path).stem,
        "difficulty": 3.0,
        "mode": 3,
        "key_count": 4,
    }


if __name__ == "__main__":
    unittest.main()
