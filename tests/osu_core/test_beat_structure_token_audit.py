import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_structure_token_audit import audit_beat_structure_token_index


KNOWN_5MS_PATH = (
    "79839/Danny Baranowsky - The Battle of Lil' Slugger "
    "(Ch 1 Boss Extended Cut) (250bpm) (Staiain) [Insane].osu"
)


def _write_osu(
    path: Path,
    *,
    timing_lines: list[str],
    hitobject_lines: list[str],
    circle_size: int = 4,
    mode: int = 3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class BeatStructureTokenAuditTests(unittest.TestCase):
    def test_audit_writes_cache_report_with_hard_gates_and_fair_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / "1001" / "structure.osu"
            _write_osu(
                beatmap_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,500,1,0,0:0:0:0:",
                    "192,192,500,1,0,0:0:0:0:",
                    "320,192,1000,128,0,1500:0:0:0:0:",
                ],
            )
            index_path = root / "index.parquet"
            cache_path = root / "cache.parquet"
            report_path = root / "report.json"
            pd.DataFrame.from_records([_index_row("1001/structure.osu", beatmap_id=1)]).to_parquet(
                index_path,
                index=False,
            )

            report = audit_beat_structure_token_index(
                source_index_path=index_path,
                dataset_root=dataset_root,
                cache_path=cache_path,
                report_path=report_path,
                motif_ngram=2,
            )
            cache_df = pd.read_parquet(cache_path)
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))

        row = cache_df.iloc[0]
        self.assertTrue(bool(row["ok"]))
        self.assertTrue(bool(row["hard_gate_ok"]))
        self.assertEqual(int(row["event_count"]), 4)
        self.assertEqual(int(row["beat_record_count"]), 4)
        self.assertEqual(int(row["chord_group_count"]), 1)
        self.assertEqual(int(row["lane_mismatch_count"]), 0)
        self.assertEqual(int(row["action_mismatch_count"]), 0)
        self.assertEqual(int(row["chord_group_mismatch_count"]), 0)
        self.assertEqual(int(row["ln_pair_mismatch_count"]), 0)
        self.assertEqual(int(row["snap_bound_violation_count"]), 0)
        self.assertEqual(int(row["collapsed_ln_unflagged_count"]), 0)
        self.assertGreater(int(row["v2_token_count"]), 0)

        self.assertEqual(report["hard_gates"]["status"], "PASS")
        self.assertTrue(bool(report["hard_gates"]["pass"]))
        self.assertEqual(report["event_totals"]["event_count"], 4)
        self.assertEqual(report_payload["cache_path"], cache_path.as_posix())
        self.assertEqual(report["fair_comparison"]["current_v2_1"]["event_count"], 4)
        self.assertEqual(report["fair_comparison"]["beat_structure"]["event_count"], 4)
        self.assertGreater(report["fair_comparison"]["current_v2_1"]["entropy_bits"], 0.0)
        self.assertGreater(report["fair_comparison"]["beat_structure"]["entropy_bits"], 0.0)
        self.assertIn("density_controlled", report["motif_recurrence"])

    def test_unflagged_collapsed_ln_fails_hard_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / "1001" / "collapsed.osu"
            _write_osu(
                beatmap_path,
                timing_lines=["0,240,4,2,0,80,1,0"],
                hitobject_lines=["320,192,99387,128,0,99392:0:0:0:0:"],
            )
            index_path = root / "index.parquet"
            cache_path = root / "cache.parquet"
            pd.DataFrame.from_records([_index_row("1001/collapsed.osu", beatmap_id=999)]).to_parquet(
                index_path,
                index=False,
            )

            report = audit_beat_structure_token_index(
                source_index_path=index_path,
                dataset_root=dataset_root,
                cache_path=cache_path,
                report_path=None,
            )
            cache_df = pd.read_parquet(cache_path)

        row = cache_df.iloc[0]
        self.assertFalse(bool(row["hard_gate_ok"]))
        self.assertFalse(bool(row["ok"]))
        self.assertEqual(int(row["snapped_ln_nonpositive_count"]), 1)
        self.assertEqual(int(row["known_5ms_ln_canonicalization_exception_count"]), 0)
        self.assertEqual(int(row["collapsed_ln_unflagged_count"]), 1)
        self.assertEqual(report["hard_gates"]["status"], "FAIL")
        self.assertEqual(report["hard_gates"]["collapsed_ln_unflagged_count"], 1)

    def test_known_5ms_ln_exception_is_flagged_but_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / KNOWN_5MS_PATH
            _write_osu(
                beatmap_path,
                timing_lines=["0,240,4,2,0,80,1,0"],
                hitobject_lines=["320,192,99387,128,0,99392:0:0:0:0:"],
            )
            index_path = root / "index.parquet"
            cache_path = root / "cache.parquet"
            pd.DataFrame.from_records([_index_row(KNOWN_5MS_PATH, beatmap_id=222593)]).to_parquet(
                index_path,
                index=False,
            )

            report = audit_beat_structure_token_index(
                source_index_path=index_path,
                dataset_root=dataset_root,
                cache_path=cache_path,
                report_path=None,
            )
            cache_df = pd.read_parquet(cache_path)

        row = cache_df.iloc[0]
        self.assertTrue(bool(row["hard_gate_ok"]))
        self.assertTrue(bool(row["ok"]))
        self.assertEqual(int(row["snapped_ln_nonpositive_count"]), 1)
        self.assertEqual(int(row["known_5ms_ln_canonicalization_exception_count"]), 1)
        self.assertEqual(int(row["collapsed_ln_unflagged_count"]), 0)
        self.assertEqual(report["hard_gates"]["status"], "PASS")
        self.assertEqual(report["hard_gates"]["known_5ms_ln_canonicalization_exception_count"], 1)


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
