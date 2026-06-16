import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    _split_for_mapset,
    audit_beat_chunk_patterns,
    build_beat_chunk_pattern_rows,
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


class BeatChunkPatternAuditTests(unittest.TestCase):
    def test_chunk_boundaries_are_half_open_and_reconstruct_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / "1001" / "boundary.osu"
            _write_osu(
                beatmap_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "192,192,500,128,0,1500:0:0:0:0:",
                    "448,192,1000,1,0,0:0:0:0:",
                ],
            )
            source_cache_path = root / "beat_structure.parquet"
            chunk_cache_path = root / "chunks.parquet"
            map_cache_path = root / "maps.parquet"
            report_path = root / "report.json"
            pd.DataFrame.from_records([_source_row("1001/boundary.osu", beatmap_id=1)]).to_parquet(
                source_cache_path,
                index=False,
            )

            report = audit_beat_chunk_patterns(
                beat_structure_cache_path=source_cache_path,
                dataset_root=dataset_root,
                chunk_cache_path=chunk_cache_path,
                map_cache_path=map_cache_path,
                report_path=report_path,
                result_log_path=None,
                motif_vocab_sizes=(2,),
            )
            chunks = pd.read_parquet(chunk_cache_path)

        self.assertTrue(report["parity_summary"]["hard_gate_pass"])
        self.assertFalse(report["pass_criteria"]["heldout_available"])
        self.assertFalse(report["pass_criteria"]["research_pass"])
        self.assertEqual(report["chunk_construction_summary"]["event_count"], 4)
        self.assertEqual(report["chunk_construction_summary"]["group_count"], 4)
        self.assertEqual(len(chunks), 2)
        first = chunks.sort_values("chunk_index").iloc[0]
        second = chunks.sort_values("chunk_index").iloc[1]
        first_groups = json.loads(first["groups_json"])
        second_groups = json.loads(second["groups_json"])
        self.assertEqual([group["offset_units"] for group in first_groups], [0, 48])
        self.assertEqual([group["offset_units"] for group in second_groups], [0, 48])
        self.assertEqual(int(second["start_beat_units"]), 96)

    def test_mirror_canonical_signature_sets_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / "1001" / "mirror.osu"
            _write_osu(
                beatmap_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["448,192,0,1,0,0:0:0:0:"],
            )
            row = _source_row("1001/mirror.osu", beatmap_id=2)

            result = build_beat_chunk_pattern_rows(row, dataset_root=dataset_root)

        self.assertTrue(result.map_summary["hard_gate_ok"])
        self.assertEqual(len(result.chunk_rows), 1)
        chunk = result.chunk_rows[0]
        self.assertEqual(chunk["raw_signature"], "A:0:8:0:0")
        self.assertEqual(chunk["mirror_signature"], "A:0:1:0:0")
        self.assertEqual(chunk["canonical_signature"], "A:0:1:0:0")
        self.assertEqual(chunk["orientation"], 1)

    def test_empty_train_or_heldout_split_cannot_research_pass(self) -> None:
        train_mapset_id = _mapset_id_for_split("train")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_path = dataset_root / "0" / str(train_mapset_id) / "train_only.osu"
            _write_osu(
                beatmap_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "192,192,500,1,0,0:0:0:0:",
                    "320,192,1000,1,0,0:0:0:0:",
                ],
            )
            source_cache_path = root / "beat_structure.parquet"
            chunk_cache_path = root / "chunks.parquet"
            map_cache_path = root / "maps.parquet"
            pd.DataFrame.from_records(
                [_source_row(f"{train_mapset_id}/train_only.osu", beatmap_id=3, beatmap_set_id=train_mapset_id)]
            ).to_parquet(source_cache_path, index=False)

            report = audit_beat_chunk_patterns(
                beat_structure_cache_path=source_cache_path,
                dataset_root=dataset_root,
                chunk_cache_path=chunk_cache_path,
                map_cache_path=map_cache_path,
                report_path=None,
                result_log_path=None,
                motif_vocab_sizes=(2,),
            )

        self.assertTrue(report["pass_criteria"]["train_available"])
        self.assertFalse(report["pass_criteria"]["heldout_available"])
        self.assertFalse(report["pass_criteria"]["research_pass"])
        self.assertEqual(report["heldout_bits_event_comparison"]["best_test"], {})


def _source_row(beatmap_path: str, *, beatmap_id: int, beatmap_set_id: int = 1001) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_row_index": beatmap_id,
        "resolved_beatmap_path": "",
        "path_ok": True,
        "conversion_ok": True,
        "hard_gate_ok": True,
        "ok": True,
        "shard": "0",
        "beatmap_set_id": beatmap_set_id,
        "beatmap_id": beatmap_id,
        "beatmap_path": beatmap_path,
        "title": "Fixture",
        "artist": "Tests",
        "creator": "Unit",
        "version": Path(beatmap_path).stem,
        "difficulty": 3.0,
        "mode": 3,
        "key_count": 4,
        "density_bin": "density_mid|chord_low|ln_low",
        "density_events_per_second": 4.0,
        "chord_ratio": 0.0,
        "ln_ratio": 0.0,
        "v2_token_count": 0,
    }


def _mapset_id_for_split(split: str) -> int:
    for value in range(1, 10000):
        if _split_for_mapset(value) == split:
            return value
    raise AssertionError(f"could not find mapset id for split {split}")


if __name__ == "__main__":
    unittest.main()
