import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_motif_quality_audit import (
    _motif_features,
    audit_beat_chunk_motif_quality,
)


class BeatChunkMotifQualityAuditTests(unittest.TestCase):
    def test_motif_features_tag_trill_and_grid(self) -> None:
        features = _motif_features(["D:6:1:0:0", "D:6:2:0:0", "D:6:1:0:0", "D:6:2:0:0"])

        self.assertIn("trill_like", features["tags"])
        self.assertIn("+6:T...", features["grid"])
        self.assertIn("+6:.T..", features["grid"])

    def test_quality_audit_writes_report_and_filtered_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chunk_path = root / "chunks.parquet"
            refinement_path = root / "refinement.json"
            beat_structure_path = root / "beat_structure.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            chunks = pd.DataFrame.from_records(
                [
                    _chunk_row(1, "train", 101, 1001, "Song", "Artist", "Easy"),
                    _chunk_row(2, "train", 102, 1002, "Song B", "Artist", "Hard"),
                    _chunk_row(3, "valid", 103, 1003, "Song C", "Artist", "Hard"),
                    _chunk_row(4, "test", 104, 1004, "Song D", "Artist", "Hard"),
                ]
            )
            chunks.to_parquet(chunk_path, index=False)
            refinement_path.write_text(
                json.dumps(
                    {
                        "pass_criteria": {
                            "research_pass": True,
                            "reconstruction_pass": True,
                            "reconstruction_mismatch_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame.from_records([_beat_structure_row(index) for index in range(1, 5)]).to_parquet(
                beat_structure_path,
                index=False,
            )

            report = audit_beat_chunk_motif_quality(
                chunk_cache_path=chunk_path,
                refinement_report_path=refinement_path,
                beat_structure_cache_path=beat_structure_path,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_size=2,
                top_motif_limit=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())

        self.assertTrue(report["guard"]["research_pass"])
        self.assertIn("filtered_compression", report)
        self.assertIn("top_motifs", report["motif_support"])
        self.assertGreaterEqual(len(report["motif_support"]["top_motifs"]), 1)


def _chunk_row(
    source_row_index: int,
    split: str,
    beatmap_set_id: int,
    beatmap_id: int,
    title: str,
    artist: str,
    version: str,
) -> dict[str, object]:
    groups = [
        {
            "offset_units": 0,
            "tap_mask": 1,
            "ln_start_mask": 0,
            "ln_end_mask": 0,
            "order_signature": ".",
            "event_count": 1,
            "onset_count": 1,
            "same_lane_multi_action_count": 0,
            "same_lane_duplicate_action_count": 0,
        },
        {
            "offset_units": 6,
            "tap_mask": 2,
            "ln_start_mask": 0,
            "ln_end_mask": 0,
            "order_signature": ".",
            "event_count": 1,
            "onset_count": 1,
            "same_lane_multi_action_count": 0,
            "same_lane_duplicate_action_count": 0,
        },
        {
            "offset_units": 12,
            "tap_mask": 1,
            "ln_start_mask": 0,
            "ln_end_mask": 0,
            "order_signature": ".",
            "event_count": 1,
            "onset_count": 1,
            "same_lane_multi_action_count": 0,
            "same_lane_duplicate_action_count": 0,
        },
    ]
    signature = "A:0:1:0:0;A:6:2:0:0;A:12:1:0:0"
    return {
        "source_row_index": source_row_index,
        "split": split,
        "beatmap_set_id": beatmap_set_id,
        "beatmap_id": beatmap_id,
        "beatmap_path": f"{beatmap_id}.osu",
        "title": title,
        "artist": artist,
        "creator": "Unit",
        "version": version,
        "difficulty": 3.0,
        "density_bin": "density_mid|chord_low|ln_low",
        "density_events_per_second": 4.0,
        "chord_ratio": 0.0,
        "ln_ratio": 0.0,
        "chunk_index": 0,
        "bar_phase_half": 0,
        "num_groups": 3,
        "num_events": 3,
        "raw_signature": signature,
        "canonical_signature": signature,
        "orientation": 0,
        "is_mirror_symmetric": False,
        "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
        "canonical_groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
    }


def _beat_structure_row(source_row_index: int) -> dict[str, object]:
    return {
        "source_row_index": source_row_index,
        "event_count": 3,
        "v2_tokenization_ok": True,
        "beat_joint_counter_json": json.dumps({"beat-a": 1, "beat-b": 1, "beat-c": 1}),
        "v2_token_counter_json": json.dumps({"BOS": 1, "LANE_1_TAP": 3, "EOS": 1}),
    }


if __name__ == "__main__":
    unittest.main()
