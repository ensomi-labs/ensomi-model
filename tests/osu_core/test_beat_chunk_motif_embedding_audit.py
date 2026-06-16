import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_motif_embedding_audit import (
    _class_support_preflight,
    _embedding_labels,
    audit_beat_chunk_motif_embeddings,
)
from pulsefield_model.osu_core.beat_chunk_motif_quality_audit import _motif_features


class BeatChunkMotifEmbeddingAuditTests(unittest.TestCase):
    def test_embedding_labels_capture_action_rhythm_and_motion(self) -> None:
        labels = _embedding_labels(["D:6:1:0:0", "D:6:2:0:0", "D:12:4:0:0"])

        self.assertEqual(labels["action_label"], "tap")
        self.assertEqual(labels["rhythm_label"], "6,6,12")
        self.assertEqual(labels["motion_label"], "1,1")

    def test_class_support_marks_ln_underpowered(self) -> None:
        metadata = [
            {
                "occurrence_count": 10,
                "quality_features": _motif_features(["D:6:1:0:0", "D:6:2:0:0"]),
            },
            {
                "occurrence_count": 3,
                "quality_features": _motif_features(["D:6:0:1:0", "D:6:0:0:1"]),
            },
        ]

        support = _class_support_preflight(metadata)

        self.assertEqual(support["ln_status"], "LN_UNDERPOWERED")
        self.assertEqual(support["powered_tag_family_count"], 0)

    def test_embedding_audit_writes_train_only_control_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chunk_path = root / "chunks.parquet"
            refinement_path = root / "refinement.json"
            motif_quality_path = root / "quality.json"
            report_path = root / "report.json"
            log_path = root / "log.md"
            chunks = pd.DataFrame.from_records(
                [
                    _chunk_row(1, "train", 101, 1001, 0, 1),
                    _chunk_row(1, "train", 101, 1001, 1, 2),
                    _chunk_row(2, "train", 102, 1002, 0, 1),
                    _chunk_row(2, "train", 102, 1002, 1, 2),
                    _chunk_row(3, "valid", 103, 1003, 0, 1),
                    _chunk_row(4, "test", 104, 1004, 0, 2),
                ]
            )
            chunks.to_parquet(chunk_path, index=False)
            guard_payload = {
                "pass_criteria": {
                    "research_pass": True,
                    "reconstruction_pass": True,
                },
            }
            refinement_path.write_text(json.dumps(guard_payload), encoding="utf-8")
            motif_quality_path.write_text(json.dumps({"pass_criteria": {"research_pass": True}}), encoding="utf-8")

            report = audit_beat_chunk_motif_embeddings(
                chunk_cache_path=chunk_path,
                refinement_report_path=refinement_path,
                motif_quality_report_path=motif_quality_path,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_size=4,
                embedding_vocab_size=2,
                embedding_dim=1,
                neighbor_k=1,
                eval_token_count=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())

        self.assertEqual(report["config"]["fit_split"], "train")
        self.assertIn("class_support", report)
        self.assertIn("matrix_health", report)
        self.assertIn("density_matched_tag", report["neighbor_diagnostics"].get("summary", {}))
        self.assertIn("frequency_matched_tag", report["neighbor_diagnostics"].get("summary", {}))

    def test_embedding_audit_rejects_non_train_fit_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chunk_path = root / "chunks.parquet"
            refinement_path = root / "refinement.json"
            motif_quality_path = root / "quality.json"
            pd.DataFrame.from_records(
                [
                    _chunk_row(1, "train", 101, 1001, 0, 1),
                    _chunk_row(2, "test", 102, 1002, 0, 2),
                ]
            ).to_parquet(chunk_path, index=False)
            refinement_path.write_text(
                json.dumps({"pass_criteria": {"research_pass": True, "reconstruction_pass": True}}),
                encoding="utf-8",
            )
            motif_quality_path.write_text(json.dumps({"pass_criteria": {"research_pass": True}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train split only"):
                audit_beat_chunk_motif_embeddings(
                    chunk_cache_path=chunk_path,
                    refinement_report_path=refinement_path,
                    motif_quality_report_path=motif_quality_path,
                    fit_split="test",
                )


def _chunk_row(
    source_row_index: int,
    split: str,
    beatmap_set_id: int,
    beatmap_id: int,
    chunk_index: int,
    pattern: int,
) -> dict[str, object]:
    if pattern == 1:
        groups = [
            _group(0, 1),
            _group(6, 2),
            _group(12, 1),
        ]
        signature = "A:0:1:0:0;A:6:2:0:0;A:12:1:0:0"
    else:
        groups = [
            _group(0, 4),
            _group(6, 8),
            _group(12, 4),
        ]
        signature = "A:0:4:0:0;A:6:8:0:0;A:12:4:0:0"
    return {
        "source_row_index": source_row_index,
        "split": split,
        "beatmap_set_id": beatmap_set_id,
        "beatmap_id": beatmap_id,
        "beatmap_path": f"{beatmap_id}.osu",
        "title": "Fixture",
        "artist": "Unit",
        "creator": "Unit",
        "version": "Hard",
        "difficulty": 3.0,
        "density_bin": "density_mid|chord_low|ln_low",
        "density_events_per_second": 4.0,
        "chord_ratio": 0.0,
        "ln_ratio": 0.0,
        "chunk_index": chunk_index,
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


def _group(offset_units: int, tap_mask: int) -> dict[str, object]:
    return {
        "offset_units": offset_units,
        "tap_mask": tap_mask,
        "ln_start_mask": 0,
        "ln_end_mask": 0,
        "order_signature": ".",
        "event_count": 1,
        "onset_count": 1,
        "same_lane_multi_action_count": 0,
        "same_lane_duplicate_action_count": 0,
    }


if __name__ == "__main__":
    unittest.main()
