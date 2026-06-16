import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import (
    _delta_tokens,
    _split_integrity,
    audit_beat_chunk_tokenizer_refinement,
)


class BeatChunkTokenizerRefinementAuditTests(unittest.TestCase):
    def test_delta_tokens_preserve_order_signature(self) -> None:
        groups = [
            {
                "offset_units": 12,
                "tap_mask": 3,
                "ln_start_mask": 0,
                "ln_end_mask": 0,
                "order_signature": "T1,T0",
            },
            {
                "offset_units": 24,
                "tap_mask": 4,
                "ln_start_mask": 0,
                "ln_end_mask": 0,
                "order_signature": ".",
            },
        ]

        self.assertEqual(_delta_tokens(groups), ["D:12:3:0:0:OT1,T0", "D:12:4:0:0"])

    def test_split_integrity_reports_conflicts_and_duplicates(self) -> None:
        rows = [
            _chunk_row(1, "train", 10, 100, "a.osu", "Song", "Artist", "Easy"),
            _chunk_row(2, "test", 10, 101, "b.osu", "Song", "Artist", "Hard"),
            _chunk_row(3, "valid", 11, 101, "b.osu", "Other", "Artist", "Hard"),
        ]
        frame = pd.DataFrame.from_records(rows)

        report = _split_integrity(frame, pd.DataFrame())

        self.assertEqual(report["mapset_split_conflict_count"], 1)
        self.assertEqual(report["duplicate_beatmap_id_count"], 2)
        self.assertEqual(report["duplicate_beatmap_path_count"], 2)
        self.assertFalse(report["hard_pass"])

    def test_refinement_audit_scores_variants_and_split_matched_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chunk_path = root / "chunks.parquet"
            map_path = root / "maps.parquet"
            previous_report_path = root / "previous.json"
            beat_structure_path = root / "beat_structure.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            chunks = pd.DataFrame.from_records(
                [
                    _chunk_row(1, "train", 1001, 1, "train_a.osu", "Train", "A", "Easy"),
                    _chunk_row(2, "train", 1002, 2, "train_b.osu", "Train", "B", "Hard"),
                    _chunk_row(3, "valid", 1003, 3, "valid.osu", "Valid", "C", "Hard"),
                    _chunk_row(4, "test", 1004, 4, "test.osu", "Test", "D", "Hard"),
                ]
            )
            chunks.to_parquet(chunk_path, index=False)
            chunks.drop_duplicates("source_row_index").to_parquet(map_path, index=False)
            previous_report_path.write_text(
                json.dumps(
                    {
                        "parity_summary": {
                            "hard_gate_pass": True,
                            "event_count_mismatch_count": 0,
                            "event_order_mismatch_count": 0,
                            "lane_mismatch_count": 0,
                            "action_mismatch_count": 0,
                            "chord_group_mismatch_count": 0,
                            "ln_pair_mismatch_count": 0,
                            "chunk_boundary_reconstruction_mismatch_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame.from_records(
                [
                    _beat_structure_row(1, ok=True),
                    _beat_structure_row(2, ok=True),
                    _beat_structure_row(3, ok=True),
                    _beat_structure_row(4, ok=False),
                ]
            ).to_parquet(beat_structure_path, index=False)

            report = audit_beat_chunk_tokenizer_refinement(
                chunk_cache_path=chunk_path,
                map_cache_path=map_path,
                previous_report_path=previous_report_path,
                beat_structure_cache_path=beat_structure_path,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_sizes=(1, 2),
                top_motif_limit=2,
            )
            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())

        variants = {row["variant"] for row in report["variant_results"]}
        self.assertIn("motif_abs_mirror", variants)
        self.assertTrue(report["split_integrity"]["hard_pass"])
        self.assertTrue(report["split_matched_baselines"]["available"])
        self.assertEqual(report["split_matched_baselines"]["v2_tokenization_failure_count"], 1)
        self.assertEqual(report["pass_criteria"]["selection_policy"].split(";")[0], "choose motif tokenizer by lowest valid bits/event")
        self.assertIn("test_v2_success_only", report["valid_selected_motif"])
        self.assertTrue(report["reconstruction_checks"]["pass"])
        self.assertEqual(report["reconstruction_checks"]["mismatch_count"], 0)
        self.assertIn("bits_per_event_with_dictionary", report["valid_selected_motif"]["test"])


def _chunk_row(
    source_row_index: int,
    split: str,
    beatmap_set_id: int,
    beatmap_id: int,
    beatmap_path: str,
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
            "offset_units": 12,
            "tap_mask": 2,
            "ln_start_mask": 0,
            "ln_end_mask": 0,
            "order_signature": ".",
            "event_count": 1,
            "onset_count": 1,
            "same_lane_multi_action_count": 0,
            "same_lane_duplicate_action_count": 0,
        },
    ]
    signature = "A:0:1:0:0;A:12:2:0:0"
    return {
        "source_row_index": source_row_index,
        "split": split,
        "beatmap_set_id": beatmap_set_id,
        "beatmap_id": beatmap_id,
        "beatmap_path": beatmap_path,
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
        "num_groups": 2,
        "num_events": 2,
        "raw_signature": signature,
        "canonical_signature": signature,
        "orientation": 0,
        "is_mirror_symmetric": False,
        "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
        "canonical_groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
    }


def _beat_structure_row(source_row_index: int, *, ok: bool) -> dict[str, object]:
    return {
        "source_row_index": source_row_index,
        "event_count": 2,
        "v2_tokenization_ok": ok,
        "beat_joint_counter_json": json.dumps({"beat-token-a": 1, "beat-token-b": 1}),
        "v2_token_counter_json": json.dumps({"BOS": 1, "LANE_1_TAP": 2, "EOS": 1}) if ok else "{}",
    }


if __name__ == "__main__":
    unittest.main()
