import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.occupancy_state_tokenizer_audit import (
    _build_streams,
    _decode_stream_signatures,
    _release_code,
    _release_mask_from_code,
    audit_occupancy_state_tokenizer,
)


class OccupancyStateTokenizerAuditTests(unittest.TestCase):
    def test_release_code_uses_active_state_when_lossless(self) -> None:
        self.assertEqual(_release_code(0b0100, 0b0100), "EA")
        self.assertEqual(_release_mask_from_code(0b0100, "EA"), 0b0100)
        self.assertEqual(_release_code(0b1100, 0b0100), "EM4")
        self.assertEqual(_release_mask_from_code(0b1100, "EM4"), 0b0100)
        self.assertEqual(_release_code(0, 0b0100), "EX4")
        self.assertEqual(_release_mask_from_code(0, "EX4"), 0b0100)

    def test_stream_reconstructs_across_chunk_boundary(self) -> None:
        rows = [
            _cache_row(source=1, split="train", chunk=0, start=0, groups=[_group(90, tap=0, start=1, end=0)]),
            _cache_row(source=1, split="train", chunk=1, start=96, groups=[_group(12, tap=0, start=0, end=1)]),
        ]
        streams = _build_streams(pd.DataFrame(rows), variant="o1_occupancy_release_state")
        tokens = [token.token for token in streams[0].tokens]

        self.assertEqual(tokens, ["C:0:0", "O:90:0:1:E0", "C:96:0", "O:18:0:0:EA"])
        self.assertEqual(
            _decode_stream_signatures(tokens, variant="o1_occupancy_release_state"),
            ["A:90:0:1:0", "A:12:0:0:1"],
        )

    def test_small_audit_writes_report(self) -> None:
        rows = []
        for source, split in enumerate(["train", "train", "valid", "test"], start=1):
            rows.append(_cache_row(source=source, split=split, chunk=0, start=0, groups=[_group(0, tap=1, start=0, end=0)]))
            rows.append(
                _cache_row(
                    source=source,
                    split=split,
                    chunk=1,
                    start=96,
                    groups=[_group(0, tap=0, start=1, end=0), _group(48, tap=0, start=0, end=1)],
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            csv_path = root / "comparison.csv"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_occupancy_state_tokenizer(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                comparison_csv_path=csv_path,
                motif_vocab_size=4,
                motif_min_n=1,
                motif_max_n=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(report["reconstruction_guard"]["pass"])


def _cache_row(
    *,
    source: int,
    split: str,
    chunk: int,
    start: int,
    groups: list[dict[str, int | str]],
) -> dict[str, object]:
    signature = ";".join(
        f"A:{group['offset_units']}:{group['tap_mask']}:{group['ln_start_mask']}:{group['ln_end_mask']}"
        for group in groups
    )
    return {
        "source_row_index": source,
        "split": split,
        "beatmap_set_id": source,
        "beatmap_id": source * 10,
        "beatmap_path": f"{source}.osu",
        "title": "t",
        "artist": "a",
        "creator": "c",
        "version": "v",
        "difficulty": 1.0,
        "density_bin": "density_mid|chord_low|ln_low",
        "density_events_per_second": 1.0,
        "chord_ratio": 0.0,
        "ln_ratio": 0.0,
        "chunk_index": chunk,
        "segment_id": 0,
        "segment_chunk_index": chunk,
        "bar_phase_half": 0,
        "start_beat_units": start,
        "num_groups": len(groups),
        "num_events": sum(int(group["event_count"]) for group in groups),
        "raw_signature": signature,
        "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
    }


def _group(offset: int, *, tap: int, start: int, end: int) -> dict[str, int | str]:
    return {
        "offset_units": offset,
        "tap_mask": tap,
        "ln_start_mask": start,
        "ln_end_mask": end,
        "order_signature": ".",
        "event_count": max(1, tap.bit_count() + start.bit_count() + end.bit_count()),
    }


if __name__ == "__main__":
    unittest.main()
