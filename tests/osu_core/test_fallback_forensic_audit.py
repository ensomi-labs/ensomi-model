import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pulsefield_model.osu_core.fallback_forensic_audit import (
    _baseline_tokens,
    _boundary_bucket,
    _group_class,
    _lane_erased_tokens,
    _skeleton_tokens,
    audit_fallback_forensics,
)


class FallbackForensicAuditTests(unittest.TestCase):
    def test_token_views_keep_rhythm_but_erase_lane_identity(self) -> None:
        row = _row(
            [
                _group(0, tap=3, start=4, end=0),
                _group(24, tap=0, start=0, end=4),
            ],
        )

        self.assertEqual(_baseline_tokens(row), ["C0", "D:0:3:4:0", "D:24:0:0:4"])
        self.assertEqual(_lane_erased_tokens(row), ["C0", "L:0:T2:S1:E0", "L:24:T0:S0:E1"])
        self.assertEqual(_skeleton_tokens(row), ["C0", "K:0:tap_ln_start:A3plus", "K:24:ln_end:A1"])

    def test_group_class_and_boundary_buckets_are_stable(self) -> None:
        self.assertEqual(_group_class(1, 0, 0), "tap")
        self.assertEqual(_group_class(1, 2, 0), "tap_ln_start")
        self.assertEqual(_group_class(0, 2, 4), "ln_start_ln_end")
        self.assertEqual(_boundary_bucket(0), "exact_start")
        self.assertEqual(_boundary_bucket(6), "near_start_le12")
        self.assertEqual(_boundary_bucket(48), "middle")
        self.assertEqual(_boundary_bucket(90), "near_end_le12")

    def test_small_audit_writes_report_and_tables(self) -> None:
        rows = []
        for source, split in enumerate(["train", "train", "valid", "test"], start=1):
            rows.append(
                _cache_row(
                    source=source,
                    split=split,
                    chunk=0,
                    groups=[
                        _group(0, tap=1, start=0, end=0),
                        _group(24, tap=2, start=0, end=0),
                    ],
                )
            )
            rows.append(
                _cache_row(
                    source=source,
                    split=split,
                    chunk=1,
                    groups=[
                        _group(0, tap=0, start=1, end=0),
                        _group(48, tap=0, start=0, end=1),
                    ],
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            csv_path = root / "tables.csv"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_fallback_forensics(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                tables_csv_path=csv_path,
                motif_vocab_size=4,
                motif_min_n=1,
                motif_max_n=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("baseline_splits", report)
            self.assertIn("forensic_summary", report)
            self.assertEqual(report["invalid_transition_count"], 0)
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)


def _row(groups: list[dict[str, int | str]]) -> SimpleNamespace:
    return SimpleNamespace(bar_phase_half=0, groups_json=json.dumps(groups, separators=(",", ":"), sort_keys=True))


def _cache_row(*, source: int, split: str, chunk: int, groups: list[dict[str, int | str]]) -> dict[str, object]:
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
        "start_beat_units": chunk * 96,
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
