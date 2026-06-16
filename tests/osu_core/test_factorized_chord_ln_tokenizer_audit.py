import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.factorized_chord_ln_tokenizer_audit import (
    _reconstruct_delta_tokens,
    _residual_payload,
    _residual_tokens,
    audit_factorized_chord_ln_tokenizer,
)
from pulsefield_model.osu_core.fallback_forensic_audit import _skeleton_tokens


class FactorizedChordLnTokenizerAuditTests(unittest.TestCase):
    def test_residual_tokens_charge_exact_masks(self) -> None:
        row = _row([_group(0, tap=3, start=4, end=0), _group(24, tap=0, start=0, end=4)])

        self.assertEqual(_skeleton_tokens(row), ["C0", "K:0:tap_ln_start:A3plus", "K:24:ln_end:A1"])
        self.assertEqual(_residual_tokens(row), ["R:3:4:0", "R:0:0:4"])
        self.assertEqual(_residual_payload("R:3:4:0"), (3, 4, 0, "."))

    def test_skeleton_plus_residual_reconstructs_delta_tokens(self) -> None:
        skeleton = ["C0", "K:0:tap_ln_start:A3plus", "K:24:ln_end:A1"]
        residual = ["R:3:4:0", "R:0:0:4"]

        self.assertEqual(_reconstruct_delta_tokens(skeleton, residual), ["D:0:3:4:0", "D:24:0:0:4"])

    def test_small_audit_writes_report(self) -> None:
        rows = []
        for source, split in enumerate(["train", "train", "valid", "test"], start=1):
            rows.append(
                _cache_row(
                    source=source,
                    split=split,
                    chunk=0,
                    groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)],
                )
            )
            rows.append(
                _cache_row(
                    source=source,
                    split=split,
                    chunk=1,
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

            report = audit_factorized_chord_ln_tokenizer(
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
            self.assertTrue(report["reconstruction"]["pass"])
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)


def _row(groups: list[dict[str, int | str]]):
    return type("Row", (), {"bar_phase_half": 0, "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True)})()


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
