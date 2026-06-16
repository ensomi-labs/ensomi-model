import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.selective_wildcard_motif_audit import (
    _build_prefixed_vocab,
    _delta_from_skeleton_residual,
    _encode_row,
    _residual_tokens,
    _signature_from_hybrid_tokens,
    audit_selective_wildcard_motifs,
)


class SelectiveWildcardMotifAuditTests(unittest.TestCase):
    def test_wildcard_reconstructs_from_charged_residual(self) -> None:
        self.assertEqual(_delta_from_skeleton_residual("K:24:tap_ln_start:A2", "R:4:8:0"), "D:24:4:8:0")

    def test_exact_match_precedes_wildcard(self) -> None:
        row = _row([_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)])
        exact_vocab = _build_prefixed_vocab([(("C0", "D:0:1:0:0"), 2, 2.0)], prefix="M")
        wildcard_vocab = _build_prefixed_vocab([(("C0", "K:0:tap:A1"), 2, 2.0)], prefix="W")

        encoded, covered, _spans = _encode_row(
            row,
            exact_trie=_trie(exact_vocab),
            wildcard_trie=_trie(wildcard_vocab),
        )

        self.assertEqual(encoded[:2], ["M0", "D:24:2:0:0"])
        self.assertNotIn("R:1:0:0", encoded)
        self.assertIn(1, covered)

    def test_hybrid_signature_decoder_consumes_residual_after_wildcard(self) -> None:
        exact_vocab = _build_prefixed_vocab([], prefix="M")
        wildcard_vocab = _build_prefixed_vocab([(("C0", "K:0:tap:A1"), 2, 2.0)], prefix="W")

        signature = _signature_from_hybrid_tokens(
            ["W0", "R:2:0:0"],
            exact_id_to_motif={},
            wildcard_id_to_motif={token_id: motif for motif, token_id in wildcard_vocab.motif_to_id.items()},
        )

        self.assertEqual(signature, "A:0:2:0:0")

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

            report = audit_selective_wildcard_motifs(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                comparison_csv_path=csv_path,
                exact_motif_vocab_size=4,
                wildcard_vocab_sizes=(1, 2),
                motif_min_n=1,
                motif_max_n=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(report["reconstruction"]["pass"])


def _trie(vocab):
    from pulsefield_model.osu_core.duration_ln_tokenization_audit import _build_motif_trie

    return _build_motif_trie(vocab.motif_to_id)


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
