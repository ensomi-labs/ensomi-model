import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.c3_side_stream_tokenization import (
    _decode_side_stream_for_source,
    _encode_side_stream_for_source,
    _signature_from_delta_tokens,
    audit_c3_mapper_window_sidecar,
    audit_c3_exact_window_assignment_comparison,
    audit_c3_side_stream_tokenization,
)
from pulsefield_model.osu_core.context_adaptive_fallback_codec_audit import (
    _CostPlan,
    _FallbackRecord,
    _LzSpanSelection,
)
from pulsefield_model.data.mapper_sparse_windows_v2_1 import load_c3_side_stream_token_sidecar


class C3SideStreamTokenizationTests(unittest.TestCase):
    def test_side_stream_decodes_exact_mirror_and_skeleton_refs(self) -> None:
        records = [
            _record(0, tap=1),
            _record(1, tap=1),
            _record(2, tap=8),
            _record(3, tap=2),
        ]
        spans = {
            records[1].id: _span(records, start=1, previous=0, match_type="exact"),
            records[2].id: _span(records, start=2, previous=0, match_type="mirror"),
            records[3].id: _span(records, start=3, previous=0, match_type="skeleton"),
        }

        side_tokens = _encode_side_stream_for_source(records, spans)
        decoded, errors = _decode_side_stream_for_source(records, side_tokens)

        self.assertEqual(errors, [])
        self.assertEqual([decoded[record.id] for record in records], [record.token for record in records])
        self.assertIn("REF|exact|1|1", side_tokens)
        self.assertIn("REF|mirror|2|1", side_tokens)
        self.assertIn("REF|skeleton|3|1", side_tokens)

    def test_signature_from_delta_tokens_preserves_order_signature(self) -> None:
        tokens = ["C0", "D:0:1:0:0", "D:24:3:0:0:OT1,T0"]

        self.assertEqual(_signature_from_delta_tokens(tokens), "A:0:1:0:0;A:24:3:0:0:OT1,T0")

    def test_small_audit_writes_report_and_roundtrips(self) -> None:
        rows = [
            _cache_row(source=1, split="train", groups=[_group(0, tap=1), _group(24, tap=2)]),
            _cache_row(source=2, split="train", groups=[_group(0, tap=1), _group(24, tap=2)]),
            _cache_row(source=3, split="valid", groups=[_group(0, tap=1), _group(24, tap=1)]),
            _cache_row(source=4, split="test", groups=[_group(0, tap=1), _group(24, tap=1)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_c3_side_stream_tokenization(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_size=2,
                motif_min_n=2,
                motif_max_n=2,
                lz_window_fallbacks=8,
                lz_max_span=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(report["reconstruction_guard"]["pass"])
            self.assertTrue(report["pass_criteria"]["p0_roundtrip_pass"])
            self.assertGreaterEqual(report["token_stats"]["main_stream_token_count"], 0)
            self.assertGreaterEqual(report["token_stats"]["side_stream_token_count"], 0)

    def test_mapper_window_sidecar_writes_loader_compatible_rows(self) -> None:
        rows = [
            _cache_row(source=1, split="train", groups=[_group(0, tap=1), _group(24, tap=2)], chunk_sort_ms=100.0),
            _cache_row(source=2, split="train", groups=[_group(0, tap=1), _group(24, tap=2)], chunk_sort_ms=8_100.0),
            _cache_row(source=3, split="valid", groups=[_group(0, tap=1), _group(24, tap=1)], chunk_sort_ms=16_100.0),
            _cache_row(source=4, split="test", groups=[_group(0, tap=1), _group(24, tap=1)], chunk_sort_ms=24_100.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            sidecar_path = root / "sidecar.json"
            report_path = root / "report.json"
            log_path = root / "log.md"
            dataset_root = root / "dataset"
            for source in (1, 2, 3, 4):
                _write_osu(
                    dataset_root / "0" / f"{source}.osu",
                    timing_lines=["0,500,4,2,0,80,1,0"],
                    hitobject_lines=[
                        "64,192,0,1,0,0:0:0:0:",
                        "192,192,250,1,0,0:0:0:0:",
                    ],
                )
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_c3_mapper_window_sidecar(
                chunk_cache_path=cache_path,
                dataset_root=dataset_root,
                sidecar_path=sidecar_path,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_size=2,
                motif_min_n=2,
                motif_max_n=2,
                lz_window_fallbacks=8,
                lz_max_span=2,
            )
            loaded = load_c3_side_stream_token_sidecar(sidecar_path)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(report["pass_criteria"]["sidecar_generation_pass"])
            self.assertTrue(report["pass_criteria"]["p5_exact_sidecar_generation_pass"])
            self.assertEqual(report["pass_criteria"]["window_anchor_mode"], "exact_group")
            self.assertEqual(report["anchor_report"]["parse_error_count"], 0)
            self.assertTrue(report["sidecar_stats"]["token_preservation_pass"])
            self.assertEqual(report["sidecar_stats"]["missing_anchor_token_count"], 0)
            self.assertEqual(report["loader_guard"]["loaded_window_count"], len(sidecar["windows"]))
            self.assertGreater(report["sidecar_stats"]["sidecar_token_count"], 0)
            self.assertEqual(report["sidecar_stats"]["sidecar_token_count"], report["loader_guard"]["loaded_token_count"])
            self.assertTrue(all(int(token_id) > 0 for window in sidecar["windows"] for token_id in window["token_ids"]))
            expected_path = (dataset_root / "0" / "1.osu").as_posix()
            self.assertIn(expected_path, loaded)
            self.assertIn(0, loaded[expected_path])
            source2_path = (dataset_root / "0" / "2.osu").as_posix()
            self.assertIn(0, loaded[source2_path])
            self.assertNotIn(8_000, loaded[source2_path])

    def test_exact_window_assignment_comparison_detects_chunk_sort_mismatch(self) -> None:
        rows = [
            _cache_row(source=1, split="train", groups=[_group(0, tap=1), _group(24, tap=2)], chunk_sort_ms=8_100.0),
            _cache_row(source=2, split="train", groups=[_group(0, tap=1), _group(24, tap=2)], chunk_sort_ms=100.0),
            _cache_row(source=3, split="valid", groups=[_group(0, tap=1), _group(24, tap=1)], chunk_sort_ms=100.0),
            _cache_row(source=4, split="test", groups=[_group(0, tap=1), _group(24, tap=1)], chunk_sort_ms=100.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for source in (1, 2, 3, 4):
                _write_osu(
                    dataset_root / "0" / f"{source}.osu",
                    timing_lines=["0,500,4,2,0,80,1,0"],
                    hitobject_lines=[
                        "64,192,0,1,0,0:0:0:0:",
                        "192,192,250,1,0,0:0:0:0:",
                    ],
                )
            cache_path = root / "chunks.parquet"
            report_path = root / "exact_report.json"
            log_path = root / "exact_log.md"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_c3_exact_window_assignment_comparison(
                chunk_cache_path=cache_path,
                dataset_root=dataset_root,
                report_path=report_path,
                result_log_path=log_path,
                motif_vocab_size=2,
                motif_min_n=2,
                motif_max_n=2,
                lz_window_fallbacks=8,
                lz_max_span=2,
                source_limit=4,
                mismatch_rate_fail_threshold=0.01,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertEqual(report["exact_anchor_report"]["parse_error_count"], 0)
            self.assertGreater(report["comparison"]["compared_token_count"], 0)
            self.assertGreater(report["comparison"]["mismatched_token_count"], 0)
            self.assertTrue(report["pass_criteria"]["kill_chunk_sort_anchoring"])
            self.assertFalse(report["pass_criteria"]["exact_window_comparison_pass"])


def _span(
    records: list[_FallbackRecord],
    *,
    start: int,
    previous: int,
    match_type: str,
) -> _LzSpanSelection:
    return _LzSpanSelection(
        source_row_index=records[start].source_row_index,
        split=records[start].split,
        start_record_id=records[start].id,
        previous_record_id=records[previous].id,
        start_index=start,
        previous_index=previous,
        length=1,
        distance=start - previous,
        match_type=match_type,
        payload_bits=1.0,
        mode_bits=1.0,
        residual_bits=1.0 if match_type == "skeleton" else 0.0,
        raw_bits=records[start].raw_bits,
        charged_bits=2.0,
        target_crosses_chunk=False,
        reference_crosses_chunk=False,
        target_crosses_segment=False,
        reference_crosses_segment=False,
        full_group_contiguous=True,
    )


def _record(record_id: int, *, tap: int) -> _FallbackRecord:
    return _FallbackRecord(
        id=record_id,
        split="test",
        source_row_index=1,
        beatmap_set_id=1,
        beatmap_id=10,
        segment_id=0,
        chunk_index=0,
        group_index=record_id + 1,
        absolute_units=record_id * 24,
        offset_units=record_id * 24,
        bar_phase_half=0,
        event_count=max(1, tap.bit_count()),
        token=f"D:0:{tap}:0:0",
        raw_bits=20.0,
        delta=0,
        tap_mask=tap,
        ln_start_mask=0,
        ln_end_mask=0,
        order_signature=".",
        pre_hold_mask=0,
        chunk_start_hold_mask=0,
        previous_skeleton="START",
        previous_fallback=False,
        history_skeletons=(),
        boundary_bucket="middle",
        group_class="tap",
        active_count=tap.bit_count(),
        active_hold_count=0,
        chord_ln_mixed=False,
        density_bin="density_mid|chord_low|ln_low",
        artist="a",
        title="t",
        version="v",
    )


def _cache_row(
    *,
    source: int,
    split: str,
    groups: list[dict[str, int | str]],
    chunk_sort_ms: float = 0.0,
) -> dict[str, object]:
    signature = ";".join(
        f"A:{group['offset_units']}:{group['tap_mask']}:{group['ln_start_mask']}:{group['ln_end_mask']}"
        for group in groups
    )
    return {
        "source_row_index": source,
        "split": split,
        "shard": "0",
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
        "chunk_index": 0,
        "segment_id": 0,
        "segment_chunk_index": 0,
        "bar_phase_half": 0,
        "start_beat_units": 0,
        "num_groups": len(groups),
        "num_events": sum(int(group["event_count"]) for group in groups),
        "raw_signature": signature,
        "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
        "chunk_sort_ms": chunk_sort_ms,
    }


def _group(offset: int, *, tap: int) -> dict[str, int | str]:
    return {
        "offset_units": offset,
        "tap_mask": tap,
        "ln_start_mask": 0,
        "ln_end_mask": 0,
        "order_signature": ".",
        "event_count": max(1, tap.bit_count()),
    }


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


if __name__ == "__main__":
    unittest.main()
