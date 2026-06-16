import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.c3_side_stream_tokenization import (
    _decode_side_stream_for_source,
    _encode_side_stream_for_source,
    _signature_from_delta_tokens,
    audit_c3_side_stream_tokenization,
)
from pulsefield_model.osu_core.context_adaptive_fallback_codec_audit import (
    _CostPlan,
    _FallbackRecord,
    _LzSpanSelection,
)


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


def _cache_row(*, source: int, split: str, groups: list[dict[str, int | str]]) -> dict[str, object]:
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
        "chunk_index": 0,
        "segment_id": 0,
        "segment_chunk_index": 0,
        "bar_phase_half": 0,
        "start_beat_units": 0,
        "num_groups": len(groups),
        "num_events": sum(int(group["event_count"]) for group in groups),
        "raw_signature": signature,
        "groups_json": json.dumps(groups, separators=(",", ":"), sort_keys=True),
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


if __name__ == "__main__":
    unittest.main()
