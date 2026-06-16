import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pulsefield_model.osu_core.context_adaptive_fallback_codec_audit import (
    _FallbackRecord,
    _build_explicit_selector_plan,
    _build_lz_span_plan,
    _build_oracle_plan,
    _c1_bit_context,
    _lz_candidates_by_record,
    _mirror_order_signature_4,
    _mirror_mask_4,
    audit_c3_lz_hardening,
    audit_context_adaptive_fallback_codec,
)


class ContextAdaptiveFallbackCodecAuditTests(unittest.TestCase):
    def test_small_audit_writes_report_and_marks_oracle_illegal(self) -> None:
        rows = [
            _cache_row(source=1, split="train", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)]),
            _cache_row(source=2, split="train", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)]),
            _cache_row(source=3, split="valid", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=4, start=0, end=0)]),
            _cache_row(source=4, split="test", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=4, start=0, end=0)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            report_path = root / "report.json"
            log_path = root / "log.md"
            comparison_path = root / "comparison.csv"
            diagnostics_path = root / "diagnostics.csv"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_context_adaptive_fallback_codec(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                comparison_csv_path=comparison_path,
                diagnostics_csv_path=diagnostics_path,
                motif_vocab_size=4,
                motif_min_n=2,
                motif_max_n=2,
                bootstrap_samples=4,
                lz_window_fallbacks=8,
                lz_max_span=2,
                patch_max_span=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertFalse(report["limited"])
            self.assertTrue(report["reconstruction_guard"]["pass"])
            self.assertGreater(len(report["reconstruction_guard"]["rows"]), 1)
            self.assertGreater(report["fallback_record_summary"]["split_fallback_literal_count"]["test"], 0)
            self.assertGreater(len(report["variants"]["b0_r0_delta"]["splits"]["test"]["mapset_bits"]), 0)
            self.assertFalse(report["variants"]["f0_free_oracle_upper_bound"]["legal_codec"])
            self.assertTrue(report["variants"]["s1_explicit_per_fallback_selector"]["legal_codec"])
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)

    def test_explicit_selector_charges_mode_for_every_fallback(self) -> None:
        records = [_record(0, raw_bits=10.0), _record(1, raw_bits=8.0)]

        plan = _build_explicit_selector_plan(
            records,
            candidate_bits={
                "raw_literal": (10.0, 8.0),
                "c1": (5.0, 12.0),
                "c2": (7.0, 7.0),
            },
            table_model_cost_bits=3.0,
        )

        self.assertTrue(plan.legal_codec)
        self.assertEqual(plan.table_model_cost_bits, 3.0)
        self.assertTrue(all(value > 0.0 for value in plan.mode_header_bits))
        self.assertGreater(plan.replacement_bits[0], plan.payload_bits[0])

    def test_free_oracle_is_not_a_legal_codec_and_has_no_mode_cost(self) -> None:
        records = [_record(0, raw_bits=10.0)]

        plan = _build_oracle_plan(records, candidate_bits={"c1": (2.0,), "c2": (3.0,)})

        self.assertFalse(plan.legal_codec)
        self.assertEqual(plan.replacement_bits[0], 2.0)
        self.assertEqual(plan.mode_header_bits[0], 0.0)

    def test_c1_bit_context_uses_decoder_known_state_not_current_masks(self) -> None:
        record = _record(0, raw_bits=10.0, tap=15, start=8, end=4, pre_hold=2)

        context = _c1_bit_context(record, plane="tap", lane=0, delta_bucket="13to24")

        self.assertIn("pre_lane_0", context)
        self.assertIn("pre_hold_1", context)
        self.assertNotIn("tap_15", context)
        self.assertNotIn("start_8", context)
        self.assertNotIn("end_4", context)

    def test_mirror_mask_uses_four_lane_reversal(self) -> None:
        self.assertEqual(_mirror_mask_4(0b0001), 0b1000)
        self.assertEqual(_mirror_mask_4(0b0110), 0b0110)
        self.assertEqual(_mirror_mask_4(0b1010), 0b0101)

    def test_mirror_order_signature_uses_four_lane_reversal(self) -> None:
        self.assertEqual(_mirror_order_signature_4("."), ".")
        self.assertEqual(_mirror_order_signature_4("T0,T3"), "T3,T0")
        self.assertEqual(_mirror_order_signature_4("E0,S1,T2"), "E3,S2,T1")

    def test_c3_lz_plan_verifies_prior_fallback_reference(self) -> None:
        records = [
            _record(0, raw_bits=20.0, tap=1, pre_hold=1),
            _record(1, raw_bits=20.0, tap=1, pre_hold=1),
        ]

        candidates = _lz_candidates_by_record(records, window_fallbacks=8, max_span=2)
        plan = _build_lz_span_plan(records, candidates, window_fallbacks=8, max_span=2)

        self.assertEqual(plan.reconstruction_mismatch_count, 0)
        self.assertEqual(plan.mode_codes[1], "lz_exact")
        self.assertLess(plan.replacement_bits[1], records[1].raw_bits)

    def test_c3_hardening_small_audit_writes_report_and_guard_rows(self) -> None:
        rows = [
            _cache_row(source=1, split="train", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)]),
            _cache_row(source=2, split="train", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=2, start=0, end=0)]),
            _cache_row(source=3, split="valid", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=1, start=0, end=0)]),
            _cache_row(source=4, split="test", groups=[_group(0, tap=1, start=0, end=0), _group(24, tap=1, start=0, end=0)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "chunks.parquet"
            report_path = root / "hardening.json"
            log_path = root / "hardening.md"
            comparison_path = root / "hardening.csv"
            diagnostics_path = root / "hardening_diagnostics.csv"
            pd.DataFrame(rows).to_parquet(cache_path, index=False)

            report = audit_c3_lz_hardening(
                chunk_cache_path=cache_path,
                report_path=report_path,
                result_log_path=log_path,
                comparison_csv_path=comparison_path,
                diagnostics_csv_path=diagnostics_path,
                motif_vocab_size=4,
                motif_min_n=2,
                motif_max_n=2,
                bootstrap_samples=4,
                lz_window_fallbacks=8,
                lz_max_span=2,
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertFalse(report["limited"])
            self.assertIn("c3_active_all_fixed_w256", report["variants"])
            self.assertTrue(report["reconstruction_guard"]["baseline_token_stream_guard"]["pass"])
            self.assertGreater(len(report["reconstruction_guard"]["rows"]), 0)


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
        "version": f"v{source}",
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


def _group(offset: int, *, tap: int, start: int, end: int) -> dict[str, int | str]:
    return {
        "offset_units": offset,
        "tap_mask": tap,
        "ln_start_mask": start,
        "ln_end_mask": end,
        "order_signature": ".",
        "event_count": max(1, tap.bit_count() + start.bit_count() + end.bit_count()),
    }


def _record(
    record_id: int,
    *,
    raw_bits: float,
    tap: int = 1,
    start: int = 0,
    end: int = 0,
    pre_hold: int = 0,
) -> _FallbackRecord:
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
        event_count=max(1, tap.bit_count() + start.bit_count() + end.bit_count()),
        token=f"D:0:{tap}:{start}:{end}",
        raw_bits=raw_bits,
        delta=24,
        tap_mask=tap,
        ln_start_mask=start,
        ln_end_mask=end,
        order_signature=".",
        pre_hold_mask=pre_hold,
        chunk_start_hold_mask=0,
        previous_skeleton="START",
        previous_fallback=False,
        history_skeletons=(),
        boundary_bucket="middle",
        group_class="tap",
        active_count=(tap | start | end).bit_count(),
        active_hold_count=pre_hold.bit_count(),
        chord_ln_mixed=bool(tap and (start or end)),
        density_bin="density_mid|chord_low|ln_low",
        artist="a",
        title="t",
        version="v",
    )


if __name__ == "__main__":
    unittest.main()
