import json
import unittest
from types import SimpleNamespace

from pulsefield_model.osu_core.duration_ln_tokenization_audit import (
    _PairInfo,
    _duration_bucket,
    _signature_from_transformed_tokens,
    _transform_row,
    _validate_variant_names,
)


class DurationLnTokenizationAuditTests(unittest.TestCase):
    def test_short_pair_emits_charged_residual_and_reconstructs(self) -> None:
        row = _row(
            [
                _group(0, tap=1, start=2, end=0),
                _group(12, tap=4, start=0, end=2),
            ]
        )
        pair = _pair(start=0, end=12, duration=12)

        transformed = _transform_row(
            row,
            variant_name="r3_short_pair",
            pair_lookup={1: {(pair.start_key, pair.end_key): pair}},
            start_tags={},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["P:0:short:1:2:0:_:4:0:2:_", "PX:12"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_short_pair_reconstructs_with_intervening_group(self) -> None:
        row = _row(
            [
                _group(0, tap=0, start=2, end=0),
                _group(6, tap=1, start=0, end=0),
                _group(12, tap=0, start=0, end=2),
            ]
        )
        pair = _pair(start=0, end=12, duration=12)

        transformed = _transform_row(
            row,
            variant_name="r3_short_pair",
            pair_lookup={1: {(pair.start_key, pair.end_key): pair}},
            start_tags={},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["P:0:short:0:2:0:_:0:0:2:_", "PX:12", "D:6:1:0:0"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_cross_chunk_pair_falls_back_to_delta_tokens(self) -> None:
        row = _row(
            [
                _group(0, tap=0, start=2, end=0),
                _group(12, tap=0, start=0, end=0),
            ]
        )
        pair = _PairInfo(start_key=(0, 0), end_key=(0, 96), duration_units=96, duration_bucket="long")

        transformed = _transform_row(
            row,
            variant_name="r3_short_pair",
            pair_lookup={1: {(pair.start_key, pair.end_key): pair}},
            start_tags={},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["D:0:0:2:0", "D:12:0:0:0"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_chained_ln_groups_are_not_skipped_twice(self) -> None:
        row = _row(
            [
                _group(0, tap=0, start=1, end=0),
                _group(12, tap=0, start=2, end=1),
                _group(24, tap=0, start=0, end=2),
            ]
        )
        first = _pair(start=0, end=12, duration=12)
        second = _pair(start=12, end=24, duration=12)

        transformed = _transform_row(
            row,
            variant_name="r3_short_pair",
            pair_lookup={1: {(first.start_key, first.end_key): first, (second.start_key, second.end_key): second}},
            start_tags={},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["P:0:short:0:1:0:_:0:2:1:_", "PX:12", "D:24:0:0:2"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_duration_start_tag_is_lossless(self) -> None:
        row = _row(
            [
                _group(0, tap=1, start=2, end=0),
                _group(12, tap=4, start=0, end=2),
            ]
        )

        transformed = _transform_row(
            row,
            variant_name="r1_duration_start",
            pair_lookup={},
            start_tags={1: {(0, 0): "short"}},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["D:0:1:2:0:Bshort", "D:12:4:0:2"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_exact_duration_pair_has_no_untracked_residual_token(self) -> None:
        row = _row([_group(0, tap=0, start=2, end=0), _group(12, tap=0, start=0, end=2)])
        pair = _pair(start=0, end=12, duration=12)

        transformed = _transform_row(
            row,
            variant_name="r3_exact_duration",
            pair_lookup={1: {(pair.start_key, pair.end_key): pair}},
            start_tags={},
            random_bucket=lambda _pair: "micro",
        )

        self.assertEqual([token.token for token in transformed], ["PE:0:12:0:2:0:_:0:0:2:_"])
        self.assertEqual(_signature_from_transformed_tokens([token.token for token in transformed]), row.raw_signature)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_variant_names(["r0_delta", "mapper_probe"])

    def test_duration_buckets_are_fixed_musical_thresholds(self) -> None:
        self.assertEqual(_duration_bucket(3), "micro")
        self.assertEqual(_duration_bucket(12), "short")
        self.assertEqual(_duration_bucket(48), "medium")
        self.assertEqual(_duration_bucket(49), "long")


def _row(groups: list[dict[str, int | str]]) -> SimpleNamespace:
    signature = ";".join(
        f"A:{group['offset_units']}:{group['tap_mask']}:{group['ln_start_mask']}:{group['ln_end_mask']}"
        for group in groups
    )
    return SimpleNamespace(
        source_row_index=1,
        split="train",
        beatmap_set_id=10,
        beatmap_id=20,
        chunk_index=0,
        segment_id=0,
        segment_chunk_index=0,
        bar_phase_half=0,
        start_beat_units=0,
        num_groups=len(groups),
        num_events=sum(int(group["event_count"]) for group in groups),
        raw_signature=signature,
        groups_json=json.dumps(groups, separators=(",", ":"), sort_keys=True),
    )


def _group(offset: int, *, tap: int, start: int, end: int) -> dict[str, int | str]:
    return {
        "offset_units": offset,
        "tap_mask": tap,
        "ln_start_mask": start,
        "ln_end_mask": end,
        "order_signature": ".",
        "event_count": max(1, tap.bit_count() + start.bit_count() + end.bit_count()),
    }


def _pair(*, start: int, end: int, duration: int) -> _PairInfo:
    return _PairInfo(
        start_key=(0, start),
        end_key=(0, end),
        duration_units=duration,
        duration_bucket=_duration_bucket(duration),
    )


if __name__ == "__main__":
    unittest.main()
