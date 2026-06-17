from __future__ import annotations

import pytest
import torch

from pulsefield_model.models.mapper.v3.factor_target import (
    BOS_INPUT_KIND,
    END_KIND,
    EVENT_KIND,
    IGNORE_INDEX,
    collate_delta_event_factor_targets,
    delta_event_factor_target_from_v3_tokens,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


def _time_shift_tokens(vocab: MapperV3Vocab, delta_ms: int) -> list[int]:
    return [vocab.time_shift_token_id(piece) for piece in vocab.decompose_time_shift_delta(delta_ms)]


def test_delta_event_factor_target_builds_shifted_rows_and_reconstructs() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        *_time_shift_tokens(vocab, 80),
        vocab.event_token_id_from_signature("T..."),
        *_time_shift_tokens(vocab, 160),
        vocab.event_token_id_from_signature(".T.."),
        vocab.eos_id,
    ]

    target = delta_event_factor_target_from_v3_tokens(
        tokens,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    )

    assert target.target_kind == (EVENT_KIND, EVENT_KIND, END_KIND)
    assert target.target_delta == (8, 16, IGNORE_INDEX)
    assert target.target_end_gap == (IGNORE_INDEX, IGNORE_INDEX, 76)
    assert target.input_kind == (BOS_INPUT_KIND, EVENT_KIND, EVENT_KIND)
    assert target.input_delta == (0, 8, 16)
    assert target.input_kind[-1] != END_KIND
    assert target.event_row_count == 2
    assert target.end_row_count == 1
    assert target.reconstruction_mismatch_count == 0


def test_delta_event_factor_target_rejects_out_of_range_end_gap() -> None:
    vocab = MapperV3Vocab()

    with pytest.raises(ValueError, match="end gap"):
        delta_event_factor_target_from_v3_tokens(
            [],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=9000,
            chart_end_ms=9000,
            is_full_chart_end=False,
        )


def test_collate_delta_event_factor_targets_pads_rows_and_counts() -> None:
    vocab = MapperV3Vocab()
    first = delta_event_factor_target_from_v3_tokens(
        [
            *_time_shift_tokens(vocab, 80),
            vocab.event_token_id_from_signature("T..."),
            vocab.eos_id,
        ],
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    ).as_tensor_mapping()
    second = delta_event_factor_target_from_v3_tokens(
        [
            *_time_shift_tokens(vocab, 40),
            vocab.event_token_id_from_signature("T..."),
            *_time_shift_tokens(vocab, 40),
            vocab.event_token_id_from_signature(".T.."),
            vocab.eos_id,
        ],
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=1000,
        chart_end_ms=1000,
        is_full_chart_end=False,
    ).as_tensor_mapping()

    batch = collate_delta_event_factor_targets([first, second])

    assert batch["target_kind"].shape == torch.Size([2, 3])
    assert batch["row_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["target_kind"][0].tolist() == [EVENT_KIND, END_KIND, IGNORE_INDEX]
    assert batch["target_kind"][1].tolist() == [EVENT_KIND, EVENT_KIND, END_KIND]
    assert batch["row_count"].tolist() == [2, 3]
    assert batch["event_row_count"].tolist() == [1, 2]
    assert batch["end_row_count"].tolist() == [1, 1]
    assert batch["reconstruction_mismatch_count"].tolist() == [0, 0]
