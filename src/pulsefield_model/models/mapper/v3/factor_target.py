from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .replay import target_end_ms
from .vocab import MapperV3Vocab


EVENT_KIND = 0
END_KIND = 1
BOS_INPUT_KIND = 2
IGNORE_INDEX = -100
DEFAULT_DELTA_EVENT_FACTOR_TIME_CLASSES = 801


@dataclass(frozen=True)
class DeltaEventFactorTarget:
    input_kind: tuple[int, ...]
    input_delta: tuple[int, ...]
    input_signature: tuple[int, ...]
    input_end_gap: tuple[int, ...]
    target_kind: tuple[int, ...]
    target_delta: tuple[int, ...]
    target_signature: tuple[int, ...]
    target_end_gap: tuple[int, ...]
    event_row_count: int
    end_row_count: int
    reconstruction_mismatch_count: int

    @property
    def row_count(self) -> int:
        return len(self.target_kind)

    def as_tensor_mapping(self) -> dict[str, torch.Tensor]:
        return {
            "input_kind": torch.tensor(self.input_kind, dtype=torch.long),
            "input_delta": torch.tensor(self.input_delta, dtype=torch.long),
            "input_signature": torch.tensor(self.input_signature, dtype=torch.long),
            "input_end_gap": torch.tensor(self.input_end_gap, dtype=torch.long),
            "target_kind": torch.tensor(self.target_kind, dtype=torch.long),
            "target_delta": torch.tensor(self.target_delta, dtype=torch.long),
            "target_signature": torch.tensor(self.target_signature, dtype=torch.long),
            "target_end_gap": torch.tensor(self.target_end_gap, dtype=torch.long),
            "row_mask": torch.ones((self.row_count,), dtype=torch.bool),
            "row_count": torch.tensor(self.row_count, dtype=torch.long),
            "event_row_count": torch.tensor(self.event_row_count, dtype=torch.long),
            "end_row_count": torch.tensor(self.end_row_count, dtype=torch.long),
            "reconstruction_mismatch_count": torch.tensor(self.reconstruction_mismatch_count, dtype=torch.long),
        }


def delta_event_factor_target_from_v3_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    is_full_chart_end: bool,
    delta_class_count: int = DEFAULT_DELTA_EVENT_FACTOR_TIME_CLASSES,
    end_gap_class_count: int = DEFAULT_DELTA_EVENT_FACTOR_TIME_CLASSES,
) -> DeltaEventFactorTarget:
    resolved_target_end_ms = target_end_ms(
        write_start_ms=int(write_start_ms),
        write_end_ms=int(write_end_ms),
        chart_end_ms=int(chart_end_ms),
        is_full_chart_end=bool(is_full_chart_end),
    )
    event_index_by_signature = {
        vocab.event_signature(token_id): index
        for index, token_id in enumerate(vocab.event_token_ids)
    }
    current_ms = int(write_start_ms)
    anchor_ms = int(write_start_ms)
    original_event_times: list[int] = []
    reconstructed_event_times: list[int] = []
    target_kind: list[int] = []
    target_delta: list[int] = []
    target_signature: list[int] = []
    target_end_gap: list[int] = []
    reconstructed_ms = int(write_start_ms)
    for token_id_raw in token_ids:
        token_id = int(token_id_raw)
        if token_id == int(vocab.eos_id):
            continue
        if vocab.is_time_shift_token(token_id):
            current_ms += int(vocab.time_shift_value(token_id))
            continue
        if vocab.is_event_token(token_id):
            signature = vocab.event_signature(token_id)
            delta_ms = int(current_ms) - int(anchor_ms)
            delta_label = _time_grid_label(delta_ms, delta_class_count, name="event delta")
            target_kind.append(EVENT_KIND)
            target_delta.append(delta_label)
            target_signature.append(int(event_index_by_signature[signature]))
            target_end_gap.append(IGNORE_INDEX)
            original_event_times.append(int(current_ms))
            reconstructed_ms += int(delta_ms)
            reconstructed_event_times.append(int(reconstructed_ms))
            anchor_ms = int(current_ms)
            continue
        raise ValueError(f"unsupported v3 token in delta-event factor target: {token_id}")
    end_gap_ms = int(resolved_target_end_ms) - int(anchor_ms)
    end_gap_label = _time_grid_label(end_gap_ms, end_gap_class_count, name="end gap")
    target_kind.append(END_KIND)
    target_delta.append(IGNORE_INDEX)
    target_signature.append(IGNORE_INDEX)
    target_end_gap.append(end_gap_label)
    reconstructed_target_end = int(reconstructed_ms) + int(end_gap_ms)
    reconstruction_mismatch_count = int(
        tuple(original_event_times) != tuple(reconstructed_event_times)
        or int(reconstructed_target_end) != int(resolved_target_end_ms)
    )

    input_kind: list[int] = [BOS_INPUT_KIND]
    input_delta: list[int] = [0]
    input_signature: list[int] = [0]
    input_end_gap: list[int] = [0]
    for index in range(len(target_kind) - 1):
        input_kind.append(target_kind[index])
        input_delta.append(target_delta[index] if target_delta[index] != IGNORE_INDEX else 0)
        input_signature.append(target_signature[index] if target_signature[index] != IGNORE_INDEX else 0)
        input_end_gap.append(target_end_gap[index] if target_end_gap[index] != IGNORE_INDEX else 0)
    return DeltaEventFactorTarget(
        input_kind=tuple(input_kind),
        input_delta=tuple(input_delta),
        input_signature=tuple(input_signature),
        input_end_gap=tuple(input_end_gap),
        target_kind=tuple(target_kind),
        target_delta=tuple(target_delta),
        target_signature=tuple(target_signature),
        target_end_gap=tuple(target_end_gap),
        event_row_count=len(target_kind) - 1,
        end_row_count=1,
        reconstruction_mismatch_count=int(reconstruction_mismatch_count),
    )


def collate_delta_event_factor_targets(
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not targets:
        raise ValueError("collate_delta_event_factor_targets requires at least one target")
    batch_size = len(targets)
    max_rows = max(int(target["row_count"].item()) for target in targets)
    input_kind = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_delta = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_signature = torch.zeros((batch_size, max_rows), dtype=torch.long)
    input_end_gap = torch.zeros((batch_size, max_rows), dtype=torch.long)
    target_kind = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_delta = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_signature = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    target_end_gap = torch.full((batch_size, max_rows), IGNORE_INDEX, dtype=torch.long)
    row_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    row_count = torch.zeros((batch_size,), dtype=torch.long)
    event_row_count = torch.zeros((batch_size,), dtype=torch.long)
    end_row_count = torch.zeros((batch_size,), dtype=torch.long)
    reconstruction_mismatch_count = torch.zeros((batch_size,), dtype=torch.long)
    for batch_index, target in enumerate(targets):
        length = int(target["row_count"].item())
        input_kind[batch_index, :length] = target["input_kind"].to(dtype=torch.long)
        input_delta[batch_index, :length] = target["input_delta"].to(dtype=torch.long)
        input_signature[batch_index, :length] = target["input_signature"].to(dtype=torch.long)
        input_end_gap[batch_index, :length] = target["input_end_gap"].to(dtype=torch.long)
        target_kind[batch_index, :length] = target["target_kind"].to(dtype=torch.long)
        target_delta[batch_index, :length] = target["target_delta"].to(dtype=torch.long)
        target_signature[batch_index, :length] = target["target_signature"].to(dtype=torch.long)
        target_end_gap[batch_index, :length] = target["target_end_gap"].to(dtype=torch.long)
        row_mask[batch_index, :length] = target["row_mask"].to(dtype=torch.bool)
        row_count[batch_index] = target["row_count"].to(dtype=torch.long)
        event_row_count[batch_index] = target["event_row_count"].to(dtype=torch.long)
        end_row_count[batch_index] = target["end_row_count"].to(dtype=torch.long)
        reconstruction_mismatch_count[batch_index] = target["reconstruction_mismatch_count"].to(dtype=torch.long)
    return {
        "input_kind": input_kind,
        "input_delta": input_delta,
        "input_signature": input_signature,
        "input_end_gap": input_end_gap,
        "target_kind": target_kind,
        "target_delta": target_delta,
        "target_signature": target_signature,
        "target_end_gap": target_end_gap,
        "row_mask": row_mask,
        "row_count": row_count,
        "event_row_count": event_row_count,
        "end_row_count": end_row_count,
        "reconstruction_mismatch_count": reconstruction_mismatch_count,
    }


def _time_grid_label(value_ms: int, class_count: int, *, name: str) -> int:
    value = int(value_ms)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value % 10 != 0:
        raise ValueError(f"{name} must be on the 10ms grid, got {value}")
    label = value // 10
    if label >= int(class_count):
        raise ValueError(f"{name} {value}ms exceeds factor target range {(int(class_count) - 1) * 10}ms")
    return int(label)


__all__ = [
    "BOS_INPUT_KIND",
    "DEFAULT_DELTA_EVENT_FACTOR_TIME_CLASSES",
    "END_KIND",
    "EVENT_KIND",
    "IGNORE_INDEX",
    "DeltaEventFactorTarget",
    "collate_delta_event_factor_targets",
    "delta_event_factor_target_from_v3_tokens",
]
