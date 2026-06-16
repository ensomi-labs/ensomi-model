from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from pulsefield_model.models.mapper.shared.grammar import (
    _broadcast_bool_tensor,
    _broadcast_carry,
    _broadcast_window_tensor,
    _carry_at,
    _event_is_legal,
    _normalize_open_age_ms,
    _normalize_open_mask,
    _normalize_open_start_ms,
)
from pulsefield_model.models.mapper.shared.replay import MapperReplayState, ReplayError, replay_state_matches_carry

from .replay import LNCarryState, target_end_ms, transition_replay_state
from .vocab import MapperV3Vocab


def valid_token_mask(
    *,
    position: int,
    current_ms: int,
    open_mask: int | Sequence[bool] | torch.Tensor,
    open_start_ms: Sequence[int | None] | torch.Tensor,
    open_age_ms: Sequence[int] | torch.Tensor,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    vocab: MapperV3Vocab,
    chart_end_ms: int | None = None,
    min_ln_duration_ms: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_device = torch.device("cpu") if device is None else device
    target_end = target_end_ms(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    state = MapperReplayState(
        position=int(position),
        current_ms=int(current_ms),
        open_mask=_normalize_open_mask(open_mask),
        open_start_ms=_normalize_open_start_ms(open_start_ms),
        open_age_ms=_normalize_open_age_ms(open_age_ms),
    )
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=resolved_device)

    if int(position) < 0:
        if (
            bool(is_full_chart_start)
            and int(state.current_ms) == int(write_start_ms)
            and replay_state_matches_carry(state, ln_carry_in)
        ):
            mask[vocab.bos_id] = True
        return mask

    for token_id in range(vocab.size):
        if token_id in {vocab.pad_id, vocab.bos_id}:
            continue
        if min_ln_duration_ms is not None and vocab.is_event_token(token_id):
            remaining_ms = int(target_end) - int(current_ms)
            if not _event_is_legal(
                vocab.decode_event(token_id),
                state.open_mask,
                remaining_ms=remaining_ms,
                min_ln_duration_ms=min_ln_duration_ms,
            ):
                continue
        try:
            transition_replay_state(
                state,
                token_id,
                position=int(position),
                vocab=vocab,
                write_start_ms=int(write_start_ms),
                write_end_ms=int(write_end_ms),
                chart_end_ms=chart_end_ms,
                ln_carry_out=ln_carry_out,
                is_full_chart_start=bool(is_full_chart_start),
                is_full_chart_end=bool(is_full_chart_end),
            )
        except (ReplayError, ValueError):
            continue
        mask[token_id] = True
    return mask


def build_grammar_mask(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor,
    open_age_ms: torch.Tensor,
    write_start_ms: torch.Tensor | int,
    write_end_ms: torch.Tensor | int,
    ln_carry_in: LNCarryState | Mapping[str, Any],
    ln_carry_out: LNCarryState | Mapping[str, Any],
    is_full_chart_start: torch.Tensor | bool,
    is_full_chart_end: torch.Tensor | bool,
    vocab: MapperV3Vocab,
    chart_end_ms: torch.Tensor | int | None = None,
    positions: torch.Tensor | None = None,
    min_ln_duration_ms: int | None = None,
    invalid_value: float = -torch.inf,
) -> torch.Tensor:
    if current_ms.ndim == 1:
        current_ms = current_ms.unsqueeze(0)
    if open_mask.ndim == 2:
        open_mask = open_mask.unsqueeze(0)
    if open_start_ms.ndim == 2:
        open_start_ms = open_start_ms.unsqueeze(0)
    if open_age_ms.ndim == 2:
        open_age_ms = open_age_ms.unsqueeze(0)
    if current_ms.ndim != 2:
        raise ValueError(f"current_ms must have shape [B,T] or [T], got {tuple(current_ms.shape)}")
    if open_mask.shape[:2] != current_ms.shape or int(open_mask.shape[-1]) != 4:
        raise ValueError(f"open_mask must have shape {tuple(current_ms.shape)}x4, got {tuple(open_mask.shape)}")
    if open_start_ms.shape != open_mask.shape:
        raise ValueError(f"open_start_ms must match open_mask shape, got {tuple(open_start_ms.shape)}")
    if open_age_ms.shape != open_mask.shape:
        raise ValueError(f"open_age_ms must match open_mask shape, got {tuple(open_age_ms.shape)}")

    batch_size, steps = current_ms.shape
    device = current_ms.device
    if positions is None:
        positions = torch.arange(steps, dtype=torch.long, device=device).expand(batch_size, steps)
    elif positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(batch_size, steps)
    if tuple(positions.shape) != (batch_size, steps):
        raise ValueError(f"positions must have shape {(batch_size, steps)}, got {tuple(positions.shape)}")

    write_start_values = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
    write_end_values = _broadcast_window_tensor(write_end_ms, batch_size=batch_size, device=device)
    if chart_end_ms is None:
        chart_end_values = write_end_values
    else:
        chart_end_values = _target_end_tensor(
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            is_full_chart_end=is_full_chart_end,
            device=current_ms.device,
        )
    full_start_values = _broadcast_bool_tensor(is_full_chart_start, batch_size=batch_size, device=device)
    full_end_values = _broadcast_bool_tensor(is_full_chart_end, batch_size=batch_size, device=device)
    carry_in_values = _broadcast_carry(ln_carry_in, batch_size=batch_size, device=device)
    carry_out_values = _broadcast_carry(ln_carry_out, batch_size=batch_size, device=device)

    valid = torch.zeros((batch_size, steps, vocab.size), dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        ln_in = _carry_at(carry_in_values, batch_index)
        ln_out = _carry_at(carry_out_values, batch_index)
        for step in range(steps):
            valid[batch_index, step] = valid_token_mask(
                position=int(positions[batch_index, step].item()),
                current_ms=int(current_ms[batch_index, step].item()),
                open_mask=open_mask[batch_index, step],
                open_start_ms=open_start_ms[batch_index, step],
                open_age_ms=open_age_ms[batch_index, step],
                write_start_ms=int(write_start_values[batch_index].item()),
                write_end_ms=int(write_end_values[batch_index].item()),
                chart_end_ms=int(chart_end_values[batch_index].item()),
                ln_carry_in=ln_in,
                ln_carry_out=ln_out,
                is_full_chart_start=bool(full_start_values[batch_index].item()),
                is_full_chart_end=bool(full_end_values[batch_index].item()),
                vocab=vocab,
                min_ln_duration_ms=min_ln_duration_ms,
                device=device,
            )
    return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)


def _target_end_tensor(
    *,
    write_start_ms: torch.Tensor | int,
    write_end_ms: torch.Tensor | int,
    chart_end_ms: torch.Tensor | int,
    is_full_chart_end: torch.Tensor | bool,
    device: torch.device,
) -> torch.Tensor:
    write_start = torch.as_tensor(write_start_ms, dtype=torch.long, device=device).reshape(-1)
    write_end = torch.as_tensor(write_end_ms, dtype=torch.long, device=device).reshape(-1)
    chart_end = torch.as_tensor(chart_end_ms, dtype=torch.long, device=device).reshape(-1)
    full_end = torch.as_tensor(is_full_chart_end, dtype=torch.bool, device=device).reshape(-1)
    size = max(write_start.numel(), write_end.numel(), chart_end.numel(), full_end.numel())

    def expand(value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 1:
            return value.expand(size)
        if value.numel() == size:
            return value
        raise ValueError(f"cannot broadcast target-end tensor with {value.numel()} values to {size}")

    write_start = expand(write_start)
    write_end = expand(write_end)
    chart_end = expand(chart_end)
    full_end = expand(full_end)
    target_end = torch.where(full_end, chart_end, write_end)
    if bool(torch.any(target_end < write_start).item()) or bool(torch.any(target_end > write_end).item()):
        raise ValueError("chart_end_ms must be inside the write window for full-chart-end samples")
    return target_end
