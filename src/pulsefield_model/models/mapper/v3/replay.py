from __future__ import annotations

from typing import Sequence

import torch

from pulsefield_model.models.mapper.shared.replay import (
    CLOSED_OPEN_START_MS,
    LNCarryState,
    MapperReplayState,
    ReplayError,
    empty_ln_carry_state,
    format_replay_state,
    initial_replay_state,
    ln_carry_state_from_open_starts,
    ln_carry_state_tensors,
    open_mask_bits_to_tuple,
    open_mask_tuple_to_bits,
    open_start_tensor_values_to_tuple,
    open_start_tuple_to_tensor_values,
    replay_state_matches_carry,
)
from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT, LaneAction

from .vocab import MapperV3Vocab


def target_end_ms(
    *,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int | None = None,
    is_full_chart_end: bool = False,
) -> int:
    write_start = int(write_start_ms)
    write_end = int(write_end_ms)
    if write_end <= write_start:
        raise ValueError(f"write_end_ms must be after write_start_ms: {write_start}..{write_end}")
    if write_start % 10 != 0 or write_end % 10 != 0:
        raise ValueError(f"write window must be 10ms-aligned: {write_start}..{write_end}")
    if not bool(is_full_chart_end):
        return write_end
    target_end = write_end if chart_end_ms is None else int(chart_end_ms)
    if target_end < write_start or target_end > write_end:
        raise ValueError(f"chart_end_ms must be inside the write window: {write_start}..{write_end}, got {target_end}")
    if target_end % 10 != 0:
        raise ValueError(f"chart_end_ms must be 10ms-aligned: {target_end}")
    return target_end


def replay_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    validate_terminal: bool = False,
) -> list[MapperReplayState]:
    target_end = target_end_ms(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    _validate_boundary_carry_states(
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        write_start_ms=int(write_start_ms),
        target_end_ms=target_end,
    )
    state = initial_replay_state(ln_carry_in)
    states: list[MapperReplayState] = []
    for position, token_id in enumerate(token_ids):
        state = MapperReplayState(
            position=position,
            current_ms=state.current_ms,
            open_mask=state.open_mask,
            open_start_ms=state.open_start_ms,
            open_age_ms=state.open_age_ms,
            event_emitted_at_current_ms=state.event_emitted_at_current_ms,
        )
        states.append(state)
        state = transition_replay_state(
            state,
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
            chart_end_ms=chart_end_ms,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )

    if validate_terminal and not replay_state_matches_carry(state, ln_carry_out):
        raise ReplayError(
            "token sequence terminal state does not match ln_carry_out: "
            f"terminal={format_replay_state(state)} ln_carry_out={ln_carry_out}"
        )
    return states


def replay_terminal_state(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
) -> MapperReplayState:
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_start=is_full_chart_start,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=False,
    )
    if not states:
        return initial_replay_state(ln_carry_in)
    state = initial_replay_state(ln_carry_in)
    for position, token_id in enumerate(token_ids):
        state = transition_replay_state(
            MapperReplayState(
                position=position,
                current_ms=state.current_ms,
                open_mask=state.open_mask,
                open_start_ms=state.open_start_ms,
                open_age_ms=state.open_age_ms,
                event_emitted_at_current_ms=state.event_emitted_at_current_ms,
            ),
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
        )
    return state


def transition_replay_state(
    state: MapperReplayState,
    token_id: int,
    *,
    position: int,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
) -> MapperReplayState:
    target_end = target_end_ms(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    token_id = int(token_id)
    if token_id == vocab.pad_id:
        raise ReplayError("PAD is not legal in v3 replay")

    if token_id == vocab.bos_id:
        if not is_full_chart_start:
            raise ReplayError("BOS is legal only at the full-chart start")
        if int(position) != 0 or state.current_ms != int(write_start_ms):
            raise ReplayError("BOS is legal only at full-chart token position 0")
        if any(state.open_mask) or any(start is not None for start in state.open_start_ms) or any(state.open_age_ms):
            raise ReplayError("BOS requires an empty chart-start LN state")
        return _copy_state_at_position(state, position=position)

    if token_id == vocab.eos_id:
        if not is_full_chart_end:
            raise ReplayError("EOS is legal only at the full-chart end")
        if state.current_ms != int(target_end):
            raise ReplayError(f"EOS requires current_ms == chart_end_ms: {state.current_ms} != {target_end}")
        if not replay_state_matches_carry(state, ln_carry_out):
            raise ReplayError("EOS requires current LN state to equal ln_carry_out")
        return _copy_state_at_position(state, position=position)

    if vocab.is_time_shift_token(token_id):
        delta_ms = vocab.time_shift_value(token_id)
        next_ms = state.current_ms + delta_ms
        if next_ms > int(target_end):
            raise ReplayError(f"TIME_SHIFT moves past target_end_ms: {next_ms} > {target_end}")
        next_age = tuple(
            int(next_ms - open_start) if is_open and open_start is not None else 0
            for is_open, open_start in zip(state.open_mask, state.open_start_ms, strict=True)
        )
        next_state = MapperReplayState(
            position=int(position),
            current_ms=next_ms,
            open_mask=state.open_mask,
            open_start_ms=state.open_start_ms,
            open_age_ms=next_age,  # type: ignore[arg-type]
            event_emitted_at_current_ms=False,
        )
        if not is_full_chart_end and next_ms == int(target_end) and not replay_state_matches_carry(next_state, ln_carry_out):
            raise ReplayError("TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out")
        return next_state

    if vocab.is_event_token(token_id):
        if state.current_ms > int(target_end) or (state.current_ms == int(target_end) and not is_full_chart_end):
            raise ReplayError("EVENT is illegal after the target end")
        if state.event_emitted_at_current_ms:
            raise ReplayError("EVENT is illegal after an event at the same current_ms")
        next_open = list(state.open_mask)
        next_start = list(state.open_start_ms)
        next_age = list(state.open_age_ms)
        for lane, action in enumerate(vocab.decode_event(token_id)):
            is_open = state.open_mask[lane]
            if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
                raise ReplayError(f"{action.value} is illegal on open lane {lane}")
            if not is_open and action == LaneAction.HOLD_END:
                raise ReplayError(f"HOLD_END is illegal on closed lane {lane}")
            if action == LaneAction.HOLD_START:
                next_open[lane] = True
                next_start[lane] = state.current_ms
                next_age[lane] = 0
            elif action == LaneAction.HOLD_END:
                next_open[lane] = False
                next_start[lane] = None
                next_age[lane] = 0
            elif action == LaneAction.TAP:
                next_age[lane] = 0
        return MapperReplayState(
            position=int(position),
            current_ms=state.current_ms,
            open_mask=tuple(next_open),  # type: ignore[arg-type]
            open_start_ms=tuple(next_start),  # type: ignore[arg-type]
            open_age_ms=tuple(int(value) for value in next_age),  # type: ignore[arg-type]
            event_emitted_at_current_ms=True,
        )

    raise ReplayError(f"unknown mapper v3 token id: {token_id}")


def replay_state_tensors(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
):
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_start=is_full_chart_start,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=True,
    )
    return {
        "current_ms": torch.tensor([state.current_ms for state in states], dtype=torch.long),
        "open_mask": torch.tensor([state.open_mask for state in states], dtype=torch.bool),
        "open_start_ms": torch.tensor(
            [open_start_tuple_to_tensor_values(state.open_start_ms) for state in states],
            dtype=torch.long,
        ),
        "open_age_ms": torch.tensor([state.open_age_ms for state in states], dtype=torch.long),
    }


def close_labels_from_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_end: bool = False,
):
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=True,
    )
    labels = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    mask = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    for index, state in enumerate(states):
        for lane, is_open in enumerate(state.open_mask):
            mask[index, lane] = is_open
        token_id = int(token_ids[index])
        if not vocab.is_event_token(token_id):
            continue
        for lane, action in enumerate(vocab.decode_event(token_id)):
            labels[index, lane] = state.open_mask[lane] and action == LaneAction.HOLD_END
    return labels, mask


def _copy_state_at_position(state: MapperReplayState, *, position: int) -> MapperReplayState:
    return MapperReplayState(
        position=int(position),
        current_ms=state.current_ms,
        open_mask=state.open_mask,
        open_start_ms=state.open_start_ms,
        open_age_ms=state.open_age_ms,
        event_emitted_at_current_ms=state.event_emitted_at_current_ms,
    )


def _validate_boundary_carry_states(
    *,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    write_start_ms: int,
    target_end_ms: int,
) -> None:
    if int(ln_carry_in.current_ms) != int(write_start_ms):
        raise ValueError(f"ln_carry_in.current_ms must equal write_start_ms: {ln_carry_in.current_ms} != {write_start_ms}")
    if int(ln_carry_out.current_ms) != int(target_end_ms):
        raise ValueError(f"ln_carry_out.current_ms must equal target_end_ms: {ln_carry_out.current_ms} != {target_end_ms}")


__all__ = [
    "CLOSED_OPEN_START_MS",
    "LNCarryState",
    "MapperReplayState",
    "ReplayError",
    "close_labels_from_tokens",
    "empty_ln_carry_state",
    "format_replay_state",
    "initial_replay_state",
    "ln_carry_state_from_open_starts",
    "ln_carry_state_tensors",
    "open_mask_bits_to_tuple",
    "open_mask_tuple_to_bits",
    "open_start_tensor_values_to_tuple",
    "open_start_tuple_to_tensor_values",
    "replay_state_matches_carry",
    "replay_state_tensors",
    "replay_terminal_state",
    "replay_tokens",
    "target_end_ms",
    "transition_replay_state",
]
