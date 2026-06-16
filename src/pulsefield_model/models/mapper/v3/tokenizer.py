from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from pulsefield_model.models.mapper.v2_1.tokenizer import (
    MAPPER_DENSITY_FRAME_MS,
    MAPPER_DENSITY_FRAMES,
    MAPPER_WRITE_MS,
    MapperTimepoint,
    MapperTokenizationError,
    UnsupportedMapperActionError,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
    window_timepoints,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import (
    encode_full_chart_tokens as _v2_1_encode_full_chart_tokens,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import (
    encode_mapper_window as _v2_1_encode_mapper_window,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import (
    quantize_10ms_half_up,
)
from pulsefield_model.models.mapper.shared.tokenizer import (
    final_full_chart_token_before as _event_final_full_chart_token_before,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab

from .conversion import v2_1_tokens_to_v3_event_tokens
from .replay import (
    LNCarryState,
    close_labels_from_tokens,
    replay_state_tensors,
)
from .vocab import MapperV3Vocab


@dataclass(frozen=True)
class TokenizedMapperWindow:
    target_fragment_ids: list[int]
    decoder_input_ids: list[int]
    write_start_ms: int
    write_end_ms: int
    chart_end_ms: int
    is_full_chart_start: bool
    is_full_chart_end: bool
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState
    target_fragment_current_ms: torch.Tensor
    target_fragment_open_mask: torch.Tensor
    target_fragment_open_start_ms: torch.Tensor
    target_fragment_open_age_ms: torch.Tensor
    close_labels: torch.Tensor
    close_label_mask: torch.Tensor

    @property
    def seq_len(self) -> int:
        return len(self.target_fragment_ids)

    def target_fragment_tensor(self) -> torch.Tensor:
        return torch.tensor(self.target_fragment_ids, dtype=torch.long)

    def decoder_input_tensor(self) -> torch.Tensor:
        return torch.tensor(self.decoder_input_ids, dtype=torch.long)


def encode_full_chart_tokens(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperV3Vocab,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> list[int]:
    v2_vocab = MapperV21Vocab(vocab.time_shift_values_ms)
    v2_tokens = _v2_1_encode_full_chart_tokens(
        timepoints,
        vocab=v2_vocab,
        chart_start_ms=chart_start_ms,
        chart_end_ms=chart_end_ms,
    )
    return list(v2_1_tokens_to_v3_event_tokens(v2_tokens, source_vocab=v2_vocab, target_vocab=vocab).token_ids)


def encode_mapper_window(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> TokenizedMapperWindow:
    v2_vocab = MapperV21Vocab(vocab.time_shift_values_ms)
    v2_window = _v2_1_encode_mapper_window(
        timepoints,
        vocab=v2_vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_start_ms=chart_start_ms,
        chart_end_ms=chart_end_ms,
    )
    conversion = v2_1_tokens_to_v3_event_tokens(
        v2_window.target_fragment_ids,
        source_vocab=v2_vocab,
        target_vocab=vocab,
    )
    target_fragment_ids = list(conversion.token_ids)
    first_decoder_input = (
        vocab.bos_id
        if v2_window.is_full_chart_start
        else final_full_chart_token_before(
            timepoints,
            vocab=vocab,
            chart_start_ms=chart_start_ms,
            boundary_ms=write_start_ms,
        )
    )
    decoder_input_ids = [first_decoder_input, *target_fragment_ids[:-1]]
    if len(decoder_input_ids) != len(target_fragment_ids):
        raise MapperTokenizationError("decoder input and target fragment lengths must match")

    ln_carry_in = _coerce_carry(v2_window.ln_carry_in)
    ln_carry_out = _coerce_carry(v2_window.ln_carry_out)
    state_tensors = replay_state_tensors(
        target_fragment_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=v2_window.chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_start=v2_window.is_full_chart_start,
        is_full_chart_end=v2_window.is_full_chart_end,
    )
    close_labels, close_label_mask = close_labels_from_tokens(
        target_fragment_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=v2_window.chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_end=v2_window.is_full_chart_end,
    )
    return TokenizedMapperWindow(
        target_fragment_ids=target_fragment_ids,
        decoder_input_ids=decoder_input_ids,
        write_start_ms=v2_window.write_start_ms,
        write_end_ms=v2_window.write_end_ms,
        chart_end_ms=v2_window.chart_end_ms,
        is_full_chart_start=v2_window.is_full_chart_start,
        is_full_chart_end=v2_window.is_full_chart_end,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        target_fragment_current_ms=state_tensors["current_ms"],
        target_fragment_open_mask=state_tensors["open_mask"],
        target_fragment_open_start_ms=state_tensors["open_start_ms"],
        target_fragment_open_age_ms=state_tensors["open_age_ms"],
        close_labels=close_labels,
        close_label_mask=close_label_mask,
    )


def final_full_chart_token_before(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperV3Vocab,
    chart_start_ms: int,
    boundary_ms: int,
) -> int:
    return _event_final_full_chart_token_before(
        timepoints,
        vocab=vocab,
        chart_start_ms=chart_start_ms,
        boundary_ms=boundary_ms,
    )


def tokenize_hitobjects_window(
    hitobjects: Sequence[Any],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> TokenizedMapperWindow:
    return encode_mapper_window(
        hitobjects_to_mapper_timepoints(hitobjects),
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_start_ms=chart_start_ms,
        chart_end_ms=chart_end_ms,
    )


def _coerce_carry(value: Any) -> LNCarryState:
    return LNCarryState(
        current_ms=int(value.current_ms),
        open_mask=tuple(bool(item) for item in value.open_mask),  # type: ignore[arg-type]
        open_start_ms=tuple(None if item is None else int(item) for item in value.open_start_ms),  # type: ignore[arg-type]
        open_age_ms=tuple(int(item) for item in value.open_age_ms),  # type: ignore[arg-type]
    )
