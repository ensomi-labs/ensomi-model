from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from pulsefield_model.models.mapper.shared.batch import MapperBatch, MapperTokenContract
from pulsefield_model.models.mapper.shared.incremental import (
    IncrementalDecodeState,
    as_decode_batch_vector,
    as_decode_step_lane_tensor,
    as_decode_step_tensor,
    decode_position_index,
    normalize_attention_kv_cache,
    validate_incremental_decode_state,
)
from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT
from pulsefield_model.models.mapper.v2.model import (
    GLOBAL_POSITION_FEATURES,
    MapperV2Config,
    MapperV2ForwardOutput,
    MapperV2Model,
)
from pulsefield_model.models.mapper.v2_1.model import (
    _batch_bool_flag,
    _precomputed_global_attention_kv_cache,
    _validate_v21_fragment_contract,
)

from .grammar import build_grammar_mask
from .tokenizer import MAPPER_DENSITY_FRAMES
from .vocab import MapperV3Vocab


@dataclass(frozen=True)
class MapperV3Config(MapperV2Config):
    """Mapper v3 event-token config.

    The v3 target stream is shorter than v2.1 on the full audit, but the smoke
    path keeps the v2.1 sequence cap to avoid silently dropping long charts
    before trained v3 evidence exists.
    """

    max_seq_len: int = 1024


@dataclass(frozen=True)
class MapperV3ForwardOutput(MapperV2ForwardOutput):
    pass


MapperV3ModelOutput = MapperV3ForwardOutput
MapperV3IncrementalDecodeState = IncrementalDecodeState


@dataclass(frozen=True)
class MapperV3IncrementalDecodeOutput:
    decode_state: MapperV3IncrementalDecodeState
    decoder_input_token: torch.Tensor
    position: torch.Tensor
    base_logits: torch.Tensor
    logits_final: torch.Tensor
    decoder_hidden: torch.Tensor
    state_prior_bias: torch.Tensor
    state_prior_lane_action_bias: torch.Tensor
    ln_close_logits: torch.Tensor
    ln_close_event_bias: torch.Tensor
    ln_close_time_shift_bias: torch.Tensor
    grammar_mask: torch.Tensor
    global_attention_gates: torch.Tensor | None = None


class MapperV3Model(MapperV2Model):
    """Mapper V2 global-context decoder with the v3 event-token contract."""

    def __init__(
        self,
        config: MapperV3Config = MapperV3Config(),
        *,
        vocab: MapperV3Vocab | None = None,
        control_encoder: nn.Module | None = None,
    ) -> None:
        resolved_vocab = MapperV3Vocab() if vocab is None else vocab
        if config.vocab_size is not None and int(config.vocab_size) != resolved_vocab.size:
            raise ValueError(f"config vocab_size {config.vocab_size} does not match mapper v3 vocab size {resolved_vocab.size}")
        super().__init__(config, vocab=resolved_vocab, control_encoder=control_encoder)
        self.config: MapperV3Config = config
        self.vocab: MapperV3Vocab = resolved_vocab

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        *,
        control_memory_8s: torch.Tensor | None = None,
        density_teacher_8s: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> MapperV3ForwardOutput:
        if batch is None:
            batch = {key: value for key, value in kwargs.items() if value is not None}
        elif kwargs:
            merged = dict(batch)
            merged.update({key: value for key, value in kwargs.items() if value is not None})
            batch = merged
        if control_memory_8s is None:
            maybe_control_memory = batch.get("control_memory_8s")
            if isinstance(maybe_control_memory, torch.Tensor):
                control_memory_8s = maybe_control_memory
        projected_control_memory_8s = batch.get("projected_control_memory_8s")
        if projected_control_memory_8s is not None and not isinstance(projected_control_memory_8s, torch.Tensor):
            raise ValueError("projected_control_memory_8s must be a torch.Tensor")
        if projected_control_memory_8s is not None and control_memory_8s is not None:
            raise ValueError("projected_control_memory_8s cannot be supplied with control_memory_8s")
        if density_teacher_8s is None:
            maybe_density_teacher = batch.get("density_teacher_8s")
            if isinstance(maybe_density_teacher, torch.Tensor):
                density_teacher_8s = maybe_density_teacher
        has_control_context = control_memory_8s is not None or projected_control_memory_8s is not None
        if has_control_context != (density_teacher_8s is not None):
            raise ValueError("control_memory_8s and density_teacher_8s must be supplied together")
        if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
            raise ValueError("control_memory_padding_mask_8s is not supported in Phase B; supply full 8s control memory")

        mapper_batch = MapperBatch.from_mapping(
            batch,
            contract=MapperTokenContract(
                name="v3",
                vocab=self.vocab,
                uses_chart_end_for_terminal_windows=True,
            ),
        )
        decoder_input = mapper_batch.decoder_input_tokens
        loss_target_tokens = mapper_batch.target_fragment_tokens
        target_fragment_mask = mapper_batch.target_fragment_mask
        input_padding_mask = mapper_batch.input_padding_mask
        device = decoder_input.device
        fragment_states = mapper_batch.fragment_states
        current_ms = fragment_states.current_ms
        open_mask = fragment_states.open_mask
        open_start_ms = fragment_states.open_start_ms
        open_age_ms = fragment_states.open_age_ms
        write_start_ms = mapper_batch.write_start_ms
        write_end_ms = mapper_batch.write_end_ms
        chart_end_ms = mapper_batch.chart_end_ms
        target_end_ms = mapper_batch.target_end_ms
        is_full_chart_start = mapper_batch.is_full_chart_start
        is_full_chart_end = mapper_batch.is_full_chart_end
        ln_carry_in = mapper_batch.ln_carry_in.as_mapping()
        ln_carry_out = mapper_batch.ln_carry_out.as_mapping()
        valid_input_mask = target_fragment_mask.to(device=device, dtype=torch.bool)
        _validate_v21_fragment_contract(
            decoder_input_tokens=decoder_input,
            target_fragment_tokens=loss_target_tokens,
            target_fragment_mask=valid_input_mask,
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            target_end_ms=target_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
        )
        sanitized_states = fragment_states.sanitized(target_end_ms, valid_input_mask)
        current_ms = sanitized_states.current_ms
        open_mask = sanitized_states.open_mask
        open_start_ms = sanitized_states.open_start_ms
        open_age_ms = sanitized_states.open_age_ms

        if control_memory_8s is None and projected_control_memory_8s is None:
            control_memory_8s, density_teacher_8s = self._control_teacher_8s(batch)
        assert density_teacher_8s is not None
        density_teacher_8s = density_teacher_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        if projected_control_memory_8s is None:
            assert control_memory_8s is not None
            control_memory_8s = control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
            if control_memory_8s.ndim != 3 or int(control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if int(control_memory_8s.shape[-1]) != self.config.control_dim:
                raise ValueError(
                    f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                    f"got {control_memory_8s.shape[-1]}"
                )
            control_memory = self.control_projection(control_memory_8s)
        else:
            control_memory = projected_control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
            if control_memory.ndim != 3 or int(control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"projected_control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if int(control_memory.shape[-1]) != self.config.d_model:
                raise ValueError(
                    f"projected_control_memory_8s last dim must match config.d_model={self.config.d_model}, "
                    f"got {control_memory.shape[-1]}"
                )
        if tuple(density_teacher_8s.shape) != (decoder_input.shape[0], MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")

        global_memory, global_memory_padding_mask, global_position_features = self._global_context_memory(
            batch=batch,
            device=device,
            batch_size=int(decoder_input.shape[0]),
            write_start_ms=write_start_ms,
        )
        global_attention_kv_cache = _precomputed_global_attention_kv_cache(
            batch=batch,
            device=device,
            batch_size=int(decoder_input.shape[0]),
            global_memory=global_memory,
            config=self.config,
        )
        decoder_hidden, base_logits = self._decode_with_global_context(
            tokens=decoder_input,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=target_end_ms,
            difficulty=self._difficulty(batch, device=decoder_input.device),
            control_memory=control_memory,
            input_padding_mask=input_padding_mask,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
            global_attention_kv_cache=global_attention_kv_cache,
        )
        remaining_ms = (target_end_ms.reshape(-1, 1) - current_ms).clamp_min(0)
        state_prior = self.state_prior_adapter(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms,
        )
        ln_close = self.ln_close_adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
        )
        apply_grammar_mask = _batch_bool_flag(batch, key="apply_grammar_mask", default=True)
        if apply_grammar_mask:
            positions = torch.arange(decoder_input.shape[1], dtype=torch.long, device=decoder_input.device).reshape(1, -1)
            grammar_mask = build_grammar_mask(
                current_ms=current_ms,
                open_mask=open_mask,
                open_start_ms=open_start_ms,
                open_age_ms=open_age_ms,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                chart_end_ms=chart_end_ms,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
                is_full_chart_start=is_full_chart_start,
                is_full_chart_end=is_full_chart_end,
                vocab=self.vocab,
                positions=positions.expand(decoder_input.shape[0], -1),
            ).to(dtype=base_logits.dtype)
        else:
            grammar_mask = torch.zeros_like(base_logits)
        logits_final = base_logits + state_prior.vocab_bias + ln_close.event_bias + ln_close.time_shift_bias + grammar_mask
        return MapperV3ForwardOutput(
            decoder_input_tokens=decoder_input,
            loss_target_tokens=loss_target_tokens,
            state_current_ms=current_ms,
            state_open_mask=open_mask,
            state_open_start_ms=open_start_ms,
            state_open_age_ms=open_age_ms,
            base_logits=base_logits,
            logits_final=logits_final,
            decoder_hidden=decoder_hidden,
            state_prior_bias=state_prior.vocab_bias,
            state_prior_lane_action_bias=state_prior.lane_action_bias,
            ln_close_logits=ln_close.close_logits,
            ln_close_event_bias=ln_close.event_bias,
            ln_close_time_shift_bias=ln_close.time_shift_bias,
            grammar_mask=grammar_mask,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_attention_gates=self._global_attention_gates(device=device, enabled=global_memory is not None),
            global_position_features=global_position_features,
        )

    def _difficulty(self, batch: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
        from pulsefield_model.models.mapper.shared.model import _difficulty_tensor

        return _difficulty_tensor(batch, device=device, dim=self.config.difficulty_dim)

    @torch.no_grad()
    def incremental_decode_next_token(
        self,
        *,
        decode_state: MapperV3IncrementalDecodeState,
        decoder_input_token: torch.Tensor,
        current_ms: torch.Tensor,
        open_mask: torch.Tensor,
        open_start_ms: torch.Tensor,
        open_age_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        chart_end_ms: torch.Tensor,
        is_full_chart_start: torch.Tensor,
        is_full_chart_end: torch.Tensor,
        ln_carry_in: Mapping[str, Any],
        ln_carry_out: Mapping[str, Any],
        density_teacher_8s: torch.Tensor,
        control_memory_8s: torch.Tensor | None = None,
        projected_control_memory_8s: torch.Tensor | None = None,
        control_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        difficulty: torch.Tensor | None = None,
        normalized_difficulty: torch.Tensor | None = None,
        global_memory: torch.Tensor | None = None,
        global_memory_padding_mask: torch.Tensor | None = None,
        global_position_features: torch.Tensor | None = None,
        global_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        position: int | torch.Tensor | None = None,
        apply_grammar_mask: bool = True,
    ) -> MapperV3IncrementalDecodeOutput:
        if self.training:
            raise ValueError("incremental decode is inference-only; call eval() before decoding")
        if control_memory_8s is not None and projected_control_memory_8s is not None:
            raise ValueError("projected_control_memory_8s cannot be supplied with control_memory_8s")

        token = decoder_input_token.to(dtype=torch.long)
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if token.ndim != 2 or int(token.shape[1]) != 1:
            raise ValueError(f"decoder_input_token must have shape [B] or [B,1], got {tuple(token.shape)}")
        batch_size = int(token.shape[0])
        device = token.device
        cache_steps = validate_incremental_decode_state(
            decode_state,
            batch_size=batch_size,
            layers=self.config.layers,
            heads=self.config.heads,
            head_dim=self.config.d_model // self.config.heads,
        )
        position_index = decode_position_index(position, default=cache_steps)
        if cache_steps != position_index:
            raise ValueError(f"decode cache length {cache_steps} does not match next position {position_index}")
        if position_index >= self.config.max_seq_len:
            raise ValueError(f"decode position {position_index} exceeds max_seq_len={self.config.max_seq_len}")

        current_ms_step = as_decode_step_tensor(
            current_ms,
            name="current_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        open_mask_step = as_decode_step_lane_tensor(
            open_mask,
            name="open_mask",
            batch_size=batch_size,
            lanes=KEY_COUNT,
            device=device,
            dtype=torch.bool,
        )
        open_start_ms_step = as_decode_step_lane_tensor(
            open_start_ms,
            name="open_start_ms",
            batch_size=batch_size,
            lanes=KEY_COUNT,
            device=device,
            dtype=torch.long,
        )
        open_age_ms_step = as_decode_step_lane_tensor(
            open_age_ms,
            name="open_age_ms",
            batch_size=batch_size,
            lanes=KEY_COUNT,
            device=device,
            dtype=torch.long,
        )
        write_start_ms_step = as_decode_batch_vector(
            write_start_ms,
            name="write_start_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        write_end_ms_step = as_decode_batch_vector(
            write_end_ms,
            name="write_end_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        chart_end_ms_step = as_decode_batch_vector(
            chart_end_ms,
            name="chart_end_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        is_full_chart_start_step = as_decode_batch_vector(
            is_full_chart_start,
            name="is_full_chart_start",
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
        )
        is_full_chart_end_step = as_decode_batch_vector(
            is_full_chart_end,
            name="is_full_chart_end",
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
        )
        target_end_ms_step = torch.where(is_full_chart_end_step, chart_end_ms_step, write_end_ms_step)
        terminal_outside = is_full_chart_end_step & (
            (chart_end_ms_step < write_start_ms_step) | (chart_end_ms_step > write_end_ms_step)
        )
        if bool(terminal_outside.any()):
            raise ValueError("terminal chart_end_ms must fall inside [write_start_ms, write_end_ms]")

        difficulty_source = {}
        if normalized_difficulty is not None:
            difficulty_source["normalized_difficulty"] = normalized_difficulty
        elif difficulty is not None:
            difficulty_source["difficulty"] = difficulty
        difficulty_step = self._difficulty(difficulty_source, device=device)
        if int(difficulty_step.shape[0]) != batch_size:
            raise ValueError(f"difficulty batch must be {batch_size}, got {difficulty_step.shape[0]}")

        density_teacher = density_teacher_8s.detach().to(device=device, dtype=torch.float32)
        if tuple(density_teacher.shape) != (batch_size, MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")
        if projected_control_memory_8s is None:
            if control_memory_8s is None:
                raise ValueError("control_memory_8s or projected_control_memory_8s is required")
            raw_control_memory = control_memory_8s.detach().to(device=device, dtype=torch.float32)
            if raw_control_memory.ndim != 3 or int(raw_control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if tuple(raw_control_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"control_memory_8s batch must be {batch_size}, got {raw_control_memory.shape[0]}")
            if int(raw_control_memory.shape[-1]) != self.config.control_dim:
                raise ValueError(
                    f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                    f"got {raw_control_memory.shape[-1]}"
                )
            control_memory = self.control_projection(raw_control_memory)
        else:
            control_memory = projected_control_memory_8s.detach().to(device=device, dtype=torch.float32)
            if control_memory.ndim != 3 or int(control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"projected_control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if tuple(control_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"projected_control_memory_8s batch must be {batch_size}, got {control_memory.shape[0]}")
            if int(control_memory.shape[-1]) != self.config.d_model:
                raise ValueError(
                    f"projected_control_memory_8s last dim must match config.d_model={self.config.d_model}, "
                    f"got {control_memory.shape[-1]}"
                )

        control_attention_kv = normalize_attention_kv_cache(
            control_attention_kv_cache,
            device=device,
            batch_size=batch_size,
            layers=self.config.layers,
            heads=self.config.heads,
            d_model=self.config.d_model,
            source_steps=int(control_memory.shape[1]),
            name="control_attention_kv_cache",
        )

        if global_position_features is not None:
            if not isinstance(global_position_features, torch.Tensor):
                raise ValueError("global_position_features must be a torch.Tensor")
            global_position_features = global_position_features.detach().to(device=device, dtype=torch.float32)
            if tuple(global_position_features.shape) != (batch_size, GLOBAL_POSITION_FEATURES):
                raise ValueError(
                    f"global_position_features must have shape [{batch_size},{GLOBAL_POSITION_FEATURES}], "
                    f"got {tuple(global_position_features.shape)}"
                )

        global_attention_kv = None
        if global_memory is not None:
            if not self.config.use_global_context:
                raise ValueError("global_memory cannot be supplied when global context is disabled")
            global_memory = global_memory.detach().to(device=device, dtype=torch.float32)
            if global_memory.ndim != 3 or tuple(global_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"global_memory must have shape [B,G,D], got {tuple(global_memory.shape)}")
            if int(global_memory.shape[-1]) != self.config.d_model:
                raise ValueError(f"global_memory last dim must be {self.config.d_model}, got {global_memory.shape[-1]}")
            if global_memory_padding_mask is None:
                raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
            global_memory_padding_mask = global_memory_padding_mask.detach().to(device=device, dtype=torch.bool)
            if tuple(global_memory_padding_mask.shape) != tuple(global_memory.shape[:2]):
                raise ValueError("global_memory_padding_mask must have shape [B,G]")
            global_attention_kv = normalize_attention_kv_cache(
                global_attention_kv_cache,
                device=device,
                batch_size=batch_size,
                layers=self.config.layers,
                heads=self.config.heads,
                d_model=self.config.d_model,
                source_steps=int(global_memory.shape[1]),
                name="global_attention_kv_cache",
            )
        elif global_attention_kv_cache is not None:
            raise ValueError("global_memory is required when global_attention_kv_cache is supplied")

        hidden, next_layer_caches = self._incremental_decode_hidden_next_token(
            token=token,
            current_ms=current_ms_step,
            write_start_ms=write_start_ms_step,
            write_end_ms=target_end_ms_step,
            difficulty=difficulty_step,
            control_memory=control_memory,
            decode_state=decode_state,
            position_index=position_index,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
            global_attention_kv_cache=global_attention_kv,
            control_attention_kv_cache=control_attention_kv,
        )
        decoder_hidden = self.output_norm(hidden)
        base_logits = self.output_head(decoder_hidden)
        remaining_ms = (target_end_ms_step.reshape(-1, 1) - current_ms_step).clamp_min(0)
        state_prior = self.state_prior_adapter(
            open_mask=open_mask_step,
            open_start_ms=open_start_ms_step,
            open_age_ms=open_age_ms_step,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms_step,
        )
        ln_close = self.ln_close_adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher,
            current_ms=current_ms_step,
            write_start_ms=write_start_ms_step,
            open_mask=open_mask_step,
            open_start_ms=open_start_ms_step,
            open_age_ms=open_age_ms_step,
            remaining_ms=remaining_ms,
        )
        positions = torch.full((batch_size, 1), position_index, dtype=torch.long, device=device)
        if bool(apply_grammar_mask):
            grammar_mask = build_grammar_mask(
                current_ms=current_ms_step,
                open_mask=open_mask_step,
                open_start_ms=open_start_ms_step,
                open_age_ms=open_age_ms_step,
                write_start_ms=write_start_ms_step,
                write_end_ms=write_end_ms_step,
                chart_end_ms=chart_end_ms_step,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
                is_full_chart_start=is_full_chart_start_step,
                is_full_chart_end=is_full_chart_end_step,
                vocab=self.vocab,
                positions=positions,
            ).to(dtype=base_logits.dtype)
        else:
            grammar_mask = torch.zeros_like(base_logits)
        logits_final = base_logits + state_prior.vocab_bias + ln_close.event_bias + ln_close.time_shift_bias + grammar_mask
        return MapperV3IncrementalDecodeOutput(
            decode_state=MapperV3IncrementalDecodeState(self_attention_kv_cache=tuple(next_layer_caches)),
            decoder_input_token=token[:, 0],
            position=positions[:, 0],
            base_logits=base_logits[:, 0],
            logits_final=logits_final[:, 0],
            decoder_hidden=decoder_hidden[:, 0],
            state_prior_bias=state_prior.vocab_bias[:, 0],
            state_prior_lane_action_bias=state_prior.lane_action_bias[:, 0],
            ln_close_logits=ln_close.close_logits[:, 0],
            ln_close_event_bias=ln_close.event_bias[:, 0],
            ln_close_time_shift_bias=ln_close.time_shift_bias[:, 0],
            grammar_mask=grammar_mask[:, 0],
            global_attention_gates=self._global_attention_gates(device=device, enabled=global_memory is not None),
        )


__all__ = [
    "MapperV3Config",
    "MapperV3ForwardOutput",
    "MapperV3IncrementalDecodeOutput",
    "MapperV3IncrementalDecodeState",
    "MapperV3Model",
    "MapperV3ModelOutput",
]
