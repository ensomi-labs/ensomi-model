from __future__ import annotations

from dataclasses import dataclass

import torch

from pulsefield_model.inference.mapper_v3_rollout import (
    MapperV3FullRollout,
    MapperV3GenerationError,
    MapperV3GenerationStep,
    _target_fragment_state_batch_v3,
    decoder_input_tokens_for_generation_v3,
    generated_v3_tokens_to_v2_1_tokens,
    grammar_constrained_window_generation_v3,
    mapper_v3_logits_fn,
    rollout_to_timepoints_v3,
    zero_control_batch_provider_v3,
)
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.models.mapper.v3 import MapperV3Config, MapperV3Model, MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def _actions(*actions: LaneAction) -> tuple[LaneAction, LaneAction, LaneAction, LaneAction]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)  # type: ignore[return-value]


def test_v3_window_generation_exports_event_timepoints_and_v2_1_tokens() -> None:
    vocab = MapperV3Vocab()
    tokens = [
        vocab.time_shift_token_id(100),
        vocab.encode_event(_actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP)),
        vocab.time_shift_token_id(400),
        vocab.eos_id,
    ]

    def logits_fn(step: MapperV3GenerationStep) -> torch.Tensor:
        logits = torch.full((vocab.size,), -1000.0)
        token_id = tokens[step.token_index]
        assert bool(step.valid_token_mask[token_id].item())
        logits[token_id] = 1000.0
        return logits

    window = grammar_constrained_window_generation_v3(
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=500,
        ln_carry_in=empty_ln_carry_state(0),
        ln_carry_out=empty_ln_carry_state(500),
        logits_fn=logits_fn,
        is_full_chart_start=True,
        is_full_chart_end=True,
        max_tokens=8,
    )

    assert window.completed
    assert window.tokens == tokens
    rollout = MapperV3FullRollout(
        chart_end_ms=500,
        windows=[window],
        tokens=list(window.tokens),
        completed=True,
        dead_end=False,
        max_tokens_exceeded=False,
    )
    timepoints = rollout_to_timepoints_v3(rollout, vocab)
    assert len(timepoints) == 1
    assert timepoints[0].time_ms == 100
    assert [action.value for action in timepoints[0].lane_actions] == ["TAP", "NONE", "TAP", "NONE"]

    v21 = MapperV21Vocab(vocab.time_shift_values_ms)
    expanded = generated_v3_tokens_to_v2_1_tokens(tokens, source_vocab=vocab)
    assert expanded == (
        v21.time_shift_token_id(100),
        v21.lane_action_token_id(0, "TAP"),
        v21.lane_action_token_id(2, "TAP"),
        v21.time_shift_token_id(400),
        v21.eos_id,
    )


def test_v3_decoder_input_requires_left_context_for_non_initial_window() -> None:
    vocab = MapperV3Vocab()
    generated = [vocab.time_shift_token_id(10)]

    assert decoder_input_tokens_for_generation_v3(
        vocab=vocab,
        generated_tokens=generated,
        is_full_chart_start=True,
    ) == [vocab.bos_id, *generated]

    try:
        decoder_input_tokens_for_generation_v3(vocab=vocab, generated_tokens=generated)
    except MapperV3GenerationError as exc:
        assert "left_context_tokens" in str(exc)
    else:
        raise AssertionError("non-initial v3 generation accepted an empty left context")


def test_v3_prefix_state_batch_replays_only_generated_tokens() -> None:
    vocab = MapperV3Vocab()
    generated = (
        vocab.time_shift_token_id(100),
        vocab.encode_event(_actions(LaneAction.HOLD_START)),
    )

    states = _target_fragment_state_batch_v3(
        generated_tokens=generated,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=8_000,
        ln_carry_in=empty_ln_carry_state(0),
        ln_carry_out=empty_ln_carry_state(8_000),
        is_full_chart_start=True,
        is_full_chart_end=False,
        device=torch.device("cpu"),
    )

    assert states["current_ms"].tolist() == [[0, 100, 100]]
    assert states["open_mask"].tolist() == [
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ],
    ]
    assert states["open_start_ms"].tolist()[0][2] == [100, -1, -1, -1]


def test_zero_control_batch_provider_matches_mapper_v3_shapes() -> None:
    model = MapperV3Model(
        MapperV3Config(
            control_dim=8,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            max_seq_len=32,
            use_global_context=True,
            global_conv_blocks=0,
        ),
        vocab=MapperV3Vocab(),
    )

    batch = zero_control_batch_provider_v3(model=model, device="cpu")(0, 8_000)

    assert tuple(batch["projected_control_memory_8s"].shape) == (1, 400, 16)
    assert tuple(batch["density_teacher_8s"].shape) == (1, 400, 1)
    assert tuple(batch["global_memory"].shape) == (1, 4, 16)


def test_mapper_v3_logits_fn_incremental_decode_appends_only_new_prefix_token() -> None:
    vocab = MapperV3Vocab()
    model = FakeIncrementalMapperV3Model(vocab.size)
    ln_carry_in = empty_ln_carry_state(0)
    ln_carry_out = empty_ln_carry_state(8_000)
    logits_fn = mapper_v3_logits_fn(
        model=model,
        vocab=vocab,
        device=torch.device("cpu"),
        normalized_difficulty=0.0,
        control_batch={
            "density_teacher_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
            "projected_control_memory_8s": torch.zeros((1, 400, 16), dtype=torch.float32),
        },
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=8_000,
        is_full_chart_start=True,
        is_full_chart_end=False,
        time_shift_length_penalty_alpha=0.0,
    )

    logits_fn(
        MapperV3GenerationStep(
            decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
            generated_tokens=(),
            state=initial_replay_state(ln_carry_in),
            valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
            token_index=0,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=8_000,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
        ),
    )
    logits = logits_fn(
        MapperV3GenerationStep(
            decoder_input_tokens=torch.tensor(
                [vocab.bos_id, vocab.time_shift_token_id(10)],
                dtype=torch.long,
            ),
            generated_tokens=(vocab.time_shift_token_id(10),),
            state=initial_replay_state(ln_carry_in),
            valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
            token_index=1,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=8_000,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
        ),
    )

    assert model.calls == [(0, vocab.bos_id), (1, vocab.time_shift_token_id(10))]
    assert int(torch.argmax(logits).item()) == vocab.time_shift_token_id(10)
    assert model.seen_chart_end_ms == [8_000, 8_000]
    assert not model.seen_sparse_state_kwargs


@dataclass(frozen=True)
class FakeIncrementalOutput:
    decode_state: object
    logits_final: torch.Tensor


class FakeIncrementalMapperV3Model:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.calls: list[tuple[int, int]] = []
        self.seen_chart_end_ms: list[int] = []
        self.seen_sparse_state_kwargs = False

    def create_empty_decode_state(self, *, batch_size: int, device: torch.device) -> object:
        assert batch_size == 1
        assert device == torch.device("cpu")
        return object()

    def incremental_decode_next_token(
        self,
        *,
        decoder_input_token: torch.Tensor,
        position: int,
        decode_state: object,
        chart_end_ms: torch.Tensor,
        **kwargs,
    ) -> FakeIncrementalOutput:
        token_id = int(decoder_input_token.item())
        self.calls.append((int(position), token_id))
        self.seen_chart_end_ms.append(int(chart_end_ms.item()))
        self.seen_sparse_state_kwargs = self.seen_sparse_state_kwargs or (
            "emitted_lane_mask" in kwargs or "last_lane_index" in kwargs
        )
        logits = torch.full((1, self.vocab_size), -1000.0, dtype=torch.float32)
        logits[0, token_id] = 1000.0
        return FakeIncrementalOutput(decode_state=decode_state, logits_final=logits)
