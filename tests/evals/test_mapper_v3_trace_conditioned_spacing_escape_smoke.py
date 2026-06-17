from __future__ import annotations

import torch

from pulsefield_model.evals.mapper_v3_trace_conditioned_spacing_escape_smoke import (
    TraceConditionedSpacingEscapeTransform,
    decision_from_aggregate,
    guard_results_from_aggregate,
)
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state, transition_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_spacing_escape_penalizes_repeated_spacing_time_shift_piece() -> None:
    vocab = MapperV3Vocab()
    event_token = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    ts_100 = vocab.time_shift_token_id(100)
    ts_60 = vocab.time_shift_token_id(60)
    ts_90 = vocab.time_shift_token_id(90)
    generated_tokens: list[int] = []
    for _ in range(5):
        if generated_tokens:
            generated_tokens.extend([ts_100, ts_60])
        generated_tokens.append(event_token)
    step = _step(vocab, generated_tokens=generated_tokens, write_start_ms=0, write_end_ms=8_000)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[ts_100] = True
    valid_mask[ts_90] = True
    step = _replace_valid_mask(step, valid_mask)
    logits = torch.zeros(vocab.size, dtype=torch.float32)
    logits[ts_100] = 5.0
    logits[ts_90] = 4.5

    transform = TraceConditionedSpacingEscapeTransform(vocab=vocab, spacing_penalty=2.0)
    transformed = transform(step, logits)

    assert transformed[ts_100].item() == 3.0
    assert transformed[ts_90].item() == 4.5
    assert transform.spacing_activation_count == 1
    assert transform.boundary_activation_count == 0
    assert transform.examples[0]["spacing_ms"] == 160
    assert transform.examples[0]["token_name"] == "TS_100"


def test_boundary_escape_boosts_rank_near_event_in_second_window() -> None:
    vocab = MapperV3Vocab()
    event_token = vocab.encode_event((LaneAction.NONE, LaneAction.TAP, LaneAction.NONE, LaneAction.NONE))
    ts_100 = vocab.time_shift_token_id(100)
    step = _step(
        vocab,
        generated_tokens=[ts_100],
        write_start_ms=8_000,
        write_end_ms=16_000,
        chart_end_ms=16_000,
        full_start=False,
        full_end=True,
    )
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[ts_100] = True
    valid_mask[event_token] = True
    step = _replace_valid_mask(step, valid_mask)
    logits = torch.full((vocab.size,), -10.0, dtype=torch.float32)
    logits[ts_100] = 3.0
    logits[event_token] = 2.0

    transform = TraceConditionedSpacingEscapeTransform(
        vocab=vocab,
        boundary_event_boost=1.5,
        boundary_event_rank_limit=2,
        boundary_event_margin_floor=-2.0,
    )
    transformed = transform(step, logits)

    assert transformed[event_token].item() == 3.5
    assert transformed[ts_100].item() == 3.0
    assert transform.boundary_activation_count == 1
    assert transform.examples[0]["event_signature"] == ".T.."
    assert transform.examples[0]["argmax_token"] == "TS_100"


def test_boundary_escape_does_not_boost_when_event_rank_is_too_low() -> None:
    vocab = MapperV3Vocab()
    event_token = vocab.encode_event((LaneAction.NONE, LaneAction.TAP, LaneAction.NONE, LaneAction.NONE))
    ts_100 = vocab.time_shift_token_id(100)
    ts_90 = vocab.time_shift_token_id(90)
    step = _step(
        vocab,
        generated_tokens=[ts_100],
        write_start_ms=8_000,
        write_end_ms=16_000,
        chart_end_ms=16_000,
        full_start=False,
        full_end=True,
    )
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[ts_100] = True
    valid_mask[ts_90] = True
    valid_mask[event_token] = True
    step = _replace_valid_mask(step, valid_mask)
    logits = torch.full((vocab.size,), -10.0, dtype=torch.float32)
    logits[ts_100] = 3.0
    logits[ts_90] = 2.5
    logits[event_token] = 2.0

    transform = TraceConditionedSpacingEscapeTransform(vocab=vocab, boundary_event_boost=1.5, boundary_event_rank_limit=1)
    transformed = transform(step, logits)

    assert transformed[event_token].item() == 2.0
    assert transform.boundary_activation_count == 0


def test_decision_routes_positive_when_primary_signal_and_guards_pass() -> None:
    aggregate = {
        "primary_case_count": 6,
        "sentinel_case_count": 4,
        "primary": {
            "rigidity_improved_count": 4,
            "second_window_improved_count": 5,
            "both_improved_count": 4,
        },
    }
    guard_results = {
        "case_coverage": True,
        "no_training": True,
        "no_tokenizer_or_default_decode_change": True,
        "no_target_leakage": True,
        "all_legal": True,
        "no_dead_end_cases": True,
        "no_max_token_cases": True,
        "primary_event_ratio_guard": True,
        "sentinel_median_event_ratio_guard": True,
        "max_boundary_ratio_guard": True,
        "mean_f1_guard": True,
    }

    decision = decision_from_aggregate(aggregate, guard_results)

    assert decision["route"] == "TEST_FULL32_TRACE_CONDITIONED_SPACING_ESCAPE"


def test_guard_and_decision_kill_overproduction() -> None:
    aggregate = {
        "primary_case_count": 6,
        "sentinel_case_count": 4,
        "all_legal": True,
        "dead_end_count": 0,
        "max_token_count": 0,
        "primary": {
            "event_ratio_over_count": 0,
            "mean_f1_delta": 0.0,
            "baseline": {"max_boundary_event_ratio": 0.04},
            "candidate": {"max_boundary_event_ratio": 0.05},
            "rigidity_improved_count": 4,
            "second_window_improved_count": 4,
            "both_improved_count": 4,
        },
        "sentinel": {
            "candidate": {"median_event_count_ratio": 1.40},
        },
    }

    guards = guard_results_from_aggregate(aggregate)
    decision = decision_from_aggregate(aggregate, guards)

    assert guards["sentinel_median_event_ratio_guard"] is False
    assert decision["route"] == "KILL_OVERPRODUCTION"


def _step(
    vocab: MapperV3Vocab,
    *,
    generated_tokens: list[int],
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int | None = None,
    full_start: bool = True,
    full_end: bool = False,
) -> MapperV3GenerationStep:
    chart_end = write_end_ms if chart_end_ms is None else int(chart_end_ms)
    carry_in = empty_ln_carry_state(write_start_ms)
    carry_out = empty_ln_carry_state(chart_end if full_end else write_end_ms)
    state = initial_replay_state(carry_in)
    for position, token_id in enumerate(generated_tokens):
        state = transition_replay_state(
            state,
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end,
            ln_carry_out=carry_out,
            is_full_chart_start=full_start,
            is_full_chart_end=full_end,
        )
    return MapperV3GenerationStep(
        decoder_input_tokens=torch.tensor([vocab.bos_id, *generated_tokens], dtype=torch.long),
        generated_tokens=tuple(generated_tokens),
        state=state,
        valid_token_mask=torch.zeros(vocab.size, dtype=torch.bool),
        token_index=len(generated_tokens),
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end,
        ln_carry_in=carry_in,
        ln_carry_out=carry_out,
        is_full_chart_start=full_start,
        is_full_chart_end=full_end,
    )


def _replace_valid_mask(step: MapperV3GenerationStep, valid_mask: torch.Tensor) -> MapperV3GenerationStep:
    return MapperV3GenerationStep(
        decoder_input_tokens=step.decoder_input_tokens,
        generated_tokens=step.generated_tokens,
        state=step.state,
        valid_token_mask=valid_mask,
        token_index=step.token_index,
        write_start_ms=step.write_start_ms,
        write_end_ms=step.write_end_ms,
        chart_end_ms=step.chart_end_ms,
        ln_carry_in=step.ln_carry_in,
        ln_carry_out=step.ln_carry_out,
        is_full_chart_start=step.is_full_chart_start,
        is_full_chart_end=step.is_full_chart_end,
    )
