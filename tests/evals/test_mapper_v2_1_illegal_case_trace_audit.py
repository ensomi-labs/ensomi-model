from __future__ import annotations

import torch

from pulsefield_model.evals.mapper_v2_1_illegal_case_trace_audit import (
    MapperV21IllegalCaseTraceTransform,
    classify_trace_result,
)
from pulsefield_model.inference.mapper_v2_1_rollout import (
    MapperV21AntiRigidSpacingLogitsTransform,
    MapperV21GenerationStep,
)
from pulsefield_model.models.mapper.v2_1.replay import (
    empty_ln_carry_state,
    initial_replay_state,
    transition_replay_state,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


def test_trace_transform_baseline_records_without_changing_logits() -> None:
    vocab = MapperV21Vocab()
    ts_100 = vocab.time_shift_token_id(100)
    tap = vocab.lane_action_token_id(0, "TAP")
    carry_in = empty_ln_carry_state(0)
    carry_out = empty_ln_carry_state(1_000)
    state = initial_replay_state(carry_in)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[ts_100] = True
    valid_mask[tap] = True
    logits = torch.zeros(vocab.size, dtype=torch.float32)
    logits[ts_100] = 2.0
    logits[tap] = 1.0
    step = MapperV21GenerationStep(
        decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
        generated_tokens=(),
        state=state,
        valid_token_mask=valid_mask,
        token_index=0,
        write_start_ms=0,
        write_end_ms=1_000,
        chart_end_ms=1_000,
        ln_carry_in=carry_in,
        ln_carry_out=carry_out,
        is_full_chart_start=True,
        is_full_chart_end=False,
    )
    transform = MapperV21IllegalCaseTraceTransform(vocab=vocab, mode="baseline")

    transformed = transform(step, logits)
    trace = transform.to_dict()

    assert torch.equal(transformed, logits)
    assert trace["step_count"] == 1
    assert trace["changed_step_count"] == 0
    assert trace["tail_records"][0]["selected_after_transform"]["token_name"] == "TS_100"
    assert trace["last_terminal_probe"]["valid_token_count"] > 0


def test_trace_guard_matches_existing_anti_rigid_transform() -> None:
    vocab = MapperV21Vocab()
    ts_100 = vocab.time_shift_token_id(100)
    ts_90 = vocab.time_shift_token_id(90)
    ts_60 = vocab.time_shift_token_id(60)
    tap = vocab.lane_action_token_id(0, "TAP")
    generated = tuple(token for _ in range(5) for token in (ts_100, ts_60, tap))
    carry_in = empty_ln_carry_state(0)
    carry_out = empty_ln_carry_state(2_000)
    state = initial_replay_state(carry_in)
    for position, token_id in enumerate(generated):
        state = transition_replay_state(
            state,
            token_id,
            position=position,
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=2_000,
            chart_end_ms=2_000,
            ln_carry_out=carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
        )
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[ts_100] = True
    valid_mask[ts_90] = True
    logits = torch.zeros(vocab.size, dtype=torch.float32)
    logits[ts_100] = 10.0
    logits[ts_90] = 9.0
    step = MapperV21GenerationStep(
        decoder_input_tokens=torch.tensor([vocab.bos_id, *generated], dtype=torch.long),
        generated_tokens=generated,
        state=state,
        valid_token_mask=valid_mask,
        token_index=len(generated),
        write_start_ms=0,
        write_end_ms=2_000,
        chart_end_ms=2_000,
        ln_carry_in=carry_in,
        ln_carry_out=carry_out,
        is_full_chart_start=True,
        is_full_chart_end=False,
    )
    reference = MapperV21AntiRigidSpacingLogitsTransform(vocab=vocab, min_repeated_spacings=4)
    trace_transform = MapperV21IllegalCaseTraceTransform(vocab=vocab, mode="guard")

    reference_logits = reference(step, logits)
    traced_logits = trace_transform(step, logits)
    trace = trace_transform.to_dict()

    assert torch.equal(traced_logits, reference_logits)
    assert torch.isneginf(traced_logits[ts_100])
    assert trace["blocked_count"] == 1
    assert trace["changed_step_count"] == 1
    changed = trace["tail_records"][0]["changed_tokens"]
    assert changed[0]["token_name"] == "TS_100"
    assert changed[0]["raw_valid_rank"] == 1
    assert trace["tail_records"][0]["selected_after_transform"]["token_name"] == "TS_90"


def test_classify_trace_marks_recent_guard_change_as_anti_rigid_damage() -> None:
    summary = {
        "rollout": {
            "completed": False,
            "dead_end": True,
            "max_tokens_exceeded": False,
        }
    }
    trace = {
        "blocked_count": 1,
        "changed_step_count": 1,
        "raw_invalid_top1_count": 0,
        "last_terminal_probe": {
            "valid_token_count": 0,
            "invalid_reason_counts": {
                "TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out": 3,
            },
        },
        "tail_records": [{"changed_token_count": 1}],
    }

    classification = classify_trace_result(summary=summary, trace=trace)

    assert classification["family"] == "anti_rigid_removed_recent_transition"
    assert classification["legal"] is False
