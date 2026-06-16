from __future__ import annotations

import torch
from pathlib import Path

from pulsefield_model.evals.mapper_v3_selective_event_gate_smoke import (
    SelectiveEventGate,
    compute_time_metrics,
    summarize_smoke,
)
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_selective_event_gate_forces_rank_near_second_window_event() -> None:
    vocab = MapperV3Vocab()
    event = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    shift = vocab.time_shift_token_id(100)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[event] = True
    valid_mask[shift] = True
    logits = torch.full((vocab.size,), -1000.0, dtype=torch.float32)
    logits[shift] = 5.0
    logits[event] = 4.0
    carry = empty_ln_carry_state(8_000)
    gate = SelectiveEventGate(vocab=vocab, max_examples=4)

    adjusted = gate(
        MapperV3GenerationStep(
            decoder_input_tokens=torch.tensor([vocab.time_shift_token_id(100)], dtype=torch.long),
            generated_tokens=(),
            state=initial_replay_state(carry),
            valid_token_mask=valid_mask,
            token_index=0,
            write_start_ms=8_000,
            write_end_ms=16_000,
            chart_end_ms=16_000,
            ln_carry_in=carry,
            ln_carry_out=empty_ln_carry_state(16_000),
            is_full_chart_start=False,
            is_full_chart_end=True,
        ),
        logits,
    )

    assert int(torch.argmax(adjusted).item()) == event
    assert gate.to_dict()["forced_count"] == 1
    assert gate.to_dict()["forced_by_window"] == {"second": 1}


def test_compute_time_metrics_matches_with_tolerance() -> None:
    metrics = compute_time_metrics(
        generated_times=[100, 8100, 8200, 15950],
        reference_times=[120, 8110, 9000, 15990],
        chart_end_ms=16_000,
    )

    assert metrics["generated_event_count"] == 4
    assert metrics["reference_event_count"] == 4
    assert metrics["second_window_event_count"] == 3
    assert metrics["boundary_event_count"] == 1
    assert metrics["timing_match_100ms"]["matched_generated"] == 3
    assert metrics["timing_match_100ms"]["f1"] == 0.75


def test_summarize_smoke_routes_full32_when_two_low_bias_cases_pass() -> None:
    summary = summarize_smoke(
        trace_summary_path=Path("trace.json"),
        baseline_summary_path=Path("baseline.json"),
        work_dir=Path("work"),
        elapsed_s=0.1,
        gate_config={"max_event_rank": 5},
        case_results=[
            _case_result("a", role="low_bias_starved", positive=True),
            _case_result("b", role="low_bias_starved", positive=True),
            _case_result("c", role="low_bias_starved", positive=False),
            _case_result("d", role="pass_like_control", positive=False),
        ],
    )

    assert summary["decision"]["route"] == "TEST_FULL32_SELECTIVE_GATE"


def _case_result(case_id: str, *, role: str, positive: bool) -> dict[str, object]:
    return {
        "case": {"case_id": case_id, "role": role},
        "checks": {
            "legal": True,
            "positive_low_bias_signal": positive,
            "overproduced": False,
            "pass_like_regressed": False,
        },
    }
