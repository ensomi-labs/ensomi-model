from __future__ import annotations

import torch

from pulsefield_model.evals.mapper_v3_generated_prefix_state_trace_audit import (
    GeneratedPrefixStateTraceCollector,
    classify_case_failure,
    decision_from_aggregate,
    trace_aggregate_from_rows,
)
from pulsefield_model.inference.mapper_v3_rollout import (
    MapperV3FullRollout,
    MapperV3GeneratedWindow,
    MapperV3GenerationStep,
)
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state, transition_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_trace_collector_records_logits_and_attaches_emitted_token() -> None:
    vocab = MapperV3Vocab()
    carry = empty_ln_carry_state(0)
    state0 = initial_replay_state(carry)
    shift_token = vocab.time_shift_token_id(100)
    event_token = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[shift_token] = True
    valid_mask[event_token] = True
    logits = torch.full((vocab.size,), -10.0, dtype=torch.float32)
    logits[shift_token] = 3.0
    logits[event_token] = 2.0

    collector = GeneratedPrefixStateTraceCollector(vocab=vocab, top_k=2)
    collector.observe(
        MapperV3GenerationStep(
            decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
            generated_tokens=(),
            state=state0,
            valid_token_mask=valid_mask,
            token_index=0,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=8_000,
            ln_carry_in=carry,
            ln_carry_out=empty_ln_carry_state(8_000),
            is_full_chart_start=True,
            is_full_chart_end=True,
        ),
        logits,
    )
    state1 = transition_replay_state(
        state0,
        shift_token,
        position=0,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=8_000,
        ln_carry_out=empty_ln_carry_state(8_000),
        is_full_chart_start=True,
        is_full_chart_end=True,
    )
    window = MapperV3GeneratedWindow(
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=8_000,
        ln_carry_in=carry,
        ln_carry_out=empty_ln_carry_state(8_000),
        tokens=[shift_token],
        states_before=[state0],
        states_after=[state1],
        terminal_state=state1,
        completed=False,
        dead_end=False,
        max_tokens_exceeded=False,
    )
    rows = collector.attach_rollout(
        MapperV3FullRollout(
            chart_end_ms=8_000,
            windows=[window],
            tokens=[shift_token],
            completed=False,
            dead_end=False,
            max_tokens_exceeded=False,
        )
    )

    assert len(rows) == 1
    assert rows[0]["argmax"]["kind"] == "time_shift"
    assert rows[0]["best_event"]["rank"] == 2
    assert rows[0]["best_event"]["margin_vs_argmax"] == -1.0
    assert rows[0]["emitted"]["time_shift_ms"] == 100
    assert rows[0]["state_after"]["current_ms"] == 100

    aggregate = trace_aggregate_from_rows(rows)
    assert aggregate["event_valid_step_count"] == 1
    assert aggregate["event_topk_step_count"] == 1
    assert aggregate["emitted_kind_counts"] == {"time_shift": 1}


def test_classify_case_failure_prefers_repeated_spacing_as_structural_primary() -> None:
    rows = [
        _event_row(index=0, time_ms=0),
        _event_row(index=2, time_ms=160),
        _event_row(index=4, time_ms=320),
        _event_row(index=6, time_ms=480),
        _event_row(index=8, time_ms=640),
        _underselect_row(index=9, time_ms=800),
    ]

    result = classify_case_failure(
        trace_rows=rows,
        generated_times=[0, 160, 320, 480, 640],
        reference_times=[0, 160, 320, 480, 640, 800],
        metrics={
            "second_window_event_share": 0.0,
            "boundary_event_ratio": 0.0,
        },
        chart_end_ms=8_000,
        rollout_completed=True,
        rollout_dead_end=False,
        rollout_max_tokens_exceeded=False,
    )

    assert result["failure_class"] == "time_shift_repetition"
    assert result["repeated_spacing_ms"] == 160
    assert result["best_event_rank"] == 1
    assert result["emitted_token"] == "EV_1000"
    assert result["secondary_flags"]["event_underselection"] is False


def test_classify_case_failure_detects_rank_near_event_underselection() -> None:
    result = classify_case_failure(
        trace_rows=[
            _underselect_row(index=0, time_ms=100),
            _underselect_row(index=1, time_ms=260),
        ],
        generated_times=[],
        reference_times=[100, 260],
        metrics={
            "second_window_event_share": 0.0,
            "boundary_event_ratio": 0.0,
        },
        chart_end_ms=8_000,
        rollout_completed=True,
        rollout_dead_end=False,
        rollout_max_tokens_exceeded=False,
    )

    assert result["failure_class"] == "event_underselection"
    assert result["best_event_rank"] == 3
    assert result["candidate_count"] == 2


def test_classify_case_failure_detects_boundary_drift_when_second_window_is_starved() -> None:
    result = classify_case_failure(
        trace_rows=[
            {
                "sequence_index": 0,
                "row_key": "8000:0",
                "write_start_ms": 8_000,
                "token_index": 0,
                "current_ms": 8_000,
                "emitted": {"kind": "time_shift", "name": "TS_4000", "time_shift_ms": 4_000},
                "argmax": {"kind": "time_shift", "name": "TS_4000", "time_shift_ms": 4_000},
                "best_event": None,
                "top_tokens": [],
            }
        ],
        generated_times=[0, 160, 320, 480, 640],
        reference_times=[0, 160, 320, 480, 640, 8_200, 8_360, 8_520],
        metrics={
            "second_window_event_share": 0.0,
            "boundary_event_ratio": 0.0,
        },
        chart_end_ms=16_000,
        rollout_completed=True,
        rollout_dead_end=False,
        rollout_max_tokens_exceeded=False,
    )

    assert result["failure_class"] == "time_shift_repetition"
    assert result["secondary_flags"]["boundary_drift"] is True


def test_decision_routes_to_spacing_repair_when_repetition_dominates() -> None:
    decision = decision_from_aggregate(
        {
            "selected_case_count": 6,
            "traced_case_count": 6,
            "concrete_class_count": 6,
            "failure_class_counts": {
                "time_shift_repetition": 5,
                "boundary_drift": 1,
            },
        }
    )

    assert decision["route"] == "TEST_BOUNDARY_OR_SPACING_STATE_REPAIR"


def _event_row(*, index: int, time_ms: int) -> dict[str, object]:
    return {
        "sequence_index": index,
        "row_key": f"0:{index}",
        "write_start_ms": 0,
        "token_index": index,
        "current_ms": time_ms,
        "emitted": {"kind": "event", "name": "EV_1000", "event_signature": "T..."},
        "argmax": {"kind": "event", "name": "EV_1000", "event_signature": "T..."},
        "best_event": {
            "kind": "event",
            "name": "EV_1000",
            "event_signature": "T...",
            "rank": 1,
            "margin_vs_argmax": 0.0,
        },
        "top_tokens": [{"kind": "event", "name": "EV_1000"}],
    }


def _underselect_row(*, index: int, time_ms: int) -> dict[str, object]:
    return {
        "sequence_index": index,
        "row_key": f"0:{index}",
        "write_start_ms": 0,
        "token_index": index,
        "current_ms": time_ms,
        "emitted": {"kind": "time_shift", "name": "TS_160", "time_shift_ms": 160},
        "argmax": {"kind": "time_shift", "name": "TS_160", "time_shift_ms": 160},
        "best_event": {
            "kind": "event",
            "name": "EV_1000",
            "event_signature": "T...",
            "rank": 3,
            "margin_vs_argmax": -0.5,
        },
        "top_tokens": [
            {"kind": "time_shift", "name": "TS_160"},
            {"kind": "event", "name": "EV_1000"},
        ],
    }
