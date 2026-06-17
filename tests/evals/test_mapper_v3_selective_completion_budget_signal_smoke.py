from __future__ import annotations

import json
from pathlib import Path

import torch

from pulsefield_model.evals.mapper_v3_selective_completion_budget_signal_smoke import (
    ControlBudgetSelectiveEventGate,
    DEFAULT_CASE_SUMMARY_PATH,
    _selected_cases,
    summarize_smoke,
)
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_control_budget_gate_forces_rank_near_event_when_under_online_budget() -> None:
    vocab = MapperV3Vocab()
    gate = ControlBudgetSelectiveEventGate(vocab=vocab, budget_slack_events=4, max_examples=4)
    event = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    shift = vocab.time_shift_token_id(100)
    gate.observe_window_batch(0, 8_000, {"density_teacher_8s": torch.zeros((1, 400, 1))})
    gate.observe_window_batch(8_000, 16_000, {"density_teacher_8s": torch.zeros((1, 400, 1))})
    gate(_step(vocab=vocab, event=event, shift=shift, generated_tokens=(event, event), window_start=0), _logits(vocab, event, shift))

    adjusted = gate(
        _step(vocab=vocab, event=event, shift=shift, generated_tokens=(), window_start=8_000),
        _logits(vocab, event, shift),
    )

    stats = gate.to_dict()
    assert int(torch.argmax(adjusted).item()) == event
    assert stats["forced_count"] == 1
    assert stats["budget_by_window"]["8000"]["event_budget"] == 6


def test_control_budget_gate_does_not_force_after_budget_exhausted() -> None:
    vocab = MapperV3Vocab()
    gate = ControlBudgetSelectiveEventGate(vocab=vocab, budget_slack_events=0, max_examples=4)
    event = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    shift = vocab.time_shift_token_id(100)
    gate.observe_window_batch(0, 8_000, {"density_teacher_8s": torch.zeros((1, 400, 1))})
    gate.observe_window_batch(8_000, 16_000, {"density_teacher_8s": torch.zeros((1, 400, 1))})
    gate(_step(vocab=vocab, event=event, shift=shift, generated_tokens=(event, event), window_start=0), _logits(vocab, event, shift))

    adjusted = gate(
        _step(
            vocab=vocab,
            event=event,
            shift=shift,
            generated_tokens=(event, event),
            window_start=8_000,
        ),
        _logits(vocab, event, shift),
    )

    assert int(torch.argmax(adjusted).item()) == shift
    assert gate.to_dict()["forced_count"] == 0
    assert gate.to_dict()["skipped_counts"]["budget_exhausted"] == 1


def test_selected_cases_uses_committed_six_primary_four_sentinel_slice() -> None:
    summary = json.loads(DEFAULT_CASE_SUMMARY_PATH.read_text(encoding="utf-8"))
    cases = _selected_cases(summary)

    assert len(cases) == 10
    assert sum(1 for case in cases if case["group"] == "primary") == 6
    assert sum(1 for case in cases if case["group"] == "sentinel") == 4
    assert cases[0]["case_id"] == "14_oomori_seiko_justadice_tv_size_remu_hard"


def test_summarize_smoke_routes_mutate_when_budget_source_missing() -> None:
    summary = summarize_smoke(
        case_summary_path=Path("cases.json"),
        baseline_summary_path=Path("baseline.json"),
        work_dir=Path("work"),
        elapsed_s=0.1,
        gate_config={},
        case_results=[_result("p0", group="primary", budget_missing=True) for _ in range(6)]
        + [_result("s0", group="sentinel", budget_missing=False) for _ in range(4)],
    )

    assert summary["decision"]["route"] == "MUTATE_BUDGET_SOURCE"


def test_summarize_smoke_routes_full32_when_all_guards_pass() -> None:
    summary = summarize_smoke(
        case_summary_path=Path("cases.json"),
        baseline_summary_path=Path("baseline.json"),
        work_dir=Path("work"),
        elapsed_s=0.1,
        gate_config={},
        case_results=[_result(f"p{index}", group="primary", positive=index < 4) for index in range(6)]
        + [_result(f"s{index}", group="sentinel") for index in range(4)],
    )

    assert summary["decision"]["route"] == "TEST_NEXT_FULL32_BUDGET_SIGNAL"


def _step(
    *,
    vocab: MapperV3Vocab,
    event: int,
    shift: int,
    generated_tokens: tuple[int, ...],
    window_start: int,
) -> MapperV3GenerationStep:
    carry = empty_ln_carry_state(window_start)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[event] = True
    valid_mask[shift] = True
    return MapperV3GenerationStep(
        decoder_input_tokens=torch.tensor([vocab.bos_id, *generated_tokens], dtype=torch.long),
        generated_tokens=generated_tokens,
        state=initial_replay_state(carry),
        valid_token_mask=valid_mask,
        token_index=len(generated_tokens),
        write_start_ms=window_start,
        write_end_ms=window_start + 8_000,
        chart_end_ms=16_000,
        ln_carry_in=carry,
        ln_carry_out=empty_ln_carry_state(window_start + 8_000),
        is_full_chart_start=window_start == 0,
        is_full_chart_end=window_start >= 8_000,
    )


def _logits(vocab: MapperV3Vocab, event: int, shift: int) -> torch.Tensor:
    logits = torch.full((vocab.size,), -1000.0, dtype=torch.float32)
    logits[shift] = 5.0
    logits[event] = 4.0
    return logits


def _result(
    case_id: str,
    *,
    group: str,
    positive: bool = False,
    budget_missing: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "group": group,
        "candidate": {"metrics": {"event_count_ratio": 1.0}},
        "comparison": {"f1_delta": 0.0, "second_window_event_share_delta": 0.2},
        "gate_stats": {"forced_count": 1},
        "checks": {
            "illegal": False,
            "event_ratio_over_cap": False,
            "duplicate_regressed": False,
            "boundary_regressed": False,
            "budget_source_missing": budget_missing,
            "positive_primary_signal": positive,
        },
    }
