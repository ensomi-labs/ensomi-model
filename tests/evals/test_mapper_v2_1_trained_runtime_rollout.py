from __future__ import annotations

import torch

from pulsefield_model.evals.mapper_v2_1_trained_runtime_rollout import V21LogitDiagnosticsCollector
from pulsefield_model.inference.mapper_v2_1_rollout import MapperV21GenerationStep
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab
from pulsefield_model.models.mapper.v2_1.replay import empty_ln_carry_state
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction


def test_v21_logit_diagnostics_records_lane_action_rank_and_kind_counts() -> None:
    vocab = MapperV21Vocab()
    carry = empty_ln_carry_state(0)
    lane_action_token = vocab.lane_action_token_id(0, LaneAction.TAP)
    shift_token = vocab.time_shift_token_id(100)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[lane_action_token] = True
    valid_mask[shift_token] = True
    valid_mask[vocab.eos_id] = True
    logits = torch.full((vocab.size,), -10.0, dtype=torch.float32)
    logits[shift_token] = 3.0
    logits[lane_action_token] = 2.0
    logits[vocab.eos_id] = 1.0

    collector = V21LogitDiagnosticsCollector(vocab=vocab, top_k=2)
    collector.observe(
        MapperV21GenerationStep(
            decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
            generated_tokens=(),
            state=_initial_state(carry),
            valid_token_mask=valid_mask,
            token_index=0,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=8_000,
            ln_carry_in=carry,
            ln_carry_out=empty_ln_carry_state(8_000),
            is_full_chart_start=True,
            is_full_chart_end=False,
        ),
        logits,
    )

    summary = collector.to_dict([shift_token])

    assert summary["step_count"] == 1
    assert summary["lane_action_valid_step_count"] == 1
    assert summary["lane_action_top1_step_count"] == 0
    assert summary["lane_action_topk_step_count"] == 1
    assert summary["argmax_kind_counts"] == {"time_shift": 1}
    assert summary["emitted_kind_counts"] == {"time_shift": 1}
    assert summary["best_lane_action_rank"]["median"] == 2.0
    assert summary["best_lane_action_margin_vs_argmax"]["median"] == -1.0


def _initial_state(carry):
    from pulsefield_model.models.mapper.v2_1.replay import MapperReplayState

    return MapperReplayState(
        position=-1,
        current_ms=int(carry.current_ms),
        open_mask=carry.open_mask,
        open_start_ms=carry.open_start_ms,
        open_age_ms=carry.open_age_ms,
        emitted_lane_mask=(False, False, False, False),
        last_lane_index=-1,
    )
