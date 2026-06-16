from __future__ import annotations

import torch

from pulsefield_model.evals.mapper_v3_trained_runtime_rollout import V3LogitDiagnosticsCollector
from pulsefield_model.inference.mapper_v3_rollout import MapperV3GenerationStep
from pulsefield_model.models.mapper.v3 import MapperV3Vocab
from pulsefield_model.models.mapper.v3.replay import empty_ln_carry_state, initial_replay_state
from pulsefield_model.models.mapper.v3.vocab import LaneAction


def test_v3_logit_diagnostics_records_event_rank_and_kind_counts() -> None:
    vocab = MapperV3Vocab()
    carry = empty_ln_carry_state(0)
    event_token = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
    shift_token = vocab.time_shift_token_id(100)
    valid_mask = torch.zeros(vocab.size, dtype=torch.bool)
    valid_mask[event_token] = True
    valid_mask[shift_token] = True
    valid_mask[vocab.eos_id] = True
    logits = torch.full((vocab.size,), -10.0, dtype=torch.float32)
    logits[shift_token] = 3.0
    logits[event_token] = 2.0
    logits[vocab.eos_id] = 1.0

    collector = V3LogitDiagnosticsCollector(vocab=vocab, top_k=2)
    collector.observe(
        MapperV3GenerationStep(
            decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
            generated_tokens=(),
            state=initial_replay_state(carry),
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
    assert summary["event_valid_step_count"] == 1
    assert summary["event_top1_step_count"] == 0
    assert summary["event_topk_step_count"] == 1
    assert summary["argmax_kind_counts"] == {"time_shift": 1}
    assert summary["emitted_kind_counts"] == {"time_shift": 1}
    assert summary["best_event_rank"]["median"] == 2.0
    assert summary["best_event_margin_vs_argmax"]["median"] == -1.0
