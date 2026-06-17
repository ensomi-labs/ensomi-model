# Target Grammar v3 Delta-Event Auxiliary Tiny Training Gate Result Report

## Scope

This gate performs a synthetic teacher-forced overfit run for the default-off v3 delta-event auxiliary objective. It does not use the full training runner, real dataset cache, rollout, tokenizer changes, or default mapper changes.

## Decision

Route: `TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE`.

- Reason: synthetic default-off and overfit trainability gates passed
- Recommended next step: Create a fixed-slice label coverage or tiny real-data training gate before any rollout or replacement work.

## Training Setup

- steps: `40`
- learning rate: `0.005`
- synthetic sequence length: `12`
- target tokens: `TS_80 EV_1000 TS_100 TS_60 EV_0100 TS_100 TS_60 EV_2000 TS_300 EV_3100 TS_300 EOS`

## Metrics

| Metric | Value |
| --- | ---: |
| initial total loss | `18.514944` |
| final total loss | `0.356188` |
| total loss ratio | `0.019238` |
| initial token loss | `4.571407` |
| final token loss | `0.153096` |
| token loss ratio | `0.033490` |
| initial auxiliary loss | `13.943537` |
| final auxiliary loss | `0.203092` |
| auxiliary loss ratio | `0.014565` |
| event labels | `4` |
| end-gap labels | `1` |
| delta head grad abs | `23.144943` |
| signature head grad abs | `22.912657` |
| end-gap head grad abs | `49.937260` |
| token embedding grad abs | `4.266740` |

## Checks

| Check | Value |
| --- | ---: |
| all_losses_finite | `True` |
| auxiliary_loss_decreased | `True` |
| default_off_auxiliary_logits_absent | `True` |
| default_off_lambda_zero | `True` |
| delta_head_gradient_nonzero | `True` |
| end_gap_head_gradient_nonzero | `True` |
| end_gap_labels_positive | `True` |
| event_labels_positive | `True` |
| final_auxiliary_loss_finite | `True` |
| initial_auxiliary_loss_positive | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| no_rollout | `True` |
| no_tokenizer_dataset_default_change | `True` |
| shared_decoder_gradient_nonzero | `True` |
| signature_head_gradient_nonzero | `True` |
| token_loss_not_worse | `True` |
| total_loss_decreased | `True` |

## What Passed

- Default-off v3 still emits no delta-event auxiliary logits.
- The enabled auxiliary objective has event and end-gap labels on the synthetic teacher-forced batch.
- Total, token, and auxiliary losses decreased under real optimizer steps.
- First-step gradients reached all three auxiliary heads and the shared token embedding.

## What Surfaced

The auxiliary objective is trainable in a synthetic overfit setting. This is stronger than the previous plumbing smoke, but it still does not prove real-data label coverage, mapper-quality improvement, rollout legality, or full v3 replacement readiness.

## What Is Not Proved

- No real dataset, fixed32, full32, or 4k run was performed.
- No autoregressive rollout quality was measured.
- No target grammar replacement was made.
- No C3 backreference or side-stream claim is tested here.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_training_gate
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3 tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/training/test_mapper_v3.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_training_gate.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=TEST_DELTA_EVENT_AUXILIARY_FIXED_SLICE_GATE aux_loss_ratio=0.014565
- 44 passed in 1.35s
- passed
- passed
- passed

## Next Step

Create a fixed-slice label coverage or tiny real-data training gate before any rollout or replacement work.
