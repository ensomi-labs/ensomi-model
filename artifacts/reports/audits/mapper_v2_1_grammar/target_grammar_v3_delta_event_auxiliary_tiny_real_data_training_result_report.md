# Target Grammar v3 Delta-Event Auxiliary Tiny Real-Data Training Result Report

## Scope

This gate trains a small v3 model on real fixed-slice teacher-forced batches with the delta-event auxiliary objective enabled. It does not roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.

## Decision

Route: `TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE`.

- Reason: real fixed-slice tiny training gate passed for the enabled delta-event auxiliary objective
- Recommended next step: Stress delta-event label coverage on a larger slice or full4k, with explicit end-gap range analysis before any default change.

## Dataset

| Metric | Value |
| --- | ---: |
| source windows | `256` |
| consumed windows | `32` |
| batch count | `8` |
| batch size | `4` |
| max seq len | `201` |
| valid target tokens | `3673` |

## Training Metrics

| Metric | Value |
| --- | ---: |
| steps | `32` |
| learning rate | `0.003` |
| initial total loss | `17.565510` |
| final total loss | `6.929854` |
| total loss ratio | `0.394515` |
| initial token loss | `4.725065` |
| final token loss | `2.678886` |
| token loss ratio | `0.566952` |
| initial auxiliary loss | `12.840445` |
| final auxiliary loss | `4.250968` |
| auxiliary loss ratio | `0.331061` |
| event labels | `1447` |
| signature labels | `1447` |
| end-gap labels | `32` |
| min event labels per batch | `125` |
| min end-gap labels per batch | `4` |
| delta head grad abs | `45.266479` |
| signature head grad abs | `39.163284` |
| end-gap head grad abs | `86.764618` |
| token embedding grad abs | `5.720287` |

## Checks

| Check | Value |
| --- | ---: |
| no_rollout | `True` |
| no_tokenizer_dataset_default_change | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| label_coverage_route_passed | `True` |
| synthetic_training_route_passed | `True` |
| consumed_windows_positive | `True` |
| real_batches_positive | `True` |
| every_batch_has_event_labels | `True` |
| every_batch_has_signature_labels | `True` |
| every_batch_has_end_gap_labels | `True` |
| all_losses_finite | `True` |
| initial_auxiliary_loss_positive | `True` |
| final_auxiliary_loss_finite | `True` |
| auxiliary_loss_decreased | `True` |
| total_loss_decreased | `True` |
| token_loss_not_materially_worse | `True` |
| delta_head_gradient_nonzero | `True` |
| signature_head_gradient_nonzero | `True` |
| end_gap_head_gradient_nonzero | `True` |
| shared_decoder_gradient_nonzero | `True` |

## What Passed

- Real fixed-slice batches carried event, signature, and end-gap labels.
- The enabled auxiliary objective optimized under real optimizer steps.
- Gradients reached all three auxiliary heads and the shared token embedding.
- No tokenizer, dataset schema, default mapper, rollout, C3 replay, or future-lookup change was used.

## What Surfaced

The delta-event auxiliary objective is now trainable on real fixed-slice teacher-forced batches. The remaining structural issue is coverage scale: the prior fixed-slice coverage audit hit the current `8000ms` end-gap boundary, so a larger/full4k range stress is required before defaults or replacement work.

## What Is Not Proved

- This does not prove autoregressive rollout quality.
- This does not prove full4k label coverage.
- This does not make v3 target replacement ready.
- This does not compare against a full v2.1/v3 trained pipeline.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_real_data_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_real_data_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=`TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE`, auxiliary loss ratio=`0.331061`, token loss ratio=`0.566952`
- 39 passed in 1.32s
- passed
- passed
- passed

## Next Step

Stress delta-event label coverage on a larger slice or full4k, with explicit end-gap range analysis before any default change.
