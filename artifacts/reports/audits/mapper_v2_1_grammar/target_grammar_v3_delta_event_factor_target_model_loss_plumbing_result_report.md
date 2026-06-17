# Target Grammar v3 Delta-Event Factor Target Model/Loss Plumbing Result Report

## Scope

This gate verifies optional production model/loss plumbing for factorized delta-event target rows. It does not train a full production model, change inference, change rollout, replace target fragments, or change mapper defaults.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD`.

- Reason: default-off behavior held and enabled factor target model/loss produced finite real-batch gradients
- Recommended next step: Create a bounded short training-smoke card with factor-target loss enabled.

## Fixed-Slice Row Metrics

| Metric | Value |
| --- | ---: |
| audited windows | `32` |
| row count | `1479` |
| event rows | `1447` |
| end rows | `32` |
| max row len | `97` |
| row mask valid count | `1479` |
| reconstruction mismatches | `0` |

## Default-Off Probe

| Metric | Value |
| --- | ---: |
| batch has factor target | `False` |
| factor outputs absent | `True` |
| lambda metric | `0.0` |

## Enabled Probe

| Metric | Value |
| --- | ---: |
| batch size | `4` |
| row count | `320` |
| event row count | `316` |
| factor loss | `19.298216` |
| total loss | `23.996616` |
| kind labels | `320` |
| delta labels | `316` |
| signature labels | `316` |
| end-gap labels | `4` |
| kind head grad abs | `17.822748` |
| delta head grad abs | `39.499653` |
| signature head grad abs | `36.908768` |
| end-gap head grad abs | `45.751347` |

## Checks

| Check | Value |
| --- | ---: |
| data_contract_route_positive | `True` |
| tiny_model_route_positive | `True` |
| audited_windows_positive | `True` |
| row_count_positive | `True` |
| row_count_matches_data_contract | `True` |
| event_rows_match_data_contract | `True` |
| end_rows_match_data_contract | `True` |
| row_mask_count_matches_rows | `True` |
| reconstruction_mismatches_zero | `True` |
| default_off_batch_lacks_factor_target | `True` |
| default_off_outputs_absent | `True` |
| default_off_loss_metric_zero | `True` |
| enabled_outputs_align_rows | `True` |
| enabled_loss_finite | `True` |
| enabled_factor_loss_positive | `True` |
| enabled_kind_labels_match_rows | `True` |
| enabled_delta_labels_match_events | `True` |
| enabled_signature_labels_match_events | `True` |
| enabled_end_gap_labels_match_windows | `True` |
| kind_head_gradient_nonzero | `True` |
| delta_head_gradient_nonzero | `True` |
| signature_head_gradient_nonzero | `True` |
| end_gap_head_gradient_nonzero | `True` |
| no_inference_rollout_or_c3_change | `True` |

## What Passed

- Default-off v3 model/loss outputs remain absent for factor-target fields.
- Enabled row-level factor logits align with collated factor rows.
- Factor loss is finite and positive on a real fixed-slice batch.
- Kind, delta, signature, and end-gap factor heads all receive gradients.

## What Surfaced

The factorized target now has production model/loss plumbing, but still needs a bounded training-smoke gate.

## What Is Not Proved

- This does not prove full training stability.
- This does not prove autoregressive factor-row inference.
- This does not prove rollout quality or complete v3 replacement readiness.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_model_loss_plumbing
uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_model_loss_plumbing.py -q
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/model.py src/pulsefield_model/models/mapper/v3/loss.py src/pulsefield_model/training/mapper_v3.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_model_loss_plumbing.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json >/dev/null
git diff --check
```

## Next Step

Create a bounded short training-smoke card with factor-target loss enabled.
