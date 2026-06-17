# Target Grammar v3 Delta-Event Factor Target Tiny Model Result Report

## Scope

This research-only gate trains a tiny causal decoder on factorized delta-event rows derived from current v3 target fragments. It does not change production tokenizer behavior, dataset schema, mapper defaults, training runner, inference, or rollout.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD`.

- Reason: factorized delta-event target rows reconstructed and trained in a tiny real-data teacher-forced model
- Recommended next step: Create a bounded production-plumbing card for factorized v3 target rows before full training.

## Dataset

| Metric | Value |
| --- | ---: |
| source windows | `256` |
| consumed windows | `32` |
| row count | `1479` |
| event rows | `1447` |
| end rows | `32` |
| max row len | `97` |
| reconstruction mismatches | `0` |

## Training Metrics

| Metric | Value |
| --- | ---: |
| steps | `80` |
| initial total loss | `20.087185` |
| final total loss | `5.122849` |
| total loss ratio | `0.255031` |
| kind loss ratio | `0.213915` |
| delta loss ratio | `0.207284` |
| signature loss ratio | `0.593649` |
| end-gap loss ratio | `0.034707` |
| kind head grad abs | `26.559746` |
| delta head grad abs | `78.464401` |
| signature head grad abs | `72.248108` |
| end-gap head grad abs | `94.191071` |
| kind embedding grad abs | `2.198227` |

## Checks

| Check | Value |
| --- | ---: |
| no_production_tokenizer_or_dataset_change | `True` |
| no_mapper_default_or_runner_change | `True` |
| no_rollout | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| full4k_bit_proxy_route_positive | `True` |
| consumed_windows_positive | `True` |
| row_count_positive | `True` |
| event_rows_positive | `True` |
| end_rows_match_windows | `True` |
| reconstruction_mismatches_zero | `True` |
| all_losses_finite | `True` |
| total_loss_decreased | `True` |
| kind_loss_decreased | `True` |
| delta_loss_decreased | `True` |
| signature_loss_decreased | `True` |
| end_gap_loss_decreased | `True` |
| kind_head_gradient_nonzero | `True` |
| delta_head_gradient_nonzero | `True` |
| signature_head_gradient_nonzero | `True` |
| end_gap_head_gradient_nonzero | `True` |
| shared_embedding_gradient_nonzero | `True` |

## What Passed

- Factorized rows reconstructed exactly from current v3 fragments.
- Shifted previous-row teacher forcing was sufficient for a tiny causal model to optimize.
- Kind, delta, signature, and end-gap heads all received gradients.
- No production tokenizer/default/dataset/training/rollout change was made.

## What Surfaced

The factorized target surface is viable enough for a production-plumbing card, but still not replacement-ready.

## What Is Not Proved

- This does not prove production mapper quality.
- This does not prove autoregressive rollout quality.
- This does not yet integrate factorized rows into the production dataset/model/training runner.
- This does not complete v3 replacement.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_tiny_model
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_tiny_model.py tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_tiny_model.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=`TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD`, total loss ratio=`0.255031`
- 14 passed in 0.64s
- passed
- passed
- passed

## Next Step

Create a bounded production-plumbing card for factorized v3 target rows before full training.
