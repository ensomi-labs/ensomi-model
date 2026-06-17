# Target Grammar v3 Delta-Event Factor Target Longer Training Result Report

## Scope

This gate runs a longer fixed-slice v3 training stability check with factorized delta-event target supervision enabled. It does not claim full4k stability, rollout quality, online factor-row inference, or complete v3 replacement readiness.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD`.

- Reason: factor-target enabled v3 training stayed finite and non-exploding across a longer fixed-slice run
- Recommended next step: Create a bounded full4k/fixed-cache factor-target training card before rollout or replacement claims.

## Run

| Metric | Value |
| --- | ---: |
| completed steps | `16` |
| max steps | `16` |
| eval steps | `[1, 8, 16]` |
| report path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_longer_training/report.json` |
| checkpoint path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_longer_training/checkpoint.pt` |
| dataset factor target | `True` |
| model factor target | `True` |
| loss factor lambda | `0.25` |

## Stability Metrics

| Metric | First Eval | Final Train | Final Eval |
| --- | ---: | ---: | ---: |
| loss/total | `9.791060` | `9.232156` | `9.270527` |
| loss/token | `4.570699` | `4.315387` | `4.325970` |
| loss/delta_event_factor_target | `20.881446` | `19.667079` | `19.778228` |
| phase/lambda_delta_event_factor_target | `0.250000` | `0.250000` | `0.250000` |
| delta_event_factor/kind_label_count | `2448.000000` | `788.000000` | `2448.000000` |
| delta_event_factor/delta_label_count | `2400.000000` | `772.000000` | `2400.000000` |
| delta_event_factor/signature_label_count | `2400.000000` | `772.000000` | `2400.000000` |
| delta_event_factor/end_gap_label_count | `48.000000` | `16.000000` | `48.000000` |

| Derived Metric | Value |
| --- | ---: |
| first eval factor loss | `20.881446` |
| final eval factor loss | `19.778228` |
| factor loss ratio | `0.947168` |
| max factor loss ratio | `1.150000` |
| all eval losses finite | `True` |

## Loss Total Accounting

| Metric | Value |
| --- | ---: |
| reported final eval total | `9.270527` |
| recomputed final eval total | `9.270527` |
| absolute delta | `0.000000` |
| matches | `True` |

## Checks

| Check | Value |
| --- | ---: |
| training_smoke_route_positive | `True` |
| completed_requested_steps | `True` |
| training_marked_complete | `True` |
| report_exists | `True` |
| checkpoint_exists | `True` |
| dataset_factor_target_enabled | `True` |
| model_factor_target_enabled | `True` |
| loss_factor_lambda_positive | `True` |
| eval_point_count_sufficient | `True` |
| all_eval_losses_finite | `True` |
| factor_loss_ratio_within_limit | `True` |
| final_eval_factor_loss_finite | `True` |
| final_train_factor_loss_finite | `True` |
| final_eval_kind_labels_positive | `True` |
| final_eval_delta_labels_positive | `True` |
| final_eval_signature_labels_positive | `True` |
| final_eval_end_gap_labels_positive | `True` |
| loss_total_recomputed_matches | `True` |
| no_inference_rollout_or_c3_change | `True` |

## What Passed

- The real v3 runner completed the longer fixed-slice factor-target run.
- Multiple eval points had finite token and factor-target losses.
- Final eval factor loss stayed within the non-explosion threshold.
- Final eval aggregate `loss/total` accounting remained exact.

## What Surfaced

Factor-target supervision is stable enough on the fixed slice to justify a bounded full4k training card.

## What Is Not Proved

- This does not prove full4k training stability.
- This does not prove generated chart quality.
- This does not implement factor-row inference or full replacement.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_longer_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_summary.json >/dev/null
git diff --check
```

## Next Step

Create a bounded full4k/fixed-cache factor-target training card before rollout or replacement claims.
