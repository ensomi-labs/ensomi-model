# Target Grammar v3 Delta-Event Factor Target Full4k Fixed-Cache Training Result Report

## Scope

This gate runs a bounded production v3 training loop over the full4k source index with cached control-teacher tensors and factorized delta-event target supervision enabled. It does not claim production convergence, rollout quality, online factor-row inference, or complete v3 replacement readiness.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD`.

- Reason: factor-target enabled v3 training stayed finite on the full4k fixed-cache path
- Recommended next step: Create a bounded longer-full4k or production-config factor-target card before rollout claims.

## Dataset And Cache

| Metric | Value |
| --- | ---: |
| source window count | `174515` |
| train window count | `174335` |
| eval window count | `180` |
| candidate windows | `180696` |
| eligible windows | `174515` |
| expected audited windows | `174515` |
| expected candidate windows | `180696` |
| mapper record cache | `artifacts/cache/stage2_mapper_v3/window_records/delta_event_factor_target_full4k_training.parquet` |
| mapper record cache existed before run | `True` |
| mapper record cache exists | `True` |
| control teacher cache files | `180696` |
| require control teacher cache | `True` |
| dataset factor target | `True` |

## Run

| Metric | Value |
| --- | ---: |
| completed steps | `8` |
| max steps | `8` |
| eval steps | `[1, 4, 8]` |
| report path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_full4k_training/report.json` |
| checkpoint path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_full4k_training/checkpoint.pt` |
| model factor target | `True` |
| loss factor lambda | `0.25` |

## Stability Metrics

| Metric | First Eval | Final Train | Final Eval |
| --- | ---: | ---: | ---: |
| loss/total | `9.603167` | `9.449050` | `9.379645` |
| loss/token | `4.537671` | `4.480091` | `4.441899` |
| loss/delta_event_factor_target | `20.261986` | `19.875836` | `19.750985` |
| phase/lambda_delta_event_factor_target | `0.250000` | `0.250000` | `0.250000` |
| delta_event_factor/kind_label_count | `12899.000000` | `860.000000` | `12899.000000` |
| delta_event_factor/delta_label_count | `12719.000000` | `844.000000` | `12719.000000` |
| delta_event_factor/signature_label_count | `12719.000000` | `844.000000` | `12719.000000` |
| delta_event_factor/end_gap_label_count | `180.000000` | `16.000000` | `180.000000` |

| Derived Metric | Value |
| --- | ---: |
| first eval factor loss | `20.261986` |
| final eval factor loss | `19.750985` |
| factor loss ratio | `0.974780` |
| max factor loss ratio | `1.200000` |
| all eval losses finite | `True` |

## Loss Total Accounting

| Metric | Value |
| --- | ---: |
| reported final eval total | `9.379645` |
| recomputed final eval total | `9.379645` |
| absolute delta | `0.000000` |
| matches | `True` |

## Checks

| Check | Value |
| --- | ---: |
| longer_training_route_positive | `True` |
| full4k_bit_proxy_route_positive | `True` |
| full4k_label_coverage_route_positive | `True` |
| source_windows_full4k_scale | `True` |
| source_matches_full4k_bit_proxy | `True` |
| candidate_windows_match_full4k_bit_proxy | `True` |
| train_windows_positive | `True` |
| eval_windows_positive | `True` |
| mapper_record_cache_configured | `True` |
| mapper_record_cache_exists | `True` |
| mapper_record_cache_metadata_exists | `True` |
| control_teacher_cache_dir_exists | `True` |
| control_teacher_cache_file_count_matches_candidates | `True` |
| control_teacher_cache_required | `True` |
| dataset_factor_target_enabled | `True` |
| completed_requested_steps | `True` |
| training_marked_complete | `True` |
| report_exists | `True` |
| checkpoint_exists | `True` |
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

- The real v3 runner completed a bounded full4k fixed-cache factor-target run.
- The training source matched prior full4k proxy-audit eligible-window counts.
- Multiple eval points had finite token and factor-target losses.
- Final eval aggregate `loss/total` accounting remained exact.

## What Surfaced

Factor-target supervision is mechanically stable on the full4k fixed-cache path, so the next question is longer full4k duration or production-config realism.

## What Is Not Proved

- This does not prove full production training convergence.
- This does not prove generated chart quality.
- This does not implement online factor-row inference.
- This does not make v3 the final mapper/planner replacement.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_full4k_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_full4k_training.py tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_full4k_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_full4k_training_summary.json >/dev/null
git diff --check
```

## Next Step

Create a bounded longer-full4k or production-config factor-target card before rollout claims.
