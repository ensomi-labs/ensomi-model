# Target Grammar v3 Delta-Event Factor Target Training Smoke Result Report

## Scope

This gate runs a short production v3 training smoke with factorized delta-event target supervision enabled. It does not claim stable training, rollout quality, online factor-row inference, or complete v3 replacement readiness.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD`.

- Reason: factor-target enabled v3 training runner completed a bounded real-data smoke with valid metrics
- Recommended next step: Create a bounded longer-training card before rollout or replacement claims.

## Training Run

| Metric | Value |
| --- | ---: |
| completed steps | `2` |
| max steps | `2` |
| is complete | `True` |
| report path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_training_smoke/report.json` |
| checkpoint path | `artifacts/runs/stage2_mapper_v3/delta_event_factor_target_training_smoke/checkpoint.pt` |
| dataset factor target | `True` |
| model factor target | `True` |
| loss factor lambda | `0.25` |

## Metrics

| Metric | Last Train | Final Train | Final Eval |
| --- | ---: | ---: | ---: |
| loss/total | `9.583796` | `9.555488` | `9.785857` |
| loss/token | `4.624063` | `4.521998` | `4.696991` |
| loss/delta_event_factor_target | `19.838928` | `20.133961` | `20.355467` |
| phase/lambda_delta_event_factor_target | `0.250000` | `0.250000` | `0.250000` |
| delta_event_factor/kind_label_count | `95.000000` | `356.000000` | `1877.000000` |
| delta_event_factor/delta_label_count | `93.000000` | `348.000000` | `1829.000000` |
| delta_event_factor/signature_label_count | `93.000000` | `348.000000` | `1829.000000` |
| delta_event_factor/end_gap_label_count | `2.000000` | `8.000000` | `48.000000` |

## Loss Total Accounting

| Metric | Value |
| --- | ---: |
| reported final eval total | `9.785857` |
| recomputed final eval total | `9.785857` |
| absolute delta | `0.000000` |
| matches | `True` |

## Checks

| Check | Value |
| --- | ---: |
| model_loss_route_positive | `True` |
| completed_requested_steps | `True` |
| training_marked_complete | `True` |
| report_exists | `True` |
| checkpoint_exists | `True` |
| dataset_factor_target_enabled | `True` |
| model_factor_target_enabled | `True` |
| loss_factor_lambda_positive | `True` |
| final_eval_loss_total_finite | `True` |
| final_eval_factor_loss_finite | `True` |
| final_train_factor_loss_finite | `True` |
| last_train_factor_loss_finite | `True` |
| final_eval_factor_lambda_positive | `True` |
| final_eval_kind_labels_positive | `True` |
| final_eval_delta_labels_positive | `True` |
| final_eval_signature_labels_positive | `True` |
| final_eval_end_gap_labels_positive | `True` |
| loss_total_recomputed_matches | `True` |
| no_inference_rollout_or_c3_change | `True` |

## What Passed

- The real v3 training runner completed the factor-target smoke.
- The report and checkpoint were written.
- Final train/eval metrics include finite factor-target losses and label counts.
- Aggregated final eval `loss/total` includes the enabled factor-target term.

## What Surfaced

Factor-target supervision is runner-compatible for a short smoke; the next gate is longer training stability.

## What Is Not Proved

- This does not prove long-run training stability.
- This does not prove generated chart quality.
- This does not implement factor-row inference or full replacement.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke
uv run --group dev pytest tests/training/test_mapper_training_runner.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/training/mapper_runner.py src/pulsefield_model/training/mapper_common.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_training_smoke.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_summary.json >/dev/null
git diff --check
```

## Next Step

Create a bounded longer-training card before rollout or replacement claims.
