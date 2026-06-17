# Target Grammar v3 Time-Shift Distance Tiny Training Gate Result Report

## Scope

This gate compares matched v3 baseline and time-shift-distance-enabled training reports. Optional rollout pairs are used only when supplied. It does not change tokenizer behavior, decode defaults, or production mapper settings.

## Decision

Route: `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE`.

- Reason: failed rollout checks: no_new_starved_cases, mean_rigid_not_worse
- Next step: Mutate the timing objective or scale before a full32 run.

## Training Reports

- baseline: `artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/train/baseline/report.json`
- enabled: `artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/train/enabled/report.json`
- baseline steps: `500`
- enabled steps: `500`
- enabled eval `loss/time_shift_distance`: `0.017952`
- token-loss delta: `-0.003974`
- token-loss regression ratio: `0.00%`
- total-loss delta: `-0.003951`
- total-loss regression ratio: `0.00%`

## Training Checks

| Check | Passed |
| --- | ---: |
| `baseline_contract_v3` | `True` |
| `baseline_dataset_contract_v3` | `True` |
| `enabled_contract_v3` | `True` |
| `enabled_dataset_contract_v3` | `True` |
| `baseline_completed_steps` | `True` |
| `enabled_completed_steps` | `True` |
| `baseline_complete_flag` | `True` |
| `enabled_complete_flag` | `True` |
| `baseline_total_loss_finite` | `True` |
| `enabled_total_loss_finite` | `True` |
| `baseline_token_loss_finite` | `True` |
| `enabled_token_loss_finite` | `True` |
| `baseline_lambda_disabled` | `True` |
| `enabled_lambda_positive` | `True` |
| `enabled_time_shift_distance_loss_finite` | `True` |
| `enabled_time_shift_distance_loss_positive` | `True` |
| `token_loss_not_materially_worse` | `True` |
| `total_loss_not_materially_worse` | `True` |
| `valid_tokens_match` | `True` |

## Rollout Diagnostics

- rollout pairs: `32`
- new starved cases: `3`
- mean rigid delta: `0.062197`
- mean F1 delta: `-0.030968`

| Check | Passed |
| --- | ---: |
| `rollout_pair_count_positive` | `True` |
| `baseline_all_legal` | `True` |
| `enabled_all_legal` | `True` |
| `no_new_starved_cases` | `False` |
| `mean_rigid_not_worse` | `False` |

## Interpretation

The time-shift distance objective did not clear the current tiny gate. Do not scale it without changing the objective, lambda, scale, or training recipe.

## What This Does Not Prove

- It does not prove v3 full32 replacement readiness.
- It does not prove full-dataset convergence.
- It does not prove generated quality unless rollout pairs are supplied and pass.
