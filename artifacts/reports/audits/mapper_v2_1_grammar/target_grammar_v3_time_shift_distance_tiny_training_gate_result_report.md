# Target Grammar v3 Time-Shift Distance Tiny Training Gate Result Report

## Scope

This gate compares matched v3 baseline and time-shift-distance-enabled training reports. Optional rollout pairs are used only when supplied. It does not change tokenizer behavior, decode defaults, or production mapper settings.

## Decision

Route: `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE`.

- Reason: failed training checks: enabled_total_loss_finite, enabled_token_loss_finite, enabled_time_shift_distance_loss_finite, enabled_time_shift_distance_loss_positive, token_loss_not_materially_worse, total_loss_not_materially_worse
- Next step: Do not run rollout; inspect scale/lambda or loss formulation.

## Training Reports

- baseline: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json`
- enabled: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json`
- baseline steps: `80`
- enabled steps: `80`
- enabled eval `loss/time_shift_distance`: `nan`
- token-loss delta: `nan`
- token-loss regression ratio: `nan`
- total-loss delta: `nan`
- total-loss regression ratio: `nan`

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
| `enabled_total_loss_finite` | `False` |
| `baseline_token_loss_finite` | `True` |
| `enabled_token_loss_finite` | `False` |
| `baseline_lambda_disabled` | `True` |
| `enabled_lambda_positive` | `True` |
| `enabled_time_shift_distance_loss_finite` | `False` |
| `enabled_time_shift_distance_loss_positive` | `False` |
| `token_loss_not_materially_worse` | `False` |
| `total_loss_not_materially_worse` | `False` |
| `valid_tokens_match` | `True` |

## Rollout Diagnostics

No rollout-pair file was supplied. A training-pass result routes to `TEST_ROLLOUT_GATE`, not full32.

## Interpretation

The time-shift distance objective did not clear the current tiny gate. Do not scale it without changing the objective, lambda, scale, or training recipe.

## What This Does Not Prove

- It does not prove v3 full32 replacement readiness.
- It does not prove full-dataset convergence.
- It does not prove generated quality unless rollout pairs are supplied and pass.
