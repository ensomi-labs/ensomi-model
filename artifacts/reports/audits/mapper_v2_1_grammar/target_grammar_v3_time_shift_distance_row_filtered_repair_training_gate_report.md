# Target Grammar v3 Time-Shift Distance Tiny Training Gate Result Report

## Scope

This gate compares matched v3 baseline and time-shift-distance-enabled training reports. Optional rollout pairs are used only when supplied. It does not change tokenizer behavior, decode defaults, or production mapper settings.

## Decision

Route: `TEST_ROLLOUT_GATE`.

- Reason: training checks passed; rollout-pair diagnostics still missing
- Next step: Run a 2-4 case rollout-pair diagnostic with full timepoint previews.

## Training Reports

- baseline: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json`
- enabled: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json`
- baseline steps: `80`
- enabled steps: `80`
- enabled eval `loss/time_shift_distance`: `0.018968`
- token-loss delta: `-0.000061`
- token-loss regression ratio: `0.00%`
- total-loss delta: `-0.000051`
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

No rollout-pair file was supplied. A training-pass result routes to `TEST_ROLLOUT_GATE`, not full32.

## Interpretation

The matched training reports pass, but no rollout-pair diagnostics were supplied. The objective has earned a tiny rollout gate, not full32 escalation.

## What This Does Not Prove

- It does not prove v3 full32 replacement readiness.
- It does not prove full-dataset convergence.
- It does not prove generated quality unless rollout pairs are supplied and pass.
