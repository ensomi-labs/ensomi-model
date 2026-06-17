# Target Grammar v3 Time-Shift Distance Loss Stage 1 Result Report

## Scope

This gate tests synthetic loss plumbing only. It does not train, roll out, change tokenizer behavior, change decode policy, or change inference.

## Decision

Route: `TEST_TIME_SHIFT_DISTANCE_TINY_TRAINING_GATE`.

- Reason: synthetic separation, gradient, disabled-default behavior, and config exposure passed
- Recommended next step: Create a tiny v3 training/rollout card with rigid-grid and second-window guards.
- Next card: `target_grammar_v3_time_shift_distance_tiny_training_gate`

## Checks

| Check | Value |
| --- | ---: |
| `matching_loss_finite` | `True` |
| `rigid_loss_finite` | `True` |
| `rigid_loss_greater_than_matching` | `True` |
| `gradient_finite` | `True` |
| `gradient_nonzero` | `True` |
| `non_time_shift_rows_ignored` | `True` |
| `disabled_default_ok` | `True` |
| `enabled_metric_positive` | `True` |
| `config_fields_present` | `True` |

## Probe Metrics

| Probe | Metric | Value |
| --- | --- | ---: |
| `matching_vs_rigid` | `matching_loss` | `1.7908250526943448e-07` |
| `matching_vs_rigid` | `rigid_wrong_shift_loss` | `0.00034506627707742155` |
| `matching_vs_rigid` | `gap` | `0.0003448871945721521` |
| `matching_vs_rigid` | `matching_finite` | `True` |
| `matching_vs_rigid` | `rigid_finite` | `True` |
| `gradient` | `loss` | `0.00034506627707742155` |
| `gradient` | `grad_abs_sum` | `2.4627895982121117e-05` |
| `gradient` | `finite` | `True` |
| `gradient` | `nonzero` | `True` |
| `non_time_shift_ignored` | `loss` | `0.0` |
| `non_time_shift_ignored` | `ignored` | `True` |
| `disabled_default` | `off_lambda` | `0.0` |
| `disabled_default` | `on_lambda` | `0.5` |
| `disabled_default` | `off_time_shift_distance_loss` | `0.0` |
| `disabled_default` | `on_time_shift_distance_loss` | `0.00034506627707742155` |
| `disabled_default` | `disabled_default_ok` | `True` |
| `disabled_default` | `enabled_metric_positive` | `True` |
| `disabled_default` | `total_loss_delta` | `0.00017261505126953125` |
| `config` | `lambda_time_shift_distance` | `0.25` |
| `config` | `time_shift_distance_scale_ms` | `500.0` |
| `config` | `config_fields_present` | `True` |

## What This Proves

- Matching target time-shift logits score lower than rigid wrong-shift logits.
- The auxiliary loss is finite and differentiable in the synthetic probe.
- Non-time-shift target rows are ignored by the auxiliary term.
- The new config is disabled by default and exposed through `MapperV3LossConfig`.

## What Remains Unproven

- No trained rollout quality improvement is proven.
- No tiny, full32, or full-dataset v3 gate has been run for this objective.
- This does not prove the remaining v3 failure is solved by loss-side timing calibration.

## Next Step

Create a tiny v3 training/rollout card with rigid-grid and second-window guards.
