# Target Grammar v3 Conditioned Event-Distribution Objective Stage 1 Result Report

## Scope

This gate tests synthetic loss plumbing only. It does not train, roll out, change tokenizer behavior, or change decode policy.

## Decision

Route: `TEST_SHORT_ROLLOUT_GATE`.

- Reason: synthetic conditioning, disabled-default behavior, and config exposure passed
- Recommended next step: Run the short v3 conditioned-objective training/rollout gate with overgeneration guards.
- Next card: `target_grammar_v3_conditioned_event_distribution_short_rollout_gate`

## Checks

| Check | Value |
| --- | ---: |
| `high_difficulty_overproduction_finite` | `True` |
| `high_difficulty_overproduction_ratio` | `4.0` |
| `high_difficulty_overproduction_conditioned` | `True` |
| `underproduction_finite` | `True` |
| `underproduction_gap` | `1.4321379861248715` |
| `underproduction_penalized` | `True` |
| `disabled_default_ok` | `True` |
| `enabled_metric_positive` | `True` |
| `config_fields_present` | `True` |

## Probe Metrics

| Probe | Metric | Value |
| --- | --- | ---: |
| `high_difficulty_overproduction` | `low_difficulty_loss` | `0.2763669788837433` |
| `high_difficulty_overproduction` | `high_difficulty_loss` | `1.1054679155349731` |
| `high_difficulty_overproduction` | `ratio` | `4.0` |
| `high_difficulty_overproduction` | `finite` | `True` |
| `underproduction` | `matching_loss` | `5.314737427397631e-05` |
| `underproduction` | `suppressed_loss` | `1.4321911334991455` |
| `underproduction` | `gap` | `1.4321379861248715` |
| `underproduction` | `finite` | `True` |
| `disabled_default` | `off_lambda` | `0.0` |
| `disabled_default` | `off_conditioned_loss` | `0.0` |
| `disabled_default` | `on_lambda` | `0.5` |
| `disabled_default` | `on_conditioned_loss` | `0.0019929828122258186` |
| `disabled_default` | `disabled_default_ok` | `True` |
| `disabled_default` | `enabled_metric_positive` | `True` |
| `config` | `config_fields_present` | `True` |
| `config` | `missing_fields` | `[]` |

## What This Proves

- The objective has a finite synthetic loss.
- High-difficulty sparse/zero-target overproduction receives a stronger penalty than the same low-difficulty context.
- Underproduction of target events is penalized.
- The new config is disabled by default and exposed through `MapperV3LossConfig`.

## What Remains Unproven

- No trained rollout quality improvement is proven.
- No full32 or full-dataset v3 gate has been run for this objective.
- This does not prove the remaining v3 failure is objective-side rather than target-grammar-side.

## Next Step

Run the short v3 conditioned-objective training/rollout gate with overgeneration guards.
