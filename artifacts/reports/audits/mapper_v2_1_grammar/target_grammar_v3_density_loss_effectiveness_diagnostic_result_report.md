# Target Grammar v3 Density-Loss Effectiveness Diagnostic Result Report

## Scope

This pass executes `target_grammar_v3_density_loss_effectiveness_diagnostic_experiment_card.md`. It reads the existing v3 500-step training report and the committed v3 failure-cluster diagnostic. It does not rerun training, rollout, or source-code changes.

## Guard Results

- Training report loaded: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json`
- Failure-cluster summary loaded: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_failure_cluster_diagnostic_summary.json`
- Density loss active: `True` (`lambda_density=0.05`)
- Eval history rows: `5`
- Training rerun: `false`
- Rollout rerun: `false`
- Source code changed: `false`

## Result

Decision: `TEST_NEXT`.

Reason: Density loss was active but eval density did not improve while token loss improved; next mutation needs stronger event-distribution calibration with rollout gates.

## Loss Trend

| Step | Eval token | Eval density | Weighted density | Density/token |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 2.183018 | 1.599248 | 0.079962 | 0.0366 |
| 200 | 2.031527 | 1.587372 | 0.079369 | 0.0391 |
| 300 | 2.096646 | 1.610221 | 0.080511 | 0.0384 |
| 400 | 1.905926 | 1.609627 | 0.080481 | 0.0422 |
| 500 | 1.940945 | 1.608767 | 0.080438 | 0.0414 |

Trend summary:

- Eval token loss: `2.183018` -> `1.940945` (-11.09%).
- Eval density loss: `1.599248` -> `1.608767` (0.60%).
- Train token loss: `2.240048` -> `1.683360` (-24.85%).
- Train density loss: `1.749788` -> `1.464490` (-16.30%).
- Final weighted eval density term: `0.080438`, about `4.14%` of final eval token loss.

## Failure-Cluster Context

| Failure signal | Cases |
| --- | ---: |
| `second_window_starvation` | 11 |
| `reference_mismatch_second_window_drop` | 11 |
| `starved_160_grid` | 10 |
| `undergeneration` | 9 |
| `overgeneration` | 2 |
| `easier_chart_overproduction` | 2 |

The six-case horizon gate passed, but the 32-case wide audit still failed. The failure-cluster diagnostic shows the remaining blockers are dominated by event distribution over time: second-window starvation, starved 160ms grids, undergeneration on hard cases, and overgeneration on easier/mid cases.

## What Passed

- The current v3 training setup already has density auxiliary loss enabled.
- Token loss improved on eval over the 500-step run.
- The diagnostic requires no new runtime or code changes.

## What Surfaced

- Eval density loss stayed effectively flat: the relative change was only `0.60%` from step 100 to 500.
- Density contributes a small weighted term at `lambda_density=0.05`, about `4.14%` of final eval token loss.
- The wide-audit failures remain count/distribution-shaped despite active density loss.
- Therefore the next card should not be described as merely adding density loss; it should explicitly sweep/strengthen event-distribution pressure and judge it by free-running rollout gates.

## Interpretation

This is a `TEST_NEXT` routing result. Existing evidence kills a broad "add density loss" framing because density loss was already active. It does not kill density retuning. It narrows the next experiment to a stronger event-distribution calibration with explicit rollout gates for starvation, overgeneration, boundary duplicates, and rigidity.

## Recommended Next Card

`target_grammar_v3_event_distribution_calibration`:

- selected variant should compare the current `lambda_density=0.05` checkpoint against one stronger density/count-continuity calibration, not broad architecture changes;
- primary gate: reduce 32-case second-window starvation below `6` cases and starved-160-grid below `5` cases;
- secondary gate: do not increase overgeneration above `2` cases or max boundary-event ratio above the existing wide-audit value;
- guard: all rollouts legal, v3 grammar/model tests unchanged, no default replacement.
