# Target Grammar v3 Density-Loss Effectiveness Diagnostic Experiment Card

## Hypothesis

The current v3 500-step training run already uses density auxiliary loss, but the failure-cluster diagnostic suggests event distribution is still the dominant blocker. If eval density loss stays flat while token loss improves, the next mutation should not be a vague "add density loss" card; it should test a stronger event-distribution/count-continuity objective or a density-weight sweep with rollout gates.

## Root Objective

Decide whether existing v3 evidence supports density-loss retuning as the next bounded mutation, or whether the next card should target event-count/window-continuity more directly.

## Goal Decomposition

1. Confirm whether the 500-step checkpoint used density auxiliary loss.
2. Compare eval token-loss trend against eval density-loss trend over the same run.
3. Compare density-loss trend with the observed 32-case failure clusters.
4. Choose the next event-distribution mutation family before running a costly sweep.

## Candidate Variants

### A. Existing-Artifact Density-Loss Effectiveness Diagnostic

Analyze the existing 500-step training report, the six-case horizon gate summary, and the 32-case failure-cluster diagnostic. Add no training, rollout, or source-code changes.

### B. Immediate High-Lambda Density Sweep

Train one or more new checkpoints with larger `lambda_density` values and rerun fixed-slice rollouts.

### C. New Event-Count Objective

Implement a new auxiliary count/window-continuity loss before checking whether the existing density loss is active and effective.

### D. Decode-Time Density Control

Add decode-time event-count pressure or time-shift/event balancing.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | Read existing training history and failure clusters | Density loss is active and its trend explains whether retuning is justified | Training history lacks density metrics or cannot be reconciled with failures |
| B | Train high-lambda checkpoint | Free-running starvation improves without overproduction | Expensive before knowing whether current density loss moved eval density |
| C | Add new loss | New objective directly targets failure clusters | Adds machinery before checking current loss evidence |
| D | Decode control | Fast inference-side mutation | Prior decode-only sweep already killed broad decode-only calibration |

## Selected Variant

Variant A: existing-artifact density-loss effectiveness diagnostic.

## Selection Pressure

Variant A is the cheapest way to avoid repeating the earlier decode-only mistake. The current failure clusters point at event distribution, but the existing training stack already has a density auxiliary loss. The next experiment should first verify whether that loss was active and whether it actually moved eval density before spending runtime on a sweep or adding new loss code.

## Minimal Change

Add report artifacts only:

- `target_grammar_v3_density_loss_effectiveness_diagnostic_experiment_card.md`
- `target_grammar_v3_density_loss_effectiveness_diagnostic_result_report.md`
- `target_grammar_v3_density_loss_effectiveness_diagnostic_summary.json`

No source-code changes.

## Files Likely To Change

Report artifacts under:

- `artifacts/reports/audits/mapper_v2_1_grammar/`

Read-only context:

- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_training_horizon_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_failure_cluster_diagnostic_summary.json`

## Dataset Slice

Existing fixed 32-song/256-window v3 training run and its downstream six-case and 32-case rollout audits.

## Baseline / Comparator

The existing 500-step v3 training run:

- `lambda_density: 0.05`
- final eval token loss: `1.940945`
- final eval density loss: `1.608767`
- wide-audit result: `MUTATE`
- failure clusters: `11` second-window starvation cases, `10` starved-160-grid cases, `9` undergeneration cases, `2` overgeneration cases.

## Primary Metric

Eval density-loss relative change from first recorded eval step to final eval step.

## Secondary Metrics

- Eval token-loss relative change.
- Train density-loss relative change.
- Final density-loss weight contribution (`lambda_density * loss/density`) relative to token loss.
- Failure-cluster counts after the same run.
- Whether six-case horizon gate passed while 32-case wide audit failed.

## Verify Command Or Evaluation Procedure

1. Load the 500-step training report JSON.
2. Extract `loss_config`, `history`, `final_eval_metrics`, and `final_train_metrics`.
3. Compute token-loss and density-loss deltas across eval history.
4. Load failure-cluster diagnostic summary and compare dominant failure counts.
5. Write a Markdown result report and JSON summary.
6. Validate JSON with `python3 -m json.tool`.

## Guard Check

- Do not rerun training or rollout.
- Do not change source code.
- Verify density loss was active (`lambda_density > 0`).
- Verify the loaded training history has eval density metrics.

## Qualitative Check

Confirm the interpretation does not overclaim: flat eval density loss does not prove density loss is useless; it proves the existing weak density setup did not resolve free-running event distribution on the 32-case fixed slice.

## Positive Signal

If eval density loss falls by at least 10% and the failure clusters are not density/count dominated, density retuning is not the obvious next card.

## Negative Signal

If eval density loss is flat or worse while failure clusters remain event-distribution dominated, the next card should be a stronger density-weight/count-continuity calibration with rollout gates.

## Kill Criteria

Kill a broad "add density loss" plan if the existing run already used density loss and eval density did not improve meaningfully.

## Expected Failure Modes

- Training history may be too short to estimate stable trends.
- Eval density loss may be noisy because the fixed eval split is small.
- Density loss is teacher-forced and may not directly predict free-running second-window continuity.
- The density target may not distinguish overproduction from underproduction at the right temporal granularity.

## Expected Runtime / Runtime Budget

Under 2 minutes. Stop if the required JSON reports are missing.

## Confounders

- This is not a new training result.
- The 32-case wide audit and six-case horizon gate measure different sample sets.
- Density-loss trend is indirect evidence for free-running rollout behavior.
- The effect of a higher `lambda_density` remains untested until a sweep runs.

## Result Interpretation Plan

- If eval density loss improved clearly, next card should inspect why free-running clusters still fail despite density learning.
- If eval density loss is flat, next card should test stronger event-distribution calibration with explicit rollout gates.
- If density loss was inactive, next card should simply enable it and rerun the 500-step gate.

## Result Log Template

- Loaded reports:
- Density loss active:
- Eval token-loss trend:
- Eval density-loss trend:
- Failure-cluster comparison:
- Decision:
- Next-loop action:

## Next-Loop Action

Create or run one bounded event-distribution calibration card only after this diagnostic confirms the target family.

## Closest Analogies And Novelty Layer

Closest analogies: auxiliary-loss effectiveness audit, teacher-forced metric versus free-running behavior audit, event-count calibration diagnostics.

Novelty layer: none claimed. This is training evidence triage for v3 replacement readiness.
