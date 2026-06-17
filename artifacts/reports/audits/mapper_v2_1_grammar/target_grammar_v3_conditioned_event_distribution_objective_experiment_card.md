# Target Grammar v3 Conditioned Event-Distribution Objective Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: post-CE route synthesis selected `TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE` after CE weighting, global event budget, trace selectors, and anti-rigid decode branches failed or narrowed.
- Acceptance source, if any: `target_grammar_v3_post_ce_route_synthesis_summary.json`.
- Source snapshot / evidence grade: strong negative evidence against simple decode branches; medium positive evidence that loss-side/target-distribution repair is the next bounded family; no evidence yet that a conditioned objective improves rollout quality.

## Hypothesis

If v3's remaining CE failure is a context-dependent event-distribution calibration problem, then a conditioned event objective that separately penalizes starvation and overgeneration by target event-count context should improve finite rollout gates more safely than global event-budget loss or uniform event-token CE weighting. If it behaves like another scalar event knob, it should be killed before a full training run.

## Root Objective

Test the smallest loss-side repair that can address v3 event starvation without adding decode selectors, cross-window replay, future-token dependencies, or a target grammar redesign.

## Goal Decomposition

- Subgoal 1: Add a disabled-by-default conditioned event-distribution loss surface to the shared mapper tuple loss so v3 can use it through existing loss config plumbing.
- Subgoal 2: Prove the objective is genuinely conditioned, not only a renamed global event-budget loss.
- Subgoal 3: Verify the objective has overgeneration guards in sparse/zero-target and high-difficulty contexts.
- Subgoal 4: Run a short v3 training/rollout gate only after synthetic loss tests prove the conditioning surface is active and bounded.

## Candidate Variants

- Variant A: Increase uniform event-token CE weight again.
- Variant B: Retune global half-window event-budget loss.
- Variant C: Add conditioned event-distribution loss with asymmetric under/over penalties by target event-count bucket and difficulty bucket.
- Variant D: Redesign v3 target grammar immediately.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing CE-weight full32 report | Starvation falls without rigid/event overgeneration regression | Report route is `MUTATE`; rigid cases worsen |
| B | Existing event-budget 0.05 report | Median event ratio in gate and no max-token cases | Report route is `MUTATE`; 4 max-token cases and starvation regression |
| C | Synthetic logits plus short training/rollout gate | Conditioned penalty is finite, stronger in intended buckets, and short rollout improves starvation without overgeneration | Loss is equivalent to global budget, unstable, or rollout trips legality/event-ratio guards |
| D | Representation/design-only audit | Smaller objective repair impossible to test | Variant C can be tested without grammar rewrite |

## Selected Variant

- Selected: Variant C, conditioned event-distribution loss.
- Rejected: Variant A and B because their committed reports already failed gate-level evidence.
- Deferred: Variant D until the bounded loss-side repair is killed.
- Why this is the smallest useful test: it changes only the teacher-forced objective and config surface while preserving the audited v3 grammar, reversible reconstruction, online inference contract, and existing decode path.

## Selection Pressure

- Primary pressure: reduce second-window event starvation without increasing rigid-grid overgeneration or max-token dead ends.
- Guard pressure: no decode selector, no future-token inference dependency, no cross-window replay, no target-stream mutation.
- Runtime pressure: synthetic tests and a short fixed-slice training gate before any full32 or full-dataset rerun.
- Kill pressure: if conditioning does not produce a separable loss signal or immediately reproduces scalar-budget failures, stop and move to target-grammar repair.

## Research Question

Can v3's remaining event-distribution problem be improved by a conditioned teacher-forced objective, or is the failure now target-grammar/representation-side rather than objective-side?

## Closest Analogies / Novelty Layer

- Closest analogies: class-conditional loss weighting, density-conditioned auxiliary losses, asymmetric count calibration, curriculum-like objective gating.
- Relevant taxonomy bucket: implementation family and local verification after ablation failures.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation on the training objective; the v3 representation itself is unchanged.

## Minimal Change

Add disabled-by-default loss plumbing that computes predicted event mass and target event counts by half-window, then applies asymmetric penalties conditioned by target event-count bucket and normalized difficulty bucket:

- nonzero/sparse target buckets emphasize underproduction penalty;
- zero-target and high-difficulty sparse buckets emphasize overproduction penalty;
- all terms are computed from teacher-forced logits and target tokens only;
- defaults preserve current training behavior exactly.

The first implementation may expose one compact config family, for example:

- `lambda_conditioned_event_distribution`
- `conditioned_event_under_weight`
- `conditioned_event_over_weight`
- `conditioned_event_zero_target_over_weight`
- `conditioned_event_high_difficulty_over_weight`
- `conditioned_event_high_difficulty_min`

The exact names may change if the surrounding code has an established naming pattern, but the semantics must stay bounded and auditable.

## Files Likely to Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `src/pulsefield_model/models/mapper/v3/loss.py` only if the alias needs an explicit export adjustment
- `src/pulsefield_model/training/mapper_v3.py`
- `tests/models/mapper/test_tuple_loss.py` or nearest shared mapper loss test file
- `tests/training/test_mapper_v3.py`
- `src/pulsefield_model/evals/mapper_v3_conditioned_event_distribution_gate.py`
- `tests/evals/test_mapper_v3_conditioned_event_distribution_gate.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_summary.json`

## Dataset Slice

Stage 1 uses synthetic logits/targets that cover four contexts:

- zero-target first half;
- sparse nonzero second half;
- dense target half;
- high-difficulty sparse/zero target half.

Stage 2, only if Stage 1 passes, uses the existing v3 fixed-slice short training/rollout setup with the same 32-case gate family used by the CE and anti-rigid reports, but may start with a smaller smoke subset if runtime requires it.

## Baseline / Comparator

Comparator artifacts:

- Global event budget 0.05: route `MUTATE`, 4 max-token cases, starvation worsened.
- Event-token CE weighting: route `MUTATE`, starvation improved but rigidity worsened.
- Tap-only anti-rigid: route `KILL`, one dead end and median event ratio above gate.
- Post-CE synthesis: route `TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE`.

Comparator behavior in unit tests:

- `lambda_conditioned_event_distribution=0.0` must preserve total loss and metrics except for explicitly reported zero-valued disabled metrics.
- Conditioned objective must not change logits, masks, tokenizer output, or decode legality.

## Primary Metric

Route decision from the conditioned objective gate:

- `TEST_NEXT` if the objective is conditioned in synthetic tests and the short rollout improves starvation without overgeneration/legality regressions.
- `KILL` if it is equivalent to scalar event budget, unstable, or trips overgeneration guards.
- `MUTATE_TO_TARGET_GRAMMAR_REPAIR` if the loss is well-formed but rollout evidence indicates the remaining issue is target-side.

## Secondary Metric

- Synthetic loss ratio between intended high-penalty and low-penalty contexts.
- `loss/conditioned_event_distribution` finite and disabled at zero lambda.
- Second-window starved case count versus CE baseline.
- Rigid-grid case count versus pre-CE and CE baselines.
- Median event-count ratio and max-token hit count.
- Legality and dead-end count.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest \
  tests/models/mapper/test_tuple_loss.py \
  tests/training/test_mapper_v3.py \
  tests/evals/test_mapper_v3_conditioned_event_distribution_gate.py -q

uv run python -m pulsefield_model.evals.mapper_v3_conditioned_event_distribution_gate \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_result_report.md
```

If the nearest shared loss test file has a different name, use that file and record the actual command in the result report.

## Guard Check

```bash
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_summary.json >/dev/null
git diff --check
```

The gate must also assert:

- no grammar/tokenizer defaults change;
- online inference path does not require future tokens;
- decode selectors remain unchanged;
- full disabled-default compatibility for existing configs.

## Qualitative Check

The result report must explain whether the positive signal came from conditioning, not merely from more event mass. It must list overgeneration and legality outcomes next to starvation outcomes.

## Positive Signal

- Synthetic tests show high-penalty contexts receive stronger loss than low-penalty contexts while zero lambda is behavior-preserving.
- Short rollout reduces second-window starvation versus the CE baseline.
- Median event-count ratio remains within the accepted gate range.
- Rigid-grid and dead-end counts do not regress versus the active baseline.

## Negative Signal

- Objective collapses to the same signal as global event-budget loss.
- Loss creates max-token cases, dead ends, or high event-ratio overgeneration.
- Starvation improves only by flooding events.
- Required context cannot be computed without future inference dependency.

## Kill Criteria

- Any default behavior change when the new lambda is zero.
- Missing finite-loss proof on synthetic logits.
- High-difficulty sparse/zero target overgeneration guard cannot be expressed with current batch fields.
- Short rollout triggers legality failure or max-token cases.
- The result cannot distinguish conditioned objective effect from uniform event-token CE weighting.

## Expected Failure Modes

- Half-window buckets are still too coarse and reproduce global event-budget failure.
- Difficulty conditioning is too weak or unavailable for the synthetic/test batch shape.
- Underproduction penalty improves starvation while overproduction guard is too weak.
- Target event counts are contaminated by padding or chart-end windows.
- Metrics average away the sparse second-window failure class.

## Confounders

- A short training gate cannot prove full-song quality.
- Synthetic logits prove plumbing, not learned behavior.
- Existing CE-weight checkpoint quality may dominate a short continuation run.
- Full v3 replacement still requires full 4k audit, planner/mapper training, and inference validation.

## Expected Runtime / Runtime Budget

Expected Stage 1 runtime: under one minute.

Expected Stage 2 runtime: use the existing short fixed-slice gate budget. Stop before full32 if synthetic conditioning fails or if the smoke subset trips legality/max-token guards.

## Result Interpretation Plan

- Positive result would suggest: promote to a full32 conditioned-objective training gate.
- Negative result would suggest: kill loss-side scalar/conditioned repair and create a target-grammar repair card.
- Ambiguous result would require: one diagnostic that separates starvation improvement from event flooding.
- Human owner decides: whether the remaining v3 issue is worth more objective-side work or should move to grammar repair.
- Next-loop action if positive: `TEST_CONDITIONED_EVENT_DISTRIBUTION_FULL32_GATE`.
- Next-loop action if negative: `MUTATE_TO_TARGET_GRAMMAR_REPAIR`.
- Next-loop action if ambiguous: `TEST_STARVATION_VS_FLOODING_DIAGNOSTIC`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary route:
- Synthetic conditioning checks:
- Short rollout starved cases:
- Short rollout rigid cases:
- Median event-count ratio:
- Max-token hit count:
- Legality/dead-end count:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: exact config names and thresholds should follow the shared mapper loss style during implementation, but the objective must remain conditioned and disabled by default.

## Next-Loop Action

- If positive: run full32 conditioned-objective gate.
- If negative: create target-grammar repair card.
- If ambiguous: add one starvation-versus-flooding diagnostic before any larger training run.

## Novelty Notes

- Closest analogies: class-conditional weighting, asymmetric count calibration, density-conditioned auxiliary losses.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is objective-side engineering variation on an already audited v3 representation.
