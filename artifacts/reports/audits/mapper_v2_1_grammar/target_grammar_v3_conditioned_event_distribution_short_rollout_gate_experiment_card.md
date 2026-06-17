# Target Grammar v3 Conditioned Event-Distribution Short Rollout Gate Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: Stage 1 conditioned event-distribution plumbing passed synthetic conditioning and disabled-default checks.
- Acceptance source, if any: `target_grammar_v3_conditioned_event_distribution_objective_summary.json`.
- Source snapshot / evidence grade: strong synthetic plumbing evidence; no trained quality evidence yet.

## Hypothesis

If the conditioned event-distribution objective is useful beyond plumbing, then a short v3 training/rollout gate should reduce CE-baseline second-window starvation without increasing rigid-grid overgeneration, max-token cases, dead ends, or median event ratio beyond the accepted gate. If the improvement comes only from flooding events, the objective should be killed before a full32 run.

## Root Objective

Test whether the newly added conditioned event-distribution objective gives a real short-run quality signal before spending full32 or full-dataset runtime.

## Goal Decomposition

- Subgoal 1: Train or evaluate only a short bounded v3 run with the conditioned objective enabled.
- Subgoal 2: Compare against the committed CE-weight full32 baseline and pre-CE fixed-slice baseline.
- Subgoal 3: Separate starvation improvement from event flooding.
- Subgoal 4: Route to full32 gate, target-grammar repair, or kill.

## Candidate Variants

- Variant A: Run full32 conditioned-objective training immediately.
- Variant B: Run a short conditioned-objective training/rollout gate on CE-sensitive fixed-slice cases.
- Variant C: Run only synthetic/objective diagnostics again with different weights.
- Variant D: Skip objective testing and move to target-grammar repair.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Full32 runtime estimate and current evidence | Stage 1 already proves trained benefit | No trained signal yet; full32 is premature |
| B | Short rollout gate with CE-sensitive cases | Starvation improves without overgeneration/legality regressions | Max-token/dead-end/rigid/event-ratio guards fail |
| C | Stage 1 synthetic gate | Conditioning mechanics are unknown | Stage 1 already passed |
| D | Post-CE route synthesis | Objective-side repair already killed | Conditioned objective has not been trained |

## Selected Variant

- Selected: Variant B, short conditioned-objective training/rollout gate.
- Rejected: Variant A because full32 is too expensive before a short trained signal.
- Rejected: Variant C because the synthetic gate already passed.
- Deferred: Variant D until the short trained gate fails.
- Why this is the smallest useful test: it is the first trained-quality check for the objective while keeping runtime bounded and preserving all tokenizer/decode defaults.

## Selection Pressure

- Primary pressure: reduce second-window starvation relative to CE baseline.
- Guard pressure: no legality regression, no max-token cases, no dead ends, no rigid-grid regression, no event flooding.
- Runtime pressure: stop after a short fixed-slice gate before full32.
- Kill pressure: if starvation improves only by raising event ratio or rigidity, kill or mutate to target-grammar repair.

## Research Question

Does the conditioned event-distribution objective produce a trained rollout improvement signal, or does v3 now require target-grammar repair?

## Closest Analogies / Novelty Layer

- Closest analogies: short ablation gate before full sweep, loss-weight smoke training, event-count calibration with safety guards.
- Relevant taxonomy bucket: minimal experiment and result interpretation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: training-objective engineering variation; v3 representation unchanged.

## Minimal Change

Add a short rollout gate evaluator/config that enables the already-implemented conditioned event objective with a small weight and compares the generated rollout metrics against committed baselines. The run must not alter tokenizer, grammar, decode policy, or default configs.

Suggested first training-loss knobs:

- `lambda_conditioned_event_distribution: 0.05`
- `conditioned_event_under_weight: 3.0`
- `conditioned_event_over_weight: 1.0`
- `conditioned_event_zero_target_over_weight: 2.0`
- `conditioned_event_high_difficulty_over_weight: 4.0`
- `conditioned_event_high_difficulty_min: 0.75`

These are candidate values, not defaults. The result report must record the actual values used.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_conditioned_event_distribution_short_rollout.py`
- `tests/evals/test_mapper_v3_conditioned_event_distribution_short_rollout.py`
- optional short-run config under `configs/training/` only if the evaluator cannot pass overrides directly
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_objective_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json`

## Dataset Slice

Use a short fixed-slice subset biased toward the known CE failure modes:

- all CE-starved low-bias cases if available;
- at least one pass-like control case;
- at least one high-difficulty sparse/zero-target risk case;
- stop before full32 unless the short subset passes.

If the committed tooling only supports the 32-case universe, add a case-selector option rather than changing the case definitions.

## Baseline / Comparator

Primary comparator: committed CE-weight full32 summary:

- `second_window_starved_case_count`: 5
- `rigid_case_count`: 11
- `median_event_count_ratio`: 1.1561079925153703
- `max_token_count`: 0
- `dead_end_count`: 0
- `all_legal`: true

Safety comparator: pre-CE v3 fixed-slice baseline:

- `second_window_starved_case_count`: 11
- `rigid_case_count`: 7
- `median_event_count_ratio`: 1.0
- `max_boundary_event_ratio`: 0.2

## Primary Metric

Route decision:

- `TEST_FULL32_CONDITIONED_EVENT_DISTRIBUTION` if the short gate reduces starvation versus CE without overgeneration/legality regressions.
- `KILL` if it creates max-token/dead-end/legality failures or event flooding.
- `MUTATE_TO_TARGET_GRAMMAR_REPAIR` if the objective is safe but does not improve starvation.

## Secondary Metric

- Second-window starved case count.
- Rigid-grid case count.
- Median and max event-count ratio.
- Mean F1 at 100 ms.
- Boundary event ratio.
- Dead-end count and max-token hit count.
- Per-case notes for any improvement/regression.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_conditioned_event_distribution_short_rollout.py -q

uv run python -m pulsefield_model.evals.mapper_v3_conditioned_event_distribution_short_rollout \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_result_report.md
```

If implementation reuses an existing training-comparison evaluator, record the exact command and overrides in the result report.

## Guard Check

```bash
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json >/dev/null
git diff --check
```

The evaluator must also assert:

- Stage 1 summary route is `TEST_SHORT_ROLLOUT_GATE`.
- No decode selector is enabled.
- No tokenizer/grammar defaults changed.
- No online inference future-token dependency is introduced.
- Loss overrides are explicit in the report.

## Qualitative Check

The report must include a per-case starvation-vs-flooding table. Cases that improve starvation but exceed event ratio or rigidity guards must be marked as negative, not ambiguous positives.

## Positive Signal

- Starved case count falls below CE baseline `5`.
- Rigid case count is no worse than CE baseline `11` and preferably no worse than pre-CE `7`.
- Median event-count ratio stays within `0.80..1.25`.
- No max-token cases, no dead ends, and all rollouts legal.

## Negative Signal

- Starvation remains at or above CE baseline.
- Event ratio rises above gate or max-token/dead-end failures appear.
- Rigid cases increase.
- Improvement is localized to one case while controls regress.

## Kill Criteria

- Any illegal rollout, max-token hit, or dead end.
- Median event-count ratio outside `0.80..1.25`.
- Rigid case count worse than CE baseline `11`.
- Starved case count not below CE baseline `5`.
- Result cannot separate objective effect from event flooding.

## Expected Failure Modes

- Conditioned objective behaves like CE weighting and overproduces rigid grids.
- High-difficulty overproduction guard suppresses too much and preserves starvation.
- Short training is too noisy to separate objective effect.
- Case selector accidentally omits the failure class.

## Confounders

- Short rollout is not full replacement evidence.
- Full32 and full-dataset audits may still fail after a short pass.
- Existing checkpoint initialization and seed may dominate short-run outcomes.
- Dataset slice selection can overstate targeted improvement.

## Expected Runtime / Runtime Budget

Expected runtime: bounded short-run gate. Stop immediately if setup cannot select CE-sensitive cases or if early rollouts trip legality/max-token/dead-end guards.

Do not run full32 in this card.

## Result Interpretation Plan

- Positive result would suggest: create and run a full32 conditioned-objective gate.
- Negative result would suggest: kill conditioned objective and create target-grammar repair card.
- Ambiguous result would require: one starvation-vs-flooding diagnostic or a repeated seed, not a larger full32 run by default.
- Human owner decides: whether a short positive signal is enough to spend full32 runtime.
- Next-loop action if positive: `TEST_FULL32_CONDITIONED_EVENT_DISTRIBUTION`.
- Next-loop action if negative: `MUTATE_TO_TARGET_GRAMMAR_REPAIR`.
- Next-loop action if ambiguous: `TEST_STARVATION_VS_FLOODING_DIAGNOSTIC`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Loss overrides:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary route:
- Starved cases:
- Rigid cases:
- Median event-count ratio:
- Max event-count ratio:
- Dead-end count:
- Max-token hit count:
- Legal count:
- Mean F1 100 ms:
- Per-case starvation-vs-flooding table:
- Verify command / result:
- Guard command / result:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
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
- Remaining ambiguity: exact case selector and checkpoint initialization should reuse the nearest existing v3 fixed-slice tooling.

## Next-Loop Action

- If positive: run full32 conditioned-objective gate.
- If negative: target-grammar repair card.
- If ambiguous: starvation-vs-flooding diagnostic.

## Novelty Notes

- Closest analogies: short ablation gate, guarded loss-weight training probe.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: objective-side engineering variation only.
