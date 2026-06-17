# Target Grammar v3 CE Event-Margin Viability Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: The CE residual cluster diagnostic selected `TEST_EVENT_RANKING_CALIBRATION`, while the earlier event-margin diagnostic killed global event bonus on the older 500-step baseline.
- Acceptance source, if any: active v3 full-pipeline goal plus `target_grammar_v3_ce_residual_cluster_diagnostic_result_report.md` and `target_grammar_v3_event_margin_decode_viability_result_report.md`.
- Source snapshot / evidence grade: strong evidence that CE improves starvation and F1 while increasing rigidity; strong evidence that a global event bonus is unsafe on the old baseline; medium evidence that CE has a different margin profile worth auditing before any new decode or training mutation.

## Hypothesis

If CE weighting already moves event tokens closer to top-1, then a CE-specific event-margin audit may reveal a small safe operating region for event-ranking calibration. If CE still requires large event biases or overlaps with overgenerated cases, then the next step should be selective trace instrumentation or v2.1 grammar repair, not a global decode bonus.

## Root Objective

Decide whether the CE-weight checkpoint supports a bounded event-ranking calibration experiment, or whether the v3 branch should pivot away from decode/ranking calibration.

## Goal Decomposition

- Subgoal 1: Measure required event-token bias on the CE-weight full32 rollout.
- Subgoal 2: Compare CE required-bias distribution against the old 500-step baseline.
- Subgoal 3: Decide whether the next experiment should be simple global event bonus, selective trace/oracle instrumentation, or v2.1 grammar repair.

## Candidate Variants

- Variant A: Run CE event-margin viability using the existing artifact-only evaluator. Selected.
- Variant B: Run a CE global event-bonus rollout immediately. Rejected until margins show a safe region.
- Variant C: Add a new target-side ranking loss. Deferred until decode-side margin feasibility is checked.
- Variant D: Pivot to v2.1 grammar repair immediately. Deferred until CE-specific margins are audited.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Artifact-only margin audit | CE failed cases mostly need bias `<=4` and pass-like/overgenerated cases do not overlap | CE failed cases need large bias or overgenerated cases overlap |
| B | Full32 bonus rollout | Starvation drops without overgeneration/rigidity regression | Max-token, event-ratio, or rigid-grid regression |
| C | New ranking loss | Teacher-forced ranking improves and rollout gates pass | More scalar training without a viable decode signal |
| D | v2.1 repair | v3 CE has no smaller path | CE margin evidence not yet checked |

## Selected Variant

- Selected: Variant A, CE event-margin viability.
- Rejected: Variant B because prior decode-control mutations overcorrected.
- Deferred: C and D until the artifact-only margin gate selects a route.
- Why this is the smallest useful test: it reuses existing CE rollout logit diagnostics and can kill a global bonus without new runtime behavior.

## Selection Pressure

- Primary pressure: decide whether CE supports simple event-ranking calibration.
- Guard pressure: no training, tokenizer, grammar, rollout, or default decode changes.
- Runtime pressure: under 10 seconds.
- Kill pressure: if CE failed/overgenerated cases overlap at low required bias, global event-bonus calibration is unsafe.

## Research Question

Does the CE-weight v3 checkpoint create a safe event-margin region for ranking calibration, or does it merely trade starvation for rigid/overgenerated event grids?

## Closest Analogies / Novelty Layer

- Closest analogies: logit-margin calibration, decode-policy triage, exposure-bias diagnostics.
- Relevant taxonomy bucket: local verification before bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for v3 replacement readiness.

## Minimal Change

Run `mapper_v3_event_margin_decode_viability` with two existing summaries:

- `baseline_500step`
- `ce_weight2`, selected as the baseline for the decision

Write a new CE-specific summary and report.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/mapper_v3_event_margin_decode_viability.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json`

## Dataset Slice

The same fixed 32-case, 16s real-audio v3 rollout universe from the 500-step baseline and CE-weight training gate.

## Baseline / Comparator

- Decision baseline: `ce_weight2`.
- Comparator: `baseline_500step`.

## Primary Metric

Required event-bias distribution for CE failed cases, especially starved/rigid/overgenerated cases above bias thresholds `2`, `4`, and `6`.

## Secondary Metric

- Required-bias distribution for pass-like cases.
- Event top-k minus top-1 gap.
- Median best-event rank and margin.
- Cross-variant sanity check versus old baseline.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_event_margin_decode_viability \
  --summary baseline_500step=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json \
  --summary ce_weight2=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json \
  --baseline ce_weight2 \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_event_margin_decode_viability.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must state whether CE supports simple event-bonus calibration, selective trace/oracle work, or v2.1 grammar pivot.

## Positive Signal

Most CE starved/failed cases need required event bias `<=4`, and pass-like/overgenerated cases do not need comparable bias.

## Negative Signal

CE overgenerated or rigid cases overlap with the low-bias region, or CE still needs large bias on hard failed cases.

## Kill Criteria

- Required margin fields are missing.
- CE summary cannot be parsed as a run list.
- The result cannot separate starved/rigid/overgenerated cases.
- Global event bonus would likely worsen overgenerated or rigid cases.

## Expected Failure Modes

- CE may reduce margin magnitude but also reduce event-valid ratio.
- Low required bias may occur in already-overgenerated cases.
- Median margins may hide per-step selectivity that requires trace-level instrumentation.

## Confounders

- This is an artifact-only diagnostic, not a rollout.
- CE weighting changed the trained model; margin comparisons are not a controlled decode ablation.
- A selective event gate may still work even if global bonus is killed.

## Expected Runtime / Runtime Budget

Expected runtime: under 10 seconds. Stop if required fields are missing.

## Result Interpretation Plan

- Positive result would suggest: run a bounded CE simple event-bonus or rank-selector rollout.
- Negative result would suggest: do not run global event bonus; use selective per-step trace/oracle instrumentation or pivot to v2.1 grammar repair.
- Ambiguous result would require: full per-step trace on CE residual cases.
- Human owner decides: whether one more selective v3 trace card is worth it.
- Next-loop action if positive: CE event-ranking rollout card.
- Next-loop action if negative: selective trace/oracle or v2.1 grammar card.
- Next-loop action if ambiguous: CE residual per-step trace audit.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Input summaries:
- Runtime:
- Required-bias table:
- Decision:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: this cannot prove selective per-step calibration; it only evaluates global/ranking viability from existing margins.

## Next-Loop Action

- If positive: CE event-ranking rollout card.
- If negative: selective trace/oracle or v2.1 grammar repair card.
- If ambiguous: CE residual trace audit.

## Novelty Notes

- Closest analogies: logit-margin calibration and constrained-decoding triage.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic.
