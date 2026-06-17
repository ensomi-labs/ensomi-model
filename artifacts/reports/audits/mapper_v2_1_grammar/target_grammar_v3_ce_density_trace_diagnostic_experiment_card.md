# Target Grammar v3 CE Density-Aware Trace Diagnostic Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the CE selective trace oracle found only `1/5` low-bias CE-starved cases with enough second-window opportunities, but recommended either density/control-aware trace fields or a pivot to target-grammar repair.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_ce_selective_trace_oracle_result_report.md`.
- Source snapshot / evidence grade: strong runtime legality evidence; medium negative evidence against a simple selector; missing evidence on whether the remaining opportunity signal is separable by generated-density context.

## Hypothesis

If CE-starved second-window opportunities occur mainly in locally sparse generated-density regions and pass-like controls do not show the same pattern, then a density-aware selector or training objective may still be worth one bounded mutation. If only one or two CE-starved cases show this pattern, or controls look similar, then the v3 local selector route has too little coverage and should pivot toward target-grammar/training repair.

## Root Objective

Decide whether the CE trace-oracle residual signal is density-separable enough to justify another v3 selector/training mutation, or whether the evidence now supports leaving local decode probes and improving the target grammar/training objective.

## Goal Decomposition

- Subgoal 1: Reuse the completed CE trace-oracle summaries without rerunning model inference.
- Subgoal 2: For each selected case, compute second-window opportunity counts in locally sparse generated-density context.
- Subgoal 3: Compare low-bias starved cases against the pass-like control.
- Subgoal 4: Route the next loop to density-aware selector, training objective repair, or target-grammar repair.

## Candidate Variants

- Variant A: Implement a density-aware selector immediately.
- Variant B: Add an artifact-only density-aware trace diagnostic over the existing CE trace summaries.
- Variant C: Run a longer v3 training job immediately.
- Variant D: Pivot directly to target-grammar repair without checking density separability.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Selector smoke | Improved starved cases without overproduction | Needs density-separation evidence first |
| B | Existing-trace density diagnostic | At least `3/5` CE-starved cases have sparse-context second-window opportunities | Only `0-2/5` cases have coverage or control is similar |
| C | Longer training | Better event generation after more steps | Does not identify whether selector branch is worth keeping |
| D | Synthesis-only pivot | Prior CE trace already sufficient | Could skip a cheap discriminating diagnostic |

## Selected Variant

- Selected: Variant B, artifact-only CE density-aware trace diagnostic.
- Rejected: Variant A and C because they spend mutation/training budget before checking whether the residual signal is separable.
- Deferred: Variant D until the density diagnostic resolves the CE trace ambiguity.
- Why this is the smallest useful test: it only reads existing trace artifacts and writes a report; no retraining, tokenizer edit, grammar edit, or decode-policy change.

## Selection Pressure

- Primary pressure: broad coverage across CE-starved cases, not just one success case.
- Guard pressure: unchanged model, tokenizer, target grammar, training, and runtime decode behavior.
- Runtime pressure: artifact-only, expected under one minute.
- Kill pressure: if fewer than three CE-starved cases have sparse-context second-window opportunity coverage, stop simple/selector-local v3 work.

## Research Question

Are near-boundary CE event opportunities concentrated in sparse generated-density regions in enough starved cases to support a density-aware selector/training mutation?

## Closest Analogies / Novelty Layer

- Closest analogies: oracle decoding diagnostics, classifier feature separability audit, failure-cluster ablation.
- Relevant taxonomy bucket: local verification before bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is diagnostic engineering evidence for a previously audited representation.

## Minimal Change

Add an artifact-only evaluator that:

- reads `target_grammar_v3_ce_selective_trace_oracle_summary.json`;
- loads each selected trace summary path;
- classifies second-window logit examples by opportunity status;
- measures generated-event density in the previous `1000ms`;
- counts sparse-context opportunities per case;
- reports whether the density-aware signal covers enough CE-starved cases to justify another local v3 mutation.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_ce_density_trace_diagnostic.py`
- `tests/evals/test_mapper_v3_ce_density_trace_diagnostic.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_summary.json`
- Trace summaries under `artifacts/tmp/mapper_v3_ce_selective_trace_oracle/rollouts/`

## Dataset Slice

The six cases from the CE selective trace oracle:

- five low-bias CE-starved cases;
- one pass-like control from the same CE rollout slice.

## Baseline / Comparator

Baseline is the CE selective trace-oracle result: `1/5` low-bias starved cases met the simple second-window opportunity threshold, with full traces and no legality failures.

## Primary Metric

Number of low-bias CE-starved cases with at least `10` second-window opportunity steps where the generated-event count in the previous `1000ms` is at most `1`.

## Secondary Metric

- Sparse-context opportunity count in the pass-like control.
- Second-window generated event count.
- Second-window event-argmax share.
- Longest consecutive opportunity run.
- Median previous-`1000ms` generated-event density for opportunity steps.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_ce_density_trace_diagnostic \
  --trace-oracle-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_summary.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_ce_density_trace_diagnostic.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_density_trace_diagnostic_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must distinguish "opportunities exist" from "opportunities exist in sparse generated-density context across enough starved cases."

## Positive Signal

At least `3/5` CE-starved cases have `>=10` sparse-context second-window opportunities, and the pass-like control has fewer than `10`.

## Negative Signal

Fewer than `3/5` CE-starved cases have sparse-context opportunity coverage, or the pass-like control has a similar sparse-context opportunity count.

## Kill Criteria

- Required trace summary paths are missing.
- Any selected trace is incomplete or illegal.
- The diagnostic cannot read per-step logit examples or generated timepoints.

## Expected Failure Modes

- Existing trace summaries may omit enough timepoints for density context if `timepoint_preview_limit` was too small.
- Generated-density context can be too weak a proxy for musical density.
- A selector might need reference-free audio/control features not captured by this artifact-only diagnostic.

## Confounders

- Generated density is produced by the flawed v3 rollout itself.
- This does not evaluate whether choosing the event token improves F1.
- The pass-like control set has one case only.
- The CE trace slice is fixed-slice, not full-dataset inference.

## Expected Runtime / Runtime Budget

Expected runtime: under one minute. Stop if any required trace path is missing.

## Result Interpretation Plan

- Positive result would suggest: create a bounded density-aware selector or event-budget objective card.
- Negative result would suggest: kill simple/local v3 selector work and pivot to target-grammar/training repair.
- Ambiguous result would require: add audio/control features to the trace only if they define a concrete selector guard.
- Human owner decides: whether to spend another v3 mutation or move to grammar repair.
- Next-loop action if positive: `TEST_DENSITY_AWARE_SELECTOR_OR_OBJECTIVE`.
- Next-loop action if negative: `MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR`.
- Next-loop action if ambiguous: `TEST_CONTROL_FEATURE_TRACE_ONLY`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
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
- Remaining ambiguity: whether density context separates the one strong CE opportunity case from the other CE-starved failures.

## Next-Loop Action

- If positive: define one density-aware selector/training objective card.
- If negative: pivot to target-grammar/training repair.
- If ambiguous: add control/audio trace fields only if they create a concrete selector guard.

## Novelty Notes

- Closest analogies: oracle decoding diagnostics and feature-separability audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: evaluator instrumentation only.
