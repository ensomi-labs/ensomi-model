# Target Grammar v3 CE Residual Cluster Diagnostic Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: The C3/v3 target-complexity gate selected `MUTATE_TO_V3_GRAMMAR_REPAIR`, and the CE-weight v3 training gate is the strongest current v3 rollout signal: it improved starvation and F1 while failing on rigid-grid cases.
- Acceptance source, if any: active full-pipeline v3 goal plus `target_grammar_v3_event_token_ce_weight_training_gate_result_report.md`, `target_grammar_v3_ce_antirigid_full32_result_report.md`, and `c3_v3_target_complexity_comparison_result_report.md`.
- Source snapshot / evidence grade: strong evidence that v3 is reversible and lower-bit; strong evidence that C3 should not be the next mapper target; strong evidence that CE weighting helps continuation but worsens rigidity; medium evidence that decode anti-rigid control overcorrects event count.

## Hypothesis

The current v3 branch should use the CE-weight checkpoint as the best baseline for the next grammar repair. If CE residual failures are mostly rigid high-F1 grids with fewer starvation/boundary failures than the 500-step baseline, the next bounded mutation should target timing diversity around CE, not another broad event-density or C3 target experiment. If CE still shares the same starvation/boundary-clamp failures with the baseline, the next mutation should target continuation/completion semantics or pivot to v2.1 grammar improvement.

## Root Objective

Choose the next smallest v3/v2.1 grammar-repair experiment after C3 mapper-target diminishing returns and CE-weight partial success.

## Goal Decomposition

- Subgoal 1: Compare baseline 500-step v3 and CE-weight v3 on the same 32 fixed-slice cases.
- Subgoal 2: Classify residual failures as starvation, boundary clamp, rigid grid, undergeneration, or overgeneration.
- Subgoal 3: Decide whether the next bounded experiment should be CE timing-diversity repair, continuation/completion repair, v2.1 grammar work, or richer instrumentation.

## Candidate Variants

- Variant A: Rerun the existing generated-state diagnostic with CE included. Selected because it reuses a tested evaluator and needs no training.
- Variant B: Train another scalar loss immediately. Rejected because density, event-budget, and continuation-jump scalar paths already failed.
- Variant C: Run another anti-rigid decode sweep. Rejected because hard-block and soft-penalty full32 gates failed on event inflation.
- Variant D: Pivot directly to v2.1 grammar repair. Deferred until CE residual clusters are recorded.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Artifact-only residual diagnostic | CE residuals isolate a smaller next lever | CE shares baseline starvation/boundary failures |
| B | New scalar-loss training | Improves all full32 gates | Prior scalar losses regress legality/distribution |
| C | Decode suppression | Removes rigidity without event inflation | Prior suppression overproduces |
| D | v2.1 repair | v3 has no smaller positive route | CE residual evidence not yet checked |

## Selected Variant

- Selected: Variant A, CE residual cluster diagnostic.
- Rejected: B and C because prior evidence already shows diminishing returns for broad scalar losses and decode suppression.
- Deferred: D until the diagnostic decides whether CE leaves a tractable v3 repair target.
- Why this is the smallest useful test: it only reads existing summaries and uses the established generated-state classifier.

## Selection Pressure

- Primary pressure: identify whether CE residual failures are mainly rigidity or continuation failure.
- Guard pressure: no training, tokenization, rollout, inference, or defaults change.
- Runtime pressure: under 10 seconds.
- Kill pressure: if CE residuals are not separable from baseline failure clusters, stop CE-specific repair and pivot to continuation instrumentation or v2.1 grammar work.

## Research Question

After CE weighting improves v3 continuation but worsens rigidity, is there a narrow timing-diversity repair target left, or should the branch pivot away from v3 scalar/decoding mutations?

## Closest Analogies / Novelty Layer

- Closest analogies: generated-state error slicing, exposure-bias diagnostics, constrained-decoding audit selection, symbolic target grammar repair.
- Relevant taxonomy bucket: local verification before bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for selecting the next v3/v2.1 grammar repair.

## Minimal Change

Run the existing `mapper_v3_continuation_cluster_diagnostic` evaluator on two existing summaries:

- `baseline_500step`
- `ce_weight2`

Write a new summary and result report. No source changes are expected unless the existing evaluator cannot consume the CE summary.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/mapper_v3_continuation_cluster_diagnostic.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json`

## Dataset Slice

The fixed 32-case, 16s real-audio v3 rollout universe used by the 500-step baseline and CE-weight training gate.

## Baseline / Comparator

- Baseline: `baseline_500step`.
- Comparator: `ce_weight2`.

## Primary Metric

Failure-class deltas:

- starved count;
- rigid count;
- boundary-clamped count;
- undergeneration/overgeneration count;
- mean F1;
- median event-count ratio.

## Secondary Metric

- Event-valid ratio.
- Event top-k minus top-1 ratio.
- Shared failure overlap by case id.
- Highest-leverage residual CE cases.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_continuation_cluster_diagnostic \
  --summary baseline_500step=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json \
  --summary ce_weight2=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json \
  --baseline baseline_500step \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_continuation_cluster_diagnostic.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_residual_cluster_diagnostic_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must state whether CE residuals support a timing-diversity repair, continuation/completion repair, richer instrumentation, or v2.1 grammar pivot.

## Positive Signal

CE reduces starvation/boundary failures enough that residual failure is mostly rigid-grid timing diversity with legal rollouts.

## Negative Signal

CE residual failures remain dominated by the same starvation/boundary-clamp cases as the baseline, or CE introduces a new overgeneration family.

## Kill Criteria

- Required summaries are missing or incompatible.
- Case ids cannot be matched across variants.
- CE residuals do not isolate a smaller repair target.

## Expected Failure Modes

- The existing classifier may be too coarse for CE-specific timing diversity.
- The CE summary may lack event-rank fields for some cases.
- A residual rigid-grid label may mix high-F1 harmless grids with low-F1 collapse.

## Confounders

- This is not a new rollout; it reuses existing CE artifacts.
- CE weighting is a training-side intervention, not a grammar change.
- A positive residual cluster does not prove the next timing-diversity mutation will work.

## Expected Runtime / Runtime Budget

Expected runtime: under 10 seconds. Stop if the summaries cannot be parsed or matched.

## Result Interpretation Plan

- Positive result would suggest: create a bounded CE timing-diversity repair card.
- Negative result would suggest: pivot to continuation/completion instrumentation or v2.1 grammar repair.
- Ambiguous result would require: add per-step trace instrumentation on CE residual cases.
- Human owner decides: whether to spend another v3 loop or move to v2.1 grammar work.
- Next-loop action if positive: CE timing-diversity repair card.
- Next-loop action if negative: v2.1 grammar or continuation instrumentation card.
- Next-loop action if ambiguous: CE residual trace audit.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Input summaries:
- Runtime:
- Failure-class table:
- CE residual cases:
- Guard result:
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
- Remaining ambiguity: this selects a next lever; it does not prove the repair.

## Next-Loop Action

- If positive: create a CE timing-diversity repair card.
- If negative: pivot to continuation instrumentation or v2.1 grammar improvement.
- If ambiguous: trace the CE residual cases.

## Novelty Notes

- Closest analogies: rollout failure clustering and grammar/decode ablation selection.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering triage for v3 replacement readiness.
