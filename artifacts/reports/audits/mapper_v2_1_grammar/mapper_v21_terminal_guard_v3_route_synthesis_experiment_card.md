# Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: compare the newly passed v2.1 terminal LN-start guard repair against the current v3/C3 route before changing defaults.
- Acceptance source, if any: `mapper_v21_terminal_ln_start_guard_repair_full32_result_report.md`
- Source snapshot / evidence grade: strong local artifact evidence; full32 real-audio v2.1 repair passed, v3/C3 prior summaries already committed.

## Hypothesis

The v2.1 terminal LN-start guard repair invalidates the previous v2.1 legality blocker, but it should not replace the v3 route unless it also beats the current v3 candidate on the same fixed-slice quality/readiness gates. An artifact-only synthesis can route the next work without rerunning rollout.

## Root Objective

Preserve progress toward a final local, reversible, lower-bit v3 target grammar while using the now-passing v2.1 terminal guard as a stronger comparator and short-term grammar hardening result.

## Goal Decomposition

- Subgoal 1: Verify that the new v2.1 terminal guard full32 artifact passes legality, max-token, starvation, and F1-regression gates.
- Subgoal 2: Recompare the repaired v2.1 metrics with the existing same-slice v3 metrics.
- Subgoal 3: Route the next branch among v2.1 opt-in/default gate, v3 generated-prefix repair, or renewed C3 target-side integration.

## Candidate Variants

- Variant A: artifact-only route synthesis using the terminal-guard full32 summary, the fixed-slice v2.1/v3 comparison summary, and current v3/C3 synthesis summaries.
- Variant B: rerun full32 real-audio v2.1/v3 comparison with terminal guard wired directly into the original comparison evaluator.
- Variant C: broaden the v2.1 terminal guard immediately to a larger dataset slice.
- Variant D: ignore the v2.1 repair and continue the existing v3 generated-prefix trace route.

## Local Verification Matrix

- Variant A: pass if all required summaries exist, schemas load, per-case v2.1 repair/v3 comparison is computable, and the route is driven by explicit guards.
- Variant B: pass if rerun completes and reproduces legality/quality; fail if runtime dominates without changing interpretation.
- Variant C: pass if it answers default-readiness; fail as premature if same-slice v3 comparison has not been synthesized.
- Variant D: pass if the v2.1 repair is irrelevant; fail because it leaves a new comparator unaccounted.

## Selected Variant

- Selected: Variant A.
- Rejected: Variant B is more expensive than needed; Variant C is premature; Variant D ignores new evidence.
- Why this is the smallest useful test: it consumes already committed artifacts and directly answers the next routing question with no training or rollout rerun.

## Selection Pressure

- Primary pressure: route the next bounded experiment after the v2.1 terminal guard full32 pass.
- Guard pressure: no default change, no training, no rollout rerun, no tokenizer mutation.
- Runtime pressure: under 1 minute.
- Kill pressure: missing artifacts, schema mismatch, v2.1 repair fails hard guards, or v3 representation readiness is not proven.

## Research Question

Does the v2.1 terminal LN-start guard repair change the branch route from v3 generated-prefix repair to v2.1 default promotion or C3 integration?

## Closest Analogies / Novelty Layer

- Closest analogies: codec-vs-target representation audits, teacher-forcing target replacement gates, artifact-only route synthesis.
- Relevant taxonomy bucket: local verification / route synthesis after bounded experiments.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering route selection, not a new representation claim.

## Minimal Change

Add an artifact-only evaluator that loads existing summaries, computes v2.1 terminal-guard vs v3 same-slice checks, writes JSON and Markdown reports, and has focused unit tests.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v21_terminal_guard_v3_route_synthesis.py`
- `tests/evals/test_mapper_v21_terminal_guard_v3_route_synthesis.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_guard_v3_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_guard_v3_route_synthesis_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_full32_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json`

## Dataset Slice

The existing fixed-slice full32 real-audio cases represented in the committed summaries.

## Baseline / Comparator

- Baseline: previous v2.1/v3 fixed-slice comparison summary and v3 full-dataset representation audit.
- Candidate: v2.1 terminal LN-start guard full32 summary.

## Primary Metric

Route decision with explicit guard results.

## Secondary Metric

Same-slice candidate-vs-v3 deltas: mean F1 delta, starved-count delta, event-count-ratio delta, dominant-spacing-ratio delta, and legality counts.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v21_terminal_guard_v3_route_synthesis
uv run --group dev pytest tests/evals/test_mapper_v21_terminal_guard_v3_route_synthesis.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_guard_v3_route_synthesis_summary.json >/dev/null
```

## Guard Check

`git diff --check` and focused pytest must pass. The evaluator must report artifact-only/no-training/no-rollout-rerun/no-default-change guards as true.

## Qualitative Check

The report must clearly distinguish short-term v2.1 hardening from final v3 replacement readiness, and it must not resurrect C3 production-input claims.

## Positive Signal

The synthesis records v2.1 terminal guard as a validated opt-in comparator while preserving the v3 generated-prefix repair route for final replacement.

## Negative Signal

The synthesis finds that the v2.1 repair fails hidden comparison guards or that required v3/C3 evidence is missing/stale.

## Kill Criteria

- Any required artifact missing.
- v2.1 terminal guard full32 is not legal or has new starvation/max-token failures.
- v3 representation audit no longer proves exact reconstruction and bit/token reduction.
- C3 production-input legality is incorrectly treated as true.
- Report cannot compute same-slice v2.1-vs-v3 metrics.

## Expected Failure Modes

- Summary schema drift.
- Case IDs fail to align between terminal-guard and original comparison summaries.
- Ambiguous route because v2.1 beats v3 on short-slice rollout metrics but v3 remains the only lower-bit local replacement grammar.

## Confounders

The fixed-slice mapper-quality comparison is not a full 4k rollout audit. The v2.1 terminal guard uses a v2.1 checkpoint and anti-rigid guard, while v3 replacement requires a trained v3 checkpoint with broader rollout quality.

## Expected Runtime / Runtime Budget

Under 1 minute. Stop if required artifacts are missing or the per-case comparison cannot be computed.

## Result Interpretation Plan

- Positive result would suggest: keep v2.1 terminal guard as opt-in/stability comparator and continue v3 generated-prefix repair.
- Negative result would suggest: rerun or broaden the v2.1/v3 comparison before routing.
- Ambiguous result would require: a direct rerun comparison or broader v2.1 terminal guard gate.
- Human owner decides: whether v2.1 terminal guard should become default after broader validation.
- Next-loop action if positive: execute the next v3 generated-prefix/state repair or create a v2.1 broader default gate.
- Next-loop action if negative: repair synthesis inputs or rerun comparison.
- Next-loop action if ambiguous: create a direct rerun comparison card.

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
- Remaining ambiguity: fixed-slice comparison cannot prove full replacement readiness.

## Next-Loop Action

- If positive: keep v2.1 guard opt-in and continue v3 generated-prefix repair/full-pipeline quality gates.
- If negative: rerun direct comparison or broaden v2.1 guard before route selection.
- If ambiguous: create a direct v2.1 terminal-guard vs v3 rerun card.

## Novelty Notes

- Closest analogies: route synthesis and ablation-gate reconciliation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering verification and routing only.
