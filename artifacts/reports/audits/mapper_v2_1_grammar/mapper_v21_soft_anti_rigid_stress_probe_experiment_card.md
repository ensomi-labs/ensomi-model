# Mapper v2.1 Soft Anti-Rigid Stress Probe Experiment Card

## Mode

- Mode: executor
- Route: TEST
- Source idea: hard-block anti-rigid spacing reduced v2.1 rigidity on the fixed slice but failed legality/starvation guards.
- Acceptance source, if any: `mapper_v21_v3_fixed_slice_decode_comparison_result_report.md`
- Source snapshot / evidence grade: local runtime evidence, strong for the fixed 32-map slice, weak for held-out/full-4K replacement.

## Hypothesis

A finite anti-rigid spacing penalty can retain part of the hard-block rigidity reduction while avoiding the dead-end and starvation failures that made the hard-block guard non-scalable.

## Root Objective

Decide whether the v2.1 anti-rigid path should mutate from hard blocking to a softer decode constraint, or be killed as a brittle local intervention.

## Goal Decomposition

- Subgoal 1: test the exact hard-block failure cases without rerunning the full 32-map sweep.
- Subgoal 2: compare soft penalty against committed v2.1 baseline, hard-block guard, and v3 metrics on the same cases.
- Subgoal 3: preserve legality and continuation before spending another full fixed-slice run.

## Candidate Variants

- Variant A: finite penalty `4.0`, `hard_block=False`, same repeated-spacing detector, on four stress cases.
- Variant B: finite penalty `2.0`, weaker intervention, on the same stress cases.
- Variant C: boundary-aware hard block that disables blocking near window end.
- Variant D: immediate full 32-map soft-penalty sweep.

## Local Verification Matrix

- Variant A: smallest check is four stress-case soft rollouts; pass if all are legal, no new starvation, and rigidity still drops versus baseline.
- Variant B: same check, but likely too weak if logits strongly favor the rigid shift.
- Variant C: targets the `7990ms` dead-end symptom but does not address F1/starvation regressions away from the boundary.
- Variant D: stronger evidence but wasteful before checking whether soft penalty fixes the known failures.

## Selected Variant

- Selected: Variant A.
- Rejected: B is a fallback if A is too aggressive; C is narrower than the observed failures; D is too much runtime before the stress probe.
- Why this is the smallest useful test: it reuses committed 32-map hard-block evidence and only reruns the candidate mutation on cases most likely to refute it quickly.

## Selection Pressure

- Primary pressure: soft guard must complete legally on all selected stress cases.
- Guard pressure: soft guard must not introduce starvation versus committed v2.1 baseline on the selected cases.
- Runtime pressure: four real-audio 16s rollouts should finish in minutes.
- Kill pressure: any soft-guard dead-end, max-token failure, or new starvation kills this penalty value before a full sweep.

## Research Question

Is the hard-block failure caused by infinite suppression specifically, or by the repeated-spacing detector itself?

## Closest Analogies / Novelty Layer

- Closest analogies: repetition penalty, constrained decoding, finite logit penalty, timing-grid regularization.
- Relevant taxonomy bucket: engineering variation on decode-time grammar constraints.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering variation, not a new token representation.

## Minimal Change

- Add a small stress-probe evaluator that reads the committed hard-block fixed-slice summary.
- Select four cases: the two illegal/dead-end cases plus the high-regression stress cases `29` and `31`.
- Run only the soft-penalty v2.1 variant for those cases.
- Compare soft metrics against committed v2.1 baseline, hard-block guard, and v3 metrics.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v21_soft_anti_rigid_stress_probe.py`
- `tests/evals/test_mapper_v21_soft_anti_rigid_stress_probe.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_result_report.md`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`

## Dataset Slice

- `04_oomori_seiko_justadice_tv_size_remu_normal`
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- Prefix: `0-16000ms`.
- Real audio and selected normalized difficulty from the committed hard-block summary.

## Baseline / Comparator

- Committed v2.1 baseline metrics from `mapper_v21_v3_fixed_slice_decode_comparison_summary.json`.
- Committed hard-block guard metrics from the same summary.
- Committed v3 500-step fixed-slice metrics from the same summary.

## Primary Metric

- Soft guard legality: all four soft rollouts complete with no dead-end and no max-token failure.
- Continuation safety: soft guard introduces zero new starved cases versus committed v2.1 baseline.

## Secondary Metric

- Soft versus baseline dominant-spacing ratio delta.
- Soft versus hard-block F1 delta.
- Soft versus v3 F1/starvation.
- Event-count ratio and second-window share.

## Verify Command / Evaluation Procedure

1. Read the committed hard-block fixed-slice summary.
2. Run soft-penalty v2.1 rollout for the four selected cases.
3. Recompute generated/reference metrics with the existing timing metric implementation.
4. Compare against committed baseline, hard-block, and v3 metrics.
5. Write summary JSON and Markdown report.

## Guard Check

- `uv run --group dev pytest tests/evals/test_mapper_v21_soft_anti_rigid_stress_probe.py tests/evals/test_mapper_v21_v3_fixed_slice_decode_comparison.py tests/inference/test_mapper_v2_1_rollout.py -q`
- `uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_summary.json`

## Qualitative Check

Inspect whether soft penalty fixes the two `7990ms` dead-end patterns and whether cases `29`/`31` still collapse into low-F1 or starved outputs.

## Positive Signal

- All four soft rollouts are legal.
- Soft guard has zero new starved cases versus committed v2.1 baseline.
- Soft guard improves dominant-spacing ratio versus baseline in at least three of four cases.
- Soft guard mean F1 is no worse than hard-block mean F1 by more than `0.02`.

## Negative Signal

- Any soft rollout is illegal.
- Any new starvation appears versus committed v2.1 baseline.
- Soft guard does not reduce rigidity in at least three cases.
- Soft guard F1 is materially worse than hard-block on the selected stress cases.

## Kill Criteria

Kill penalty `4.0` immediately if it dead-ends, max-token fails, or creates new starvation on the stress set. Do not scale to 32 cases unless the stress probe passes.

## Expected Failure Modes

- Penalty `4.0` may still be too strong and behave like a hard block.
- Penalty `4.0` may be too weak to disrupt high-confidence rigid grids.
- Baseline metrics for illegal committed cases are partial, so interpret baseline-vs-soft deltas carefully.
- The stress set is intentionally biased toward failures and is not a general quality estimate.

## Confounders

- This is decode-only and does not change tokenizer bits or training targets.
- The v2.1 checkpoint is still the matched 200-step checkpoint.
- The selected cases are from the training slice, not held-out data.
- Timing F1 can reward regular grids.

## Expected Runtime / Runtime Budget

Expected runtime is under 10 minutes. Stop on missing checkpoint, repeated runtime exception, or if the first two stress cases both fail legality.

## Result Interpretation Plan

- Positive result would suggest: finite penalty deserves a full 32-map fixed-slice comparison.
- Negative result would suggest: the repeated-spacing detector or this penalty value is brittle; mutate to a different detector or return to v3/C3 diagnostics.
- Ambiguous result would require: a small penalty sweep over `2.0`, `4.0`, and `6.0` on the same stress cases.
- Human owner decides: whether soft decode constraints are worth another loop.
- Next-loop action if positive: run a full 32-map soft-penalty comparison.
- Next-loop action if negative: kill penalty `4.0` and define a different grammar mutation.
- Next-loop action if ambiguous: bounded penalty sweep, not a full run.

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
- Remaining ambiguity: whether penalty `4.0` is the right finite strength; this probe answers only that selected value.

## Next-Loop Action

- If positive: full 32-map soft-penalty comparison.
- If negative: kill this penalty value and mutate away from hard blocking.
- If ambiguous: small penalty sweep on the same stress set.

## Novelty Notes

- Closest analogies: repetition penalty and constrained decoding.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation only.
