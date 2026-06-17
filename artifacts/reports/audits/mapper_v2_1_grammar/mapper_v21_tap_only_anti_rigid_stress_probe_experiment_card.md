# Mapper v2.1 Tap-Only Anti-Rigid Stress Probe Experiment Card

## Mode

- Mode: executor
- Route: TEST
- Source idea: simple repeated-spacing suppression failed because case `05` is an LN alternation where changing one early spacing creates a first-window dead-end.
- Acceptance source, if any: `mapper_v21_soft_penalty_sweep_stress_result_report.md`
- Source snapshot / evidence grade: local runtime evidence plus generated action inspection; strong for the four stress cases, weak for full-slice quality.

## Hypothesis

The anti-rigid detector should target tap-only repeated grids, not LN start/end alternations. If suppression is skipped when the recent equal-spacing run contains LN actions, the invariant case `05` should stay legal/non-starved while tap-grid stress cases can still receive rigidity reduction.

## Root Objective

Decide whether the v2.1 anti-rigid branch has a viable detector-side mutation after hard-block and finite-penalty schedules failed.

## Goal Decomposition

- Subgoal 1: add an opt-in tap-only recent-run filter to the anti-rigid transform.
- Subgoal 2: rerun the same four stress cases with hard-block suppression enabled only for tap-only repeated runs.
- Subgoal 3: compare against committed v2.1 baseline, hard-block, soft-penalty sweep, and v3 context metrics.

## Candidate Variants

- Variant A: require the recent repeated-spacing run to contain only TAP lane actions.
- Variant B: disable suppression only within the first 2 seconds.
- Variant C: disable suppression near the first-window boundary.
- Variant D: abandon v2.1 anti-rigid work and return to v3/C3 diagnostics.

## Local Verification Matrix

- Variant A: pass if all four stress rollouts are legal, no new starvation appears versus baseline, case `05` no longer fails, and at least two cases still reduce rigidity.
- Variant B: may avoid case `05`, but arbitrary time threshold would also miss genuine early tap grids.
- Variant C: targets the 7990ms dead-end symptom but not the causal early LN perturbation.
- Variant D: reasonable if detector mutation fails, but premature before testing the LN-specific hypothesis.

## Selected Variant

- Selected: Variant A.
- Rejected: B/C are symptom-based schedules; D is a next-loop fallback if A fails.
- Why this is the smallest useful test: it changes only the detector predicate and reruns the existing four-case stress probe.

## Selection Pressure

- Primary pressure: all four tap-only rollouts must complete legally.
- Guard pressure: zero new starved cases versus committed v2.1 baseline.
- Runtime pressure: four real-audio stress rollouts should finish quickly.
- Kill pressure: if case `05` still dead-ends or starves, the LN-specific hypothesis is false.

## Research Question

Is the anti-rigid failure caused by applying tap-grid suppression to LN alternations?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding, repetition blocking with token-class filters, rhythm-grid regularization.
- Relevant taxonomy bucket: decode-time grammar engineering.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation only.

## Minimal Change

- Add an opt-in `require_tap_only_run` flag to `MapperV21AntiRigidSpacingLogitsTransform`.
- The flag requires all lane actions in the repeated-spacing run to be TAP, otherwise no candidate is suppressed.
- Add a stress-probe evaluator for the four committed cases.

## Files Likely to Change

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `tests/inference/test_mapper_v2_1_rollout.py`
- `src/pulsefield_model/evals/mapper_v21_tap_only_anti_rigid_stress_probe.py`
- `tests/evals/test_mapper_v21_tap_only_anti_rigid_stress_probe.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`

## Dataset Slice

- `04_oomori_seiko_justadice_tv_size_remu_normal`
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- Prefix: `0-16000ms`.
- Real audio and normalized difficulty from the committed hard-block fixed-slice summary.

## Baseline / Comparator

- Committed v2.1 baseline metrics.
- Committed hard-block anti-rigid metrics.
- Committed soft penalty sweep metrics.
- v3 500-step fixed-slice metrics as context only.

## Primary Metric

- Tap-only guard legality: all four stress cases legal.
- Tap-only continuation safety: zero new starved cases versus committed v2.1 baseline.
- Detector-specific check: case `05` legal and non-starved.

## Secondary Metric

- Rigid-improved case count versus baseline.
- Mean F1 delta versus baseline and hard-block.
- Transform candidate/block counts.
- Difference from hard-block and baseline outcomes.

## Verify Command / Evaluation Procedure

1. Run unit tests for the new tap-only detector predicate.
2. Run the tap-only stress evaluator on the four cases.
3. Aggregate metrics against committed comparators.
4. Validate JSON and record the result report.

## Guard Check

- `uv run --group dev pytest tests/inference/test_mapper_v2_1_rollout.py tests/evals/test_mapper_v21_tap_only_anti_rigid_stress_probe.py -q`
- `uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_summary.json`

## Qualitative Check

Confirm that case `05` does not trigger suppression on the LN alternation and that tap-grid cases still trigger suppression when the run is tap-only.

## Positive Signal

- All four stress rollouts are legal.
- No new starvation versus baseline.
- Case `05` legal and non-starved.
- At least two cases reduce dominant-spacing ratio versus baseline.

## Negative Signal

- Case `05` still dead-ends or starves.
- Tap-only mode suppresses nothing useful and improves fewer than two cases.
- Tap-only mode matches baseline exactly across all four cases.
- Tap-only mode recreates hard-block starvation on case `29`.

## Kill Criteria

Kill the tap-only detector mutation if it fails legality/continuation on the stress set, or if it avoids failures only by becoming identical to baseline.

## Expected Failure Modes

- Some rigid failures may include LN actions and therefore be skipped.
- Tap-only filtering may be too conservative.
- The four-case stress set is intentionally biased and does not estimate full-slice performance.
- Greedy decode may amplify a single allowed/suppressed token.

## Confounders

- Decode-only change; no target grammar or training effect.
- v2.1 checkpoint is undertrained.
- Timing F1 can reward regular grids.
- This does not satisfy v3 replacement requirements.

## Expected Runtime / Runtime Budget

Expected runtime is under 10 minutes. Stop on missing checkpoint or repeated runtime exception.

## Result Interpretation Plan

- Positive result would suggest: run a full 32-map tap-only comparison before defaults.
- Negative result would suggest: kill v2.1 anti-rigid suppression family and pivot back to v3/C3 diagnostics.
- Ambiguous result would require: inspect transform examples/logit margins on skipped cases.
- Human owner decides: whether v2.1 decode grammar work remains worth another loop.
- Next-loop action if positive: full fixed-slice tap-only comparison.
- Next-loop action if negative: return to v3/C3 diagnostics.
- Next-loop action if ambiguous: logit-margin/skipped-candidate diagnostic.

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
- Remaining ambiguity: a pass only proves stress-case viability, not full fixed-slice readiness.

## Next-Loop Action

- If positive: full fixed-slice tap-only comparison.
- If negative: kill anti-rigid suppression family and pivot to v3/C3.
- If ambiguous: logit-margin/skipped-candidate diagnostic.

## Novelty Notes

- Closest analogies: class-filtered repetition penalty and constrained decoding.
- Novelty layer, if any: none.
- Representation novelty vs engineering variation: engineering variation only.
