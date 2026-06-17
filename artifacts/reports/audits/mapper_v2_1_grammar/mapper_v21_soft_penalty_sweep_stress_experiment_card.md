# Mapper v2.1 Soft Penalty Sweep Stress Experiment Card

## Mode

- Mode: executor
- Route: TEST
- Source idea: penalty `4.0` behaved exactly like the hard-block anti-rigid guard on the stress set.
- Acceptance source, if any: `mapper_v21_soft_anti_rigid_stress_probe_result_report.md`
- Source snapshot / evidence grade: local runtime evidence on four fixed-slice stress cases; strong for the specific failure cases, weak for general quality.

## Hypothesis

Weaker finite penalties can separate from hard-block behavior under greedy decoding. If a weaker penalty keeps all stress cases legal, avoids new starvation, and still reduces rigidity in at least part of the stress set, then soft anti-rigid suppression may deserve a full 32-map comparison. If weaker penalties either do nothing or still reproduce hard-block failure modes, this decode-suppression family should be deprioritized.

## Root Objective

Decide whether the v2.1 anti-rigid decode path has a viable finite-penalty regime, or whether the repeated-spacing detector/suppression family should be killed before more fixed-slice runtime.

## Goal Decomposition

- Subgoal 1: test whether penalties below `4.0` alter greedy outputs on the same four stress cases.
- Subgoal 2: enforce legality and continuation as hard guards before considering rigidity improvement.
- Subgoal 3: identify one selected penalty value for a possible full fixed-slice run, or kill the penalty schedule.

## Candidate Variants

- Variant A: sweep penalties `0.5`, `1.0`, and `2.0` on the four stress cases.
- Variant B: run only penalty `2.0`.
- Variant C: run a wider sweep including `0.25`, `0.5`, `1.0`, `2.0`, and `3.0`.
- Variant D: stop penalty work and design a new detector immediately.

## Local Verification Matrix

- Variant A: pass if at least one penalty is legal on all four cases, introduces zero new starvation, differs from hard-block on at least one case, and improves rigidity versus baseline in at least two cases.
- Variant B: faster but cannot distinguish too-weak from too-strong penalty behavior.
- Variant C: more complete but spends extra runtime before knowing whether any weak penalty changes outputs.
- Variant D: reasonable if penalty `4.0` were enough evidence, but it would leave open whether the failure was only due to penalty strength.

## Selected Variant

- Selected: Variant A.
- Rejected: B is under-instrumented; C is larger than necessary; D is premature because `4.0` may simply be too strong.
- Why this is the smallest useful test: three weaker penalties over four known stress cases should decide whether finite penalties can separate from hard blocking.

## Selection Pressure

- Primary pressure: a candidate penalty must make all four stress rollouts legal.
- Guard pressure: it must introduce zero new starved cases versus committed v2.1 baseline.
- Runtime pressure: 12 real-audio 16s rollouts should finish in minutes.
- Kill pressure: if all penalties fail legality/starvation or all output exactly the hard-block pattern, kill the simple finite-penalty schedule.

## Research Question

Is there a finite anti-rigid penalty strength that changes greedy v2.1 outputs without recreating hard-block dead-end/starvation behavior?

## Closest Analogies / Novelty Layer

- Closest analogies: repetition penalty, finite logit penalty, constrained decoding, timing-grid regularization.
- Relevant taxonomy bucket: decode-time engineering variation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation only.

## Minimal Change

- Add a sweep evaluator that wraps the existing `run_soft_anti_rigid_stress_probe` logic for penalties `0.5`, `1.0`, and `2.0`.
- Reuse committed baseline/hard-block/v3 comparators from `mapper_v21_v3_fixed_slice_decode_comparison_summary.json`.
- Write one sweep summary and one result report.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v21_soft_penalty_sweep_stress.py`
- `tests/evals/test_mapper_v21_soft_penalty_sweep_stress.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`
- `src/pulsefield_model/evals/mapper_v21_soft_anti_rigid_stress_probe.py`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`

## Dataset Slice

- `04_oomori_seiko_justadice_tv_size_remu_normal`
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- Prefix: `0-16000ms`.
- Real audio and normalized difficulty from the committed hard-block fixed-slice summary.

## Baseline / Comparator

- Committed v2.1 baseline metrics from `mapper_v21_v3_fixed_slice_decode_comparison_summary.json`.
- Committed hard-block metrics from the same summary.
- Penalty `4.0` result from `mapper_v21_soft_anti_rigid_stress_probe_summary.json`.
- Existing v3 500-step fixed-slice metrics as a context comparator, not a training-budget-matched winner.

## Primary Metric

For each penalty:

- all four rollouts legal,
- zero new starved cases versus committed v2.1 baseline,
- not identical to hard-block outcomes on all four cases.

## Secondary Metric

- Rigid-improved case count versus baseline.
- Mean F1 delta versus baseline and hard-block.
- Mean dominant-spacing ratio delta versus baseline.
- Starved count and event-count ratio.

## Verify Command / Evaluation Procedure

1. Run the sweep evaluator for penalties `0.5`, `1.0`, and `2.0`.
2. Aggregate each penalty independently.
3. Select the best passing penalty by legality first, then starvation, then rigidity improvement, then F1.
4. Write summary JSON and Markdown report.

## Guard Check

- `uv run --group dev pytest tests/evals/test_mapper_v21_soft_penalty_sweep_stress.py tests/evals/test_mapper_v21_soft_anti_rigid_stress_probe.py tests/inference/test_mapper_v2_1_rollout.py -q`
- `uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_summary.json`

## Qualitative Check

Check whether penalties below `4.0` actually change the case outputs, especially case `05` dead-end and case `29` starvation. If a weaker penalty leaves outputs identical to baseline, record it as too weak, not as a quality pass.

## Positive Signal

- At least one penalty is legal on all four cases.
- The selected penalty introduces zero new starvation versus baseline.
- The selected penalty improves rigidity versus baseline in at least two of four cases.
- The selected penalty differs from hard-block outcomes on at least one stress case.

## Negative Signal

- Every penalty has a dead-end or max-token failure.
- Every penalty introduces new starvation versus baseline.
- Every penalty either matches hard-block exactly or fails to change the baseline.
- No penalty improves rigidity in at least two cases.

## Kill Criteria

Kill the simple finite-penalty schedule if no tested penalty clears the primary metric. Do not run a full 32-map penalty sweep in that case.

## Expected Failure Modes

- Low penalties may be too weak to change greedy argmax.
- Medium penalties may still be effectively hard under high-confidence logits.
- A penalty can improve rigidity while damaging continuation.
- Stress cases are biased and do not estimate average chart quality.

## Confounders

- Decode-only experiment; no tokenizer or training change.
- Greedy decoding may hide softer effects that sampling would reveal.
- The v2.1 checkpoint is still undertrained and not a final full-pipeline model.
- Timing F1 can reward regular grids.

## Expected Runtime / Runtime Budget

Expected runtime is under 25 minutes for 12 real-audio stress rollouts. Stop if two penalties in a row fail by runtime exception rather than metric failure.

## Result Interpretation Plan

- Positive result would suggest: run a full 32-map comparison for the selected weak penalty.
- Negative result would suggest: kill simple finite penalties and mutate detector/schedule, or return to v3/C3 diagnostics.
- Ambiguous result would require: inspect logit margins before another rollout sweep.
- Human owner decides: whether the decode-suppression family remains worth pursuing.
- Next-loop action if positive: full fixed-slice run for selected penalty.
- Next-loop action if negative: design a detector-side mutation or stop v2.1 anti-rigid work.
- Next-loop action if ambiguous: logit-margin diagnostic on the stress cases.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Penalties:
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
- Remaining ambiguity: if weak penalties are too weak, this card will not distinguish detector weakness from model-confidence weakness without a later logit-margin diagnostic.

## Next-Loop Action

- If positive: full fixed-slice run for the selected penalty.
- If negative: kill simple finite penalties and mutate detector/schedule.
- If ambiguous: logit-margin diagnostic on stress cases.

## Novelty Notes

- Closest analogies: repetition penalty and finite constrained decoding.
- Novelty layer, if any: none.
- Representation novelty vs engineering variation: engineering variation only.
