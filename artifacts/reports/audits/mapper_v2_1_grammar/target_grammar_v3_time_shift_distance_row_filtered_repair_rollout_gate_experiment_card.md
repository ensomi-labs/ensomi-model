# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Rollout Gate Experiment Card

## Mode

- Mode: planner
- Route: `TEST`
- Source idea: the row-filtered repair passed the diagnostic and report-only tiny training gate, which routed to `TEST_ROLLOUT_GATE`.
- Acceptance source: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_summary.json`
- Source snapshot / evidence grade: medium local evidence; training is only 80 steps, but the baseline/enabled comparison is matched.

## Hypothesis

If the repaired time-shift distance auxiliary is numerically viable and directionally useful, then a small high-risk rollout pair gate should stay legal and should not introduce new second-window starvation or rigid-spacing regression versus the matched baseline checkpoint.

## Root Objective

Move the C3/v3 grammar path toward full-pipeline use by testing whether the repaired time-shift distance auxiliary survives the first generated-output gate before any full32 or full-dataset escalation.

## Goal Decomposition

- Subgoal 1: run matched baseline/enabled rollouts on high-risk fixed-slice cases using the same tiny-gate checkpoints.
- Subgoal 2: compare legality, max-token/dead-end status, rigid-spacing ratio, second-window event share, event-count ratio, and F1@100ms.
- Subgoal 3: route to full32 only if rollout checks pass; otherwise mutate or kill the objective before spending larger runtime.

## Candidate Variants

- Variant A: 3-case high-risk rollout pair gate from prior 32-case worst-case groups.
- Variant B: 2-case smoke gate with only the most starved/rigid cases.
- Variant C: full 32-case rollout immediately.
- Variant D: run synthetic-audio rollouts only.

## Local Verification Matrix

- Variant A: pass if all enabled rollouts are legal, no new starved cases appear, and mean dominant-spacing ratio does not worsen by more than `0.02`.
- Variant B: useful for plumbing only, but too narrow to cover overgeneration and starvation separately.
- Variant C: rejected until a small real-audio gate passes.
- Variant D: rejected because timing behavior must be checked on real audio/beatmaps.

## Selected Variant

- Selected: Variant A.
- Rejected: B, C, and D.
- Why this is the smallest useful test: three cases cover rigid-grid, starvation, and overgeneration without running the full32 suite.

## Selection Pressure

- Primary pressure: enabled rollouts remain legal and do not create new starved cases.
- Guard pressure: report-only training route remains `TEST_ROLLOUT_GATE`; no code changes are expected for rollout execution.
- Runtime pressure: six real-audio rollouts should finish locally without full32 runtime.
- Kill pressure: if enabled rollouts are illegal, max-token out, or materially worse on rigid/starved metrics, do not escalate this objective.

## Research Question

Does the row-filtered time-shift distance auxiliary improve or at least preserve early generated-output behavior on high-risk v3 rollout cases?

## Closest Analogies / Novelty Layer

- Closest analogies: tiny rollout safety gates after teacher-forced auxiliary loss changes; matched checkpoint A/B rollout probes.
- Relevant taxonomy bucket: training-objective validation and generated-output safety gate.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering validation of an auxiliary loss repair, not representation novelty.

## Minimal Change

Run existing rollout/evaluator tooling. Do not change tokenizer, grammar masks, loss code, model code, decode defaults, or training configs.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_pairs.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_result_report.md`
- rollout summaries/reports under `artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_rollout_gate/`

## Read-Only Context Files

- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_tiny_training_gate.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json`
- `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json`

## Dataset Slice

Use three real-audio fixed-slice cases selected from prior v3 worst-case groups:

- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: rigid `0.9744`, second-window share `0.0`, F1 `0.2481`
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: rigid `0.9802`, event-count ratio `1.3247`
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: event-count ratio `2.1579`

Each case uses `chart_end_ms=16000`, real audio, greedy decode, `max_tokens_per_window=512`, and full timepoint previews.

## Baseline / Comparator

- Baseline checkpoint: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/checkpoint.pt`
- Enabled checkpoint: `artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/checkpoint.pt`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`

## Primary Metric

The tiny-gate rollout comparator routes to `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE` only if all rollout checks pass.

## Secondary Metric

- per-case legality and completed flags
- dominant-spacing ratio delta
- second-window event-share delta
- event-count ratio delta
- F1@100ms delta
- new starved case count

## Verify Command / Evaluation Procedure

1. Run baseline and enabled trained runtime rollouts for each selected case.
2. Write a rollout-pair JSON file with baseline/enabled summaries plus beatmap path and chart end.
3. Rerun `pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate` with `--rollout-pairs-json`.
4. Write summary/report artifacts and inspect route.

## Guard Check

Run the existing rollout/model guard:

```bash
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py -q
```

## Qualitative Check

Inspect timepoint previews for repeated 100/160/320ms grids, second-window collapse, event-count explosions, dead ends, and max-token cases.

## Positive Signal

Route `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE` if all training and rollout checks pass.

## Negative Signal

Route `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE` if enabled rollouts are illegal, introduce new starved cases, exceed max tokens, or worsen mean dominant-spacing ratio beyond the gate.

## Kill Criteria

Kill or reformulate the current auxiliary if the enabled checkpoint fails legality/max-token checks or materially worsens both rigid and starvation behavior versus baseline on this high-risk slice.

## Expected Failure Modes

- 80-step checkpoints are too undertrained for stable rollout interpretation.
- The auxiliary is teacher-forced stable but does not improve greedy decode behavior.
- Real-audio timing extraction dominates the comparison noise.
- Enabled and baseline both fail, making the auxiliary effect ambiguous.

## Confounders

- This is not a full32 or full-dataset result.
- The selected cases are intentionally high-risk and may overstate failure rates.
- The baseline/enabled checkpoints are tiny 80-step runs, not production candidates.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes for six real-audio rollouts plus comparator and tests. Stop after this small gate; do not expand to full32 in the same card.

## Result Interpretation Plan

- Positive result would suggest: create a full32 repaired-loss training/rollout card.
- Negative result would suggest: mutate or kill the time-shift distance objective before full32.
- Ambiguous result would require: add a narrower rollout diagnostic separating undertraining from objective effect.
- Human owner decides: whether to keep the auxiliary in the C3/v3 path.
- Next-loop action if positive: `TEST` full32 repaired-loss gate.
- Next-loop action if negative: `MUTATE` or `KILL` this auxiliary.
- Next-loop action if ambiguous: `MUTATE` into a more diagnostic rollout card.

## Result Log Template

- Experiment: Target grammar v3 time-shift distance row-filtered repair rollout gate
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
- Closed loop complete: no
- Remaining ambiguity: full32/full-dataset behavior is out of scope.

## Next-Loop Action

- If positive: `TEST` full32 repaired-loss gate.
- If negative: `MUTATE` or `KILL` the objective.
- If ambiguous: `MUTATE` into a narrower diagnostic rollout card.

## Novelty Notes

- Closest analogies: matched checkpoint rollout safety gates after an auxiliary loss change.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering validation only.
