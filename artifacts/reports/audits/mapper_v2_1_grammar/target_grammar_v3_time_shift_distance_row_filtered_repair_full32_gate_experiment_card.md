# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Full32 Gate Experiment Card

## Mode

- Mode: planner
- Route: `TEST`
- Source idea: the repaired time-shift distance auxiliary passed the real-batch diagnostic, matched 80-step training gate, and 3-case high-risk rollout gate.
- Acceptance source: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_summary.json`
- Source snapshot / evidence grade: medium; small rollout gate passed, but 80-step enabled rollouts were behavior-identical to baseline.

## Hypothesis

At a 500-step horizon, the repaired time-shift distance auxiliary may create a measurable generated-output effect while preserving the legality and starvation/rigidity guards that passed at 80 steps. If a matched full32 500-step gate still shows identical or worse behavior, the auxiliary has diminishing marginal returns and should be mutated or deprioritized.

## Root Objective

Move target grammar v3 toward full-pipeline readiness by testing whether the repaired time-shift distance objective remains useful beyond the tiny gate, before changing defaults or claiming v3 replacement readiness.

## Goal Decomposition

- Subgoal 1: train matched 500-step baseline and enabled checkpoints on the fixed 32-song/256-window slice.
- Subgoal 2: run full32 real-audio greedy rollouts for both checkpoints using the existing fixed-slice case universe.
- Subgoal 3: compare training, legality, rigid-spacing, second-window starvation, event-count ratio, and F1@100ms.
- Subgoal 4: route to broader v3 work only if the objective passes full32 guards; otherwise mutate or kill this auxiliary.

## Candidate Variants

- Variant A: matched baseline/enabled 500-step full32 gate with `event_token_loss_weight=2.0`, enabled `lambda_time_shift_distance=0.5`.
- Variant B: train only enabled 500-step and compare to the older 500-step horizon checkpoint.
- Variant C: change lambda/scale before full32.
- Variant D: skip full32 and move directly to full 4k training.

## Local Verification Matrix

- Variant A: pass if enabled training is finite and full32 rollouts do not regress legality, new-starvation, or mean rigid ratio versus matched baseline.
- Variant B: rejected because the older horizon checkpoint used a different loss recipe, so attribution would be weak.
- Variant C: rejected until the current lambda has a full32 result.
- Variant D: rejected because full32 evidence is required before larger-scale training.

## Selected Variant

- Selected: Variant A.
- Rejected: B, C, and D.
- Why this is the smallest useful test: it preserves the tiny-gate recipe and scales only runtime/coverage.

## Selection Pressure

- Primary pressure: enabled full32 rollouts must remain legal with no new starved cases and no material rigid-spacing regression.
- Guard pressure: existing mapper v3 training/rollout/model tests must pass; summary JSON must validate.
- Runtime pressure: stop after matched 500-step training plus full32 rollout comparison.
- Kill pressure: if enabled full32 is behavior-identical and not better on any timing metric, treat improvement signal as weak even if safety passes.

## Research Question

Does the row-filtered time-shift distance auxiliary produce a useful full32 generated-output signal at 500 steps, or does it remain a numerically stable but behaviorally inert training term?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary loss ablation, matched checkpoint rollout A/B test, teacher-forced versus free-running validation.
- Relevant taxonomy bucket: training objective validation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering validation of the v3 target grammar training objective, not representation novelty.

## Minimal Change

Add matched baseline/enabled 500-step configs and run existing training, rollout, and comparator tooling. Do not change tokenizer, grammar masks, model code, loss code, or decode defaults.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_baseline.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_enabled.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_pairs.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_result_report.md`
- rollout summaries/reports under `artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_summary.json`
- `src/pulsefield_model/training/mapper_v3.py`
- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_tiny_training_gate.py`

## Dataset Slice

Training uses the fixed 32-song / 256-window cache-backed v3 slice:

- index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- mapper record cache: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`
- control teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

Rollout uses the 32 cases from `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`.

## Baseline / Comparator

Direct comparator:

- baseline config: `lambda_time_shift_distance=0.0`, `event_token_loss_weight=2.0`
- enabled config: `lambda_time_shift_distance=0.5`, `event_token_loss_weight=2.0`
- both train for 500 steps from the same control checkpoint and seed on the same fixed slice.

Historical context:

- older 500-step horizon checkpoint routed `MUTATE` on wide audit and used a different loss recipe, so it is not the direct comparator.

## Primary Metric

Route from the full32 rollout comparator:

- `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE` or stronger only if training and rollout checks pass.
- `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE` if enabled training or rollout checks fail.

## Secondary Metric

- enabled `loss/time_shift_distance` finite and positive
- token/total loss regression ratios
- full32 legality
- new starved case count
- mean dominant-spacing ratio delta
- mean F1@100ms delta
- mean second-window event-share delta
- mean event-count ratio delta
- count of cases where enabled differs from baseline

## Verify Command / Evaluation Procedure

1. Train baseline:
   `uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_baseline.yaml`
2. Train enabled:
   `uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_enabled.yaml`
3. Run 32 baseline/enabled real-audio rollout pairs with `max_tokens_per_window=512`, greedy decode, and full timepoint previews.
4. Run the existing time-shift tiny-gate comparator with `--rollout-pairs-json` pointing to the generated full32 pair manifest.
5. Write result report and summary.

## Guard Check

```bash
uv run --group dev pytest tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json >/tmp/full32_time_shift_distance_summary.valid.json
```

## Qualitative Check

Inspect worst-case previews for repeated fixed grids, second-window collapse, event-count explosions, and cases where enabled differs from baseline. A safety pass with zero behavioral differences should be interpreted as weak positive evidence only.

## Positive Signal

- both runs complete 500 steps
- enabled time-shift loss is finite and positive
- full32 enabled rollouts are all legal
- no new starved cases
- mean dominant-spacing ratio delta <= `0.02`
- at least some cases show nonzero rollout metric deltas without harming guard metrics

## Negative Signal

- enabled training has non-finite loss
- enabled full32 rollouts produce illegal/dead-end/max-token cases
- enabled introduces any new starved case
- mean dominant-spacing ratio worsens by more than `0.02`
- enabled and baseline are identical across full32, indicating weak behavioral leverage

## Kill Criteria

Kill or deprioritize the current time-shift distance auxiliary if it fails full32 safety checks or remains behaviorally inert despite a finite 500-step training signal.

## Expected Failure Modes

- The auxiliary remains too small relative to CE to affect greedy decode.
- The expected-value distance target smooths over multimodal timing buckets.
- The event-token CE weighting recipe still dominates over timing calibration.
- Full32 rollouts expose rigid/starved regressions not visible in the 3-case gate.

## Confounders

- Full32 is still not full 4k.
- Greedy decode may hide teacher-forced calibration gains.
- Matched configs include event-token CE weight `2.0`, which had mixed prior evidence.
- Same-seed tiny/full32 runs may produce identical checkpoints if the auxiliary gradients stay too small.

## Expected Runtime / Runtime Budget

Expected runtime is several hours for two 500-step trainings plus 64 real-audio rollouts. Stop on non-finite training, missing checkpoints, or widespread rollout legality failure.

## Result Interpretation Plan

- Positive result would suggest: test a broader repaired-loss v3 mapper/planner gate.
- Negative result would suggest: mutate or kill this auxiliary and focus on other v3 grammar/training repairs.
- Ambiguous result would require: isolate whether the objective is too weak, too smooth, or hidden by greedy decode.
- Human owner decides: whether to keep the auxiliary in the v3 path.
- Next-loop action if positive: `TEST` broader v3 pipeline gate.
- Next-loop action if negative: `MUTATE` or `KILL` the auxiliary.
- Next-loop action if ambiguous: `MUTATE` into lambda/scale or diagnostic card.

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
- Closed loop complete: no
- Remaining ambiguity: full32 execution and result interpretation are pending.

## Next-Loop Action

- If positive: `TEST` broader v3 full-pipeline gate.
- If negative: `MUTATE`/`KILL` this auxiliary and consider v2.1 grammar improvement or other v3 objectives.
- If ambiguous: `MUTATE` to lambda/scale or decode-sensitivity diagnostics.

## Novelty Notes

- Closest analogies: auxiliary loss ablation and matched rollout safety gates.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering validation only.
