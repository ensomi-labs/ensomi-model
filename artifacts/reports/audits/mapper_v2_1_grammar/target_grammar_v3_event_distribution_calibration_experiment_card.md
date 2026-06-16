# Target Grammar v3 Event-Distribution Calibration Experiment Card

## Hypothesis

The current v3 500-step checkpoint is legal but event-distribution calibrated too weakly: it improves token loss and passes the six-case continuation gate, yet the 32-case fixed-slice gate still has second-window starvation, 160ms grid collapse, hard-chart undergeneration, easier-chart overgeneration, and one zero-reference boundary duplicate. A stronger density/event-distribution training weight can reduce starvation without breaking legality or increasing overproduction.

## Root Objective

Move v3 toward full-pipeline replacement readiness by testing whether a stronger event-distribution training signal improves free-running fixed-slice rollout quality before scaling to full 4K training.

## Goal Decomposition

1. Keep the v3 target grammar fixed: reversible event-group tokens, no cross-window replay, no future target dependency.
2. Keep the fixed 32-song/256-window slice, model shape, control checkpoint, teacher cache, decode policy, and 500-step horizon fixed.
3. Mutate only one training-side calibration knob: density auxiliary weight.
4. Compare against the existing `lambda_density=0.05` 500-step checkpoint on the same 32-case 16s fixed-slice gate.
5. Decide whether stronger event-distribution pressure is worth scaling, needs mutation, or should be killed.

## Candidate Variants

### A. Stronger Density Weight, Same Training Setup

Train a fresh 500-step v3 checkpoint with `lambda_density=0.20` on the same fixed slice, then run all 32 fixed-slice 16s real-audio rollouts with greedy alpha `0.00`.

### B. Add A New Count-Continuity Loss

Implement a new auxiliary objective that directly penalizes first/second-window count mismatch or expected onset mass imbalance.

### C. Decode-Time Event/Time-Shift Balancing

Add decode-time event pressure or time-shift penalties to force continuation.

### D. Larger Training Horizon Or Full-Slice Scaling

Train longer or on a larger slice before changing calibration.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | One 500-step high-density run plus 32-case gate | Starvation drops materially without legality/boundary/overgeneration regression | Starvation persists, or overproduction/boundary artifacts increase |
| B | Implement and unit-test new loss | Better matches failure shape than density loss | Adds new machinery before exhausting existing loss knobs |
| C | Rollout sweep only | Cheap to run | Prior decode-only sweep already failed or introduced artifacts |
| D | Longer/larger run | Could improve generalization | More expensive before testing the diagnosed calibration lever |

## Selected Variant

Variant A: stronger density weight, same training setup.

## Selection Pressure

Variant A is the smallest executable mutation after the density-loss effectiveness diagnostic. The current run already used `lambda_density=0.05`, but eval density stayed flat. A four-times stronger density weight tests whether the existing auxiliary path has usable leverage before adding a new count-continuity loss. It also keeps all comparators stable.

## Minimal Change

- No tokenizer, grammar, model, runtime, or decode-policy source changes.
- Train a fresh 500-step v3 checkpoint on the fixed slice with `loss.lambda_density=0.20`.
- Run all 32 fixed-slice 16s rollouts with greedy decoding and logit diagnostics.
- Aggregate the same wide-audit metrics and re-run failure-cluster labels.

## Files Likely To Change

Report artifacts:

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_distribution_calibration_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_distribution_calibration_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_distribution_calibration_summary.json`

Temporary runtime artifacts:

- `artifacts/tmp/mapper_v3_event_distribution_calibration/`

No source files should change for Variant A.

## Dataset Slice

Training:

- Fixed index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Eval split: same fixed-slice settings as the 500-step horizon gate: `eval_fraction=0.25`, `eval_size=32`, `final_train_eval_size=16`.

Rollout:

- Manifest: `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- All 32 unique fixed-slice beatmaps.
- Prefix: `16000ms`.

## Baseline / Comparator

Existing v3 500-step fixed-slice wide audit:

- checkpoint: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`
- `lambda_density=0.05`
- all legal: `true`
- mean F1 @100ms: `0.640`
- median F1 @100ms: `0.679`
- mean event-count ratio: `0.971`
- median event-count ratio: `1.000`
- mean dominant-spacing ratio: `0.749`
- starved cases: `11`
- rigid cases: `7`
- max boundary-event ratio: `0.200`
- max duplicate/nonincreasing spacing ratio: `0.750`
- max event-count ratio: `2.158`

Failure-cluster baseline:

- second-window starvation: `11`
- starved-160-grid: `10`
- undergeneration: `9`
- overgeneration: `2`
- zero-reference boundary duplicate: `1`

## Primary Metric

Reduction in second-window starvation count on the same 32-case fixed-slice 16s gate.

## Secondary Metrics

- Starved-160-grid count.
- Undergeneration and overgeneration counts.
- Mean and median F1 @100ms.
- Mean and median event-count ratio.
- Mean and median dominant-spacing ratio.
- Rigid-case count.
- Max boundary-event ratio.
- Max duplicate/nonincreasing spacing ratio.
- Event top-1/top-k logit diagnostics.
- Eval token-loss and density-loss trends.

## Verify Command Or Evaluation Procedure

1. Train fresh v3 checkpoint with `lambda_density=0.20`, same fixed-slice setup, `500` steps.
2. Run all 32 fixed-slice 16s rollouts using the new checkpoint and the existing manifest.
3. Aggregate timing/count/rigidity/boundary/logit metrics.
4. Reuse the failure-cluster labeling rules from the prior diagnostic.
5. Compare against the committed `lambda_density=0.05` wide audit and failure clusters.
6. Validate the summary JSON with `python3 -m json.tool`.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_model.py -q`
- All 32 rollouts must be legal: no dead-end and no max-token failure.

## Qualitative Check

Inspect representative cases:

- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`
- `23_usao_knight_rider_kuo_kyoka_expert`
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`
- `30_goreshit_one_way_to_hannover_cokiiplay_autophobia`

The desired qualitative change is continuation into the second 8s window without merely adding dense regular grids or boundary duplicates.

## Positive Signal

The high-density checkpoint passes all of:

- all 32 rollouts legal;
- second-window starvation count drops from `11` to at most `6`;
- starved-160-grid count drops from `10` to at most `5`;
- overgeneration count stays at or below `2`;
- max boundary-event ratio stays at or below `0.200`;
- max duplicate/nonincreasing spacing ratio stays below `0.750`;
- mean F1 does not fall below `0.590`.

## Negative Signal

- Starvation remains above `8` cases.
- Starvation improves only by overproducing regular grids.
- Overgeneration increases above `2` cases.
- Boundary or duplicate artifacts worsen.
- Legality regresses.
- Eval density loss improves but free-running cluster counts do not.

## Kill Criteria

Kill the simple density-weight path if the positive signal fails. Mutate toward a more explicit event-count/window-continuity objective or revisit v2.1 grammar work rather than scaling the same high-density setting.

## Expected Failure Modes

- Larger density weight may encourage overproduction.
- Density target may be too coarse to fix timing placement.
- Fixed-slice training may overfit same-audio grid patterns.
- Greedy rollout may still collapse even if teacher-forced density improves.
- Zero-reference intros need a separate suppression rule.

## Expected Runtime / Runtime Budget

Expected runtime: 45-90 minutes for training, 32 rollouts, aggregation, and guards. Stop on non-finite training metrics, missing checkpoint, rollout dead-end/max-token failure, or unavailable MPS/torch runtime.

## Confounders

- Fresh 500-step run is not an exact continuation of the baseline checkpoint.
- Fixed-slice evidence is not held-out or full 4K evidence.
- F1 @100ms can reward regular grids.
- Density loss is teacher-forced while the gate is free-running.
- Same-audio families can dominate aggregate behavior.

## Result Interpretation Plan

- If positive, run a broader v3 training/rollout audit before replacement.
- If starvation drops but overgeneration rises, mutate to count-balanced density or difficulty-conditioned density.
- If density loss improves but rollout clusters do not, implement a direct event-count/window-continuity objective.
- If legality regresses, repair training/runtime stability before further scaling.
- If no calibration path helps, return to v2.1 grammar improvement while preserving v3 as a reversible target candidate.

## Result Log Template

- Training command:
- Training metrics:
- Rollout command:
- Aggregate 32-case metrics:
- Cluster comparison:
- Representative cases:
- Guard results:
- Decision:
- Next-loop action:

## Next-Loop Action

Use the result to decide whether v3 should proceed to a larger calibrated training run, mutate to explicit event-count continuity, or pause in favor of v2.1 grammar improvements.

## Closest Analogies And Novelty Layer

Closest analogies: auxiliary-loss weight sweep, density/count calibration for sequence generation, teacher-forced loss versus free-running rollout validation, exposure-bias triage.

Novelty layer: none claimed. This is an engineering calibration experiment for the v3 replacement path.
