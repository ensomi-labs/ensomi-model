# Target Grammar v3 Event-Budget Training Gate Experiment Card

## Hypothesis

The v3 500-step baseline fails the fixed 32-case rollout gate mainly through event-count and first/second-window continuity errors. The previous scalar density mutation (`lambda_density=0.20`) regressed legality and event distribution because it pushed a framewise lane-onset proxy too broadly. A modest explicit event-budget loss (`lambda_event_budget=0.05`) should reduce second-window starvation and hard-chart undergeneration while preserving legal online v3 rollout.

## Root Objective

Move target grammar v3 toward full-pipeline replacement readiness while preserving the representation requirements already established by the v3 full-dataset audit: reversible beatmap-event reconstruction, lower-bit teacher-forcing targets than v2.1, no complex cross-window replay, and online inference without future target-derived inputs.

## Goal Decomposition

1. Keep the v3 tokenizer, grammar, model shape, runtime path, decode policy, training slice, control checkpoint, and teacher cache fixed.
2. Mutate only one training-side objective after the event-budget plumbing gate: enable `lambda_event_budget=0.05`.
3. Train a fresh 500-step checkpoint on the same fixed 32-song/256-window slice used by the baseline and density-calibration runs.
4. Rerun all 32 fixed-slice 16s real-audio rollouts with greedy decoding and the same max-token guard.
5. Compare against the committed 500-step baseline and the killed high-density run on legality, event count, second-window share, rigid-grid collapse, boundary events, and timing F1.

## Candidate Variants

### A. Modest Event-Budget Weight

Train fresh 500-step v3 with `lambda_density=0.05` and `lambda_event_budget=0.05`, then run the full 32-case fixed-slice rollout gate.

### B. Strong Event-Budget Weight

Train with `lambda_event_budget=0.20` to force count matching more aggressively.

### C. Decode-Time Event Budget Clamp

Keep the baseline checkpoint and enforce per-window event-count priors during rollout.

### D. Planner-Side Event Budget Target

Add a planner/control target that predicts event budget before mapper decoding.

## Local Verification Matrix

| Variant | Smallest Check | Pass Condition | Fail Condition |
| --- | --- | --- | --- |
| A | 500-step train plus 32-case rollout | Starved cases drop without legality, boundary, or overgeneration regression | Starvation persists or legality/tail metrics regress |
| B | Same gate with larger weight | Larger improvement than A | Repeats density-style overpressure and overgeneration |
| C | Rollout-only sweep | Count improves without timing artifacts | Masks training weakness or needs hand-coded future-dependent behavior |
| D | Target-design card | Planner signal can be computed online and improves mapper conditioning | Too broad before local mapper objective is tested |

## Selected Variant

Variant A: modest event-budget weight.

## Selection Pressure

Variant A is the smallest valid continuation after the event-budget loss plumbing gate. It directly targets the diagnosed failure shape, keeps the previous density loss at the baseline value, and avoids the high-pressure path that already failed for scalar density. Variant B is deliberately rejected for this loop because the density run showed that stronger event-distribution pressure can break legality and overgenerate. Variants C and D are deferred until the mapper-side local objective is tested in free-running rollout.

## Minimal Change

- No tokenizer, grammar, model architecture, runtime, or decode-policy source changes.
- Generate a temporary training config with `lambda_event_budget=0.05`.
- Train a fresh 500-step v3 checkpoint using the existing fixed-slice cache and control teacher cache.
- Run all 32 fixed-slice 16s real-audio rollouts with greedy decoding, max `512` tokens per window, and logit diagnostics.
- Aggregate the same fixed-slice metrics used by the 500-step baseline and event-distribution calibration reports.

## Files Likely To Change

Report artifacts:

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json`

Temporary runtime artifacts:

- `artifacts/tmp/mapper_v3_event_budget_training_gate/`

No source files should change for the selected variant.

## Dataset Slice

Training:

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Mapper record cache: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Training split settings: `eval_fraction=0.25`, `eval_size=32`, `final_train_eval_size=16`
- Training horizon: `500` steps

Rollout:

- Manifest: `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- Cases: all 32 unique fixed-slice beatmaps
- Prefix: `0-16000ms`
- Decode: greedy `temperature=0.0`, no time-shift penalties
- Max tokens per window: `512`

## Baseline / Comparator

Primary baseline:

- `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `all_legal=true`
- second-window starved cases: `11`
- rigid cases: `7`
- median event-count ratio: `1.000`
- mean F1 @100ms: `0.640`
- max boundary-event ratio: `0.200`

Negative comparator:

- `target_grammar_v3_event_distribution_calibration_summary.json`
- `lambda_density=0.20`
- route `MUTATE`
- `all_legal=false`
- second-window starved cases: `17`
- overgeneration cases: `5`
- median second-window share: `0.045`

## Primary Metric

Pass the fixed-slice event-distribution gate:

- all 32 rollouts legal;
- no rollout hits max tokens;
- second-window starved cases less than baseline `11`;
- rigid cases no worse than baseline `7`;
- median event-count ratio remains in `[0.80, 1.25]`;
- max boundary-event ratio no worse than baseline `0.200`.

## Secondary Metric

- Mean and median timing F1 @100ms.
- Mean and median second-window event share.
- Mean and median dominant-spacing ratio.
- Under/overgeneration case counts.
- Worst cases by starvation, rigid ratio, event-count ratio, boundary ratio, and low F1.
- Teacher-forced final eval `loss/event_budget` is finite and reported.

## Verify Command Or Evaluation Procedure

1. Generate a temporary YAML config for the selected variant.
2. Run 500-step v3 training with the fixed slice and existing teacher cache.
3. Run runtime-backed v3 rollouts for all manifest cases.
4. Aggregate generated timepoints against canonical reference events parsed from each `.osu`.
5. Validate the summary JSON.
6. Run focused regression guards:

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
```

## Guard Check

- The run must not change source code.
- The run must not rely on future target tokens during inference.
- The summary must preserve the manifest case count of `32`.
- Any max-token or dead-end case is a failure, even if aggregate timing metrics improve.

## Qualitative Check

Inspect the first and last generated/reference timing previews for the worst starvation and worst overgeneration cases. Treat improvement as weak if aggregate metrics improve only by emitting boundary duplicates or dense fixed grids.

## Positive Signal

`TEST_NEXT` if the selected variant passes the primary gate and improves at least one continuity metric, especially lower starvation or higher median second-window share, without worsening legality or event-count tails.

## Negative Signal

`MUTATE` if event-budget training is legal but does not improve starvation, increases rigid-grid collapse, increases boundary duplicates, or creates over/undergeneration tails comparable to the killed density run.

## Kill Criteria

Kill or downweight this objective if the 32-case rollout has any dead-end, max-token failure, non-finite event-budget loss, starved cases at least baseline `11`, or overgeneration/boundary artifacts worse than both the baseline and density-comparator envelope.

## Expected Failure Modes

- The event-budget objective improves teacher-forced metrics but does not affect greedy rollout.
- Count matching raises event mass but still places events on rigid 150/160ms grids.
- The per-half budget is too coarse and shifts errors into boundary duplicates near `8000ms`/`16000ms`.
- `lambda_event_budget=0.05` is too weak to move the model in 500 steps.

## Expected Runtime / Runtime Budget

Expected runtime is roughly one bounded 500-step mapper training run plus 32 real-audio runtime rollouts. Stop early if training fails to complete, if the checkpoint cannot load through `ModelRuntime`, or if the first few rollouts repeatedly hit max tokens or dead-end.

## Confounders

- The training slice is not held out; this is a calibration gate, not replacement evidence.
- The fixed 32-case manifest reuses known failing families and may overstate or understate full 4K behavior.
- Greedy decode may understate improvements that require sampling or calibrated decode penalties.
- Event-count F1 can be inflated by rigid grids hitting nearby reference events.

## Result Interpretation Plan

- If all primary gates pass, move to a wider v3 training/rollout audit using the same objective.
- If legality passes but starvation does not improve, mutate the objective toward ordered timing continuity or event-placement calibration rather than increasing raw weight.
- If event count improves but rigid grids worsen, separate count budget from timing grammar and test timing-specific regularization.
- If the run regresses like `lambda_density=0.20`, kill mapper-side scalar count pressure and move to planner-side budget conditioning.

## Result Log Template

```markdown
# Target Grammar v3 Event-Budget Training Gate Result Report

## Scope

## Training Result

## 32-Case Rollout Result

## Baseline Comparison

## Passed

## Surfaced

## Decision

## Next Step
```

## Next-Loop Action

If `TEST_NEXT`, train a wider v3 checkpoint with the selected objective and rerun the same fixed-slice plus held-out rollout gates. If `MUTATE`, use the surfaced failure cluster to choose either ordered timing continuity, a smaller/larger event-budget weight, or planner-side event-budget conditioning.

## Closest Analogies And Novelty Layer

Closest analogies are auxiliary count losses, CTC-style expected token-count regularizers, and sequence-level length/budget penalties in autoregressive models. This is engineering variation at the mapper training-objective layer, not representation novelty. The representation novelty claim remains with the already audited v3 event-group grammar, not this loss.
