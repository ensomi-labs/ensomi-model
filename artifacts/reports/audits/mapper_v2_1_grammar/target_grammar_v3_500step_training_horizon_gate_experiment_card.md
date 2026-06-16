# Target Grammar v3 500-Step Training Horizon Gate Experiment Card

## Hypothesis

The 200-step v3 checkpoint may still be undertrained: zero-event collapse was fixed between 20 and 200 steps, and the remaining rigid-grid/second-window failures may improve with a modest additional training horizon before changing loss or grammar. A 500-step from-scratch run on the same fixed slice should show whether training horizon alone is still a productive next lever.

## Root Objective

Move target grammar v3 toward full-pipeline replacement by determining whether the current blocker is still mostly training horizon, or whether new loss/training machinery is required.

## Goal Decomposition

- Keep the v3 representation fixed: reversible event-group tokens, no tokenizer/grammar changes.
- Keep the dataset slice, control checkpoint, teacher cache, model shape, and decode policy fixed.
- Compare 500-step v3 free-running behavior against the committed 200-step v3 multicase baseline and matched v2.1 baseline.
- Decide whether the next loop should scale v3 training, mutate v3 loss/training, or deprioritize v3 in favor of v2.1 grammar work.

## Candidate Variants

- Variant A: train a fresh 500-step v3 checkpoint on the same fixed 32-song/256-window slice and run the same multicase real-audio timing audit.
- Variant B: implement v3 resume support and continue the 200-step checkpoint to 500 steps.
- Variant C: add token-class/event-time loss reweighting before another training run.
- Variant D: jump to a larger dataset slice or full 4K training immediately.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | Fresh 500-step run plus same multicase rollouts | Lower rigid-grid ratio and restored sparse/dense second-window continuity | Same rigid grid/starvation despite lower token loss |
| B | Resume 200-step to 500-step | Isolates incremental horizon | Resume is explicitly unimplemented for mapper v3 |
| C | Loss-code mutation | Could target event/time-shift balance directly | Changes more variables before proving horizon is exhausted |
| D | Larger slice | More representative | Too expensive before small-slice horizon is falsified |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is the smallest executable training-side test. It avoids unimplemented resume support and avoids adding new loss code before proving that horizon alone has diminishing returns. It keeps all known comparators stable.

## Minimal Change

- No code changes to tokenizer, grammar, model, loss, or runtime.
- Train v3 from control initialization for `500` MPS steps on the same fixed slice.
- Run greedy alpha `0.00` real-audio rollouts for the same sparse/moderate/dense cases and both `8000`/`16000` ms prefixes.
- Aggregate the same timing-quality metrics as the v3 multicase and matched v2.1 audits.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_training_horizon_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_training_horizon_gate_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_training_horizon_gate_summary.json`

## Dataset Slice

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Train/eval split: deterministic split with `eval_fraction=0.25`, `eval_size=32`, `final_train_eval_size=16`.
- Real-audio cases: same three cases and two prefixes from `artifacts/tmp/mapper_v3_multicase_timing_quality/manifest.json`.

## Baseline / Comparator

- v3 200-step multicase audit: legal, mean F1 @100ms `0.643`, mean dominant-spacing ratio `0.993`, sparse/dense 16s second-window share `0.02`, route `MUTATE`.
- matched v2.1 200-step baseline: legal, mean F1 @100ms `0.745`, mean dominant-spacing ratio `1.000`, no sparse/dense 16s starvation, route `MUTATE`.
- v3 decode-only sweep: no tested decode policy passed; route `KILL` for decode-only calibration.

## Primary Metric

Change from v3 200-step to v3 500-step on:

- mean dominant-spacing ratio across the six rollouts,
- sparse/dense 16s second-window event share.

## Secondary Metric

- Timing F1 @100ms.
- Generated/reference event-count ratio.
- Boundary-event ratio.
- Completion/dead-end/max-token flags.
- Training eval token loss trend.
- Event top-1/top-k logit diagnostics.

## Verify Command Or Evaluation Procedure

1. Train a fresh v3 checkpoint for `500` steps on the fixed slice.
2. Run the same six real-audio greedy rollouts with selected-map normalized difficulties.
3. Aggregate metrics with the same matcher used by the previous multicase audits.
4. Compare against the v3 200-step and matched v2.1 summaries.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_model.py -q`
- Validate the result summary JSON with `uv run python -m json.tool`.

## Qualitative Check

Inspect whether the generated times move away from fixed `160/200ms` grids and whether sparse/dense 16s cases continue through the second window without boundary-only artifacts.

## Positive Signal

The 500-step checkpoint:

- completes all six rollouts legally,
- reduces mean dominant-spacing ratio below `0.90`,
- raises sparse and dense 16s second-window share to at least `0.25`,
- keeps max boundary-event ratio at or below `0.05`,
- and does not reduce mean F1 @100ms by more than `0.05` versus the 200-step v3 baseline.

## Negative Signal

- Eval token loss improves but rigid-grid ratio stays above `0.95`.
- Sparse/dense 16s second-window share remains below `0.10`.
- More training causes max-token, dead-end, or boundary-duplication failures.
- Event count increases mainly by producing denser regular grids.

## Kill Criteria

If the 500-step run lowers loss but fails the positive signal, treat horizon-only training as insufficient for the next v3 replacement step. Mutate to a bounded loss/training calibration card rather than scaling the same setup.

## Expected Failure Modes

- Fresh 500-step run may not be directly step-continuous with the 200-step checkpoint because mapper v3 resume is not implemented.
- Loss may continue improving while free-running quality remains grid-prior dominated.
- Greedy generation may remain exposure-biased even if teacher-forced token loss improves.
- The fixed slice may be too small to learn robust timing.

## Expected Runtime / Runtime Budget

Expected runtime is under 45 minutes for training, six rollouts, aggregation, and guards. Stop on non-finite training metrics, missing checkpoint, or rollout legality failure.

## Confounders

- This is still a small fixed-slice diagnostic, not a full 4K dataset audit.
- v3 and v2.1 token losses are not directly comparable.
- Timing F1 can be inflated by regular grids.
- A fresh 500-step run tests horizon from initialization, not exact continuation from the committed 200-step checkpoint.

## Result Interpretation Plan

- If positive, run a wider v3 multicase/full-slice audit before replacement.
- If loss improves but quality gates fail, mutate toward loss/training calibration.
- If legality regresses, repair runtime/training stability before scaling.
- If v3 still fails while v2.1 remains more stable on continuation, consider parallel v2.1 grammar improvement.

## Result Log Template

- Training metrics:
- Runtime legality:
- Matched timing table:
- 500-step vs 200-step comparison:
- 500-step vs matched v2.1 comparison:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the result to choose between wider v3 training, v3 loss/training calibration, or renewed v2.1 grammar work.

## Closest Analogies And Novelty Layer

Closest analogies are teacher-forcing versus free-running diagnostics, autoregressive exposure-bias checks, and controlled training-horizon ablations. There is no novelty claim; this is evidence hardening for the v3 event-group target representation.
