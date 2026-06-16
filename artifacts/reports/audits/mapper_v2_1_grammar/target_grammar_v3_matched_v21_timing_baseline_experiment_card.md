# Target Grammar v3 Matched v2.1 Timing Baseline Experiment Card

## Hypothesis

The rigid-grid timing and second-window starvation seen in the 200-step v3 checkpoint may be a general short-training/free-running mapper failure rather than a v3 target-grammar-specific failure. A matched 200-step v2.1 baseline on the same fixed slice and six real-audio cases should clarify whether the next mutation should target v3 specifically or mapper training/decode calibration more generally.

## Root Objective

Harden the v3 replacement decision by comparing v3 free-running timing failures against a matched v2.1 baseline before scaling v3 training or changing v3 grammar/loss.

## Goal Decomposition

- Train a v2.1 checkpoint with the same model/control shape, fixed slice, cache-backed setup, learning rate, seed, and 200-step budget as the v3 gate.
- Run the same six real-audio timing cases used in the v3 multicase audit.
- Measure timing F1, rigid-grid ratio, event-count ratio, boundary ratio, and second-window share.
- Interpret whether v3 is uniquely worse, comparable to v2.1, or better on the current undertrained setting.

## Candidate Variants

- Variant A: train a matched 200-step v2.1 checkpoint and add a minimal v2.1 runtime rollout audit harness mirroring the v3 diagnostic.
- Variant B: use the existing 3-step v2.1 checkpoint from the real-config cache-backed comparison.
- Variant C: compare v3 against the long-running configured v2.1 checkpoint path if available.
- Variant D: skip v2.1 baseline and mutate v3 timing/loss immediately.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | 200-step v2.1 train + same six rollouts | Matched training budget and comparable timing metrics | Training/runtime harness fails or exceeds budget |
| B | Existing 3-step v2.1 rollout | Fast | Not comparable to v3 200-step checkpoint |
| C | Existing long checkpoint | Stronger quality baseline | Checkpoint may be unavailable or trained on different setup/resume state |
| D | v3 calibration now | Direct mutation | Risks solving a general mapper issue with v3-specific changes |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is the smallest fair baseline. It controls for model size, slice, control-teacher cache, training horizon, and real-audio cases. It requires modest code only because v2.1 rollout primitives and runtime loading already exist.

## Minimal Change

- Add a small v2.1 trained runtime rollout audit module mirroring the existing v3 eval shape.
- Train v2.1 for 200 MPS steps on the same fixed 32-song/256-window slice.
- Run the same three cases x two prefixes with selected-map normalized difficulty and alpha `0.00`.
- Write a result report and machine-readable summary.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v2_1_trained_runtime_rollout.py`
- `tests/evals/test_mapper_v2_1_trained_runtime_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_summary.json`
- This experiment card.

## Dataset Slice

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Real-audio cases: same three cases and two prefixes from `target_grammar_v3_multicase_timing_quality_result_report.md`.

## Baseline / Comparator

The v3 200-step multicase timing audit:

- All six rollouts legal.
- Mean timing F1 @100ms: `0.643`.
- Mean dominant-spacing ratio: `0.993`.
- Sparse/dense 16s second-window event share: `0.02`.
- Route: `MUTATE`.

## Primary Metric

Difference between matched v2.1 and v3 mean dominant-spacing ratio and second-window event share.

## Secondary Metric

- Timing F1 @100ms.
- Generated/reference event-count ratio.
- Boundary-event ratio.
- Completion/dead-end/max-token flags.
- Training eval token loss trend.

## Verify Command Or Evaluation Procedure

1. Train v2.1 for `200` MPS steps on the fixed slice.
2. Run v2.1 real-audio runtime rollout for the same six cases.
3. Aggregate generated timing metrics using the same reference-time matcher as the v3 multicase audit.
4. Compare v2.1 against the v3 summary.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v2_1_rollout.py tests/evals/test_mapper_v2_1_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q`

## Qualitative Check

Inspect whether v2.1 also collapses to rigid grids and window starvation. If v2.1 is materially better, v3-specific calibration is more urgent. If v2.1 is similarly bad, the failure is probably short-training/free-running mapper calibration.

## Positive Signal

- v2.1 shows the same rigid-grid/window-starvation pattern, supporting a general training/decode calibration path rather than v3-specific grammar changes.
- Or v2.1 is much better, giving a concrete quality target for v3.

## Negative Signal

- v2.1 runtime cannot complete the same real-audio path.
- v2.1 output is too sparse/dense to compare.
- Metrics are not computed consistently with the v3 multicase audit.

## Kill Criteria

If a matched v2.1 checkpoint cannot be trained or run through real-audio session inference, stop and repair v2.1 runtime comparability before making v3/v2.1 quality claims.

## Expected Failure Modes

- Training takes longer than expected.
- v2.1 output can dead-end due to sparse lane-action grammar constraints.
- v2.1 may need a different max-token cap because lane-action tokens are longer than v3 event tokens.

## Expected Runtime / Runtime Budget

Expected runtime is under 2 hours for training, six rollouts, aggregation, and guards. Stop on non-finite training metrics, missing checkpoint, or rollout legality failure.

## Confounders

- Both checkpoints are undertrained.
- v2.1 and v3 target token semantics differ.
- Timing F1 can be inflated by regular grids near dense references.
- Reference beatmap matching is a proxy, not a full chart quality metric.

## Result Interpretation Plan

- If v2.1 is also rigid-grid dominated, mutate toward general mapper timing/decode calibration.
- If v2.1 is much less rigid and has better second-window continuity, mutate v3 training/loss/decode specifically.
- If v2.1 cannot run legally, fix baseline comparability before further replacement decisions.

## Result Log Template

- Training metrics:
- Runtime legality:
- Matched timing table:
- v2.1 vs v3 aggregate comparison:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the matched baseline to choose between v3-specific calibration, shared mapper free-running calibration, or v2.1 grammar improvement.

## Closest Analogies And Novelty Layer

Closest analogies are controlled baseline comparison, teacher-forcing versus free-running diagnostics, and constrained autoregressive timing evaluation. There is no novelty claim; this is a verification baseline for the v3 target grammar decision.
