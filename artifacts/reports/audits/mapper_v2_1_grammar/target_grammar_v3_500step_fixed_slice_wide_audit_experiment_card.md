# Target Grammar v3 500-Step Fixed-Slice Wide Audit Experiment Card

## Hypothesis

The 500-step v3 checkpoint passed the six-case horizon gate, but that may be an artifact of the handpicked sparse/moderate/dense examples. Auditing all 32 unique beatmaps from the fixed slice at a 16s prefix should determine whether the 500-step improvement generalizes enough to justify a larger v3 audit before adding loss machinery.

## Root Objective

Harden the target grammar v3 replacement path by testing whether the 500-step checkpoint's legal continuation and reduced grid collapse hold across the full fixed training-slice case universe.

## Goal Decomposition

- Keep v3 representation, checkpoint, decode policy, runtime, and selected-map difficulties fixed.
- Expand from 3 selected maps to all 32 unique fixed-slice beatmaps.
- Measure legality, second-window starvation, rigid-grid dominance, boundary artifacts, event-count ratio, and F1.
- Decide whether the next loop should widen v3 evaluation further, scale training, or mutate training/loss.

## Candidate Variants

- Variant A: run all 32 unique fixed-slice beatmaps at `16000ms` with greedy alpha `0.00`.
- Variant B: run a 12-map stratified subset by difficulty and audio family.
- Variant C: run the full 4K dataset audit immediately.
- Variant D: skip wider audit and move directly to loss/training calibration.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | 32 unique fixed-slice 16s rollouts | Improvement holds across the complete small-slice universe | Many legal/timing regressions outside the six selected maps |
| B | 12 stratified rollouts | Faster signal | Leaves fixed-slice generalization under-verified |
| C | Full 4K audit | Direct replacement evidence | Too expensive before fixed-slice generalization is checked |
| D | Loss/training mutation | Directly targets known artifacts | Premature if 500-step horizon generalizes |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is exhaustive for the current fixed training slice while staying bounded. It is the smallest audit that can refute the six-case result as selection bias.

## Minimal Change

- No model, tokenizer, grammar, loss, decode-policy, or training-code changes.
- Build a manifest from all unique `(shard, audio_path, beatmap_path, difficulty)` rows in the fixed 32-song/256-window index.
- Run the existing v3 runtime rollout evaluator for each case at `16000ms`, greedy `temperature=0.0`, alpha `0.00`, selected-map normalized difficulty.
- Aggregate the same timing-quality metrics used by prior v3 audits.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`

## Dataset Slice

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Unique maps: all 32 unique beatmaps in that index.
- Audio root: `dataset/{shard}/{audio_path}`.
- Beatmap root: `dataset/{shard}/{beatmap_path}`.
- Prefix: `0-16000ms`.
- Difficulty: `normalize_difficulty(difficulty)` from the selected map row.

## Baseline / Comparator

- v3 500-step six-case horizon gate: all legal, mean F1 `0.762`, mean dominant-spacing ratio `0.733`, sparse/dense 16s min second-window share `0.353`, route `TEST_NEXT`.
- v3 200-step multicase audit: legal but rigid/starved, route `MUTATE`.
- v3 decode-only sweep: no policy passed, route `KILL`.

## Primary Metric

Fixed-slice generalization gate:

- all rollouts legal,
- second-window starved case count,
- rigid-case count where dominant-spacing ratio is at least `0.95`.

## Secondary Metric

- Mean and median timing F1 @100ms.
- Mean/median generated/reference event-count ratio.
- Mean/median dominant-spacing ratio.
- Boundary-event ratio.
- Duplicate/non-increasing spacing ratio.
- Metrics by difficulty band.
- Worst-case timing previews.

## Verify Command Or Evaluation Procedure

1. Generate the 32-case manifest from the fixed index.
2. Run the 500-step checkpoint through runtime-backed v3 rollout for all 32 cases.
3. Aggregate metrics against selected reference beatmaps.
4. Compare aggregate and tail metrics against the six-case gate.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_model.py -q`
- Validate the result summary JSON with `uv run python -m json.tool`.

## Qualitative Check

Inspect worst cases by low F1, high rigid ratio, starvation, boundary artifacts, and event-count ratio. If pass/fail is driven by one map family or difficulty band, record that explicitly.

## Positive Signal

The 32-case audit:

- has all rollouts legal,
- has second-window starved case count at most `3`,
- has rigid-case count at most `8`,
- has max boundary-event ratio at most `0.05`,
- has median event-count ratio in `[0.50, 2.00]`,
- and has mean F1 @100ms at least `0.65`.

## Negative Signal

- More than `3` cases starve the second window.
- More than `8` cases remain rigid-grid dominated.
- Legal completion fails on any case.
- Median event-count ratio exceeds `2.00`, indicating broad overproduction.
- Worst cases cluster by difficulty or audio family, showing the six-case gate did not generalize.

## Kill Criteria

If the wide audit fails the positive signal, do not run a full 4K replacement audit yet. Mutate to loss/training calibration or case-family diagnosis using the failed fixed-slice clusters.

## Expected Failure Modes

- Some short or sparse maps may naturally have low second-window event share.
- Dense regular grids can inflate F1.
- Shared audio families may make the 32-map audit less independent than 32 unrelated songs.
- Real-audio runtime cost may be higher than the six-case gate.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes for 32 real-audio 16s rollouts, aggregation, and guards. Stop on missing checkpoint, repeated runtime failure, or more than one hour of rollout runtime.

## Confounders

- The cases are from the training slice, so this is not held-out generalization.
- The checkpoint is still only 500 steps.
- Full 4K dataset replacement requirements remain unproven.
- Timing F1 is a proxy, not chart-quality proof.

## Result Interpretation Plan

- If positive, move to a held-out or larger fixed-slice v3 audit before changing loss.
- If mixed, cluster failures and define a targeted calibration card.
- If negative, treat the six-case pass as insufficient and prioritize loss/training calibration or v2.1 grammar improvement.

## Result Log Template

- Manifest summary:
- Runtime legality:
- Aggregate metrics:
- Difficulty-band metrics:
- Worst cases:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the result to choose between a larger v3 audit, targeted v3 calibration, or renewed v2.1 grammar work.

## Closest Analogies And Novelty Layer

Closest analogies are fixed-slice validation, tail-risk auditing, and teacher-forcing versus free-running diagnostics. There is no novelty claim; this is representation-readiness verification for target grammar v3.
