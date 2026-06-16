# Target Grammar v3 Real-Config Cache-Backed Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 has passed representation, smoke, tiny training, and broader tiny paired-training gates; the next hardening step should verify the full-size mapper configuration and cached control-teacher path before any longer replacement run.
- Acceptance source, if any: active C3/v3 goal requires the new target grammar to become usable in the full mapper pipeline from training to inference after full audit and hardening.
- Source snapshot / evidence grade: strong representation evidence, moderate tiny-training evidence, weak real-config training evidence.

## Hypothesis

On a deterministic multi-mapset slice of eligible mapper windows, v3 can run through the same cache-backed real d384/l4 mapper training stack as v2.1 with finite metrics, expected token-contract reports, and lower target valid-token count.

## Root Objective

Harden v3 from tiny comparison evidence toward the production mapper path by verifying:

- real d384/l4 mapper model config,
- real d384/l3 frozen control config,
- existing control-teacher cache loading,
- MPS training path,
- matched v2.1/v3 report comparability.

## Goal Decomposition

- Subgoal 1: Build a deterministic slice containing mapper-eligible 8s write windows, not merely 2s control rows.
- Subgoal 2: Run v2.1 and v3 with matched real configs, cache-backed control teacher tensors, and comparable split/step settings.
- Subgoal 3: Compare report contracts, completed steps, finite losses, eval valid tokens, and dataset/cache metadata.
- Subgoal 4: Preserve existing v2.1/v3 unit and inference guards.

## Candidate Variants

- Variant A: 32 beatmaps x 8 eligible windows, d384/l4 configs, existing d384 control-teacher cache, 3 MPS optimizer steps, small eval cap.
- Variant B: More tiny CPU training on a larger eligible-window slice.
- Variant C: Full d384 training on the complete local dataset.
- Variant D: Train v3 only, then compare to the existing 44k v2.1 checkpoint report.

## Local Verification Matrix

- Variant A: Directly tests the missing real-config/cache/MPS path while keeping runtime bounded. Pass/fail is observable from two report JSONs and the comparison harness.
- Variant B: Cheaper, but mostly repeats tiny evidence and does not verify production-size cache-backed mechanics.
- Variant C: Closest to replacement, but too slow for a fail-fast hardening gate.
- Variant D: Useful later, but unfair for first real-config comparison because v3 would be from scratch while v2.1 is trained.

## Selected Variant

- Selected: Variant A.

## Selection Pressure

Variant A is selected because it targets the strongest remaining infrastructure uncertainty before longer training: whether v3 behaves under the real mapper/control dimensions and cache contract. It keeps the dataset large enough to avoid the prior 16-window weakness while keeping runtime bounded.

## Minimal Change

No production code change.

Run existing training entrypoints and comparison harness with a generated deterministic index slice:

- first 32 source-order beatmaps with at least 8 mapper-eligible starts,
- first 8 eligible starts per selected beatmap,
- `256` source rows and expected `256` mapper-eligible windows before train/eval split.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_real_config_cache_backed_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_real_config_cache_backed_comparison_summary.json`
- temporary ignored outputs under `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/`

Read-only context:

- `configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml`
- `configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml`
- `src/pulsefield_model/training/mapper_v2_1.py`
- `src/pulsefield_model/training/mapper_v3.py`
- `src/pulsefield_model/evals/mapper_v3_training_comparison.py`

## Dataset Slice

- Base index: `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Selection: source-order deterministic `32` beatmaps x `8` eligible 8s starts.
- Expected rows: `256`
- Expected unique beatmaps: `32`
- Required cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Required cache hit gate: `256/256`

## Baseline / Comparator

- Comparator: v2.1 sparse lane-action mapper using the same real config shape and same selected slice, from scratch for this short run.
- Existing evidence:
  - full v3 dataset audit reconstruction mismatches: `0`,
  - full v3 dataset token reduction: `20.54%`,
  - broader tiny comparison v3 valid-token reduction: `23.16%`.

## Primary Metric

- v2.1 and v3 both complete `3` optimizer steps with finite final eval loss under the real d384/l4 cache-backed path.

## Secondary Metric

- v3 final eval valid-token ratio versus v2.1.
- Expected positive direction: ratio `< 1.0`.
- Dataset/cache metadata:
  - expected selected rows: `256`,
  - expected eligible source windows: `256`,
  - expected train/eval split: non-empty,
  - expected cache hits in sampled dataset path.

## Verify Command or Evaluation Procedure

1. Generate selected index under `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/`.
2. Run `run_mapper_v2_1_phase_b_training(...)` and `run_mapper_v3_phase_b_training(...)` with:
   - `max_steps=3`,
   - `eval_every=3`,
   - `save_every=3`,
   - `batch_size=2`,
   - `device_name=mps`,
   - `eval_fraction=0.25`,
   - `eval_size=32`,
   - `final_train_eval_size=16`,
   - real d384/l4 mapper config,
   - real d384/l3 control config,
   - `require_control_teacher_cache=True`,
   - existing d384 control-teacher cache.
3. Run `pulsefield_model.evals.mapper_v3_training_comparison` with `--min-completed-steps 3`.

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Qualitative Check

Inspect the generated reports for:

- correct mapper token contracts,
- finite loss metrics,
- non-zero valid-token counts,
- real d384 model/control configs,
- `require_control_teacher_cache=True`,
- no fallback to tiny control dimensions.

## Positive Signal

- Both real-config runs complete 3 steps.
- Both reports use the expected token contracts.
- Both final eval losses are finite.
- v3 has lower eval valid-token count than v2.1.
- Guard tests pass.

## Negative Signal

- MPS/cache path fails,
- missing teacher-cache entries,
- v3 real config shape mismatch,
- v3 valid-token count is not lower on the matched slice,
- guard tests fail.

## Kill Criteria

Kill immediate longer real-config v3 training if v3 cannot complete this cache-backed real-config gate without weakening legality, cache, or model-shape checks.

## Expected Failure Modes

- Selected slice may contain records without cache entries.
- MPS may hit unsupported operation or memory pressure.
- Real d384 shape may expose a v3-only model/loss bug not covered by tiny tests.
- Short 3-step losses may be noisy and should not be interpreted as quality.

## Expected Runtime / Runtime Budget

- Expected runtime: 10-30 minutes on local MPS.
- Stop condition: any hard failure in index/cache gate, training run, comparison harness, or guards.

## Confounders

- Three steps is an infrastructure and comparability gate, not convergence.
- v2.1 and v3 vocabularies have different token semantics, so per-token loss is not a quality ranking.
- Eval valid-token reduction is target-length evidence, not generated chart-quality evidence.

## Result Interpretation Plan

- Positive: proceed to a longer cache-backed v3 real-config training run or trained full-song inference/session-runtime route.
- Negative due to infrastructure: repair the failing integration layer and rerun this card.
- Negative due to token metric: re-check v3 tokenizer coverage on this slice before longer training.
- Ambiguous due to runtime/noise: repeat with fewer eval windows or CPU tiny fallback only to isolate the failure.

## Result Log Template

- Experiment: Target grammar v3 real-config cache-backed comparison
- Date:
- Commit / run id:
- Selected rows / beatmaps:
- Cache hit count:
- v2.1 report:
- v3 report:
- Completed steps:
- Eval losses:
- Eval valid-token counts:
- Token ratio / reduction:
- Guards:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run a longer real-config v3 training/inference comparison or begin the trained v3 full-song inference gate.
- If negative: repair the specific cache/model/training incompatibility before any longer v3 run.

## Closest Analogies and Novelty Layer

- Closest analogies: v2.1 sparse mapper training, teacher-forced event-token sequence models, cache-backed frozen-control conditioning.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering validation of an already audited representation, not a novelty claim.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: this does not prove trained mapper quality or replacement readiness.
