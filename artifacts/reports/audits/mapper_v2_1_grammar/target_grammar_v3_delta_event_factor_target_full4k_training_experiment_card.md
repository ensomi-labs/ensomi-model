# Target Grammar v3 Delta-Event Factor Target Full4k Fixed-Cache Training Experiment Card

## Hypothesis

The factorized delta-event target branch can run through the production v3 training runner on the full4k source index with the fixed cached control-teacher tensors: the full source/eligible window surface is available, cached control memory is hit during sampled training/eval batches, factor-target labels remain present, losses stay finite across multiple eval points, and aggregate loss accounting remains exact.

## Root Objective

Move v3 from fixed-slice factor-target stability toward a full-pipeline replacement path. The previous longer fixed-slice run proved short-horizon stability on a 32-song cache-backed slice. This card tests the next boundary: full4k index and full fixed teacher-cache compatibility under a bounded real-runner training loop.

## Goal Decomposition

- Subgoal 1: Instantiate the production v3 training dataset over the full4k index.
- Subgoal 2: Reuse the fixed control-teacher cache without missing sampled cache entries.
- Subgoal 3: Run the production v3 training runner with factor-target branch and loss enabled.
- Subgoal 4: Capture at least three eval points with finite token and factor-target losses.
- Subgoal 5: Verify positive factor-label counts at final eval.
- Subgoal 6: Verify final eval factor loss is not exploding relative to the first eval.
- Subgoal 7: Preserve no-rollout, no-inference, no-C3, no-future-lookup, and no-default-change guards.

## Candidate Variants

- Variant A: Full4k index, fixed control-teacher cache, bounded CPU run, small factor-target-enabled v3 model, `8` steps, evals at step `1`, `4`, and `8`.
- Variant B: Full production v3 config on full4k with global context and MPS settings.
- Variant C: Cache-coverage preflight only, no training.
- Variant D: Repeat the fixed 32-song longer training gate with more steps.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Full4k source index is used; eligible-window count is positive and comparable to prior full4k audits; report/checkpoint written; >=3 eval points; finite losses; positive factor labels; final/first factor-loss ratio <= threshold | Dataset/cache miss, empty split, missing factor metrics, non-finite loss, exploding factor loss, runner/report failure |
| B | Stronger production realism if it works | Too expensive and confounds full-index compatibility with production-scale training |
| C | Fast cache-readiness evidence | Does not test optimizer/loss/report behavior on the full4k path |
| D | More optimizer evidence on known slice | Adds no full4k index/cache evidence |

## Selected Variant

Variant A: bounded full4k/fixed-cache production-runner stability gate with factor-target loss enabled.

## Selection Pressure

Variant A is the smallest experiment that adds new evidence beyond the fixed-slice result. It exercises the full4k index, full fixed cache, production collate/model/loss/training/report path, and repeated evals while keeping runtime and failure diagnosis bounded. Variant B is deferred until this gate proves the full4k path is mechanically stable. Variant C is too weak for training readiness. Variant D repeats already-positive evidence.

## Minimal Change

- Add `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_full4k_training.py`.
- Reuse `run_mapper_v3_phase_b_training` with the full4k index and fixed control-teacher cache.
- Use a dedicated v3 mapper-record cache path for full4k eligible-window reuse.
- Parse the runner report for dataset/cache/stability metrics.
- Add focused tests for full4k route decisions and cache/plumbing failure routes.
- Record summary and markdown result report.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_full4k_training.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_full4k_training.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_full4k_training_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_full4k_training_result_report.md`

## Read-Only Context

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json`
- `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

## Dataset Slice

Use the full4k control-window index:

`artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`

The runner may split this full source through `eval_size=16`, `final_train_eval_size=16`, `batch_size=4`, `max_steps=8`, and `eval_every=4`. Training remains bounded, but source dataset construction and mapper-record eligibility are full4k.

## Baseline / Comparator

- Full4k bit proxy route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`.
- Full4k label coverage route: `TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD`.
- Fixed-slice longer training route: `TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD`.
- Fixed-slice longer training factor-loss ratio: `0.947168`.

## Primary Metric

Full4k fixed-cache training stability:

- `completed_steps == 8`
- source dataset row count is full4k scale and eligible mapper windows are positive
- at least `3` eval points
- every eval `loss/total`, `loss/token`, and `loss/delta_event_factor_target` is finite
- final eval factor-loss ratio versus first eval is `<= 1.20`
- report and checkpoint exist

## Secondary Metrics

- train/eval/final-train window counts
- mapper-record cache path in training report
- control-teacher cache directory and `require_control_teacher_cache`
- first/final eval factor loss
- final train factor loss
- final eval factor label counts
- final eval aggregate `loss/total` recomputation delta

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_full4k_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_full4k_training.py tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_full4k_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_full4k_training_summary.json >/dev/null
git diff --check
```

## Guard Check

- No inference or rollout changes.
- No C3 backreference or future lookup.
- No mapper default changes.
- No tokenizer changes.
- No dataset schema changes.
- Generated mapper-record caches, checkpoints, and run reports are not committed; only audit reports are committed.
- This remains bounded full4k training stability, not full production training or replacement readiness.

## Qualitative Check

The report must state that this proves only full4k fixed-cache short-horizon training compatibility. It must not claim generated-chart quality, full production training convergence, online factor-row inference, or final v3 replacement readiness.

## Positive Signal

The full4k real-runner gate completes with finite metrics at all eval points, positive factor labels, correct total-loss accounting, report/checkpoint artifacts, and final/first eval factor-loss ratio `<= 1.20`.

## Negative Signal

Cache misses, empty train/eval split, non-finite losses, missing factor metrics, missing eval history, exploding factor loss, incorrect aggregate loss accounting, or missing report/checkpoint.

## Kill Criteria

- Any sampled training/eval batch fails because cached control teacher tensors are missing.
- Any eval `loss/total` or `loss/delta_event_factor_target` is non-finite.
- Final/first eval factor-loss ratio exceeds `1.20`.
- Factor-target labels disappear from eval metrics.

## Expected Failure Modes

- Full4k mapper-record cache may need to be built on the first run, increasing runtime without invalidating the experiment.
- Sampled full4k windows may have noisier loss movement than the fixed slice.
- Cache coverage might match candidate-window count but still fail on stale keys if index metadata changed.
- Short full4k training can show local loss noise; pass means mechanical stability only, not quality or convergence.

## Expected Runtime / Runtime Budget

Expected runtime: cache-hit rerun under 5 minutes on CPU. First-time full4k mapper-record cache construction may take about 60-70 minutes on local CPU/disk and should be treated as setup cost for later full4k cards.

Stop condition: fail on dataset construction error, cache miss, runner crash, non-finite loss, missing metrics, missing report/checkpoint, or non-explosion-ratio failure.

## Confounders

- This is not a full training run.
- This uses a small model shape for runtime control.
- This does not enable global full-song context.
- This does not remove current token targets.
- This does not implement factor-row autoregressive inference.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_LONGER_OR_PRODUCTION_CONFIG_CARD`.
- If cache/dataset/report completeness fails: route to `MUTATE_FACTOR_TARGET_FULL4K_CACHE_OR_DATASET`.
- If losses stay finite but factor loss explodes: route to `MUTATE_FACTOR_TARGET_FULL4K_LR_OR_LOSS_WEIGHT`.
- If factor metrics or total-loss accounting fail: route to `MUTATE_FACTOR_TARGET_FULL4K_PLUMBING`.

## Result Log Template

```text
route:
source_window_count:
train_window_count:
eval_window_count:
completed_steps:
eval_steps:
first_eval_factor_loss:
final_eval_factor_loss:
factor_loss_ratio:
eval_loss_total_min:
eval_loss_total_max:
final_eval_label_counts:
loss_total_recomputed:
cache:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to run a longer full4k or production-config card, or repair full4k cache/dataset/loss plumbing first.

## Closest Analogies And Novelty Layer

Closest analogies are staged full-dataset training-readiness gates, cache-backed data-loader audits, and auxiliary-head training ramps. This is engineering validation for the already-audited v3 factorized target, not a novelty claim. The tested layer is full4k fixed-cache training compatibility.
