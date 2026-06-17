# Target Grammar v3 Delta-Event Factor Target Training Smoke Experiment Card

## Hypothesis

The factorized delta-event row target can run through the real v3 training runner with the production dataset, model, loss, optimizer, evaluation, checkpoint, and report machinery enabled for a short fixed-slice smoke. The smoke should prove runner compatibility and metric accounting, not final quality: factor-target fields are requested only when enabled, losses remain finite, factor-target loss appears in train/eval reports, and `loss/total` includes the factor-target weighted term.

## Root Objective

Move the audited v3 factorized target closer to a full mapper replacement path. The data-contract and model/loss plumbing gates proved row construction, default-off behavior, real-batch loss, and gradients. This card tests the next production boundary: the actual training runner can train/evaluate/report with factor-target supervision enabled.

## Goal Decomposition

- Subgoal 1: Repair mapper metric finalization so enabled v3 auxiliary/factor losses are included in aggregated `loss/total`.
- Subgoal 2: Run a short production v3 training smoke on the fixed real-data slice with `use_delta_event_factor_target=True` and `lambda_delta_event_factor_target > 0`.
- Subgoal 3: Verify the training report and checkpoint are written and contain the enabled model/loss/dataset flags.
- Subgoal 4: Verify train/eval metrics include finite factor-target loss, label counts, and lambda metrics.
- Subgoal 5: Preserve default training behavior and avoid inference/rollout changes.

## Candidate Variants

- Variant A: Add a bounded evaluator that calls `run_mapper_v3_phase_b_training` with a small model, fixed index, cached control teacher tensors, factor-target branch enabled, and a short step budget; repair metric finalization as needed.
- Variant B: Keep using the model/loss plumbing evaluator and skip the real training runner.
- Variant C: Launch a larger real training run immediately.
- Variant D: Replace the main token loss with factor-target loss in this smoke.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Training completes; report/checkpoint written; final metrics finite; factor-target loss and lambda present; `loss/total` includes factor-target weighted term; dataset report says factor target enabled | Runner crash, missing factor labels, missing metrics, bad total-loss accounting, non-finite loss |
| B | Fast | Does not test DataLoader/optimizer/eval/report/checkpoint path |
| C | More signal | Too expensive before runner compatibility is proven |
| D | Closer to replacement | Too broad before proving auxiliary/factor smoke stability |

## Selected Variant

Variant A: short real training-run smoke with factor-target branch/loss enabled and metric-finalizer repair.

## Selection Pressure

Variant A is the smallest production-facing step after model/loss plumbing. It exercises the real training runner without claiming final quality or replacing the main token stream. Variant B stalls before the runner boundary. Variant C spends too much runtime before a basic smoke. Variant D changes the objective from runner compatibility to replacement training too early.

## Minimal Change

- Update `default_mapper_metric_finalizer` to include existing optional mapper losses plus `loss/delta_event_factor_target` when their lambdas are enabled.
- Add `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_training_smoke.py`.
- The evaluator calls `run_mapper_v3_phase_b_training` with a small v3 model, fixed slice, cached control teacher tensors, `include_full_song_context=False`, `use_global_context=False`, and factor-target loss enabled.
- Add tests for finalizer accounting and evaluator route decisions.
- Record summary and markdown result report.

## Files Likely To Change

- `src/pulsefield_model/training/mapper_runner.py`
- `src/pulsefield_model/training/mapper_common.py`
- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_training_smoke.py`
- `tests/training/test_mapper_training_runner.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_result_report.md`

## Read-Only Context

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

## Dataset Slice

Use the fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

Training smoke uses a tiny random train/eval split from this index with `eval_size=8`, `final_train_eval_size=8`, and a short step budget.

## Baseline / Comparator

- Previous model/loss route: `TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD`.
- Previous real-batch factor loss: finite/positive `19.298216`.
- Current default v3 training runner works without factor-target fields.
- Current metric finalizer must include enabled factor-target losses before training-smoke reports are trusted.

## Primary Metric

Production training smoke completes and reports valid factor-target supervision:

- `completed_steps == max_steps`
- final eval `loss/total` finite
- final eval `loss/delta_event_factor_target` finite and positive
- final eval `phase/lambda_delta_event_factor_target > 0`
- report dataset has `include_delta_event_factor_target: true`
- checkpoint file exists

## Secondary Metrics

- last train factor-target loss
- final train factor-target loss
- final eval factor label counts
- final eval `loss/total` equals recomputed token plus weighted component losses within tolerance
- elapsed runtime

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_training_smoke
uv run --group dev pytest tests/training/test_mapper_training_runner.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/training/mapper_runner.py src/pulsefield_model/training/mapper_common.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_training_smoke.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_summary.json >/dev/null
git diff --check
```

## Guard Check

- No inference or rollout changes.
- No C3 backreference or future lookup.
- Default model/loss/training configs remain factor-target disabled.
- The smoke remains a runner compatibility gate, not a full training claim.
- Generated checkpoints/runs are not committed; only audit reports are committed.

## Qualitative Check

The result report must say this is a training-run smoke only. It must not claim stable training, mapper quality, rollout quality, online factor-row inference, or final v3 replacement readiness.

## Positive Signal

The real v3 training runner completes the short smoke with finite factor-target metrics, correct metric accounting, a written report/checkpoint, and explicit enabled dataset/model/loss flags.

## Negative Signal

Runner crash, missing nested factor labels, missing factor metrics, non-finite loss, incorrect aggregate `loss/total`, missing checkpoint/report, or default config regression.

## Kill Criteria

- Training runner cannot process factor-target batches.
- Final eval factor-target loss is missing or non-finite.
- Aggregated `loss/total` excludes the enabled factor-target term after the finalizer repair.

## Expected Failure Modes

- Metric finalizer may overwrite `loss/total` and drop new auxiliary terms.
- Eval split may be too large for a fast smoke unless bounded.
- Factor target branch may exceed `max_seq_len` if the smoke config is too small.
- Cached control teacher dimensions must remain `384` even with small mapper `d_model`.

## Expected Runtime / Runtime Budget

Expected runtime: under 3 minutes on CPU with cached control teacher tensors.

Stop condition: fail on runner crash, missing cache, non-finite factor loss, missing report/checkpoint, or incorrect total-loss accounting.

## Confounders

- This does not train long enough to establish model quality.
- This does not remove current token targets.
- This does not implement factor-row autoregressive inference.
- This does not prove full4k training stability.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD`.
- If metric accounting fails: route to `MUTATE_FACTOR_TARGET_TRAINING_METRIC_ACCOUNTING`.
- If runner/data/model integration fails: route to `MUTATE_FACTOR_TARGET_TRAINING_SMOKE`.

## Result Log Template

```text
route:
completed_steps:
max_steps:
report_path:
checkpoint_path:
final_eval_loss_total:
final_eval_factor_loss:
final_eval_lambda_factor:
factor_label_counts:
loss_total_recomputed:
loss_total_delta:
dataset_include_factor_target:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to run a longer factor-target training card or repair training-run/metric accounting first.

## Closest Analogies And Novelty Layer

Closest analogies are training-smoke gates, auxiliary-head training gates, and staged representation migration checks. This is engineering validation for an already-audited representation, not a novelty claim. The tested layer is production training-run compatibility.
