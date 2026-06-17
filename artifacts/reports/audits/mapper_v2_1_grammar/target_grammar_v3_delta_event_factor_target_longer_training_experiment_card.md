# Target Grammar v3 Delta-Event Factor Target Longer Training Experiment Card

## Hypothesis

The factorized delta-event target branch remains stable beyond the two-step smoke when trained through the real v3 runner on the fixed real-data slice: multiple eval checkpoints have finite token and factor losses, factor-label counts remain positive, aggregate loss accounting remains correct, and final eval factor loss does not explode relative to the first eval.

## Root Objective

Move the factorized v3 target toward a full mapper replacement path. The previous smoke proved runner compatibility. This card tests the next necessary boundary before any rollout or full-dataset training: short-horizon stability under repeated optimizer/eval/report cycles.

## Goal Decomposition

- Subgoal 1: Run the production v3 training runner for more than a smoke with factor-target branch/loss enabled.
- Subgoal 2: Capture at least three eval points, including an early eval and final eval.
- Subgoal 3: Verify all eval points have finite `loss/total`, `loss/token`, and `loss/delta_event_factor_target`.
- Subgoal 4: Verify factor-label counts remain positive at eval.
- Subgoal 5: Verify final eval factor loss is not exploding relative to the first eval.
- Subgoal 6: Preserve the no-rollout/no-inference/no-C3/no-future-lookup guard.

## Candidate Variants

- Variant A: Run a 16-step fixed-slice training stability gate with evals at step 1, 8, and 16 using the same small production v3 model shape from the smoke.
- Variant B: Jump directly to a full-cache/full4k training run.
- Variant C: Run only the two-step smoke again with stricter assertions.
- Variant D: Replace the main token objective with factor target loss only.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Training completes; >=3 eval points; finite losses; positive factor labels; final/first eval factor-loss ratio <= threshold; report/checkpoint written | Non-finite loss, missing factor metrics, exploding factor loss, runner/report failure |
| B | More representative if it works | Too expensive before short stability is known |
| C | Fast | Does not test stability beyond the smoke already passed |
| D | Closer to replacement | Too broad before auxiliary/factor stability is established |

## Selected Variant

Variant A: 16-step fixed-slice production-runner stability gate with factor-target loss enabled.

## Selection Pressure

Variant A is the smallest experiment that adds new evidence beyond the smoke: multiple optimizer/eval cycles and a non-explosion check. Variant B is deferred until this gate passes. Variant C repeats existing evidence. Variant D changes too much at once and would confound branch stability with replacement objective design.

## Minimal Change

- Add `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_longer_training.py`.
- Reuse the production `run_mapper_v3_phase_b_training` path with the small factor-target-enabled v3 model.
- Parse the runner report history for eval-point stability.
- Add focused route/accounting tests.
- Record summary and markdown result report.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_longer_training.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_result_report.md`

## Read-Only Context

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_training_smoke_summary.json`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

## Dataset Slice

Use the fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

Training uses a bounded train/eval split from this index with `eval_size=16`, `final_train_eval_size=16`, `batch_size=4`, `max_steps=16`, and `eval_every=8`.

## Baseline / Comparator

- Training-smoke route: `TEST_DELTA_EVENT_FACTOR_TARGET_LONGER_TRAINING_CARD`.
- Smoke completed 2 steps with final eval factor loss `20.355467` and exact aggregate loss accounting.
- Current gate must add stability evidence beyond the smoke: multiple eval points and a non-explosion ratio.

## Primary Metric

Training stability:

- `completed_steps == 16`
- at least `3` eval points
- every eval `loss/total`, `loss/token`, and `loss/delta_event_factor_target` is finite
- final eval factor-loss ratio versus first eval is `<= 1.15`

## Secondary Metrics

- final eval factor loss
- first eval factor loss
- min/max eval factor loss
- final train factor loss
- final eval factor label counts
- final eval aggregate `loss/total` recomputation delta
- report/checkpoint existence

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_longer_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_longer_training.py tests/evals/test_mapper_v3_delta_event_factor_target_training_smoke.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_longer_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_longer_training_summary.json >/dev/null
git diff --check
```

## Guard Check

- No inference or rollout changes.
- No C3 backreference or future lookup.
- Default configs remain disabled.
- Generated checkpoints/runs are not committed; only audit reports are committed.
- This remains fixed-slice training stability, not full4k training or replacement readiness.

## Qualitative Check

The report must state that this is short-horizon training stability only. It must not claim generated-chart quality, full4k stability, factor-row inference readiness, or final v3 replacement.

## Positive Signal

The 16-step real-runner training completes with finite metrics at all eval points, positive factor labels, correct total-loss accounting, and final/first eval factor-loss ratio `<= 1.15`.

## Negative Signal

Non-finite losses, missing factor metrics, missing eval history, exploding factor loss, incorrect aggregate loss accounting, or missing report/checkpoint.

## Kill Criteria

- Any eval `loss/total` or `loss/delta_event_factor_target` is non-finite.
- Final/first eval factor-loss ratio exceeds `1.15`.
- Factor-target labels disappear from eval metrics.

## Expected Failure Modes

- Loss may be finite but drift upward on the fixed eval slice.
- Eval history parsing may miss the first/final eval.
- Short fixed-slice training can show noisy improvements; pass means stability only, not quality.

## Expected Runtime / Runtime Budget

Expected runtime: under 5 minutes on CPU with cached control teacher tensors.

Stop condition: fail on runner crash, non-finite loss, missing metrics, missing report/checkpoint, or non-explosion-ratio failure.

## Confounders

- This is not a full training run.
- This is not full4k training stability.
- This does not remove current token targets.
- This does not implement factor-row autoregressive inference.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_FULL4K_TRAINING_CARD`.
- If finite but exploding: route to `MUTATE_FACTOR_TARGET_LR_OR_LOSS_WEIGHT`.
- If runner/metrics fail: route to `MUTATE_FACTOR_TARGET_LONGER_TRAINING_PLUMBING`.

## Result Log Template

```text
route:
completed_steps:
eval_steps:
first_eval_factor_loss:
final_eval_factor_loss:
factor_loss_ratio:
eval_loss_total_min:
eval_loss_total_max:
final_eval_label_counts:
loss_total_recomputed:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to run a bounded full4k/fixed-cache factor-target training card or repair learning-rate/loss-weight/plumbing first.

## Closest Analogies And Novelty Layer

Closest analogies are short-horizon training stability gates and auxiliary-head training ramps. This is engineering validation for the already-audited v3 factorized target, not a novelty claim. The tested layer is training stability beyond smoke.
