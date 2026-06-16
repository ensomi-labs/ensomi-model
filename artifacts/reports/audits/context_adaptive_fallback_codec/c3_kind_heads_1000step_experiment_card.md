# C3 Kind-Heads 1000-Step Experiment Card

## Hypothesis

P18 showed that reduced C3 kind heads continue to improve with additional training and that `REF`/`RES` are no longer the main blockers. A 1000-step run should determine whether the same representation can catch or beat the reduced unigram prior, or whether the remaining `RAW` gap requires a structured target redesign.

## Root Objective

Move C3 toward legal full-pipeline use by testing whether the current best legal target-side C3 auxiliary representation has enough training trajectory to become useful, or whether the next step must be a RAW-focused structured target.

## Goal Decomposition

1. Keep the P18 representation unchanged: reduced top-512-per-kind sidecar, three kind heads, kind-balanced C3 loss.
2. Increase only the training horizon from 500 to 1000 steps.
3. Preserve full-pipeline legality: C3 is target-side supervision only.
4. Evaluate with the same top-K diagnostics and main-loss guard.
5. Decide whether to keep training this representation or mutate to RAW-structured targets.

## Candidate Variants

### A. 1000-step kind-head run from scratch

Run the P18 config from scratch for 1000 CPU steps with the same seed and dataset.

### B. Resume P18 to 1000 steps

Resume the P18 checkpoint and continue training.

### C. RAW-focused structured audit

Stop training and inspect/model `RAW` tokens directly.

### D. 2000-step kind-head run

Run longer immediately to test convergence.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | 1000-step run plus diagnostics | Recall catches/beats unigram or clearly closes the RAW gap | Still materially below unigram |
| B | Resume validation | Faster continuation | Reject if resume strictness makes setup brittle |
| C | Offline RAW audit | More targeted | Reject until the 1000-step trajectory is known |
| D | Longer run | Stronger convergence signal | Too much runtime before 1000-step evidence |

## Selected Variant

A. 1000-step kind-head run from scratch.

## Selection Pressure

P18 improved recall@20 from `0.025858` to `0.058575` and brought `REF` close to unigram. The simplest next check is whether more training closes the remaining `RAW` gap. A from-scratch 1000-step run avoids resume-coupling and changes only one variable: training horizon.

## Minimal Change

Add one run config copied from P18 with:

- `run_name=c3_kind_heads_reduced_sidecar_1000step`
- `output_dir=artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step`
- `max_steps=1000`
- `eval_every=200`
- `save_every=1000`
- `log_every=200`

No model or loss code changes are expected.

## Files Likely To Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_result_report.md`

## Dataset Slice

Use the P16/P17/P18 reduced sidecar and same split:

- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_reduced_per_kind_top512_le3.json`
- seed: `20260616`
- `eval_fraction=0.01`
- `eval_size=8`
- `final_train_eval_size=16`
- CPU, batch size 1

## Baseline / Comparator

Primary comparator is P18:

- model recall@20: `0.058575`
- hits@20: `111 / 1895`
- `RAW` recall@20: `0.069087`
- `REF` recall@20: `0.059859`
- `RES` recall@20: `0.046235`
- reduced unigram recall@20: `0.089710`
- final eval `loss/token`: `2.316796`

## Primary Metric

Model recall@20 over reduced C3 positive labels.

## Secondary Metric

- `RAW`/`REF`/`RES` recall@20
- model sample hit rate@20
- reduced unigram recall@20
- final eval `loss/token`
- final eval `loss/c3_auxiliary`
- C3 auxiliary loss trend over eval checkpoints

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml

uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect whether top-20 recovery catches unigram overall, and whether `RAW` specifically closes the gap while `REF` and `RES` stay healthy.

## Positive Signal

Model recall@20 reaches at least 90% of reduced unigram recall@20, or beats unigram on two kinds and closes the `RAW` gap by at least 50% relative to P18, while main token loss stays within +2% of the nearest no-C3/previous comparator.

## Negative Signal

Recall@20 remains materially below unigram and the remaining gap is still mostly `RAW`.

## Kill Criteria

If 1000 steps still leaves the model far below unigram, stop small bag-style training probes and move to a RAW-focused structured target audit.

## Expected Failure Modes

- Longer training may keep reducing BCE without enough top-K improvement.
- The model may overfit frequent `REF`/`RES` labels while `RAW` remains hard.
- The small eval split may make final decisions noisy.
- A flat bag target may be the wrong form for `RAW`.

## Expected Runtime / Runtime Budget

Expected runtime is under 5 minutes on CPU. Stop if training exceeds 20 minutes or fails to write a checkpoint.

## Confounders

- This is still a bag target, not ordered C3 generation.
- The reduced sidecar only covers frequent labels.
- The run is still a small CPU probe.

## Result Interpretation Plan

- If recall catches unigram, keep the kind-head path and test a larger production-adjacent run.
- If recall improves but remains below unigram, mutate to RAW-focused structured targets.
- If recall stalls, kill bag-style reduced C3 auxiliary objectives.

## Result Log Template

```text
completed:
max_steps:
final_eval_loss_token:
final_eval_loss_c3_auxiliary:
model_recall_at_20:
p18_model_recall_at_20:
unigram_recall_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, run a production-adjacent kind-head check. If negative or ambiguous, do a RAW-focused structured C3 target audit.

## Closest Analogies And Novelty Layer

Closest analogies are learning-curve checks for multi-task auxiliary heads and frequency-prior comparisons. There is no novelty claim. The tested layer is whether the current C3 representation is worth scaling toward full-pipeline use.
