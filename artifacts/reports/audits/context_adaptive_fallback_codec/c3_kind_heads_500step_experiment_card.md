# C3 Kind-Heads 500-Step Experiment Card

## Hypothesis

P17 showed that reduced C3 kind heads improve target recovery, especially `REF`, but 100 steps may be too short for the auxiliary representation to outrun a train-unigram prior. A 500-step run of the same kind-head setup should clarify whether the gap to unigram is closing or whether bag-style reduced C3 has reached a ceiling.

## Root Objective

Move C3 toward legal full-pipeline use by determining whether the best current reduced/decomposed auxiliary target continues to improve with modest additional training before switching to structured C3 grammar/generation.

## Goal Decomposition

1. Keep the P17 representation unchanged: reduced top-512-per-kind sidecar, three kind heads, kind-balanced C3 loss.
2. Increase only the training horizon from 100 to 500 steps.
3. Preserve full-pipeline legality: C3 remains target-side supervision only.
4. Re-evaluate with the same top-K diagnostics and main-loss guard.

## Candidate Variants

### A. 500-step kind-head run from scratch

Run the P17 config from scratch for 500 CPU steps with the same seed and dataset.

### B. Resume P17 to 500 steps

Resume the P17 checkpoint and continue training.

### C. 1000-step kind-head run

Run longer to improve convergence evidence.

### D. Structured C3 grammar prototype

Stop bag targets and design explicit grammar/generation targets.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | 500-step run plus diagnostics | Recall improves materially over P17 and main loss is safe | Recall stalls or remains far below unigram |
| B | Resume checkpoint validation | Faster continuation | Reject if strict resume config makes comparison brittle |
| C | Longer run | Stronger convergence evidence | Too much runtime before knowing 500-step trend |
| D | New card/design | Directly targets production form | Premature before testing P17’s trajectory |

## Selected Variant

A. 500-step kind-head run from scratch.

## Selection Pressure

Running from scratch avoids resume-config coupling and keeps the comparison simple: same seed, same dataset, same architecture, same loss, only longer training. 500 steps is a bounded runtime increase over P17 and should be enough to reveal whether the top-K recovery curve is still moving.

## Minimal Change

Add one run config copied from P17 with:

- `run_name=c3_kind_heads_reduced_sidecar_500step`
- `output_dir=artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_500step`
- `max_steps=500`
- `eval_every=100`
- `save_every=500`
- `log_every=100`

No model or loss code changes are expected.

## Files Likely To Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_result_report.md`

## Dataset Slice

Use the P16/P17 reduced sidecar and same split:

- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_reduced_per_kind_top512_le3.json`
- seed: `20260616`
- `eval_fraction=0.01`
- `eval_size=8`
- `final_train_eval_size=16`
- CPU, batch size 1

## Baseline / Comparator

Primary comparator is P17:

- model recall@20: `0.025858`
- hits@20: `49 / 1895`
- `RAW` recall@20: `0.032787`
- `REF` recall@20: `0.017606`
- `RES` recall@20: `0.021136`
- reduced unigram recall@20: `0.089710`
- final eval `loss/token`: `2.705861`

## Primary Metric

Model recall@20 over reduced C3 positive labels.

## Secondary Metric

- `RAW`/`REF`/`RES` recall@20
- sample hit rate@20
- reduced unigram recall@20
- final eval `loss/token`
- final eval `loss/c3_auxiliary`
- C3 auxiliary loss trend over eval checkpoints

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_enabled.yaml

uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_500step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect whether recall is closing the gap to reduced unigram, and whether `REF` continues improving rather than the model only increasing `RAW` hits.

## Positive Signal

Model recall@20 improves over P17 by at least 50% relative, `REF` recall improves over P17, all three kinds recover labels, and main token loss stays within +2% of P13 baseline.

## Negative Signal

Recall@20 improves weakly or regresses, `REF` stalls, or the run still remains far below unigram after 500 steps.

## Kill Criteria

If 500 steps do not produce a clear trajectory toward unigram, stop spending runtime on bag-style reduced C3 and move to structured C3 grammar/generation.

## Expected Failure Modes

- Longer training may improve BCE loss without improving top-K recovery.
- The small eval split can overstate or understate kind-specific changes.
- The frequency prior may remain stronger than the model.
- The auxiliary task may need ordered/structured targets rather than bag labels.

## Expected Runtime / Runtime Budget

Expected runtime is under 5 minutes on CPU. Stop if training exceeds 15 minutes or fails to write a checkpoint.

## Confounders

- This is still a bag target, not ordered C3 generation.
- The reduced sidecar only covers frequent labels.
- Training remains a small CPU probe, not a production-scale mapper run.

## Result Interpretation Plan

- If recall approaches or beats unigram, keep kind-head C3 and test a larger run.
- If recall improves but remains far below unigram, mutate to structured targets.
- If recall stalls, kill bag-style reduced auxiliary objectives.

## Result Log Template

```text
completed:
max_steps:
final_eval_loss_token:
final_eval_loss_c3_auxiliary:
model_recall_at_20:
p17_model_recall_at_20:
unigram_recall_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, run a larger kind-head check. If negative or only mildly positive, move to structured C3 target grammar/generation.

## Closest Analogies And Novelty Layer

Closest analogies are longer-horizon auxiliary target probes and learning-curve checks for multi-task heads. There is no novelty claim. The tested layer is whether the current C3 representation has enough learnable signal to justify further integration work.
