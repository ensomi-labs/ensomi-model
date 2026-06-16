# C3 Auxiliary Positive-Weight Experiment Card

## Hypothesis

P14 failed because the full-vocab C3 bag auxiliary loss was dominated by absent labels. Adding a bounded positive-label weight to the existing C3 auxiliary BCE should improve top-K positive-label recovery over the P14 unweighted checkpoint without materially regressing the main mapper token loss.

## Root Objective

Move C3 toward legal full-pipeline use by testing whether the target-side auxiliary path can learn recoverable C3 labels once the observed class-imbalance failure is addressed.

## Goal Decomposition

1. Keep C3 legal for full-pipeline use: target-side labels only, no target-derived input conditioning.
2. Mutate only the failed part of P14: the unweighted full-vocab bag BCE objective.
3. Evaluate with the P14 diagnostics gate, not only with auxiliary BCE loss.
4. Preserve mapper safety: main token loss should remain near the P13 baseline.

## Candidate Variants

### A. Scalar positive-weighted BCE

Add `c3_auxiliary_positive_weight` to the C3 auxiliary loss and run the same 100-step small CPU config with `positive_weight=128`.

### B. Kind-balanced positive weights

Compute separate weights for `RAW`, `REF`, and `RES` labels and apply kind-specific BCE weighting.

### C. Top-frequency restricted sub-vocabulary

Train the auxiliary head only over frequent C3 labels to reduce extreme sparsity.

### D. Separate RAW/REF/RES heads

Replace the single full-vocab head with kind-specific heads.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | 100-step weighted run plus P14 diagnostics | recall@20 beats P14 unweighted and train-unigram, main token loss within +2% | recall@20 still below unigram or main loss regresses |
| B | Offline kind counts plus new loss path | Non-RAW target imbalance is clearly addressed | More code before scalar weighting is tested |
| C | Coverage audit of frequent labels | Frequent subset covers enough eval positives | Loses long-tail C3 structure by design |
| D | Model/head redesign smoke | Kind heads produce recoverable labels | Too large before testing direct imbalance fix |

## Selected Variant

A. Scalar positive-weighted BCE with `c3_auxiliary_positive_weight=128`.

## Selection Pressure

P14 already isolated the failure to positive-label recovery. A scalar positive weight is the smallest mutation that changes positive-vs-negative gradient pressure while leaving the model head, sidecar, dataset split, and diagnostics unchanged. If this fails, the full-vocab bag objective is unlikely to be worth more training without decomposition.

## Minimal Change

Add a disabled-by-default `c3_auxiliary_positive_weight` loss config field. When `lambda_c3_auxiliary > 0`, pass it into the existing C3 bag BCE. Default `1.0` preserves P12/P13/P14 behavior.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `tests/models/mapper/v2_1/test_model.py`
- `tests/training/test_mapper_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_result_report.md`

## Dataset Slice

Use the same slice as P13/P14:

- index: `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- seed: `20260616`
- `eval_fraction=0.01`
- `eval_size=8`
- `final_train_eval_size=16`
- `batch_size=1`
- CPU, 100 steps

## Baseline / Comparator

Primary comparator is the P14 diagnostic result for the P13 unweighted auxiliary checkpoint:

- model recall@20: `0.000436`
- unigram recall@20: `0.074139`
- main final eval token loss from P13 enabled: `2.706115`
- main final eval token loss from P13 baseline: `2.705927`

## Primary Metric

Weighted-run P14-style model recall@20 over positive C3 bag labels.

## Secondary Metric

- recall@K for K in `{1, 3, 5, 10, 20}`
- kind recall@20 for `RAW`, `REF`, and `RES`
- final eval `loss/token`
- final eval `loss/c3_auxiliary`

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml

uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_auxiliary_positive_weight_enabled/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect whether weighted predictions recover actual positive labels or merely shift to different frequent/non-target labels. Non-`RAW` recovery matters because P14 had zero `REF` and zero `RES` hits.

## Positive Signal

- model recall@20 is above train-unigram recall@20 (`0.074139`), or
- if it does not beat unigram yet, it improves over P14 by at least 10x and recovers at least one non-`RAW` kind, with main token loss within +2%.

## Negative Signal

Model recall@20 remains below unigram with no non-`RAW` recovery, or main token loss regresses beyond +2%.

## Kill Criteria

If positive weighting does not produce a large top-K recovery improvement, stop spending runtime on scalar full-vocab bag auxiliary BCE and mutate toward decomposed C3 targets.

## Expected Failure Modes

- Positive weighting may raise auxiliary loss but still fail ranking.
- The selected weight may overemphasize C3 and slightly hurt token loss.
- Recovery may improve only for frequent `RAW` labels.
- The small eval slice may produce noisy kind-level results.

## Expected Runtime / Runtime Budget

Expected runtime is similar to P13: one 100-step CPU training run plus diagnostics. Stop if the run exceeds 30 minutes or fails to write a checkpoint.

## Confounders

- P15 reuses the small P13 architecture and 100-step horizon.
- Full-vocab top-K remains a hard metric for a sparse multilabel target.
- A scalar weight cannot address per-kind or per-token frequency skew.

## Result Interpretation Plan

- If weighted BCE beats unigram and keeps token loss stable, keep this path for a longer controlled run.
- If weighted BCE improves but does not beat unigram, mutate toward kind-balanced or decomposed heads.
- If weighted BCE remains near P14, kill scalar full-vocab bag BCE.

## Result Log Template

```text
completed: true/false
positive_weight:
final_eval_loss_token:
final_eval_loss_c3_auxiliary:
model_recall_at_20:
unigram_recall_at_20:
p14_unweighted_recall_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
main_loss_regression:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, rerun with a longer budget or add online diagnostics. If ambiguous or negative, move to kind-balanced/decomposed C3 target representation.

## Closest Analogies And Novelty Layer

Closest analogies are class-imbalanced multilabel BCE, positive-class reweighting, and auxiliary target regularization. There is no novelty claim. The layer under test is whether a legal C3 target-side objective can produce recoverable labels under the current mapper architecture.
