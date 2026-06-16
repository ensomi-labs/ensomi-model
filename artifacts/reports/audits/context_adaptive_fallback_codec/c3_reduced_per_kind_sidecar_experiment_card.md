# C3 Reduced Per-Kind Sidecar Experiment Card

## Hypothesis

P14/P15 failed because the auxiliary target asked the mapper to rank 14,294 sparse labels in one flat space. A reduced per-kind sidecar that keeps the top 512 train labels from each C3 kind (`RAW`, `REF`, `RES`) should make positive C3 label recovery more learnable while preserving most held-out label coverage.

## Root Objective

Move C3 toward legal full-pipeline use by reformulating the target-side auxiliary representation into a smaller structured target that can be trained and diagnosed before attempting any production mapper-token replacement.

## Goal Decomposition

1. Keep C3 legal: target-side labels only, no C3 input conditioning.
2. Reduce the flat C3 label space from 14,294 labels to a train-derived per-kind subset.
3. Preserve enough C3 coverage that the experiment still probes beatmap structure.
4. Evaluate with top-K diagnostics against a train-unigram baseline.

## Candidate Variants

### A. Top-512 labels per kind, remapped sidecar

Select the top 512 train labels from each of `RAW`, `REF`, and `RES`, remap them into a 1,536-label sidecar, train the existing C3 auxiliary head, and diagnose top-K recovery.

### B. Top-256 labels per kind

Smaller 768-label sidecar with lower positive coverage.

### C. Top-1024 labels per kind

Larger 3,072-label sidecar with high coverage but closer to the failed sparse-label setup.

### D. Separate kind-specific heads

Implement three model heads and three losses.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | Coverage audit plus 100-step train/diagnostics | High held-out coverage and model recall@20 improves materially | Recall remains below unigram with no useful lift |
| B | Coverage audit | Simpler target | Reject if held-out coverage is too low |
| C | Coverage audit | Better coverage | Reject if target remains too large for a fast recovery test |
| D | Unit-test model heads | More explicit decomposition | Too much architecture before testing reduced-label feasibility |

## Selected Variant

A. Top-512 labels per kind, remapped sidecar.

## Selection Pressure

The coverage audit found:

| Top-N per kind | Vocab size | Train coverage | Eval coverage | RAW eval | REF eval | RES eval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 768 | 0.6179 | 0.6324 | 0.6526 | 0.7048 | 0.5815 |
| 512 | 1536 | 0.8018 | 0.8264 | 0.8019 | 0.8554 | 0.8449 |
| 1024 | 3072 | 0.9297 | 0.9372 | 0.9164 | 0.9578 | 0.9542 |

Top-512 is the smallest option that preserves more than 80% held-out coverage across all kinds. Top-256 throws away too much structure. Top-1024 may still be too sparse for a fast negative/positive result.

## Minimal Change

Add a sidecar reduction tool that:

1. reconstructs the P13/P14 train split from the run config;
2. counts unique train C3 labels per window;
3. selects the top 512 labels per kind;
4. writes a reduced/remapped sidecar with all windows retained but non-selected tokens dropped.

Then run the existing mapper auxiliary path with:

- `c3_auxiliary_vocab_size=1536`
- `lambda_c3_auxiliary=0.05`
- `c3_auxiliary_positive_weight=64.0`

## Files Likely To Change

- `src/pulsefield_model/evals/c3_reduced_sidecar.py`
- `tests/evals/test_c3_reduced_sidecar.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_result_report.md`

The generated reduced sidecar lives under `artifacts/cache/c3_mapper_window_sidecar/` and is treated as a large generated cache artifact.

## Dataset Slice

Use the same slice as P13/P14/P15:

- index: `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- source sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- seed: `20260616`
- `eval_fraction=0.01`
- `eval_size=8`
- `final_train_eval_size=16`
- CPU, 100 training steps

## Baseline / Comparator

Primary comparator is P15/P14 full-vocab diagnostics:

- P14 unweighted recall@20: `0.000436`
- P15 weighted recall@20: `0.000872`
- full-vocab train-unigram recall@20: `0.074139`

For the reduced sidecar, diagnostics also compute a reduced train-unigram comparator on the reduced train split.

## Primary Metric

Reduced-sidecar model recall@20 over positive reduced C3 bag labels.

## Secondary Metric

- model recall@K for K in `{1, 3, 5, 10, 20}`
- reduced train-unigram recall@20
- kind recall@20 for `RAW`, `REF`, `RES`
- retained label coverage versus the exact sidecar
- final eval `loss/token`
- final eval `loss/c3_auxiliary`

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_reduced_sidecar \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml \
  --source-sidecar artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json \
  --output-sidecar artifacts/cache/c3_mapper_window_sidecar/c3_reduced_per_kind_top512_le3.json \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_generation_summary.json \
  --top-per-kind 512

uv run python -m pulsefield_model.training.mapper_v2_1 \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_enabled.yaml

uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_reduced_per_kind_sidecar_enabled/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect whether reduced predictions recover labels across all three kinds, and whether improvement is real compared with the reduced train-unigram baseline rather than only compared with the failed full-vocab heads.

## Positive Signal

Model recall@20 beats the reduced train-unigram recall@20, or improves over P15 by at least 10x while recovering all three kinds and keeping main token loss within +2%.

## Negative Signal

Model recall@20 remains below reduced unigram and does not recover all three kinds, or main token loss regresses beyond +2%.

## Kill Criteria

If the reduced 1,536-label target still cannot beat unigram or recover broad kind coverage, stop using a flat bag auxiliary objective and move to explicit grammar/decomposed heads.

## Expected Failure Modes

- Reduced train-unigram may become a stronger comparator because rare labels are removed.
- The model may still learn label priors instead of audio/chart-conditioned structure.
- The selected 512-per-kind target may underrepresent rare C3 motifs.
- Positive weight `64.0` may be too weak or too strong for the reduced space.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes: sidecar generation, one 100-step CPU training run, diagnostics, and focused tests. Stop if sidecar generation or training exceeds 45 minutes.

## Confounders

- This is still a bag-label objective, not ordered C3 generation.
- The eval split is small.
- The reducer uses the train split for label selection, so it tests frequent-pattern recovery rather than full C3 coverage.

## Result Interpretation Plan

- If positive, keep reduced/decomposed targets alive and test a longer run or separate kind heads.
- If ambiguous, try explicit kind heads with the same selected vocab.
- If negative, kill flat bag targets and move to structured grammar/token generation.

## Result Log Template

```text
generated_vocab_size:
exact_eval_positive_labels:
reduced_eval_positive_labels:
retained_eval_coverage:
final_eval_loss_token:
final_eval_loss_c3_auxiliary:
model_recall_at_20:
reduced_unigram_recall_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, either run longer or split the reduced vocabulary into separate kind heads. If negative, stop flat bag objectives and design a structured C3 target grammar.

## Closest Analogies And Novelty Layer

Closest analogies are sub-vocabulary selection, frequency-truncated multilabel targets, and target-side auxiliary regularization. There is no novelty claim. The tested layer is representation engineering for making C3 target supervision usable in the mapper pipeline.
