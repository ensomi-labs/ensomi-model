# C3 Auxiliary-Target Diagnostics Experiment Card

## Hypothesis

The P13 C3 auxiliary target loss decrease reflects learned chart-local C3 structure, not only a bias toward globally common side-stream labels. A post-hoc diagnostic should show model top-K recovery above a train-unigram baseline, and should reveal whether recovery is concentrated in `RAW`, `REF`, or `RES` tokens.

## Root Objective

Move C3 toward legal full-pipeline use by hardening the target-side auxiliary path with observable diagnostics before deciding whether to keep auxiliary regularization or decompose C3 into a richer target grammar.

## Goal Decomposition

1. Verify that the trained P13 checkpoint can recover held-out C3 bag labels above a simple common-label comparator.
2. Identify which C3 token kinds (`RAW`, `REF`, `RES`) are actually recovered.
3. Preserve the full-pipeline legality constraint: C3 labels are target-side supervision only, not target-derived mapper inputs.

## Candidate Variants

### A. Post-hoc top-K checkpoint diagnostics

Load the P13 auxiliary checkpoint, rebuild the same eval split, compute model top-K label recovery, compare against train-unigram top-K labels, and split target/hit counts by token kind.

### B. Add online training metrics

Patch the loss to emit top-K and kind metrics during training, then rerun the 100-step comparison.

### C. Longer auxiliary training run

Run a larger auxiliary target comparison and rely on loss curves plus final loss.

### D. Grammar decomposition prototype

Start replacing the bag target with ordered or kind-specific C3 generation heads.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | Load P13 checkpoint and score the existing eval split | JSON report includes top-K model vs unigram and kind metrics | Checkpoint/dataset cannot be reconstructed, or metrics show no signal beyond unigram |
| B | Unit-test online metric math | Metrics are correct and training still passes | Adds training runtime before knowing whether diagnostics are useful |
| C | Resume/run longer | Aux loss continues improving without main loss drift | Still cannot explain what was learned |
| D | Prototype new head | Ordered/kind target is trainable | Too large before diagnosing P13's learned signal |

## Selected Variant

A. Post-hoc top-K checkpoint diagnostics.

## Selection Pressure

Variant A is fastest and most diagnostic. It reuses P13's exact checkpoint and eval split, adds no new training, and directly tests the main ambiguity surfaced by P13: common-label bias versus learned window-conditioned C3 recovery. B is useful later if A passes. C and D are premature.

## Minimal Change

Add one reusable evaluation module for C3 auxiliary diagnostics and focused metric tests. Run it against the P13 enabled checkpoint. Commit only the evaluator, tests, experiment card, JSON summary, and result report.

## Files Likely To Change

- `src/pulsefield_model/evals/c3_auxiliary_diagnostics.py`
- `tests/evals/test_c3_auxiliary_diagnostics.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_diagnostics_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_diagnostics_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_diagnostics_result_report.md`

## Dataset Slice

Use the same P13 enabled run config:

- index: `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- split: same seed `20260616`, `eval_fraction=0.01`, `eval_size=8`
- expected eval windows: 142

## Baseline / Comparator

Train-unigram top-K C3 labels computed from the train split, using unique bag labels per mapper window.

## Primary Metric

Model micro recall@K over positive C3 bag labels on eval windows, compared with train-unigram micro recall@K for K in `{1, 3, 5, 10, 20}`.

## Secondary Metric

Per-kind target counts, hit counts, and recall@K for `RAW`, `REF`, and `RES`, plus sample hit rate@K and precision@K.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_auxiliary_target_learnability_enabled/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect the result report for:

- model top-K recall versus unigram recall;
- whether `REF` and `RES` are recovered or only `RAW`;
- whether predicted top-K kind distribution collapses.

## Positive Signal

Model recall@20 is above train-unigram recall@20 by at least 10% relative, and at least one non-`RAW` kind has nonzero model recall@20.

## Negative Signal

Model recall@20 is tied with or worse than unigram, or recovery is entirely common `RAW` labels despite nonzero `REF`/`RES` targets.

## Kill Criteria

If model top-K recovery does not beat unigram at K=20, do not spend more runtime on the current full-vocab bag auxiliary head without changing the objective or label structure.

## Expected Failure Modes

- Eval split reconstruction differs from P13 if config loading changes.
- The auxiliary head may rank frequent labels well but fail rare `REF`/`RES` tokens.
- Full-vocab BCE may produce calibrated loss improvement without useful top-K recovery.
- Eval slice is small, so per-kind metrics may have high variance.

## Expected Runtime / Runtime Budget

Expected runtime is under 5 minutes on CPU because no training is performed. Stop if checkpoint scoring exceeds 15 minutes or cannot reconstruct the P13 eval split.

## Confounders

- The eval split is only eight map groups and 142 windows.
- Train-unigram is a strong baseline for globally frequent C3 patterns.
- Top-K bag recovery does not test ordered side-stream generation.

## Result Interpretation Plan

- If model beats unigram and recovers non-`RAW` kinds, keep the auxiliary path and consider online diagnostics or kind-specific heads.
- If model beats unigram only on `RAW`, mutate toward kind-balanced or decomposed C3 supervision.
- If model does not beat unigram, kill the current full-vocab bag auxiliary head as a production direction.

## Result Log Template

```text
completed: true/false
eval_window_count:
positive_sample_count:
positive_label_count:
model_recall_at_20:
unigram_recall_at_20:
relative_lift_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, add online diagnostics or test a kind-balanced auxiliary loss. If negative, mutate away from full-vocab bag auxiliary toward decomposed C3 representation.

## Closest Analogies And Novelty Layer

Closest analogies are multilabel auxiliary prediction, bag-of-events regularization, and retrieval-style top-K evaluation. There is no novelty claim here. The layer under test is engineering feasibility and representation observability for C3 target-side supervision.
