# C3 Kind-Heads Reduced Sidecar Experiment Card

## Hypothesis

P16 improved C3 recovery by reducing the label space, but `REF` remained weak and the reduced flat bag still lost to unigram. Splitting the auxiliary target into separate `RAW`, `REF`, and `RES` heads, with kind-balanced auxiliary loss over the same reduced sidecar, should improve non-`RAW` recovery without regressing main mapper token loss.

## Root Objective

Move C3 toward legal full-pipeline use by testing whether decomposing the reduced target by token kind produces recoverable target-side C3 structure beyond a flat frequency prior.

## Goal Decomposition

1. Preserve full-pipeline legality: C3 labels remain target-side supervision only.
2. Reuse the P16 reduced sidecar so the data target is stable.
3. Replace the single flat C3 auxiliary head with three kind-specific heads whose logits are concatenated for existing diagnostics.
4. Balance the C3 auxiliary loss across kind slices so `REF` is not drowned out by `RAW`/`RES`.
5. Evaluate with the same top-K diagnostics and main-loss guard.

## Candidate Variants

### A. Kind heads plus kind-balanced loss

Use three auxiliary heads with vocab sizes `[512, 512, 512]`, concatenate logits, and compute per-kind BCE losses averaged equally.

### B. Kind-balanced loss only

Keep one head but average BCE by kind slices.

### C. Kind heads only

Use three heads but keep the existing flat BCE over concatenated logits.

### D. Longer P16 reduced flat run

Run P16 longer to see whether the gap closes without architecture/loss changes.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Reject condition |
| --- | --- | --- | --- |
| A | 100-step run plus diagnostics | Improves P16 recall and especially `REF`, main loss safe | Still below P16 or no non-RAW improvement |
| B | Unit-test loss slices | Faster change | Reject because P16 suggested representation decomposition, not only weighting |
| C | Unit-test head concat | Isolates architecture | Reject because weak `REF` also needs loss balance |
| D | Resume/run longer | Fewer code changes | Reject because it does not address the observed kind imbalance |

## Selected Variant

A. Kind heads plus kind-balanced loss.

## Selection Pressure

P16 showed a real signal but still weak `REF` recovery:

- `RAW` recall@20: `0.021077`
- `REF` recall@20: `0.003521`
- `RES` recall@20: `0.022457`

The selected variant is the smallest decomposition that changes both capacity allocation and loss pressure by kind while preserving the P16 sidecar and diagnostics.

## Minimal Change

Add disabled-by-default config fields:

- model: `use_c3_auxiliary_kind_heads`
- model/loss: `c3_auxiliary_kind_vocab_sizes`
- loss: `c3_auxiliary_kind_balance`

When enabled, the model builds one C3 auxiliary head per kind and concatenates logits. The loss optionally computes BCE per kind slice and averages slice losses equally. Default behavior remains unchanged.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `tests/models/mapper/v2_1/test_model.py`
- `tests/training/test_mapper_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_result_report.md`

## Dataset Slice

Use the P16 reduced sidecar and the same training slice:

- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_reduced_per_kind_top512_le3.json`
- seed: `20260616`
- `eval_fraction=0.01`
- `eval_size=8`
- `final_train_eval_size=16`
- CPU, 100 steps

## Baseline / Comparator

Primary comparator is P16 reduced flat target:

- model recall@20: `0.018997`
- hits@20: `36 / 1895`
- `RAW` recall@20: `0.021077`
- `REF` recall@20: `0.003521`
- `RES` recall@20: `0.022457`
- reduced unigram recall@20: `0.089710`
- final eval `loss/token`: `2.706093`

## Primary Metric

Model recall@20 over reduced C3 positive labels.

## Secondary Metric

- `REF` recall@20
- `RAW`/`RES` recall@20
- model sample hit rate@20
- reduced unigram recall@20
- final eval `loss/token`
- final eval `loss/c3_auxiliary`

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_enabled.yaml

uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_enabled/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_diagnostics_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_diagnostics_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Qualitative Check

Inspect whether `REF` recovery improves without collapsing `RAW`/`RES`, and whether model recall moves closer to reduced unigram rather than only improving total auxiliary loss.

## Positive Signal

Model recall@20 improves over P16, `REF` recall@20 improves over P16, all three kinds recover labels, and main token loss stays within +2% of the P13 baseline.

## Negative Signal

Recall@20 does not improve over P16, or `REF` remains flat/near zero, or main token loss regresses beyond +2%.

## Kill Criteria

If kind heads plus kind-balanced loss do not improve P16, do not continue with bag-style auxiliary C3 objectives; move to explicit structured C3 grammar/generation.

## Expected Failure Modes

- Kind-specific heads may not help if the decoder hidden state lacks enough C3 signal.
- Equal kind loss may overemphasize rare/weak labels and hurt total C3 calibration.
- Top-K over concatenated logits can still be dominated by `RAW`/`RES` unless the heads calibrate similarly.
- The 100-step run may be too short to judge final convergence, but it is enough for a fast negative.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes: one 100-step CPU run, diagnostics, and focused tests. Stop if training exceeds 45 minutes or fails to write a checkpoint.

## Confounders

- This is still a bag target, not ordered C3 generation.
- P16 reduced sidecar keeps frequent labels only.
- The eval split is small.

## Result Interpretation Plan

- If recall and `REF` improve, keep kind decomposition and test longer or with online diagnostics.
- If total recall improves but `REF` does not, mutate to explicit `REF` target design.
- If nothing improves, kill bag-style C3 auxiliary objectives and design structured C3 decoding.

## Result Log Template

```text
completed:
kind_vocab_sizes:
final_eval_loss_token:
final_eval_loss_c3_auxiliary:
model_recall_at_20:
p16_model_recall_at_20:
unigram_recall_at_20:
raw_recall_at_20:
ref_recall_at_20:
res_recall_at_20:
decision: TEST_NEXT / MUTATE / KILL
```

## Next-Loop Action

If positive, run longer or add online kind diagnostics. If negative, move from bag auxiliary labels to structured C3 target grammar/generation.

## Closest Analogies And Novelty Layer

Closest analogies are multi-task auxiliary heads, per-label-group multilabel classification, and class-balanced BCE. There is no novelty claim. The layer under test is C3 representation engineering for legal target-side mapper integration.
