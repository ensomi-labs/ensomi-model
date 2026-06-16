# C3 Mapper Sidecar Tensor Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: continue from P1a mapper shadow metadata toward full-pipeline C3 use.
- Acceptance source, if any: active thread goal, P0 side-stream result, P1a shadow-field result.
- Source snapshot / evidence grade: strong local evidence for C3 codec correctness; weak evidence for model utility.

## Hypothesis

Precomputed C3 side-stream token IDs can be joined to mapper v2.1 windows, padded by the mapper collate function, and carried through the training loss path behind disabled-by-default flags without changing current mapper defaults.

## Root Objective

Move C3 from audit-only metadata toward a batchable full-pipeline tokenization signal while avoiding a premature mapper architecture change.

## Goal Decomposition

- Subgoal 1: define a bounded sidecar schema for per-window C3 token IDs.
- Subgoal 2: expose optional per-sample C3 side-stream token tensors in `MapperV21WindowDataset`.
- Subgoal 3: prove collation and mapper training loss compatibility with the optional tensors present.

## Candidate Variants

- Variant A: metadata-only summary continuation.
- Variant B: fixed padded sidecar token IDs keyed by beatmap path and window start.
- Variant C: dense per-window scalar features only.
- Variant D: direct model conditioning on C3 tensors.

## Local Verification Matrix

- Variant A: verify metadata survives collate; already passed in P1a, but not a batchable token signal.
- Variant B: verify sample tensors, padded collate tensors, sidecar availability masks, and training loss path with tensors present.
- Variant C: verify scalar tensors, but it would discard tokenization structure too early.
- Variant D: verify model forward changes, but it is too large before sidecar packing is proven.

## Selected Variant

- Selected: Variant B, fixed padded sidecar token IDs.
- Rejected: Variant A is already done; Variant C is too lossy; Variant D is premature.
- Why this is the smallest useful test: it preserves the side-stream token nature of C3 while avoiding tokenizer recomputation inside the mapper dataset and avoiding model architecture edits.

## Selection Pressure

- Primary pressure: optional C3 side-stream token tensors can be created and batched.
- Guard pressure: default mapper samples, collate output, config loading, and training tests remain unchanged.
- Runtime pressure: unit tests only; no full LE<=3 cache run required.
- Kill pressure: stop if the sidecar requires invasive dataset or model rewrites.

## Research Question

Can the C3 side-stream representation enter the mapper data/training pipeline as a bounded optional tensor artifact before changing the mapper model?

## Closest Analogies / Novelty Layer

- Closest analogies: sequence sidecar features, auxiliary token streams, retrieval/codebook side channels, compressed residual streams.
- Relevant taxonomy bucket: representation engineering and pipeline instrumentation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is an engineering variation that tests whether the previously audited representation can be carried by the existing mapper pipeline.

## Minimal Change

Add disabled-by-default C3 side-stream token sidecar support to the mapper v2.1 dataset and collate path. Add optional training config plumbing for a sidecar JSON file. Do not feed the C3 tensors into the model.

## Files Likely to Change

- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `src/pulsefield_model/training/mapper_common.py`
- `src/pulsefield_model/training/mapper_v2_1.py`
- `tests/models/mapper/v2_1/test_data_windows.py`
- `tests/models/mapper/v2_1/test_model.py`
- `tests/training/test_mapper_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_sidecar_tensor_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_final_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_shadow_fields_result_report.md`

## Dataset Slice

Use existing mapper v2.1 synthetic unit-test records with synthetic C3 sidecar token IDs. Do not require full-cache artifacts for unit tests.

## Baseline / Comparator

Baseline is current mapper v2.1 behavior with no C3 sidecar args: no C3 tensor fields, unchanged collate output, unchanged trainer config semantics.

## Primary Metric

Enabled dataset samples and batches include padded C3 side-stream token tensors with correct masks and availability flags.

## Secondary Metric

The mapper training loss path accepts a raw batch containing C3 sidecar tensors without changing model outputs or loss inputs.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
```

## Guard Check

- No default C3 tensor fields appear when disabled.
- No default mapper model/training architecture changes.
- Unknown config-key tests still protect the training config surface.
- Sidecar token IDs use `0` as pad and positive IDs as payload.
- Sidecar tensors are bounded by a configurable max-token cap.

## Qualitative Check

Inspect one enabled batch. It should expose C3 as an auxiliary side-stream token tensor, not as a replacement for mapper target tokens.

## Positive Signal

- Disabled-path tests pass.
- Enabled sample includes C3 token IDs and metadata.
- Collate pads variable-length C3 side-stream tensors.
- Training loss path runs with C3 tensors present.

## Negative Signal

- Default samples gain new fields.
- Collate needs broad refactoring.
- Trainer/model rejects unknown C3 tensor fields.
- Sidecar lookup cannot use stable beatmap/window keys.

## Kill Criteria

Kill this variant if it requires model architecture changes, recomputing C3 spans in `__getitem__`, or changing default mapper training behavior.

## Expected Failure Modes

- Window start keys mismatch between sidecar and mapper records.
- C3 token sequences are too long without a cap.
- Training tensor mover ignores the new fields; if future model consumption is needed, the mover must be extended.
- Sidecar schema is insufficient for full reconstruction; this card only tests pipeline carriage.

## Confounders

Synthetic token IDs prove data plumbing, not C3 learning value. A positive result does not imply model quality improvement.

## Expected Runtime / Runtime Budget

Focused tests should finish in under 5 seconds.

## Result Interpretation Plan

- Positive result would suggest: C3 can be carried through mapper data/training as an optional side-stream tensor artifact.
- Negative result would suggest: return to sidecar-key/schema design before touching model inputs.
- Ambiguous result would require: a small generated sidecar from real P0 data.
- Human owner decides: whether to proceed to model conditioning or a real-cache sidecar builder.
- Next-loop action if positive: generate a real C3 mapper-window sidecar from P0 and measure token length distribution.
- Next-loop action if negative: mutate sidecar schema or revert to metadata-only probe.
- Next-loop action if ambiguous: add a tiny real-data sidecar smoke audit.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: real-cache sidecar generation and model utility are deferred.

## Next-Loop Action

- If positive: build a real C3 mapper-window sidecar distribution audit.
- If negative: mutate the sidecar schema before model work.
- If ambiguous: add a real-data smoke sidecar.

## Novelty Notes

- Closest analogies: auxiliary token streams and sidecar features for sequence models.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering bridge from audited representation to mapper pipeline.
