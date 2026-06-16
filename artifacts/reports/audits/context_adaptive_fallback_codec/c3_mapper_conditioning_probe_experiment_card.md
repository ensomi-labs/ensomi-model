# C3 Mapper Conditioning Probe Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P5 exact C3 sidecar generation passed correctness gates but surfaced cross-window reference-span risk before model conditioning.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_exact_mapper_window_sidecar_result_report.md`.
- Source snapshot / evidence grade: strong local evidence that exact C3 sidecar tensors are loader-compatible and cap-tractable; no evidence yet that the mapper can consume them as trainable input.

## Hypothesis

A disabled-by-default pooled C3 side-stream token conditioning path can be added to Mapper v2.1 so exact C3 sidecar tensors influence logits and receive gradients when enabled, while default mapper behavior remains unchanged.

## Root Objective

Move C3 from a verified exact sidecar artifact into the mapper model boundary as a minimal full-pipeline conditioning signal.

## Goal Decomposition

- Subgoal 1: preserve default Mapper v2.1 behavior when C3 conditioning is disabled.
- Subgoal 2: add an opt-in conditioning path that consumes existing `c3_side_stream_*` batch tensors.
- Subgoal 3: prove enabled conditioning changes logits and routes gradients to C3 conditioning parameters.

## Candidate Variants

- Variant A: pooled C3 token embedding added as a per-window decoder bias.
- Variant B: per-token C3 cross-attention memory.
- Variant C: cross-window-aware C3 packing before model input.
- Variant D: train a full model immediately with exact C3 sidecar.

## Local Verification Matrix

- Variant A: smallest model-surface change; directly tests trainable conditioning without solving cross-window `REF` semantics.
- Variant B: more expressive, but introduces a new attention memory and decode-cache questions before any signal exists.
- Variant C: addresses P5 cross-window concern, but is data-packing work and still needs a model consumer.
- Variant D: premature because there is not yet a model path that consumes C3 tokens.

## Selected Variant

- Selected: Variant A, pooled side-stream embedding with a gated additive projection.
- Rejected: B is too large for the first conditioning gate; C is a follow-up if pooled conditioning is weak or semantically unclear; D skips the integration proof.
- Why this is the smallest useful test: existing dataset and collate code already emit padded C3 token tensors, so the model can consume them without changing sidecar schema or mapper target tokenization.

## Selection Pressure

- Primary pressure: enabled C3 conditioning changes finite logits and gradients reach C3 embedding/projection parameters.
- Guard pressure: disabled model output matches current behavior and existing mapper tests pass.
- Runtime pressure: synthetic unit tests only; no full training run.
- Kill pressure: stop if conditioning requires changing target tokenization, default mapper sample schema, or incremental decode behavior.

## Research Question

Can exact C3 side-stream tokens enter Mapper v2.1 as a trainable auxiliary feature behind an opt-in flag, despite unresolved cross-window `REF` semantics?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary token feature embeddings, side-channel conditioning, prompt/prefix-style pooled metadata, retrieval-code features.
- Relevant taxonomy bucket: representation engineering integration.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is engineering integration of a verified C3 representation, not a new modeling claim.

## Minimal Change

Add Mapper v2.1 config fields:

- `use_c3_side_stream_conditioning=False`,
- `c3_side_stream_vocab_size=0`,
- `c3_side_stream_embedding_dim=64`,
- `c3_side_stream_scale_init=0.03`.

When enabled, validate and embed `c3_side_stream_tokens`, masked-average over available sidecar tokens, project to `d_model`, scale with a trainable scalar gate, and add the result to decoder hidden states before output logits/adapters consume them.

## Files Likely to Change

- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `tests/models/mapper/v2_1/test_model.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_probe_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `src/pulsefield_model/training/mapper_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_result_report.md`

## Dataset Slice

Synthetic mapper v2.1 windows with synthetic C3 side-stream token IDs. No real-cache sidecar is required for model-unit verification.

## Baseline / Comparator

Baseline is Mapper v2.1 with C3 batch tensors present but ignored. Comparator is Mapper v2.1 with `use_c3_side_stream_conditioning=True`.

## Primary Metric

- Disabled parity: logits match between disabled model runs with and without C3 tensors.
- Enabled influence: logits change when C3 tokens change under the same mapper inputs.
- Gradient flow: C3 conditioning embedding/projection/gate parameters receive finite gradients.

## Secondary Metric

- Shape validation catches invalid C3 token IDs and bad masks.
- Existing forward/loss and incremental decode tests still pass.
- Training config parsing accepts the new model fields through the existing dataclass-key path.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- `use_c3_side_stream_conditioning=False` remains the default.
- Mapper datasets can still carry C3 tensors without requiring model conditioning.
- Incremental decode remains unchanged and does not accept C3 side-stream tensors in this card.
- No full training-quality claim is made.

## Qualitative Check

Inspect model config and tests. The C3 path should be clearly optional, pooled, and additive rather than a replacement for mapper target tokens.

## Positive Signal

- Disabled parity passes.
- Enabled C3 token changes alter logits.
- C3 conditioning parameters receive gradients.
- Existing mapper/C3 tests pass.

## Negative Signal

- Default logits change.
- C3 tensors are silently ignored even when enabled.
- Gradient does not reach C3 conditioning parameters.
- Validation accepts out-of-range C3 token IDs.

## Kill Criteria

Kill this conditioning variant if it changes default mapper behavior, requires target-token changes, breaks existing training tests, or cannot route gradients from the mapper loss into C3 conditioning parameters.

## Expected Failure Modes

- Very small conditioning scale can make enabled-vs-disabled logit differences hard to observe.
- Token IDs from sidecar exceed configured C3 vocab size.
- Padding or unavailable-window masks leak pad embeddings into the pooled feature.
- Incremental decode parity expectations need to remain untouched because C3 conditioning is not implemented for inference yet.

## Confounders

- Synthetic tests prove plumbing and gradient flow, not model quality.
- Pooled conditioning ignores `REF` span semantics and cross-window dependencies.
- A positive result still needs a small training/eval run before claiming usefulness.

## Expected Runtime / Runtime Budget

Focused tests should complete in seconds. Stop before any real training run in this card.

## Result Interpretation Plan

- Positive result would suggest: run a tiny opt-in training smoke with the exact P5 sidecar and compare no-C3 vs C3 loss traces.
- Negative result would suggest: repair validation/gradient plumbing or mutate to cross-attention memory.
- Ambiguous result would require: add a deterministic synthetic task where C3 tokens carry a known signal.
- Human owner decides: whether pooled C3 conditioning is the first training variant worth spending runtime on.
- Next-loop action if positive: tiny training smoke with exact C3 sidecar.
- Next-loop action if negative: mutate conditioning architecture.
- Next-loop action if ambiguous: add a controlled synthetic learning probe.

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
- Remaining ambiguity: model utility remains deferred until a tiny training/eval smoke.

## Next-Loop Action

- If positive: run tiny exact-sidecar training smoke.
- If negative: mutate model conditioning.
- If ambiguous: add synthetic learning probe.

## Novelty Notes

- Closest analogies: auxiliary side-channel embeddings and pooled feature conditioning.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering integration of C3 into mapper model inputs.
