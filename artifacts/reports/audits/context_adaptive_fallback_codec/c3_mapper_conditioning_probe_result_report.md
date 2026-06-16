# C3 Mapper Conditioning Probe Result Report

## Scope

This P6 pass implements the first model-side C3 consumption path for Mapper v2.1.

The path is intentionally minimal:

- disabled by default,
- consumes the existing `c3_side_stream_*` batch tensors,
- masked-averages C3 side-stream token embeddings per mapper window,
- projects the pooled side-stream feature to `d_model`,
- adds it to decoder hidden states before logits and LN-close adapters consume them.

This does not decode C3 `REF`/`RES` semantics. It tests whether the exact C3 sidecar can become a trainable auxiliary conditioning signal in the full mapper pipeline.

## Implemented Config

Added to `MapperV21Config`:

- `use_c3_side_stream_conditioning=False`,
- `c3_side_stream_vocab_size=0`,
- `c3_side_stream_embedding_dim=64`,
- `c3_side_stream_scale_init=0.03`.

When enabled, `c3_side_stream_vocab_size` is interpreted as the max non-pad sidecar token id. The model allocates one additional pad slot at id `0`.

## Result

Decision: TEST_NEXT.

P6 passes as a model-integration probe. C3 can now enter Mapper v2.1 as an opt-in trainable side-stream feature without changing default mapper behavior.

| Gate | Result |
| --- | ---: |
| Default behavior unchanged | pass |
| Enabled C3 tokens influence logits | pass |
| C3 conditioning receives gradients | pass |
| Out-of-range C3 token validation | pass |
| Existing training boundary tests | pass |
| Broader C3/mapper verifier suite | 44 passed |

## What Passed

- With `use_c3_side_stream_conditioning=False`, C3 tensors are ignored and logits match the no-C3 path.
- With conditioning enabled, changing C3 side-stream token ids changes finite base logits under identical mapper inputs.
- Gradients reach:
  - `c3_side_stream_embedding`,
  - `c3_side_stream_projection`,
  - `c3_side_stream_scale`.
- Invalid C3 token ids are rejected instead of silently wrapping or indexing out of range.
- Existing dataset/training config plumbing remains compatible with C3 sidecar tensors.
- Incremental decode remains unchanged; enabled C3 conditioning explicitly rejects incremental decode for now.

## What This Proves

P6 proves the full pipeline has a real model consumption path:

1. exact C3 sidecar generation exists,
2. mapper dataset/collate can carry C3 token tensors,
3. training batches can include those tensors,
4. Mapper v2.1 can condition logits on them,
5. the loss can backpropagate into C3 conditioning parameters.

This is the first point where C3 is no longer only an audit artifact or ignored sidecar tensor.

## What This Does Not Prove

- It does not prove C3 improves validation loss or generation quality.
- It does not solve cross-window `REF`/`RES` semantics.
- It does not implement C3-conditioned incremental decoding.
- It does not tune the C3 embedding dimension, scale, pooling, or cap.

## Verification

```bash
uv run python -m py_compile src/pulsefield_model/models/mapper/v2_1/model.py tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

Observed:

- focused model/training slice: 15 passed,
- broader C3/mapper verifier suite: 44 passed.

## Interpretation

This is a positive integration result, not a training-result claim.

The next bounded experiment should run a tiny opt-in training smoke using the exact P5 sidecar:

- baseline: Mapper v2.1 with C3 tensors carried but conditioning disabled,
- variant: Mapper v2.1 with pooled C3 conditioning enabled,
- metric: short-run train/eval loss trace plus no-regression guards,
- kill criteria: config/runtime instability, missing sidecar coverage, or worse-than-baseline loss on the tiny slice.

If the tiny smoke is positive, then C3 is ready for a longer controlled training comparison. If negative, mutate the conditioning path toward cross-window-aware packing or C3 cross-attention memory.
