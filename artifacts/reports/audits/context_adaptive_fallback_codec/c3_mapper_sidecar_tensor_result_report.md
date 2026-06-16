# C3 Mapper Sidecar Tensor Result Report

## Scope

This pass implemented P2a from `c3_mapper_sidecar_tensor_experiment_card.md`: carry precomputed C3 side-stream token IDs through the mapper v2.1 data and training boundary as disabled-by-default optional tensors.

It does not feed C3 into the model architecture and does not replace mapper target tokens.

## Implemented Contract

`MapperV21WindowDataset` now supports optional C3 side-stream token tensors:

- `include_c3_side_stream_token_tensors=False`
- `c3_side_stream_token_ids_by_beatmap_path=None`
- `c3_side_stream_max_tokens=256`

When enabled, each sample can include:

- `c3_side_stream_tokens`
- `c3_side_stream_token_mask`
- `c3_side_stream_available`
- `c3_side_stream_token_count`
- `c3_side_stream_truncated`

Collation pads variable-length C3 token sequences with pad id `0` and emits batch masks/counts. The training config can also point at a JSON sidecar through:

- `include_c3_side_stream_token_tensors`
- `c3_side_stream_token_sidecar_path`
- `c3_side_stream_max_tokens`

Providing a sidecar path enables the optional tensors automatically.

## What Passed

- Default mapper samples and batches do not contain C3 side-stream tensor fields.
- Enabled mapper samples expose bounded C3 side-stream token tensors.
- Collation pads variable-length C3 side-stream token sequences and preserves availability/count/truncation fields.
- JSON sidecar loading works for per-window `beatmap_path` plus `window_start_ms` keys.
- Training config parsing and CLI forwarding accept the C3 sidecar options.
- The shared mapper tensor mover recognizes the C3 side-stream tensor fields.
- Mapper v2.1 forward/loss still run with C3 sidecar tensors present and ignored.

## What Surfaced

P2a proves C3 can be carried as a batchable optional side-stream token artifact. It does not prove that the model should consume the side-stream, nor that the P0 full-cache side-stream has already been projected into mapper-window sidecar files.

The next unresolved issue is real-cache sidecar generation:

- map fallback-substream C3 tokens to mapper write windows,
- measure per-window token length and truncation rates,
- decide whether the max-token cap is acceptable,
- then design model conditioning only if the sidecar distribution is tractable.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
```

Observed:

- P2 focused suite: `17 passed in 0.77s`
- combined C3/mapper suite: `28 passed in 0.95s`
- broader focused audit suite: `39 passed in 0.99s`

## Result Interpretation

P2a passes. C3 now has a full-pipeline carriage path up to mapper data loading, batching, training config, tensor movement, and model/loss tolerance.

This is still not a training-improvement claim. The representation remains side-stream-shaped, and the next evidence needed is a real-cache mapper-window sidecar distribution audit.

Recommended next experiment:

`TEST_NEXT`: generate a real C3 mapper-window sidecar from the P0 side-stream artifacts and report per-window token length, availability, and truncation rates before adding model conditioning.
