# C3 Mapper Shadow Fields Result Report

## Scope

This pass implemented P1a from `c3_mapper_shadow_fields_experiment_card.md`: expose C3 side-stream information to the mapper v2.1 dataset as disabled-by-default shadow metadata.

It does not change mapper model inputs, default mapper sample keys, tensor shapes, record filtering, or training code.

## Implemented Contract

`MapperV21WindowDataset` now accepts:

- `include_c3_side_stream_metadata=False`
- `c3_side_stream_metadata_by_beatmap_path=None`
- `c3_side_stream_contract="r0_delta_main_plus_c3_fallback_side_stream_v1"`

When disabled, mapper sample metadata is unchanged and contains no `c3_side_stream` key.

When enabled, each sample metadata object includes:

- `enabled`
- `contract`
- `beatmap_path`
- `window_start_ms`
- `window_end_ms`
- `summary_available`
- `summary`

The summary is an optional caller-provided per-beatmap payload. The dataset does not require the full C3 cache to instantiate ordinary mapper samples.

## What Passed

- Disabled-path mapper tests pass.
- Enabled mapper sample emits C3 metadata.
- Collation preserves the nested C3 metadata in the existing metadata list.
- No model/training code was changed.
- Combined focused C3/P0/P1 mapper suite passes.

## What Surfaced

P1a is only a visibility bridge. It proves that the full pipeline can carry C3 side-stream metadata without disturbing mapper defaults, but it does not yet feed C3 tokens or tensors into the model.

The remaining design question is how to pack side-stream information for learning:

- keep it as a sidecar/stateful stream,
- tensorize bounded per-window summaries,
- or build a compact grammar extension over mapper tokens.

The P0 token count result still matters here: explicit main+side tokenization is `1.566805x` the baseline encoded token count, so naive token concatenation is not the right default.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Observed:

- mapper focused suite: `8 passed in 0.62s`
- combined focused suite: `19 passed in 0.80s`

## Result Interpretation

P1a passes. C3 can now be surfaced inside mapper dataset samples as opt-in shadow metadata. This is enough to design the next mapper-side probe, but not enough to claim training benefit.

Recommended next experiment:

`TEST_NEXT`: define a mapper probe card for bounded C3 side-stream tensors or sidecar lookup, with a sequence-length budget and an explicit no-regression gate for default mapper training.
