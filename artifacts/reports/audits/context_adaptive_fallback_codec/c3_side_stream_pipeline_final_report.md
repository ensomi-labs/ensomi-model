# C3 Side-Stream Pipeline Tokenization Final Report

## Scope

This pass implemented P0 from `c3_side_stream_pipeline_experiment_card.md`: an artifact-only encoder/decoder for the hardened C3 fallback side stream. It does not change mapper dataset defaults, training, or inference.

The token contract is:

- main stream keeps `r0_delta` motif/control tokens;
- fallback atom payloads in the main stream become `F` placeholders;
- side stream emits `RAW`, `REF`, and `RES` tokens in fallback-substream order;
- selected C3 references use only prior same-chart fallback records.

## Full-Cache Result

- full cache: `true`
- runtime: `172.295s`
- selected C3 variant: `a2_skeleton_residual_all_fallback_w256`
- P0 round-trip: pass
- side-stream reference errors: `0`
- side-stream payload mismatches: `0`
- full token-stream mismatches: `0`
- group signature mismatches: `0`
- mapper-timepoint-compatible mismatches: `0`
- missing side payloads: `0`

## Token Statistics

- baseline encoded tokens: `5,652,564`
- main stream tokens: `5,652,564`
- fallback placeholders: `2,786,696`
- side stream tokens: `3,203,904`
- side stream raw tokens: `1,167,043`
- side stream ref tokens: `417,208`
- side stream residual tokens: `1,619,653`
- combined main+side tokens: `8,856,468`
- combined/main baseline ratio: `1.566805`

This is lossless and pipeline-facing, but it is not yet a compact learned sequence format. The next step should measure whether optional shadow fields can be batched ergonomically before changing model inputs.

## Span Statistics

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- mean span length: `3.882124`
- noncontiguous main-stream spans: `199,906`
- target cross-chunk spans: `167,597`
- reference cross-chunk spans: `168,631`

The side-stream shape remains necessary. A flat-token replacement would hide too much noncontiguous and cross-chunk state.

## What Passed

- P0 artifact-only side-stream encode/decode round-trips on full cache.
- The side stream explicitly represents noncontiguous and cross-chunk C3 spans.
- Current mapper defaults remain unchanged.
- Mapper disabled-path tests pass.

## What Surfaced

The representation is ready for optional pipeline shadow fields, not default model input replacement. The combined token count is `1.57x` the baseline encoded token count because P0 keeps the main stream placeholders and emits explicit side-stream residual payloads. This is acceptable for an artifact validation pass, but the mapper probe needs sequence packing or stateful side-channel design.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --limit-chunks 2000 --report-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_side_stream_pipeline_report.json --result-log-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_side_stream_pipeline_result_log.md
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Observed:

- focused P0/C3 tests: `11 passed`
- smoke P0 audit: pass, `limited=true`
- full P0 audit: pass, `limited=false`
- mapper disabled-path tests: `7 passed`
- combined focused suite: `18 passed`

## Recommendation

`TEST_NEXT`: implement optional mapper dataset shadow fields behind a disabled-by-default flag. Do not change mapper model inputs yet.
