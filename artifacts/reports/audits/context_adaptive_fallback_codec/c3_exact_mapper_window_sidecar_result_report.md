# C3 Exact Mapper-Window Sidecar Result Report

## Scope

This P5 pass implements the mutation required by P4: replace P3 `chunk_sort_ms` sidecar window assignment with exact per-group snapped-time assignment.

The generated sidecar keeps the existing loader contract:

- `beatmap_path`,
- `window_start_ms`,
- `token_ids`,
- same C3 token vocabulary and pad/id convention.

The semantic change is the window anchor:

- P3: `(source_row_index, segment_id, chunk_index)` using `chunk_sort_ms`,
- P5: `(source_row_index, segment_id, absolute_units)` using exact group snapped time reconstructed from the beatmap.

## Commands

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --window-anchor-mode exact_group --limit-chunks 2000 --sidecar-path artifacts/cache/c3_mapper_window_sidecar/smoke_c3_exact_mapper_window_sidecar.json --report-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_exact_mapper_window_sidecar_report.json --result-log-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_exact_mapper_window_sidecar_result_log.md
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --window-anchor-mode exact_group
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
```

## Result

Decision: P5 exact sidecar generation passes the hard correctness and loader gates, but surfaces a secondary modeling risk around cross-window reference spans.

| Check | Result |
| --- | ---: |
| P5 exact sidecar generation pass | true |
| Full cache pass | true |
| Window anchor mode | exact_group |
| Parsed sources | 9,242 / 9,242 |
| Parse errors | 0 |
| Exact group anchors | 9,719,273 |
| Missing anchor tokens | 0 |
| Reconstruction pass | true |
| Loader guard pass | true |
| Token preservation pass | true |
| Sidecar tokens | 3,203,904 |
| Traced side-stream tokens | 3,203,904 |
| Sidecar windows | 173,268 |
| Windows with tokens | 166,801 |
| Token vocab size | 14,294 |
| Runtime | 359.858 s |
| Sidecar artifact size | 73M |

## Cap Distribution

| Metric | P3 chunk-sort | P5 exact |
| --- | ---: | ---: |
| Sidecar tokens | 3,203,904 | 3,203,904 |
| Windows | 173,149 | 173,268 |
| Windows with tokens | 166,125 | 166,801 |
| Token vocab size | 14,294 | 14,294 |
| Tokens/window p95 | 61 | 60 |
| Tokens/window p99 | 101 | 101 |
| Tokens/window max | 268 | 274 |
| Cap 256 truncated windows | 1 | 1 |
| Cap 256 overflow tokens | 12 | 18 |

Exact timing does not materially worsen the cap profile. At cap 256, only 1 window truncates, with 18 overflow tokens.

## What Passed

- Exact beatmap reparsing covered the full cache: 9,242 parsed sources, 0 parse errors.
- Every traced C3 side-stream token found an exact group-time anchor: 0 missing anchor tokens.
- The C3 side stream still reconstructs exactly: 0 side reference errors, 0 side payload mismatches, 0 full token-stream mismatches.
- The sidecar loader reads the generated exact artifact: 173,268 loaded windows and 3,203,904 loaded tokens.
- The tensor/mapper guard tests still pass with the unchanged loader contract.
- P4's timing-alignment blocker is removed for sidecar generation.

## What Surfaced

Exact timing increases cross-window reference exposure:

| Metric | P3 chunk-sort | P5 exact |
| --- | ---: | ---: |
| Target cross-window span rate | 9.428% | 10.585% |
| Reference cross-window span rate | 9.154% | 10.351% |
| Target/reference same-window span rate | not primary | 31.961% |

This is not a correctness failure of the sidecar artifact. It is a model-conditioning risk: a per-window sidecar row can contain `REF` and `RES` tokens whose compression semantics depend on fallback records in another mapper window.

## Interpretation

P5 proves that exact-timing C3 sidecar generation is feasible at full-cache scale. The generated artifact is loader-compatible, token-preserving, and cap-tractable.

P5 does not prove that the mapper should consume the sidecar as-is. The remaining issue is no longer timing alignment; it is how to condition on C3 reference-style tokens when roughly 10% of LZ-style spans cross 8s mapper-window boundaries.

## Recommendation

Do not return to chunk-sort anchoring. Exact per-group timing should be the sidecar generation path.

Before model conditioning, run one bounded packing/conditioning decision experiment:

1. either keep exact per-window token rows and let the model learn `REF`/`RES` token embeddings as auxiliary features,
2. or add a cross-window-aware packing mode for `REF` spans before feeding C3 tokens to the mapper.

The next Experiment Card should decide between those two options with a small training or data-only probe, not a broad mapper rewrite.
