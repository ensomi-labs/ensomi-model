# C3 Mapper-Window Sidecar Generation Result Report

## Scope

This pass implemented P3 from `c3_mapper_window_sidecar_generation_experiment_card.md`: generate a real mapper-window C3 side-stream sidecar from the full P0-selected C3 plan.

The generated sidecar uses mapper-compatible keys:

- `beatmap_path`: `dataset_root/shard/beatmap_path`
- `window_start_ms`: 8s mapper write-window start
- `token_ids`: positive C3 side-stream token IDs, with pad id `0`

This pass does not change mapper model conditioning.

## Full-Cache Result

- full cache: `true`
- runtime: `188.754s`
- sidecar path: `artifacts/cache/c3_mapper_window_sidecar/c3_mapper_window_sidecar_le3.json`
- sidecar size: `73M`
- sidecar rows/windows: `173,149`
- loaded beatmaps: `9,242`
- token vocab size: `14,294`
- traced side-stream tokens: `3,203,904`
- sidecar token count: `3,203,904`
- token preservation: pass
- missing anchor tokens: `0`
- loader guard: pass
- reconstruction guard: pass

## Window Length Distribution

All observed source windows:

- mean tokens/window: `18.504`
- p50: `12`
- p90: `43`
- p95: `61`
- p99: `101`
- max: `268`

Nonzero-token windows:

- count: `166,125`
- mean tokens/window: `19.286`
- p50: `12`
- p90: `44`
- p95: `62`
- p99: `103`
- max: `268`

Cap sweep:

- cap `64`: `7,446` windows truncated, `4.300%` of windows, `5.996%` token overflow
- cap `128`: `561` windows truncated, `0.324%` of windows, `0.461%` token overflow
- cap `256`: `1` window truncated, `0.00058%` of windows, `12` overflow tokens
- cap `512`: `0` windows truncated

The default P2 cap of `256` is nearly sufficient by length.

## What Passed

- The generated sidecar preserves all traced P0 side-stream tokens exactly.
- The sidecar loads through `load_c3_side_stream_token_sidecar`.
- Mapper-compatible resolved paths are emitted.
- Empty-token rows are included for observed windows, so availability can be distinguished from missing sidecar.
- The sidecar length distribution is tractable under the current default cap.

## What Surfaced

The selected P3 projection uses `chunk_sort_ms`, the earliest snapped event time in each 2-beat chunk, as the mapper-window anchor. That is not exact group timing.

Boundary-risk metrics are high:

- boundary-risk token count: `792,701`
- boundary-risk token rate: `24.742%`
- target cross-mapper-window span rate: `9.428%`
- reference cross-mapper-window span rate: `9.154%`

This means the generated sidecar is valid as a real artifact and distribution audit, but should not yet be treated as final training input. Exact per-group timing should be compared on a bounded real-cache slice before model conditioning.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
uv run python -m py_compile src/pulsefield_model/osu_core/c3_side_stream_tokenization.py tests/osu_core/test_c3_side_stream_tokenization.py
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --limit-chunks 2000 --sidecar-path artifacts/cache/c3_mapper_window_sidecar/smoke_c3_mapper_window_sidecar.json --report-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_mapper_window_sidecar_report.json --result-log-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_mapper_window_sidecar_result_log.md
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar
```

Observed:

- focused P3/P2 tests: `9 passed in 0.74s`
- combined C3/mapper tests: `29 passed in 1.01s`
- broader focused audit suite: `40 passed in 1.04s`
- smoke sidecar: pass, `2,876` tokens, `290` windows
- full sidecar: pass, `3,203,904` tokens, `173,149` windows

## Result Interpretation

P3 passes the artifact-generation and tractability gates, but fails promotion pressure for final model input because chunk-sort anchoring risk is too high.

Recommended next experiment:

`MUTATE`: implement an exact per-group timing comparison on a bounded source slice and measure how often chunk-sort window assignment differs from exact group-time assignment. If mismatch is low, keep the current sidecar builder; if mismatch is high, replace the P3 anchoring with exact timing before model conditioning.
