# C3 Side-Stream Pipeline Tokenization Experiment Card

## Hypothesis

The hardened `r0_delta + C3 skeleton-residual fallback side-stream` codec can be exposed as a lossless pipeline-facing beatmap tokenization artifact without changing mapper training semantics yet. If the artifact round-trips and keeps sequence/runtime overhead bounded, it becomes a safe candidate for mapper-side probes.

## Root Objective

Turn the hardened C3 result into an optional full-pipeline tokenization interface that preserves exact beat-chunk and mapper-timepoint reconstruction.

## Goal Decomposition

- Define a two-stream token contract: unchanged motif/main stream plus C3 fallback side-stream.
- Encode beat-chunk cache rows into the two-stream representation.
- Decode the two-stream representation back to beat-chunk group tokens.
- Convert decoded groups back through mapper timepoint-compatible structure where possible.
- Measure sequence length, side-stream length, span usage, cross-chunk references, and per-window dataset impact.

## Candidate Variants

- P0 artifact-only encoder/decoder: no mapper dataset change, just cache-level two-stream tokenization.
- P1 dataset shadow fields: add optional side-stream tensors/statistics to mapper samples behind a flag.
- P2 flat-token approximation: emit C3 spans in the main stream only when spans are main-stream contiguous.
- P3 grammar-extension decoder: main stream carries fallback-read markers that consume C3 side-stream tokens.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| P0 | full cache encode/decode round-trip has zero group-token mismatches | Required before any mapper-facing work. |
| P1 | existing mapper dataset tests pass with feature disabled and optional fields enabled in smoke | If disabled path changes, reject. |
| P2 | retains most C3 gain while staying flat-contiguous | If weak, do not force flat-token replacement. |
| P3 | reconstructs noncontiguous/cross-chunk spans with bounded state | If too complex, keep C3 as offline codec only. |

## Selected Variant

Start with P0 artifact-only encoder/decoder. Do not change default mapper training or inference code in this card.

## Selection Pressure

P0 is selected because hardening showed `199,906` selected spans are noncontiguous in the main group stream and `167,597` selected spans cross chunk boundaries. A flat-token mapper change would be premature.

## Minimal Change

Add a small optional C3 side-stream tokenization artifact module and tests. Reuse the hardened C3 plan builder and reconstruction guards. Do not alter existing mapper dataset defaults.

## Files Likely To Change

- `src/pulsefield_model/osu_core/context_adaptive_fallback_codec_audit.py`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `tests/osu_core/test_c3_side_stream_tokenization.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_result_log.md`

## Dataset Slice

Use synthetic smoke rows first, then the full LE<=3 beat-chunk cache with `limit_chunks=None`.

## Baseline / Comparator

Baseline is the unchanged `r0_delta` beat-chunk token stream and current mapper v2.1 tokenizer behavior. This card should not claim model-quality improvement; it only validates pipeline representation readiness.

## Primary Metric

Lossless reconstruction mismatch count.

## Secondary Metric

Main-stream token count, side-stream token count, selected span count, cross-chunk span count, noncontiguous span count, encode/decode runtime, mapper-window sequence length impact, and disabled-path mapper dataset test status.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Beat-chunk group reconstruction mismatches must be `0`.
- Mapper timepoint reconstruction mismatches must be `0` for supported rows.
- Existing mapper dataset defaults must remain unchanged.
- C3 references must use only prior same-chart fallback side-stream records.
- Noncontiguous and cross-chunk spans must be represented explicitly, not hidden as flat contiguous tokens.

## Qualitative Check

Inspect examples of noncontiguous spans, cross-chunk spans, clean-trace maps, dirty-trace maps, and flat-contiguous-only spans. Confirm the token contract is understandable enough for a mapper-side probe.

## Positive Signal

- Full-cache P0 round-trip has zero mismatches.
- Disabled mapper path remains unchanged.
- Optional side-stream artifact exposes bounded sequence/runtime statistics.
- P2 flat-contiguous-only statistics show whether a flat token approximation is viable.

## Negative Signal

- Any round-trip mismatch.
- Side-stream decode requires future context.
- Mapper defaults change.
- Noncontiguous/cross-chunk state is too complex to batch or window.

## Kill Criteria

Kill mapper-side promotion if P0 cannot round-trip, if side-stream references are not locally decodable, or if full-cache runtime is not practical after basic optimization.

## Expected Failure Modes

- Windowed mapper samples lack enough prior fallback history for references crossing window starts.
- Cross-chunk spans require state not currently exposed in dataset samples.
- Flat-contiguous approximation loses most of the C3 gain.
- Optional side-stream tensors complicate collation more than expected.

## Expected Runtime / Runtime Budget

Smoke tests should run in seconds. Full-cache artifact generation should target less than 10 minutes.

## Confounders

The hardening result is a compression/representation result, not yet a learned mapper-quality result. Pipeline readiness must be judged by round-trip and dataset ergonomics first.

## Result Interpretation Plan

- If P0 passes, proceed to P1 dataset shadow fields.
- If P1 passes, design a mapper-side probe.
- If P2 is weak, do not pursue flat token replacement.
- If P3 is required, treat C3 as a side-stream grammar extension, not a normal vocabulary expansion.

## Result Log Template

Record command, commit, dirty flag, runtime, full-cache flag, reconstruction counts, token counts, side-stream span stats, mapper disabled-path test status, and recommendation.

## Next-Loop Action

Implement P0 artifact-only encoder/decoder.
