# C3 Exact Window Assignment Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P3 generated a real mapper-window sidecar, but surfaced high `chunk_sort_ms` boundary-risk.
- Acceptance source, if any: active thread goal plus P3 result report.
- Source snapshot / evidence grade: strong local evidence for sidecar tractability; unresolved evidence for window assignment correctness.

## Hypothesis

For a bounded high-boundary-risk real-map slice, exact per-group snapped-time window assignment will either validate the current `chunk_sort_ms` anchor or show that P3 must be replaced with exact timing before model conditioning.

## Root Objective

Resolve the remaining timing-alignment risk blocking use of C3 side-stream tokenization as full-pipeline mapper input.

## Goal Decomposition

- Subgoal 1: select a bounded source slice biased toward P3 boundary-risk cases.
- Subgoal 2: rebuild exact group snapped times for those sources by reparsing beatmaps.
- Subgoal 3: compare exact group-time mapper windows against current chunk-sort mapper windows for traced C3 side tokens.

## Candidate Variants

- Variant A: compare chunk-sort vs exact windows for all chunk rows only.
- Variant B: compare chunk-sort vs exact windows for traced C3 side tokens on a high-risk source slice.
- Variant C: rebuild exact sidecar for the full cache.
- Variant D: accept P3 sidecar without exact timing comparison.

## Local Verification Matrix

- Variant A: fastest, but it measures chunks rather than side-stream token assignments.
- Variant B: directly measures the P3 failure mode and remains bounded.
- Variant C: highest fidelity, but too expensive before knowing whether mismatch is material.
- Variant D: unsafe because P3 boundary-risk token rate was `24.742%`.

## Selected Variant

- Selected: Variant B, traced C3 token comparison on a high-risk bounded source slice.
- Rejected: A is indirect; C is too large for this decision point; D ignores the surfaced risk.
- Why this is the smallest useful test: it compares exactly the token-to-window assignment needed by the mapper sidecar, while limiting reparsing to a small real-map slice.

## Selection Pressure

- Primary pressure: exact-vs-chunk-sort C3 token window mismatch rate.
- Guard pressure: existing P0/P2/P3 tests keep passing.
- Runtime pressure: source-limit defaults to a bounded value; no full-cache reparsing.
- Kill pressure: stop if exact group timing cannot be reconstructed with the existing beat representation path.

## Research Question

Is `chunk_sort_ms` anchoring accurate enough for mapper-window C3 side-stream tokens, or must P3 sidecar generation use exact per-group snapped times?

## Closest Analogies / Novelty Layer

- Closest analogies: alignment audits, time-index sidecar validation, auxiliary sequence windowing checks.
- Relevant taxonomy bucket: representation engineering validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is an engineering validation of timing alignment for the C3 representation.

## Minimal Change

Add a P4 audit mode to `c3_side_stream_tokenization.py`:

- select high-risk sources from the chunk cache,
- compute exact group snapped-time anchors by reparsing selected beatmaps,
- compare exact vs chunk-sort mapper window assignment for traced C3 side tokens,
- write JSON/markdown reports.

Do not change mapper model architecture.

## Files Likely to Change

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `tests/osu_core/test_c3_side_stream_tokenization.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_window_assignment_comparison_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/osu_core/beat_chunk_pattern_audit.py`
- `src/pulsefield_model/osu_core/beat_representation.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_window_sidecar_generation_result_report.md`

## Dataset Slice

Use synthetic exact-timing unit tests first. Then run a real-cache high-boundary-risk slice, default `source_limit=64`.

## Baseline / Comparator

Baseline is P3 `chunk_sort_ms` assignment. Comparator is exact per-group snapped-time assignment from reparsed beatmaps.

## Primary Metric

C3 side-stream token window mismatch rate:

- `mismatched_token_count / compared_token_count`

## Secondary Metric

- compared source count,
- parse error count,
- missing exact group count,
- mismatch counts by token kind,
- fallback-record mismatch rate,
- examples of mismatched assignments.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --exact-window-compare --source-limit 64
```

## Guard Check

- Existing P0 audit mode still works.
- Existing P3 mapper-sidecar mode still works.
- The exact comparison reports parse/missing-anchor failures instead of silently dropping them.
- The source slice records which source rows were selected and why.

## Qualitative Check

Inspect mismatch examples. They should identify beatmap path, source row, record id, chunk window, exact window, chunk time, and exact time.

## Positive Signal

- Parse errors are zero or rare.
- Missing exact group count is zero or rare.
- Exact-vs-chunk side-token mismatch rate is low enough to keep P3 anchoring.

## Negative Signal

- Mismatch rate is material on the high-risk slice.
- Mismatches cluster around 8s boundaries, confirming the P3 caveat.
- Exact group lookup cannot reconstruct many records.

## Kill Criteria

Kill current P3 anchoring if exact comparison shows material mismatch, using `1%` token mismatch as the default fail threshold for this bounded high-risk slice.

## Expected Failure Modes

- Reparsed timing path differs from the chunk-cache build config.
- Segment/group keys do not align with `_FallbackRecord` keys.
- The high-risk slice overestimates full-cache mismatch.
- Low-risk sources may still need a later confirmation slice.

## Confounders

This is intentionally biased toward high boundary risk. A high mismatch rate is actionable; a low mismatch rate should still be confirmed on a broader random slice before final model conditioning.

## Expected Runtime / Runtime Budget

Unit tests should finish in seconds. Real-cache `source_limit=64` comparison should finish under 2 minutes.

## Result Interpretation Plan

- Positive result would suggest: keep P3 chunk-sort anchoring and proceed to cap/model probe after a random-slice confirmation.
- Negative result would suggest: mutate P3 to exact per-group timing sidecar before model conditioning.
- Ambiguous result would require: compare high-risk and random slices separately.
- Human owner decides: whether the mismatch rate is acceptable for model input.
- Next-loop action if positive: run random-slice confirmation or model-conditioning probe.
- Next-loop action if negative: implement exact-timing sidecar generation.
- Next-loop action if ambiguous: add stratified high-risk/random comparison.

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
- Remaining ambiguity: model utility remains deferred.

## Next-Loop Action

- If positive: confirm on random slice or start model-conditioning probe.
- If negative: replace P3 sidecar anchoring with exact per-group timing.
- If ambiguous: run stratified comparison.

## Novelty Notes

- Closest analogies: time-window alignment audits for auxiliary sidecar features.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering validation of C3 side-stream timing alignment.
