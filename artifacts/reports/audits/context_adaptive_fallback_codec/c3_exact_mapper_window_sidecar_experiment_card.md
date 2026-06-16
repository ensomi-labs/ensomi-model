# C3 Exact Mapper-Window Sidecar Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P4 rejected P3 `chunk_sort_ms` window anchoring for C3 side-stream tokens.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_exact_window_assignment_comparison_result_report.md`.
- Source snapshot / evidence grade: strong local negative evidence for chunk-sort anchoring; strong local evidence that exact group anchors can be reconstructed on a high-risk slice.

## Hypothesis

Replacing P3 chunk-sort sidecar window assignment with exact per-group snapped-time assignment will preserve all traced C3 side-stream tokens, keep loader compatibility, and keep per-window token lengths tractable enough to remain a candidate full-pipeline sidecar.

## Root Objective

Remove the timing-alignment blocker that prevents C3 side-stream tokenization from being used as mapper-side full-pipeline input.

## Goal Decomposition

- Subgoal 1: build mapper-window sidecar rows using exact group snapped times instead of chunk timestamps.
- Subgoal 2: verify reconstruction, token preservation, exact-anchor coverage, and loader compatibility on synthetic and real-cache runs.
- Subgoal 3: compare exact-timing sidecar length/cap distribution against the prior P3 distribution before model conditioning.

## Candidate Variants

- Variant A: keep P3 sidecar code and post-correct windows after JSON generation.
- Variant B: add an exact anchor mode inside the existing sidecar builder and make exact group timing the mapper-side generation path.
- Variant C: create a separate standalone exact-sidecar builder that duplicates P3 generation.
- Variant D: skip sidecar regeneration and move directly to model conditioning with P4 warnings.

## Local Verification Matrix

- Variant A: easy to bolt on, but risks preserving chunk-owned empty windows and hiding missing exact anchors.
- Variant B: minimally invasive and directly fixes the anchor lookup where tokens are assigned.
- Variant C: isolates risk, but duplicates tokenization and reporting logic.
- Variant D: rejected because P4 showed 6.797% high-risk token-window mismatch.

## Selected Variant

- Selected: Variant B, exact anchor mode inside the existing mapper-window sidecar audit.
- Rejected: A is too indirect; C creates unnecessary duplicate logic; D ignores a failed timing gate.
- Why this is the smallest useful test: the C3 codec, sidecar schema, loader, and reports already exist; only the window-anchor source needs to change.

## Selection Pressure

- Primary pressure: exact sidecar preserves all traced side-stream tokens with zero missing exact anchors.
- Guard pressure: existing C3, P4 comparison, mapper sidecar tensor, and mapper training tests keep passing.
- Runtime pressure: smoke run should finish in seconds; full-cache run should stay in P3-like runtime.
- Kill pressure: stop if exact beatmap reparsing cannot cover the real-cache sidecar or if cap distribution becomes materially worse.

## Research Question

Can C3 be materialized as a mapper-window sidecar using exact per-group timing, without losing token preservation or tractable sequence lengths?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary token sidecars, alignment-audited sequence features, time-indexed residual streams.
- Relevant taxonomy bucket: representation engineering validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is engineering hardening of a C3 side-stream representation, not a novelty claim.

## Minimal Change

Extend `c3_side_stream_tokenization.py` so `--mapper-sidecar` can use `exact_group` window anchors:

- parse selected beatmaps with the existing beat representation path,
- build anchors keyed by `(source_row_index, segment_id, beat_offset_numerator)`,
- assign traced C3 side tokens by each fallback record's `absolute_units`,
- report exact anchor parse/missing counts,
- retain the existing sidecar JSON schema and loader contract.

## Files Likely to Change

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `tests/osu_core/test_c3_side_stream_tokenization.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_window_assignment_comparison_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_window_sidecar_generation_result_report.md`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`

## Dataset Slice

Use synthetic unit tests first, then run:

- smoke real-cache sidecar with `--limit-chunks 2000`,
- full real-cache sidecar if smoke passes.

## Baseline / Comparator

Baseline is P3 chunk-sort sidecar generation. Comparator is exact per-group snapped-time sidecar generation under the same C3 tokenization plan and JSON loader contract.

## Primary Metric

Exact sidecar generation pass:

- reconstruction pass,
- sidecar token preservation pass,
- missing exact anchor token count is 0,
- parse error count is 0,
- loader guard pass.

## Secondary Metric

- sidecar token count,
- window count,
- windows with tokens,
- token vocab size,
- tokens/window p95, p99, max,
- cap 256 truncation rate and overflow token count,
- target/reference cross-window span rates under exact anchors,
- exact anchor count and parsed source count.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --window-anchor-mode exact_group --limit-chunks 2000 --sidecar-path artifacts/cache/c3_mapper_window_sidecar/smoke_c3_exact_mapper_window_sidecar.json --report-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_exact_mapper_window_sidecar_report.json --result-log-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_exact_mapper_window_sidecar_result_log.md
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --window-anchor-mode exact_group
```

## Guard Check

- P0 C3 side-stream audit still roundtrips.
- P4 exact comparison test still detects chunk-sort mismatch.
- The sidecar loader still reads the generated JSON into per-beatmap/per-window token lists.
- Default mapper training remains disabled with respect to C3 model conditioning.

## Qualitative Check

Inspect sample sidecar rows and the result report. The rows should keep the same loader contract while the report explicitly says window assignment uses exact group snapped time.

## Positive Signal

- Full-cache exact sidecar generation passes.
- Missing exact anchors and parse errors are 0.
- Cap 256 truncation remains rare enough to continue toward a model-conditioning probe.
- The generated artifact remains loader-compatible.

## Negative Signal

- Exact anchors cannot be built for many sources.
- Side-stream tokens fail to preserve exactly.
- Per-window lengths become materially worse than P3.
- Loader compatibility breaks.

## Kill Criteria

Kill exact sidecar promotion if full-cache generation has parse errors, missing exact anchor tokens, token preservation failure, reconstruction failure, loader failure, or cap 256 truncation greater than 5% of windows.

## Expected Failure Modes

- Beatmap files unavailable or path resolution mismatch.
- Beat representation path differs from chunk-cache generation enough to miss group anchors.
- Exact grouping creates additional windows and changes cap distribution.
- Sorting or token-id assignment remains stable but row distribution changes.

## Confounders

- Exact sidecar JSON may be large and ignored locally, so committed reports are more important than the full artifact.
- Smoke `limit_chunks` can truncate sources and overstate parse coverage; full-cache run is required before model-conditioning claims.
- Exact timing may reduce boundary-risk semantics but increase cross-window references because the representation itself can span windows.

## Expected Runtime / Runtime Budget

- Unit tests: seconds.
- Smoke sidecar: under 1 minute.
- Full exact sidecar: expected under 10 minutes; stop if reparsing is clearly pathological.

## Result Interpretation Plan

- Positive result would suggest: exact-timing C3 sidecar is a valid full-pipeline artifact candidate; next design a disabled-by-default model-conditioning probe.
- Negative result would suggest: mutate sidecar construction or grouping before model work.
- Ambiguous result would require: stratified source-slice diagnostics for parse/missing-anchor or cap failures.
- Human owner decides: whether to accept exact sidecar distribution as good enough for model input.
- Next-loop action if positive: implement a disabled-by-default mapper conditioning probe.
- Next-loop action if negative: inspect exact-anchor failures or sidecar packing/cap alternatives.
- Next-loop action if ambiguous: run focused diagnostics on failed source classes.

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
- Remaining ambiguity: model utility remains deferred until exact sidecar generation passes.

## Next-Loop Action

- If positive: design disabled-by-default C3 model conditioning.
- If negative: repair exact anchoring or sidecar packing.
- If ambiguous: run source-class diagnostics.

## Novelty Notes

- Closest analogies: auxiliary token sidecars and time-aligned sequence features.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering validation and hardening of C3 mapper-side use.
