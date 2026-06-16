# C3 Mapper-Window Sidecar Generation Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: continue from P2 sidecar tensor support toward a real C3 mapper-window sidecar artifact.
- Acceptance source, if any: active thread goal plus P2 result report.
- Source snapshot / evidence grade: strong local evidence for C3 codec correctness and mapper tensor carriage; missing evidence for real-cache sidecar distribution.

## Hypothesis

The full-cache P0 C3 side-stream can be projected into mapper-compatible 8s window sidecar rows keyed by resolved mapper `beatmap_path` and `window_start_ms`, with bounded token lengths and explicit truncation/boundary-risk metrics.

## Root Objective

Produce a real mapper-window C3 sidecar artifact that can be consumed by the P2 mapper dataset path, before changing model conditioning.

## Goal Decomposition

- Subgoal 1: reuse the hardened P0 selected C3 plan to emit side-stream tokens with target-record anchors.
- Subgoal 2: map each target record to a mapper-compatible beatmap path and 8s `window_start_ms`.
- Subgoal 3: write a JSON sidecar using the P2 loader schema and report token length/truncation/coverage distributions.

## Candidate Variants

- Variant A: exact per-group time sidecar by reparsing every beatmap and reconstructing group times.
- Variant B: `chunk_sort_ms` anchored sidecar from the existing full chunk cache.
- Variant C: beat-unit sidecar keyed by source row and beat units instead of mapper windows.
- Variant D: no sidecar file, report-only distribution audit.

## Local Verification Matrix

- Variant A: highest fidelity, but reparses 9k+ maps and changes the runtime/scope too much for this pass.
- Variant B: mapper-compatible and fast enough because `chunk_sort_ms` already exists in the full cache; must report boundary precision caveat.
- Variant C: lower risk internally but not directly consumable by `MapperV21WindowDataset`.
- Variant D: too weak for the full-pipeline goal because P2 already needs a real sidecar path.

## Selected Variant

- Selected: Variant B, `chunk_sort_ms` anchored mapper-window sidecar.
- Rejected: Variant A is a follow-up if boundary risk is unacceptable; Variant C is not pipeline-compatible; Variant D is not enough progress.
- Why this is the smallest useful test: it produces a real sidecar consumable by the existing mapper dataset path without reparsing maps or changing model code.

## Selection Pressure

- Primary pressure: generated sidecar rows load through `load_c3_side_stream_token_sidecar` and contain the same total side-stream token count as the traced P0 side stream.
- Guard pressure: P0 reconstruction still passes; mapper P2 tests still pass.
- Runtime pressure: smoke should finish in seconds; full cache may take P0-like runtime but should avoid map reparsing.
- Kill pressure: stop if sidecar generation cannot preserve token counts, cannot emit mapper-compatible paths, or produces unacceptable truncation under the default cap.

## Research Question

Can the audited C3 side-stream be materialized as a real mapper-window sidecar artifact with tractable per-window token lengths?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary sequence sidecars, retrieval side channels, compressed residual streams, per-window feature sidecars.
- Relevant taxonomy bucket: representation engineering and pipeline instrumentation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering bridge from audited C3 representation into mapper-window artifacts.

## Minimal Change

Add a P3 audit/build function to `c3_side_stream_tokenization.py` that emits:

- JSON sidecar: `schema_version`, `contract`, `token_vocab`, and `windows`.
- JSON report and markdown result log.
- CLI flag to run sidecar generation separately from P0 audit.

Do not change mapper model architecture.

## Files Likely to Change

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `tests/osu_core/test_c3_side_stream_tokenization.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_window_sidecar_generation_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_sidecar_tensor_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_final_report.md`

## Dataset Slice

Use synthetic unit-test chunk rows for local tests, then run a limited smoke slice. If smoke passes, run full LE<=3 cache.

## Baseline / Comparator

Baseline is no real mapper-window C3 sidecar artifact. P2 only proved synthetic sidecar tensor carriage.

## Primary Metric

Sidecar token preservation:

- `sidecar_token_count == traced_side_stream_token_count`
- generated sidecar loads through the P2 loader.

## Secondary Metric

Window tractability:

- p50/p90/p95/p99/max sidecar tokens per window,
- default cap truncation rate at `256`,
- source/window coverage,
- boundary-risk proxy from chunks near 8s window boundaries.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar --limit-chunks 2000 --sidecar-path artifacts/cache/c3_mapper_window_sidecar/smoke_c3_mapper_window_sidecar.json --report-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_mapper_window_sidecar_report.json --result-log-path artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_mapper_window_sidecar_result_log.md
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --mapper-sidecar
```

## Guard Check

- Existing P0 audit path still works.
- P2 sidecar loader accepts generated sidecar.
- Generated `beatmap_path` values match mapper dataset resolution: `dataset_root/shard/beatmap_path`.
- Token IDs are positive and pad id remains `0`.
- Sidecar includes empty-token rows for observed source windows, so "available but zero tokens" is distinguishable from missing sidecar.

## Qualitative Check

Inspect generated sidecar sample rows and report. The artifact should look like an auxiliary side-stream, not a replacement target sequence.

## Positive Signal

- Sidecar generation passes smoke and full cache.
- Token count is preserved exactly.
- Full-cache default cap truncation is low enough to justify a model-side probe or at least a cap sweep.
- Mapper-compatible paths are emitted.

## Negative Signal

- Token count mismatch.
- P2 loader rejects the sidecar.
- Most windows truncate at the default cap.
- Boundary-risk proxy is too high to trust `chunk_sort_ms` anchoring.

## Kill Criteria

Kill this variant if generated sidecar rows cannot preserve all traced side-stream tokens, if mapper-compatible keys cannot be produced from the chunk cache, or if full-cache generation is too large/slow to be practical.

## Expected Failure Modes

- `chunk_sort_ms` anchoring can misassign records near 8s boundaries.
- Side-stream reference spans can cross mapper windows, so per-window sidecar rows are not independently decodable.
- JSON sidecar may be large because P0 emits explicit RAW/REF/RES tokens.
- Limited `--limit-chunks` slices may omit later context and understate full-cache window lengths.

## Confounders

This pass uses chunk-level earliest event time, not exact group time. A positive result proves pipeline artifact generation and tractability, not final exact timing alignment.

## Expected Runtime / Runtime Budget

- Unit tests: seconds.
- Smoke sidecar: under 30 seconds.
- Full sidecar: P0-like runtime, expected under 5 minutes.

## Result Interpretation Plan

- Positive result would suggest: C3 can be materialized as a real mapper-window sidecar and the next step can be cap sweep or model conditioning.
- Negative result would suggest: either exact time reconstruction or a different sidecar packing is needed before model work.
- Ambiguous result would require: exact sidecar generation on a smaller real-map slice.
- Human owner decides: whether `chunk_sort_ms` anchoring is acceptable or exact reparsing is required.
- Next-loop action if positive: add a small model-conditioning probe or cap sweep.
- Next-loop action if negative: mutate to exact per-group time sidecar.
- Next-loop action if ambiguous: run exact sidecar on a small source slice and compare assignments.

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
- Remaining ambiguity: exact per-group timing and model utility are deferred.

## Next-Loop Action

- If positive: run a cap sweep or model-conditioning probe.
- If negative: implement exact per-group time sidecar on a bounded slice.
- If ambiguous: compare chunk-sort and exact sidecars on a small slice.

## Novelty Notes

- Closest analogies: auxiliary token sidecars and compressed residual streams.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering materialization of the already audited C3 representation.
