# C3 Wide-Sweep Runtime-Bound Repair Experiment Card

## Hypothesis

The full-cache `--c3-wide-sweep` runtime bottleneck is caused by recomputing equivalent exact/mirror/skeleton LZ candidate options for each A6/A7/A8 policy. A single largest-window candidate superset can be computed once and filtered by policy window, phase filter, and match type without changing legal candidate semantics.

## Root Objective

Make the C3 hardening A6/A7/A8 decomposition runnable as a routine verification gate so C3 promotion evidence includes bounded-window, pointer-code, and source-filter diagnostics.

## Goal Decomposition

- Reproduce the current C3 hardening candidate semantics on a small synthetic test.
- Avoid rebuilding mirror/skeleton match options for every wide policy.
- Preserve charged pointer/mode/residual costs and reconstruction guards.
- Run the full `--c3-wide-sweep` audit or return a clear runtime/failure result.

## Candidate Variants

- V1 shared largest-window superset: build options once for the widest requested history and filter by policy.
- V2 per-window incremental cache: build options for each window but reuse precomputed atom/mirror/skeleton keys.
- V3 bounded wide gate: remove full-history from the default wide command and keep it as a separate diagnostic.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| V1 | Synthetic test proves filtered 16/32/phase options equal direct construction | If equal, V1 can replace repeated construction. |
| V2 | Profiling shows key recomputation dominates | If not, V2 is insufficient. |
| V3 | Full wide still too slow after V1/V2 or full-history is unnecessary | If needed, report as a mutation, not a silent metric change. |

## Selected Variant

V1 shared largest-window superset. It is the smallest semantic-preserving change and directly targets the observed stack: `_lz_candidate_options_by_record -> _lz_match_length -> _mirror_atom_key -> _mirror_mask_4`.

## Selection Pressure

Choose the repair only if it preserves candidate options for smaller windows and phase filters on synthetic coverage. Do not change the selected C3 variant or any cost model just to make runtime look better.

## Minimal Change

Add a policy-filter helper that derives per-policy options from a shared superset by checking distance, phase filter, and allowed match types. Update `_build_c3_hardening_plans` to build that superset once per widest needed history instead of once per policy window/phase.

## Files Likely To Change

- `src/pulsefield_model/osu_core/context_adaptive_fallback_codec_audit.py`
- `tests/osu_core/test_context_adaptive_fallback_codec_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_wide_sweep_runtime_bound_repair_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_wide_sweep_runtime_bound_repair_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_wide_sweep_runtime_bound_repair_summary.json`

## Dataset Slice

Use synthetic rows for semantic equivalence tests. Final audit uses the full existing LE<=3 beat-chunk cache with `limit_chunks=None` if runtime is acceptable.

## Baseline / Comparator

Baseline is the current direct `_lz_candidate_options_by_record` construction per policy. V1 must produce identical filtered candidate option signatures for representative window and phase policies on synthetic tests.

## Primary Metric

`--c3-wide-sweep` completes and writes a valid report, or the result report documents the remaining runtime blocker with stack/context.

## Secondary Metric

Focused unit tests pass; C3 hardening reconstruction guards remain zero mismatches; selected wide-sweep variant and deltas are recorded.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening --c3-wide-sweep
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json >/dev/null
```

## Guard Check

- Candidate signatures for synthetic direct-vs-filtered policies must match exactly.
- Full hardening reconstruction mismatches must remain `0`.
- No default mapper tokenization, training data, or inference path changes.
- No metric is removed or silently narrowed.

## Qualitative Check

Inspect that A6 16/32/64/128/512/1024/full-history, A7 pointer-code variants, and A8 phase-filter variants are present in the completed report when the full run succeeds.

## Positive Signal

The full wide sweep completes under the expected 5 to 20 minute audit budget and preserves C3 hardening gates.

## Negative Signal

The full wide sweep still exceeds the runtime budget, candidate equivalence fails, or reconstruction guard failures appear.

## Kill Criteria

Kill this repair if the filtered superset changes candidate semantics or if wide sweep remains impractical after the shared-superset change. In that case, mutate to V3 with full-history as a separate bounded diagnostic.

## Expected Failure Modes

- Full-history options still dominate runtime even when computed once.
- Filtering by phase after candidate construction preserves semantics but does not reduce memory enough.
- Synthetic tests miss a policy interaction that appears on full cache.

## Expected Runtime / Runtime Budget

Synthetic tests should finish in seconds. Full wide sweep should finish within 20 minutes; stop and report if it remains active beyond that budget.

## Confounders

The C3 hardening JSON `code_dirty` field may be set by artifact writes during the audit. Treat git status before/after the run as the authoritative source for source-code dirtiness.

## Result Interpretation Plan

If the wide sweep completes, promote the C3 hardening gate from non-wide-only to full A6/A7/A8 decomposition. If it does not complete, keep the C3 compression result but mark full wide decomposition as an engineering blocker.

## Result Log Template

Record command, commit, runtime, candidate-equivalence test result, whether full wide sweep completed, selected variant, reconstruction guard status, A6/A7/A8 availability, and next-loop action.

## Next-Loop Action

If V1 passes, rerun and commit the full C3 hardening wide report. If V1 fails, create a V3 bounded-wide mutation card before changing the audit scope.

## Closest Analogies And Novelty Layer

Closest analogy is LZ candidate indexing and dynamic-programming cache reuse. This is an audit-harness engineering repair, not representation novelty.
