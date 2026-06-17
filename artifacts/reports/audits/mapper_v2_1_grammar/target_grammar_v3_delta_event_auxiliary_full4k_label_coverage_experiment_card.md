# Target Grammar v3 Delta-Event Auxiliary Full4K Label Coverage Experiment Card

## Hypothesis

The v3 delta-event auxiliary label surface covers the full available 4K mapper dataset under the current 8s delta and end-gap head ranges: every valid v3 event token receives a non-negative event-delta and event-signature label, every valid window receives exactly one non-negative terminal end-gap label, no label exceeds the configured head range, and no C3 replay or future target lookup is required.

## Root Objective

Harden v3 toward a final target grammar path that is reversible, lower-bit than v2.1, teacher-forcing friendly, online-local, and replacement-ready only after full4k evidence. This card removes the current coverage-scale blocker surfaced by the fixed-slice audit, where the max end-gap hit the `8000ms` boundary.

## Goal Decomposition

- Subgoal 1: Audit all mapper-eligible windows from the current 4K control index.
- Subgoal 2: Derive event-delta, event-signature, and terminal end-gap labels from v3 tokenized target fragments in a single pass.
- Subgoal 3: Verify current head ranges: `delta_event_delta_max_ms=8000` and `delta_event_end_gap_max_ms=8000`.
- Subgoal 4: Report max/unique distributions for event deltas, end gaps, and signatures, with examples for any failures.
- Subgoal 5: Preserve tokenizer, dataset schema, model defaults, rollout behavior, and online-local semantics.

## Candidate Variants

- Variant A: One-pass full4k label/range audit over the `ControlWindowDataset` records, tokenizing each eligible window once and aggregating coverage.
- Variant B: Reuse `MapperV3WindowDataset` with a new record cache, then run the existing fixed-slice coverage evaluator on the full index.
- Variant C: Run a larger-but-not-full sampled coverage audit.
- Variant D: Move directly to longer enabled-auxiliary training or rollout.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Full index scanned, zero label errors, zero negative/out-of-range labels, event/signature/end-gap counts match eligible windows/events | Any missing label, range miss, unsupported token, or runtime blocker before full scan |
| B | Same as A after cache build | Double tokenization makes the run opaque and slower; cache construction can dominate evidence collection |
| C | Faster coverage signal | Does not answer the explicit full4k replacement blocker |
| D | Training/rollout metrics | Premature if full4k labels exceed current head ranges |

## Selected Variant

Variant A: one-pass full4k label/range audit.

## Selection Pressure

Variant A directly answers the current blocker with the least confounding: label coverage and range safety across the full index. Variant B is a fallback only if one-pass parity with current dataset tokenization fails. Variant C is insufficient for replacement readiness. Variant D would waste training time if the label surface is invalid at full scale.

## Minimal Change

- Add one full4k label-coverage evaluator that reuses current v3 tokenization and the existing fixed-slice `window_label_coverage` semantics.
- Add focused tests for route selection and aggregation behavior.
- Generate full4k summary/report artifacts.
- Do not change tokenizer, mapper defaults, dataset schema, loss, training runner, inference, or rollout.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_full4k_label_coverage.py`
- `tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_result_report.md`

## Read-Only Context

- `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json`
- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`
- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py`

## Dataset Slice

Full available 4K mapper control index:

`artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`

The main run must use `--limit all`. Bounded `--limit` runs are allowed only as preflight and must not be interpreted as full4k evidence.

## Baseline / Comparator

- Full v3 event-token representation audit: `174515` full-audit windows, `0` reconstruction mismatches, positive token and bit reduction versus v2.1.
- Fixed-slice delta-event label coverage: `256` windows, `11983` event labels, `256` end-gap labels, `0` range errors, max end-gap `8000ms`.
- Tiny real-data training gate: `32` windows, auxiliary loss ratio `0.331061`, all training checks passed.

## Primary Metric

Full4k label error and range miss count.

Pass threshold: `0` label errors, `0` negative labels, `0` out-of-range delta labels, and `0` out-of-range end-gap labels.

## Secondary Metrics

- audited window count
- event token / event label / signature label count
- end-gap label count
- max event delta ms
- max end-gap ms
- unique event delta count
- unique end-gap count
- top event deltas, end gaps, and event signatures
- elapsed time and scan progress

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_full4k_label_coverage --progress-interval 5000
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_full4k_label_coverage.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json >/dev/null
git diff --check
```

## Guard Check

- tokenizer changed: false
- dataset schema changed: false
- mapper default behavior changed: false
- training run: false
- rollout run: false
- C3 backreference used: false
- future lookup used: false
- full v3 representation audit route remains positive
- tiny real-data training route remains positive

## Qualitative Check

The report must explicitly state whether the fixed-slice `8000ms` end-gap boundary survives full4k. If any label exceeds range, include failure examples and route to range mutation before more training.

## Positive Signal

The current 8s delta/end-gap heads cover the full4k label surface exactly, with counts matching events/windows and no negative/range failures.

## Negative Signal

Any event delta or end gap is negative or greater than `8000ms`, event labels do not match event-token count, end-gap labels do not match audited windows, or tokenization errors occur on the full index.

## Kill Criteria

- Any hard label semantics failure not attributable to invalid source input.
- Full index cannot be scanned without changing tokenizer semantics.
- End-gap/delta ranges fail so broadly that simple head-range mutation is not a bounded fix.

## Expected Failure Modes

- Full4k exposes end-gap values greater than `8000ms`.
- Rare terminal windows have chart-end/end-gap semantics different from fixed-slice examples.
- Runtime is long because tokenization has to parse beatmaps and replay state.
- Unsupported input windows appear in the full index.

## Expected Runtime / Runtime Budget

Expected runtime: tens of minutes on CPU. Use progress logs every `1000` audited windows. Stop only on repeated hard failures or explicit user interruption; do not call a partial run full4k evidence.

## Confounders

- This is a label/range audit, not trained mapper quality.
- It does not prove delta-event should become the generated target representation; it only proves the auxiliary/factorized label surface is full-scope-safe.
- Passing this audit still leaves a full training/inference comparison before replacement.

## Result Interpretation Plan

- If full4k coverage passes: route to `TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD`.
- If only range checks fail: route to `MUTATE_DELTA_EVENT_HEAD_RANGE_OR_BUCKETING`.
- If label semantics fail: route to `KILL_DELTA_EVENT_AUXILIARY_FULL4K_LABEL_SURFACE`.
- If runtime is interrupted or incomplete: route to `MUTATE_FULL4K_LABEL_AUDIT_RUNTIME`.

## Result Log Template

```text
route:
source_windows:
audited_windows:
tokenization_errors:
event_label_count:
signature_label_count:
end_gap_label_count:
out_of_range_delta_count:
out_of_range_end_gap_count:
max_event_delta_ms:
max_end_gap_ms:
unique_event_delta_count:
unique_end_gap_count:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether delta-event factorization deserves a full4k bit-proxy/factor-target card, or whether the label range/semantics need mutation before more training.

## Closest Analogies And Novelty Layer

Closest analogies are full-corpus target-label audits for factorized sequence heads, duration/range bucketing checks, and teacher-forced target decomposition. There is no novelty claim here. The layer being tested is representation-surface validity at full dataset scope.
