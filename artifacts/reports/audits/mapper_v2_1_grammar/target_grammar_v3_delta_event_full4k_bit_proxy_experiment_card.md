# Target Grammar v3 Delta-Event Full4K Bit Proxy Experiment Card

## Hypothesis

The full4k delta-event factorized target proxy is materially more compact than current v3 under the same unigram-proxy family used by the bounded delta-event audit: event rows carry `delta_ms` and event signature, each window carries one terminal end-gap row, reconstruction is exact, sequence rows are fewer than v3 tokens, and factorized total bits are not worse than current v3.

## Root Objective

Harden v3 toward a final target grammar that is reversible, lower-bit than v2.1, teacher-forcing friendly, online-local, and does not require C3-style cross-window replay. The previous full4k label coverage gate proved the label surface is range-safe; this card tests whether that full4k factorized surface is actually a better representation target before any default or generated-target change.

## Goal Decomposition

- Subgoal 1: Reuse current v3 tokenization and scan the full 4K mapper scope used by the full v3 representation audit.
- Subgoal 2: Count current v3 tokens and delta-event factorized rows over the same audited windows.
- Subgoal 3: Verify exact reconstruction of event times/signatures and terminal target end from the delta-event proxy.
- Subgoal 4: Compute full4k unigram proxy bits for current v3, factorized delta-event, and flat delta/event pairs.
- Subgoal 5: Route to a factor-target/model-card only if full4k compression and reconstruction both pass.

## Candidate Variants

- Variant A: One-pass full4k proxy audit that tokenizes each mapper-eligible v3 window once and computes v3, factorized, and flat-pair counters directly.
- Variant B: Derive proxy bits from `target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json`.
- Variant C: Extend the existing fixed-slice proxy evaluator to use `MapperV3WindowDataset` over the full index.
- Variant D: Move straight to implementing a generated factorized target/model head.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Full-scope window count matches `174515`, reconstruction mismatches are zero, row ratio < 1, factorized bit ratio <= 1 | Any mismatch, incomplete scope, factorized bits regress, runtime cannot finish |
| B | Would be fast if full distributions existed | Current full4k label summary stores only top distributions, so exact bits cannot be computed |
| C | Reuses older evaluator structure | Full dataset construction is opaque and slower; previous probe showed non-instrumented construction can stall |
| D | Could reach implementation faster | Premature because full4k bit viability has not been proven |

## Selected Variant

Variant A: one-pass full4k delta-event bit proxy audit.

## Selection Pressure

Variant A is the only bounded option that gives exact full4k compression evidence without changing target grammar or defaults. Variant B is rejected because the existing full4k coverage artifact intentionally omits full counters. Variant C duplicates slow dataset construction. Variant D skips a required representation gate.

## Minimal Change

- Add one evaluator that scans the full4k index, tokenizes v3 windows once, computes current-v3 and delta-event proxy counters, verifies reconstruction, writes JSON and Markdown artifacts, and returns a route.
- Add focused tests for bit formulas, route selection, and reconstruction guard behavior.
- Do not change tokenizer, mapper defaults, dataset schema, loss, training runner, inference, or rollout.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_full4k_bit_proxy.py`
- `tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_result_report.md`

## Read-Only Context

- `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_summary.json`
- `src/pulsefield_model/evals/target_grammar_v3_delta_event_proxy_audit.py`
- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_full4k_label_coverage.py`

## Dataset Slice

Full available 4K mapper scope:

`artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`

The positive gate requires parity with the existing full v3 audit window count: `174515` audited mapper-eligible windows. Bounded `--limit` runs are preflight only.

## Baseline / Comparator

- Current v3 full dataset audit: `174515` audited windows, zero reconstruction mismatches, positive token/bit reduction versus v2.1.
- Full4k delta-event label coverage: `174515` audited windows, `9720062` event labels, `174515` end-gap labels, zero range errors, max event delta `7980ms`, max end-gap `8000ms`.
- Fixed-slice delta-event proxy: row ratio `0.386820`, factorized bit ratio vs v3 `0.747341`.

## Primary Metric

Full4k factorized proxy total-bit ratio versus current v3 unigram total bits.

Pass threshold: `proxy_factorized_total_bit_ratio_vs_v3 <= 1.0`.

## Secondary Metrics

- sequence row ratio versus v3 token count
- flat-pair total-bit ratio versus v3
- flat delta/event pair cardinality
- event reconstruction mismatch count
- end reconstruction mismatch count
- v3 token count
- proxy row count
- unique event deltas, end gaps, event signatures, and flat delta/event pairs

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_full4k_bit_proxy --progress-interval 5000
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_full4k_bit_proxy.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json >/dev/null
git diff --check
```

## Guard Check

- no training
- no rollout
- no tokenizer/default/dataset schema change
- no C3 backreference
- no future lookup
- audited window count matches full v3 audit and full4k label coverage
- reconstruction mismatches are zero

## Qualitative Check

The report must state whether full4k supports a factorized target follow-up, a bucketing mutation, or killing the delta-event factorization path. It must not claim trained mapper quality or replacement readiness.

## Positive Signal

Full scope completes, reconstruction is exact, proxy rows are fewer than current v3 tokens, factorized total bits are not worse than v3, and flat-pair cardinality is tractable enough to discuss as a secondary target family.

## Negative Signal

Any reconstruction mismatch, incomplete full scope, factorized bit regression, row count not lower than v3, or flat-pair vocabulary too large for a reasonable target head.

## Kill Criteria

- Any event/end reconstruction mismatch that is not invalid source data.
- Sequence row ratio >= 1.0.
- Factorized bit ratio > 1.0 at full4k scope.

## Expected Failure Modes

- Full4k distributions are less concentrated than fixed-slice distributions, erasing the bounded-slice bit advantage.
- Flat delta/event pairs are too numerous even if factorized bits pass.
- Runtime is long because exact full4k counters require another tokenization pass.

## Expected Runtime / Runtime Budget

Expected runtime: about one hour on CPU, similar to the full4k label coverage audit. Use progress logs every `5000` audited windows. Stop only on hard reconstruction errors or explicit interruption; partial runs are not full4k evidence.

## Confounders

- Unigram target-stream bits are a representation proxy, not conditional model likelihood.
- A factorized target can have fewer bits but still be harder to train or decode.
- Current v3 already proved a v2.1 bit advantage under a different train/eval unigram setup; this card compares within a full-scope self-unigram proxy family for v3 versus delta-event.

## Result Interpretation Plan

- If full4k factorized bits pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`.
- If rows pass but bits regress: route to `MUTATE_DELTA_EVENT_BUCKETING`.
- If reconstruction fails: route to `KILL_DELTA_EVENT_FACTOR_TARGET`.
- If full scope is incomplete: route to `MUTATE_FULL4K_PROXY_AUDIT_RUNTIME`.

## Result Log Template

```text
route:
audited_windows:
v3_token_count:
proxy_row_count:
sequence_length_ratio_vs_v3:
v3_unigram_total_bits:
proxy_factorized_total_bits:
proxy_factorized_total_bit_ratio_vs_v3:
flat_pair_total_bit_ratio_vs_v3:
event_reconstruction_mismatches:
end_reconstruction_mismatches:
unique_event_deltas:
unique_end_gaps:
unique_event_signatures:
unique_flat_delta_event_pairs:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to create a factorized target/model-card for delta-event v3, mutate delta bucketing/ranges, or stop this factorization path and return to simpler v3/v2.1 grammar hardening.

## Closest Analogies And Novelty Layer

Closest analogies are factorized token heads, duration-conditioned event generation, and target-stream entropy audits for sequence representations. There is no novelty claim here. The tested layer is representation-surface compression and exact reversibility at full dataset scope, not a trained model contribution.
