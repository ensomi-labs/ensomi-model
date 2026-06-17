# Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_delta_event_auxiliary_tiny_training_gate` passed synthetic trainability and recommended a fixed-slice label coverage or tiny real-data training gate.
- Acceptance source, if any: `target_grammar_v3_delta_event_auxiliary_tiny_training_gate_result_report.md`
- Source snapshot / evidence grade: strong synthetic trainability evidence; no real-data label coverage or real training evidence yet.

## Hypothesis

The default v3 delta-event auxiliary label extraction has complete coverage on the fixed 32-song / 256-window real v3 slice: every v3 event token receives a valid event-delta and event-signature label, every audited window receives exactly one valid end-gap label, all labels fit the current 8s head ranges, and no tokenizer/default/rollout behavior changes are required.

## Root Objective

Move v3 toward a real factorized target grammar that is reversible, lower-bit than v2.1, teacher-forcing friendly, online-local, and does not depend on C3-style cross-window replay. This card checks whether the auxiliary factorization is valid on real fixed-slice v3 target fragments before any training job.

## Goal Decomposition

- Subgoal 1: Reuse the fixed 32-song / 256-window v3 comparison index from the representation audit.
- Subgoal 2: Derive event-delta, event-signature, and terminal end-gap labels from current teacher-forced v3 target fragments.
- Subgoal 3: Verify label coverage and range safety for the current default heads: delta max 8000ms and end-gap max 8000ms.
- Subgoal 4: Compare counts against the current v3 target stream: event labels must match v3 event token count; end-gap labels must match audited window count.

## Candidate Variants

- Variant A: fixed-slice label coverage audit using current v3 tokenizer and auxiliary label semantics.
- Variant B: tiny real-data training run with the auxiliary objective enabled.
- Variant C: full fixed32 training run with auxiliary objective enabled.
- Variant D: move directly to rollout or target replacement.

## Local Verification Matrix

- Variant A: pass if label derivation succeeds on all 256 windows, no labels are out of range, event label count equals event-token count, and one end-gap label exists per window.
- Variant B: reject until label coverage is known; otherwise training failures could be label/range issues.
- Variant C: reject as too broad before fixed-slice label coverage.
- Variant D: reject because rollout/replacement cannot interpret an unvalidated real-data label surface.

## Selected Variant

- Selected: Variant A.
- Rejected: B/C/D are later gates.
- Why this is the smallest useful test: it reads existing fixed-slice data and does not train, roll out, or change runtime behavior.

## Selection Pressure

- Primary pressure: zero label errors and zero out-of-range labels on the fixed slice.
- Guard pressure: event-label count equals v3 event-token count; end-gap-label count equals audited window count; no tokenizer/default/runtime/C3/future context change.
- Runtime pressure: under 5 minutes.
- Kill pressure: any range error, count mismatch, invalid negative delta/end gap, or inability to tokenize the fixed index.

## Minimal Change

Add a bounded evaluator that reads `MapperV3WindowDataset` over the fixed 256-window index, tokenizes each record, derives label coverage stats, and writes JSON/Markdown artifacts.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py`
- `tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_result_report.md`

## Read-Only Context Files

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_summary.json`
- `src/pulsefield_model/models/mapper/v3/loss.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`

## Dataset Slice

Fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

## Baseline / Comparator

Current v3 target fragments on the same fixed slice, plus the committed delta-event proxy audit counts.

## Primary Metric

Label coverage pass/fail:

- `label_error_count == 0`
- `out_of_range_delta_count == 0`
- `out_of_range_end_gap_count == 0`

## Secondary Metric

Audited window count, v3 token count, event token count, event label count, end-gap label count, max event delta, max end-gap, unique delta count, unique end-gap count, and top label distributions.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3/test_model.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json >/dev/null
git diff --check
```

## Guard Check

The evaluator must report `no_training=true`, `no_rollout=true`, `tokenizer_changed=false`, `dataset_schema_changed=false`, `default_behavior_changed=false`, `uses_c3_backreference=false`, and `uses_future_lookup=false`.

## Qualitative Check

The report must state this is a fixed-slice label coverage audit only, not real training, rollout quality, full4k coverage, or target replacement readiness.

## Positive Signal

All fixed-slice v3 fragments produce valid labels under current head ranges; event/end-gap counts match expected comparator counts.

## Negative Signal

Any label error, range miss, mismatch between event tokens and event labels, or missing end-gap label.

## Kill Criteria

- Label error count > 0.
- Event label count != v3 event token count.
- End-gap label count != audited window count.
- Any event delta or end gap is negative or exceeds current head range.
- The fixed index cannot be loaded/tokenized.

## Expected Failure Modes

- Terminal chart-end windows create end gaps outside the default 8s range.
- Padded or EOS rows create ambiguous end-gap placement.
- Event deltas derived from current-ms state diverge from target tokens.
- A fixed-slice record has unsupported beatmap content.

## Expected Runtime / Runtime Budget

Under 5 minutes. Stop before any real training run.

## Confounders

This fixed 256-window slice is not the full 4k dataset. Passing this audit does not prove full-dataset label coverage, trainability, rollout quality, or replacement readiness.

## Result Interpretation Plan

- Positive result: create a tiny real-data training gate for the auxiliary objective.
- Negative result: mutate label extraction/head ranges before any real training.
- Ambiguous result: run a broader label coverage audit before training.
- Human owner decides whether to proceed from label coverage to real training.

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
- Remaining ambiguity: passing this fixed slice does not prove full4k coverage or trained mapper quality.

## Next-Loop Action

- If positive: create tiny real-data auxiliary training gate.
- If negative: mutate label extraction/head ranges.
- If ambiguous: broaden label coverage before training.

## Closest Analogies and Novelty Layer

- Closest analogies: label coverage audits for auxiliary decoder heads and symbolic music event-delta representations.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is Pulsefield-specific target-grammar verification, not a broad novelty claim.
