# Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 generated-prefix failures are dominated by standalone time-shift repetition; test a structural target proxy that binds each event to its preceding delta.
- Acceptance source, if any: `mapper_v21_terminal_guard_v3_route_synthesis_result_report.md`
- Source snapshot / evidence grade: strong local artifact evidence; generated-prefix trace, trace-conditioned decode, completion-budget, continuation-jump, v3 full-dataset, C3 target-complexity, and v2.1 terminal-guard reports are committed.

## Hypothesis

A factorized delta-event target proxy can remove the standalone time-shift decision surface that caused v3 generated-prefix time-shift repetition, while preserving local reversible beatmap-event reconstruction and reducing sequence length versus current v3. If the proxy loses bit pressure or reconstruction, kill it before training.

## Root Objective

Move toward a final v3-family target grammar that is reversible, lower-bit than v2.1, local, teacher-forcing friendly, and usable for online inference without C3-style cross-window replay or future target lookup.

## Goal Decomposition

- Subgoal 1: Convert existing v3 target fragments into a factorized event sequence: `DELTA_TO_EVENT + EVENT_SIGNATURE`, plus a local end-gap marker for window completion.
- Subgoal 2: Verify exact reconstruction of event times/signatures and window terminal time from the proxy sequence.
- Subgoal 3: Compare token count and factorized unigram bit proxy against current v3 and v2.1 committed representation baselines.

## Candidate Variants

- Variant A: factorized delta-event proxy with one event row per event token and one end-gap row per window.
- Variant B: flat combined `(delta,event_signature)` token vocabulary.
- Variant C: add a continuation/end marker to current v3 but keep standalone time-shift tokens.
- Variant D: rerun decode/training-side repairs without changing representation.

## Local Verification Matrix

- Variant A: pass if reconstruction mismatches are zero, sequence length is below current v3, factorized bit proxy is not worse than current v3 total-bit proxy on the bounded slice, and no C3/backreference/future context is needed.
- Variant B: pass only if the flat vocabulary remains tractable; fail if unique token count explodes.
- Variant C: pass only if token count/bit proxy improves while preserving current v3 conversion; likely weak against the traced standalone TS repetition.
- Variant D: already has negative evidence from trace-conditioned spacing escape, selective completion budget, and continuation-jump gates.

## Selected Variant

- Selected: Variant A.
- Rejected: Variant B risks a high-cardinality sparse vocabulary; Variant C does not remove the standalone TS attractor; Variant D repeats killed families.
- Why this is the smallest useful test: it is an artifact/data audit only, reusing the existing fixed 32-song/256-window slice and v3 tokenizer. It does not train or alter runtime.

## Selection Pressure

- Primary pressure: lower sequence length and non-worse factorized bits versus current v3 on the bounded slice.
- Guard pressure: exact event/window reconstruction, no future lookup, no C3 replay, no tokenizer/default changes.
- Runtime pressure: under 5 minutes on the fixed 256-window slice.
- Kill pressure: any reconstruction mismatch, bit proxy worse than current v3, or flat-factor cardinality too large for a plausible mapper target.

## Research Question

Can the v3 grammar be structurally mutated so generated decoding emits event-bearing steps rather than standalone time-shift steps, without losing the representation wins that made v3 preferable to C3 target integration?

## Closest Analogies / Novelty Layer

- Closest analogies: event-sequence/delta-time symbolic music representations; factorized timing/event heads; CTC-like event timing streams in sequence models.
- Relevant taxonomy bucket: representation and target grammar mutation, not a codec-side C3 extension.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is a representation-policy variation inside Pulsefield, not a broad novelty claim.

## Minimal Change

Add a bounded evaluator that reads the fixed v3 slice through `MapperV3WindowDataset`, converts target fragments into a delta-event proxy, verifies reconstruction, estimates factorized unigram bits, and writes JSON/Markdown artifacts.

## Files Likely to Change

- `src/pulsefield_model/evals/target_grammar_v3_delta_event_proxy_audit.py`
- `tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_result_report.md`

## Read-Only Context Files

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_guard_v3_route_synthesis_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json`

## Dataset Slice

The existing fixed 32-song / 256-window v3 comparison index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`.

## Baseline / Comparator

- Current v3 event-token grammar on the same 256-window slice.
- Committed v2.1/v3 representation baselines from `target_grammar_v3_full_dataset_audit_summary.json`.
- C3 target-complexity artifact only as a guard against reintroducing C3 replay/future/state costs.

## Primary Metric

Proxy total bits versus current v3 target-fragment total bits on the same slice.

## Secondary Metric

Sequence length ratio, unique delta count, unique event-signature count, flat pair count, reconstruction mismatch count, end-gap distribution, and event-row count.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.target_grammar_v3_delta_event_proxy_audit
uv run --group dev pytest tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_summary.json >/dev/null
```

## Guard Check

`git diff --check` and focused pytest must pass. The evaluator must report no training, no rollout, no tokenizer/default changes, no target-derived C3 input, and no future lookup.

## Qualitative Check

The report must explicitly state that this is a proxy representation audit, not trained rollout evidence or replacement readiness.

## Positive Signal

Reconstruction mismatches are zero, proxy sequence length is lower than current v3, proxy total bits are not worse than current v3, and flat/factor cardinalities look tractable enough for a follow-up model-head card.

## Negative Signal

Any reconstruction mismatch, no sequence reduction, proxy bit regression, or cardinality explosion.

## Kill Criteria

- Event-time/signature reconstruction mismatches > 0.
- Terminal end-gap/window mismatches > 0.
- Proxy total-bit ratio versus current v3 > 1.0.
- Proxy sequence-length ratio versus current v3 >= 1.0.
- Flat `(delta,event_signature)` vocabulary is so large on 256 windows that a flat head is clearly not viable; in that case flat Variant B is killed even if factorized Variant A survives.
- Audit cannot run from the bounded index without building a full 4k cache.

## Expected Failure Modes

- The end-gap marker erases information needed for window completion.
- Factorized delta/event bits become worse than current v3 despite fewer rows.
- Delta cardinality is too high, requiring bucketing or a continuous timing head.
- The bounded slice is too small to extrapolate to full 4k.

## Confounders

This audit estimates target bits with unigram proxies only. It does not model conditional dependence between delta and event signature, and it does not prove training/inference quality.

## Expected Runtime / Runtime Budget

Under 5 minutes. Stop if the bounded index or dataset root is unavailable.

## Result Interpretation Plan

- Positive result would suggest: create a bounded model-head/training smoke for a factorized delta-event v3 variant.
- Negative result would suggest: keep current v3 representation and pivot to v2.1 grammar hardening/default validation.
- Ambiguous result would require: run the proxy on the full v3 dataset audit scope or add a small delta bucketing ablation.
- Human owner decides: whether the factorized proxy deserves a real target grammar implementation.
- Next-loop action if positive: make a delta-event model/loss plumbing card.
- Next-loop action if negative: create a v2.1 terminal guard default/broader validation card.
- Next-loop action if ambiguous: add a bucketing/full-scope representation audit card.

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
- Remaining ambiguity: bounded-slice target proxy does not prove full-dataset or trained rollout readiness.

## Next-Loop Action

- If positive: create the factorized delta-event model-head/training smoke card.
- If negative: pivot to v2.1 terminal guard broader/default validation.
- If ambiguous: run full-scope or delta-bucket ablation before model changes.

## Novelty Notes

- Closest analogies: event-delta symbolic music tokenization and factorized timing/event prediction heads.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: a Pulsefield target-grammar engineering variation aimed at a known generated-prefix failure.
