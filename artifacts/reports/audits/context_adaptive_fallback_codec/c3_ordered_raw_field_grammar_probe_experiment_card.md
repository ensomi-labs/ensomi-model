# C3 Ordered RAW Field Grammar Probe Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: C3 hardening passed full reconstruction and compression gates, but P10-P23 showed the current mapper-side C3 auxiliary path has weak RAW recovery and no legal inference-time input source.
- Acceptance source, if any: active C3/v3 full-pipeline goal plus `c3_raw_context_bucket_probe_result_report.md`, `c3_kind_logit_calibration_result_report.md`, and `casf_c3_to_v3_route_update_report.md`.
- Source snapshot / evidence grade: strong evidence for C3 codec legality and compression; strong evidence that target-derived C3 input conditioning is illegal for inference; medium evidence that bag/kind auxiliary targets have diminishing returns; weak evidence for any ordered grammar replacement.

## Hypothesis

The C3 auxiliary path may be failing because it converts an ordered fallback-substream codec into a per-window bag/ranking target. If RAW C3 tokens are decomposed into ordered field-level grammar symbols, the target may preserve the structure that made C3 compressive while reducing the RAW label space enough to justify a later teacher-forcing target or auxiliary-sequence experiment.

## Root Objective

Decide whether C3 still deserves another bounded pipeline-facing experiment as an ordered target representation, or whether the practical replacement path should pivot back to v3/v2.1 grammar work.

## Goal Decomposition

- Subgoal 1: Preserve the proven C3 legality boundary: C3 labels remain target-side only and are never used as inference inputs.
- Subgoal 2: Measure whether ordered RAW field factorization reduces label entropy and label-space size versus exact C3 RAW tokens.
- Subgoal 3: Verify that ordered field reconstruction can round-trip back to exact C3 side-stream tokens before any model training.
- Subgoal 4: Compare the result against the failed bag/kind auxiliary path and the simpler v3 target-grammar route.

## Candidate Variants

- Variant A: Context-conditioned RAW factor heads. Rejected for this next step because P23 found only one supported positive context bucket and weak fields still mostly trailed unigram.
- Variant B: Post-hoc kind-logit calibration. Rejected because P21 improved RAW only by damaging non-RAW recovery and still trailed unigram.
- Variant C: Ordered C3 RAW field grammar probe. Selected because it tests whether the lost signal is order/structure rather than another classifier-head tuning issue.
- Variant D: Direct full C3 target grammar replacement. Deferred because C3 has cross-window references and side-stream semantics; a smaller ordered-field audit should fail or pass first.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Train context-conditioned factor heads | Weak fields beat unigram in supported buckets | Context lift remains narrow or noisy |
| B | Post-hoc calibration sweep | Held-out recall improves without non-RAW damage | RAW gain requires collapsing REF/RES |
| C | Ordered field grammar audit | Exact C3 side-stream tokens reconstruct from ordered fields and charged field bits improve over exact RAW labels | Reconstruction mismatch or no entropy/label-space reduction |
| D | Full C3 target grammar | Legal replay and online state are defined | Requires complex cross-window replay before proving target simplicity |

## Selected Variant

- Selected: Variant C, an ordered C3 RAW field grammar probe.
- Rejected: A and B because the last diagnostics already showed diminishing returns.
- Deferred: D until ordered-field accounting proves that C3 can become a tractable target sequence rather than only a codec artifact.
- Why this is the smallest useful test: it requires no mapper training, no inference changes, and can fail on reconstruction or entropy before spending model runtime.

## Selection Pressure

- Primary pressure: reduce RAW target complexity while preserving exact ordered side-stream reconstruction.
- Guard pressure: no target-derived C3 labels may become model inputs; no inference-time C3 sidecar dependency is introduced.
- Runtime pressure: run on a bounded slice first, then full exact sidecar only if the parser/reconstruction gate passes.
- Kill pressure: if ordered factorization cannot reconstruct exact C3 tokens or does not materially reduce RAW entropy/label space, stop C3 mapper-side pursuit and pivot to v3/v2.1 grammar work.

## Research Question

Is the remaining C3 mapper-side blocker caused by bagging/ranking away ordered side-stream structure, or is C3 fundamentally a codec-side representation that should not drive the next mapper target grammar?

## Closest Analogies / Novelty Layer

- Closest analogies: LZ-style backreference streams, residual coding, factorized categorical targets, byte-pair/field tokenization, and teacher-forced sequence targets.
- Relevant taxonomy bucket: representation/interface audit for an auto-research candidate after compression evidence.
- Novelty layer, if any: the possible novelty is at the representation-policy layer, not the general LZ/factorization method.
- Representation novelty vs engineering variation: C3's chart-local fallback reference policy is the representation claim; ordered RAW field splitting is an engineering test of mapper feasibility.

## Minimal Change

Add an artifact-only evaluator that:

- loads `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`;
- parses ordered side-stream token sequences;
- splits `RAW` tokens into ordered field grammar symbols while leaving `REF` and `RES` as explicit kind-tagged symbols or simple typed payloads;
- reconstructs exact side-stream tokens from the field grammar and reports mismatches;
- computes train/eval/test or deterministic split unigram bits for exact C3 token labels versus ordered field symbols;
- reports sequence-length ratio, label-space size, RAW entropy, REF/RES preservation, and cross-window/reference statistics.

No mapper training, rollout, or production inference path changes are part of this card.

## Files Likely To Change

- `src/pulsefield_model/evals/c3_ordered_raw_field_grammar_probe.py`
- `tests/evals/test_c3_ordered_raw_field_grammar_probe.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `src/pulsefield_model/evals/c3_raw_structure_audit.py`
- `src/pulsefield_model/evals/c3_raw_factor_probe.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_final_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_c3_to_v3_route_update_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_result_report.md`

## Dataset Slice

Stage 1 bounded slice:

- Use the smoke exact sidecar if present: `artifacts/cache/c3_mapper_window_sidecar/smoke_c3_exact_mapper_window_sidecar.json`.
- Otherwise sample the first 1,024 windows from `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`.

Stage 2 full-cache audit, only if Stage 1 passes:

- Full exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`.
- Expected source stats from P5: token vocab size `14,294`, sidecar tokens `3,203,904`, windows `173,268`.

## Baseline / Comparator

Primary C3 comparator:

- Exact C3 side-stream token labels from P5/P11.
- Reduced top-512-per-kind auxiliary route from P17-P23.

Route comparator:

- v3 full-dataset target grammar remains the practical teacher-forcing baseline: reconstruction mismatches `0`, token reduction about `20.54%`, eval total-bit reduction about `5.43%`, no C3-style target-derived input.

## Primary Metric

Exact side-stream reconstruction mismatch count after ordered RAW field encode/decode.

Acceptance requires `0` mismatches.

## Secondary Metrics

- Ordered-field charged bits per original C3 side-stream token versus exact C3 token unigram bits.
- RAW-only bits per RAW token before and after field splitting.
- Field vocabulary sizes and entropy by RAW field.
- Sequence-length ratio versus exact side-stream tokens.
- REF/RES preservation rate.
- Cross-window-reference token share carried through the ordered grammar.

## Verify Command / Evaluation Procedure

Stage 1:

```bash
uv run python -m pulsefield_model.evals.c3_ordered_raw_field_grammar_probe \
  --sidecar artifacts/cache/c3_mapper_window_sidecar/smoke_c3_exact_mapper_window_sidecar.json \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_ordered_raw_field_grammar_probe_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/smoke_c3_ordered_raw_field_grammar_probe_result_report.md \
  --max-windows 1024
```

Stage 2, only after Stage 1 passes:

```bash
uv run python -m pulsefield_model.evals.c3_ordered_raw_field_grammar_probe \
  --sidecar artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_ordered_raw_field_grammar_probe.py tests/evals/test_c3_raw_factor_probe.py tests/osu_core/test_c3_side_stream_tokenization.py -q
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The result report must answer whether ordered field splitting makes C3 look like a viable emitted target sequence, or whether it remains a side-stream codec whose value does not translate cleanly into mapper training.

## Positive Signal

- Reconstruction mismatches are `0`.
- Ordered RAW field splitting reduces RAW label entropy or charged bits by at least `0.05` bits per RAW token versus exact RAW token labels.
- Sequence-length expansion is bounded enough to justify a later tiny training card, preferably `<= 2.0x` exact side-stream tokens.
- REF/RES structure remains explicitly recoverable and no target-derived sidecar is introduced as an inference input.

## Negative Signal

- Any reconstruction mismatch.
- Ordered splitting expands sequence length without reducing charged bits.
- RAW fields remain dominated by unigram priors with no useful factor entropy reduction.
- Cross-window/reference semantics remain too large to describe without C3-specific replay machinery.

## Kill Criteria

- Kill ordered C3 target-grammar pursuit if exact side-stream reconstruction is not lossless.
- Kill if the field grammar requires future target context beyond already-emitted C3 tokens.
- Kill if sequence expansion and field costs erase the C3 compression advantage.
- Kill if the audit concludes that C3 still needs a side-stream replay model more complex than v3's local event grammar.

## Expected Failure Modes

- `RAW` tokens may be too sparse or syntactically shallow for field splitting to help.
- Splitting may reduce vocabulary but lengthen the sequence too much.
- REF/RES tokens may dominate the remaining complexity after RAW is split.
- Full-cache sidecar loading may be memory-heavy; bounded Stage 1 should catch parser bugs first.

## Confounders

- Compression-oriented entropy is not the same as mapper learnability.
- The exact sidecar is target-derived; this audit is target-design only.
- The reduced top-512-per-kind sidecar may understate full-vocab RAW complexity.
- v3 is a simpler local target grammar and may remain preferable even if ordered C3 fields improve entropy.

## Expected Runtime / Runtime Budget

- Stage 1 expected runtime: under 30 seconds.
- Stage 2 expected runtime: under 5 minutes on CPU.
- Stop if Stage 1 reconstruction fails or if full sidecar memory usage is unreasonable.

## Result Interpretation Plan

- Positive result would suggest: write a tiny ordered C3 sequence-head or target-grammar smoke card.
- Negative result would suggest: stop C3 mapper-side integration and focus on v3/v2.1 grammar improvement.
- Ambiguous result would require: compare exact C3 token, reduced C3 token, and ordered-field targets on the same bounded training slice.
- Human owner decides: whether C3's stronger codec gain is worth the extra state/sequence complexity.
- Next-loop action if positive: bounded ordered C3 teacher-forcing smoke.
- Next-loop action if negative: pivot to v3 continuation/grammar repair or v2.1 grammar improvement.
- Next-loop action if ambiguous: report a three-way target complexity table before training.

## Result Log Template

- Experiment: C3 ordered RAW field grammar probe
- Date:
- Commit / run id:
- Sidecar path:
- Window count:
- Token count:
- Runtime:
- Reconstruction mismatches:
- Exact C3 bits/token:
- Ordered field bits/token:
- RAW exact bits/RAW token:
- RAW ordered bits/RAW token:
- Sequence-length ratio:
- REF/RES preservation:
- Cross-window-reference share:
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
- Remaining ambiguity: a positive entropy/reconstruction result would still not prove mapper quality; it would only justify one bounded ordered-target smoke.

## Next-Loop Action

- If positive: create and run a tiny ordered C3 target-sequence smoke.
- If negative: stop spending mapper-side work on C3 and continue with v3/v2.1 grammar repair.
- If ambiguous: produce a target-complexity comparison before training.

## Novelty Notes

- Closest analogies: LZ backreference codecs, residual streams, factorized categorical sequence targets, and grammar/side-channel tokenization.
- Novelty layer, if any: representation-policy layer for combining corpus-global motif tokens with chart-local fallback references.
- Representation novelty vs engineering variation: ordered RAW field splitting is engineering variation used to test whether the representation can become a mapper target.
