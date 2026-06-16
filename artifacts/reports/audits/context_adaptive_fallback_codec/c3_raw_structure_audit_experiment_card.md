# C3 RAW-Structure Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P19 reduced C3 kind-head training improved `REF` and `RES`, but `RAW` remained far below the reduced unigram comparator.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_kind_heads_1000step_result_report.md`.
- Source snapshot / evidence grade: strong local evidence from full C3 codec hardening, exact sidecar generation, legal target-role audit, and P16-P19 auxiliary-target diagnostics; weak evidence for the internal structure of RAW misses.

## Hypothesis

P19's remaining C3 auxiliary-target failure is concentrated in identifiable RAW token substructures rather than uniformly across all RAW labels. A post-hoc RAW audit should show whether misses are dominated by exact-token frequency, RAW field buckets, composite order signatures, or ranking/calibration just outside top-20.

## Root Objective

Move C3 toward legal full-pipeline use by deciding whether the next target-side representation should split RAW into smaller structured heads, keep the reduced kind-head bag objective, or mutate toward ordered C3 generation.

## Goal Decomposition

- Subgoal 1: Reuse the P19 checkpoint, config, and reduced sidecar without retraining.
- Subgoal 2: Measure RAW hit/miss structure against both model predictions and reduced train-unigram predictions.
- Subgoal 3: Produce a bounded next-loop decision for RAW target representation.

## Candidate Variants

- Variant A: Post-hoc RAW structure audit of the P19 checkpoint.
- Variant B: Run another longer P19-style training horizon.
- Variant C: Immediately implement split RAW heads by guessed token fields.
- Variant D: Jump to ordered C3 generation.

## Local Verification Matrix

- Variant A: Passes if it can attribute RAW misses by token field, frequency bucket, composite/order-signature status, and rank depth without changing training.
- Variant B: Passes only if blind longer training is likely to cross the RAW unigram gap; P19 deliberately rejected this due to weak evidence.
- Variant C: Passes only if the split dimensions are already known; current evidence does not identify them.
- Variant D: Passes only if bag objectives are clearly exhausted; P19 shows `REF`/`RES` are learnable, so this is premature.

## Selected Variant

- Selected: Variant A, post-hoc RAW structure audit.
- Rejected: B because P19 called for diagnosis before more runtime; C because split dimensions are not yet grounded; D because it is a larger grammar change before RAW failure is localized.
- Why this is the smallest useful test: it adds observability around an existing checkpoint and sidecar, and can fail quickly without new model training.

## Selection Pressure

- Primary pressure: identify the dominant RAW failure mode using the P19 model-vs-unigram gap.
- Guard pressure: preserve C3 legality by using target-side labels only; do not consume C3 labels as model inputs.
- Runtime pressure: complete on the existing P19 eval slice in under 2 minutes on CPU.
- Kill pressure: if RAW misses are diffuse and not rank-near, do not spend more effort on bagged RAW auxiliaries.

## Research Question

Why does P19 recover `REF` and `RES` but still underperform reduced unigram on `RAW`, and what is the smallest justified mutation of the C3 target representation?

## Closest Analogies / Novelty Layer

- Closest analogies: multilabel error slicing, token-frequency baseline analysis, target-factorization probes, retrieval rank diagnostics.
- Relevant taxonomy bucket: local verification and representation audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering observability for the C3 representation, not a novel modeling method.

## Minimal Change

Add a reusable evaluator that loads a C3 auxiliary checkpoint and computes RAW-only diagnostics:

- RAW target counts, model hits, unigram hits, overlap, and misses.
- RAW token parser fields from strings like `RAW|D:12:2:0:0` and `RAW|D:0:0:0:15:OE3,E0,E1,E2`.
- RAW grouping by field count, field values, order-signature/composite tail, and train-frequency bucket.
- RAW rank-depth diagnostics for targets inside top-20, top-50, top-100, top-200, and missing from those cutoffs.
- Top missed and top model-only RAW tokens for qualitative inspection.

## Files Likely to Change

- `src/pulsefield_model/evals/c3_raw_structure_audit.py`
- `tests/evals/test_c3_raw_structure_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/c3_auxiliary_diagnostics.py`
- `src/pulsefield_model/evals/c3_reduced_sidecar.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_result_report.md`

## Dataset Slice

Use the same eval split as P19:

- config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- checkpoint: `artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt`
- sidecar: reduced top-512-per-kind sidecar referenced by the config
- eval windows: expected `142`
- positive RAW labels: expected `854`

## Baseline / Comparator

Primary comparator is the reduced train-unigram top-K predictor already used by P19 diagnostics.

## Primary Metric

RAW model-vs-unigram hit gap by structured RAW bucket at K=20.

## Secondary Metric

- RAW target rank coverage at K=20, 50, 100, and 200.
- RAW miss concentration by frequency bucket and parsed token fields.
- Model-only RAW predictions by frequency and token field.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_raw_structure_audit \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_raw_structure_audit.py tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py -q
```

## Qualitative Check

Inspect top RAW misses and bucket tables for whether the model misses semantically coherent groups or merely loses to high-frequency unigram labels.

## Positive Signal

At least one RAW subgroup explains a large share of the gap or shows rank-near behavior where many missed targets appear by K=50 or K=100. That would justify a smaller structured RAW mutation.

## Negative Signal

RAW misses are diffuse across fields and frequency buckets, with targets rarely appearing even by K=200. That would argue against another bag-head refinement.

## Kill Criteria

Kill blind reduced-kind-head horizon extension if the audit shows RAW misses are not rank-near and not concentrated in a tractable structural bucket.

## Expected Failure Modes

- RAW token parsing may be too syntactic to expose real semantics.
- The reduced top-512 RAW vocabulary may hide failures outside the reduced sidecar.
- Top-K bag ranks may not reflect ordered side-stream generation difficulty.
- The P19 eval slice is small, so rare buckets may be noisy.

## Confounders

- Train-unigram is strong for RAW because RAW labels include frequent literal payloads.
- Model and unigram may hit different RAW labels in the same window.
- Kind heads concatenate logits for diagnostics; per-kind scoring can differ from full concatenated scoring.
- RAW token strings encode fields whose exact semantic names are not fully asserted here.

## Expected Runtime / Runtime Budget

Expected under 2 minutes on CPU. Stop if the P19 checkpoint, config, or reduced sidecar is missing.

## Result Interpretation Plan

- Positive result would suggest: split RAW into one or two grounded subheads, or add RAW-specific calibration/ranking diagnostics.
- Negative result would suggest: stop refining bagged RAW and design an ordered C3 target grammar card.
- Ambiguous result would require: run the same audit on a larger eval slice or include full exact sidecar labels.
- Human owner decides: whether RAW split or ordered generation is the next research direction.
- Next-loop action if positive: RAW split-head Experiment Card.
- Next-loop action if negative: ordered C3 target-grammar Experiment Card.
- Next-loop action if ambiguous: larger-slice RAW audit or full-sidecar RAW coverage audit.

## Result Log Template

- Experiment: C3 RAW-structure audit
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
- Remaining ambiguity: the audit can identify structural RAW failure modes, but it will not by itself prove full-pipeline generation benefit.

## Next-Loop Action

- If positive: create a RAW split-head card.
- If negative: create an ordered C3 target-grammar card.
- If ambiguous: repeat on a larger or full-sidecar diagnostic slice.

## Novelty Notes

- Closest analogies: error slicing, representation factorization probes, multilabel rank diagnostics.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation for observability and target-shape selection.
