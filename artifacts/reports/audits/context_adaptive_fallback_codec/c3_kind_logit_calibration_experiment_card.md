# C3 Kind-Logit Calibration Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P20 showed the P19 RAW gap is much larger in full concatenated logits than in RAW-only logits, and many missed RAW labels are rank-near by K=50/K=200.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_raw_structure_audit_result_report.md`.
- Source snapshot / evidence grade: strong evidence for C3 codec correctness and target-side learnability; medium evidence that RAW failure is partly cross-kind calibration rather than pure RAW separability; weak evidence for any specific training-time fix.

## Hypothesis

If P20's RAW blocker is mainly kind-head calibration, then a simple post-hoc transformation of the P19 logits should improve full-vocab top-20 C3 recovery, especially RAW recall@20, without changing model weights. If no small calibration improves held-out recall, the next useful mutation is a structured RAW target split rather than more global kind-head tuning.

## Root Objective

Move C3 toward legal full-pipeline use by deciding whether the reduced kind-head auxiliary path needs calibration/ranking changes or RAW target factorization.

## Goal Decomposition

- Subgoal 1: Reuse the P19 checkpoint and reduced sidecar without retraining.
- Subgoal 2: Collect eval logits and targets once, preserving target-side-only C3 legality.
- Subgoal 3: Sweep small post-hoc transformations and compare against identity and reduced unigram on a held-out diagnostic split.

## Candidate Variants

- Variant A: RAW-only bias sweep, adding a scalar to RAW logits before top-K.
- Variant B: Per-kind bias sweep over RAW/REF/RES scalar offsets.
- Variant C: Per-kind scale plus bias sweep over a small grid.
- Variant D: Implement a new train-time calibration loss or split head immediately.

## Local Verification Matrix

- Variant A: Passes if RAW recall@20 improves without collapsing non-RAW recovery; cheap but may be too narrow.
- Variant B: Passes if global kind competition is the issue; still cheap and directly tests P20's cross-kind signal.
- Variant C: Passes if logits need both offset and dispersion correction; slightly broader but still post-hoc and bounded.
- Variant D: Rejected unless A-C show a meaningful calibration signal; otherwise it changes training before proving calibration is the right lever.

## Selected Variant

- Selected: A-C in one post-hoc calibration sweep, with identity and unigram comparators.
- Rejected: D because the current question can fail quickly without training.
- Why this is the smallest useful test: it answers whether calibration can recover P20's rank-near RAW labels using existing logits and no model changes.

## Selection Pressure

- Primary pressure: improve held-out full-vocab model recall@20 and RAW recall@20 over identity.
- Guard pressure: do not use target-derived C3 labels as model inputs; do not claim generation or mapper-quality improvement.
- Runtime pressure: complete on the existing P19 eval split in under 2 minutes on CPU.
- Kill pressure: if no calibrated transform improves held-out recall by at least 5 hits or closes at least 25% of the identity-vs-unigram RAW gap, stop calibration and move to RAW split/factorization.

## Research Question

Is P19's remaining reduced-kind-head failure mainly a post-hoc kind-logit calibration problem, or does RAW require a structured target representation?

## Closest Analogies / Novelty Layer

- Closest analogies: logit calibration, temperature scaling, class-prior adjustment, multilabel ranking threshold tuning.
- Relevant taxonomy bucket: local verification of representation/model-interface failure.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation; this tests whether a known calibration mechanism explains the observed C3 target failure.

## Minimal Change

Add a reusable evaluator that:

- loads the P19 model/config/reduced sidecar,
- collects eval logits and target token sets,
- scores identity model logits and reduced train-unigram top-K,
- splits eval samples into calibration and held-out partitions by stable alternating index,
- selects the best transform on the calibration partition,
- reports held-out metrics for identity, selected calibration, oracle-best-on-heldout diagnostic, and unigram,
- writes JSON and Markdown reports.

## Files Likely to Change

- `src/pulsefield_model/evals/c3_kind_logit_calibration.py`
- `tests/evals/test_c3_kind_logit_calibration.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/c3_auxiliary_diagnostics.py`
- `src/pulsefield_model/evals/c3_raw_structure_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_summary.json`

## Dataset Slice

Use the same P19 eval split:

- config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- checkpoint: `artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt`
- sidecar: reduced top-512-per-kind sidecar referenced by config
- expected eval windows: `142`
- calibration partition: alternating eval samples with even sample index
- held-out partition: alternating eval samples with odd sample index

## Baseline / Comparator

- Identity P19 logits.
- Reduced train-unigram top-K comparator from P19 diagnostics.
- Oracle-best-on-heldout calibration is reported only as a diagnostic upper bound, not as selected evidence.

## Primary Metric

Held-out model micro recall@20 over all reduced C3 positive labels.

## Secondary Metric

- Held-out RAW recall@20.
- Held-out REF and RES recall@20.
- Full-vocab top-20 predicted kind mix.
- Absolute hit improvement over identity.
- RAW gap closure versus reduced unigram.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_kind_logit_calibration \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_kind_logit_calibration.py tests/evals/test_c3_raw_structure_audit.py tests/evals/test_c3_auxiliary_diagnostics.py -q
```

## Qualitative Check

Inspect the selected transform. A useful calibration should be small and interpretable, not an extreme bias that simply floods top-K with RAW and erases REF/RES.

## Positive Signal

On held-out samples, selected calibration improves total recall@20 by at least 5 hits or closes at least 25% of the RAW identity-vs-unigram gap while preserving non-RAW recall within 10% relative of identity.

## Negative Signal

Selected calibration does not improve held-out recall, or it improves RAW only by collapsing `REF`/`RES` recovery.

## Kill Criteria

Kill post-hoc/global calibration as the next C3 path if selected calibration fails the positive signal and the oracle-heldout best is also weak. Move to RAW split/factorization.

## Expected Failure Modes

- The P19 eval split is small, so held-out estimates are noisy.
- Calibration may overfit the calibration half and fail held-out.
- A global kind bias may not address field-specific RAW underperformance.
- A post-hoc transform may improve diagnostics but not translate to train-time robustness.

## Confounders

- Unigram is a strong comparator for frequent RAW labels.
- Sample-level positive label counts vary widely.
- The selected transform operates on already-trained logits; train-time calibration might behave differently.
- Held-out split is diagnostic, not a fully independent dataset.

## Expected Runtime / Runtime Budget

Expected under 2 minutes on CPU. Stop if checkpoint/config/sidecar loading fails.

## Result Interpretation Plan

- Positive result would suggest: add a train-time calibration/ranking card for C3 kind heads.
- Negative result would suggest: create a RAW split/factorization card based on P20 buckets.
- Ambiguous result would require: repeat on a larger eval slice or use train/calibration/valid partitions.
- Human owner decides: whether calibration is worth a training card or whether to move directly to RAW factorization.
- Next-loop action if positive: calibrated C3 kind-head training card.
- Next-loop action if negative: RAW split-head/factorization card.
- Next-loop action if ambiguous: larger-slice calibration audit.

## Result Log Template

- Experiment: C3 kind-logit calibration audit
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
- Remaining ambiguity: even a positive post-hoc calibration result would not prove full-pipeline generation benefit; it only selects the next legal target-side training mutation.

## Next-Loop Action

- If positive: write a calibrated kind-head training Experiment Card.
- If negative: write a RAW split/factorization Experiment Card.
- If ambiguous: repeat the audit on a larger slice.

## Novelty Notes

- Closest analogies: logit calibration, temperature scaling, class-prior biasing.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation for C3 auxiliary target diagnostics.
