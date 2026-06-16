# C3 RAW Factor-Marginal Probe Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P21 rejected global kind-logit calibration because viable non-RAW-preserving calibration did not improve held-out recovery, while unconstrained calibration only worked by overpromoting RAW and damaging REF/RES.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_kind_logit_calibration_result_report.md` and `c3_raw_structure_audit_result_report.md`.
- Source snapshot / evidence grade: strong evidence that C3 side-stream representation is legal/lossless; medium evidence that reduced kind heads learn C3 structure; strong evidence that current RAW exact-token bag recovery is the blocker; weak evidence for whether RAW factor heads would be learnable.

## Hypothesis

Even when P19 misses exact RAW tokens at top-20, its RAW logits may already rank the underlying RAW fields well. If field-level recall is high, RAW should be factorized into smaller target heads. If field-level recall is also weak, RAW factorization is unlikely to be enough and ordered C3 generation should move up.

## Root Objective

Move C3 toward legal full-pipeline use by determining whether the next target-side model change should split RAW into factor heads rather than keep exact RAW token labels.

## Goal Decomposition

- Subgoal 1: Reuse P19 checkpoint logits without retraining.
- Subgoal 2: Parse reduced RAW tokens into syntactic fields from strings like `RAW|D:12:8:0:0`.
- Subgoal 3: Score field-level top-K recovery by marginalizing existing RAW token logits over shared field values.
- Subgoal 4: Decide whether a RAW factor-head training card is justified.

## Candidate Variants

- Variant A: Data-only RAW factor vocabulary audit.
- Variant B: Post-hoc RAW factor-marginal probe over existing P19 logits.
- Variant C: Directly implement RAW factor heads in the mapper and train.
- Variant D: Skip factor heads and design ordered C3 target grammar.

## Local Verification Matrix

- Variant A: Passes if factor vocab sizes are tractable, but does not test model signal.
- Variant B: Passes if it shows whether P19 hidden/logit signal already contains recoverable factor information; no training needed.
- Variant C: Rejected before B because it changes model/loss without proving factor signal.
- Variant D: Rejected before B because P19/P20 still show rank-near exact RAW labels, so a smaller factorization may answer the question first.

## Selected Variant

- Selected: Variant B, post-hoc RAW factor-marginal probe.
- Rejected: A as too weak alone; C as too large before signal verification; D as premature before testing factor recoverability.
- Why this is the smallest useful test: it uses existing logits and labels to test the exact factorization hypothesis with no new training runtime.

## Selection Pressure

- Primary pressure: field-level model recall@K should beat field-unigram recall@K for at least two high-value fields.
- Guard pressure: do not use C3 labels as model inputs; use only target-side labels and frozen P19 logits.
- Runtime pressure: complete on the P19 eval split in under 2 minutes on CPU.
- Kill pressure: if factor recall does not beat unigram and joint factor coverage is weak, do not implement RAW factor heads yet.

## Research Question

Does the P19 reduced kind-head model contain recoverable RAW factor information even when exact RAW token top-20 recall trails unigram?

## Closest Analogies / Novelty Layer

- Closest analogies: target factorization probes, marginal decoding, multilabel field heads, structured prediction error slicing.
- Relevant taxonomy bucket: local verification for representation factorization.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation for choosing the next C3 target-side representation.

## Minimal Change

Add a reusable evaluator that:

- loads the P19 model/config/reduced sidecar,
- collects eval logits and RAW target labels,
- parses RAW tokens into `field_1`, `field_2`, `field_3`, and `field_4`,
- aggregates RAW token logits into field-value scores using max and logsumexp marginalization,
- compares model field recall@1/3/5 against train-unigram field recall@1/3/5,
- reports joint factor coverage for exact RAW targets when all four fields are in their top-K lists,
- writes JSON and Markdown reports.

## Files Likely to Change

- `src/pulsefield_model/evals/c3_raw_factor_probe.py`
- `tests/evals/test_c3_raw_factor_probe.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/c3_auxiliary_diagnostics.py`
- `src/pulsefield_model/evals/c3_raw_structure_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_structure_audit_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_logit_calibration_summary.json`

## Dataset Slice

Use the same P19 eval split:

- config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- checkpoint: `artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt`
- sidecar: reduced top-512-per-kind sidecar referenced by config
- expected eval windows: `142`
- expected RAW labels: `854`

## Baseline / Comparator

- Identity exact RAW token recovery from P20/P19.
- Train-unigram field-value top-K baseline built from the P19 train split.
- RAW-only exact-token top-20 result from P20 as context, not as the primary comparator.

## Primary Metric

Model-vs-unigram field recall@3 for `field_1`, `field_2`, `field_3`, and `field_4` using max-marginal RAW token logits.

## Secondary Metric

- Field recall@1 and recall@5.
- Logsumexp-marginal field recall.
- Joint factor coverage at K=3 and K=5.
- Per-field target value counts and top misses.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_raw_factor_probe \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_result_report.md \
  --device cpu \
  --batch-size 2
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_raw_factor_probe.py tests/evals/test_c3_raw_structure_audit.py tests/evals/test_c3_auxiliary_diagnostics.py -q
```

## Qualitative Check

Inspect whether the fields that P20 identified as underperforming, especially `field_2`, become recoverable under factor marginals. If not, factorization is less promising.

## Positive Signal

At K=3, max-marginal model recall beats field-unigram recall on at least two fields and joint factor coverage@5 is materially above exact RAW token recall@20.

## Negative Signal

Field recalls do not beat unigram, or only easy fields improve while `field_2` remains weak.

## Kill Criteria

Do not implement RAW factor heads if the probe shows no field-level signal beyond unigram and joint factor coverage@5 remains close to exact RAW token recall@20.

## Expected Failure Modes

- Max marginalization may overstate factor recoverability by letting different exact tokens support different fields.
- Logsumexp may favor high-frequency field values and mimic unigram.
- The reduced top-512 RAW vocabulary may hide long-tail factor issues.
- Field names are syntactic, not final semantic names.

## Confounders

- `field_2` may encode lane/action masks whose difficulty depends on chord density.
- Multi-label windows can contain several RAW field values.
- Factor recovery does not prove ordered side-stream generation.
- Strong unigram field baselines may still be hard to beat for common fields.

## Expected Runtime / Runtime Budget

Expected under 2 minutes on CPU. Stop if checkpoint/config/sidecar loading fails.

## Result Interpretation Plan

- Positive result would suggest: create a RAW factor-head training Experiment Card.
- Negative result would suggest: move toward ordered C3 target grammar or richer context, not simple field heads.
- Ambiguous result would require: repeat on a larger/full sidecar slice or add per-density/chord buckets.
- Human owner decides: whether factor heads are worth a model change.
- Next-loop action if positive: RAW factor-head smoke/training card.
- Next-loop action if negative: ordered C3 target-grammar card.
- Next-loop action if ambiguous: larger RAW factor probe.

## Result Log Template

- Experiment: C3 RAW factor-marginal probe
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
- Remaining ambiguity: a positive factor-marginal probe would justify a factor-head training card, but would not itself prove generation quality or full-pipeline readiness.

## Next-Loop Action

- If positive: write and run a RAW factor-head smoke/training card.
- If negative: write an ordered C3 target-grammar card.
- If ambiguous: repeat on a larger RAW factor slice.

## Novelty Notes

- Closest analogies: marginal structured decoding, factorized target probes, multilabel field-head diagnostics.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation for C3 target-shape selection.
