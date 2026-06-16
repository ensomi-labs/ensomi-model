# C3 RAW Context-Bucket Probe Experiment Card

## Hypothesis

P22 showed that RAW exact-token recall trails unigram, while factor marginals retain usable structure. The next failure may be context mixing: the same RAW field values may need to be interpreted under local density, chord, or hold context. If so, stratifying RAW factor recovery by control-v3 context buckets should reveal that weak fields are concentrated in specific density/chord/hold regimes rather than uniformly weak.

## Root Objective

Harden the C3 target-side path enough to decide whether RAW should become a factorized auxiliary target, a richer target grammar, or a killed mapper-side branch.

## Goal Decomposition

- Verify that the existing P19 checkpoint, reduced C3 sidecar, and eval split remain loadable.
- Add post-hoc bucket diagnostics without changing training, inference, or model inputs.
- Measure whether RAW factor weakness is context-local and actionable.
- Preserve the legality constraint: C3 labels stay target-side only.

## Candidate Variants

- A: Exact-token calibration rerun. Rejected because P21 already showed global kind-logit calibration overpromotes RAW and harms non-RAW.
- B: RAW factor-head training. Rejected for this turn because P22 did not prove enough field-level lift versus unigram.
- C: RAW context-bucket post-hoc probe. Selected because it is fast, legal, and can fail before adding heads.
- D: Full target grammar decomposition. Deferred because it is larger and should use context evidence from this probe.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Compare with P21 report | Held-out hits without non-RAW damage | Non-RAW preservation below gate |
| B | Train 100-500 step smoke | Factor heads beat unigram in weak fields | Same weak fields trail unigram |
| C | Post-hoc bucket probe | At least one context bucket shows field lift or clear concentration of misses | Buckets show uniformly weak fields |
| D | Grammar card only | Explicit replay semantics defined | Needs cross-window reference semantics first |

## Selected Variant

C: post-hoc RAW context-bucket probe.

## Selection Pressure

This variant is the cheapest way to decide whether RAW factorization should mutate into context-conditioned heads or whether the remaining issue is ordered grammar/replay semantics. It does not spend training runtime and does not create a production inference path from target-derived C3 labels.

## Minimal Change

Extend `c3_raw_factor_probe` to compute mean `density_level`, `chord_ratio`, and `hold_occupancy` from the existing control-v3 target source for each eval mapper window, bucket those means, and report RAW field recall/miss concentration per bucket.

## Files Likely To Change

- `src/pulsefield_model/evals/c3_raw_factor_probe.py`
- `tests/evals/test_c3_raw_factor_probe.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_result_report.md`

## Dataset Slice

Use the same P22 config/checkpoint:

- config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- checkpoint: `artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt`
- eval split: 142 windows from that config

## Baseline / Comparator

P22 raw factor probe:

- exact full-vocab RAW recall@20: `0.084309`
- exact RAW-only recall@20: `0.185012`
- max-marginal joint factor coverage@5: `0.553864`
- improved fields@3: `field_3`
- weak fields@3: `field_1`, `field_2`, `field_4`

Within this probe, compare model max-marginal factor recall against train-unigram factor recall in each context bucket.

## Primary Metric

Per-bucket model-minus-unigram RAW factor recall@3 for `field_1`, `field_2`, `field_3`, and `field_4`.

## Secondary Metric

Per-bucket RAW label count, positive sample count, and missed weak-field value concentration.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_raw_factor_probe \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_result_report.md \
  --batch-size 2 \
  --top-k 1 3 5
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_c3_raw_factor_probe.py -q
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_context_bucket_probe_summary.json >/dev/null
```

## Qualitative Check

The report must state whether context buckets make RAW factor weakness local/actionable or broadly uniform.

## Positive Signal

At least two weak fields show a bucket with nontrivial support and model-minus-unigram recall@3 greater than zero, or miss concentration clearly identifies one context bucket family for a next targeted factor grouping.

## Negative Signal

Weak fields remain below unigram across all supported density/chord/hold buckets and misses are not concentrated enough to guide a smaller mutation.

## Kill Criteria

- Existing P22 overall metrics regress because of accounting changes.
- Bucket context cannot be loaded from the same eval split.
- Context buckets use target-derived C3 labels as model inputs.
- No supported bucket provides a stronger signal than the original factor probe.

## Expected Failure Modes

- Eval split is too small for fine buckets.
- Context buckets expose only generic high-frequency effects.
- Density/chord/hold means are too coarse relative to C3 RAW token syntax.

## Expected Runtime / Runtime Budget

Expected runtime: under 1 minute on CPU after checkpoint load. Stop if runtime exceeds 5 minutes or if dataset/context loading fails.

## Confounders

- Context features are target-side supervision signals, so this is a diagnostic for target design, not an inference input proof.
- Eval size is only 142 windows.
- RAW labels are capped by the reduced sidecar top-512-per-kind vocabulary.

## Result Interpretation Plan

- `TEST_NEXT`: context buckets reveal actionable field lift or concentrated misses; create a bounded context-conditioned RAW factor-head card.
- `MUTATE`: context buckets reveal some localization but not enough for model changes; refine grouping or move toward ordered grammar.
- `KILL`: context buckets are uniformly weak; do not spend more runtime on simple RAW factor heads.

## Result Log Template

- command:
- elapsed:
- overall P22 metric preservation:
- strongest positive buckets:
- weakest buckets:
- decision:
- next-loop action:

## Next-Loop Action

Choose between context-conditioned RAW factor heads and ordered C3 target grammar decomposition.

## Closest Analogies And Novelty Layer

Closest analogies: multitask auxiliary labels, factorized categorical targets, and context-stratified error analysis. The novelty layer, if any, is not the general method; it is applying chart-local C3 fallback side-stream structure as target-side supervision for osu!mania mapper tokenization.
