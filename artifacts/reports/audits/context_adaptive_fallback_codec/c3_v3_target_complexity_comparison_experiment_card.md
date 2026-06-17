# C3/v3 Target Complexity Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: C3 hardening proved chart-local fallback-substream compression, but ordered RAW field grammar and mapper-side auxiliary probes surfaced sequence, calibration, and target-leakage costs. The prior ordered RAW card explicitly left an ambiguous next-loop action: compare C3, v3, and v2.1 target complexity before spending more training runtime.
- Acceptance source, if any: active C3/v3 full-pipeline goal plus `c3_ordered_raw_field_grammar_probe_result_report.md`, `c3_v3_post_branch_route_synthesis_result_report.md`, and `target_grammar_v3_full_dataset_audit_result_report.md`.
- Source snapshot / evidence grade: strong evidence for C3 codec legality and compression; strong evidence for v3 full-dataset reconstruction and local target compression; strong evidence that current C3 input conditioning is not inference-legal; medium evidence that ordered C3 target grammar is too expensive.

## Hypothesis

If C3 is still mainly a side-stream codec rather than a practical emitted mapper target, then a three-way complexity gate will show that C3 has stronger compression evidence but worse target-sequence/state cost than v3. In that case, the next full-pipeline work should mutate toward v3/v2.1 grammar repair rather than more C3 mapper-side integration.

## Root Objective

Decide whether to keep pursuing C3 as a mapper-facing target grammar, or pivot the next experiments toward v3/v2.1 grammar improvement.

## Goal Decomposition

- Subgoal 1: Compare C3 side-stream and ordered RAW target costs against v3 and v2.1 using existing authoritative audit artifacts.
- Subgoal 2: Preserve C3's proven representation boundary: target-derived side-stream tokens are not treated as production inference inputs.
- Subgoal 3: Produce a concrete route decision before more mapper training or rollout experiments.

## Candidate Variants

- Variant A: Run another C3 auxiliary or kind-head training job. Rejected because P17-P23 did not beat unigram comparators and P10 killed target-derived input conditioning.
- Variant B: Build a full C3 emitted target grammar immediately. Rejected because ordered RAW splitting failed sequence/bit gates and C3 has cross-window reference semantics.
- Variant C: Three-way target complexity audit over existing artifacts. Selected because it can falsify C3 target-grammar promotion without new training.
- Variant D: Jump directly to v3 training mutation. Deferred until this audit records why C3 is no longer the best next mapper-target path.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Longer C3 auxiliary training | C3 beats unigram and improves main loss | More training only improves absent-label calibration |
| B | Full C3 target grammar design | Online replay/state can be defined simply | Cross-window references require complex replay |
| C | Artifact-only complexity audit | C3 target costs are bounded and competitive with v3 | C3 combined/ordered sequence and replay costs exceed v3 |
| D | v3 mutation | Training/rollout improves fixed-slice gates | Skips unresolved C3 evidence boundary |

## Selected Variant

- Selected: Variant C, artifact-only three-way target complexity audit.
- Rejected: A and B because the last C3 diagnostics showed diminishing mapper-side returns.
- Deferred: D until the audit explicitly closes or preserves the C3 route.
- Why this is the smallest useful test: it uses already generated full-cache artifacts, requires no retraining, and directly answers whether C3 should keep consuming mapper-target research time.

## Selection Pressure

- Primary pressure: prefer the representation with exact reconstruction, lower/bounded target cost, and simple online teacher-forcing/inference semantics.
- Guard pressure: no target-derived C3 side-stream can become an inference input.
- Runtime pressure: no new full-cache tokenization or training; read existing JSON artifacts only.
- Kill pressure: if C3 requires side-stream replay or sequence expansion that is materially worse than v3's local event grammar, stop C3 mapper-target pursuit for now.

## Research Question

Does C3 still deserve the next mapper-facing target grammar experiment, or has its marginal value shifted to codec-side evidence while v3/v2.1 grammar repair is the practical path?

## Closest Analogies / Novelty Layer

- Closest analogies: LZ side-channel codecs, target grammar selection, factorized symbolic music event targets, representation cost audits.
- Relevant taxonomy bucket: representation/interface selection after local verification.
- Novelty layer, if any: representation-policy layer, not a novel compression algorithm.
- Representation novelty vs engineering variation: the C3 chart-local fallback reference policy is the representation claim; this audit is engineering selection pressure for pipeline integration.

## Minimal Change

Add a small evaluator that reads existing C3 and v3 audit summaries, extracts comparable target-complexity metrics, computes guard results, and writes one summary JSON plus one result report.

No model, tokenizer, replay, training, rollout, or production-default behavior changes are part of this card.

## Files Likely To Change

- `src/pulsefield_model/evals/c3_v3_target_complexity_comparison.py`
- `tests/evals/test_c3_v3_target_complexity_comparison.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`

## Dataset Slice

Existing full-cache artifacts only:

- C3 hardening/full side stream: 9,242 beatmaps from the C3 audit cache.
- C3 exact mapper-window sidecar: 173,268 mapper windows.
- C3 ordered RAW field grammar full sidecar audit.
- v3 full-dataset audit: 174,515 mapper windows.
- v2.1 sparse target stream as the v3 audit baseline comparator.

## Baseline / Comparator

- Baseline: v2.1 sparse lane-action target grammar.
- Candidate 1: v3 local event-token target grammar.
- Candidate 2: C3 exact side-stream and ordered RAW field grammar candidate.

## Primary Metric

Route decision from target complexity guards:

- v3 exact reconstruction and positive bit/token reduction;
- C3 exact reconstruction and sidecar generation legality;
- C3 combined/ordered sequence cost and cross-window reference cost versus v3's local target contract.

## Secondary Metric

- C3 combined main+side token ratio.
- C3 ordered/exact sequence ratio.
- C3 ordered bits per original token versus exact C3 bits per token.
- C3 target/reference cross-window span rates.
- v3 token reduction, total-bit reduction, and reconstruction mismatch count.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.c3_v3_target_complexity_comparison \
  --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json \
  --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_result_report.md
uv run --group dev pytest tests/evals/test_c3_v3_target_complexity_comparison.py -q
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json >/dev/null
git diff --check
```

## Guard Check

- The evaluator must not read beatmap targets directly; it only reads prior audit artifacts.
- It must not change mapper defaults.
- It must not claim v3 replacement readiness from representation-only metrics.
- It must explicitly mark C3 target-derived input conditioning as not production-legal.

## Qualitative Check

The report must state which route is favored next and why: continue C3 target grammar, mutate C3, or pivot to v3/v2.1 grammar repair.

## Positive Signal

`TEST_NEXT_C3_TARGET` only if C3's target-sequence cost is bounded enough to compete with v3 and its replay/state cost is described without target leakage or complex future lookup.

## Negative Signal

`MUTATE_TO_V3_GRAMMAR_REPAIR` if C3 remains reconstructive but side-stream-shaped, sequence-expanded, or cross-window-stateful while v3 remains local, shorter, and fully reconstructive.

## Kill Criteria

- Kill immediate C3 mapper-target pursuit if ordered C3 expands sequence length beyond `2.0x` exact side-stream tokens and worsens charged bits.
- Kill immediate C3 mapper-target pursuit if cross-window reference semantics remain required for a material share of spans.
- Kill any C3 route that requires target-derived labels as inference inputs.

## Expected Failure Modes

- Metrics may not be perfectly normalized between C3 codec bits and v3 target bits.
- C3's codec compression win may remain scientifically strong while still losing as an emitted target grammar.
- v3 may win representation simplicity while still failing trained rollout quality.

## Confounders

- This is a target/interface audit, not a mapper-quality audit.
- Existing artifacts come from different harnesses, so the decision should be about route selection, not exact absolute compression ranking.
- v2.1 remains the current safer default until v3 mapper training and inference pass rollout gates.

## Expected Runtime / Runtime Budget

Expected runtime: under 10 seconds plus tests. Stop if any required input artifact is missing or malformed.

## Result Interpretation Plan

- Positive result would suggest: create a bounded emitted C3 target grammar card.
- Negative result would suggest: stop C3 mapper-target work for now and focus on v3/v2.1 grammar repair.
- Ambiguous result would require: add missing normalization metrics before choosing.
- Human owner decides: whether C3's stronger codec result is worth extra implementation complexity.
- Next-loop action if positive: C3 emitted-target grammar smoke card.
- Next-loop action if negative: v3/v2.1 grammar repair card based on fixed-slice rollout failure clusters.
- Next-loop action if ambiguous: add a stricter common-slice target-cost audit.

## Result Log Template

- Experiment: C3/v3 target complexity comparison
- Date:
- Commit / run id:
- Input artifacts:
- Runtime:
- v2.1 baseline metrics:
- v3 metrics:
- C3 side-stream metrics:
- C3 ordered RAW metrics:
- Primary route decision:
- Guard results:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: a negative C3 target-complexity result does not invalidate C3 as codec evidence; it only deprioritizes C3 as the next mapper-emitted target path.

## Next-Loop Action

- If positive: create and run a C3 emitted-target grammar smoke.
- If negative: focus the next experiment on v3/v2.1 grammar repair.
- If ambiguous: add stricter common-slice normalization before training.

## Novelty Notes

- Closest analogies: LZ side-stream codecs, residual target decomposition, symbolic music event grammar selection, and representation-cost gating.
- Novelty layer, if any: representation-policy layer.
- Representation novelty vs engineering variation: this audit is engineering selection pressure over existing representation candidates.
