# Target Grammar v3 500-Step Failure-Cluster Diagnostic Experiment Card

## Hypothesis

The v3 500-step fixed-slice wide-audit failures are not one generic failure. They cluster into a small number of actionable modes: second-window starvation on mid/high difficulty 160ms grids, rigid same-audio grid copying, low-reference or zero-reference boundary duplication, and easier-chart overproduction.

## Root Objective

Decide the next v3 mutation after the 32-case fixed-slice `MUTATE` result without running another expensive training or rollout pass.

## Goal Decomposition

1. Separate legality from quality: all v3 500-step rollouts were legal, so diagnose timing/event distribution failures rather than grammar validity.
2. Separate model starvation from reference sparsity: compare generated first/second-window shares with reference first/second-window shares.
3. Separate rigid timing from useful alignment: identify when high dominant-spacing ratio coexists with high F1 versus low F1.
4. Separate count failures from placement failures: classify undergeneration, overgeneration, duplicate same-time events, and boundary-heavy emissions.
5. Convert clusters into one bounded next-loop card recommendation.

## Candidate Variants

### A. Existing-Artifact Failure Cluster Diagnostic

Read the existing 32 rollout summaries plus the wide-audit summary. Enrich each case with reference-event distribution, duplicate-time counts, boundary counts, and logit confidence diagnostics already captured during rollout.

### B. Immediate Loss/Training Calibration

Skip diagnostics and change training loss or decode policy based on the visible starvation/rigidity symptoms.

### C. New Rollout Probe Sweep

Run more decode-policy or checkpoint rollouts on selected bad cases.

### D. Full Dataset v3 Runtime Audit

Scale the current v3 checkpoint to a wider or full dataset rollout audit.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | Parse all 32 existing wide-audit runs and assign cluster labels | Every failed case gets at least one interpretable label; cluster counts point to one next-loop mutation | Labels are diffuse, contradictory, or mostly unassigned |
| B | Compare with wide-audit aggregate only | A single dominant failure is already obvious | Multiple overlapping failures remain unresolved |
| C | Select bad cases and run new rollouts | New rollouts expose a failure not visible in current artifacts | Runtime cost increases before explaining existing failures |
| D | Launch broader audit | Full-scale failure pattern needed before next mutation | Existing fixed-slice failures are still unclassified |

## Selected Variant

Variant A: existing-artifact failure cluster diagnostic.

## Selection Pressure

Variant A is the cheapest test that can fail quickly and directly answers the wide-audit next-step request. Variants B, C, and D spend implementation or runtime before identifying whether the next mutation should target event-count calibration, second-window continuity, duplicate/boundary suppression, or same-audio rigidity.

## Minimal Change

Add report artifacts only:

- `target_grammar_v3_500step_failure_cluster_diagnostic_experiment_card.md`
- `target_grammar_v3_500step_failure_cluster_diagnostic_result_report.md`
- `target_grammar_v3_500step_failure_cluster_diagnostic_summary.json`

No model, tokenizer, training, decode, or runtime source code changes.

## Files Likely To Change

Report artifacts under:

- `artifacts/reports/audits/mapper_v2_1_grammar/`

Read-only context:

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/rollouts/*_summary.json`
- referenced source beatmap files from the manifest or wide-audit summary

## Dataset Slice

All 32 unique beatmaps from:

`artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`

Use the same `16000ms` prefix window as the wide audit.

## Baseline / Comparator

Comparator is the existing v3 500-step fixed-slice wide audit:

- all legal: `true`
- mean F1 @100ms: `0.640`
- median F1 @100ms: `0.679`
- starved cases: `11`
- rigid cases: `7`
- positive signal passed: `false`

## Primary Metric

Actionable cluster assignment coverage: fraction of non-pass-like cases assigned to at least one concrete failure cluster.

## Secondary Metrics

- cluster counts and overlaps;
- generated versus reference second-window share delta;
- undergeneration and overgeneration counts;
- dominant spacing and dominant-spacing ratio;
- same-time duplicate count and duplicate/nonincreasing spacing ratio;
- boundary-event count and ratio;
- logit event top-1/top-k ratios, best-event-rank median, and margin median;
- difficulty-band and same-audio-family clustering.

## Verify Command Or Evaluation Procedure

1. Load the 32-case wide-audit summary JSON.
2. Load each rollout summary JSON referenced by the wide audit.
3. Use generated timepoints already stored in rollout summaries.
4. Use wide-audit reference counts and first/last reference previews; if available, use source beatmap paths to compute full reference first/second-window shares.
5. Assign rule-based cluster labels.
6. Write a Markdown result report and JSON summary.
7. Validate JSON with `python3 -m json.tool`.

## Guard Check

- Do not rerun training or rollout.
- Do not change source code.
- Preserve the original wide-audit metrics.
- Verify all 32 cases are represented exactly once.

## Qualitative Check

Inspect representative cases for each cluster and confirm labels match visible symptoms from the wide-audit report: Camellia/Billx/USao starvation, Namirin same-audio rigid grid, Goreshit zero-reference boundary duplicate, and easier-chart overproduction.

## Positive Signal

At least 90% of non-pass-like cases receive an interpretable failure label, and the largest clusters imply a concrete next-loop mutation.

## Negative Signal

Most failures cannot be explained by the available artifacts, or the cluster labels conflict with the wide-audit worst-case tables.

## Kill Criteria

Kill immediate broader rollout scaling if this diagnostic confirms multiple unresolved quality clusters. Proceed to a targeted mutation card instead.

## Expected Failure Modes

- The rollout summary may only store reference previews, limiting reference distribution diagnostics.
- Zero-reference prefixes make F1 incomparable and must be handled separately.
- Rigid high-F1 cases may be acceptable locally but still undesirable for replacement readiness.
- Logit confidence metrics may be too coarse to distinguish model uncertainty from decode-policy collapse.

## Expected Runtime / Runtime Budget

Under 2 minutes. Stop if all 32 rollout summaries cannot be loaded or if the diagnostic would require new model inference.

## Confounders

- This is a training-slice fixed-prefix diagnostic, not held-out evidence.
- The 16s prefix may exaggerate intro sparsity or boundary behavior.
- Same-audio cases can share generated timing patterns because the audio/control path is identical or similar.
- F1 @100ms does not distinguish musically acceptable alternate timing from wrong placement.

## Result Interpretation Plan

- If starvation/under-generation dominates, next card should target event-continuity or count calibration.
- If rigid high-F1 same-audio grids dominate, next card should target timing diversity/conditioning rather than legality.
- If boundary/duplicate artifacts dominate, next card should target decode cleanup or event-time uniqueness.
- If overproduction dominates easier charts, next card should target event-count calibration and difficulty conditioning.
- If no dominant cluster emerges, run a narrower new rollout probe with additional instrumentation.

## Result Log Template

- Commands/procedure:
- Loaded case count:
- Cluster definitions:
- Cluster counts:
- Cluster overlaps:
- Representative cases:
- What passed:
- What surfaced:
- Decision:
- Next-loop action:

## Next-Loop Action

Create one targeted mutation card based on the dominant cluster. Do not scale v3 to replacement training until the failure mode is reduced on the fixed-slice gate.

## Closest Analogies And Novelty Layer

Closest analogies: error clustering, ablation triage, sequence-model exposure-bias diagnostics, and rhythm-generation timing/count calibration.

Novelty layer: none claimed. This is engineering evidence collection for the v3 target-grammar replacement path, not a representation novelty claim.
