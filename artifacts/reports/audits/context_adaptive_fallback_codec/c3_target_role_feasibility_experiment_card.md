# C3 Target-Role Feasibility Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P10 killed promotion of the current C3 side stream as production input conditioning because it is target-derived.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_full_pipeline_role_audit_result_report.md`.
- Source snapshot / evidence grade: strong sidecar correctness and role evidence; weak evidence for which legal target-side integration family should come next.

## Hypothesis

The exact C3 mapper-window side stream is more immediately usable as a target-side auxiliary signal or predicted plan than as a full replacement for the mapper token grammar.

## Root Objective

Choose the next legal full-pipeline C3 integration family after ruling out target-derived input conditioning.

## Goal Decomposition

- Subgoal 1: measure the exact C3 sidecar as a target signal: coverage, sequence size, token-kind mix, cap pressure, and vocabulary concentration.
- Subgoal 2: compare target-side implementation families: auxiliary target, two-stage predicted plan, and full target grammar replacement.
- Subgoal 3: select the smallest implementation family that moves C3 toward the full pipeline without target leakage.

## Candidate Variants

- Variant A: C3 auxiliary target head predicting window-level side-stream summaries or bag/sequence targets while keeping mapper output unchanged.
- Variant B: two-stage predicted C3 plan: predict C3 side-stream tokens first, then condition mapper on predicted C3.
- Variant C: full C3 target grammar replacement where the mapper emits `RAW`/`REF`/`RES` structure directly.
- Variant D: continue current pooled C3 input conditioning.

## Local Verification Matrix

- Variant A: requires no inference-time sidecar and can be measured using existing exact sidecar labels; risk is that it may regularize but not directly change output tokenization.
- Variant B: legal in principle, but requires a new plan generator and error-propagation audit before mapper conditioning.
- Variant C: most direct use of C3 as tokenization, but requires decode grammar, replay, and cross-window reference semantics.
- Variant D: rejected by P10 as target-derived input conditioning.

## Selected Variant

- Selected: data-only feasibility audit before choosing A, B, or C.
- Rejected: implementing A/B/C directly before measuring target shape is premature; D is killed.
- Why this is the smallest useful test: it uses the exact sidecar and existing reports to decide the next bounded code path without touching production code.

## Selection Pressure

- Primary pressure: identify whether C3 target labels are tractable enough for a small auxiliary-head implementation, or too sequence-heavy/semantic for that path.
- Guard pressure: do not promote target-derived input conditioning; do not claim quality improvement.
- Runtime pressure: local JSON/report analysis only.
- Kill pressure: if C3 labels are extremely sparse, extremely long, or dominated by huge open vocabulary, avoid auxiliary target and move to target-grammar decomposition instead.

## Research Question

What is the smallest legal target-side role for exact C3 that can move toward full-pipeline use?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary target feasibility audit, target-token redesign sizing, two-stage latent-plan sizing.
- Relevant taxonomy bucket: representation integration validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this classifies how to integrate an already-validated C3 representation; it does not claim a new representation result.

## Minimal Change

Add audit artifacts only. Analyze:

- exact C3 mapper-window sidecar JSON,
- exact sidecar report,
- P9/P10 reports,
- token vocabulary string prefixes and frequency concentration.

No production code changes are planned.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_target_role_feasibility_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_target_role_feasibility_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_target_role_feasibility_result_report.md`

## Read-Only Context Files

- `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_full_pipeline_role_audit_result_report.md`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`

## Dataset Slice

Use the full-cache exact sidecar:

- sidecar windows: expected `173,268`
- sidecar tokens: expected `3,203,904`
- token vocab size: expected `14,294`
- mapper window: 8 seconds

## Baseline / Comparator

Baseline legal full-pipeline mapper uses the current sparse v2.1 lane-action target stream.

Comparators:

- C3 auxiliary target: C3 labels are predicted as extra target supervision.
- C3 predicted plan: C3 labels are generated before mapper decoding.
- C3 target grammar: mapper emits C3 `RAW`/`REF`/`RES` tokens directly.

## Primary Metric

- Selected next implementation family: `auxiliary_target`, `two_stage_plan`, `target_grammar`, or `kill_c3_mapper_integration`.

## Secondary Metric

- Token coverage: windows with tokens, token count, tokens/window quantiles.
- Token-kind mix: `RAW`, `REF`, `RES`.
- Vocabulary shape: token vocab size, top-token concentration, kind-specific vocab sizes.
- Cap pressure: overflow rates at caps 64, 128, 256, 512, and 1024.
- Cross-window reference exposure from sidecar report.

## Verify Command / Evaluation Procedure

```bash
uv run python - <<'PY'
import json
from collections import Counter
from pathlib import Path
sidecar = json.loads(Path("artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json").read_text())
report = json.loads(Path("artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json").read_text())
print(len(sidecar["windows"]), len(sidecar["token_vocab"]), report["sidecar_stats"]["sidecar_token_count"])
PY
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
```

## Guard Check

- Do not edit production code.
- Do not run more mapper training.
- Do not reuse target-derived C3 labels as inference inputs.
- Keep the result as integration-family selection, not model-quality evidence.

## Qualitative Check

Inspect whether C3 labels are better suited to:

- a small auxiliary loss,
- a separate planner/predictor,
- or a full target grammar/tokenizer redesign.

## Positive Signal

- Sidecar target shape is compact enough for a bounded auxiliary-target card.
- Vocabulary concentration suggests a small head or compressed labels could be tested.
- Cross-window exposure is acknowledged before choosing a grammar path.

## Negative Signal

- Labels are too long/open-vocabulary for auxiliary supervision.
- Full target grammar replacement would require unresolved cross-window state.
- The audit cannot distinguish A/B/C.

## Kill Criteria

Kill immediate auxiliary-head implementation if sidecar labels are too sequence-heavy, too open-vocabulary, or too semantically stateful to be represented as a bounded target without first decomposing token kinds.

## Expected Failure Modes

- Top-token concentration may be low because `RAW`/`RES` payloads are high-cardinality.
- A bag target could discard order and span semantics.
- A full grammar path may require more decode-state design than this audit can cover.

## Confounders

- Sidecar token IDs are local to the sidecar vocabulary and encode string payloads; ID frequency alone may hide semantic structure.
- The same window can contain target/reference spans whose semantics cross 8s mapper boundaries.
- Auxiliary-target success would not itself prove generation quality.

## Expected Runtime / Runtime Budget

Expected under 10 minutes. Stop if the exact sidecar or exact sidecar report is missing.

## Result Interpretation Plan

- Positive result would suggest: create a C3 auxiliary-target smoke Experiment Card.
- Negative result would suggest: create a C3 target-grammar decomposition card before model changes.
- Ambiguous result would require: add sidecar token-kind/timing metadata to make the target labels inspectable.
- Human owner decides: whether to prioritize direct target grammar or auxiliary regularization.
- Next-loop action if positive: auxiliary target smoke card.
- Next-loop action if negative: target grammar decomposition card.
- Next-loop action if ambiguous: sidecar metadata enrichment card.

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
- Code execution allowed after this card: yes, for audit artifacts only
- Closed loop complete: yes
- Remaining ambiguity: the model change is intentionally deferred until target shape is measured.

## Next-Loop Action

- If positive: auxiliary target smoke card.
- If negative: target grammar decomposition card.
- If ambiguous: sidecar metadata enrichment card.

## Novelty Notes

- Closest analogies: auxiliary target sizing and target-token redesign feasibility.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is integration triage for C3, not a new C3 result.
