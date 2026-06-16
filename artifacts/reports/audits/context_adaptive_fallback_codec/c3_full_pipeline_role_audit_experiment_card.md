# C3 Full-Pipeline Role Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P9 showed C3 side-stream conditioning is stable, but the exact C3 sidecar is derived from the target beatmap and incremental decode still rejects C3 conditioning.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_mapper_conditioning_longer_result_report.md`.
- Source snapshot / evidence grade: strong local artifacts for sidecar correctness; strong code evidence for current inference incompatibility; weak evidence for a legal generation-time C3 role.

## Hypothesis

The current C3 side-stream tensor path is valid as an offline target-side audit/conditioning probe, but it is not yet a legal full-pipeline mapper input because generation has no exogenous source for target-derived C3 tokens.

## Root Objective

Decide the next C3 integration direction before spending more training runtime: input conditioning, auxiliary target prediction, or target-token grammar replacement.

## Goal Decomposition

- Subgoal 1: verify whether current C3 sidecar rows are derived from target beatmap/fallback payloads.
- Subgoal 2: verify whether current inference/rollout can legally supply C3 side-stream tokens.
- Subgoal 3: map the observed state to the next bounded implementation family.

## Candidate Variants

- Variant A: continue longer mapper training with current pooled C3 side-stream conditioning.
- Variant B: add incremental decode support for externally supplied C3 conditioning.
- Variant C: audit current C3 role/provenance and decide whether it is a legal input or a target-side representation.
- Variant D: immediately redesign mapper output tokens around C3 `RAW`/`REF`/`RES`.

## Local Verification Matrix

- Variant A: technically easy, but may deepen an oracle-conditioning path if C3 tokens are target-derived.
- Variant B: only useful if a legal generation-time C3 source exists; otherwise it just makes target leakage easier to run.
- Variant C: smallest check that answers whether the current training probe can be promoted.
- Variant D: probably the right family if C3 is target-side, but too broad without a role audit.

## Selected Variant

- Selected: Variant C, full-pipeline role/provenance audit.
- Rejected: A and B are premature until legality is known; D is too broad before the role decision.
- Why this is the smallest useful test: it uses existing reports and code paths to decide the next integration family without changing production code.

## Selection Pressure

- Primary pressure: classify the current C3 path as legal input, target-side oracle probe, auxiliary target, or target grammar candidate.
- Guard pressure: cite concrete files, reports, and commands; do not claim quality improvement or novelty.
- Runtime pressure: local inspection and small runtime probes only.
- Kill pressure: if current C3 side stream is target-derived and not available in rollout, stop treating it as a production input.

## Research Question

Can the current exact C3 mapper-window side stream be used as a full-pipeline mapper input, or must it be converted into an output/auxiliary target representation?

## Closest Analogies / Novelty Layer

- Closest analogies: teacher-forced target-side feature leakage audit, auxiliary target design audit, representation-role classification.
- Relevant taxonomy bucket: representation integration validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is a correctness/role audit for the C3 representation, not a novelty claim.

## Minimal Change

Add audit artifacts only. Run targeted local checks against:

- C3 sidecar generation/provenance code,
- exact sidecar report,
- mapper dataset sidecar loader,
- mapper model C3 conditioning path,
- v2.1 rollout/incremental decode path.

No production code changes are planned in this card.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_full_pipeline_role_audit_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_full_pipeline_role_audit_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_full_pipeline_role_audit_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_result_report.md`

## Dataset Slice

Use existing full-cache C3 sidecar and mapper reports:

- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- exact sidecar report: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- P9 reports: `artifacts/runs/stage2_mapper_v2_1/c3_conditioning_longer_* / report.json`

## Baseline / Comparator

Baseline full-pipeline mapper inference uses generated target-token prefix, control/density context, grammar state, and no target-derived C3 side stream.

Comparator is the P7-P9 C3 conditioning probe, where exact C3 side-stream tokens are loaded from a precomputed beatmap-derived sidecar.

## Primary Metric

- Role classification: `legal_generation_input`, `offline_oracle_probe`, `auxiliary_target_candidate`, or `target_grammar_candidate`.

## Secondary Metric

- Inference compatibility: whether `generate_full_song_rollout_v2_1` can supply C3 tokens.
- Incremental decode compatibility: whether `MapperV21Model.incremental_decode_next_token` accepts C3-conditioned configs.
- Provenance evidence: whether sidecar tokens are derived from beatmap fallback payloads.
- Training evidence status from P9.

## Verify Command / Evaluation Procedure

```bash
rg -n "audit_c3_mapper_window_sidecar|_build_mapper_window_sidecar_payload|token_ids|incremental decode with C3|incremental_decode_next_token|generate_full_song_rollout_v2_1" src/pulsefield_model
uv run python - <<'PY'
from pathlib import Path
import json
for path in [
    Path("artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json"),
    Path("artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_summary.json"),
]:
    payload = json.loads(path.read_text())
    print(path, sorted(payload.keys())[:12])
PY
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Do not edit production code.
- Do not run more mapper training in this card.
- Do not call target-derived C3 tokens a legal inference input unless there is an inference-time source.
- Preserve the P9 result as stability evidence only.

## Qualitative Check

Read the exact sidecar and inference code paths and identify whether C3 is an input feature, auxiliary target, or output-token representation.

## Positive Signal

- The audit produces a clear next implementation family and avoids more training on a role-invalid path.
- If C3 is target-derived, the report explicitly prevents promotion as default input conditioning.

## Negative Signal

- Provenance remains ambiguous.
- Inference has a legal source for C3 tokens but the audit misses it.
- The report overstates P9 as quality evidence.

## Kill Criteria

Kill the current pooled C3 input-conditioning promotion path if C3 tokens are target-derived and generation cannot supply them without using the target chart.

## Expected Failure Modes

- The audit may decide against additional training despite stable loss.
- Some future C3 role may require a broader output-token redesign.
- Existing reports may not expose enough token provenance; source inspection may be required.

## Confounders

- Teacher-forced training can use target-derived auxiliary signals without failing loss checks.
- Pooled C3 conditioning may look stable even if it is not deployable.
- A future two-stage generator could legally predict C3 first, but that is not the current pipeline.

## Expected Runtime / Runtime Budget

Expected under 10 minutes. Stop if provenance cannot be resolved from current source and artifacts.

## Result Interpretation Plan

- Positive result would suggest: move C3 toward auxiliary target/output representation, or define a legal two-stage predictor.
- Negative result would suggest: if C3 has a valid exogenous source, implement inference-side C3 conditioning.
- Ambiguous result would require: add explicit provenance metadata to the sidecar generator.
- Human owner decides: whether to spend architecture time on C3 target tokenization or abandon mapper-side C3.
- Next-loop action if positive: create a C3 auxiliary-target or target-grammar Experiment Card.
- Next-loop action if negative: create an incremental-decode C3 input-conditioning card.
- Next-loop action if ambiguous: create a sidecar provenance metadata card.

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
- Remaining ambiguity: the next implementation family depends on the role classification.

## Next-Loop Action

- If positive: create a C3 auxiliary-target or target-grammar Experiment Card.
- If negative: create an incremental-decode C3 input-conditioning card.
- If ambiguous: add explicit sidecar provenance metadata.

## Novelty Notes

- Closest analogies: target-side leakage audit and auxiliary-target role classification.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this protects the C3 representation from being evaluated through a deployment-invalid input path.
