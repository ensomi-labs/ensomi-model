# Target Grammar v3 Post-CE Route Synthesis Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: CE weighting, trace-oracle diagnostics, density-aware trace, and broad/tap-only anti-rigid decode gates now form a complete local-v3 mutation branch, but their decisions are spread across reports.
- Acceptance source, if any: active thread goal plus the committed CE/tap-only anti-rigid kill.
- Source snapshot / evidence grade: strong full-dataset representation evidence; strong pipeline-mechanics evidence; strong negative evidence against simple local decode policies; medium evidence that remaining v3 work must be training/target-distribution repair.

## Hypothesis

If all simple local decode branches fail at least one full32 gate and scalar event-distribution losses have already failed or shown weak pressure, then the next v3 work should stop decode-policy tweaks and move to a bounded conditioned event-distribution objective or target-grammar repair. If any branch still has an untested positive gate, the next card should target that narrower branch instead.

## Root Objective

Make the current v3 branch state auditable after CE and anti-rigid experiments, then select the next smallest experiment aligned with full v3 replacement readiness.

## Goal Decomposition

- Subgoal 1: Load authoritative summaries for representation, pipeline, CE weighting, trace, density trace, event-budget, density loss, and anti-rigid decode gates.
- Subgoal 2: Classify each implementation family as passed, killed, mutated, or next-test.
- Subgoal 3: Check whether any simple decode-policy family remains worth testing.
- Subgoal 4: Select one next bounded card family: conditioned training objective, target-grammar repair, or C3/v2.1 fallback.

## Candidate Variants

- Variant A: Continue CE anti-rigid decode tuning.
- Variant B: Continue low-bias/selective event decode tuning.
- Variant C: Add a conditioned event-distribution training objective plumbing card.
- Variant D: Redesign v3 grammar immediately.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Broad, soft, and tap-only anti-rigid full32 decisions | One route remains `TEST_NEXT` | All route `KILL` or fail legality/event ratio |
| B | CE trace and density trace decisions | Broad sparse-context opportunity coverage | Only `1/5` CE-starved cases covered |
| C | Existing loss/gate evidence | Global objectives failed but localized/conditioned objective not tested | If scalar losses already solve the issue |
| D | Representation-only design | Training/target repair impossible to bound | Smaller objective plumbing can answer first |

## Selected Variant

- Selected: Variant C, conditioned event-distribution objective plumbing as the next card family if synthesis confirms A and B are killed.
- Rejected: A and B if their latest summaries remain negative.
- Deferred: D until a smaller conditioned training-objective card fails.
- Why this is the smallest useful test: it preserves the audited v3 grammar and training pipeline while testing a more targeted objective surface before redesigning representation.

## Selection Pressure

- Primary pressure: do not spend another full32 runtime on a killed decode family.
- Guard pressure: v3 remains local, reversible, teacher-forcing friendly, and online inference does not depend on future tokens.
- Runtime pressure: synthesis is artifact-only and should finish under one minute.
- Kill pressure: if all decode families are killed, do not create another decode selector card.

## Research Question

After CE weighting and anti-rigid/timing diagnostics, what is the next falsifiable v3 repair branch: more decode tuning, conditioned training objective, or target-grammar redesign?

## Closest Analogies / Novelty Layer

- Closest analogies: branch triage after ablation studies, experiment-family meta-analysis, failure-mode routing.
- Relevant taxonomy bucket: result interpretation and next-loop selection.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is research workflow evidence, not representation novelty.

## Minimal Change

Add an artifact-only evaluator that reads the existing JSON summaries, produces a branch table, checks explicit route predicates, and writes JSON/Markdown synthesis artifacts.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_post_ce_route_synthesis.py`
- `tests/evals/test_mapper_v3_post_ce_route_synthesis.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_result_report.md`

## Read-Only Context Files

- `target_grammar_v3_full_dataset_audit_summary.json`
- `target_grammar_v3_real_config_cache_backed_comparison_summary.json`
- `target_grammar_v3_real_audio_session_rollout_summary.json`
- `target_grammar_v3_density_loss_effectiveness_diagnostic_summary.json`
- `target_grammar_v3_event_budget_training_gate_summary.json`
- `target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `target_grammar_v3_ce_residual_cluster_diagnostic_summary.json`
- `target_grammar_v3_ce_event_margin_viability_summary.json`
- `target_grammar_v3_ce_selective_trace_oracle_summary.json`
- `target_grammar_v3_ce_density_trace_diagnostic_summary.json`
- `target_grammar_v3_ce_antirigid_full32_summary.json`
- `target_grammar_v3_ce_antirigid_soft_full32_summary.json`
- `target_grammar_v3_ce_tap_only_antirigid_full32_summary.json`
- C3/v3 target complexity summary.

## Dataset Slice

Artifact summaries from the committed v3/C3 audit branch. No new training or rollout dataset is used.

## Baseline / Comparator

Comparator is the pre-CE v3 500-step wide audit and the CE-weight full32 gate:

- CE improves starvation versus pre-CE but worsens rigidity.
- Anti-rigid decode improves rigidity but fails event-count/legality gates.
- Selector traces explain only one CE-starved case.

## Primary Metric

Route decision:

- `TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE` if decode families are killed and representation/pipeline mechanics remain valid.
- `TEST_DECODE_BRANCH` only if an existing decode family has a positive full32 route.
- `MUTATE_TO_TARGET_GRAMMAR_REPAIR` if training objective evidence is already exhausted.

## Secondary Metric

- Count of killed decode families.
- Count of positive representation/pipeline gates.
- Explicit next-card recommendation.
- Evidence table with route, reason, and key metrics for every source family.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_post_ce_route_synthesis \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_post_ce_route_synthesis.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_ce_route_synthesis_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must clearly separate what is proved, what is killed, what remains unproven, and the next bounded card.

## Positive Signal

Synthesis selects a single next bounded card family and gives explicit evidence for killing simple decode branches.

## Negative Signal

Synthesis finds a still-positive decode branch or lacks enough authoritative summaries to route.

## Kill Criteria

- Missing required summaries.
- Contradictory route extraction that prevents classification.
- Any route claim not backed by an inspected artifact.

## Expected Failure Modes

- Some summaries store `decision` as a string and others as a mapping.
- Artifact naming drift can hide required reports.
- Synthesis can overclaim if it treats narrow smoke tests as full replacement evidence.

## Confounders

- Artifact-only synthesis cannot prove the next objective will work.
- The next recommendation depends on current reports and should change if a later training run exists.
- Full v3 replacement still requires longer training and real-audio quality evidence.

## Expected Runtime / Runtime Budget

Expected runtime: under one minute.

## Result Interpretation Plan

- Positive result would suggest: create a conditioned event-distribution objective plumbing card.
- Negative result would suggest: run the still-open branch or repair missing artifacts.
- Ambiguous result would require: one more artifact-level failure inspection, not model training.
- Human owner decides: whether to pursue conditioned objective or grammar redesign after this route.
- Next-loop action if positive: `TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE_PLUMBING`.
- Next-loop action if negative: `TEST_OPEN_BRANCH`.
- Next-loop action if ambiguous: `MUTATE_SYNTHESIS_INPUTS`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Source summaries:
- Runtime:
- Primary route:
- Decode families killed:
- Representation gates passed:
- Pipeline gates passed:
- Failed or missing checks:
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
- Remaining ambiguity: whether the conditioned event-distribution objective should be loss-side only or include target-side grammar repair.

## Next-Loop Action

- If positive: create a conditioned event-distribution objective plumbing card.
- If negative: test the still-open branch.
- If ambiguous: repair synthesis inputs.

## Novelty Notes

- Closest analogies: ablation family triage and branch selection after failed gates.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: route synthesis only.
