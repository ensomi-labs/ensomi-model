# Target Grammar v3 Post-Teacher-Forced Exposure Route Synthesis Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the teacher-forced v3 time-shift logit audit passed, while generated rollouts still show timing/continuation collapse and multiple decode/loss branches have failed rollout gates.
- Acceptance source, if any: `target_grammar_v3_teacher_forced_time_shift_logit_audit_result_report.md`.
- Source snapshot / evidence grade: high for fixed-slice teacher-forced timing logits; medium-high for existing 32-case generated-prefix rollout diagnostics; medium for cross-branch route synthesis because it is artifact-only.

## Hypothesis

If teacher-forced time-shift ranks are healthy but generated-prefix rollout, decode-policy, event-margin, and conditioned-objective gates still fail, then the next useful v3 experiment is not another scalar teacher-forced timing loss or global decode knob. It should route to a generated-prefix state trace or target-grammar repair audit; if that evidence is already saturated or narrow, the branch should pivot toward v2.1 grammar improvement.

## Root Objective

Move toward the requested full-pipeline v3 replacement or a justified v2.1 grammar pivot by reconciling the newest teacher-forced timing evidence with existing generated-prefix failure evidence.

## Goal Decomposition

- Subgoal 1: prove whether teacher-forced timing collapse is still a live hypothesis after the new audit.
- Subgoal 2: check whether decode-only and current training-objective mutations have already failed rollout gates.
- Subgoal 3: route the next bounded experiment to generated-prefix trace, target-grammar repair, or v2.1 grammar improvement without changing defaults.

## Candidate Variants

- Variant A: rerun time-shift-distance training with a new lambda. This ignores that teacher-forced ranks are healthy and the prior full32 objective worsened rollout.
- Variant B: run another decode-policy sweep. This ignores that deterministic decode-only policies were killed and stochastic sampling changed shape without stable quality.
- Variant C: synthesize teacher-forced and generated-prefix evidence, then route to a narrower generated-prefix state trace or pivot.
- Variant D: immediately pivot to v2.1 grammar. This may be right soon, but it skips one cheap exposure-route check after the new teacher-forced audit.

## Local Verification Matrix

- Variant A: reject unless teacher-forced recall@5 is weak or predicted shift concentration is extreme.
- Variant B: reject unless the prior decode sweep had a passing deterministic policy or a stable non-greedy policy.
- Variant C: pass if all required artifacts exist, checks are explicit, and the decision can be derived from current summaries without new training.
- Variant D: defer unless current summaries already prove generated-prefix trace opportunities are exhausted or v3 local probes are too narrow.

## Selected Variant

- Selected: Variant C, artifact-only post-teacher-forced exposure route synthesis.
- Rejected: A, B, and immediate D.
- Why this is the smallest useful test: it uses committed artifacts, runs in seconds, and decides whether the next card should inspect generated-prefix state exposure or stop spending v3-local runtime.

## Selection Pressure

- Primary pressure: distinguish teacher-forced timing-logit collapse from generated-prefix exposure/state collapse.
- Guard pressure: no training, no rollout rerun, no tokenizer change, no mapper default change.
- Runtime pressure: must run from existing JSON/Markdown artifacts in seconds.
- Kill pressure: if required artifacts are missing or disagree on schema/route, do not route to another implementation card.

## Research Question

After healthy teacher-forced time-shift ranks, do existing generated-prefix failures justify a prefix-state trace/target-grammar repair audit, or should the branch pivot away from v3 local probes toward v2.1 grammar improvement?

## Closest Analogies / Novelty Layer

- Closest analogies: exposure-bias triage, generated-prefix versus teacher-forced calibration audit, route synthesis after ablation gates.
- Relevant taxonomy bucket: artifact-grounded model diagnostic and experiment routing.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering route synthesis for v3 target-grammar research.

## Minimal Change

Add one route-synthesis evaluator that reads existing summaries and writes a JSON summary plus Markdown result report.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_post_teacher_forced_exposure_route_synthesis.py`
- `tests/evals/test_mapper_v3_post_teacher_forced_exposure_route_synthesis.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_time_shift_full32_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_c3_diminishing_returns_route_synthesis_summary.json`

## Dataset Slice

No new dataset pass. The synthesis uses:

- teacher-forced eval split: `56` windows / `3886` time-shift rows from the fixed v3 report;
- existing generated-prefix fixed 32-case summaries;
- existing short CE-sensitive conditioned-objective rollout gate.

## Baseline / Comparator

Baseline evidence:

- teacher-forced time-shift audit route and metrics;
- baseline generated-state continuation failure counts;
- prior decode-only and training-objective gate decisions.

Comparator:

- routes and guard outcomes from existing v3 local probes.

## Primary Metric

Route decision from explicit evidence checks:

- teacher-forced timing healthy;
- generated-prefix failure persists;
- decode-only killed;
- global event bonus killed;
- conditioned objective failed short rollout;
- current expected-shift distance objective failed full32 rollout;
- C3 mapper-side diminishing returns recorded.

## Secondary Metric

- teacher-forced recall@5 and median rank;
- generated continuation baseline starved/rigid/boundary counts;
- conditioned short-rollout starved delta and event-ratio guard;
- decode sweep best-policy pass flag;
- event-margin required-bias pressure.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_post_teacher_forced_exposure_route_synthesis
uv run --group dev pytest tests/evals/test_mapper_v3_post_teacher_forced_exposure_route_synthesis.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json >/tmp/post_teacher_forced_exposure_route.valid.json
```

## Guard Check

- all required source summaries load as JSON objects;
- the teacher-forced audit has faithful global context and nonzero time-shift rows;
- no command trains, reruns rollout, changes tokenizer behavior, or changes mapper defaults;
- summary JSON validates.

## Qualitative Check

The report must explicitly state what was proved, what was not proved, and why the next route is narrower than another v3 scalar loss.

## Positive Signal

Route to `TEST_GENERATED_PREFIX_STATE_TRACE` if teacher-forced ranks are healthy and generated-prefix failures persist while prior decode/loss branches are killed or inconclusive.

## Negative Signal

Route to `MUTATE_TIMING_OBJECTIVE` only if teacher-forced ranks are weak or the existing generated-prefix summaries contradict the exposure-collapse interpretation.

## Kill Criteria

- required source artifact missing;
- teacher-forced metrics are not faithful or nonzero;
- route depends on a smoke-only artifact;
- synthesis recommends training despite killed/failed rollout gates without a new mechanism.

## Expected Failure Modes

- Existing artifacts use slightly different route labels.
- Some summaries omit fields needed for secondary metrics.
- Route synthesis may be too coarse to justify implementation, in which case the next card should be trace instrumentation rather than training.

## Confounders

- The teacher-forced audit is fixed-slice, not full 4k.
- Generated-prefix summaries cover fixed 32-case or short selected subsets, not all charts.
- Existing rollout summaries may use different checkpoints and branch objectives.

## Expected Runtime / Runtime Budget

Expected runtime: under 5 seconds for synthesis and under 1 minute for focused tests. Stop on missing source summaries or failed JSON validation.

## Result Interpretation Plan

- Positive result would suggest: create or implement a generated-prefix state trace audit on high-leverage cases.
- Negative result would suggest: return to timing objective design only if teacher-forced collapse is actually visible.
- Ambiguous result would require: a narrower trace-instrumentation card before more training.
- Human owner decides: whether to keep spending v3-local probes or pivot to v2.1 grammar.
- Next-loop action if positive: implement generated-prefix state trace audit or target-grammar repair card.
- Next-loop action if negative: mutate timing objective with explicit generated-prefix guard.
- Next-loop action if ambiguous: write trace instrumentation only, no training.

## Result Log Template

- Experiment: target_grammar_v3_post_teacher_forced_exposure_route_synthesis
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
- Remaining ambiguity: the synthesis can route the next experiment, but cannot by itself prove full-pipeline v3 replacement readiness.

## Next-Loop Action

- If positive: implement generated-prefix state trace audit.
- If negative: mutate timing objective with generated-prefix rollout guards.
- If ambiguous: add trace instrumentation only.

## Novelty Notes

- Closest analogies: exposure-bias triage and route synthesis after ablation gates.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for target-grammar route selection.
