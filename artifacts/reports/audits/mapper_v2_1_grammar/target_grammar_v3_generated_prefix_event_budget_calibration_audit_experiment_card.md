# Target Grammar v3 Generated-Prefix Event-Budget Calibration Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: after C3 target-side work showed diminishing mapper-facing returns, continue the v3/v2.1 grammar route by diagnosing the generated-prefix event-count and timing calibration failure that survived teacher-forced timing, scalar losses, continuation-jump training, and local spacing escape.
- Acceptance source, if any: active thread goal plus committed v3/C3 reports.
- Source snapshot / evidence grade: strong local evidence from committed full-dataset representation audits, fixed-slice rollout gates, generated-prefix traces, and C3 target-complexity comparison; weak evidence for any new repair until this audit runs.

## Hypothesis

The remaining v3 rollout failure is not a representation irreversibility problem and not solved by local decode repair. It is a generated-prefix event-budget/completion calibration problem: the model repeatedly enters a rigid first-window timing attractor, underuses second-window event opportunities, and broad spacing escape overcorrects into sentinel overproduction. An artifact-only calibration audit can decide whether the next useful mutation should be a selective planner-side completion/budget signal, a grammar-level continuation target, or a pivot back to v2.1 grammar improvement.

## Root Objective

Move v3 toward full-pipeline replacement readiness by identifying the smallest next mechanism that can address free-running continuation without violating v3's core contract: reversible beatmap-event reconstruction, lower bits/tokens than v2.1, local online inference, and no C3-style target-history replay.

## Goal Decomposition

- Subgoal 1: Reconcile the latest negative v3 branches into one generated-prefix failure model: continuation-jump loss, conditioned/event-budget objectives, and spacing escape all fail through event-count calibration, legality, rigidity, or overproduction.
- Subgoal 2: Quantify whether the six traced generated-prefix failures contain rank-near event opportunities and second-window budget deficits that a selective completion/budget signal could plausibly target.
- Subgoal 3: Verify sentinel behavior so that any future repair is constrained against the known failure mode: fixing primary starvation by flooding easier/control charts.

## Candidate Variants

- Variant A: Artifact-only generated-prefix event-budget calibration audit. Read committed trace, rollout, and escape summaries; compute primary and sentinel budget errors, rank-near opportunity rates, second-window deficits, overproduction risk, and route synthesis.
- Variant B: Lower-weight continuation-jump retrain. Already locally trainable but the `lambda_continuation_jump=0.05` gate failed rollout legality, starvation, rigidity, event-ratio, and F1 guards.
- Variant C: Another trace-conditioned spacing/decode repair. Already killed because the six primary cases improved only by overproducing, and all four sentinel controls exceeded the event-ratio cap.
- Variant D: Immediate v2.1 grammar-improvement card. Plausible fallback, but premature until the current v3 failures are classified into "selective budget signal exists" versus "v3 local target grammar branch has diminishing returns".

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | Artifact-only synthesis over committed summaries | Produces a single route with explicit primary/sentinel evidence and no new training | Missing artifacts, mixed route, or no measurable budget/completion signal |
| B | Reuse continuation-jump training report | Failed: 3 dead ends, 19 starved cases, 24 rigid cases, median event ratio 0.631579 | Killed for this loop |
| C | Reuse spacing-escape smoke report | Failed: primary and sentinel event-ratio guards false | Killed for this loop |
| D | Create v2.1 grammar card now | Allowed only if A shows no selective v3 budget signal | Deferred until A classifies the v3 branch |

## Selected Variant

- Selected: Variant A, artifact-only generated-prefix event-budget calibration audit.
- Rejected: Variant B because teacher-forced continuation pressure did not transfer to generated-state rollout; Variant C because local decode repair floods; Variant D because the current v3 branch still has one cheap diagnostic before pivoting.
- Why this is the smallest useful test: it uses committed reports and at most one lightweight eval script. It does not train, change defaults, or modify tokenization.

## Selection Pressure

- Primary pressure: choose a path that can fail quickly before another training run.
- Guard pressure: preserve v3 legality, no future target input, no target-derived C3 conditioning, and no default decode change.
- Runtime pressure: artifact-only, expected under one minute on existing summaries.
- Kill pressure: if the audit cannot isolate a selective budget/completion signal, stop v3 scalar/decode repair expansion and pivot to v2.1 grammar improvement or a more structural grammar mutation.

## Research Question

Do generated-prefix v3 failures contain a selective, measurable event-budget/completion signal that can support another bounded v3 mutation, or have the recent branches shown enough diminishing returns to return to v2.1 grammar improvement?

## Closest Analogies / Novelty Layer

- Closest analogies: exposure-bias diagnostics for autoregressive models, event-count calibration, duration/budget control, and constrained sequence completion audits.
- Relevant taxonomy bucket: representation and training-objective audit, not a new model architecture.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: v3 event-token grammar is the representation candidate; this audit is engineering/diagnostic selection pressure around generated-prefix calibration.

## Minimal Change

- Add an eval-only audit that reads committed v3 summaries and produces a report plus summary JSON.
- No training, no rollout rerun, no tokenizer change, no grammar/vocab change, no model architecture change, and no default behavior change.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_generated_prefix_event_budget_calibration_audit.py`
- `tests/evals/test_mapper_v3_generated_prefix_event_budget_calibration_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_event_budget_calibration_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_event_budget_calibration_audit_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_event_budget_calibration_audit_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_continuation_jump_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_result_report.md`

## Dataset Slice

Artifact-only slice:

- six generated-prefix traced primary failures from `target_grammar_v3_generated_prefix_state_trace_audit_summary.json`;
- four sentinel controls from `target_grammar_v3_trace_conditioned_spacing_escape_summary.json`;
- 32-case fixed-slice aggregates from continuation-jump, event-budget, and baseline reports where available.

## Baseline / Comparator

- Generated-prefix trace baseline: 6/6 failures classified as `time_shift_repetition`, mean event ratio `0.533579`, mean second-window share `0.034722`, mean dominant-spacing ratio `0.942535`.
- Spacing-escape smoke: primary mean event ratio `1.222994`, primary second-window share `0.491164`, but primary event-ratio guard failed; sentinel median event ratio `1.634997`, sentinel guard failed.
- Continuation-jump training gate: rollout illegal with 3 dead ends, 19 starved cases, 24 rigid cases, median event ratio `0.631579`.
- Event-budget training gate: median event ratio `0.716418`, 18 starved cases, 6 overgeneration cases, 4 max-token cases.
- Matched v2.1 timing baseline: v2.1 also rigid, but v3 has worse sparse/dense second-window starvation.

## Primary Metric

Route decision with evidence:

- `TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL` if primary failures show rank-near event opportunities plus second-window event deficits while sentinels demonstrate broad-release overproduction risk.
- `MUTATE_TO_GRAMMAR_LEVEL_CONTINUATION_TARGET` if deficits exist but rank-near opportunities are weak or too coupled to rigid timing state.
- `PIVOT_TO_V2_1_GRAMMAR_IMPROVEMENT` if evidence shows no selective v3 budget signal beyond already failed scalar/decode repairs.

## Secondary Metric

- Primary rank-near opportunity count and share from generated-prefix traces.
- Primary second-window event-share deficit versus reference.
- Primary and sentinel event-ratio delta under spacing escape.
- Failure overlap across continuation-jump, event-budget, conditioned objective, and spacing escape.
- Explicit count of guard failures by family: starvation, rigidity, legality, overgeneration, max-token, event-ratio.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_generated_prefix_event_budget_calibration_audit.py -q
uv run python -m pulsefield_model.evals.mapper_v3_generated_prefix_event_budget_calibration_audit
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_event_budget_calibration_audit_summary.json >/dev/null
git diff --check
```

## Guard Check

- Audit must be artifact-only.
- No model checkpoint training.
- No rollout rerun.
- No grammar, tokenizer, vocab, replay, or inference default changes.
- No target-derived C3 input conditioning.
- Missing required artifacts must produce a failing audit result rather than silently weakening evidence.

## Qualitative Check

The report must state in plain terms why each killed family failed: scalar loss undertransfers, local decode floods, event-budget loss is blunt, and C3 is codec-side strong but target-sequence/state costly. The recommended next step must be exactly one bounded route, not a broad menu.

## Positive Signal

Positive signal for one more v3 mutation:

- at least 4/6 primary traced failures have rank-near event opportunities during second-window deficit;
- broad spacing escape improves primary continuation but overproduces sentinels, proving a selective gate is needed;
- the report selects a single budget/completion route with explicit guard metrics.

## Negative Signal

Negative signal:

- rank-near opportunities are absent or too sparse;
- the only successful continuation behavior comes from flooding;
- failure evidence is indistinguishable from already killed scalar/decode branches;
- the report cannot recommend a narrower v3 mechanism than "train more" or "add another loss".

## Kill Criteria

Kill further v3 scalar/decode repair expansion if:

- the audit cannot isolate selective budget/completion evidence;
- the next route would require target-derived inference inputs;
- the next route would require C3-style cross-window replay inside v3;
- the evidence points only to broad event-count pressure already killed by event-budget and spacing-escape guards.

## Expected Failure Modes

- Existing summaries may not expose enough per-step fields for rank-near opportunity counting.
- Primary traces may be too small to support a route beyond "needs full32 diagnostic".
- Sentinel overproduction may dominate and force a pivot even if primary continuation improves.
- The audit may show that v3 and v2.1 share a deeper generated-state timing attractor not addressable by v3 grammar alone.

## Confounders

- The primary trace set is only six cases and intentionally high-overlap/failure-focused.
- Sentinel controls are four cases, not a full replacement-quality validation.
- Undertrained fixed-slice checkpoints can exaggerate exposure bias.
- F1 can improve under dense regular grids without solving musical timing quality.

## Expected Runtime / Runtime Budget

Expected under one minute. Stop if required committed summary files are missing or schema assumptions fail.

## Result Interpretation Plan

- Positive result would suggest: create a bounded selective completion/budget signal card with hard sentinel overproduction guards.
- Negative result would suggest: pivot away from v3 scalar/decode repair and define a v2.1 grammar-improvement card or a structural v3 grammar mutation.
- Ambiguous result would require: one full32 generated-prefix trace aggregation before any training.
- Human owner decides: whether this evidence is enough to keep v3 as the mapper-facing branch or to prioritize v2.1 grammar.
- Next-loop action if positive: write a selected completion/budget Experiment Card.
- Next-loop action if negative: write a v2.1 grammar-improvement Experiment Card.
- Next-loop action if ambiguous: write a full32 trace aggregation card, not a training card.

## Result Log Template

- Experiment: target_grammar_v3_generated_prefix_event_budget_calibration_audit
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
- Remaining ambiguity: the audit may find the existing trace schema insufficient; if so, the next route is a full32 trace aggregation card rather than training.

## Next-Loop Action

- If positive: create a selective completion/budget signal card with primary and sentinel event-ratio guards.
- If negative: pivot to a v2.1 grammar-improvement card or structural v3 grammar mutation.
- If ambiguous: aggregate generated-prefix trace evidence on the 32-case fixed slice before any new loss or decode policy.

## Novelty Notes

- Closest analogies: exposure-bias diagnostics, sequence budget calibration, and duration/completion control in autoregressive generation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering selection pressure around the existing v3 target grammar, not a new representation claim.
