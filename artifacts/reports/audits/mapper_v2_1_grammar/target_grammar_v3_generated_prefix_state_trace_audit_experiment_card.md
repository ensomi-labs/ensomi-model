# Target Grammar v3 Generated-Prefix State Trace Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: post-teacher-forced exposure synthesis routed to `TEST_GENERATED_PREFIX_STATE_TRACE`.
- Acceptance source, if any: `target_grammar_v3_post_teacher_forced_exposure_route_synthesis_result_report.md`.
- Source snapshot / evidence grade: high that teacher-forced time-shift ranks are healthy; medium-high that generated-prefix rollouts still fail; medium that a trace can isolate the mechanism without another training run.

## Hypothesis

The v3 checkpoint's remaining rollout failure starts under generated prefixes: either repeated time-shift choices create state drift, event tokens become rank-near but underselected, LN/carry state diverges from the target path, or a boundary/terminal state causes late spill. A per-step trace on high-overlap failed cases should identify the first actionable failure class before any new v3 loss, C3 integration, or v2.1 pivot.

## Root Objective

Decide whether v3 still has a small local repair path after healthy teacher-forced timing logits and failed decode/objective shortcuts, or whether the branch should pivot to target-grammar/v2.1 grammar work.

## Goal Decomposition

- Subgoal 1: collect generated-prefix state/logit traces on shared starved/rigid/boundary failure cases.
- Subgoal 2: align trace steps with reference target timepoints and window boundaries.
- Subgoal 3: classify the earliest failure mechanism per case into timing collapse, event underselection, carry/state mismatch, boundary drift, or no actionable trace.

## Candidate Variants

- Variant A: full32 rerun with full trace logging. More complete, but too expensive and noisy for a first mechanism audit.
- Variant B: artifact-only synthesis from existing summaries. Already done; it routes the next card but cannot classify first failure step.
- Variant C: six-case generated-prefix state trace on highest-overlap cases using the existing 500-step v3 checkpoint. This is selected.
- Variant D: immediate v2.1 grammar pivot. Plausible soon, but it skips a cheap mechanism check after the new teacher-forced result.

## Local Verification Matrix

- Variant A: reject unless six-case trace cannot reproduce the failure or lacks enough steps.
- Variant B: reject as already complete and insufficient for mechanism classification.
- Variant C: pass if it can run without training, emit per-step state/logit rows, and classify at least one failure mechanism.
- Variant D: defer unless Variant C is uninstrumentable or returns no actionable generated-prefix mechanism.

## Selected Variant

- Selected: Variant C, six-case generated-prefix state trace.
- Rejected: A, B, and immediate D.
- Why this is the smallest useful test: it targets the exact cases where multiple v3 branches share starvation/boundary/rigid failures, uses existing checkpoints, and avoids spending another training run before mechanism evidence.

## Selection Pressure

- Primary pressure: identify the first generated-prefix mechanism that diverges from healthy teacher-forced behavior.
- Guard pressure: no training, no tokenizer change, no default decode change, no production conditioning.
- Runtime pressure: run at most six real-audio fixed-slice rollouts with bounded per-step logs.
- Kill pressure: if trace collection cannot align generated state to reference cases or cannot classify failures, do not train a new objective.

## Research Question

When v3 rollouts fail despite healthy teacher-forced time-shift ranks, what is the first generated-prefix failure mechanism on high-overlap starved/rigid/boundary cases?

## Closest Analogies / Novelty Layer

- Closest analogies: exposure-bias trace audit, scheduled-sampling error attribution, per-step logit/state instrumentation.
- Relevant taxonomy bucket: model diagnostics and failure localization.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for v3 target-grammar replacement readiness.

## Minimal Change

Add an artifact-only evaluator that runs or reads bounded generated-prefix rollouts with expanded per-step trace rows and writes a summary/report. It should reuse the existing v3 runtime rollout path and logit observer where possible.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_generated_prefix_state_trace_audit.py`
- `tests/evals/test_mapper_v3_generated_prefix_state_trace_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_teacher_forced_exposure_route_synthesis_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json`
- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`
- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json`
- `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/inference/mapper_v3_rollout.py`

## Dataset Slice

Six high-overlap fixed-slice cases from the generated-state continuation diagnostic:

- `14_oomori_seiko_justadice_tv_size_remu_hard`
- `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- `19_nekodex_circles_famoss_hard`
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- `18_billiummoto_four_veiled_stars_aries_famoss_hard`

Use the same chart-end and real-audio setup as the existing fixed-slice rollout summaries where available.

## Baseline / Comparator

Baseline:

- current greedy v3 generated-prefix rollout trace on each selected case.

Comparators:

- teacher-forced time-shift audit aggregate: recall@5 `0.923314`, median rank `1`;
- reference event/timepoint sequence for each selected chart;
- existing continuation failure class for each selected case.

## Primary Metric

First failure class per case:

- `time_shift_repetition`: repeated fixed spacing starts before target boundary and dominates later events;
- `event_underselection`: event token is valid/rank-near but time shift or EOS wins repeatedly;
- `state_or_carry_mismatch`: generated replay state diverges from reference carry/open-mask expectations;
- `boundary_drift`: generated state parks near the 8s boundary or spills to terminal time;
- `unclassified`: trace does not provide an actionable first mechanism.

## Secondary Metric

- first failure step index and current ms;
- best event rank/margin at failure;
- time-shift argmax and top-k distribution before failure;
- repeated-spacing run length;
- event-valid/top-k/top-1 ratios before and after failure;
- boundary event share and second-window event share.

## Verify Command / Evaluation Procedure

Planned command:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_generated_prefix_state_trace_audit
uv run --group dev pytest tests/evals/test_mapper_v3_generated_prefix_state_trace_audit.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/inference/test_mapper_v3_rollout.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json >/tmp/generated_prefix_state_trace.valid.json
```

## Guard Check

- no training;
- no tokenizer, grammar, or default decode change;
- all selected cases have trace rows or explicit missing-case reason;
- rollouts do not use target-derived C3 sidecar conditioning;
- summary JSON validates.

## Qualitative Check

Inspect the first 20 trace rows around each first failure and verify that the assigned failure class is understandable from current ms, top tokens, valid token kinds, and emitted token.

## Positive Signal

At least four of six selected cases receive a concrete first failure class, and at least one class points to a bounded next repair such as trace-conditioned event ranking, state/carry repair, boundary continuation repair, or target-grammar mutation.

## Negative Signal

Most cases are unclassified, require huge event biases, or show no low-bias/rank-near opportunities. That would argue against another v3 local training objective and toward target-grammar/v2.1 grammar work.

## Kill Criteria

- fewer than four selected cases can be traced;
- trace rows cannot be aligned to generated current-ms state;
- failure classification depends on target leakage;
- the result recommends a new training run without a concrete generated-prefix mechanism.

## Expected Failure Modes

- Existing runtime rollout only stores aggregate logit diagnostics, requiring a small observer extension.
- Some selected case paths may need lookup from existing fixed-slice summaries.
- Per-step logs may be large; cap rows around first failure and aggregate the rest.

## Confounders

- Six cases are not full32/full4k.
- Real-audio preparation may differ from cached teacher-forced eval records.
- Greedy traces do not prove stochastic or beam behavior.
- Reference alignment may be approximate around same-ms chords and LN events.

## Expected Runtime / Runtime Budget

Expected runtime: under 20 minutes for six bounded real-audio rollouts on CPU/MPS. Stop after two repeated trace failures or if one case exceeds the existing token cap.

## Result Interpretation Plan

- Positive result would suggest: implement the smallest repair matching the dominant first failure class.
- Negative result would suggest: stop v3 local probes and draft a v2.1 grammar improvement or target-grammar redesign card.
- Ambiguous result would require: narrower instrumentation on one case, not training.
- Human owner decides: whether to continue v3 local repair or pivot to v2.1 grammar.
- Next-loop action if positive: one repair card tied to the dominant class.
- Next-loop action if negative: v2.1 grammar improvement card.
- Next-loop action if ambiguous: one-case trace instrumentation card.

## Result Log Template

- Experiment: target_grammar_v3_generated_prefix_state_trace_audit
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
- Remaining ambiguity: the trace can classify failures, but cannot by itself prove a repair or full-pipeline replacement.

## Next-Loop Action

- If positive: create the matching repair Experiment Card.
- If negative: create a v2.1 grammar improvement or target-grammar redesign Experiment Card.
- If ambiguous: instrument one case more deeply before training.

## Novelty Notes

- Closest analogies: exposure-bias trace audit, per-step logit/state attribution.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for deciding whether v3 remains worth local repair.
