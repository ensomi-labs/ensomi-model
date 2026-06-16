# Target Grammar v3 Generated-State Continuation Diagnostic Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: C3 mapper-side paths now show diminishing returns, while v3 rollout gates repeatedly surface first-window starvation, rigid timing, and boundary clamping after same-ms duplicate loops were fixed.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_same_ms_event_guard_result_report.md`, `target_grammar_v3_event_budget_training_gate_result_report.md`, `target_grammar_v3_continuation_jump_training_gate_result_report.md`, and `c3_ordered_raw_field_grammar_probe_result_report.md`.
- Source snapshot / evidence grade: strong evidence that v3 reconstructs and is lower-token; strong evidence that current free-running v3 quality fails; medium evidence that scalar auxiliary losses are the wrong lever; weak evidence for the next grammar mutation.

## Hypothesis

The remaining v3 failure is a generated-state timing attractor rather than a basic grammar legality problem. If existing rollouts show that event tokens are often valid and rank-near while generated times collapse to boundary-adjacent first-window patterns, the next mutation should target completion/timing semantics or decode calibration, not another scalar teacher-forced loss. If event tokens are not valid or dead ends cluster under a specific grammar state, then the next mutation should be grammar-state repair.

## Root Objective

After C3 diminishing returns, identify the smallest v3/v2.1 grammar-side mutation that could improve full-pipeline readiness without broad replacement or more blind training.

## Goal Decomposition

- Subgoal 1: Compare the same 32-case fixed-slice rollout universe across baseline 500-step v3, same-ms guard, event-budget loss, and continuation-jump loss.
- Subgoal 2: Classify starved/rigid/boundary cases by generated-state signatures using existing run summaries.
- Subgoal 3: Separate grammar unavailability from model ranking/calibration signals using event-valid, event-top-k, and event-top-1 counts.
- Subgoal 4: Produce a next-loop route: decode/target timing mutation, grammar-state repair, or pivot to v2.1 grammar improvement.

## Candidate Variants

- Variant A: Run another 500-step scalar-loss training gate. Rejected because event-budget and continuation-jump scalar losses failed rollout gates.
- Variant B: Directly change v3 grammar completion semantics. Rejected until we know whether the failure is grammar unavailability or model ranking.
- Variant C: Generated-state continuation diagnostic over existing 32-case summaries. Selected because it is fast, non-invasive, and directly informs the next mutation.
- Variant D: Pivot immediately to v2.1 grammar improvement. Deferred until the diagnostic confirms v3 failures are not addressable by a small local grammar/decode mutation.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Reuse prior reports | Scalar loss improves all gates | Prior scalar losses already failed |
| B | Static code review of grammar | Dead-end states are impossible or isolated | Needs evidence from generated states |
| C | Aggregate existing summaries | Failure classes and next lever are identifiable | Summaries lack enough state/logit evidence |
| D | Synthesis report only | v3 local route is exhausted | v3 still has a small testable mutation |

## Selected Variant

- Selected: Variant C, generated-state continuation diagnostic.
- Rejected: A because it spends training runtime without a new mechanism.
- Deferred: B and D until diagnostic evidence selects the lever.
- Why this is the smallest useful test: it reuses already-generated full32 summaries and can fail before any training or production-code grammar change.

## Selection Pressure

- Primary pressure: identify whether starved/rigid failures are mostly model-ranking/decode calibration or grammar-state unavailability.
- Guard pressure: no training, inference behavior, tokenization, or defaults change.
- Runtime pressure: under one minute on CPU.
- Kill pressure: if the summaries cannot distinguish the failure class, create a richer rollout instrumentation card instead of guessing.

## Research Question

After C3 target-side paths failed to become mapper-ready, what is the next smallest v3/v2.1 grammar-side experiment supported by current rollout evidence?

## Closest Analogies / Novelty Layer

- Closest analogies: generated-state error analysis, exposure-bias diagnostics, constrained decoding audits, and grammar/decode ablation selection.
- Relevant taxonomy bucket: local verification and failure-mode interpretation before a bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering evidence for choosing between v3 grammar repair and v2.1 grammar improvement.

## Minimal Change

Add an artifact-only evaluator that reads existing v3 rollout summary JSON files and writes a diagnostic report. It should:

- load aggregate and per-run rows from 500-step baseline, same-ms guard, event-budget loss, and continuation-jump loss summaries;
- classify each run as starved, rigid, boundary-clamped, late-spill, dead-end, overgenerated, or max-token;
- compute event top-k/top-1 gap and event-valid ratios where available;
- compare failure overlap by `case_id` across variants;
- recommend the next bounded route.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_continuation_cluster_diagnostic.py`
- `tests/evals/test_mapper_v3_continuation_cluster_diagnostic.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_same_ms_event_guard_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json`
- `artifacts/tmp/mapper_v3_continuation_jump_guard/continuation_jump_guard_rollout_summary.json`
- `src/pulsefield_model/inference/mapper_v3_rollout.py`
- `src/pulsefield_model/models/mapper/v3/grammar.py`

## Dataset Slice

Use the existing fixed 32-song / 256-window v3 rollout case universe from recent reports.

Variants:

- `baseline_500step`: `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `same_ms_guard`: `target_grammar_v3_same_ms_event_guard_summary.json`
- `event_budget_lambda005`: `target_grammar_v3_event_budget_training_gate_summary.json`
- `continuation_jump_lambda005`: `artifacts/tmp/mapper_v3_continuation_jump_guard/continuation_jump_guard_rollout_summary.json`

## Baseline / Comparator

Primary comparator: `baseline_500step`, because it has the best currently observed v3 rollout aggregate among the compared 500-step paths.

Secondary comparators: same-ms guard, event-budget loss, and continuation-jump loss.

## Primary Metric

Failure-class counts by variant:

- starved case count;
- rigid case count;
- boundary-clamped case count;
- late-spill case count;
- dead-end count;
- max-token count.

## Secondary Metric

- Median and mean event top-k/top-1 gap.
- Event-valid step ratio.
- Failure overlap count by case id across variants.
- Boundary clamp share among starved cases.
- Cases where event top-k is high but event top-1 is low.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_continuation_cluster_diagnostic \
  --summary baseline_500step=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json \
  --summary same_ms_guard=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_same_ms_event_guard_summary.json \
  --summary event_budget_lambda005=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_training_gate_summary.json \
  --summary continuation_jump_lambda005=artifacts/tmp/mapper_v3_continuation_jump_guard/continuation_jump_guard_rollout_summary.json \
  --baseline baseline_500step \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_continuation_cluster_diagnostic.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The result report must state whether the next mutation should be:

- decode/target timing calibration;
- grammar-state repair;
- richer instrumentation;
- or pivot to v2.1 grammar improvement.

## Positive Signal

The diagnostic identifies one dominant failure lever with enough evidence to write a bounded next Experiment Card.

## Negative Signal

The available summaries cannot separate grammar unavailability from model ranking or decode-policy collapse.

## Kill Criteria

- Required summaries are missing or incompatible.
- Case ids cannot be matched across variants.
- The diagnostic only restates existing aggregate counts without adding a decision-relevant failure class.
- The result would require guessing a grammar change without evidence.

## Expected Failure Modes

- Existing summaries lack full generated states and only expose compressed per-run diagnostics.
- Boundary-clamp heuristics may over-count valid late first-window events.
- Event-top-k counts may be too coarse because they aggregate across all steps.
- The continuation-jump summary currently lives under ignored `artifacts/tmp`.

## Confounders

- The compared variants are not identical models; scalar-loss variants were retrained.
- Some summaries report reference second-window share and others do not.
- Event top-k availability does not prove an event sequence is musically correct.
- Generated timepoint previews are enough for classifying boundary patterns but not full replay semantics.

## Expected Runtime / Runtime Budget

Expected runtime: under 10 seconds. Stop if any required summary cannot be parsed.

## Result Interpretation Plan

- Positive result would suggest: create the next bounded grammar/decode mutation card named by the diagnostic.
- Negative result would suggest: instrument full v3 rollout state/logit traces before changing grammar.
- Ambiguous result would require: compare a small rerun with full per-step state traces on the worst shared cases.
- Human owner decides: whether the evidence is enough to pivot from v3 to v2.1 grammar improvement.
- Next-loop action if positive: write and run the selected bounded mutation.
- Next-loop action if negative: instrumentation card, not training.
- Next-loop action if ambiguous: worst-case trace audit.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Input summaries:
- Baseline:
- Runtime:
- Failure-class table:
- Top overlap cases:
- Event-rank/top-k evidence:
- Boundary-clamp evidence:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected next lever:
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
- Remaining ambiguity: the diagnostic can choose the next lever, but it cannot itself prove a grammar fix.

## Next-Loop Action

- If decode/timing ranked attractor dominates: create a bounded decode/target timing-calibration card.
- If grammar-state unavailability dominates: create a minimal grammar-state repair card.
- If evidence is too coarse: create a full state/logit instrumentation card.
- If v3 has no small local mutation: pivot to v2.1 grammar improvement.

## Novelty Notes

- Closest analogies: constrained-generation failure clustering, exposure-bias diagnostics, and rollout trace triage.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for v3 replacement readiness.
