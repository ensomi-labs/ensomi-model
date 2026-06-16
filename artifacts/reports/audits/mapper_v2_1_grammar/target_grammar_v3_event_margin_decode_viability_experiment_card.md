# Target Grammar v3 Event-Margin Decode Viability Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the generated-state continuation diagnostic selected decode timing calibration, but prior decode-policy sweeps already killed the existing flat/delta time-shift knobs. Before adding another runtime selector, check whether the recorded event logit margins are small enough for a bounded event-bonus calibration to be plausible.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_generated_state_continuation_diagnostic_result_report.md`, `target_grammar_v3_decode_policy_continuation_sweep_result_report.md`, and `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`.
- Source snapshot / evidence grade: strong evidence that v3 is legal/reversible and shorter as a teacher-forcing target; strong evidence that full32 free-running rollout still fails via starvation/rigidity; medium evidence that simple scalar losses and current decode knobs are exhausted; weak evidence about whether an event-bonus selector is viable.

## Hypothesis

The remaining v3 continuation failure is not solved by a simple global event-token bonus if starved or rigid cases require large median logit-margin shifts to make event tokens win. If many failed cases have best valid event tokens with median margin worse than `-4.0`, then a global decode bonus would likely overproduce on already-passable cases before fixing the hard failures. If failed cases have low required bias, then a small bounded runtime selector is worth testing next.

## Root Objective

Decide whether the next v3 step should be a small event-bonus decode calibration, a richer selective trace/oracle experiment, or a pivot toward v2.1 grammar improvement after C3 and scalar v3 paths show diminishing returns.

## Goal Decomposition

- Subgoal 1: Measure required event-token bias from existing per-case median best-event margins on the 32-case v3 500-step fixed-slice audit.
- Subgoal 2: Compare required bias for starved, rigid, pass-like, undergenerated, and overgenerated cases.
- Subgoal 3: Classify whether a simple global event bonus has a plausible safe operating region.
- Subgoal 4: Recommend the next bounded route without retraining or changing runtime defaults.

## Candidate Variants

- Variant A: Add a global event bonus to v3 runtime and rerun full32. Rejected for now because previous decode sweeps already failed and a global bonus can create max-token/duplicate overproduction.
- Variant B: Run an artifact-only event-margin viability diagnostic. Selected because it can fail quickly using existing full32 logit diagnostics and directly tests whether a simple bonus is plausible.
- Variant C: Add a selective per-window budget selector using density/control features. Deferred until Variant B shows that a bonus magnitude is not obviously unsafe.
- Variant D: Pivot immediately to v2.1 grammar improvement. Deferred until the event-margin evidence kills the simple v3 decode lever.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Full32 rerun with event bonus | Starvation drops without overgeneration/max-token regression | Large bias needed or prior stochastic behavior predicts overproduction |
| B | Existing summary audit | Failed cases mostly need bias `<= 4.0` while pass-like cases need less or no bias | Failed cases mostly need bias `> 4.0` and pass-like/overgenerated cases overlap |
| C | Worst-case trace replay | Selective gate chooses events only in starved windows | Needs more state than current summaries expose |
| D | Synthesis report only | v3 small decode route is exhausted | v3 still has a plausible local lever |

## Selected Variant

- Selected: Variant B, event-margin decode viability diagnostic.
- Rejected: Variant A because it would change runtime behavior before checking whether event margins imply a safe event-bonus range.
- Deferred: Variant C and D until the diagnostic selects a route.
- Why this is the smallest useful test: it uses already-generated full32 summaries, needs no training, and can distinguish "event top-k is enough" from "event top-k is too low-margin for simple decode calibration."

## Selection Pressure

- Primary pressure: decide if simple global event-bonus calibration should be killed or tested.
- Guard pressure: no tokenizer, grammar, model, runtime-default, or training behavior changes.
- Runtime pressure: under one minute on CPU.
- Kill pressure: if failed cases need large median event bias or pass-like cases already overgenerate, do not run another global decode sweep.

## Research Question

Do the v3 500-step full32 logit margins support a simple event-token decode bonus as the next timing-calibration experiment?

## Closest Analogies / Novelty Layer

- Closest analogies: logit-margin calibration, constrained-decoding ablation triage, exposure-bias diagnostics, and offline decision-boundary audits.
- Relevant taxonomy bucket: local verification and failure-mode interpretation before a bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering evidence about a decode policy, not representation novelty.

## Minimal Change

Add an artifact-only evaluator that reads one or more v3 rollout summary JSON files and writes a viability report. It should:

- compute required event bias as `max(0, -best_event_margin_median)`;
- bucket cases by required bias thresholds `2`, `4`, `6`, and `8`;
- classify cases as starved, rigid, undergenerated, overgenerated, or pass-like from existing rollout metrics;
- compare failed-case required bias against pass-like/overgenerated cases;
- recommend one route: `TEST_SIMPLE_EVENT_BONUS`, `TEST_SELECTIVE_TRACE_ORACLE`, `KILL_GLOBAL_EVENT_BONUS`, or `MUTATE_TO_V2_1_GRAMMAR`.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_event_margin_decode_viability.py`
- `tests/evals/test_mapper_v3_event_margin_decode_viability.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_state_continuation_diagnostic_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_distribution_calibration_summary.json`

## Dataset Slice

Use the existing 32-case v3 500-step fixed-slice wide-audit summary. Optionally compare same-ms guard, event-budget, and continuation-jump summaries if they expose the same margin fields.

Primary input:

- `baseline_500step`: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`

## Baseline / Comparator

Baseline comparator is the current 500-step v3 rollout: all legal, `11` second-window-starved cases, `7` rigid cases by wide-audit definition, median event ratio `1.0`, and mean F1 `0.639566`.

## Primary Metric

Failed-case required-bias distribution, especially the count and share of starved or rigid cases with required event bias `> 4.0`.

## Secondary Metric

- Required bias distribution for pass-like and overgenerated cases.
- Event top-k minus top-1 gap.
- Median best-event rank by failure class.
- Number of starved cases with required bias `<= 2.0`, `<= 4.0`, `<= 6.0`, and `> 6.0`.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_event_margin_decode_viability \
  --summary baseline_500step=artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_result_report.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_event_margin_decode_viability.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must explain whether "event top-k exists" is strong enough for decode calibration, or whether large negative event margins mean global event bias is unsafe.

## Positive Signal

Most starved/rigid cases need required event bias `<= 4.0`, pass-like/overgenerated cases do not need comparable bias, and the route is `TEST_SIMPLE_EVENT_BONUS`.

## Negative Signal

Most starved/rigid cases need required event bias `> 4.0` or `> 6.0`, while pass-like or already-overgenerated cases overlap enough that a global bonus would likely increase artifacts.

## Kill Criteria

- Required margin fields are missing from the baseline summary.
- Case ids cannot be parsed.
- The diagnostic cannot separate starved/rigid failures from pass-like cases.
- The result only restates top-k availability without quantifying margin magnitude.

## Expected Failure Modes

- Median margin hides per-step windows where a selective event would help.
- Existing summaries do not expose full per-step logit distributions.
- Some summaries have `null` F1 or missing reference second-window share.
- Required-bias thresholds are heuristics, not a direct rerollout result.

## Confounders

- A selective event gate might succeed even when global event bias is unsafe.
- Stochastic sampling improved continuation in a three-case probe but overgenerated; this diagnostic cannot measure stochastic stability.
- The fixed 32-case slice is intentionally adversarial and may overstate full-dataset tail risk.
- Median logit margin does not prove a specific event token is musically correct.

## Expected Runtime / Runtime Budget

Expected runtime: under 10 seconds. Stop if required summary fields are missing.

## Result Interpretation Plan

- Positive result would suggest: run a bounded simple event-bonus rerollout on high-leverage cases.
- Negative result would suggest: kill global event-bonus decode calibration; either add selective per-step trace instrumentation or pivot to v2.1 grammar improvement.
- Ambiguous result would require: trace-level oracle replay on the shared worst cases.
- Human owner decides: whether a selective trace/oracle v3 path is worth one more card before v2.1 grammar work.
- Next-loop action if positive: create and run `target_grammar_v3_simple_event_bonus_rollout`.
- Next-loop action if negative: create either a selective trace/oracle card or a v2.1 grammar improvement card.
- Next-loop action if ambiguous: add full per-step generated-state/logit traces for worst shared cases.

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
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: current summaries cannot prove a selective per-step event gate would fail; they can only kill or support a simple global event-bonus route.

## Next-Loop Action

- If positive: run a bounded event-bonus rollout smoke on high-leverage cases.
- If negative: stop global decode-bonus work and either instrument selective traces or pivot to v2.1 grammar improvement.
- If ambiguous: add per-step trace collection on the shared worst cases.

## Novelty Notes

- Closest analogies: logit-margin calibration and constrained-decoding failure triage.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is an engineering viability check for runtime calibration.
