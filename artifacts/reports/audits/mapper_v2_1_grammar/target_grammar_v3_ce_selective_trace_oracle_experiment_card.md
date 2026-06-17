# Target Grammar v3 CE Selective Trace Oracle Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the CE margin viability report killed a global event-token bonus but found all remaining CE-starved cases have median required event bias at or below `4.0`.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_ce_event_margin_viability_result_report.md`.
- Source snapshot / evidence grade: strong evidence against global bonus; medium evidence that the remaining CE starvation might be near a per-step decision boundary; weak evidence about whether the near-boundary events occur after `8000ms` where the starvation is observed.

## Hypothesis

If the five remaining CE-starved cases contain many second-window steps where a valid event token is rank-near and within a small margin of the selected non-event token, then a context-adaptive selective event selector remains worth one bounded v3 experiment. If the opportunities are absent, first-window-only, or indistinguishable from pass-like/overgenerated controls, then the CE result does not justify another local v3 decode mutation.

## Root Objective

Decide whether CE leaves a real selective-decode opening in v3, or whether the remaining compression/quality work should move to target grammar or training-side structure instead of another runtime selector.

## Goal Decomposition

- Subgoal 1: Select all CE-starved cases whose median required event bias is at most `4.0`.
- Subgoal 2: Rerun selected cases with full per-step logit examples while preserving existing decode behavior.
- Subgoal 3: Count near-boundary valid event opportunities by first window and second window.
- Subgoal 4: Compare against available controls from the same CE rollout slice.
- Subgoal 5: Route the next loop to selective selector smoke, density-aware trace, v2.1/v3 grammar work, or kill.

## Candidate Variants

- Variant A: Implement a CE selective event gate immediately.
- Variant B: Run a CE-specific thresholded trace oracle over all five CE-starved low-bias cases.
- Variant C: Re-run another full CE decode sweep with a global bonus.
- Variant D: Pivot immediately to v2.1/v3 target grammar repair.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Selector smoke on CE-starved cases | Second-window event share improves without overproduction | Needs trace evidence first |
| B | Trace-only rerun with `low_bias_threshold=4.0` | Most CE-starved cases have second-window opportunities | Opportunities absent, first-window-only, or matched by controls |
| C | Full32 global decode sweep | Broad starved improvement without overgeneration | CE margin report already shows global bonus is unsafe |
| D | Synthesis-only pivot | v3 local route exhausted | CE low-bias exception remains unchecked |

## Selected Variant

- Selected: Variant B, CE-specific thresholded trace oracle.
- Rejected: Variant A and C because they change decode behavior before establishing opportunity placement.
- Deferred: Variant D until the CE low-bias exception is checked.
- Why this is the smallest useful test: it changes only evaluator configurability and reruns a small existing slice without retraining, tokenizer edits, grammar edits, or inference default changes.

## Selection Pressure

- Primary pressure: second-window opportunity evidence in CE-starved cases.
- Guard pressure: unchanged tokenizer, grammar, weights, training, and decode defaults.
- Runtime pressure: at most six 16s real-audio trace reruns, expected under 20 minutes.
- Kill pressure: if CE-starved cases do not expose second-window opportunities under the same threshold that made them low-bias, stop simple v3 selector work.

## Research Question

Does CE convert remaining v3 starvation into a local selectable event-boundary problem, or does the per-step trace show that the margin evidence is not placed where a selector can use it?

## Closest Analogies / Novelty Layer

- Closest analogies: oracle decoding diagnostics, constrained decoding trace analysis, failure-case ablation, and classifier-threshold audit.
- Relevant taxonomy bucket: local verification before bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering evidence about decode feasibility, not a representation contribution.

## Minimal Change

Add optional low/high-bias threshold parameters to the existing low-bias trace oracle and allow the CE training-gate summary `runs` list to serve as the manifest source. Then run the trace oracle with `low_bias_threshold=4.0` against the CE margin summary and CE rollout summaries.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_low_bias_trace_oracle.py`
- `tests/evals/test_mapper_v3_low_bias_trace_oracle.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- CE per-case rollout summaries under `artifacts/tmp/mapper_v3_event_token_ce_weight_training_gate/rollouts/`

## Dataset Slice

Primary CE-starved low-bias cases:

- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`
- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- `27_billiummoto_four_veiled_stars_aries_insane`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`

Controls:

- available pass-like control from the same CE rollout slice, selected by the existing oracle;
- high-bias starved control only if the CE slice contains one above the configured high-bias threshold.

## Baseline / Comparator

Comparator is the CE event-margin viability report with baseline `ce_weight2`: `5` starved cases, `14` rigid cases, `12` overgenerated cases, and all starved cases at `required_event_bias <= 4.0`.

## Primary Metric

Per CE-starved case, count of second-window opportunity steps where:

- `current_ms >= 8000`;
- argmax kind is not `event`;
- best valid event rank is at most `5`;
- best-event margin is at least `-2.0`.

## Secondary Metric

- First-window opportunity count.
- Opportunity share among second-window steps.
- Trace coverage versus diagnostic step count.
- Dead-end and max-token guard status.
- Whether pass-like controls also show the same opportunity pattern.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_low_bias_trace_oracle \
  --margin-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_event_margin_viability_summary.json \
  --manifest artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_result_report.md \
  --work-dir artifacts/tmp/mapper_v3_ce_selective_trace_oracle \
  --low-bias-limit 5 \
  --low-bias-threshold 4.0 \
  --device auto
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_low_bias_trace_oracle.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_selective_trace_oracle_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must distinguish "valid events are near the boundary" from "valid events are near the boundary in the second window where CE still starves."

## Positive Signal

At least three of five CE-starved low-bias cases have `>=10` second-window opportunity steps, with full traces and no legality failures, and the pass-like control does not make the signal indistinguishable.

## Negative Signal

Fewer than three CE-starved cases have meaningful second-window opportunities, opportunities are mostly first-window-only, or the control pattern suggests a selector would likely overproduce.

## Kill Criteria

- Required CE rollout summaries or audio paths are missing.
- Full trace examples do not cover all diagnostic steps.
- Any selected CE-starved case fails unchanged runtime legality.
- The diagnostic cannot classify opportunities by window.

## Expected Failure Modes

- CE summaries have manifest rows, but the current oracle only accepts a JSON list and needs a manifest loader extension.
- `low_bias_threshold=4.0` selects cases the original oracle skipped because it was built around a `2.0` threshold.
- MPS/auto reruns can shift exact logits enough to alter marginal opportunity counts.
- A positive opportunity count may still not imply musical correctness or safe selector behavior.

## Confounders

- Median required bias is not window-specific.
- The pass-like control is small and cannot represent all overgenerated cases.
- Trace opportunity does not prove that choosing the event token improves F1.
- The fixed 16s slice can overweight intro/early-chart structure.

## Expected Runtime / Runtime Budget

Expected runtime: under 20 minutes. Stop after one repeated runtime failure on the same case or if trace capture is incomplete after increasing example capacity once.

## Result Interpretation Plan

- Positive result would suggest: create one bounded CE selective selector smoke with strict anti-overproduction guards.
- Negative result would suggest: kill simple v3 selector work and pivot to target grammar or training-side structural instrumentation.
- Ambiguous result would require: add density/control features to trace before any decode mutation.
- Human owner decides: whether the evidence warrants another v3 selector card or a grammar-focused card.
- Next-loop action if positive: `TEST_CE_SELECTIVE_EVENT_GATE_SMOKE`.
- Next-loop action if negative: `KILL_SIMPLE_V3_SELECTOR` and `MUTATE_TO_TARGET_GRAMMAR_REPAIR`.
- Next-loop action if ambiguous: `TEST_DENSITY_AWARE_TRACE`.

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
- Remaining ambiguity: whether CE low-bias margin evidence appears in second-window trace steps.

## Next-Loop Action

- If positive: create a CE selective event-gate smoke card.
- If negative: pivot to target grammar repair.
- If ambiguous: run density-aware trace only if it can define a selector guard before implementation.

## Novelty Notes

- Closest analogies: oracle decoding diagnostics and thresholded classifier audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: evaluator instrumentation only.
