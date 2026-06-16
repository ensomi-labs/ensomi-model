# Target Grammar v3 Low-Bias Trace Oracle Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_event_margin_decode_viability` killed a global event-bonus route, but surfaced three low-bias starved cases where best valid event tokens are near the decision boundary. Before pivoting away from v3 local decode work, trace those cases at per-step resolution and check whether the low-bias opportunities occur in the second window where starvation happens.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_event_margin_decode_viability_result_report.md`.
- Source snapshot / evidence grade: strong evidence against global event bonus; medium evidence that only a selective per-step gate could still help; weak evidence about whether useful opportunities happen after `8000ms`.

## Hypothesis

If the low-bias starved cases contain many second-window steps where a valid event token is rank-near and within a small margin of the selected time-shift token, then a selective state-aware event gate remains worth one more v3 experiment. If those opportunities are absent or mostly first-window-only, then the remaining v3 decode path has poor leverage and we should pivot toward v2.1 grammar improvement or richer training-side instrumentation rather than adding another selector.

## Root Objective

Decide whether the only remaining small v3 decode path after the global-bonus kill is viable enough to test, or whether the evidence now supports stopping local v3 decode probes.

## Goal Decomposition

- Subgoal 1: Rerun the low-bias starved cases with full per-step logit examples, preserving existing runtime behavior.
- Subgoal 2: Count low-bias event opportunities by first window (`<8000ms`) and second window (`>=8000ms`).
- Subgoal 3: Compare against one high-bias starved control and one pass-like control.
- Subgoal 4: Route the next loop to selective selector test, richer trace/oracle, or v2.1 grammar improvement.

## Candidate Variants

- Variant A: Implement a selective event selector immediately. Rejected because we do not yet know whether low-bias opportunities occur in the starved second window.
- Variant B: Full per-step trace oracle on three low-bias starved cases plus two controls. Selected because it is small, runtime-backed, and directly tests the only remaining simple v3 decode hypothesis.
- Variant C: Run another full32 decode sweep. Rejected because global decode knobs already failed and this would spend runtime without a new mechanism.
- Variant D: Pivot immediately to v2.1 grammar improvement. Deferred until Variant B checks the low-bias exception.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Selector smoke on low-bias cases | Event gate improves second-window share | Needs trace evidence first |
| B | Trace-only rerun | Second-window low-bias opportunities exist in most low-bias cases | Opportunities are absent or first-window-only |
| C | Full32 policy sweep | Broad continuation improves | Prior decode-policy sweep already killed this family |
| D | Synthesis-only pivot | v3 local route exhausted | Low-bias exception remains unchecked |

## Selected Variant

- Selected: Variant B, low-bias trace oracle.
- Rejected: A and C because they change behavior before checking opportunity placement.
- Deferred: D until the low-bias exception is resolved.
- Why this is the smallest useful test: it modifies only eval diagnostics and reruns at most five existing manifest cases without retraining or runtime-default changes.

## Selection Pressure

- Primary pressure: determine whether low-bias event opportunities exist in the second window.
- Guard pressure: no tokenizer, grammar, model, training, or inference default changes.
- Runtime pressure: under 15 minutes for five real-audio 16s rollouts.
- Kill pressure: if low-bias cases do not show second-window opportunities, stop simple v3 decode probing.

## Research Question

Are the low-bias starved v3 cases actually fixable by a selective event gate, or are their near-boundary event logits concentrated outside the failure window?

## Closest Analogies / Novelty Layer

- Closest analogies: oracle decoding diagnostics, constrained-decoding trace analysis, exposure-bias localization, and failure-case ablation.
- Relevant taxonomy bucket: local verification before a bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering evidence about decode feasibility, not representation novelty.

## Minimal Change

Add optional full-example capture to the existing v3 runtime rollout diagnostic, then add an artifact-only evaluator that:

- reads the v3 event-margin summary and fixed-slice manifest;
- selects the three low-bias starved cases from the baseline report;
- adds one high-bias starved control and one pass-like control;
- reruns those cases with `collect_logit_diagnostics=true` and enough example capacity to cover all steps;
- classifies each trace step by window, argmax kind, best-event rank, and margin;
- reports whether low-bias event opportunities appear after `8000ms`.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_low_bias_trace_oracle.py`
- `tests/evals/test_mapper_v3_trained_runtime_rollout.py`
- `tests/evals/test_mapper_v3_low_bias_trace_oracle.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- existing per-case rollout summaries under `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/rollouts/`

## Dataset Slice

Primary low-bias starved cases from baseline:

- `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv`
- `23_usao_knight_rider_kuo_kyoka_expert`
- `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`

Controls:

- one high-bias starved case, preferably `14_oomori_seiko_justadice_tv_size_remu_hard`;
- one pass-like case with stable second-window behavior, selected from the baseline summary.

## Baseline / Comparator

Baseline is the current 500-step v3 fixed-slice rollout summary: `11` starved cases, global event-bonus route killed because `8/11` starved cases need median event bias `>4.0`.

## Primary Metric

Per low-bias case, count of second-window opportunity steps where:

- `current_ms >= 8000`;
- argmax kind is not `event`;
- best valid event rank is at most `5`;
- best-event margin is at least `-2.0`.

## Secondary Metric

- First-window opportunity count.
- Opportunity share among all steps and among second-window steps.
- Best-event margin/rank distribution by window.
- Whether trace examples cover the full rollout step count.
- Dead-end and max-token guard status.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_low_bias_trace_oracle \
  --margin-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_margin_decode_viability_summary.json \
  --manifest artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_result_report.md \
  --device auto
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_low_bias_trace_oracle.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The result report must distinguish "events are rank-near somewhere" from "events are rank-near in the starved second window."

## Positive Signal

At least two of the three low-bias starved cases have `>=10` second-window low-bias opportunities and no rollout legality failures. This routes to a bounded selective selector smoke.

## Negative Signal

Fewer than two low-bias starved cases have meaningful second-window opportunities, or opportunities are mostly first-window-only. This kills the simple selective decode route.

## Kill Criteria

- Required checkpoints, manifest rows, or audio files are missing.
- Full trace examples do not cover all diagnostic steps.
- Any selected case fails runtime legality under unchanged baseline decode.
- The diagnostic cannot classify opportunities by window.

## Expected Failure Modes

- Real-audio runtime reruns are slower than expected.
- MPS/auto device nondeterminism changes exact margins slightly.
- Low-bias median cases may still have sparse second-window opportunities because the median is dominated by first-window steps.
- A true selective selector may need reference-free density features that this trace does not evaluate.

## Confounders

- The low-bias threshold came from median margins, not a per-window oracle.
- The high-bias and pass-like controls are small and do not represent the full 32-case slice.
- Trace opportunity does not prove musical correctness of the event token.
- The fixed-slice cases are adversarial and not a full dataset result.

## Expected Runtime / Runtime Budget

Expected runtime: under 15 minutes. Stop after one repeated runtime failure on the same case.

## Result Interpretation Plan

- Positive result would suggest: create `target_grammar_v3_selective_event_gate_smoke`.
- Negative result would suggest: stop simple v3 decode probing and create a v2.1 grammar improvement card or richer training-side trace plan.
- Ambiguous result would require: one deeper trace with density/control features on the low-bias cases.
- Human owner decides: whether to spend another v3 selector card or pivot immediately to v2.1 grammar work.
- Next-loop action if positive: implement a bounded selector smoke only for the traced cases.
- Next-loop action if negative: pivot to v2.1 grammar improvement, preserving v3 artifacts.
- Next-loop action if ambiguous: add density/control features to trace before changing decode behavior.

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
- Remaining ambiguity: trace opportunity does not prove the selected event is musically correct.

## Next-Loop Action

- If positive: create and run a selective event-gate smoke on the traced cases.
- If negative: pivot to v2.1 grammar improvement or broader training-side instrumentation.
- If ambiguous: add density/control-aware trace fields.

## Novelty Notes

- Closest analogies: oracle decoding diagnostics and constrained-decoding trace analysis.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic only.
