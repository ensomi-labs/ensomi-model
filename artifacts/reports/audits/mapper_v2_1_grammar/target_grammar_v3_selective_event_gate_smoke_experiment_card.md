# Target Grammar v3 Selective Event-Gate Smoke Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_low_bias_trace_oracle` found that all three low-bias starved cases have many second-window rank-near event opportunities. The global event-bonus route is killed, so the next bounded v3 decode test is a selective gate that acts only on those opportunities.
- Acceptance source, if any: active thread goal plus `target_grammar_v3_event_margin_decode_viability_result_report.md` and `target_grammar_v3_low_bias_trace_oracle_result_report.md`.
- Source snapshot / evidence grade: strong evidence against global event bonus; medium evidence that low-bias second-window opportunities exist; weak evidence that forcing those opportunities improves chart metrics rather than overproducing.

## Hypothesis

A selective second-window event gate can reduce starvation on the low-bias v3 cases without causing max-token, duplicate, or overgeneration failures. If a bounded gate improves second-window share on at least two of the three low-bias starved cases while preserving legality and event-count ratio, then the low-bias v3 decode path remains viable for one more targeted experiment. If it overproduces, regresses timing F1, or fails controls, the selective decode path should be killed and the next work should pivot to v2.1 grammar improvement or training-side instrumentation.

## Root Objective

Test whether the only remaining simple v3 decode hypothesis has enough evidence to continue toward v3 replacement readiness, without changing tokenizer, grammar, training, or runtime defaults.

## Goal Decomposition

- Subgoal 1: Add an eval-only logits transform hook that leaves default v3 runtime behavior unchanged.
- Subgoal 2: Implement a selective event gate using only online-available state: current timestamp, valid-token mask, event rank, event margin, per-window forced count, and minimum forced-event gap.
- Subgoal 3: Rerun the three low-bias starved cases plus the high-bias and pass-like controls from the trace oracle.
- Subgoal 4: Compare generated metrics against the fixed-slice 500-step baseline rows.

## Candidate Variants

- Variant A: Global event bonus. Rejected because `target_grammar_v3_event_margin_decode_viability` killed this path.
- Variant B: Uncapped selective event forcing on every low-bias opportunity. Rejected because the trace oracle showed up to `98` second-window opportunities in one case, making overproduction likely.
- Variant C: Capped second-window selective event gate with minimum forced-event gap. Selected because it tests the low-bias exception while explicitly guarding overproduction.
- Variant D: Immediate v2.1 grammar pivot. Deferred until the low-bias exception is tested with actual generation behavior.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing margin diagnostic | Broad event bonus safe | Killed: most starved cases need large bias |
| B | Trace opportunity count | Many opportunities imply fix | Too many opportunities imply overproduction |
| C | Five-case selective smoke | Starvation improves without legality/count/F1 regression | Overproduction, duplicate collapse, or weak second-window improvement |
| D | Synthesis-only pivot | v3 decode path exhausted | Low-bias selective path remains untested |

## Selected Variant

- Selected: Variant C, capped second-window selective event gate.
- Rejected: A and B for safety/evidence reasons.
- Deferred: D until this smoke result is known.
- Why this is the smallest useful test: it changes only diagnostic-time logits, uses the same runtime checkpoint and five already-traced cases, and can fail before any training or default inference change.

## Selection Pressure

- Primary pressure: improve second-window event share on low-bias starved cases.
- Guard pressure: no dead-end, no max-token failure, no event-count overproduction, no large timing-F1 regression, and pass-like control must not regress.
- Runtime pressure: under 15 minutes for five real-audio 16s rollouts.
- Kill pressure: if the gate needs aggressive forcing that creates overproduction or F1 regression, stop simple v3 decode probing.

## Research Question

Can a bounded online selective event gate turn observed second-window low-bias opportunities into improved v3 rollout metrics without breaking legality or controls?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding with local reranking, oracle-informed decode ablation, exposure-bias mitigation, and runtime calibration smoke tests.
- Relevant taxonomy bucket: local verification before a bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation on decode policy, not representation novelty.

## Minimal Change

Add an optional logits transform hook to the v3 rollout path with default `None`, then add an eval-only smoke script that:

- selects the same five cases from the low-bias trace oracle summary;
- wraps v3 logits with a selective gate only for `current_ms >= 8000`;
- forces the best valid event only when argmax is not an event, best-event rank is `<=5`, and margin is `>=-2.0`;
- caps forced events per 8s window and enforces a minimum gap between forced events;
- recomputes event-count ratio, second-window share, boundary/duplicate ratios, dominant spacing, and timing F1 against source beatmaps;
- compares candidate metrics to baseline fixed-slice rows.

## Files Likely to Change

- `src/pulsefield_model/inference/mapper_v3_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_selective_event_gate_smoke.py`
- `tests/inference/test_mapper_v3_rollout.py`
- `tests/evals/test_mapper_v3_selective_event_gate_smoke.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- per-case baseline summaries under `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/rollouts/`

## Dataset Slice

Use the five cases selected by the trace oracle:

- low-bias starved: `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv`
- low-bias starved: `23_usao_knight_rider_kuo_kyoka_expert`
- low-bias starved: `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`
- high-bias starved control: `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- pass-like control: `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz`

## Baseline / Comparator

Comparator is the v3 500-step fixed-slice wide-audit row for each selected case. Baseline low-bias cases are legal but second-window-starved:

- case `10`: second-window share `0.0526`, event ratio `0.4176`, F1 `0.5116`;
- case `23`: second-window share `0.0612`, event ratio `0.4712`;
- case `26`: second-window share `0.0213`, event ratio `0.3456`.

## Primary Metric

Count of low-bias starved cases whose second-window event share improves by at least `0.15` while remaining legal and keeping event-count ratio `<=1.25`.

## Secondary Metric

- Timing F1 @100ms delta versus baseline.
- Generated/reference event-count ratio.
- Boundary-event ratio.
- Duplicate/nonincreasing spacing ratio.
- Dominant spacing ratio.
- Forced event count by case and by window.
- Pass-like control F1/event-count regression.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_selective_event_gate_smoke \
  --trace-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_low_bias_trace_oracle_summary.json \
  --baseline-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_result_report.md \
  --device auto
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_selective_event_gate_smoke.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/inference/test_mapper_v3_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The report must separate improved continuation from overproduction. A higher second-window share is not positive if event-count ratio or F1 degrades materially.

## Positive Signal

At least two of three low-bias starved cases improve second-window share by `>=0.15`, stay legal, keep event-count ratio `<=1.25`, and do not regress F1 by more than `0.05`. The pass-like control must not regress F1 by more than `0.05` or event-count ratio by more than `0.25`.

## Negative Signal

The gate improves fewer than two low-bias cases, overproduces any low-bias case above event-count ratio `1.25`, causes duplicate/boundary collapse, or regresses the pass-like control.

## Kill Criteria

- Any selected baseline rerun has dead-end or max-token failure.
- The transform changes behavior when disabled.
- Candidate summary cannot compute reference metrics.
- The gate requires reference/future information.
- The pass-like control regresses beyond the guard.

## Expected Failure Modes

- Forced events may improve second-window share but lower precision/F1.
- Forced events may change state so later rank-near opportunities disappear.
- The per-window cap may be too weak to move the hard cases or too strong for controls.
- Low-bias cases may still be musically wrong even if event-count ratio improves.

## Confounders

- This is only a five-case smoke, not a full32 or full4k result.
- F1 over timestamps ignores lane correctness.
- The event gate uses logits and grammar state only, but its thresholds are hand-selected from the trace oracle.
- Real-audio runtime reruns may differ slightly by device.

## Expected Runtime / Runtime Budget

Expected runtime: under 15 minutes. Stop after one repeated runtime failure on the same case.

## Result Interpretation Plan

- Positive result would suggest: run a full32 selective-gate audit before any v3 replacement claim.
- Negative result would suggest: kill simple v3 decode probing and pivot to v2.1 grammar improvement or training-side v3 instrumentation.
- Ambiguous result would require: threshold/cap sensitivity on the same five cases, not full32.
- Human owner decides: whether a full32 selective gate audit is worth running.
- Next-loop action if positive: create `target_grammar_v3_selective_event_gate_full32_audit`.
- Next-loop action if negative: create a v2.1 grammar improvement Experiment Card.
- Next-loop action if ambiguous: run one small threshold sensitivity diagnostic.

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
- Remaining ambiguity: this smoke can prove a local decode lever is worth auditing; it cannot prove full pipeline replacement readiness.

## Next-Loop Action

- If positive: run a full32 selective-gate audit.
- If negative: pivot to v2.1 grammar improvement or training-side instrumentation.
- If ambiguous: run a bounded threshold sensitivity diagnostic.

## Novelty Notes

- Closest analogies: local constrained-decoding reranking and oracle-informed decode ablation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering decode-policy test only.
