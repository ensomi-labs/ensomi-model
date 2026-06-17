# Target Grammar v3 Trace-Conditioned Spacing Escape Smoke Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: generated-prefix state trace routed to `TEST_BOUNDARY_OR_SPACING_STATE_REPAIR`.
- Acceptance source, if any: `target_grammar_v3_generated_prefix_state_trace_audit_result_report.md`.
- Source snapshot / evidence grade: high that six selected generated-prefix failures are traceable; high that all six have primary `time_shift_repetition`; medium that secondary `boundary_drift` and `event_underselection` are actionable rather than downstream symptoms.

## Hypothesis

The v3 500-step generated-prefix failures are not caused by missing teacher-forced time-shift knowledge, but by local rollout-state collapse into repeated spacing and boundary starvation. A narrow trace-conditioned decode smoke that only activates after repeated generated spacing and near boundary/starvation states should reduce rigidity and improve second-window continuation on the six traced cases without recreating the full32 anti-rigid overgeneration failure.

## Root Objective

Decide whether v3 has a small local spacing/boundary repair path after the generated-prefix trace, or whether this branch should stop local decode repairs and pivot to training-side state/objective work or v2.1/target-grammar mutation.

## Goal Decomposition

- Subgoal 1: test whether a generated-history-conditioned spacing escape can break the 160 ms repeated-spacing loop on the same six traced cases.
- Subgoal 2: test whether boundary/starvation continuation can improve without target leakage or C3 sidecar conditioning.
- Subgoal 3: guard against the known failure mode from prior anti-rigid full32 tests: event overproduction and dead-end behavior.

## Candidate Variants

- Variant A: rerun broad hard/soft anti-rigid decode on the six traced cases. This is too close to a killed family: full32 anti-rigid reduced rigidity but failed median event-ratio and, for tap-only, legality/dead-end gates.
- Variant B: start a new training objective immediately. This skips the local mechanism check and repeats the mistake of moving from trace evidence to training without proving a bounded repair surface.
- Variant C: selected trace-conditioned spacing/boundary escape smoke. Implement an opt-in eval-only logits transform that activates only from generated history and model logits: repeated-spacing escape plus boundary/starvation event opportunity, with strict event-count and legality guards.
- Variant D: pivot directly to v2.1 grammar mutation. Plausible if Variant C fails, but premature before testing the trace-local mechanism that just became observable.

## Local Verification Matrix

- Variant A: reject unless old full32 kill evidence is invalid. Current evidence says it is valid: hard/soft anti-rigid failed median event-ratio, and tap-only had dead-end behavior.
- Variant B: reject unless no cheap generated-prefix repair can be tested. Current trace gives a cheap local repair target.
- Variant C: pass if the smoke can run without training, use no target labels in decode, improve rigidity/continuation on at least four of six traced cases, and keep sentinel overproduction guards clean.
- Variant D: defer unless Variant C is uninstrumentable, overproduces, or fails to improve continuation.

## Selected Variant

- Selected: Variant C, trace-conditioned spacing/boundary escape smoke.
- Rejected: A, B, and immediate D.
- Why this is the smallest useful test: it uses the exact generated-prefix mechanism from the trace, runs bounded real-audio rollouts, avoids changing defaults, and includes sentinel controls for the known anti-rigid failure mode.

## Selection Pressure

- Primary pressure: improve generated-prefix continuation by reducing repeated-spacing dominance and increasing second-window event share on the six traced failures.
- Guard pressure: no training, no tokenizer/grammar/default change, no target-derived labels or C3 sidecar input in decode, no median event-ratio overproduction.
- Runtime pressure: at most ten real-audio fixed-slice rollouts: six traced failures plus four overproduction/dead-end sentinels.
- Kill pressure: stop if the policy behaves like the killed broad anti-rigid family.

## Research Question

Can a trace-conditioned, generated-history-only spacing/boundary escape improve the v3 generated-prefix failure cases without the overgeneration/dead-end failure that killed broad anti-rigid decode?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding, exposure-bias recovery heuristics, repetition penalties, boundary-aware sequence continuation guards.
- Relevant taxonomy bucket: model diagnostics and bounded decode intervention.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for deciding whether the v3 target grammar remains worth local repair.

## Minimal Change

Add an eval-only smoke runner with a local logits transform. The transform may inspect only generated prefix state, valid-token mask, current logits, and rollout-local counters. It must not inspect reference events, target tokens, future chart content, or C3 side-stream labels during decoding.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_trace_conditioned_spacing_escape_smoke.py`
- `tests/evals/test_mapper_v3_trace_conditioned_spacing_escape_smoke.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_result_report.md`
- `src/pulsefield_model/evals/mapper_v3_generated_prefix_state_trace_audit.py`
- `src/pulsefield_model/evals/mapper_v3_ce_antirigid_decode_stress.py`
- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `src/pulsefield_model/inference/mapper_v3_rollout.py`

## Dataset Slice

Primary traced failures:

- `14_oomori_seiko_justadice_tv_size_remu_hard`
- `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- `19_nekodex_circles_famoss_hard`
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- `18_billiummoto_four_veiled_stars_aries_famoss_hard`

Sentinel controls from prior anti-rigid failure reports:

- `23_usao_knight_rider_kuo_kyoka_expert`
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`
- `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`
- `01_hatsuki_yura_guren_yasha_a_m_d_normal`

Use the same chart-end, real-audio, mapper checkpoint, control checkpoint, and normalized difficulty as the existing fixed-slice summary for each case.

## Baseline / Comparator

Baseline:

- Current 500-step v3 generated-prefix fixed-slice summary for primary cases.
- Existing CE anti-rigid full32 reports as negative comparator for overgeneration/dead-end failure modes.

Comparator metrics:

- Generated event count ratio.
- Second-window event share.
- Dominant spacing ratio.
- Timing F1@100 ms.
- Legality/dead-end/max-token rollout status.

## Primary Metric

Primary pass condition on the six traced failures:

- at least four of six reduce dominant spacing ratio by at least `0.15` or reach `<= 0.75`; and
- at least four of six improve second-window event share by at least `0.10` absolute or reach `>= 0.15`; and
- no primary case has event-count ratio above `1.25`.

## Secondary Metric

- mean/median event-count ratio over primary and sentinel cases;
- sentinel overproduction count with event-count ratio `> 1.25`;
- dead-end and max-token case counts;
- timing F1@100 ms delta;
- transform activation count, repeated-spacing activation count, boundary activation count;
- first-failure reclassification using the generated-prefix trace classifier if trace rows are collected.

## Verify Command / Evaluation Procedure

Planned commands:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_trace_conditioned_spacing_escape_smoke
uv run --group dev pytest tests/evals/test_mapper_v3_trace_conditioned_spacing_escape_smoke.py tests/evals/test_mapper_v3_generated_prefix_state_trace_audit.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/inference/test_mapper_v3_rollout.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_summary.json >/tmp/trace_conditioned_spacing_escape.valid.json
```

## Guard Check

- no training;
- no tokenizer, grammar, or default decode change;
- no reference or future target access inside the transform;
- all selected cases have either candidate rollout metrics or explicit missing-case reason;
- all candidate rollouts are legal;
- zero dead-end cases and zero max-token cases;
- sentinel median event-count ratio stays `<= 1.25`;
- max boundary-event ratio does not exceed the baseline by more than `0.05` absolute.

## Qualitative Check

Inspect activation examples for at least two primary cases and two sentinel cases. Verify that activations are explainable from generated-prefix state and logits only, not target timing. Confirm that any continuation gain does not come from indiscriminate event spam.

## Positive Signal

At least four of six primary traced failures improve both rigidity and second-window continuation while all guard checks pass and sentinel controls do not overproduce.

## Negative Signal

The policy either repeats the old anti-rigid failure mode, does not improve second-window continuation, only shifts events into overproduction, or depends on target/reference information.

## Kill Criteria

- any transform implementation requires reference event labels during decode;
- any primary or sentinel candidate rollout is illegal, dead-ended, or max-tokened;
- sentinel median event-count ratio `> 1.25`;
- fewer than four primary cases improve rigidity;
- fewer than four primary cases improve second-window continuation;
- timing F1 mean drops by more than `0.05` absolute on primary cases;
- result recommends default decode changes before a full32 gate.

## Expected Failure Modes

- The transform may reduce repetition but overproduce events, matching the killed broad anti-rigid family.
- Repetition may be a symptom of missing learned density/structure rather than a decodable local state issue.
- Boundary/starvation activations may occur too late to recover second-window structure.
- Sentinels may reveal the policy is not selective enough.

## Confounders

- The six primary cases are selected for a specific failure class and are not full32/full-dataset evidence.
- The sentinel set is small and only protects against known overgeneration/dead-end patterns.
- Greedy decode behavior may differ from stochastic or beam policies.
- Real-audio cache preparation and checkpoint state must match prior fixed-slice setup.

## Expected Runtime / Runtime Budget

Expected runtime: under 30 minutes for ten bounded real-audio rollouts on CPU/MPS. Stop after two consecutive rollout infrastructure failures or immediately after any implementation path requires target leakage.

## Result Interpretation Plan

- Positive result would suggest: create a full32 opt-in smoke card for the same trace-conditioned policy before any default change.
- Negative result would suggest: kill local decode repair for this branch and pivot to training-side state/objective work or v2.1/target-grammar mutation.
- Ambiguous result would require: one-case trace instrumentation around activation rows, not broad training.
- Human owner decides: whether a positive smoke is worth a full32 gate.
- Next-loop action if positive: full32 trace-conditioned spacing escape gate.
- Next-loop action if negative: training-side state/objective card or v2.1 grammar mutation card.
- Next-loop action if ambiguous: one-case activation trace audit.

## Result Log Template

- Experiment: target_grammar_v3_trace_conditioned_spacing_escape
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
- Remaining ambiguity: the smoke can decide whether the trace-local repair surface exists, but it cannot prove full32 or production readiness.

## Next-Loop Action

- If positive: create and run a full32 opt-in trace-conditioned spacing escape gate.
- If negative: stop local decode repairs and route to training-side state/objective work or v2.1/target-grammar mutation.
- If ambiguous: run a one-case activation trace audit.

## Novelty Notes

- Closest analogies: constrained decoding, repetition penalty, exposure-bias recovery, boundary-aware sequence continuation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation and diagnostic selection pressure, not a representation contribution.
