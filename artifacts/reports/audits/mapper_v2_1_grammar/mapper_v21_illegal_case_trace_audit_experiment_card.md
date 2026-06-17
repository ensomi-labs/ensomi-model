# Mapper v2.1 Illegal-Case Trace Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: continue from the C3/v3/v2.1 audit chain after C3 proved codec-side structure, v3 decode repairs stalled, and the v2.1 anti-rigid guard failed legality.
- Acceptance source, if any: `mapper_v21_v3_fixed_slice_decode_comparison_result_report.md` recommended inspecting illegal cases before scaling the anti-rigid hard block.
- Source snapshot / evidence grade: local committed reports and runtime summaries; evidence grade is high for the two illegal fixed-slice cases, medium for any generalization beyond that slice.

## Hypothesis

The remaining v2.1 grammar weakness is not that v2.1 needs a broader anti-rigid suppression rule. The two observed dead ends are likely caused by different terminal-window mechanics, such as LN carry boundary constraints, same-time lane ordering, missing legal time-shift alternatives, or anti-rigid suppression removing the only viable transition. A per-step legality trace can classify those mechanics and define a smaller repair than another blind decode penalty.

## Root Objective

Improve the fallback production path when C3 mapper integration and v3 rollout repair show diminishing returns: keep v2.1's stronger F1/starvation behavior, but remove the illegal dead-end modes before proposing any grammar mutation or decode guard for scaling.

## Goal Decomposition

- Subgoal 1: Reproduce the two known v2.1 illegal fixed-slice cases under the same checkpoint, control checkpoint, chart end, and greedy decode settings.
- Subgoal 2: Capture enough per-step state to explain why generation dead-ended: valid token class counts, top logits before and after any transform, replay state, LN carry target, same-time lane state, and anti-rigid candidate decisions.
- Subgoal 3: Classify each illegal case into a bounded repair family: no-op baseline issue, anti-rigid removes viable transition, LN carry boundary mismatch, terminal time-shift/EOS issue, same-time lane ordering issue, or model logit collapse despite legal alternatives.

## Candidate Variants

- Variant A: Scale the existing v2.1 hard-block anti-rigid guard.
  Rejected before execution because the 32-case comparison already killed it: legality failed, mean F1 regressed by `-0.036039`, and it introduced new starvation.
- Variant B: Try another anti-rigid suppression predicate immediately.
  Rejected before execution because soft-penalty and tap-only probes already failed the stress-set legality/usefulness gate, and they showed cases `04` and `05` need opposite behavior.
- Variant C: Pivot directly to another structural v3 mutation.
  Rejected for this loop because the latest v3 completion-budget and scalar loss routes failed transfer to free-running rollout, while v2.1 still has better fixed-slice mean F1 and lower starvation when legal.
- Variant D: Instrument v2.1 illegal cases before changing behavior.
  Selected because it is the smallest test that can distinguish grammar-state bugs, decode-transform damage, terminal-boundary constraints, and pure model preference failures.

## Local Verification Matrix

- Variant A: Would require all 32 v2.1 guard rollouts legal and no F1/starvation regression; already failed in `mapper_v21_v3_fixed_slice_decode_comparison_result_report.md`.
- Variant B: Would require stress-set all-legal and no new starvation; already failed for soft penalty and tap-only probes.
- Variant C: Would require evidence that v3's failure is a grammar-local issue with a bounded mutation; current reports point to generated-state exposure and event-budget calibration instead.
- Variant D: Passes locally if it reproduces both illegal cases and emits a trace that identifies the last live transition, valid alternatives, and whether the anti-rigid transform changed legality or only ranking.

## Selected Variant

- Selected: Variant D, a v2.1 illegal-case trace audit over the known illegal cases and two adjacent stress cases for contrast.
- Rejected: new guard, new penalty, broad v3 mutation, and C3 mapper-side conditioning.
- Why this is the smallest useful test: the prior reports already killed rule mutations without explaining the exact dead-end state. A trace audit is the narrowest artifact that can make the next mutation evidence-based.

## Selection Pressure

- Primary pressure: classify both illegal cases into concrete repair families.
- Guard pressure: do not change mapper defaults, tokenizer, trained weights, grammar legality, or rollout behavior in this audit.
- Runtime pressure: reuse the 32-case baseline summary and existing runtime rollout path; avoid full training.
- Kill pressure: stop if the illegal cases cannot be reproduced, if the checkpoint/control artifacts are missing, or if the trace does not capture enough state to distinguish candidate repair families.

## Research Question

Why does v2.1 dead-end on case `04` in baseline mode and case `05` under the anti-rigid guard, and what is the smallest legal grammar/decode repair family suggested by the failure trace?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding trace analysis, finite-state grammar debugging, beam-search dead-end diagnosis, and legality-mask introspection.
- Relevant taxonomy bucket: implementation/verification audit, not representation novelty.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is an engineering diagnostic for the existing v2.1 grammar; it does not propose a new representation.

## Minimal Change

Add a default-off eval script that reruns selected v2.1 rollout cases with a logits observer and optional anti-rigid transform, then writes a JSON summary and Markdown report. The script should not change generation semantics. If the existing observer lacks pre-transform context, add a minimal trace wrapper around `MapperV21AntiRigidSpacingLogitsTransform` that records candidate decisions and top-k logits without altering outputs.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v2_1_illegal_case_trace_audit.py`
- `tests/evals/test_mapper_v2_1_illegal_case_trace_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `src/pulsefield_model/models/mapper/v2_1/grammar.py`
- `src/pulsefield_model/models/mapper/v2_1/replay.py`
- `src/pulsefield_model/evals/mapper_v2_1_anti_rigid_spacing_guard.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_anti_rigid_stress_probe_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_result_report.md`

## Dataset Slice

Primary slice:

- `04_oomori_seiko_justadice_tv_size_remu_normal`, v2.1 baseline mode, chart end `7990ms`.
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`, v2.1 anti-rigid guard mode, chart end `7990ms`.

Contrast slice:

- The same two cases under the opposite mode, because case `04` was repaired by hard block while case `05` was damaged by it.
- Optional stress contrast cases `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` and `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`, because prior anti-rigid probes showed starvation/F1 regressions there.

## Baseline / Comparator

Baseline is the committed v2.1 fixed-slice comparison:

- v2.1 baseline legal: `False`, mean F1 `0.744079`, starved `2`.
- v2.1 guard legal: `False`, mean F1 `0.708040`, starved `3`.
- v3 500 wide legal: `True`, mean F1 `0.639566`, starved `11`.

The trace audit compares baseline mode and anti-rigid guard mode only for the selected cases; v3 remains read-only context.

## Primary Metric

For each illegal mode/case, produce a non-empty trace classification with:

- terminal window index and terminal replay state;
- last step with at least one valid token;
- valid token counts by token class at the last 16 steps;
- top-k valid logits and top-k raw logits at the last 16 steps;
- whether anti-rigid candidate suppression fired;
- whether the suppressed token was the only viable time-shift, a high-rank viable transition, or irrelevant;
- whether the next target boundary requires a specific LN carry state.

## Secondary Metric

- Reproduced legality status for the four primary mode/case combinations.
- Timepoint count, terminal ms, token count, second-window share, F1@100ms, dominant spacing ratio, and starved flag for each traced rollout.
- Count of steps with zero valid tokens before completion.
- Count of steps where valid non-EOS tokens exist but all high-rank logits point outside the valid mask.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_mapper_v2_1_illegal_case_trace_audit.py tests/inference/test_mapper_v2_1_rollout.py -q
uv run python -m pulsefield_model.evals.mapper_v2_1_illegal_case_trace_audit
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_summary.json >/dev/null
```

## Guard Check

- The audit must not change existing v2.1 rollout outputs when tracing is disabled.
- Existing v2.1 rollout tests must still pass.
- The trace must record only generated-prefix state and legal runtime metadata; no reference beatmap events may influence decode.
- If anti-rigid mode is enabled, the transform's output logits must match the existing transform for the same step.
- Reported classification must include enough evidence to justify or reject a next grammar mutation.

## Qualitative Check

Manually inspect the final 16 traced steps for cases `04` and `05`. Confirm whether the failure is caused by terminal-boundary legality, LN open/close state, same-time lane ordering, anti-rigid suppression, or model ranking collapse.

## Positive Signal

- Both known illegal cases are reproduced.
- Each illegal case receives a concrete, evidence-backed failure class.
- At least one next-loop repair can be stated as a narrower mutation than "scale anti-rigid suppression".
- The trace distinguishes case `04` and case `05` mechanics, matching the prior observation that they require opposite behavior.

## Negative Signal

- Illegal cases do not reproduce under the same configuration.
- The trace only says "dead end" without exposing the valid-mask/logit/state reason.
- The failure classes are unrelated across cases and require a broad rewrite.
- The trace suggests the model strongly prefers invalid tokens while grammar remains healthy, implying grammar mutation is not the right next loop.

## Kill Criteria

Kill this route if the audit cannot reproduce the illegal cases, if it requires changing generation semantics to observe the failure, if checkpoint/control artifacts are unavailable, or if traces cannot classify failures beyond already-known aggregate metrics.

## Expected Failure Modes

- The observer captures post-mask logits only and misses the raw model preference; mitigate by recording both raw and valid-filtered top-k where available.
- The dead end occurs in a later full-song carry state that is hard to isolate; mitigate by recording window-level carry-in/out and terminal state.
- The anti-rigid wrapper accidentally changes ranking; guard by asserting transformed logits match the existing transform on a synthetic test.
- The two cases have unrelated causes; report that and avoid forcing one repair family.

## Confounders

- The fixed-slice cases come from a 32-map runtime audit, not the full 4k dataset.
- v2.1's stronger F1 may partly reflect over-rigid timing that is musically undesirable.
- The chart end is near `7990ms`, so terminal-window behavior may be overrepresented.
- Greedy decoding can make a finite penalty behave like a hard block.

## Expected Runtime / Runtime Budget

Unit tests should finish in seconds. The traced runtime audit should finish in under 10 minutes because it runs two to four selected real-audio cases, not training or a full 32-case sweep. Stop after 15 minutes or after the two primary illegal traces are produced.

## Result Interpretation Plan

- Positive result would suggest: write a follow-up grammar mutation card targeted to the observed failure class.
- Negative result would suggest: stop v2.1 grammar mutation and return to v3 structural grammar or mapper training calibration.
- Ambiguous result would require: add one narrower trace field, not a new decode rule.
- Human owner decides: whether v2.1 is worth repairing further versus returning to v3.
- Next-loop action if positive: create `mapper_v21_legality_first_grammar_repair_experiment_card.md` with exactly one repair family.
- Next-loop action if negative: mark v2.1 anti-rigid repair as killed/deferred and pick a structural v3 card.
- Next-loop action if ambiguous: extend this trace audit once with the missing observation, then decide.

## Result Log Template

- Experiment: mapper v2.1 illegal-case trace audit
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
- Remaining ambiguity: the exact repair family is intentionally unresolved until the trace classifies the two illegal cases.

## Next-Loop Action

- If positive: implement one legality-first v2.1 grammar/decode repair card targeted to the observed failure class.
- If negative: stop v2.1 anti-rigid repair and return to v3 structural grammar mutation.
- If ambiguous: add the single missing trace dimension and rerun only the primary two cases.

## Novelty Notes

- Closest analogies: constrained decoder trace audit, legality-mask debugging, finite-state grammar repair.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering verification only; no representation novelty is claimed.
