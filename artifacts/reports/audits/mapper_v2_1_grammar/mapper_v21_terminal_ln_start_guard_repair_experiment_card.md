# Mapper v2.1 Terminal LN-Start Guard Repair Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: execute the next-loop action from `mapper_v21_illegal_case_trace_audit_result_report.md`.
- Acceptance source, if any: `mapper_v21_illegal_case_trace_audit_summary.json` reproduced both primary illegal cases and classified them as `zero_valid_terminal_state` and `anti_rigid_path_damage`.
- Source snapshot / evidence grade: local runtime trace evidence; high for cases `04` and `05`, medium for full 32-case generalization.

## Hypothesis

The v2.1 dead ends are caused by allowing new non-carryable LN starts too close to a non-final write-window boundary. At `7990ms`, the model can open an LN, but the next `TS_10` to `8000ms` is illegal because `ln_carry_out` is empty, and same-time lane order prevents closing the new LN afterward. A default-off terminal LN-start guard using the existing `min_ln_duration_ms` legality primitive should prevent this zero-valid terminal state without another broad anti-rigid suppression rule.

## Root Objective

Repair the v2.1 fallback production path enough to keep its stronger F1/starvation profile while removing known legality failures, after C3 mapper-side integration and v3 decode repairs showed diminishing returns.

## Goal Decomposition

- Subgoal 1: Expose the existing v2.1 `min_ln_duration_ms` grammar option through runtime generation without changing defaults.
- Subgoal 2: Verify that `min_ln_duration_ms=20` fixes the traced terminal zero-valid failures for case `04` baseline and case `05` anti-rigid guard.
- Subgoal 3: Verify that the guard does not create new starvation, max-token loops, or broad F1 regression on the four stress cases and, if the stress gate passes, on the full 32-case fixed slice.

## Candidate Variants

- Variant A: Scale or retune anti-rigid hard/soft/tap-only suppression.
  Rejected before execution because hard, soft, and tap-only anti-rigid variants already failed legality or quality gates.
- Variant B: Allow cross-window LN carry in v2.1 runtime.
  Rejected for this loop because it changes the v2.1 carry contract and starts moving toward a different replay model rather than repairing the observed local failure.
- Variant C: Relax same-time lane ordering near the window boundary.
  Rejected because v2.1 canonical lane ordering is a core grammar invariant; relaxing it risks duplicate/ambiguous same-time actions.
- Variant D: Block non-carryable `HOLD_START` when remaining write-window time is below a minimum duration.
  Selected because the grammar already implements this check behind `min_ln_duration_ms`, it directly targets the traced terminal state, and it can be tested default-off.

## Local Verification Matrix

- Variant A: would need both illegal cases legal and no F1/starvation regression; prior hard/soft/tap-only reports already failed.
- Variant B: would need a new carry-aware dataset/runtime contract; too broad before testing a local guard.
- Variant C: would need proof that order relaxation is lossless and unambiguous; current evidence shows order violations are invalid, not missing representation.
- Variant D: passes local verification if a synthetic unit test blocks `HOLD_START` at `remaining_ms=10` with empty carry-out, allows legal non-terminal/longer starts, and allows carry-out starts when `ln_carry_out` explicitly expects them.

## Selected Variant

- Selected: Variant D, terminal non-carryable LN-start guard using `min_ln_duration_ms=20`.
- Rejected: new anti-rigid rule, cross-window carry rewrite, same-time lane-order relaxation.
- Why this is the smallest useful test: it uses an existing legality primitive, keeps the default behavior unchanged, and targets the exact state in both illegal traces.

## Selection Pressure

- Primary pressure: make the two primary illegal mode/case combinations legal.
- Guard pressure: do not regress legal contrast cases, do not create max-token loops, and do not increase starvation on the stress set.
- Runtime pressure: run selected real-audio cases first; only run the full 32-case fixed slice if the stress gate passes.
- Kill pressure: kill the repair if it cannot fix case `04` baseline and case `05` anti-rigid guard together.

## Research Question

Is v2.1 legality recoverable with a default-off terminal LN-start duration guard, or do the traced failures require a broader grammar/replay redesign?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained-decoding minimum-duration guard, finite-state decoder dead-end prevention, legality-mask hardening.
- Relevant taxonomy bucket: engineering repair / grammar safety gate.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation in v2.1 runtime grammar; no representation novelty.

## Minimal Change

Thread an optional `min_ln_duration_ms` argument through v2.1 runtime generation and rollout helpers, defaulting to `None`. Add an eval that runs `min_ln_duration_ms=20` as a default-off terminal LN-start guard. Do not change mapper defaults, tokenizer, vocabulary, trained weights, or dataset cache schema.

## Files Likely to Change

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `src/pulsefield_model/evals/mapper_v2_1_trained_runtime_rollout.py`
- `src/pulsefield_model/evals/mapper_v21_terminal_ln_start_guard_repair.py`
- `tests/inference/test_mapper_v2_1_rollout.py`
- `tests/evals/test_mapper_v21_terminal_ln_start_guard_repair.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/models/mapper/v2_1/grammar.py`
- `src/pulsefield_model/models/mapper/v2_1/replay.py`
- `src/pulsefield_model/evals/mapper_v2_1_illegal_case_trace_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_illegal_case_trace_audit_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_tap_only_anti_rigid_stress_probe_result_report.md`

## Dataset Slice

Primary required gate:

- `04_oomori_seiko_justadice_tv_size_remu_normal`, baseline + terminal LN-start guard.
- `04_oomori_seiko_justadice_tv_size_remu_normal`, anti-rigid + terminal LN-start guard.
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`, baseline + terminal LN-start guard.
- `05_usao_knight_rider_kuo_kyoka_dnm_s_normal`, anti-rigid + terminal LN-start guard.

Stress contrast gate:

- The four stress cases from the soft/tap-only probes: cases `04`, `05`, `29`, and `31`, each with baseline + guard and anti-rigid + guard.

Widening gate:

- The committed 32-case fixed-slice comparison universe, only if the stress gate passes.

## Baseline / Comparator

Primary comparator is the committed v2.1 trace audit:

- case `04` baseline: illegal, terminal `7990ms`, family `zero_valid_terminal_state`.
- case `04` guard: legal, anti-rigid blocks `19`.
- case `05` baseline: legal.
- case `05` guard: illegal, terminal `7990ms`, family `anti_rigid_path_damage`, one blocked `TS_300` at `1600ms`.

Secondary comparator is the full 32-case v2.1/v3 fixed-slice report:

- v2.1 baseline: legal `False`, mean F1 `0.744079`, starved `2`.
- v2.1 anti-rigid guard: legal `False`, mean F1 `0.708040`, starved `3`.
- v3 500 wide: legal `True`, mean F1 `0.639566`, starved `11`.

## Primary Metric

Primary stress metric:

- all four primary mode/case combinations legal;
- case `04` baseline + terminal guard completes;
- case `05` anti-rigid + terminal guard completes;
- zero max-token cases.

## Secondary Metric

- F1@100ms delta versus the corresponding unguarded comparator.
- Starved count versus the corresponding unguarded comparator.
- Generated event-count ratio.
- Dominant spacing ratio.
- Count of blocked `HOLD_START` tokens by remaining time bucket.
- Count of windows ending with non-empty open mask that does not match carry-out.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/inference/test_mapper_v2_1_rollout.py tests/evals/test_mapper_v21_terminal_ln_start_guard_repair.py -q
uv run python -m pulsefield_model.evals.mapper_v21_terminal_ln_start_guard_repair
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_terminal_ln_start_guard_repair_summary.json >/dev/null
```

If the stress gate passes:

```bash
uv run python -m pulsefield_model.evals.mapper_v21_terminal_ln_start_guard_repair --full32
```

## Guard Check

- Default v2.1 rollout behavior is unchanged when `min_ln_duration_ms=None`.
- Synthetic grammar tests show `HOLD_START` is blocked only when `remaining_ms < min_ln_duration_ms` and no matching carry-out start exists.
- Existing v2.1 rollout tests still pass.
- The repair must not use reference beatmap events during generation.
- The repair must not change token IDs, replay semantics, training data, or mapper model weights.

## Qualitative Check

Inspect the final 16 trace steps for any remaining illegal primary case. If case `04` or case `05` still dead-ends at `7990ms`, verify whether the terminal open mask is empty, carry-matched, or still blocked by same-time lane order.

## Positive Signal

- Primary gate: both previously illegal primary runs become legal.
- Stress gate: all four stress cases are legal under baseline + terminal guard and anti-rigid + terminal guard.
- No new starved cases on the stress set.
- Mean F1 on the stress set drops by no more than `0.03` versus the relevant comparator.
- If widened, full 32-case legality becomes `True` with no max-token cases.

## Negative Signal

- Case `04` baseline remains illegal.
- Case `05` anti-rigid remains illegal.
- The guard fixes legality only by starving the second window.
- The guard causes broad F1 collapse or event-count undergeneration.
- Full 32-case widening creates new dead ends or max-token loops.

## Kill Criteria

Kill this repair if either primary illegal run remains illegal, if the stress gate introduces any max-token case, if new stress starvation appears, or if stress mean F1 regresses by more than `0.03` against the relevant comparator.

## Expected Failure Modes

- The model may choose a different terminal lane action that still opens a non-carryable LN at `7990ms`.
- Blocking late `HOLD_START` may push the model into too many taps or too few events.
- The guard may fix case `04` but not the anti-rigid-damaged case `05`.
- A `20ms` minimum may be too weak or too strong; if so, mutate by sweeping `20/30/40ms` only after the primary failure is reproduced.

## Confounders

- The traced failures occur at the first write-window boundary, not necessarily throughout the chart.
- v2.1 has no production cross-window LN carry in this rollout path, so this repair is a guard against that contract rather than a long-term LN modeling solution.
- Anti-rigid path damage in case `05` begins earlier at `1600ms`; the terminal guard may make the path legal while leaving quality worse.
- Greedy decoding can amplify small logit changes.

## Expected Runtime / Runtime Budget

Focused tests should finish in seconds. The primary/stress real-audio gate should finish in under 15 minutes. The optional full 32-case widening may take 30 to 60 minutes. Stop after the primary gate if either known illegal run remains illegal.

## Result Interpretation Plan

- Positive result would suggest: promote the terminal LN-start guard to a wider v2.1 grammar hardening gate, still default-off.
- Negative result would suggest: v2.1 legality needs a deeper terminal-policy or carry-state mutation, and another anti-rigid suppression rule is not justified.
- Ambiguous result would require: one bounded sweep over `min_ln_duration_ms=20/30/40`, not a new grammar rewrite.
- Human owner decides: whether a default-off v2.1 repair is worth widening versus returning to v3 structural grammar.
- Next-loop action if positive: run full 32-case fixed-slice widening and compare against v3.
- Next-loop action if negative: kill this repair family and return to v3 structural grammar/card selection.
- Next-loop action if ambiguous: run the small duration sweep only on the four stress cases.

## Result Log Template

- Experiment: mapper v2.1 terminal LN-start guard repair
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
- Remaining ambiguity: whether `20ms` is the right minimum; do not sweep until the selected value is tested on the primary/stress gate.

## Next-Loop Action

- If positive: execute the optional full 32-case widening gate and compare against v3.
- If negative: stop this repair and return to v3 structural grammar selection.
- If ambiguous: run a small `20/30/40ms` stress-only sweep.

## Novelty Notes

- Closest analogies: minimum-duration constrained decoding, terminal-state validity guard, finite-state decode hardening.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering hardening of v2.1 grammar only.
