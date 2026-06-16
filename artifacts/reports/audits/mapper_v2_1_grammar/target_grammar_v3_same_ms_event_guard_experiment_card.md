# Target Grammar v3 Same-ms Event Guard Experiment Card

## Hypothesis

The v3 event-budget training gate exposed a grammar/replay legality gap: several high-difficulty rollouts reached `15980-15990ms` and then emitted hundreds of event groups at the same timestamp until the max-token guard fired. Canonical v3 targets already pack all lane actions at one timestamp into one event token, so generation should not need more than one event group at the same `current_ms`. Adding an online replay-state guard that requires a time shift after any event at the current timestamp should prevent same-ms event loops without changing the v3 representation or requiring future target information.

## Root Objective

Move target grammar v3 closer to full-pipeline replacement readiness by hardening online generation legality while preserving the audited representation contract: reversible beatmap-event reconstruction, lower-bit teacher-forcing target than v2.1, no complex cross-window replay, and no future target-derived inference inputs.

## Goal Decomposition

1. Confirm the failure mode is repeated same-ms event emission, not ordinary dense timing.
2. Add the smallest grammar/replay state needed to represent whether an event has already been emitted at the current timestamp.
3. Keep teacher-forced target tokenization unchanged, because v3 already represents same-time chords as a single event group.
4. Verify existing v3 replay/tokenization tests still pass.
5. Run a decode-only gate on the already trained checkpoints, prioritizing the four max-token cases and then the full fixed 32-case manifest if smoke passes.

## Candidate Variants

### A. Replay-State Same-ms Event Guard

Track `event_emitted_at_current_ms` in v3 replay/generation state. Event tokens are invalid when the flag is true; time-shift tokens reset the flag.

### B. Decode-Time Duplicate Suppression Only

Filter event tokens during rollout if the previously emitted token was an event at the same timestamp, without changing replay/grammar state.

### C. Training Loss Penalty For Duplicate Events

Add a loss penalty for event mass after an event at the same timestamp.

### D. Lower Event-Budget Weight

Rerun training with a smaller scalar `lambda_event_budget`.

## Local Verification Matrix

| Variant | Smallest Check | Pass Condition | Fail Condition |
| --- | --- | --- | --- |
| A | Unit replay/rollout tests plus max-token-case decode smoke | Repeated same-ms events become impossible and max-token loops disappear | Existing valid target replay breaks or max-token loop persists |
| B | Rollout-only filter smoke | Stops loop cheaply | Grammar/runtime paths diverge and bug remains in valid-token masks |
| C | Synthetic loss test | Penalizes duplicate mass | Requires training and does not enforce online legality |
| D | 500-step train | Might reduce overpressure | Does not address grammar gap; expensive after two scalar-loss failures |

## Selected Variant

Variant A: replay-state same-ms event guard.

## Selection Pressure

Variant A directly targets the observed failure with a grammar invariant that matches the v3 target representation. It is smaller and more causally aligned than another training run. It also keeps online inference self-contained: the state bit depends only on previously emitted tokens and current replay state.

## Minimal Change

- Extend v3 generation/replay state with `event_emitted_at_current_ms`.
- Initialize the flag as false at window start.
- Set the flag true after an event token.
- Reset the flag to false after a time-shift token.
- Mark event tokens invalid when the flag is true.
- Preserve EOS/time-shift behavior and LN carry semantics.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/v3/replay.py`
- `src/pulsefield_model/models/mapper/shared/grammar.py` or the v3 valid-token mask path, depending on current structure
- `src/pulsefield_model/inference/mapper_v3_rollout.py` if rollout state adapters need the new field
- `tests/models/mapper/v3/test_event_token_smoke.py`
- `tests/inference/test_mapper_v3_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_same_ms_event_guard_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_same_ms_event_guard_summary.json`

## Dataset Slice

Decode/eval only; no training.

Primary smoke:

- Four max-token cases from the event-budget training gate:
  - `23_usao_knight_rider_kuo_kyoka_expert`
  - `25_nekodex_circles_famoss_insane`
  - `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`
  - `27_billiummoto_four_veiled_stars_aries_insane`
- Checkpoint: `artifacts/tmp/mapper_v3_event_budget_training_gate/train/run_after_nan_fix/checkpoint.pt`

Comparator smoke:

- Same four cases on the baseline 500-step checkpoint:
  - `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`

Full gate if smoke passes:

- Fixed 32-case manifest: `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`

## Baseline / Comparator

Event-budget training gate:

- max-token cases: `4`
- repeated same-ms spacing ratio in max-token cases: about `0.91`
- boundary-event ratio in max-token cases: about `0.91`
- route: `MUTATE`

500-step baseline fixed-slice audit:

- max-token cases: `0`
- starved cases: `11`
- rigid cases: `7`
- route: `MUTATE`

## Primary Metric

On the four event-budget max-token cases:

- no case hits max tokens;
- duplicate-or-non-increasing spacing ratio drops below `0.05`;
- boundary-event ratio drops below `0.20`;
- generated event count remains finite and not more than `2.0x` reference count for nonzero-reference cases.

## Secondary Metric

On the full fixed 32-case gate if smoke passes:

- max-token case count;
- second-window starvation count;
- rigid-case count;
- median event-count ratio;
- mean F1 @100ms;
- boundary-event ratio;
- no regression in existing unit/rollout tests.

## Verify Command Or Evaluation Procedure

1. Add the replay/grammar state guard.
2. Run focused v3 replay/rollout tests.
3. Run the four max-token cases with the event-budget checkpoint and aggregate duplicate/boundary metrics.
4. If no max-token failure remains, rerun the fixed 32-case manifest.
5. Validate summary JSON and run regression tests:

```bash
uv run --group dev pytest tests/models/mapper/v3/test_event_token_smoke.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
```

## Guard Check

- Do not change v3 token vocabulary or target tokenization.
- Do not require future target information.
- Do not allow cross-window replay complexity.
- Do not hide dead-end cases; if the guard creates dead ends, that is a real failure.

## Qualitative Check

Inspect max-token case timing previews. A pass requires the terminal repeated `15980/15990ms` event wall to disappear, not merely move to another timestamp.

## Positive Signal

`TEST_NEXT` if the four max-token cases no longer hit max tokens and duplicate/boundary ratios fall without creating dead ends. A stronger positive signal is full 32-case improvement or no regression against the baseline checkpoint.

## Negative Signal

`MUTATE` if the guard creates dead ends, blocks valid teacher-forced replay, or simply converts same-ms loops into rigid time-shift starvation.

## Kill Criteria

Kill this grammar mutation if existing v3 tokenization/replay tests fail in a way that implies valid target sequences require repeated event groups at the same timestamp, or if max-token cases still loop after the guard.

## Expected Failure Modes

- The repeated event wall disappears but the model dead-ends near `write_end_ms`.
- The model replaces same-ms loops with dense 10/20ms event grids.
- Some LN edge case genuinely needs two event tokens at the same timestamp; this would contradict the current canonical v3 target assumption and must be audited.

## Expected Runtime / Runtime Budget

Expected runtime is a few minutes for tests and four real-audio rollouts. Full 32-case rerun is allowed only if the four-case smoke passes.

## Confounders

- This tests generation legality, not training quality.
- Removing duplicate loops may expose the underlying event-count/timing weakness more clearly.
- The event-budget checkpoint is already behaviorally failed; a pass only proves the guard fixes one failure class.

## Result Interpretation Plan

- If smoke passes and full 32-case improves, keep the guard and move to a conditioned/ordered timing objective.
- If smoke passes but full 32-case still fails on starvation/rigidity, keep the guard as legality hardening and mutate training objectives separately.
- If smoke fails through dead ends, inspect EOS/time-shift grammar near `write_end_ms`.

## Result Log Template

```markdown
# Target Grammar v3 Same-ms Event Guard Result Report

## Scope

## Code Change

## Four-Case Smoke

## Full 32-Case Gate

## Passed

## Surfaced

## Decision
```

## Next-loop Action

If the guard passes, preserve it as v3 grammar hardening and run the next mutation against the remaining starvation/rigid-grid failures. If it fails, return to grammar-state design before any new training.

## Closest Analogies And Novelty Layer

Closest analogies are autoregressive grammar masks, finite-state constrained decoding, and duplicate-token suppression in structured sequence generation. This is v3 grammar-engineering hardening, not representation novelty; the representation novelty remains the event-group target grammar.
