# Mapper v2.1 Anti-Rigid Spacing Guard Experiment Card

## Hypothesis

The matched v2.1 timing baseline and the current v3 rollout gates share a rigid 150/160ms timing-grid failure. A small generated-state decode transform that suppresses the first canonical time-shift piece after repeated identical event spacings can reduce rigid-grid collapse without changing tokenizer, teacher-forcing targets, model weights, or mapper defaults.

## Root Objective

Improve the v2.1 grammar/decode surface after C3 mapper-side and v3 scalar-loss paths showed diminishing returns, while preserving v2.1 legality and default behavior.

## Goal Decomposition

- Detect generated-state runs of repeated event spacings online from already emitted v2.1 tokens.
- Suppress the next continuation of the same spacing only when an alternate valid time-shift exists.
- Keep the transform disabled by default and make default generation parity explicit.
- Run a small deterministic verification before any real-audio rollout gate.

## Candidate Variants

- A: hard-block the first canonical time-shift token for the repeated spacing after four repeated event spacings.
- B: subtract a finite logit penalty from that first canonical time-shift token.
- C: add a training-side anti-grid loss.
- D: leave v2.1 unchanged and continue only with v3 training instrumentation.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| A hard block | deterministic mask/logit unit test blocks the repeated spacing piece and leaves alternatives valid | Fastest way to test generated-state control; if it dead-ends or changes defaults, kill. |
| B finite penalty | same deterministic setup changes rank but not legality | Safer than A but weaker signal; use only if A is too disruptive. |
| C training loss | requires retraining and rollout | Too expensive before an online decode smoke. |
| D no v2.1 change | no new evidence | Not enough movement after C3/v3 diminishing returns. |

## Selected Variant

Variant A: optional hard-block transform, disabled by default.

## Selection Pressure

The selected variant is the smallest generated-state intervention that directly targets the observed rigid-grid symptom. It avoids target leakage, avoids model retraining, and can fail quickly with unit tests before runtime rollouts.

## Minimal Change

Add an optional v2.1 `logits_transform` hook mirroring the existing v3 rollout hook, plus a small `MapperV21AntiRigidSpacingLogitsTransform` helper. Do not change default v2.1 rollout behavior or training configs.

## Files Likely To Change

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `src/pulsefield_model/evals/mapper_v2_1_trained_runtime_rollout.py`
- `tests/inference/test_mapper_v2_1_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_anti_rigid_spacing_guard_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_anti_rigid_spacing_guard_result_report.md`

## Dataset Slice

Initial implementation smoke uses deterministic generated-token examples. If this passes, the next runtime gate should use the six matched v2.1 timing-baseline cases from `target_grammar_v3_matched_v21_timing_baseline_summary.json`.

## Baseline / Comparator

Baseline is unchanged v2.1 greedy runtime decode. The matched v2.1 baseline currently has all six cases legal, mean F1 `0.745`, and dominant-spacing ratio `1.000` in every case.

## Primary Metric

For the deterministic smoke: repeated-spacing continuation token is blocked and at least one alternate time-shift remains valid.

For the follow-up runtime gate: dominant-spacing ratio decreases on at least two of the six matched baseline cases without legality regression.

## Secondary Metric

Second-window event share, timing F1@100ms, event-count ratio, boundary-event ratio, duplicate/non-increasing time ratio, and max-token/dead-end counts.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/inference/test_mapper_v2_1_rollout.py -q
```

Follow-up runtime gate:

Create a separate runtime gate that reuses `run_trained_v21_runtime_rollout_smoke` with the opt-in transform on the six matched v2.1 timing-baseline cases.

## Guard Check

- Default v2.1 generation fixture remains unchanged when the transform is absent.
- Transform never blocks the repeated spacing piece unless another time-shift token is valid.
- Transform uses only already generated tokens and current replay state.
- No target beatmap, reference chart, or future tokens are consulted.

## Qualitative Check

Inspect the blocked spacing and token examples. Confirm that a 160ms event-grid run maps to blocking `TS_100`, because v2.1 represents 160ms through canonical decomposition rather than a literal `TS_160` token.

## Positive Signal

- Unit tests pass.
- The transform blocks the next repeated-spacing time-shift piece after the repeat threshold.
- Default rollout behavior is unchanged.
- Follow-up runtime gate reduces rigid spacing without dead-end, max-token, or large F1 regression.

## Negative Signal

- Any default-generation parity regression.
- Transform blocks when no alternate time-shift exists.
- Runtime gate creates dead ends, max-token cases, boundary duplicates, or second-window starvation.
- Rigid ratio does not improve in the matched v2.1 cases.

## Kill Criteria

Kill the hard-block variant if it changes default behavior, causes any legality failure in the six-case runtime gate, or reduces mean F1 by more than `0.05` while failing to reduce dominant-spacing ratio on at least two cases.

## Expected Failure Modes

- Blocking `TS_100` after a 160ms run may force unnatural chord extension if lane-action logits dominate.
- The model may reconstruct the same spacing through another time-shift decomposition.
- Fixed-grid collapse may be driven by model logits rather than grammar-valid timing choices.
- Reducing rigidity may reduce timing F1 because the rigid baseline matches many reference events accidentally.

## Expected Runtime / Runtime Budget

Unit smoke should finish in under one second. Six-case runtime follow-up should stay under the existing matched-baseline rollout budget. Stop before wider rollout if the deterministic guard or default parity fails.

## Confounders

Timing F1 rewards dense regular grids, so lower rigidity can initially look worse. This experiment should treat F1 as a guard, not the primary positive signal.

## Result Interpretation Plan

- If deterministic smoke fails, kill the transform implementation.
- If deterministic smoke passes but runtime legality fails, mutate to a finite penalty.
- If rigidity decreases without guard regressions, run the six-case runtime gate and then compare with v3 decode timing calibration.
- If rigidity does not move, pivot to training-side v3 instrumentation or a richer v2.1 timing-state grammar.

## Result Log Template

Record command, commit, dirty flag, selected variant, blocked spacing/token examples, default parity result, test results, and whether the next action is runtime gate, mutation, or kill.

## Next-Loop Action

If the implementation smoke passes, run the six-case v2.1 runtime anti-rigid gate. If it fails, do not scale; mutate to finite penalties or return to v3 generated-state instrumentation.

## Closest Analogies And Novelty Layer

Closest analogies are anti-repetition decoding penalties, n-gram blocking, and rhythm-variation constraints in symbolic sequence generation. This is not a novel decoding method. The research layer is whether a tiny online generated-state timing guard can repair the observed osu!mania mapper rigid-grid failure without target leakage or retraining.
