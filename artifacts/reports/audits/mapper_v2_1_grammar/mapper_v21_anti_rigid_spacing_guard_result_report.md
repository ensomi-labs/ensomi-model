# Mapper v2.1 Anti-Rigid Spacing Guard Result Report

## Scope

This pass implements the first bounded step from `mapper_v21_anti_rigid_spacing_guard_experiment_card.md`: an opt-in v2.1 generated-state decode transform for repeated timing grids.

It does not change mapper v2.1 tokenization, training data, model weights, default rollout behavior, or inference defaults.

## Decision

Decision: `TEST_NEXT`.

The implementation smoke passed. The next step is a six-case runtime gate on the matched v2.1 timing-baseline cases.

## Implemented

- Added `MapperV21LogitsTransform`, mirroring the existing v3 rollout hook.
- Threaded `logits_transform` through `grammar_constrained_window_generation_v2_1`, `generate_full_song_rollout_v2_1`, and `run_trained_v21_runtime_rollout_smoke`.
- Added `MapperV21AntiRigidSpacingLogitsTransform`, disabled unless explicitly passed.
- The transform observes only already generated tokens and current replay state.
- The transform suppresses the first canonical time-shift piece after a repeated event-spacing run.

Important representation detail:

- v2.1 has no literal `TS_160` token.
- A 160ms event grid is represented by canonical decomposition, such as `TS_100 + TS_60`.
- Therefore the guard blocks `TS_100` after a repeated 160ms event-spacing run, provided another time-shift token is valid.

## What Passed

- Default v2.1 deterministic generation fixture remains unchanged.
- Explicit `logits_transform` can change v2.1 generation only when provided.
- Anti-rigid transform blocks `TS_100` after five generated events with four repeated 160ms spacings.
- The no-alternative guard keeps `TS_100` valid when no alternate time-shift token is available.
- Focused rollout tests pass.

## What Surfaced

This is an implementation smoke, not a quality result. It proves the online generated-state hook and deterministic repeated-spacing detection work. It does not yet prove reduced rigidity on real audio.

The main risk remains that hard-blocking a canonical time-shift piece may push the model into unnatural chord extension or a different decomposition of the same spacing. The runtime gate must measure that directly.

## Verification

Command:

```bash
uv run --group dev pytest tests/inference/test_mapper_v2_1_rollout.py -q
```

Observed:

- `10 passed in 0.64s`

Broader guard:

```bash
uv run --group dev pytest tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

Observed:

- `29 passed in 0.80s`

## Next Step

Run a six-case runtime gate using the matched v2.1 200-step checkpoint:

- baseline artifact: `target_grammar_v3_matched_v21_timing_baseline_summary.json`
- checkpoint: `artifacts/tmp/mapper_v21_200step_timing_baseline/train/run/checkpoint.pt`
- target cases: the same sparse/moderate/dense 8s and 16s cases
- primary signal: dominant-spacing ratio decreases on at least two cases
- guard: no dead ends, no max-token cases, no large F1 regression, and no new second-window starvation

If hard-block fails the runtime gate, mutate to a finite penalty transform rather than scaling this as a grammar default.
