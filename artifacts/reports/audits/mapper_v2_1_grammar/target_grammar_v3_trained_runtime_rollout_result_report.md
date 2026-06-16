# Target Grammar v3 Trained Runtime Rollout Result Report

## Scope

This pass verifies that a trained v3 checkpoint can cross the inference-runtime boundary and execute the v3 online rollout path. It is a runtime handoff smoke, not a generated chart-quality or convergence result.

## Experiment Card

- Source card: `target_grammar_v3_trained_runtime_rollout_experiment_card.md`
- Selected variant: extend `ModelRuntime` v3 checkpoint detection, then run a bounded synthetic runtime-backed v3 rollout using the real-config v3 checkpoint from the prior gate.
- Date: `2026-06-17`

## Implementation

- `ModelRuntime` now detects v3 checkpoints and constructs:
  - `MapperV3Config`
  - `MapperV3Model`
  - `MapperV3Vocab`
- Embedded `control_encoder.*` checkpoint keys are still filtered for inference loading.
- Existing v2/v2.1 runtime behavior remains unchanged.
- Added eval module: `pulsefield_model.evals.mapper_v3_trained_runtime_rollout`
- Added runtime unit coverage for v3 checkpoint loading.

## Result

Decision: `TEST_NEXT`.

- Mapper checkpoint: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/v3/run/checkpoint.pt`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Device: `mps`
- Runtime mapper version: `v3`
- Runtime mapper type: `MapperV3Model`
- Runtime vocab type: `MapperV3Vocab`
- Filtered embedded control keys: `129`
- Loaded mapper keys after filtering: `157`
- Loaded control keys: `129`
- Chart end: `1000` ms
- Runtime-backed rollout windows: `1`
- Generated v3 tokens: `11`
- Expanded v2.1 tokens: `11`
- Generated event timepoints: `0`
- Completed: `true`
- Dead end: `false`
- Max tokens exceeded: `false`

## What Passed

- v3 checkpoint detection works through `ModelRuntime`.
- v3 runtime loading preserves the inference contract: optimizer state, history, and training RNG state are not read.
- Embedded control-encoder weights are filtered from the mapper checkpoint before strict mapper load.
- The separately loaded control checkpoint still loads strictly.
- `SessionRuntime.prepare_mapper_window(...)` supplied v3-compatible runtime batches with projected control memory, density feature, and global context fields.
- `generate_full_song_rollout_v3(...)` executed online without future target tokens.
- The generated v3 token stream expanded to v2.1-equivalent tokens without replay/conversion errors.
- The bounded rollout terminated without dead-end or max-token failure.

## What Surfaced

- The smoke generated no event timepoints on synthetic 1s audio. This is acceptable for runtime handoff, but it gives no chart-quality signal.
- The checkpoint used here is still only the 3-step real-config checkpoint, so generation quality should not be interpreted.
- Runtime support is now present, but default stream/session replacement is still not justified until a longer v3 checkpoint and fuller real-audio inference comparison pass.

## Commands

Rollout eval:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_trained_runtime_rollout \
  --mapper-checkpoint-path artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/v3/run/checkpoint.pt \
  --control-checkpoint-path artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt \
  --summary-output artifacts/tmp/mapper_v3_trained_runtime_rollout/summary.json \
  --report-output artifacts/tmp/mapper_v3_trained_runtime_rollout/report.md \
  --device mps \
  --chart-end-ms 1000 \
  --max-tokens-per-window 256 \
  --temperature 0.0
```

Guards:

```bash
uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
```

- runtime/v3 rollout guard: `12 passed`
- inference/model guard: `26 passed`
- comparison/training guard: `18 passed`

## Interpretation

Positive for the trained runtime handoff gate: v3 now has an inference-runtime loading path and can execute the existing online rollout code from a trained checkpoint. This removes a concrete blocker between v3 training artifacts and full pipeline inference experiments.

Still not proven: longer trained v3 quality, real-audio generated chart quality, full-song v3 replacement, planner integration, and default session-runtime replacement.

## Next Step

Run a longer real-config v3 checkpoint and a fuller session-runtime inference comparison on real audio. If that passes, begin the controlled default/runtime replacement gate.
