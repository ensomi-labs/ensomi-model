# Target Grammar v3 Real-Audio Session Rollout Result Report

## Scope

This pass verifies a bounded real-audio v3 session-runtime path. It trains a short real-config v3 checkpoint, loads it through runtime, prepares actual MP3 audio with BeatThis timing, and runs a capped online v3 rollout. It is not a generated chart-quality or replacement result.

## Experiment Card

- Source card: `target_grammar_v3_real_audio_session_rollout_experiment_card.md`
- Selected variant: 20-step real-config v3 checkpoint plus 16s real-audio session rollout.
- Date: `2026-06-17`

## Training Result

- Training report: `artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/report.json`
- Checkpoint: `artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt`
- Device: `mps`
- Mapper shape: d384/l4 with global context
- Completed steps: `20`
- Final eval total loss: `3.211091`
- Final eval token loss: `3.120049`
- Final eval valid tokens: `6366`
- Final train total loss: `3.128490`
- Source windows: `256`
- Train windows: `200`
- Eval windows: `56`
- Dropped windows: `0`
- Control-teacher cache required: `true`
- Mapper token contract: `v3_event_groups`

## Real-Audio Rollout Result

Decision: `TEST_NEXT`.

- Audio: `dataset/0/1000722/audio.mp3`
- Audio length: `89465` ms
- Audio frames: `4474`
- Timing provider: `beat-this`
- Timing checkpoint: `final0`
- Timing fit score: `0.893885`
- Full-control cache windows: `12`
- Rollout prefix: `16000` ms
- Runtime mapper version: `v3`
- Runtime-backed rollout windows: `2`
- Generated v3 tokens: `215`
- Expanded v2.1 tokens: `215`
- Generated event timepoints: `0`
- Completed: `true`
- Dead end: `false`
- Max tokens exceeded: `false`

Window details:

| Window | Span | Tokens | Terminal ms | Completed | Dead end | Max tokens |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | `0-8000` | `107` | `8000` | `true` | `false` | `false` |
| 2 | `8000-16000` | `108` | `16000` | `true` | `false` | `false` |

## What Passed

- v3 completed a longer real-config MPS training gate than the prior 3-step checkpoint.
- The trained checkpoint loaded as runtime mapper version `v3`.
- Real audio preparation used actual MP3 input, packed mel extraction, BeatThis timing, and fitted dense timing.
- Session full-control preparation completed for the real audio.
- The runtime-backed v3 online rollout completed two 8s windows.
- No dead-end or max-token failure occurred.
- Generated v3 tokens expanded to v2.1-equivalent tokens without replay/conversion errors.
- Guard tests passed:
  - runtime/v3 rollout guard: `12 passed`
  - inference/model guard: `26 passed`
  - comparison/training guard: `18 passed`

## What Surfaced

- The 20-step checkpoint still generated `0` event timepoints on the 16s real-audio prefix. This is the main quality/undertraining issue surfaced by the gate.
- Runtime mechanics are no longer the immediate blocker; trained generation quality is now the bottleneck.
- This does not justify default/session replacement. It only proves the real-audio session path can execute legally.

## Commands

Rollout eval:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_trained_runtime_rollout \
  --mapper-checkpoint-path artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt \
  --control-checkpoint-path artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt \
  --summary-output artifacts/tmp/mapper_v3_real_audio_session_rollout/real_audio_rollout_summary.json \
  --report-output artifacts/tmp/mapper_v3_real_audio_session_rollout/real_audio_rollout_report.md \
  --device mps \
  --audio-path dataset/0/1000722/audio.mp3 \
  --real-audio \
  --chart-end-ms 16000 \
  --max-tokens-per-window 512 \
  --temperature 0.0
```

Guards:

```bash
uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
```

## Interpretation

Positive for real-audio session-runtime viability: v3 can now be trained, loaded, prepared with real audio/timing, and decoded online through the session path without legality failure.

Negative for quality readiness: the current short-trained checkpoint produces no events on this prefix. The next work should improve trained event generation before any default replacement.

## Next Step

Run longer v3 training and compare real-song v3 inference against v2.1. Do not replace defaults until v3 produces non-empty and musically plausible event streams under real-audio session inference.
