# Target Grammar v3 Zero-Event Decode Diagnostic Result Report

## Scope

This pass explains the zero-event result from the real-audio v3 session rollout. It does not change tokenizer semantics, model architecture, training loss, or replacement defaults.

## Experiment Card

- Source card: `target_grammar_v3_zero_event_decode_diagnostic_experiment_card.md`
- Selected variant: expose existing time-shift penalty knobs and collect compact logit-category diagnostics.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- Reason: event tokens are reachable and small time-shift penalty produces non-empty legal output
- Checkpoint: `artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt`
- Audio: `dataset/0/1000722/audio.mp3`
- Prefix: `0-16000` ms
- Decode: greedy, `temperature=0.0`, top-k diagnostics at `k=5`

## Sweep Results

| Alpha | Tokens | Events | Event-valid steps | Event top-1 | Event top-5 | Argmax kinds | Completed | Dead end | Max tokens | Median event rank | Median event margin | Timepoints |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| `0.00` | `215` | `0` | `214/215` | `0` | `108` | `{'eos': 1, 'time_shift': 214}` | `True` | `False` | `False` | `3.0` | `-1.587000` | `-` |
| `0.05` | `217` | `2` | `216/217` | `2` | `110` | `{'eos': 1, 'event': 2, 'time_shift': 214}` | `True` | `False` | `False` | `3.0` | `-1.325145` | `7950ms:NNTN, 15950ms:NNTN` |
| `0.10` | `217` | `2` | `216/217` | `2` | `110` | `{'eos': 1, 'event': 2, 'time_shift': 214}` | `True` | `False` | `False` | `3.0` | `-1.275145` | `7950ms:NNTN, 15950ms:NNTN` |
| `0.20` | `217` | `2` | `216/217` | `2` | `110` | `{'eos': 1, 'event': 2, 'time_shift': 214}` | `True` | `False` | `False` | `2.5` | `-1.175145` | `7950ms:NNTN, 15950ms:NNTN` |

## What Passed

- The baseline zero-event result reproduced exactly under diagnostics.
- Event tokens were valid on nearly every generated step, so the online grammar/state path is not blocking event emission.
- Baseline event tokens appeared in top-5 on `108/215` steps, meaning the model has some event signal even in the 20-step checkpoint.
- Alpha `0.05`, `0.10`, and `0.20` all completed legally and emitted `2` event timepoints without dead-end or max-token failure.
- Guard tests passed: runtime/v3 diagnostic `13 passed`, inference/model `26 passed`, training/comparison `18 passed`, synthetic regression completed.

## What Surfaced

- Greedy baseline argmax selected time-shift tokens on `214/215` steps and EOS once; event top-1 count was `0`.
- Mild time-shift penalty changes two steps from time-shift to event, proving the zero-event result is decode/calibration sensitive.
- The emitted events are at `7950` ms and `15950` ms, immediately before window boundaries, and both are single-lane lane-3 taps. That is not a musical quality pass.
- Median best-event margin remains negative under all settings, so most steps still prefer time-shifts over events.

## Interpretation

The zero-event rollout is not a v3 representation impossibility. It is a trained-logit calibration/free-running decode issue on an undertrained checkpoint. The v3 path should continue, but replacement remains blocked until longer training and decode/loss calibration produce non-empty, non-boundary-artifact, musically plausible event streams.

## Commands

Sweep command shape:

```bash
for alpha in 0.00 0.05 0.10 0.20; do
  uv run python -m pulsefield_model.evals.mapper_v3_trained_runtime_rollout \
    --mapper-checkpoint-path artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt \
    --control-checkpoint-path artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt \
    --summary-output artifacts/tmp/mapper_v3_zero_event_decode_diagnostic/alpha_${alpha/./_}_summary.json \
    --report-output artifacts/tmp/mapper_v3_zero_event_decode_diagnostic/alpha_${alpha/./_}_report.md \
    --device mps \
    --audio-path dataset/0/1000722/audio.mp3 \
    --real-audio \
    --chart-end-ms 16000 \
    --max-tokens-per-window 512 \
    --temperature 0.0 \
    --time-shift-length-penalty-alpha "$alpha" \
    --collect-logit-diagnostics \
    --logit-top-k 5
done
```

Guards:

```bash
uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Next Step

Run a bounded longer v3 training plus decode-policy comparison. Track event count, event timing distribution, boundary-event ratio, and v2.1-vs-v3 generated density before any default replacement.
