# Target Grammar v3 Real-Audio Session Rollout Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 has passed representation, real-config training, and synthetic trained-runtime handoff; the next gate should use real audio and the session runtime path before any default replacement.
- Acceptance source, if any: active goal requires v3 full mapper pipeline from training to inference before replacement.
- Source snapshot / evidence grade: strong representation evidence, moderate real-config training evidence, moderate synthetic runtime evidence, missing real-audio session evidence.

## Hypothesis

A slightly longer real-config v3 checkpoint can run through the real-audio session runtime path with actual packed mel and BeatThis timing, producing a legal online rollout without dead-end or max-token failure.

## Root Objective

Move v3 from synthetic runtime smoke toward real session inference:

- train a bounded v3 real-config checkpoint beyond the prior 3-step smoke,
- load it through `ModelRuntime`,
- prepare real audio through `SessionRuntime.prepare_audio(...)`,
- use actual BeatThis timing and mel extraction,
- run `generate_full_song_rollout_v3(...)` on a capped real-audio prefix,
- record generated token/timepoint behavior without claiming quality.

## Goal Decomposition

- Subgoal 1: Produce a short but nontrivial v3 checkpoint on the existing 256 eligible-window real-config slice.
- Subgoal 2: Extend the trained rollout eval to support real audio instead of synthetic mel/timing.
- Subgoal 3: Run a capped real-audio session rollout with the trained v3 checkpoint.
- Subgoal 4: Preserve runtime, training, model, and inference guards.

## Candidate Variants

- Variant A: Train v3 for 20 MPS steps on the existing 32-beatmap/256-window slice, then run a 16s real-audio session rollout on one selected local audio.
- Variant B: Reuse the prior 3-step v3 checkpoint and run real-audio rollout only.
- Variant C: Run paired v2.1/v3 longer training and export both real-audio rollouts.
- Variant D: Switch the default stream runtime to v3 and test manually.

## Local Verification Matrix

- Variant A: Best balance; adds a slightly stronger checkpoint and tests real-audio session mechanics while staying bounded.
- Variant B: Cheaper, but mostly repeats the prior weak 3-step checkpoint.
- Variant C: More comparable, but too broad for the first real-audio v3 session gate.
- Variant D: Too early because generated quality and longer-checkpoint behavior are not established.

## Selected Variant

- Selected: Variant A.

## Selection Pressure

Variant A is selected because it targets the next missing layer without changing defaults. It can fail fast on training, checkpoint loading, real-audio preparation, BeatThis timing, session window provider, or online generation.

## Minimal Change

- Add a real-audio mode to `pulsefield_model.evals.mapper_v3_trained_runtime_rollout`.
- Run existing v3 training entrypoint for 20 MPS steps using the same real-config/cache-backed 256-window slice.
- Run the eval on a capped 16s real-audio prefix.

No default stream/runtime replacement in this card.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_real_audio_session_rollout_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_real_audio_session_rollout_summary.json`
- temporary ignored outputs under `artifacts/tmp/mapper_v3_real_audio_session_rollout/`

Read-only context:

- `src/pulsefield_model/training/mapper_v3.py`
- `src/pulsefield_model/inference/model_runtime.py`
- `src/pulsefield_model/inference/session_runtime.py`
- `src/pulsefield_model/inference/mapper_v3_rollout.py`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

## Dataset Slice

Training:

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Windows: expected `256` mapper-eligible windows
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

Inference:

- Beatmap/audio: first selected local map from the real-config slice
- Audio: `dataset/0/1000722/audio.mp3` or another short selected audio if needed
- Capped chart/prefix length: `16000` ms

## Baseline / Comparator

- Prior v3 runtime smoke:
  - synthetic 1s audio,
  - 3-step checkpoint,
  - one rollout window,
  - `11` generated v3 tokens,
  - `0` event timepoints,
  - no dead-end/max-token failure.

## Primary Metric

- Real-audio v3 session rollout completes the capped prefix without dead-end or max-token failure.

## Secondary Metric

- 20-step training completes with finite eval loss.
- Runtime detects mapper version `v3`.
- Rollout token count is positive.
- Event timepoint count is recorded.
- Expanded v2.1 token count is recorded.
- Real-audio preparation uses actual mel extraction and BeatThis timing, not synthetic provider mode.

## Verify Command or Evaluation Procedure

1. Train v3 for 20 MPS steps on the existing selected slice.
2. Run real-audio rollout eval:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_trained_runtime_rollout \
  --mapper-checkpoint-path artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt \
  --control-checkpoint-path artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt \
  --summary-output artifacts/tmp/mapper_v3_real_audio_session_rollout/rollout_summary.json \
  --report-output artifacts/tmp/mapper_v3_real_audio_session_rollout/rollout_report.md \
  --device mps \
  --audio-path dataset/0/1000722/audio.mp3 \
  --real-audio \
  --chart-end-ms 16000 \
  --max-tokens-per-window 512 \
  --temperature 0.0
```

3. Run guards:

```bash
uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Existing v3 runtime/checkpoint loader tests pass.
- Existing v2.1/v3 inference/model tests pass.
- Existing v3/v2.1 training/comparison tests pass.
- No default stream config is changed.

## Qualitative Check

Inspect result summary for:

- `real_audio=true`,
- mapper version `v3`,
- finite train/eval loss,
- no rollout dead-end,
- no max-token failure,
- generated tokens and event count,
- whether zero event count persists.

## Positive Signal

- 20-step v3 training completes.
- Real-audio session runtime prepares audio/control successfully.
- Online v3 rollout finishes the capped prefix without legality failure.
- Generated token stream expands to v2.1-equivalent tokens.

## Negative Signal

- Training fails under real config,
- real audio mel/BeatThis preparation fails,
- runtime loading regresses,
- rollout dead-ends,
- rollout hits max-token cap,
- guards fail.

## Kill Criteria

Kill immediate default/runtime replacement if real-audio v3 session rollout cannot execute without legality or max-token failure.

## Expected Failure Modes

- BeatThis runtime may be slow or fail on local device.
- A 20-step checkpoint may still generate low-quality or empty-event output.
- Real-audio session provider may expose field/shape assumptions not covered by synthetic smoke.
- MPS memory pressure could require lowering chart prefix length.

## Expected Runtime / Runtime Budget

- Expected runtime: 10-30 minutes.
- Stop condition: training failure, real-audio preparation failure, rollout legality failure, or guard regression.

## Confounders

- 20 steps is still not enough for quality claims.
- Event count is quality-adjacent but not a sufficient quality metric.
- A successful 16s prefix does not prove full-song replacement.

## Result Interpretation Plan

- Positive: proceed to longer v3 training and a fuller real-song inference comparison.
- Negative training: repair training stability before inference.
- Negative rollout with successful training: isolate runtime provider/generation policy.
- Zero-event output: treat as quality weakness requiring longer training/policy work, not a runtime failure if legality holds.

## Result Log Template

- Experiment: Target grammar v3 real-audio session rollout
- Date:
- Commit / run id:
- Training checkpoint:
- Training steps/loss:
- Audio path:
- Real audio mode:
- Chart prefix:
- Mapper version:
- Generated windows:
- Generated v3 tokens:
- Event timepoints:
- Expanded v2.1 tokens:
- Completed/dead-end/max-token status:
- Guards:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run longer v3 training and real-song v3-vs-v2.1 inference comparison.
- If negative: repair the failing training/runtime/generation layer before replacement.

## Closest Analogies and Novelty Layer

- Closest analogies: v2/v2.1 cached stream inference, v3 synthetic runtime rollout, teacher-forced event-token sequence inference.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering validation of v3 in the real-audio inference pipeline.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: generated quality and full-song/default replacement remain unproven.
