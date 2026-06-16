# Target Grammar v3 Zero-Event Decode Diagnostic Experiment Card

## Hypothesis

The real-audio v3 rollout produced zero event timepoints because greedy online decoding preferred legal time-shift tokens over event tokens, not because the v3 grammar/runtime made event tokens unreachable. A bounded decode-policy/logit diagnostic should identify whether the next step is longer training, decode calibration, loss calibration, or grammar repair.

## Root Objective

Harden the v3 real-audio result before escalating toward full-pipeline replacement by explaining the zero-event rollout failure mode on the current trained checkpoint.

## Goal Decomposition

- Verify event-token reachability under the current online grammar mask.
- Measure whether model logits rank event tokens near the selected time-shift tokens.
- Test whether a small existing time-shift length penalty changes event emission without legality failure.
- Preserve the real-audio session-runtime path and avoid changing training/model semantics in this pass.

## Candidate Variants

- Variant A: expose existing v3 time-shift penalty knobs in the trained runtime rollout eval, add compact logit-category diagnostics, and sweep a few penalty values on the same 16s real-audio prefix.
- Variant B: run longer v3 training immediately and repeat rollout.
- Variant C: change the loss/event weighting to force event tokens earlier.
- Variant D: change the v3 grammar to require at least one event before completion.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | Existing checkpoint, same real audio, penalty/logit sweep | Runs complete; reports event-token reachability/ranks and event counts by policy | Instrumentation breaks runtime or cannot distinguish event reachability from preference |
| B | Longer checkpoint | Non-empty rollout after more training | Still zero events, but without knowing whether policy/logits caused it |
| C | Loss tweak | Higher event probability under teacher forcing | Requires new training semantics before proving decode bottleneck |
| D | Grammar tweak | Non-empty forced output | May create musically invalid events and hides model preference failure |

## Selected Variant

Variant A.

## Selection Pressure

Variant A gives the fastest negative or positive evidence and changes only evaluation observability plus already-supported rollout policy parameters. It avoids spending training time before determining whether the current checkpoint already assigns meaningful mass to events.

## Minimal Change

- Add v3 trained rollout eval arguments for `time_shift_length_penalty_alpha` and `time_shift_delta_penalty_alpha`.
- Add optional logit diagnostics that categorize masked greedy argmax and top-k tokens as `event`, `time_shift`, `eos`, or `other`.
- Record best-event rank/margin summary, event-valid step counts, and emitted event count in JSON/report output.
- Do not change model architecture, training loss, tokenizer, grammar validity, or default replacement behavior.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_zero_event_decode_diagnostic_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_zero_event_decode_diagnostic_summary.json`

## Dataset Slice

- Existing checkpoint: `artifacts/tmp/mapper_v3_real_audio_session_rollout/train/run/checkpoint.pt`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Audio: `dataset/0/1000722/audio.mp3`
- Prefix: `0-16000` ms, two 8s windows.

## Baseline / Comparator

The committed real-audio rollout gate with `time_shift_length_penalty_alpha=0.0`, `temperature=0.0`, and `0` generated event timepoints.

## Primary Metric

Generated event timepoint count per penalty setting.

## Secondary Metric

- Event-token valid step ratio.
- Greedy argmax category counts.
- Event top-1/top-5 counts.
- Best-event rank and best-event logit margin against masked argmax.
- Completion/dead-end/max-token flags.

## Verify Command Or Evaluation Procedure

Run the trained runtime rollout eval on the same real-audio prefix for a small penalty sweep, at minimum `0.0`, `0.05`, `0.10`, and `0.20`, with logit diagnostics enabled.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py -q`
- `uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q`

## Qualitative Check

Inspect the generated event count and logit-category report. A non-empty output caused only by aggressive penalties is diagnostic, not a quality pass.

## Positive Signal

At least one penalty setting emits event timepoints without dead-end/max-token failure, or the baseline logit diagnostics show event tokens frequently in top-k with small negative margin.

## Negative Signal

Event tokens are valid but rarely top-k and have large negative margins across the rollout, or penalties still produce zero events or legality failures.

## Kill Criteria

If event tokens are not valid at event-capable steps, stop longer-training escalation and inspect grammar/state construction. If event tokens are valid but consistently far below time shifts, do not replace defaults; treat trained calibration/data/loss as the next bottleneck.

## Expected Failure Modes

- Real-audio BeatThis preparation dominates runtime.
- Aggressive time-shift penalty produces dense or implausible events.
- Logit diagnostics add memory overhead if every raw vector is stored; keep only aggregate counters and a small example list.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes on local MPS because no new training is required. Stop after four penalty settings or first runtime/legality failure.

## Confounders

- The checkpoint is only trained for 20 steps.
- One audio prefix cannot prove generalization.
- A time-shift penalty changes decode policy, not representation quality.
- Greedy decoding may understate stochastic event probability.

## Result Interpretation Plan

- If baseline already ranks event tokens near top-k, continue with decode calibration and modest longer training.
- If penalties produce non-empty legal output, treat v3 runtime as viable but greedy policy/calibration as the bottleneck.
- If penalties do not help and event margins are large, prioritize longer training/loss calibration before full-pipeline replacement.
- If event tokens are unavailable under the mask, inspect grammar/replay state before any training escalation.

## Result Log Template

- Command(s):
- Penalty settings:
- Baseline event count:
- Best nonzero event count:
- Completion/dead-end/max-token:
- Event-valid step ratio:
- Top-k event evidence:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

If Variant A confirms a calibration/decode bottleneck, run a bounded longer-training plus decode-policy comparison. If it reveals grammar unreliability, mutate back to grammar/state diagnostics.

## Closest Analogies And Novelty Layer

Closest analogies are constrained autoregressive decoding with length penalties, token-category calibration, and teacher-forced versus free-running exposure diagnostics. There is no novelty claim here; this is an engineering verification layer for the v3 event-group representation.
