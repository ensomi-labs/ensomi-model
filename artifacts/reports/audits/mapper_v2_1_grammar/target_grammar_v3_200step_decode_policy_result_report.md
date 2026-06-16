# Target Grammar v3 200-Step Decode-Policy Result Report

## Scope

This pass tests whether a 10x longer v3 training horizon fixes the zero-event real-audio rollout without changing tokenizer, grammar, architecture, or loss semantics.

## Experiment Card

- Source card: `target_grammar_v3_200step_decode_policy_experiment_card.md`
- Selected variant: 200-step cache-backed v3 training on the fixed 32-song/256-window slice, followed by the same real-audio alpha sweep.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- Reason: 200-step training breaks zero-event greedy collapse, but generated timing is still patterned and not replacement-ready.
- Training checkpoint: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/checkpoint.pt`
- Training report: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/report.json`
- Real audio: `dataset/0/1000722/audio.mp3`
- Prefix: `0-16000` ms

## Training Metrics

| Steps | Eval total loss | Eval token loss |
| ---: | ---: | ---: |
| `20 baseline` | `3.211091` | `3.120049` |
| `50` | `2.575035` | `2.485749` |
| `100` | `2.268819` | `2.183018` |
| `150` | `2.162278` | `2.076264` |
| `200` | `2.116529` | `2.031527` |

## Rollout Sweep

| Alpha | Tokens | Events | Event-valid steps | Event top-1 | Event top-5 | Boundary events | Median event rank | Median event margin | Completed | Timing summary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `0.00` | `285` | `50` | `284/285` | `50` | `185` | `1 (2.0%)` | `2.0` | `-0.552951` | `True` | `first=[160, 320, 480, 640]... last=[7520, 7680, 7840, 15980] spacing={'160': 48, '8140': 1}` |
| `0.05` | `285` | `50` | `284/285` | `50` | `185` | `1 (2.0%)` | `2.0` | `-0.502951` | `True` | `first=[160, 320, 480, 640]... last=[7520, 7680, 7840, 15980] spacing={'160': 48, '8140': 1}` |
| `0.10` | `288` | `53` | `287/288` | `53` | `188` | `4 (7.5%)` | `2.0` | `-0.452058` | `True` | `first=[160, 320, 480, 640]... last=[15980, 15980, 15980, 16000] spacing={'0': 2, '20': 1, '160': 48, '8140': 1}` |
| `0.20` | `288` | `53` | `287/288` | `53` | `188` | `4 (7.5%)` | `2.0` | `-0.352058` | `True` | `first=[160, 320, 480, 640]... last=[15980, 15980, 15980, 16000] spacing={'0': 2, '20': 1, '160': 48, '8140': 1}` |

## Ground-Truth Reference

These are not formal paired inference labels because session inference only conditions on audio/difficulty, but they bound plausible event density and timing for maps in the same beatmap set.

| Beatmap | GT events 0-16s | First events | Last events | Alpha 0 timing F1 @100ms |
| --- | ---: | --- | --- | ---: |
| `Ohara Yuiko - Zero Centimeters (TV Size) (-Mikan) [Asha's HD].osu` | `57` | `[480, 550, 610, 1390, 1780, 2170]` | `[14450, 15220, 15320, 15420, 15810, 16000]` | `0.523` |
| `Ohara Yuiko - Zero Centimeters (TV Size) (-Mikan) [Shini's NM].osu` | `44` | `[610, 1390, 2170, 2950, 3150, 3340]` | `[13860, 14450, 15420, 15610, 15810, 16000]` | `0.511` |

## What Passed

- The 200-step run completed and produced a valid checkpoint/report despite the shell wrapper failing after training on a post-run attribute print.
- Eval token loss improved substantially: `3.120049` at 20 steps to `2.031527` at 200 steps.
- Greedy alpha `0.00` now emits `50` event timepoints instead of `0`.
- All alpha settings completed two online windows without dead-end or max-token failure.
- Event top-1 improved from `0/215` at 20 steps to `50/285` at alpha `0.00`.

## What Surfaced

- Event count improved, but timing quality is poor: alpha `0.00` emits a rigid `160ms` event grid from `160ms` through `7840ms`, then only one late event at `15980ms`.
- The second 8s window is effectively event-starved, so the model has a window-transition/free-running context failure.
- Penalty settings `0.10` and `0.20` add duplicate/boundary events near `15980-16000ms`, so stronger penalty can worsen artifacts.
- Generated count is near reference map counts (`44-57`), but timing F1 at 100ms is only about `0.51-0.52` against the two likely reference maps.

## Verification

- Runtime/v3 diagnostic guard: `13 passed`
- Training/comparison guard: `18 passed`
- Inference/model guard: `26 passed`
- Synthetic regression: `TEST_NEXT`, `7` timepoints
- Summary JSON validation: passed

## Interpretation

Training horizon was a real bottleneck for zero-event collapse. However, the 200-step checkpoint is not chart-quality ready: it learned a crude periodic event prior and still fails the second-window continuation. This keeps v3 on the `TEST_NEXT` path but blocks replacement.

## Next Step

Run a bounded timing-quality diagnostic over multiple real-audio prefixes/maps, or mutate training/loss/decode calibration to reduce rigid-grid and window-transition artifacts before broader v3 replacement work.
