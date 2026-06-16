# Target Grammar v3 Multicase Timing-Quality Result Report

## Scope

This pass audits the 200-step v3 checkpoint across multiple real-audio cases and prefix lengths. It does not change tokenizer semantics, model architecture, loss, or decode policy.

## Experiment Card

- Source card: `target_grammar_v3_multicase_timing_quality_experiment_card.md`
- Selected variant: three fixed-slice audio/map cases, prefixes `8000` and `16000` ms, alpha `0.00`, selected-map normalized difficulty.
- Date: `2026-06-17`

## Result

Decision: `MUTATE`.

- Reason: legal multicase rollouts reveal systematic rigid-grid timing and window-continuation artifacts
- Checkpoint: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/checkpoint.pt`
- Policy: greedy, `temperature=0.0`, `time_shift_length_penalty_alpha=0.00`
- Runs: `6`
- Mean timing F1 @100ms: `0.643`
- Mean dominant-spacing ratio: `0.993`

## Case Results

| Case | Prefix | Norm diff | Gen/Ref events | F1 @100ms | Dominant spacing | Rigid ratio | Boundary | 2nd-window share | Legal | First generated times |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sparse` | `8000` | `-0.455` | `49/28` | `0.701` | `160` | `1.000` | `0.000` | `0.000` | `True` | `[160, 320, 480, 640, 800, 960]` |
| `sparse` | `16000` | `-0.455` | `50/57` | `0.523` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` |
| `moderate` | `8000` | `-0.080` | `39/46` | `0.847` | `200` | `1.000` | `0.000` | `0.000` | `True` | `[200, 400, 600, 800, 1000, 1200]` |
| `moderate` | `16000` | `-0.080` | `79/86` | `0.824` | `200` | `1.000` | `0.013` | `0.494` | `True` | `[200, 400, 600, 800, 1000, 1200]` |
| `dense` | `8000` | `0.325` | `49/64` | `0.602` | `160` | `1.000` | `0.000` | `0.000` | `True` | `[160, 320, 480, 640, 800, 960]` |
| `dense` | `16000` | `0.325` | `50/144` | `0.361` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` |

## What Passed

- All six real-audio rollouts completed legally with no dead-end or max-token failure.
- The checkpoint emits non-empty event streams under selected-map normalized difficulty, so the zero-event problem is no longer the immediate blocker.
- The diagnostic covers sparse, moderate, and dense reference maps from the fixed training slice.

## What Surfaced

- Timing remains dominated by fixed grids: sparse/dense cases produce a `160ms` grid; the moderate case produces a `200ms` grid.
- Rigid-grid ratio is `1.0` for every 8s run and at least `0.9796` for every 16s run.
- Sparse and dense 16s rollouts still starve the second window: second-window event share is only `0.02`.
- Timing F1 is misleadingly high in some cases because dense regular grids land near many reference events; it does not prove music-conditioned timing quality.
- The model reacts to difficulty/audio enough to change grid spacing/count, but not enough to reproduce irregular beatmap timing.

## Verification

- Runtime/v3 diagnostic guard: `13 passed`
- Inference/model guard: `26 passed`
- Summary JSON validation: passed

## Interpretation

This is a `MUTATE` result. v3 remains viable as a representation, but the current free-running mapper is not ready for broader replacement training. The next experiment should target timing calibration/window continuation, or establish a matched v2.1 generated timing baseline before deciding whether the failure is v3-specific.

## Next Step

Create a bounded calibration experiment for rigid-grid timing and second-window starvation, or run a matched v2.1 generated timing baseline on the same six cases.
