# Target Grammar v3 500-Step Training Horizon Gate Result Report

## Scope

This pass trains a fresh 500-step v3 checkpoint on the same fixed slice as the 200-step gate and reruns the six real-audio greedy timing cases. It does not change tokenizer, grammar, model, loss, or decode policy.

## Experiment Card

- Source card: `target_grammar_v3_500step_training_horizon_gate_experiment_card.md`
- Selected variant: fresh 500-step v3 training horizon gate.
- Comparator: 200-step v3 multicase audit and matched 200-step v2.1 baseline.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- Reason: 500-step horizon improved continuation and reduced rigid-grid collapse enough for a wider v3 audit
- Checkpoint: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`
- Final eval total loss: `2.026466`
- Final eval token loss: `1.940945`
- Runs: `6`
- All legal: `True`
- Mean timing F1 @100ms: `0.762` vs v3-200 `0.643` vs v2.1 `0.745`
- Mean dominant-spacing ratio: `0.733` vs v3-200 `0.993` vs v2.1 `1.000`
- Sparse/dense 16s min second-window share: `0.353` vs v3-200 `0.020` vs v2.1 `0.490`
- Positive signal passed: `True`

## Case Results

| Case | Prefix | Norm diff | Gen/Ref events | F1 @100ms | Dominant spacing | Rigid ratio | Boundary | 2nd-window share | Legal | First generated times |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sparse` | `8000` | `-0.455` | `73/28` | `0.515` | `100` | `0.986` | `0.014` | `0.000` | `True` | `[410, 820, 920, 1020, 1120, 1220]` |
| `sparse` | `16000` | `-0.455` | `123/57` | `0.589` | `100` | `0.582` | `0.008` | `0.407` | `True` | `[410, 820, 920, 1020, 1120, 1220]` |
| `moderate` | `8000` | `-0.080` | `48/46` | `0.979` | `150` | `1.000` | `0.000` | `0.000` | `True` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `moderate` | `16000` | `-0.080` | `102/86` | `0.894` | `150` | `0.980` | `0.020` | `0.529` | `True` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `dense` | `8000` | `0.325` | `97/64` | `0.795` | `80` | `0.500` | `0.021` | `0.000` | `True` | `[820, 900, 970, 1050, 1120, 1200]` |
| `dense` | `16000` | `0.325` | `150/144` | `0.803` | `150` | `0.349` | `0.000` | `0.353` | `True` | `[820, 900, 970, 1050, 1120, 1200]` |

## Aggregate Comparison

| Metric | v3 500-step | v3 200-step | matched v2.1 |
| --- | ---: | ---: | ---: |
| Mean F1 @100ms | `0.762` | `0.643` | `0.745` |
| Mean event-count ratio | `1.592` | `0.918` | `1.202` |
| Mean dominant-spacing ratio | `0.733` | `0.993` | `1.000` |
| Sparse/dense min 16s second-window share | `0.353` | `0.020` | `0.490` |
| Max boundary-event ratio | `0.021` | `0.020` | `0.000` |

## What Passed

- Training completed 500 steps with finite metrics and a valid checkpoint.
- All six greedy real-audio rollouts completed legally with no dead-end or max-token failure.
- Sparse/dense 16s second-window starvation was repaired relative to the 200-step v3 baseline.

## What Surfaced

- Teacher-forced eval loss improved modestly versus 200 steps: token loss `1.940945` vs `2.031527`, total loss `2.026466` vs `2.116529`.
- The checkpoint emits many more events, and several individual cases still show obvious regular spacing patterns.
- The positive-signal gate passed: aggregate dominant-spacing ratio dropped below `0.90`, sparse/dense continuation recovered, and legality held.
- More horizon is still a live v3 lever, but this result is not replacement approval because regular-grid and overproduction risks remain.

## Interpretation

This is a TEST_NEXT result for a wider v3 audit, not replacement approval. More training horizon repaired v3 window continuation and reduced aggregate grid collapse, but case-level regular grids and event overproduction remain quality risks.

## Verification

- Runtime/v3 diagnostic guard: `13 passed`.
- Training-comparison guard: `18 passed`.
- v3 rollout/model guard: `9 passed`.
- Summary JSON validation: passed.

## Next Step

Run a wider v3 horizon audit on more fixed-slice or full-dataset cases before adding loss machinery, carrying forward the same continuation, rigidity, boundary, and event-count guards.
