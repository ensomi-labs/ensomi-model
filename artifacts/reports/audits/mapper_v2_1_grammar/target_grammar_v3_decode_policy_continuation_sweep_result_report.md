# Target Grammar v3 Decode-Policy Continuation Sweep Result Report

## Scope

This pass tests existing v3 online decode knobs on the 200-step checkpoint. It does not change tokenizer, grammar, model, loss, or training.

## Experiment Card

- Source card: `target_grammar_v3_decode_policy_continuation_sweep_experiment_card.md`
- Selected variant: deterministic time-shift penalty sweep plus one stochastic probe.
- Cases: sparse/moderate/dense real-audio `16000ms` prefixes from the multicase audit.
- Date: `2026-06-17`

## Result

Decision: `KILL`.

- Reason: no tested decode policy reduced rigid-grid collapse while preserving legality and continuation guards
- Checkpoint: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/checkpoint.pt`
- Policies: `5`
- Rollouts: `15`
- Best policy by guard ordering: `sample_t0_80_p0_95`
- Any policy passed positive signal: `False`

## Policy Results

| Policy | Legal | Max-token fails | Mean F1 | Mean event ratio | Mean rigid ratio | Sparse/dense min 2nd share | Max boundary | Duplicate/noninc spacing | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sample_t0_80_p0_95` | `True` | `0` | `0.712` | `1.434` | `0.635` | `0.525` | `0.232` | `0.136` | `False` |
| `delta_0_50` | `True` | `0` | `0.569` | `0.714` | `0.986` | `0.020` | `0.020` | `0.000` | `False` |
| `flat_0_05` | `True` | `0` | `0.569` | `0.714` | `0.986` | `0.020` | `0.020` | `0.000` | `False` |
| `greedy_base` | `True` | `0` | `0.569` | `0.714` | `0.986` | `0.020` | `0.020` | `0.000` | `False` |
| `flat_0_05_delta_0_50` | `False` | `1` | `0.491` | `1.589` | `0.955` | `0.020` | `0.886` | `0.295` | `False` |

## Case Results

| Policy | Case | Gen/Ref | F1 @100ms | Dominant spacing | Rigid ratio | 2nd-window share | Boundary | Legal | First generated times | Last generated times |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `delta_0_50` | `dense` | `50/144` | `0.361` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `delta_0_50` | `moderate` | `79/86` | `0.824` | `200` | `1.000` | `0.494` | `0.000` | `True` | `[200, 400, 600, 800, 1000, 1200]` | `[14800, 15000, 15200, 15400, 15600, 15800]` |
| `delta_0_50` | `sparse` | `50/57` | `0.523` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `flat_0_05` | `dense` | `50/144` | `0.361` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `flat_0_05` | `moderate` | `79/86` | `0.824` | `200` | `1.000` | `0.494` | `0.000` | `True` | `[200, 400, 600, 800, 1000, 1200]` | `[14800, 15000, 15200, 15400, 15600, 15800]` |
| `flat_0_05` | `sparse` | `50/57` | `0.523` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `flat_0_05_delta_0_50` | `dense` | `428/144` | `0.126` | `0` | `0.885` | `0.886` | `0.886` | `False` | `[160, 320, 480, 640, 800, 960]` | `[15980, 15980, 15980, 15980, 15980, 15980]` |
| `flat_0_05_delta_0_50` | `moderate` | `79/86` | `0.824` | `200` | `1.000` | `0.494` | `0.000` | `True` | `[200, 400, 600, 800, 1000, 1200]` | `[14800, 15000, 15200, 15400, 15600, 15800]` |
| `flat_0_05_delta_0_50` | `sparse` | `50/57` | `0.523` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `greedy_base` | `dense` | `50/144` | `0.361` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `greedy_base` | `moderate` | `79/86` | `0.824` | `200` | `1.000` | `0.494` | `0.000` | `True` | `[200, 400, 600, 800, 1000, 1200]` | `[14800, 15000, 15200, 15400, 15600, 15800]` |
| `greedy_base` | `sparse` | `50/57` | `0.523` | `160` | `0.980` | `0.020` | `0.020` | `True` | `[160, 320, 480, 640, 800, 960]` | `[7200, 7360, 7520, 7680, 7840, 15980]` |
| `sample_t0_80_p0_95` | `dense` | `155/144` | `0.676` | `160` | `0.299` | `0.658` | `0.232` | `True` | `[60, 220, 380, 540, 700, 860]` | `[15990, 15990, 16000, 16000, 16000, 16000]` |
| `sample_t0_80_p0_95` | `moderate` | `125/86` | `0.777` | `160` | `0.766` | `0.392` | `0.000` | `True` | `[60, 220, 380, 540, 700, 860]` | `[15040, 15200, 15360, 15520, 15680, 15840]` |
| `sample_t0_80_p0_95` | `sparse` | `101/57` | `0.684` | `160` | `0.840` | `0.525` | `0.000` | `True` | `[410, 530, 850, 1010, 1170, 1330]` | `[15040, 15200, 15360, 15520, 15680, 15840]` |

## What Passed

- The baseline, flat, delta, and stochastic policies completed legal sparse/moderate/dense 16s rollouts except for the combined-penalty dense case.
- The stochastic probe showed the grammar can emit across the second window under non-greedy selection.
- The aggregation covers the same reference beatmaps and timing proxy as the prior multicase audit.

## What Surfaced

- Deterministic flat and delta penalties did not materially change the greedy timing pattern on the three 16s cases.
- Combining flat and delta penalties made the dense case overflow the token cap with 428 generated timepoints, a legality failure.
- Stochastic sampling reduced rigid-grid dominance and improved sparse/dense continuation, but overproduced events and remains unproven for stability.
- No policy satisfied the positive signal; decode-only calibration is not enough to move v3 toward replacement readiness.

## Interpretation

This is a KILL result for decode-only calibration as the next v3 replacement path. Existing deterministic knobs are too weak until they fail legality, while one stochastic probe changes timing shape without proving stable, music-conditioned quality.

## Verification

- Runtime/v3 diagnostic guard: `13 passed`.
- v3 rollout/model guard: `9 passed`.
- Summary JSON validation: passed.

## Next Step

Create a bounded v3 training/loss calibration card using rigid-grid ratio, sparse/dense second-window share, and max-token legality as guards.
