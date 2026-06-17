# Target Grammar v3 Teacher-Forced Time-Shift Logit Audit Result Report

## Scope

This artifact-only audit scores target time-shift rows under teacher forcing using the existing 500-step v3 checkpoint. It does not train, rerun rollout, change tokenizer behavior, or change mapper defaults.

## Decision

Route: `TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC`.

- Reason: teacher-forced time-shift ranks are good enough that free-running state exposure is the likely bottleneck
- Next step: Create a generated-state exposure diagnostic on high-leverage fixed-slice cases.

## Checks

| Check | Passed |
| --- | ---: |
| `checkpoint_exists` | `True` |
| `training_report_exists` | `True` |
| `checkpoint_model_v3` | `True` |
| `training_report_contract_v3` | `True` |
| `dataset_non_empty` | `True` |
| `time_shift_rows_positive` | `True` |
| `rank_metrics_finite` | `True` |
| `faithful_global_context` | `True` |
| `no_training_or_rollout` | `True` |

## Metrics

- eval windows: `56`
- global context disabled for smoke: `False`
- target time-shift rows: `3886`
- recall@1 / @3 / @5: `0.581060` / `0.863870` / `0.923314`
- median / p90 target rank: `1.000000` / `5.000000`
- target time-shift NLL: `1.287371`
- mean target time-shift probability: `0.479082`
- argmax top shift: `100` ms at share `0.319609`
- argmax 200ms share: `0.048121`
- argmax rigid-proxy piece share (`60/100/200`): `0.527535`
- target rigid-proxy piece share (`60/100/200`): `0.502831`
- time-shift vocab has 160ms token: `False`
- JS divergence target-vs-argmax: `0.031412`

## Top Target Shifts

| Shift ms | Count |
| ---: | ---: |
| `100` | `1142` |
| `60` | `657` |
| `80` | `624` |
| `90` | `283` |
| `50` | `281` |
| `70` | `250` |
| `200` | `155` |
| `300` | `138` |
| `40` | `116` |
| `10` | `58` |
| `20` | `58` |
| `30` | `52` |

## Top Predicted Argmax Shifts

| Shift ms | Count |
| ---: | ---: |
| `100` | `1242` |
| `80` | `907` |
| `60` | `621` |
| `50` | `391` |
| `200` | `187` |
| `70` | `183` |
| `90` | `135` |
| `20` | `65` |
| `10` | `43` |
| `300` | `40` |
| `40` | `29` |
| `30` | `24` |

## Poor-Rank Examples

| Beatmap | Window | Step | Current ms | Target | Rank | Top shifts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `0` | `0` | `0` | `30` | `20` | `800, 600, 200, 400, 80` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `0` | `8000` | `10` | `17` | `70, 80, 90, 200, 100` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `40` | `9510` | `300` | `9` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `154` | `14560` | `50` | `7` | `80, 100, 70, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `166` | `14880` | `50` | `7` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `156` | `14610` | `50` | `7` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `0` | `5` | `350` | `200` | `6` | `100, 80, 300, 70, 40` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `70` | `11090` | `70` | `6` | `80, 100, 300, 40, 90` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `170` | `14980` | `50` | `6` | `80, 100, 70, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `168` | `14930` | `50` | `6` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `162` | `14770` | `50` | `6` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `160` | `14720` | `50` | `6` | `80, 100, 70, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `158` | `14660` | `60` | `5` | `80, 70, 100, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `8000` | `164` | `14820` | `60` | `5` | `80, 100, 70, 40, 60` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `0` | `18` | `1400` | `50` | `5` | `10, 20, 60, 70, 50` |
| dataset/0/1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu | `0` | `67` | `4340` | `10` | `4` | `60, 50, 70, 10, 20` |

## Interpretation

The model can rank target shifts under teacher forcing, but generated state still collapses.

## What This Does Not Prove

- It does not prove rollout quality improvement.
- It does not prove v3 replacement readiness.
- It does not evaluate the full 4k dataset.
- It does not prove all timing objectives are bad or good.
