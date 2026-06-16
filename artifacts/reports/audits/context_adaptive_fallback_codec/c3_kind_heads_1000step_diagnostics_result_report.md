# C3 Auxiliary-Target Diagnostics Result Report

## Scope

This P14 pass scores the P13 enabled C3 auxiliary checkpoint without additional training. It rebuilds the same eval split, compares model top-K C3 bag-label recovery against a train-unigram baseline, and splits recovery by C3 token kind.

## Result

Decision: KILL.

- completed: `True`
- eval windows: `142`
- positive samples: `139`
- positive labels: `1895`
- model recall@20: `0.074406`
- unigram recall@20: `0.089710`
- relative lift@20: `-0.170588`
- elapsed: `11.94s`

## Top-K Recovery

| K | Model recall | Unigram recall | Relative lift | Model sample hit rate | Model precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.005805 | 0.005805 | 0.000000 | 0.079137 | 0.079137 |
| 3 | 0.012665 | 0.016887 | -0.250000 | 0.158273 | 0.057554 |
| 5 | 0.019525 | 0.030079 | -0.350877 | 0.223022 | 0.053237 |
| 10 | 0.040633 | 0.053826 | -0.245098 | 0.410072 | 0.055396 |
| 20 | 0.074406 | 0.089710 | -0.170588 | 0.633094 | 0.050719 |

## Kind Recovery At Selected K

| Kind | Target labels | Model recall | Unigram recall | Model hits | Unigram hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 854 | 0.084309 | 0.141686 | 72 | 121 |
| REF | 284 | 0.063380 | 0.063380 | 18 | 18 |
| RES | 757 | 0.067371 | 0.040951 | 51 | 31 |

## What Passed

- The checkpoint and exact C3 sidecar were loadable under the P13 config.
- The diagnostics preserve the legality constraint: C3 appears only as target-side labels.
- The report now exposes whether the auxiliary head beats a common-label baseline and which token kinds are recovered.

## What Surfaced

The auxiliary head does not beat the train-unigram comparator at K=20: model hits 141/1895 positive labels while unigram hits 170/1895. The P13 BCE loss decrease therefore did not translate into useful positive-label recovery; it was likely dominated by absent-label calibration. The current full-vocab bag auxiliary head should not be promoted without changing the objective.

## Next Step

Kill or reformulate the full-vocab bag auxiliary target before spending more training runtime.
