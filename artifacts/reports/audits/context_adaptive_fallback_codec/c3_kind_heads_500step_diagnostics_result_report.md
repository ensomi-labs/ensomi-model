# C3 Auxiliary-Target Diagnostics Result Report

## Scope

This P14 pass scores the P13 enabled C3 auxiliary checkpoint without additional training. It rebuilds the same eval split, compares model top-K C3 bag-label recovery against a train-unigram baseline, and splits recovery by C3 token kind.

## Result

Decision: KILL.

- completed: `True`
- eval windows: `142`
- positive samples: `139`
- positive labels: `1895`
- model recall@20: `0.058575`
- unigram recall@20: `0.089710`
- relative lift@20: `-0.347059`
- elapsed: `11.46s`

## Top-K Recovery

| K | Model recall | Unigram recall | Relative lift | Model sample hit rate | Model precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.005277 | 0.005805 | -0.090909 | 0.071942 | 0.071942 |
| 3 | 0.011609 | 0.016887 | -0.312500 | 0.158273 | 0.052758 |
| 5 | 0.014776 | 0.030079 | -0.508772 | 0.187050 | 0.040288 |
| 10 | 0.023747 | 0.053826 | -0.558824 | 0.258993 | 0.032374 |
| 20 | 0.058575 | 0.089710 | -0.347059 | 0.532374 | 0.039928 |

## Kind Recovery At Selected K

| Kind | Target labels | Model recall | Unigram recall | Model hits | Unigram hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 854 | 0.069087 | 0.141686 | 59 | 121 |
| REF | 284 | 0.059859 | 0.063380 | 17 | 18 |
| RES | 757 | 0.046235 | 0.040951 | 35 | 31 |

## What Passed

- The checkpoint and exact C3 sidecar were loadable under the P13 config.
- The diagnostics preserve the legality constraint: C3 appears only as target-side labels.
- The report now exposes whether the auxiliary head beats a common-label baseline and which token kinds are recovered.

## What Surfaced

The auxiliary head does not beat the train-unigram comparator at K=20: model hits 111/1895 positive labels while unigram hits 170/1895. The P13 BCE loss decrease therefore did not translate into useful positive-label recovery; it was likely dominated by absent-label calibration. The current full-vocab bag auxiliary head should not be promoted without changing the objective.

## Next Step

Kill or reformulate the full-vocab bag auxiliary target before spending more training runtime.
