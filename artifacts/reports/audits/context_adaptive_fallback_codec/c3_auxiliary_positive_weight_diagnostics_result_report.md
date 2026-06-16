# C3 Auxiliary-Target Diagnostics Result Report

## Scope

This P14 pass scores the P13 enabled C3 auxiliary checkpoint without additional training. It rebuilds the same eval split, compares model top-K C3 bag-label recovery against a train-unigram baseline, and splits recovery by C3 token kind.

## Result

Decision: KILL.

- completed: `True`
- eval windows: `142`
- positive samples: `141`
- positive labels: `2293`
- model recall@20: `0.000872`
- unigram recall@20: `0.074139`
- relative lift@20: `-0.988235`
- elapsed: `13.41s`

## Top-K Recovery

| K | Model recall | Unigram recall | Relative lift | Model sample hit rate | Model precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000000 | 0.004797 | -1.000000 | 0.000000 | 0.000000 |
| 3 | 0.000000 | 0.013956 | -1.000000 | 0.000000 | 0.000000 |
| 5 | 0.000000 | 0.024858 | -1.000000 | 0.000000 | 0.000000 |
| 10 | 0.000436 | 0.044483 | -0.990196 | 0.007092 | 0.000709 |
| 20 | 0.000872 | 0.074139 | -0.988235 | 0.014184 | 0.000709 |

## Kind Recovery At Selected K

| Kind | Target labels | Model recall | Unigram recall | Model hits | Unigram hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 1065 | 0.000939 | 0.113615 | 1 | 121 |
| REF | 332 | 0.003012 | 0.054217 | 1 | 18 |
| RES | 896 | 0.000000 | 0.034598 | 0 | 31 |

## What Passed

- The checkpoint and exact C3 sidecar were loadable under the P13 config.
- The diagnostics preserve the legality constraint: C3 appears only as target-side labels.
- The report now exposes whether the auxiliary head beats a common-label baseline and which token kinds are recovered.

## What Surfaced

The auxiliary head does not beat the train-unigram comparator at K=20: model hits 2/2293 positive labels while unigram hits 170/2293. The P13 BCE loss decrease therefore did not translate into useful positive-label recovery; it was likely dominated by absent-label calibration. The current full-vocab bag auxiliary head should not be promoted without changing the objective.

## Next Step

Kill or reformulate the full-vocab bag auxiliary target before spending more training runtime.
