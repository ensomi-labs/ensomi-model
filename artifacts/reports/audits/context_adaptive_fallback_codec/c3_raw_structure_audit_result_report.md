# C3 RAW-Structure Audit Result Report

## Scope

This P20 pass audits the P19 reduced kind-head checkpoint without retraining. It focuses only on RAW C3 labels because P19 tied unigram on REF, beat unigram on RES, and left RAW as the main blocker.

## Result

Decision: MUTATE.

- eval windows: `142`
- RAW positive samples: `121`
- RAW positive labels: `854`
- RAW vocab size: `512`
- model RAW recall@20: `0.084309`
- unigram RAW recall@20: `0.141686`
- model minus unigram RAW recall@20: `-0.057377`
- model RAW-only recall@20: `0.185012`
- unigram RAW-only recall@20: `0.201405`
- model RAW recall@50: `0.200234`
- unigram RAW recall@50: `0.208431`
- model RAW recall@100: `0.306792`
- rank-near rate ranks 21-200: `0.324356`
- elapsed: `11.44s`

## Key Findings

- RAW-only top-20 gap is `-0.016393`, versus full-vocab top-20 gap `-0.057377`. This points to ranking/calibration pressure across concatenated kind heads, not only RAW separability.
- Full-vocab K=50 nearly closes the RAW gap: model `0.200234` versus unigram `0.208431`.
- Rank-near targets are substantial: `0.324356` of RAW labels are between full-vocab ranks 21 and 200.
- The reduced eval RAW targets are not tail/composite-heavy: `has_tail=no` covers `854` / `854` RAW labels.
- Syntactic `field_2` underperforming buckets: `12` (-0.250000), `3` (-0.236842), `4` (-0.168539), `2` (-0.144330).
- Syntactic `field_2` overperforming buckets: `10` (0.208333), `6` (0.173913), `8` (0.055556).

## Rank Depth

| Bucket | Full-vocab count | Full-vocab rate | RAW-only count | RAW-only rate |
| --- | ---: | ---: | ---: | ---: |
| 1to20 | 72 | 0.084309 | 158 | 0.185012 |
| 21to50 | 99 | 0.115925 | 114 | 0.133489 |
| 51to100 | 91 | 0.106557 | 112 | 0.131148 |
| 101to200 | 87 | 0.101874 | 183 | 0.214286 |
| over200 | 505 | 0.591335 | 287 | 0.336066 |

## Key Buckets

| Bucket type | Bucket | Targets | Model hit@20 | Unigram hit@20 | Model recall | Unigram recall | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| field_1 | 12 | 491 | 54 | 92 | 0.109980 | 0.187373 | -0.077393 |
| field_1 | 6 | 168 | 17 | 29 | 0.101190 | 0.172619 | -0.071429 |
| field_1 | 8 | 33 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 3 | 19 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 18 | 17 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 16 | 6 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 4 | 6 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 48 | 6 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 36 | 4 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 0 | 2 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 9 | 2 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 10 | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 2 | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_1 | 24 | 98 | 1 | 0 | 0.010204 | 0.000000 | 0.010204 |
| field_2 | 12 | 28 | 1 | 8 | 0.035714 | 0.285714 | -0.250000 |
| field_2 | 3 | 38 | 0 | 9 | 0.000000 | 0.236842 | -0.236842 |
| field_2 | 4 | 89 | 10 | 25 | 0.112360 | 0.280899 | -0.168539 |
| field_2 | 2 | 97 | 10 | 24 | 0.103093 | 0.247423 | -0.144330 |
| field_2 | 1 | 82 | 2 | 8 | 0.024390 | 0.097561 | -0.073171 |
| field_2 | 0 | 322 | 18 | 30 | 0.055901 | 0.093168 | -0.037267 |
| field_2 | 5 | 22 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 11 | 12 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 9 | 12 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 13 | 8 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 7 | 4 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 14 | 3 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| field_2 | 8 | 90 | 22 | 17 | 0.244444 | 0.188889 | 0.055556 |
| field_2 | 6 | 23 | 4 | 0 | 0.173913 | 0.000000 | 0.173913 |
| field_2 | 10 | 24 | 5 | 0 | 0.208333 | 0.000000 | 0.208333 |
| has_tail | no | 854 | 72 | 121 | 0.084309 | 0.141686 | -0.057377 |

## Top RAW Misses

| Token | Target count | Train count | Model hits@20 | Fields | Tail |
| --- | ---: | ---: | ---: | --- | --- |
| `RAW|D:12:8:0:0` | 17 | 13689 | 15 | `12,8,0,0` | `.` |
| `RAW|D:6:4:0:0` | 16 | 10715 | 5 | `6,4,0,0` | `.` |
| `RAW|D:6:2:0:0` | 13 | 10580 | 5 | `6,2,0,0` | `.` |
| `RAW|D:12:2:0:0` | 11 | 17362 | 5 | `12,2,0,0` | `.` |
| `RAW|D:12:10:0:0` | 11 | 5304 | 5 | `12,10,0,0` | `.` |
| `RAW|D:12:0:4:0` | 10 | 10322 | 8 | `12,0,4,0` | `.` |
| `RAW|D:12:0:2:0` | 10 | 10273 | 1 | `12,0,2,0` | `.` |
| `RAW|D:12:4:0:0` | 9 | 16991 | 5 | `12,4,0,0` | `.` |
| `RAW|D:12:3:0:0` | 9 | 8539 | 0 | `12,3,0,0` | `.` |
| `RAW|D:6:8:0:0` | 9 | 7635 | 6 | `6,8,0,0` | `.` |
| `RAW|D:12:5:0:0` | 9 | 5344 | 0 | `12,5,0,0` | `.` |
| `RAW|D:12:0:4:8` | 9 | 3548 | 0 | `12,0,4,8` | `.` |
| `RAW|D:12:1:0:0` | 8 | 13542 | 2 | `12,1,0,0` | `.` |
| `RAW|D:12:12:0:0` | 8 | 8534 | 1 | `12,12,0,0` | `.` |
| `RAW|D:6:1:0:0` | 8 | 7499 | 0 | `6,1,0,0` | `.` |
| `RAW|D:6:0:0:4` | 8 | 5054 | 0 | `6,0,0,4` | `.` |
| `RAW|D:12:0:4:2` | 8 | 4437 | 0 | `12,0,4,2` | `.` |
| `RAW|D:12:0:0:8` | 8 | 4355 | 0 | `12,0,0,8` | `.` |
| `RAW|D:12:0:8:2` | 8 | 3871 | 0 | `12,0,8,2` | `.` |
| `RAW|D:12:0:8:1` | 8 | 3735 | 0 | `12,0,8,1` | `.` |

## Top Unigram-Only RAW Targets

| Token | Target count | Train count | Model hits@20 | Unigram hits@20 | Fields | Tail |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `RAW|D:12:8:0:0` | 17 | 13689 | 15 | 17 | `12,8,0,0` | `.` |
| `RAW|D:6:4:0:0` | 16 | 10715 | 5 | 16 | `6,4,0,0` | `.` |
| `RAW|D:6:2:0:0` | 13 | 10580 | 5 | 13 | `6,2,0,0` | `.` |
| `RAW|D:12:2:0:0` | 11 | 17362 | 5 | 11 | `12,2,0,0` | `.` |
| `RAW|D:12:0:4:0` | 10 | 10322 | 8 | 10 | `12,0,4,0` | `.` |
| `RAW|D:12:0:2:0` | 10 | 10273 | 1 | 10 | `12,0,2,0` | `.` |
| `RAW|D:12:4:0:0` | 9 | 16991 | 5 | 9 | `12,4,0,0` | `.` |
| `RAW|D:12:3:0:0` | 9 | 8539 | 0 | 9 | `12,3,0,0` | `.` |
| `RAW|D:12:1:0:0` | 8 | 13542 | 2 | 8 | `12,1,0,0` | `.` |
| `RAW|D:12:12:0:0` | 8 | 8534 | 1 | 8 | `12,12,0,0` | `.` |
| `RAW|D:12:0:8:0` | 4 | 8444 | 2 | 4 | `12,0,8,0` | `.` |

## What Passed

- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.
- The audit preserves the C3 legality constraint: labels are target-side only.
- RAW diagnostics now expose frequency, parsed-field, tail/composite, and rank-depth structure.

## What Surfaced

RAW still trails unigram at K=20, but a meaningful share of missed labels appears between ranks 21 and 200. The next step should use the bucket tables to choose a grounded RAW split or calibration probe.

## Next Step

Test a smaller RAW split or calibration mutation, because many misses are rank-near by K=200.
