# C3 RAW Context-Bucket Probe Result Report

## Scope

This P23 pass extends the P22 RAW factor-marginal probe with density/chord/hold context buckets. It tests whether weak RAW fields are localized enough to justify context-conditioned RAW factor heads before changing training.

## Result

Decision: MUTATE.

- eval windows: `142`
- RAW positive samples: `121`
- RAW positive labels: `854`
- exact full-vocab RAW recall@20: `0.084309`
- exact RAW-only recall@20: `0.185012`
- max-marginal joint factor coverage@5: `0.553864`
- joint lift vs full exact: `0.469555`
- improved fields@3: `field_3`
- weak fields@3: `field_1, field_2, field_4`
- supported positive context buckets: `1`
- elapsed: `92.21s`

## Field Recall At K=3

| Field | Vocab | Max recall | Logsumexp recall | Unigram recall | Max hits | Unigram hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| field_1 | 15 | 0.745520 | 0.752688 | 0.752688 | 208 | 210 |
| field_2 | 16 | 0.387446 | 0.426407 | 0.439394 | 179 | 203 |
| field_3 | 16 | 0.591892 | 0.575676 | 0.556757 | 219 | 206 |
| field_4 | 16 | 0.578680 | 0.581218 | 0.583756 | 228 | 230 |

## Joint Factor Coverage

| K | Max | Logsumexp | Unigram |
| ---: | ---: | ---: | ---: |
| 1 | 0.008197 | 0.000000 | 0.000000 |
| 3 | 0.201405 | 0.208431 | 0.225995 |
| 5 | 0.553864 | 0.622951 | 0.631148 |

## Context Bucket Recall At K=3

| Feature | Bucket | Field | Targets | Model recall | Unigram recall | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| density_level | high | field_1 | 278 | 0.744604 | 0.751799 | -0.007194 |
| density_level | high | field_2 | 461 | 0.388286 | 0.440347 | -0.052061 |
| density_level | high | field_3 | 369 | 0.590786 | 0.555556 | 0.035230 |
| density_level | high | field_4 | 393 | 0.577608 | 0.582697 | -0.005089 |
| chord_ratio | low | field_1 | 264 | 0.734848 | 0.738636 | -0.003788 |
| chord_ratio | low | field_2 | 428 | 0.392523 | 0.443925 | -0.051402 |
| chord_ratio | low | field_3 | 352 | 0.585227 | 0.548295 | 0.036932 |
| chord_ratio | low | field_4 | 373 | 0.576408 | 0.581769 | -0.005362 |
| chord_ratio | mid | field_2 | 34 | 0.323529 | 0.382353 | -0.058824 |
| chord_ratio | mid | field_4 | 21 | 0.619048 | 0.619048 | 0.000000 |
| hold_occupancy | low | field_1 | 259 | 0.745174 | 0.741313 | 0.003861 |
| hold_occupancy | low | field_2 | 422 | 0.383886 | 0.436019 | -0.052133 |
| hold_occupancy | low | field_3 | 313 | 0.629393 | 0.587859 | 0.041534 |
| hold_occupancy | low | field_4 | 336 | 0.613095 | 0.613095 | 0.000000 |
| hold_occupancy | mid | field_1 | 20 | 0.750000 | 0.900000 | -0.150000 |
| hold_occupancy | mid | field_2 | 40 | 0.425000 | 0.475000 | -0.050000 |
| hold_occupancy | mid | field_3 | 57 | 0.385965 | 0.385965 | 0.000000 |
| hold_occupancy | mid | field_4 | 58 | 0.379310 | 0.413793 | -0.034483 |

## Strongest Context Signals

| Feature | Bucket | Field | Targets | Delta |
| --- | --- | --- | ---: | ---: |
| hold_occupancy | low | field_1 | 259 | 0.003861 |
| hold_occupancy | low | field_4 | 336 | 0.000000 |
| chord_ratio | mid | field_4 | 21 | 0.000000 |
| chord_ratio | low | field_1 | 264 | -0.003788 |
| density_level | high | field_4 | 393 | -0.005089 |
| chord_ratio | low | field_4 | 373 | -0.005362 |
| density_level | high | field_1 | 278 | -0.007194 |
| hold_occupancy | mid | field_4 | 58 | -0.034483 |

## Top Max-Marginal Missed Values At K=3

| Field | Missed values |
| --- | --- |
| field_1 | `8` (21), `18` (12), `3` (10), `24` (7), `16` (6) |
| field_2 | `4` (54), `1` (46), `3` (30), `10` (21), `12` (20) |
| field_3 | `2` (44), `8` (39), `4` (12), `3` (12), `9` (9) |
| field_4 | `1` (45), `4` (27), `2` (22), `9` (14), `3` (14) |

## What Passed

- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.
- RAW factor scoring is post-hoc; no C3 labels are used as model inputs.
- Max, logsumexp, and train-unigram factor comparators are all reported.
- Density/chord/hold buckets are loaded from the same control-v3 target source used for mapper supervision.

## What Surfaced

Joint factor coverage improves over exact token recovery, and one context bucket shows weak-field lift over unigram. The signal is not broad enough for heads yet; refine grouping before training.

## Next Step

Refine RAW context grouping or add an ordered grammar probe.
