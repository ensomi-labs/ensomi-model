# C3 RAW Factor-Marginal Probe Result Report

## Scope

This P22 pass tests whether P19's RAW logits contain recoverable field-level structure even when exact RAW token top-20 recovery trails unigram.

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
- elapsed: `11.95s`

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

## What Surfaced

Joint factor coverage improves over exact token recovery, but too few fields beat unigram at K=3. Factorization has signal, but the field design or metric needs tightening before model changes.

## Next Step

Refine RAW factor grouping or repeat with density/chord buckets before adding heads.
