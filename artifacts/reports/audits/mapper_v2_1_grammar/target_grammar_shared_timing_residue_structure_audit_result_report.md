# Target Grammar Shared Timing-Residue Structure Audit Result Report

## Scope

This artifact-only audit measures timing structure in current v2.1 teacher-forcing target streams and in the same streams after v3 event-token conversion. It compares that target timing structure with committed generated-rollout summaries that showed rigid-grid collapse.

It does not change tokenizer semantics, model heads, loss, decode policy, training configs, or inference defaults.

## Result

Decision: `TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION`.

- Reason: teacher-forcing targets preserve richer timing residue than the generated rigid-grid rollouts
- Train windows scored: `4096` / `174373`
- Eval windows scored: `142` / `142`
- v2.1/v3 time-shift parity mismatches: `0`
- v3 reconstruction mismatches: `0`
- Target event-interval effective vocab: `22.690931`
- Target interval `160/200ms` share: `9.65%`
- Target rigid-window share >=0.95: `1.58%`
- v3 generated mean dominant-spacing ratio: `0.993197`
- matched v2.1 generated mean dominant-spacing ratio: `1.000000`
- elapsed: `64.03s`

## Target Timing Metrics

| Metric | Value |
| --- | ---: |
| time-shift entropy bits | `3.274811` |
| time-shift effective vocab | `9.678685` |
| time-shift top1 share | `30.89%` |
| time-shift top3 share | `51.03%` |
| event-interval entropy bits | `4.504044` |
| event-interval effective vocab | `22.690931` |
| event-interval top1 share | `12.89%` |
| event-interval top3 share | `29.92%` |
| event-interval 160ms share | `6.88%` |
| event-interval 200ms share | `2.78%` |
| event-interval <=200ms share | `88.63%` |
| mean window dominant interval ratio | `0.508760` |
| median window dominant interval ratio | `0.485714` |
| rigid-window share >=0.80 | `7.66%` |
| rigid-window share >=0.90 | `2.90%` |
| rigid-window share >=0.95 | `1.58%` |

## Top Event Intervals

| Interval ms | Count |
| ---: | ---: |
| `80` | 29883 |
| `100` | 19980 |
| `70` | 19490 |
| `150` | 18828 |
| `160` | 15944 |
| `170` | 13406 |
| `120` | 12309 |
| `90` | 11867 |
| `110` | 11754 |
| `50` | 11391 |
| `60` | 10740 |
| `200` | 6434 |
| `40` | 5215 |
| `180` | 5099 |

## Top Time-Shift Tokens

| Shift ms | Count |
| ---: | ---: |
| `100` | 109233 |
| `80` | 36650 |
| `50` | 34531 |
| `70` | 34342 |
| `60` | 28636 |
| `200` | 20329 |
| `10` | 17848 |
| `20` | 17668 |
| `90` | 15734 |
| `40` | 12889 |
| `30` | 11411 |
| `300` | 9502 |
| `400` | 2064 |
| `500` | 631 |

## Comparator Snapshot

- v3 multicase route: `MUTATE`; generated systematic rigid grid: `True`.
- decode-policy sweep route: `KILL`.
- matched v2.1 route: `MUTATE`; generated systematic rigid grid: `True`.
- generated-state diagnostic route: `TEST_DECODE_TIMING_CALIBRATION`.
- v3 full-dataset route: `TEST_NEXT`.
- C3 target-complexity route: `MUTATE_TO_V3_GRAMMAR_REPAIR`.
- factorized event route: `KILL_FACTORIZE_EVENT_SIGNATURE`.

## What Passed

- v2.1/v3 time-shift parity: `True`.
- v3 reconstruction exact on audited slice: `True`.
- Generated rigid-grid evidence available: `True`.
- Target timing residue is rich enough for calibration follow-up: `True`.

## What Surfaced

The target streams already contain timing diversity, and v3 preserves the same time-shift sequence as v2.1. The generated rigid grids therefore look more like calibration/exposure failure than an event-token representation failure.

This audit cannot approve v3 replacement. It only decides whether the next small experiment should target timing calibration/embedding or mutate the timing grammar itself.

## Next Step

Create a bounded timing loss or timing-embedding calibration card with rigid-grid and second-window guards.
