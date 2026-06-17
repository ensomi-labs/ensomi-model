# Target Grammar v3 Factorized Event-Signature Proxy Result Report

## Scope

This artifact-only audit tests whether the current v3 4-lane event token has enough internal structure to justify a future factorized target grammar. It does not change the mapper vocab, tokenizer, model heads, training loop, inference, or replay.

The proxy replaces each current v3 event signature with local factors: lane-count bucket, lane-mask bucket, and active-lane action symbols. Time-shift and special tokens stay unchanged. Each factor sequence is checked for exact reconstruction of the original v3 event signature.

## Result

Decision: `KILL_FACTORIZE_EVENT_SIGNATURE`.

- Reason: factorized event-signature proxy failed the v2.1 total-bit guard
- Train windows scored: `4096` / `174373`
- Eval windows scored: `142` / `142`
- Full-dataset proxy: `False`
- Runtime fallback: `True`
- Reconstruction mismatches: `0`
- v2.1 total bits: `118923.707809`
- current v3 total bits: `112653.448087`
- factorized total bits: `198878.260998`
- factorized vs v2.1 total-bit ratio: `1.672318`
- factorized vs current v3 total-bit ratio: `1.765399`
- event bits/original-event reduction: `-147.06%`
- total token expansion ratio: `2.060300`
- event factor expansion ratio: `3.614794`
- elapsed: `65.12s`

## Metric Table

| Stream | Eval tokens | Vocab | Bits/token | Total bits |
| --- | ---: | ---: | ---: | ---: |
| v2.1 | 28073 | 35 | 4.236231 | 118923.707809 |
| current v3 | 22471 | 278 | 5.013281 | 112653.448087 |
| factorized proxy | 46297 | 45 | 4.295705 | 198878.260998 |
| current v3 events only | 9112 | 255 | 5.072868 | 46223.970218 |
| factorized event factors only | 32938 | 22 | 3.467198 | 114202.557495 |

## Factor Vocabulary

- lane-count factors: `4`
- lane-mask factors: `15`
- action factors: `3`
- total event-factor vocab: `22`
- full factorized stream vocab: `45`

## Top Lane Masks

| Mask factor | Count |
| --- | ---: |
| `EV_MASK_0100` | 1500 |
| `EV_MASK_0010` | 1482 |
| `EV_MASK_0001` | 1095 |
| `EV_MASK_1000` | 1059 |
| `EV_MASK_0011` | 591 |
| `EV_MASK_1100` | 582 |
| `EV_MASK_1001` | 493 |
| `EV_MASK_0110` | 347 |
| `EV_MASK_1101` | 337 |
| `EV_MASK_1010` | 324 |
| `EV_MASK_1011` | 317 |
| `EV_MASK_0101` | 301 |

## Top Event Signatures

| Signature | Count | Non-empty lanes |
| --- | ---: | ---: |
| `.T..` | 1373 | 1 |
| `..T.` | 1360 | 1 |
| `...T` | 1010 | 1 |
| `T...` | 962 | 1 |
| `..TT` | 427 | 2 |
| `TT..` | 425 | 2 |
| `T..T` | 362 | 2 |
| `T.T.` | 193 | 2 |
| `.TT.` | 192 | 2 |
| `.T.T` | 179 | 2 |
| `TT.T` | 121 | 3 |
| `T.TT` | 106 | 3 |

## What Passed

- Exact factor reconstruction: `True`.
- Exact current v3 reconstruction back to v2.1: `True`.
- Online-local factors only: `True`.
- No cross-window side stream: `True`.
- Factorized total bits below v2.1: `False`.
- Event bits/original-event reduction is meaningful: `False`.

## What Surfaced

Exact reconstruction passed, but the factorized stream gave back the current v3 representation advantage and scored worse than v2.1 under the unigram bit proxy. This is not a good target grammar replacement candidate from the current evidence.

In this run the factorized event stream is worse even after normalizing back to the original event count: factor tokens have a smaller vocabulary, but the count/mask/action sequence is long enough that event bits per original event increase.

## Comparator Snapshot

- Full v3 audit route: `TEST_NEXT`; full summary available: `True`.
- Conditioned short rollout route: `KILL`; reason: `conditioned objective tripped a hard safety or event-count calibration guard: median_event_count_ratio_in_range`.
- CE-weight training gate route: `MUTATE`; reason: `event CE weighting reduced starvation but worsened rigid-grid cases beyond the baseline gate`.

## Next Step

Kill factorized event-signature repair and test another target-side or v2.1 grammar repair candidate.
