# Target Grammar v3 Event-Token Smoke Result Report

## Scope

This smoke audit tests v3 as a complete local event-token target grammar: one 4-lane event token per non-empty same-time group. It compares against v2.1 sparse lane-action target streams on the same bounded split and verifies exact expansion back to v2.1 tokens.

## Result

Decision: `TEST_NEXT`.

- train windows scored: `4096` / `174373`
- eval windows scored: `142`
- baseline eval tokens: `28073`
- v3 eval tokens: `22471`
- token reduction: `19.96%`
- total-bit reduction: `5.27%`
- v3 bits/token: `5.013281`
- event tokens: `9112`
- multi-lane events: `3976`
- reconstruction mismatches: `0`
- elapsed: `64.67s`

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

- V3 event tokens expanded back to the exact v2.1 sparse target stream on the eval split.
- The candidate is local and teacher-forcing friendly: time shifts plus one current event-group token.
- No C3-style cross-window reference or future target lookup is needed.

## What Surfaced

The v3 event-token grammar passed the smoke gate: exact reconstruction holds, target streams are shorter, and the simple total-bit proxy improves. This is still bounded-slice evidence; the updated goal requires a full 4K dataset audit before replacement.

## Next Step

Scale the v3 event-token audit to the full 4K dataset and add dataset/model integration gates.
