# Target Grammar v3 Full Dataset Audit Result Report

## Scope

This full-dataset audit tests v3 as a complete local event-token target grammar: one 4-lane event token per non-empty same-time group. It compares against v2.1 sparse lane-action target streams and verifies exact expansion back to v2.1 tokens.

## Result

Decision: `TEST_NEXT`.

- train windows scored: `174373` / `174373`
- eval windows scored: `142` / `142`
- baseline eval tokens: `28073`
- v3 eval tokens: `22471`
- token reduction: `19.96%`
- total-bit reduction: `5.43%`
- full-audit windows: `174515`
- full-audit token reduction: `20.54%`
- full-audit reconstruction mismatches: `0`
- full-chart terminal windows: `9360`
- terminal event-at-chart-end windows: `9360`
- cross-window LN windows: `80316`
- v3 bits/token: `4.971069`
- event tokens: `9112`
- multi-lane events: `3976`
- reconstruction mismatches: `0`
- elapsed: `2555.38s`

## Top Event Signatures

| Signature | Count | Non-empty lanes |
| --- | ---: | ---: |
| `..T.` | 1260273 | 1 |
| `.T..` | 1258042 | 1 |
| `T...` | 876807 | 1 |
| `...T` | 869797 | 1 |
| `..TT` | 495183 | 2 |
| `TT..` | 492854 | 2 |
| `T..T` | 391601 | 2 |
| `.TT.` | 229055 | 2 |
| `.T.T` | 222373 | 2 |
| `T.T.` | 219630 | 2 |
| `T.TT` | 139012 | 3 |
| `TT.T` | 138082 | 3 |

## What Passed

- V3 event tokens expanded back to the exact v2.1 sparse target stream on the audited windows.
- The candidate is local and teacher-forcing friendly: time shifts plus one current event-group token.
- No C3-style cross-window reference or future target lookup is needed.

## What Surfaced

The v3 event-token grammar passed the audit gate: exact reconstruction holds on the audited scope, target streams are shorter, and the simple total-bit proxy improves. This permits the next representation gate, but trained mapper quality still needs full-pipeline validation.

## Next Step

Create the v3 dataset/model/training/inference replacement Experiment Card.
