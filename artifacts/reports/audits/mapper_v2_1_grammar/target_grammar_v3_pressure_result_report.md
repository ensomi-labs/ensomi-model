# Target Grammar v3 Pressure Result Report

## Scope

This audit simulates a reversible v3 target grammar without changing the production tokenizer. The candidate replaces same-time v2.1 sparse lane-action runs with train-derived `GROUP|signature` tokens, leaves time shifts and EOS unchanged, and falls back to literal v2.1 lane-action tokens for out-of-dictionary groups.

## Result

Decision: `TEST_NEXT`.

- train windows scored: `4096` / `174373`
- eval windows scored: `142`
- baseline eval tokens: `28073`
- baseline total unigram bits: `118923.71`
- selected K: `128`
- selected eval tokens: `23478`
- selected token reduction: `16.37%`
- selected total-bit reduction: `-0.65%`
- best total-bit K: `256` (`5.27%`)
- best token-count K: `256` (`19.96%`)
- selected group coverage: `95.47%`
- selected multi-lane group coverage: `89.61%`
- reconstruction mismatches: `0`
- elapsed: `67.44s`

## Candidate Sweep

| K | Eval tokens | Token reduction | Total-bit reduction | Bits/token | Group coverage | Multi-lane coverage | Mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 25841 | 7.95% | -5.61% | 4.860361 | 76.16% | 50.43% | 0 |
| 32 | 25368 | 9.64% | -5.68% | 4.954042 | 82.41% | 59.68% | 0 |
| 64 | 24725 | 11.93% | -5.11% | 5.055693 | 88.92% | 74.60% | 0 |
| 128 | 23478 | 16.37% | -0.65% | 5.098287 | 95.47% | 89.61% | 0 |
| 256 | 22471 | 19.96% | 5.27% | 5.013281 | 100.00% | 100.00% | 0 |

## Top Train Groups

| Signature | Count | Non-empty lanes |
| --- | ---: | ---: |
| `..T.` | 29879 | 1 |
| `.T..` | 29874 | 1 |
| `...T` | 21106 | 1 |
| `T...` | 21027 | 1 |
| `..TT` | 11210 | 2 |
| `TT..` | 11192 | 2 |
| `T..T` | 9218 | 2 |
| `.T.T` | 4917 | 2 |
| `T.T.` | 4873 | 2 |
| `.TT.` | 4560 | 2 |
| `..S.` | 3427 | 1 |
| `.S..` | 3073 | 1 |

## What Passed

- Exact reconstruction guard passed for every tested K.
- The grammar shape satisfies the local target constraints: current event group only, literal fallback, no C3-style references, and no future target dependency.
- The selected top-K candidate cleared the pressure gate through token-count reduction; its simple unigram total-bit proxy did not improve.

## What Surfaced

Top-128 group fallback is pressure-positive only through token count: it preserves exact v2.1 reconstruction and shortens the stream, but the simple train-unigram total-bit proxy worsens. This means the representation is locally legal and shorter, while dictionary entropy still needs a trained-loss or dictionary-size check before promotion.

## Next Step

Create a v3 group-token grammar smoke card for vocab, tokenizer, replay expansion, masks, and a dictionary-size/trained-loss guard.
