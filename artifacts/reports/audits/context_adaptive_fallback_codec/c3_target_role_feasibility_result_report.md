# C3 Target-Role Feasibility Result Report

## Scope

This P11 pass follows P10's role audit. Since exact C3 side-stream tokens are target-derived, this audit measures whether C3 is better pursued next as:

- an auxiliary target,
- a two-stage predicted plan,
- or a full target grammar replacement.

No production code was changed and no mapper training was run.

## Decision

Decision: TEST_NEXT.

Selected next family: C3 auxiliary target.

The next bounded card should add a disabled-by-default auxiliary head that predicts C3 sidecar labels as target supervision while keeping the existing mapper output and inference path unchanged.

Rejected for immediate implementation:

- current input conditioning: killed by P10 because it consumes target-derived labels as input;
- two-stage C3 plan: legal, but larger than needed before testing learnability;
- full target grammar: likely important later, but requires a dedicated decode/replay grammar for `RAW`/`REF`/`RES` and cross-window references.

## Target Shape

The exact sidecar is compact enough for an auxiliary-target smoke:

| Metric | Value |
| --- | ---: |
| Sidecar windows | 173,268 |
| Windows with C3 tokens | 166,801 |
| Window coverage | 96.268% |
| Sidecar token count | 3,203,904 |
| Token vocab size | 14,294 |
| Mean tokens/window | 18.491 |
| P50 tokens/window | 12 |
| P90 tokens/window | 43 |
| P95 tokens/window | 60 |
| P99 tokens/window | 101 |
| Max tokens/window | 274 |

Cap pressure is acceptable:

| Cap | Overflow token rate | Truncated window rate |
| ---: | ---: | ---: |
| 64 | 5.923% | 4.258% |
| 128 | 0.443% | 0.331% |
| 256 | 0.000562% | 0.000577% |
| 512 | 0.000% | 0.000% |
| 1024 | 0.000% | 0.000% |

At cap 256, only 1 window truncates and only 18 tokens overflow. That is small enough for a bounded auxiliary target experiment using the existing dataset tensor cap.

## Token Kind Shape

| Kind | Count | Rate | Vocab size |
| --- | ---: | ---: | ---: |
| `RAW` | 1,167,043 | 36.426% | 7,039 |
| `REF` | 417,208 | 13.022% | 2,020 |
| `RES` | 1,619,653 | 50.552% | 5,235 |

Interpretation:

- `RES` dominates token count, so a full grammar must model residual payloads, not only reference pointers.
- `REF` is a smaller and more structured subspace, which is useful for later decomposition.
- A single 14,294-way sequence target is possible, but the kind split suggests an auxiliary smoke should at least report kind-specific loss/accuracy.

## Vocabulary Concentration

| Top-K tokens | Token mass |
| ---: | ---: |
| 10 | 4.861% |
| 50 | 16.531% |
| 100 | 24.826% |
| 500 | 56.963% |
| 1,000 | 74.016% |
| 2,000 | 88.081% |
| 5,000 | 97.964% |

The top 1,000 tokens cover about 74% of token occurrences and the top 5,000 cover about 98%. This is favorable for an auxiliary smoke with either:

- a full-vocab target plus top-K diagnostics, or
- a top-K plus `OTHER` target as the first cheap probe.

## Cross-Window Semantics

The sidecar is target-window aligned, but C3 references are not purely local:

| Metric | Value |
| --- | ---: |
| Target cross-window span rate | 10.585% |
| Reference cross-window span rate | 10.351% |
| Target/reference same-window span rate | 31.961% |

This is the main reason not to jump directly into full target grammar replacement. A generator-side C3 grammar needs explicit cross-window reference state. An auxiliary target can test learnability without needing to decode the C3 stream yet.

## What Passed

- The exact C3 label stream is broad-coverage: 96.3% of mapper windows have C3 labels.
- Per-window target length is tractable: mean 18.5, p99 101.
- Existing cap 256 is effectively lossless for full-cache sidecar labels.
- Vocabulary concentration is good enough to support a cheap top-K diagnostic.
- Focused side-stream/data-window tests passed.

## What Surfaced

- Direct target grammar remains stateful because about 10% of spans cross mapper-window boundaries.
- The label space is not tiny: 14,294 token ids, with `RAW` and `RES` carrying most of the payload.
- A bag-only target would discard order and reference/residual sequence semantics.
- Auxiliary-target success would be a learnability/regularization signal, not full generation readiness.

## Interpretation

P11 selects auxiliary target as the next bounded implementation.

That does not mean auxiliary target is the final desired form of C3. It means it is the cheapest legal test after P10:

- it avoids target leakage,
- it does not require inference-side C3 input,
- it reuses existing sidecar tensors,
- it can expose whether C3 labels are learnable from the mapper hidden/control state before building a full C3 grammar.

## Recommended Next Card

Create a C3 auxiliary-target smoke experiment:

- add disabled-by-default model/loss config flags;
- predict C3 token labels from a pooled mapper/window representation;
- start with top-K token classification or multi-label/bag diagnostics;
- report total C3 auxiliary loss plus kind-specific metrics for `RAW`, `REF`, and `RES`;
- keep the main mapper output, rollout, and default configs unchanged.

Positive signal would be finite decreasing auxiliary loss and no regression in existing mapper tests. Negative signal would mutate toward target-grammar decomposition instead of spending more time on auxiliary regularization.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
```

Observed:

- focused C3 side-stream/data-window suite: `10 passed in 0.83s`
