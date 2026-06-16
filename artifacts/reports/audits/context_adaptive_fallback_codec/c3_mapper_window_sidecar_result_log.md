# C3 Mapper-Window Sidecar Generation Result Log

## Summary

- Recommendation: MUTATE: full-cache C3 sidecar works, but chunk-sort window anchoring risk is high; compare against exact per-group timing.
- P3 sidecar generation pass: True
- Full cache: True
- Runtime seconds: 188.754157
- Limited: False
- Code dirty: True
- Reconstruction pass: True
- Loader guard pass: True
- Sidecar token preservation pass: True
- Sidecar tokens: 3203904
- Traced side-stream tokens: 3203904
- Window count: 173149
- Windows with tokens: 166125
- Token vocab size: 14294
- Tokens/window p95: 61.0
- Tokens/window p99: 101.0
- Tokens/window max: 268
- Cap sweep: {'64': {'truncated_window_count': 7446, 'truncated_window_rate': 0.043003424795984964, 'overflow_token_count': 192094, 'overflow_token_rate': 0.059956228401350356, 'max_overflow_tokens': 204}, '128': {'truncated_window_count': 561, 'truncated_window_rate': 0.0032399840599714696, 'overflow_token_count': 14758, 'overflow_token_rate': 0.004606255368450491, 'max_overflow_tokens': 140}, '256': {'truncated_window_count': 1, 'truncated_window_rate': 5.775372655920623e-06, 'overflow_token_count': 12, 'overflow_token_rate': 3.7454305746988673e-06, 'max_overflow_tokens': 12}, '512': {'truncated_window_count': 0, 'truncated_window_rate': 0.0, 'overflow_token_count': 0, 'overflow_token_rate': 0.0, 'max_overflow_tokens': 0}, '1024': {'truncated_window_count': 0, 'truncated_window_rate': 0.0, 'overflow_token_count': 0, 'overflow_token_rate': 0.0, 'max_overflow_tokens': 0}}
- Boundary-risk token rate: 0.24741721349953058
- Target cross-window span rate: 0.09427911257694004

## Interpretation

MUTATE: full-cache C3 sidecar works, but chunk-sort window anchoring risk is high; compare against exact per-group timing.
