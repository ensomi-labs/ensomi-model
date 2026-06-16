# C3 Mapper-Window Sidecar Generation Result Log

## Summary

- Recommendation: MUTATE: full-cache exact-timing C3 sidecar works, but cross-window reference spans are high; inspect packing before model conditioning.
- Sidecar generation pass: True
- P5 exact sidecar generation pass: True
- Window anchor mode: exact_group
- Full cache: True
- Runtime seconds: 359.858104
- Limited: False
- Code dirty: True
- Anchor count: 9719273
- Parsed sources: 9242
- Parse errors: 0
- Missing anchor tokens: 0
- Reconstruction pass: True
- Loader guard pass: True
- Sidecar token preservation pass: True
- Sidecar tokens: 3203904
- Traced side-stream tokens: 3203904
- Window count: 173268
- Windows with tokens: 166801
- Token vocab size: 14294
- Tokens/window p95: 60.0
- Tokens/window p99: 101.0
- Tokens/window max: 274
- Cap sweep: {'64': {'truncated_window_count': 7377, 'truncated_window_rate': 0.04257566313456611, 'overflow_token_count': 189756, 'overflow_token_rate': 0.05922649367771319, 'max_overflow_tokens': 210}, '128': {'truncated_window_count': 574, 'truncated_window_rate': 0.0033127871274557332, 'overflow_token_count': 14190, 'overflow_token_rate': 0.004428971654581411, 'max_overflow_tokens': 146}, '256': {'truncated_window_count': 1, 'truncated_window_rate': 5.771406145393264e-06, 'overflow_token_count': 18, 'overflow_token_rate': 5.618145862048301e-06, 'max_overflow_tokens': 18}, '512': {'truncated_window_count': 0, 'truncated_window_rate': 0.0, 'overflow_token_count': 0, 'overflow_token_rate': 0.0, 'max_overflow_tokens': 0}, '1024': {'truncated_window_count': 0, 'truncated_window_rate': 0.0, 'overflow_token_count': 0, 'overflow_token_rate': 0.0, 'max_overflow_tokens': 0}}
- Boundary-risk token rate: 0.24909547851621022
- Target cross-window span rate: 0.10585367490556269

## Interpretation

MUTATE: full-cache exact-timing C3 sidecar works, but cross-window reference spans are high; inspect packing before model conditioning.
