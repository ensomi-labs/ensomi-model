# C3 Exact Window Assignment Comparison Result Report

## Scope

This P4 audit tests whether the P3 mapper-window sidecar can keep using `chunk_sort_ms` as the window anchor for C3 fallback side-stream tokens.

The run is intentionally bounded:

- source slice: 64 high-boundary-risk real beatmaps,
- source selection: highest `chunk_sort_ms` near-window-boundary exposure,
- chunk cache: untruncated,
- exact comparison: not full-cache,
- baseline anchor: P3 `chunk_sort_ms` window,
- comparator anchor: exact per-group snapped time reconstructed from the beatmap.

Command:

```bash
uv run python -m pulsefield_model.osu_core.c3_side_stream_tokenization --exact-window-compare --source-limit 64
```

## Result

Decision: MUTATE.

The exact comparison rejects current P3 chunk-sort anchoring on the high-risk slice.

| Check | Result |
| --- | ---: |
| Exact comparison pass | false |
| Kill chunk-sort anchoring | true |
| Token mismatch threshold | 1.000% |
| Token window mismatch rate | 6.797% |
| Compared side-stream tokens | 76,464 |
| Mismatched side-stream tokens | 5,197 |
| Compared fallback records | 66,002 |
| Mismatched fallback records | 4,520 |
| Record mismatch rate | 6.848% |
| Missing chunk anchors | 0 |
| Missing exact anchors | 0 |
| Parse errors | 0 |
| Parsed sources | 64 / 64 |
| Exact group anchors | 218,621 |
| Boundary-risk token rate | 26.572% |

Mismatch by token kind:

| Token kind | Compared | Mismatched | Mismatch rate |
| --- | ---: | ---: | ---: |
| raw | 26,867 | 1,831 | 6.815% |
| ref | 10,462 | 677 | 6.471% |
| residual | 39,135 | 2,689 | 6.871% |

## What Passed

- Exact timing reconstruction worked on the bounded real-map slice: 64 of 64 sources parsed successfully.
- The traced C3 side-stream tokens could all be matched to exact group anchors: missing exact anchor tokens were 0.
- The P3 chunk-side lookup did not drop tokens: missing chunk anchor tokens were 0.
- The comparison directly measured side-stream token assignment, not only chunk rows.
- Mismatch was not isolated to a single token class; raw, ref, and residual tokens all showed similar mismatch rates.

## What Failed

The current P3 sidecar uses the chunk's `chunk_sort_ms` to assign all traced side-stream tokens in that chunk to a mapper window. On high-boundary-risk maps, that approximation moves a material fraction of C3 tokens into the wrong 8s mapper window compared with exact per-group snapped time.

Observed failure:

- mismatch rate: 6.797%,
- kill threshold: 1%,
- kill condition triggered: yes.

This is large enough that chunk-sort anchoring should not be used for model conditioning.

## What This Proves

The audit proves that P3's length/cap tractability result is not enough. The C3 side-stream can be generated and matched back to beatmap structure, but its mapper-window assignment must use exact per-group timing before it becomes trustworthy model input.

More specifically:

- C3 tokenization itself remains viable.
- P3 sidecar tensor shape and cap analysis remain useful.
- P3 window anchoring is the weak link.
- The next implementation should preserve the C3 token stream but replace `chunk_sort_ms` window assignment with exact group snapped-time assignment.

## What This Does Not Prove

- It does not estimate the full-cache exact mismatch rate.
- It does not measure model utility.
- It does not prove the high-risk slice is representative of random or low-risk maps.
- It does not decide whether the mapper should consume the C3 side-stream as embeddings, discrete ids, pooled features, or another conditioning form.

## Surfaced Issue

The issue is not fallback representation coverage. It is timing alignment.

The current P3 sidecar treats a chunk as the timing owner for side-stream tokens. But a C3 side token is anchored to a fallback record, and that record corresponds to an exact beat group. Near 8s mapper-window boundaries, the chunk-level timestamp can disagree with the group's snapped timestamp. When that happens, the side-stream token lands in the wrong mapper window.

Example class of mismatch:

- chunk window: 48,000 ms,
- exact group window: 56,000 ms,
- token kind: `ref`,
- reason: `chunk_sort_ms` and exact snapped group time fall on opposite sides of an 8s boundary.

## Interpretation

This is a clean negative result for P3 chunk-sort anchoring, not a negative result for C3 tokenization.

Recommendation:

1. Replace P3 sidecar generation with exact per-group snapped-time window assignment.
2. Re-run the mapper-window sidecar cap audit using exact timing.
3. Only after exact sidecar generation passes should we run a disabled-by-default model-conditioning probe.

Next-loop action: mutate P3 into exact-timing sidecar generation.
