# C3 Ordered RAW Field Grammar Probe Result Report

## Scope

This pass tests whether exact C3 side-stream tokens can be decomposed into an ordered RAW field grammar without losing reconstruction. It is an artifact-only target-design probe: no mapper training, rollout, or inference input path is changed.

## Result

Decision: `MUTATE`.

- sidecar: `artifacts/cache/c3_mapper_window_sidecar/smoke_c3_exact_mapper_window_sidecar.json`
- windows: `310`
- exact C3 tokens: `2876`
- ordered symbols: `11570`
- ordered/exact sequence ratio: `4.022949`
- reconstruction mismatches: `0`
- missing token ids: `0`
- selected split: `test`
- exact C3 bits/token: `10.351670`
- ordered bits/original token: `17.533398`
- RAW exact bits/RAW token: `8.514221`
- RAW ordered bits/RAW token: `28.427295`

## Label Space

- exact C3 vocab size: `1193`
- ordered symbol vocab size: `688`
- RAW exact vocab size: `574`
- RAW ordered symbol vocab size: `69`
- RAW part-position vocab sizes: `{'0': 1, '1': 20, '2': 16, '3': 15, '4': 15}`

## Pass Criteria

- `reconstruction_pass`: `True`
- `raw_bits_reduction_pass`: `False`
- `sequence_expansion_bounded_pass`: `False`
- `label_space_reduction_pass`: `True`
- `no_missing_token_ids_pass`: `True`

## Interpretation

The ordered field representation is lossless and reduces label space, but its bit or sequence budget is not yet good enough for training. Treat this as a mutation signal, not promotion.

## What Passed

- Exact side-stream reconstruction is lossless under ordered RAW field encoding.
- All sidecar token ids resolved through the token vocabulary.
- RAW field splitting reduces the distinct RAW-side symbol space.

## What Surfaced

- Ordered RAW field splitting is target-design evidence only; it does not prove mapper learnability.
- Ordered/exact sequence ratio is 4.022949, so any training card must budget sequence length explicitly.
- RAW ordered bits per RAW token are 28.427295 versus exact RAW 8.514221.
- The RAW bit-reduction gate did not pass.
- The sequence-expansion gate did not pass.
- The mapper sidecar preserves REF tokens but does not itself expose exact cross-window reference share.

## Next Step

Do not train yet; refine the ordered grammar or compare against v3/v2.1 target complexity before spending model runtime.
