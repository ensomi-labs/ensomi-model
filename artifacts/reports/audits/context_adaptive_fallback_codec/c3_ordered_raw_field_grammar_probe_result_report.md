# C3 Ordered RAW Field Grammar Probe Result Report

## Scope

This pass tests whether exact C3 side-stream tokens can be decomposed into an ordered RAW field grammar without losing reconstruction. It is an artifact-only target-design probe: no mapper training, rollout, or inference input path is changed.

## Result

Decision: `MUTATE`.

- sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- windows: `173268`
- exact C3 tokens: `3203904`
- ordered symbols: `10206813`
- ordered/exact sequence ratio: `3.185742`
- reconstruction mismatches: `0`
- missing token ids: `0`
- selected split: `test`
- exact C3 bits/token: `10.877115`
- ordered bits/original token: `19.659257`
- RAW exact bits/RAW token: `9.475498`
- RAW ordered bits/RAW token: `30.189757`

## Label Space

- exact C3 vocab size: `14294`
- ordered symbol vocab size: `7582`
- RAW exact vocab size: `7039`
- RAW ordered symbol vocab size: `327`
- RAW part-position vocab sizes: `{'0': 1, '1': 94, '2': 16, '3': 16, '4': 16, '5': 182}`

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
- Ordered/exact sequence ratio is 3.185742, so any training card must budget sequence length explicitly.
- RAW ordered bits per RAW token are 30.189757 versus exact RAW 9.475498.
- The RAW bit-reduction gate did not pass.
- The sequence-expansion gate did not pass.
- The mapper sidecar preserves REF tokens but does not itself expose exact cross-window reference share.

## Next Step

Do not train yet; refine the ordered grammar or compare against v3/v2.1 target complexity before spending model runtime.
