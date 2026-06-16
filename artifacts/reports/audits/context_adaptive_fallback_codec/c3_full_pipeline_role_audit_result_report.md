# C3 Full-Pipeline Role Audit Result Report

## Scope

This P10 pass audits whether the current exact C3 mapper-window side stream can be promoted as a full-pipeline mapper input.

It does not change production code and does not run more mapper training. The point is to decide whether more training on pooled C3 conditioning is meaningful for deployment.

## Decision

Decision: MUTATE.

Current role classification:

- current path: offline target-derived conditioning probe
- legal production input: no
- current pooled input-conditioning promotion: kill
- next family: C3 auxiliary target or target grammar candidate

## Evidence Chain

### Sidecar Provenance

`audit_c3_mapper_window_sidecar` builds the mapper sidecar from the beatmap-derived chunk cache, `r0_delta` fallback records, and the selected C3 plan. It traces side-stream tokens, assigns them to exact mapper windows, and writes a sidecar.

Evidence:

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:301`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:347`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:358`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:364`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:382`

`_build_mapper_window_sidecar_payload` writes traced C3 tokens into rows keyed by `beatmap_path` and `window_start_ms`, with `token_ids` attached to the target beatmap window.

Evidence:

- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:1286`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:1340`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:1356`
- `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py:1371`

This means the current sidecar is not an audio/control feature. It is a tokenization of the target chart's fallback substream.

### Dataset Path

`MapperV21WindowDataset` tokenizes the target beatmap window and, when enabled, attaches C3 side-stream tokens from a `beatmap_path` plus `window_start_ms` lookup.

Evidence:

- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:91`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:118`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:149`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:212`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:296`

This is valid for offline probes and teacher-forced audits. It is not automatically valid as a generation-time input.

### Model Path

The mapper model currently treats C3 side-stream tokens as conditioning input: it embeds, mean-pools, projects, scales, and adds the vector into `decoder_hidden`.

Evidence:

- `src/pulsefield_model/models/mapper/v2_1/model.py:415`
- `src/pulsefield_model/models/mapper/v2_1/model.py:427`
- `src/pulsefield_model/models/mapper/v2_1/model.py:462`
- `src/pulsefield_model/models/mapper/v2_1/model.py:465`

P8/P9 show this does not destabilize training. They do not prove deployability because the input is target-derived.

### Inference Path

`generate_full_song_rollout_v2_1` takes a model, vocab, chart end, control-window provider, generation parameters, and generated-token prefix state. It does not have a C3 side-stream provider.

Evidence:

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py:250`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py:292`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py:293`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py:408`

`MapperV21Model.incremental_decode_next_token` also explicitly rejects C3-conditioned configs.

Evidence:

- `src/pulsefield_model/models/mapper/v2_1/model.py:469`
- `src/pulsefield_model/models/mapper/v2_1/model.py:502`

So the current C3 conditioning path cannot be used by the current rollout pipeline without either leaking target beatmap information or adding a new model that predicts/provides C3 first.

## Existing Positive Evidence Still Holds

The exact sidecar remains strong:

| Check | Value |
| --- | ---: |
| P5 exact sidecar generation pass | true |
| Reconstruction pass | true |
| Loader guard pass | true |
| Sidecar tokens | 3,203,904 |
| Windows with tokens | 166,801 |
| Token vocab size | 14,294 |
| Missing anchor tokens | 0 |
| Token preservation pass | true |
| Target cross-window span rate | 10.585% |
| Reference cross-window span rate | 10.351% |
| Target/reference same-window span rate | 31.961% |

The P9 conditioning probe remains stable:

| Metric | Value |
| --- | ---: |
| Baseline final eval loss | 2.806852 |
| C3-enabled final eval loss | 2.806887 |
| Absolute delta | +0.0000355 |
| Relative delta | +0.00127% |
| P9 kill gate triggered | false |

Interpretation: P9 is still useful as a stability/gradient-path probe, but not as production-input evidence.

## What Passed

- Sidecar provenance is clear enough to classify the current role.
- Current C3 sidecar correctness remains strong.
- Mapper tests covering model, dataset, and trainer paths pass.
- The audit did not require production-code changes.

## What Surfaced

- Current C3 side-stream tokens are target-derived.
- Current model conditioning consumes those target-derived tokens as input.
- Current rollout has no legal generation-time source for them.
- Incremental decode rejects C3-conditioned configs.
- More training on pooled input conditioning would not prove full-pipeline readiness.

## Result Interpretation

The current C3 representation should not be promoted as a default mapper input.

The next integration family should be one of:

- auxiliary target: predict C3 side-stream tokens or summaries as an auxiliary head while keeping mapper output unchanged;
- target grammar / tokenization: make C3 `RAW`/`REF`/`RES` structure part of what the mapper emits and decodes;
- two-stage generation: predict a C3 plan from audio/control first, then condition the mapper on the predicted plan.

The smallest next card should prefer auxiliary target or target grammar over input conditioning, because those paths do not require target-derived sidecar tokens at inference time.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Observed:

- focused mapper suite: `20 passed in 0.84s`
