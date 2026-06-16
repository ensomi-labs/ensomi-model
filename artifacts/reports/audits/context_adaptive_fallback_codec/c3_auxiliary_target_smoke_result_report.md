# C3 Auxiliary-Target Smoke Result Report

## Scope

This P12 pass implements the smallest legal C3 target-side smoke after P10/P11:

- C3 labels are target supervision, not input conditioning.
- Mapper output tokens and rollout stay unchanged.
- The new auxiliary head/loss are disabled by default.

The selected variant is a full-vocab multi-label bag target over exact C3 sidecar token ids.

## Implementation

Added disabled-by-default Mapper V2.1 config:

- `use_c3_auxiliary_target`
- `c3_auxiliary_vocab_size`
- `lambda_c3_auxiliary`

When enabled, the model mean-pools decoder hidden states over valid target positions and emits `c3_auxiliary_logits`. The V2.1 loss converts `c3_side_stream_tokens` into a multi-hot target and adds `lambda_c3_auxiliary * loss/c3_auxiliary`.

C3 side-stream conditioning remains separate. In this smoke config:

- `use_c3_side_stream_conditioning=false`
- `use_c3_auxiliary_target=true`

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke.yaml
```

Observed:

- focused model/training tests: `17 passed in 0.90s`
- broader focused tests: `22 passed in 0.84s`
- training smoke: completed `2/2` steps

## Smoke Result

| Metric | Value |
| --- | ---: |
| Completed steps | 2 |
| Complete | true |
| Device | cpu |
| Parameter count | 15,542,563 |
| C3 sidecar tensors loaded | true |
| C3 input conditioning enabled | false |
| C3 auxiliary target enabled | true |
| C3 auxiliary vocab size | 14,294 |
| `lambda_c3_auxiliary` | 0.05 |

Dataset proof:

- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- `c3_side_stream_max_tokens=256`
- train windows: `174,472`
- eval windows: `43`
- final train-eval windows: `2`

## Loss Metrics

| Split | `loss/total` | `loss/token` | `loss/c3_auxiliary` | C3 samples | C3 tokens | C3 positive labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Last train | 3.345135 | 3.177376 | 0.713438 | 1 | 30 | 25 |
| Final eval | 3.496222 | 3.366773 | 0.714097 | 43 | 206 | 191 |
| Final train-eval | 3.362753 | 3.224133 | 0.713017 | 2 | 23 | 22 |

Eval history:

| Step | Train total | Train C3 aux | Eval total | Eval C3 aux |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.869721 | 0.715415 | 3.505681 | 0.715327 |
| 2 | 3.345135 | 0.713438 | 3.496222 | 0.714097 |

Total-loss accounting matched the configured formula. Final eval had zero delta between reported total and recomputed total.

## What Passed

- Default disabled path stays inert: no auxiliary logits unless enabled.
- Auxiliary target path produces logits of the requested C3 vocab size.
- Auxiliary loss is finite.
- Auxiliary-head gradients are nonzero in unit tests.
- Real-data smoke completes with C3 sidecar labels loaded.
- C3 labels are not used as model conditioning in the smoke config.

## What Surfaced

- This is a wiring smoke, not a learnability result.
- Bag supervision discards C3 token order and `REF`/`RES` sequence semantics.
- Full-vocab BCE is workable for a smoke but should be compared with top-K or kind-aware diagnostics before longer runs.
- This does not make C3 generation-ready.

## Interpretation

Decision: TEST_NEXT.

P12 proves a legal target-side C3 supervision path can be wired through the mapper without target-derived input conditioning. That moves C3 closer to full-pipeline use than P8/P9 input conditioning, but it is not yet the new tokenization in the full pipeline.

Recommended next step: run a longer auxiliary-target learnability comparison with fixed baseline and C3-aux configs. If C3 auxiliary loss decreases cleanly without harming main mapper loss, keep the auxiliary path for regularization/probing. If it stays flat or noisy, mutate toward top-K/kind-aware labels or start the target-grammar decomposition card.
