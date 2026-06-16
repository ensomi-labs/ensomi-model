# C3 Reduced Per-Kind Sidecar Result Report

## Scope

This P16 pass tests a reduced C3 target representation after P14/P15 killed flat full-vocab bag BCE. The reduced sidecar keeps the top 512 train labels from each C3 kind:

- `RAW`: 512 labels
- `REF`: 512 labels
- `RES`: 512 labels

The resulting auxiliary target vocabulary has 1,536 labels instead of 14,294. C3 remains target-side only; input conditioning is disabled.

## Commands

```bash
uv run python -m pulsefield_model.evals.c3_reduced_sidecar --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml --source-sidecar artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json --output-sidecar artifacts/cache/c3_mapper_window_sidecar/c3_reduced_per_kind_top512_le3.json --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_generation_summary.json --top-per-kind 512
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_enabled.yaml
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_enabled.yaml --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_reduced_per_kind_sidecar_enabled/checkpoint.pt --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_diagnostics_summary.json --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_reduced_per_kind_sidecar_diagnostics_result_report.md --device cpu --batch-size 2
```

## Result

Decision: TEST_NEXT for the reduced/decomposed C3 path, not production promotion.

P16 does not beat the reduced train-unigram baseline, but it is the first auxiliary-target variant with broad, non-trivial positive-label recovery.

| Metric | P14 full vocab | P15 full vocab weighted | P16 reduced per-kind |
| --- | ---: | ---: | ---: |
| Auxiliary vocab size | 14,294 | 14,294 | 1,536 |
| Model recall@20 | 0.000436 | 0.000872 | 0.018997 |
| Hits@20 | 1 / 2293 | 2 / 2293 | 36 / 1895 |
| Unigram recall@20 | 0.074139 | 0.074139 | 0.089710 |
| Beats unigram | false | false | false |

P16 improves model recall@20 by `21.78x` over P15 and `43.56x` over P14, but still trails reduced unigram by a large margin.

## Sidecar Coverage

| Coverage | Value |
| --- | ---: |
| Train positive-label coverage | 0.801789 |
| Eval positive-label coverage | 0.826428 |
| Eval RAW coverage | 0.801878 |
| Eval REF coverage | 0.855422 |
| Eval RES coverage | 0.844866 |

The reduced target retains enough held-out labels to be a meaningful C3 structure probe.

## Training Metrics

| Metric | Value |
| --- | ---: |
| Completed steps | 100 |
| Final eval `loss/token` | 2.706093 |
| Final eval `loss/c3_auxiliary` | 1.056063 |
| Main token-loss delta vs P13 baseline | +0.00616% |
| `c3_auxiliary_positive_weight` | 64.0 |

Main mapper safety passed. The run is well inside the +2% token-loss regression gate.

Auxiliary eval loss decreased from `1.098192` to `1.056063`, a `-3.836%` relative change.

## Kind Recovery

| Kind | Target labels | Model hits@20 | Model recall@20 | Unigram hits@20 | Unigram recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 854 | 18 | 0.021077 | 121 | 0.141686 |
| REF | 284 | 1 | 0.003521 | 18 | 0.063380 |
| RES | 757 | 17 | 0.022457 | 31 | 0.040951 |

This passes the P16 alternate positive signal: recall improves by more than 10x over P15 and all three token kinds are recovered at least once.

## What Passed

- The reducer created a train-derived 1,536-label sidecar with balanced per-kind selection.
- The reduced sidecar preserved `82.64%` of eval positive labels.
- Training completed and remained main-loss safe.
- Top-K recovery improved substantially over both full-vocab variants.
- The model recovered `RAW`, `REF`, and `RES` labels, whereas P14/P15 were nearly empty.

## What Surfaced

The reduced flat bag target still does not beat unigram. That means this is not yet full-pipeline-ready target tokenization. The recovery improvement is real enough to keep the reduced/decomposed branch alive, but the model is still not using enough chart/audio-conditioned information to outperform a frequency prior.

## Interpretation

P16 changes the conclusion from “flat bag C3 is dead” to “flat full-vocab C3 is dead; reduced/decomposed C3 is viable enough to test once more.” The next experiment should stop treating all C3 labels as one bag and either:

- split the reduced target into separate `RAW`, `REF`, and `RES` heads; or
- run a longer reduced-sidecar comparison with online diagnostics to see whether the gap to unigram closes.

The safer next research move is separate kind heads because P16 shows `REF` remains weak even after balanced vocabulary selection.

## Verification

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Result: `29 passed in 0.88s`.
