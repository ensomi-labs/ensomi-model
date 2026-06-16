# C3 Kind-Heads Reduced Sidecar Result Report

## Scope

This P17 pass tests the next decomposition after P16: keep the reduced 1,536-label C3 sidecar, but replace the single auxiliary head with three 512-label kind heads and use kind-balanced auxiliary loss.

C3 remains target-side only. Input conditioning is disabled.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_enabled.yaml
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_enabled.yaml --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_enabled/checkpoint.pt --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_diagnostics_summary.json --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_reduced_sidecar_diagnostics_result_report.md --device cpu --batch-size 2
```

## Result

Decision: TEST_NEXT for kind-decomposed reduced C3, not production promotion.

P17 improves over P16 on total recall and especially `REF`, while staying main-loss safe. It still does not beat reduced unigram, so this remains a research direction, not a full-pipeline-ready tokenizer.

| Metric | P16 reduced flat | P17 kind heads |
| --- | ---: | ---: |
| Auxiliary vocab size | 1,536 | 1,536 |
| Model recall@20 | 0.018997 | 0.025858 |
| Hits@20 | 36 / 1895 | 49 / 1895 |
| Reduced unigram recall@20 | 0.089710 | 0.089710 |
| Beats unigram | false | false |
| Final eval `loss/token` | 2.706093 | 2.705861 |
| Final eval `loss/c3_auxiliary` | 1.056063 | 1.049002 |

Model recall@20 improved by `+36.11%` relative to P16.

## Kind Recovery

| Kind | P16 recall@20 | P17 recall@20 | P17 hits | Unigram recall@20 |
| --- | ---: | ---: | ---: | ---: |
| RAW | 0.021077 | 0.032787 | 28 / 854 | 0.141686 |
| REF | 0.003521 | 0.017606 | 5 / 284 | 0.063380 |
| RES | 0.022457 | 0.021136 | 16 / 757 | 0.040951 |

The selected target improved: `REF` recall is `5x` P16. `RAW` improved by `1.56x`. `RES` slipped slightly, but all three kinds still recover labels.

## Training Metrics

| Metric | Value |
| --- | ---: |
| Completed steps | 100 |
| Kind heads | `[512, 512, 512]` |
| Kind-balanced loss | true |
| `c3_auxiliary_positive_weight` | 64.0 |
| Final eval `loss/token` | 2.705861 |
| Main token-loss delta vs P13 baseline | -0.00243% |

Main mapper safety passed. The run is inside the +2% token-loss regression gate.

Auxiliary eval loss decreased from `1.102703` to `1.049002`, a `-4.870%` relative change.

## What Passed

- Optional kind-head C3 auxiliary path is wired and validated.
- Optional kind-balanced C3 loss is wired and validated.
- Training completed and wrote a checkpoint/report.
- Main mapper token loss stayed safe.
- Total C3 recall@20 improved over P16.
- `REF` recovery improved materially, which was the P17 target.

## What Surfaced

The result is still below reduced unigram. Kind decomposition helps, but it does not yet prove chart/audio-conditioned C3 recovery beyond a frequency prior. The diagnostics helper reports `KILL` because its generic positive gate requires beating unigram; under the P17 card, the result is `TEST_NEXT` because it improves over P16 and fixes the weakest kind enough to justify one more bounded test.

## Interpretation

The evidence now supports this hierarchy:

- flat full-vocab C3 bag BCE: killed;
- scalar positive-weighted full-vocab C3 bag BCE: killed;
- reduced flat C3 bag target: viable but below unigram;
- reduced kind-decomposed C3 target: better than reduced flat, still below unigram.

The next step should not be production integration. It should either run the kind-head setup longer with online diagnostics, or move from bag labels to a structured C3 target grammar. Since P17 improved `REF`, a longer kind-head run is now a reasonable bounded check before abandoning bag-style reduced targets.

## Verification

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Result: `30 passed in 0.83s`.
