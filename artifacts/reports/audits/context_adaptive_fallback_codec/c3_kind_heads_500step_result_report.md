# C3 Kind-Heads 500-Step Result Report

## Scope

This P18 pass reruns the P17 reduced C3 kind-head setup for 500 steps instead of 100. The representation is unchanged:

- reduced top-512-per-kind sidecar;
- three 512-label C3 auxiliary heads;
- kind-balanced C3 auxiliary loss;
- C3 target-side only, no input conditioning.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_enabled.yaml
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_enabled.yaml --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_500step/checkpoint.pt --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_diagnostics_summary.json --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_500step_diagnostics_result_report.md --device cpu --batch-size 2
```

## Result

Decision: TEST_NEXT for longer kind-head C3, not production promotion.

The 500-step run strongly improves over P17 and closes much of the gap to unigram, but still does not beat unigram overall.

| Metric | P17 100-step | P18 500-step | Reduced unigram |
| --- | ---: | ---: | ---: |
| Model recall@20 | 0.025858 | 0.058575 | 0.089710 |
| Hits@20 | 49 / 1895 | 111 / 1895 | 170 / 1895 |
| Sample hit rate@20 | 0.302158 | 0.532374 | 0.661871 |
| Beats unigram | false | false | n/a |

P18 recall@20 is `2.27x` P17. It still trails unigram by `34.71%` relative.

## Kind Recovery

| Kind | P17 recall@20 | P18 recall@20 | P18 hits | Unigram recall@20 |
| --- | ---: | ---: | ---: | ---: |
| RAW | 0.032787 | 0.069087 | 59 / 854 | 0.141686 |
| REF | 0.017606 | 0.059859 | 17 / 284 | 0.063380 |
| RES | 0.021136 | 0.046235 | 35 / 757 | 0.040951 |

`REF` is now close to unigram and `RES` beats unigram. The remaining overall gap is mostly `RAW`.

## Training Curve

| Step | Eval `loss/token` | Eval `loss/c3_auxiliary` | Eval `loss/total` |
| ---: | ---: | ---: | ---: |
| 100 | 2.705861 | 1.049002 | 2.859236 |
| 200 | 2.542445 | 0.999413 | 2.690549 |
| 300 | 2.456627 | 0.974079 | 2.602483 |
| 400 | 2.394974 | 0.957996 | 2.539386 |
| 500 | 2.316796 | 0.945019 | 2.460300 |

Auxiliary eval loss decreased by `-9.913%` from step 100 to step 500.

## What Passed

- The 500-step kind-head run completed and wrote a checkpoint/report.
- C3 recall@20 improved strongly over P17.
- All three C3 kinds improved substantially over P17.
- `REF` nearly caught the unigram baseline.
- `RES` exceeded the unigram baseline.
- Main mapper token loss did not regress.

## What Surfaced

The longer run shows the kind-head C3 signal is not saturated at 100 steps. This is the strongest evidence so far that the reduced/decomposed C3 auxiliary target is learnable.

The result is still not full-pipeline-ready. Overall top-20 recovery remains below unigram, and `RAW` is the main remaining gap. This means the next decision should be between:

- another longer kind-head run to see whether `RAW` continues improving; or
- a structured target design that handles `RAW` more explicitly.

## Interpretation

P18 upgrades the reduced kind-head branch from “barely viable” to “worth a larger check.” It also narrows the problem: `REF` and `RES` are no longer the main blockers in this small probe; `RAW` recovery is.

Recommended next card: either a 1000-step kind-head run with the same diagnostics, or a RAW-focused structured target audit before designing C3 generation.

## Verification

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Result: `30 passed in 0.91s`.
