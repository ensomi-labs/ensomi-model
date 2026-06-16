# C3 Kind-Heads 1000-Step Result Report

## Scope

This P19 pass reruns the reduced C3 kind-head setup for 1000 steps. The representation is unchanged from P18:

- reduced top-512-per-kind sidecar;
- three 512-label C3 auxiliary heads;
- kind-balanced C3 auxiliary loss;
- C3 target-side only, no input conditioning.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoint.pt --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_summary.json --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_diagnostics_result_report.md --device cpu --batch-size 2
```

## Result

Decision: MUTATE toward a RAW-focused structured C3 target audit.

P19 improves over P18, but it fails the positive gate: it reaches `82.94%` of reduced unigram recall@20, below the 90% target, and only closes `20.97%` of the remaining RAW gap.

| Metric | P18 500-step | P19 1000-step | Reduced unigram |
| --- | ---: | ---: | ---: |
| Model recall@20 | 0.058575 | 0.074406 | 0.089710 |
| Hits@20 | 111 / 1895 | 141 / 1895 | 170 / 1895 |
| Sample hit rate@20 | 0.532374 | 0.633094 | 0.661871 |
| Beats unigram | false | false | n/a |

P19 recall@20 is `1.27x` P18. The curve is still moving, but not enough to justify another blind horizon extension before understanding RAW.

## Kind Recovery

| Kind | P18 recall@20 | P19 recall@20 | P19 hits | Unigram recall@20 |
| --- | ---: | ---: | ---: | ---: |
| RAW | 0.069087 | 0.084309 | 72 / 854 | 0.141686 |
| REF | 0.059859 | 0.063380 | 18 / 284 | 0.063380 |
| RES | 0.046235 | 0.067371 | 51 / 757 | 0.040951 |

`REF` ties unigram and `RES` beats unigram. `RAW` remains the blocker at only `59.50%` of unigram.

## Training Curve

| Step | Eval `loss/token` | Eval `loss/c3_auxiliary` | Eval `loss/total` |
| ---: | ---: | ---: | ---: |
| 200 | 2.542445 | 0.999413 | 2.690549 |
| 400 | 2.394974 | 0.957996 | 2.539386 |
| 600 | 2.215317 | 0.938668 | 2.358604 |
| 800 | 2.092318 | 0.930328 | 2.234707 |
| 1000 | 2.012364 | 0.924094 | 2.154984 |

Auxiliary eval loss decreased by `-2.214%` from step 500 to step 1000, while recall@20 improved by `27.03%`.

## What Passed

- The 1000-step kind-head run completed and wrote a checkpoint/report.
- Recall@20 improved over P18.
- `REF` reached unigram.
- `RES` exceeded unigram.
- Main mapper token loss did not regress.

## What Surfaced

The kind-head bag objective is now clearly learnable, but the remaining error is concentrated in `RAW`. More training may still help, but the P19 gate was deliberately set to prevent indefinite horizon extension. The next useful step is to inspect and reformulate RAW structure.

## Interpretation

The evidence now says:

- reduced kind heads are the best C3 auxiliary path so far;
- `REF` and `RES` are no longer the main blockers in this slice;
- `RAW` remains too broad or too flat for the current bag target.

Recommended next card: RAW-focused structured C3 target audit. The audit should identify whether RAW misses are dominated by a few subfamilies, time-delta buckets, lane/action variants, or window-context effects, then decide whether to split RAW into smaller heads or move to ordered C3 generation.

## Verification

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/evals/test_c3_reduced_sidecar.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Result: `30 passed in 0.95s`.
