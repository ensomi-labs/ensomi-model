# C3 Auxiliary Positive-Weight Result Report

## Scope

This P15 pass tests one narrow mutation after P14 killed the unweighted full-vocab C3 bag auxiliary head: add scalar positive-label weighting to the same legal target-side auxiliary objective.

The run uses the same small CPU setup as P13/P14, with:

- `lambda_c3_auxiliary=0.05`
- `c3_auxiliary_positive_weight=128.0`
- C3 input conditioning disabled
- exact C3 sidecar loaded only as target-side labels

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml
uv run python -m pulsefield_model.evals.c3_auxiliary_diagnostics --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_enabled.yaml --checkpoint artifacts/runs/stage2_mapper_v2_1/c3_auxiliary_positive_weight_enabled/checkpoint.pt --summary-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_diagnostics_summary.json --report-output artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_positive_weight_diagnostics_result_report.md --device cpu --batch-size 2
```

## Result

Decision: KILL scalar full-vocab positive-weighted BCE.

The weighted run completed 100/100 steps and kept main token loss stable, but it still failed the actual C3 recovery gate.

| Metric | P13 baseline | P13 unweighted aux | P15 weighted aux |
| --- | ---: | ---: | ---: |
| Completed steps | n/a | 100 | 100 |
| `c3_auxiliary_positive_weight` | n/a | 1.0 | 128.0 |
| Final eval `loss/token` | 2.705927 | 2.706115 | 2.706216 |
| Final eval `loss/c3_auxiliary` | n/a | 0.546028 | 0.675291 |
| Main token-loss delta vs baseline | n/a | +0.00696% | +0.01070% |

Main mapper safety passed. The weighted run is still well inside the +2% token-loss regression gate.

## Auxiliary Loss Trend

| Step | Eval `loss/c3_auxiliary` |
| ---: | ---: |
| 20 | 0.792096 |
| 40 | 0.764599 |
| 60 | 0.736829 |
| 80 | 0.706860 |
| 100 | 0.675291 |

Weighted auxiliary loss decreased by `-14.746%`. This is a learning signal, but P14 already showed BCE loss alone is not sufficient evidence.

## Top-K Diagnostics

| Metric | P14 unweighted | P15 weighted | Train-unigram baseline |
| --- | ---: | ---: | ---: |
| Model recall@20 | 0.000436 | 0.000872 | 0.074139 |
| Hits@20 | 1 / 2293 | 2 / 2293 | 170 / 2293 |
| Relative lift vs unigram | -0.994118 | -0.988235 | n/a |

Kind recall@20 for P15:

| Kind | Target labels | P15 hits | P15 recall |
| --- | ---: | ---: | ---: |
| RAW | 1065 | 1 | 0.000939 |
| REF | 332 | 1 | 0.003012 |
| RES | 896 | 0 | 0.000000 |

## What Passed

- The positive-weight loss path is wired and validated.
- Training completed and wrote a checkpoint/report.
- Main mapper token loss stayed effectively tied to the P13 baseline.
- Weighted BCE loss decreased over the 100-step run.
- Diagnostics recovered one `REF` label, whereas P14 recovered no non-`RAW` labels.

## What Surfaced

Positive weighting improved the P14 top-20 result by only one extra hit. That is not enough to make the full-vocab bag target useful. The model still loses badly to a train-unigram baseline, which means scalar positive weighting did not fix the core ranking/recovery problem.

The evidence now closes two branches:

- unweighted full-vocab bag BCE: killed by P14;
- scalar positive-weighted full-vocab bag BCE: killed by P15.

## Interpretation

C3 remains worth pursuing, but not as a single full-vocab bag auxiliary BCE head. The next bounded direction should decompose the target so the model is not asked to rank 14,294 sparse labels in one flat space.

Recommended next card: kind-balanced or decomposed C3 target representation, likely separate `RAW`/`REF`/`RES` objectives or a smaller structured grammar target.

## Verification

```bash
uv run --group dev pytest tests/evals/test_c3_auxiliary_diagnostics.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

Result: `27 passed in 0.85s`.
