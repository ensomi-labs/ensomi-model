# C3 Auxiliary-Target Learnability Result Report

## Scope

This P13 pass runs a 100-step fixed-seed comparison for the legal C3 target-side auxiliary path introduced in P12.

The two runs use matched small CPU configs:

- baseline: exact C3 tensors are loaded, but C3 conditioning and C3 auxiliary target are disabled;
- enabled: exact C3 tensors are loaded as target labels, C3 conditioning is disabled, and the C3 auxiliary target is enabled.

This tests learnability and main-loss safety. It does not test generation quality.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Result

Decision: TEST_NEXT.

Both runs completed 100/100 steps. The C3 auxiliary target showed a clear short-run learning signal, while main mapper token loss remained effectively tied with baseline.

| Metric | Baseline | C3 auxiliary |
| --- | ---: | ---: |
| Completed steps | 100 | 100 |
| Complete | true | true |
| Parameter count | 15,070,861 | 15,542,563 |
| C3 input conditioning enabled | false | false |
| C3 auxiliary target enabled | false | true |
| `lambda_c3_auxiliary` | 0.0 | 0.05 |
| Final eval `loss/total` | 2.806852 | 2.834344 |
| Final eval `loss/token` | 2.705927 | 2.706115 |
| Final eval `loss/c3_auxiliary` | n/a | 0.546028 |
| Final train-eval `loss/token` | 2.787753 | 2.787887 |

The total-loss delta is expected because the auxiliary run includes `0.05 * loss/c3_auxiliary`.

## Auxiliary Loss Trend

| Step | Eval `loss/c3_auxiliary` |
| ---: | ---: |
| 20 | 0.684974 |
| 40 | 0.652073 |
| 60 | 0.618759 |
| 80 | 0.583185 |
| 100 | 0.546028 |

Auxiliary eval loss delta:

- absolute: `-0.138947`
- relative: `-20.285%`

This passes the P13 primary gate: the auxiliary loss is finite and decreases from first eval to final eval.

## Main Mapper Loss

| Step | Baseline eval `loss/token` | C3 auxiliary eval `loss/token` | Delta |
| ---: | ---: | ---: | ---: |
| 20 | 3.215512 | 3.215579 | +0.000067 |
| 40 | 3.051587 | 3.051674 | +0.000087 |
| 60 | 2.911257 | 2.911322 | +0.000065 |
| 80 | 2.796178 | 2.796260 | +0.000083 |
| 100 | 2.705927 | 2.706115 | +0.000188 |

Final eval token-loss delta:

- absolute: `+0.000188`
- relative: `+0.00696%`
- regression gate: `+2%`
- gate triggered: no

Final train-eval token-loss delta:

- absolute: `+0.000134`
- relative: `+0.00480%`

## Dataset Proof

Both reports recorded:

- `include_c3_side_stream_token_tensors=true`
- `c3_side_stream_token_sidecar_path=artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- `c3_side_stream_max_tokens=256`
- `train_window_count=174373`
- `eval_window_count=142`
- `final_train_eval_window_count=16`

## What Passed

- The auxiliary target learns a detectable signal over 100 steps.
- Main mapper token loss remains effectively unchanged versus baseline.
- C3 labels are used as target supervision, not input conditioning.
- Focused mapper/data/training tests passed: `22 passed`.
- The comparison stays inside the 2% main-loss guard.

## What Surfaced

- The result is still a bag-label learnability signal, not ordered C3 generation.
- Full-vocab BCE appears workable for short-run training, but it may mostly learn common labels and negatives.
- The next useful diagnostics should split metrics by top-K and C3 token kind (`RAW`, `REF`, `RES`).

## Interpretation

P13 strengthens P12 from wiring smoke to short-run learnability evidence. This is the first C3 integration result after P10 that is both legal for the full pipeline direction and empirically positive.

It does not mean C3 is already the mapper tokenization. It means C3 can be learned as target-side structure without target-derived input conditioning, and the path is worth one more bounded experiment.

Recommended next step: add richer C3 auxiliary diagnostics, preferably top-K and token-kind metrics, then rerun a controlled auxiliary comparison. If the signal remains clean, decide whether to keep auxiliary regularization or begin target-grammar decomposition.
