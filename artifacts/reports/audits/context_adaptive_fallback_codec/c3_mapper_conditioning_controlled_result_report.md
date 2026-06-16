# C3 Mapper Conditioning Controlled Comparison Result Report

## Scope

This P8 pass runs a 20-step fixed-seed controlled comparison using the exact P5 C3 sidecar.

The two runs use matched small CPU configs:

- baseline: exact C3 sidecar tensors are loaded and batched, but `use_c3_side_stream_conditioning=false`,
- enabled: same data path with `use_c3_side_stream_conditioning=true`.

This is still a small controlled run, not a production-quality result.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Result

Decision: TEST_NEXT.

Both runs completed 20/20 steps and wrote reports/checkpoints. The C3-enabled run tracked the baseline closely and stayed far inside the 5% negative gate.

| Metric | Baseline | C3 enabled |
| --- | ---: | ---: |
| Completed steps | 20 | 20 |
| Complete | true | true |
| Parameter count | 15,070,861 | 15,300,126 |
| Last train `loss/total` | 3.333956 | 3.335107 |
| Final eval `loss/total` | 3.285888 | 3.286643 |
| Final train-eval `loss/total` | 3.197277 | 3.197439 |
| C3 tensors loaded | true | true |
| C3 conditioning enabled | false | true |

C3-enabled final eval loss delta:

- absolute: `+0.000754`,
- relative: `+0.0230%`,
- kill threshold: `+5%`,
- kill triggered: no.

## Eval Curve

| Step | Baseline eval loss | C3-enabled eval loss | Enabled minus baseline |
| ---: | ---: | ---: | ---: |
| 1 | 3.469915 | 3.470787 | +0.000872 |
| 5 | 3.433085 | 3.433916 | +0.000831 |
| 10 | 3.384971 | 3.385782 | +0.000811 |
| 15 | 3.334120 | 3.334914 | +0.000794 |
| 20 | 3.285888 | 3.286643 | +0.000754 |

Both curves improve smoothly over the 20-step run. The C3-enabled run remains consistently but only slightly above baseline under this tiny setup.

## Dataset Proof

Both reports recorded:

- `include_c3_side_stream_token_tensors=true`,
- `c3_side_stream_token_sidecar_path=artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`,
- `c3_side_stream_max_tokens=256`,
- `train_window_count=174472`,
- `eval_window_count=43`,
- `final_train_eval_window_count=8`,
- `include_full_song_context=false`,
- `require_control_teacher_cache=true`.

## What Passed

- Exact C3 sidecar loaded in both controlled runs.
- C3 tensors traversed dataset, collate, training loop, and model boundary.
- Baseline and C3-enabled runs both completed 20 steps with finite losses.
- C3-enabled final eval loss was within 5% of baseline.
- Focused verifier suite passed: `15 passed`.

## What This Proves

P8 strengthens P7 from a runtime smoke to a small trend check. It shows pooled exact-C3 conditioning is training-stable on a fixed real-data slice and does not create an immediate loss regression under the small CPU setup.

This is enough evidence to justify a longer controlled comparison.

## What This Does Not Prove

- It does not prove C3 improves model quality.
- It does not validate production-size MPS training.
- It does not compare alternative C3 conditioning architectures.
- It does not solve cross-window `REF`/`RES` semantics.

## Interpretation

The result is neutral-positive for full-pipeline readiness:

- positive: exact C3 sidecar is now usable in a real training loop and remains stable for 20 steps,
- neutral: pooled C3 conditioning is not yet better than baseline on short-run loss,
- remaining risk: usefulness and architecture quality require longer controlled runs.

Recommended next step: run a longer controlled C3/no-C3 comparison on the same exact sidecar, with a fixed eval slice and enough steps to see whether the small loss offset shrinks, grows, or reverses.
