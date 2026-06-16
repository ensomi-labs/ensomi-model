# C3 Mapper Conditioning Longer Controlled Comparison Result Report

## Scope

This P9 pass runs a 100-step fixed-seed controlled comparison using the exact P5 C3 sidecar.

The two runs use matched small CPU configs:

- baseline: exact C3 sidecar tensors are loaded and batched, but `use_c3_side_stream_conditioning=false`.
- enabled: same data path with `use_c3_side_stream_conditioning=true`.

This strengthens P8, but it is still a small controlled run, not a production-quality result or inference-readiness proof.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Result

Decision: TEST_NEXT.

Both runs completed 100/100 steps and wrote reports/checkpoints. The C3-enabled run stayed far inside the 2% negative gate and the small eval-loss offset shrank over the run.

| Metric | Baseline | C3 enabled |
| --- | ---: | ---: |
| Completed steps | 100 | 100 |
| Complete | true | true |
| Parameter count | 15,070,861 | 15,300,126 |
| Last train `loss/total` | 2.951386 | 2.952055 |
| Final eval `loss/total` | 2.806852 | 2.806887 |
| Final train-eval `loss/total` | 2.886082 | 2.885851 |
| C3 tensors loaded | true | true |
| C3 conditioning enabled | false | true |

C3-enabled final eval loss delta:

- absolute: `+0.0000355`
- relative: `+0.00127%`
- kill threshold: `+2%`
- kill triggered: no

## Eval Curve

| Step | Baseline eval loss | C3-enabled eval loss | Enabled minus baseline |
| ---: | ---: | ---: | ---: |
| 20 | 3.321598 | 3.322033 | +0.000434 |
| 40 | 3.156871 | 3.157149 | +0.000278 |
| 60 | 3.015533 | 3.015660 | +0.000127 |
| 80 | 2.898683 | 2.898708 | +0.0000248 |
| 100 | 2.806852 | 2.806887 | +0.0000355 |

Both curves improve smoothly. The enabled run remains slightly above baseline on eval loss, but the gap is tiny and much smaller than in the 20-step P8 run.

## Dataset Proof

Both reports recorded:

- `include_c3_side_stream_token_tensors=true`
- `c3_side_stream_token_sidecar_path=artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- `c3_side_stream_max_tokens=256`
- `train_window_count=174373`
- `eval_window_count=142`
- `final_train_eval_window_count=16`
- `include_full_song_context=false`
- `require_control_teacher_cache=true`

## What Passed

- Exact C3 sidecar loaded in both controlled runs.
- C3 tensors traversed dataset, collate, training loop, and model boundary.
- Baseline and C3-enabled runs both completed 100 steps with finite losses.
- C3-enabled final eval loss was within 2% of baseline.
- Focused verifier suite passed: `15 passed`.

## What This Proves

P9 strengthens P8 from a 20-step canary to a 100-step small controlled comparison. It shows pooled exact-C3 conditioning remains training-stable with a larger eval slice and does not create a meaningful short-run loss regression.

It also shows the P8 eval offset did not grow. It shrank from about `+0.0230%` in P8 to about `+0.00127%` in P9.

## What This Does Not Prove

- It does not prove C3 improves mapper quality.
- It does not validate production-size MPS training.
- It does not prove pooled C3 conditioning is the right architecture.
- It does not make inference ready; incremental decode still rejects C3 conditioning.
- It does not solve cross-window `REF`/`RES` semantics for generation.

## Interpretation

The result is neutral-positive for full-pipeline readiness:

- positive: exact C3 side-stream tensors are stable through a longer real training loop.
- neutral: pooled C3 conditioning is still not measurably better than baseline on eval loss.
- surfaced risk: current C3 conditioning is a coarse per-window pooled vector, so usefulness may require an architecture that exposes token order, span type, or timing.

Recommended next step: do not promote C3 as a default mapper input yet. Either run a production-shaped MPS comparison as the final stability gate, or create a bounded architecture mutation card for ordered/timed C3 conditioning before spending larger training runtime.
