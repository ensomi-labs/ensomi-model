# C3 Mapper Conditioning Training Smoke Result Report

## Scope

This P7 pass runs the first tiny real training-loop smoke with the exact P5 C3 sidecar.

The two runs use matched small CPU configs:

- baseline: exact C3 sidecar tensors are loaded and batched, but `use_c3_side_stream_conditioning=false`,
- enabled: same data path with `use_c3_side_stream_conditioning=true`.

This is not a model-quality claim. It is a runtime/config/training-loop canary for full-pipeline C3 use.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Result

Decision: TEST_NEXT.

Both runs completed the 2-step smoke and wrote training reports/checkpoints. The enabled C3 run was effectively tied with baseline on this tiny slice and did not trigger the negative gate.

| Metric | Baseline | C3 enabled |
| --- | ---: | ---: |
| Completed steps | 2 | 2 |
| Complete | true | true |
| Parameter count | 15,070,861 | 15,300,126 |
| Last train `loss/total` | 3.309466 | 3.309666 |
| Final eval `loss/total` | 3.460522 | 3.461382 |
| Final train-eval `loss/total` | 3.327109 | 3.327249 |
| C3 tensors loaded | true | true |
| C3 conditioning enabled | false | true |

C3-enabled final eval loss delta:

- absolute: `+0.000860`,
- relative: `+0.0249%`,
- kill threshold: `+25%`,
- kill triggered: no.

## Dataset Proof

Both reports recorded:

- `include_c3_side_stream_token_tensors=true`,
- `c3_side_stream_token_sidecar_path=artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`,
- `c3_side_stream_max_tokens=256`,
- `train_window_count=174472`,
- `eval_window_count=43`,
- `final_train_eval_window_count=2`,
- `include_full_song_context=false`,
- `require_control_teacher_cache=true`.

The smoke used cached 8s control-teacher tensors and did not modify production training configs.

## What Passed

- Exact C3 sidecar JSON loaded in a real training entrypoint.
- Mapper dataset/collate emitted C3 token tensors in the training loop.
- Baseline run completed with C3 tensors carried but ignored.
- C3-enabled run completed with pooled C3 conditioning active.
- Both runs produced finite train/eval losses.
- C3-enabled loss was not materially worse than baseline on the tiny smoke.
- Focused model/training verifier suite passed: `15 passed`.

## What This Proves

P7 proves that C3 can now traverse the full minimal pipeline:

1. exact C3 sidecar artifact,
2. mapper sidecar loader,
3. mapper dataset/collate tensors,
4. mapper training loop,
5. C3-conditioned model forward,
6. optimizer step and report/checkpoint writing.

This removes the immediate full-pipeline runtime blocker.

## What This Does Not Prove

- It does not prove C3 improves validation loss.
- It does not prove the pooled C3 conditioning is the best architecture.
- It does not solve C3 cross-window `REF`/`RES` semantics.
- It does not validate production-size MPS training with C3 enabled.

## Interpretation

This is a positive smoke result. The right next step is a small controlled comparison, not a production run:

- fixed small train/eval slice,
- more than 2 steps,
- baseline versus C3-enabled,
- exact P5 sidecar,
- same seed and cached control teacher,
- report loss curves and sidecar coverage.

If that controlled run is positive or neutral, C3 can graduate to a longer training comparison. If it is negative, mutate the conditioning path toward cross-window-aware packing or cross-attention memory.
