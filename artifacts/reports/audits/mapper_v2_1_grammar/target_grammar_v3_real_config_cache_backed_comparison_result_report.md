# Target Grammar v3 Real-Config Cache-Backed Comparison Result Report

## Scope

This pass compares v2.1 and v3 on a deterministic real-config, cache-backed local slice. It is an infrastructure and target-length hardening gate, not a mapper-quality or convergence result.

## Experiment Card

- Source card: `target_grammar_v3_real_config_cache_backed_comparison_experiment_card.md`
- Selected variant: `32` beatmaps x `8` mapper-eligible 8s starts, d384/l4 mapper configs, d384/l3 frozen control config, existing control-teacher cache, `3` MPS optimizer steps.
- Date: `2026-06-17`

## Slice

- Base index: `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Selected index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Selected rows: `256`
- Selected beatmaps: `32`
- Control records: `256`
- Teacher-cache hits: `256 / 256`
- v2.1 mapper-eligible windows: `256 / 256`
- v3 mapper-eligible windows: `256 / 256`
- Dropped windows: `0`

## Runtime Config

- Device: `mps`
- Mapper shape: d384, 4 layers, 8 heads, global context enabled
- Control shape: d384, 3 layers, 8 heads, global memory enabled
- v2.1 parameters: `29,971,825`
- v3 parameters: `30,158,757`
- Optimizer steps: `3`
- Batch size: `2`
- Learning rate: `0.0001`
- Weight decay: `0.0`
- Train windows: `200`
- Eval windows: `56`
- Final train-eval windows: `16`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- `require_control_teacher_cache`: `true`

## Result

Decision: `TEST_NEXT`.

- v2.1 report: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/v21/run/report.json`
- v3 report: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/v3/run/report.json`
- v2.1 completed steps: `3`
- v3 completed steps: `3`
- v2.1 final eval loss: `2.810789`
- v3 final eval loss: `3.904120`
- v3 eval loss delta: `1.093331`
- v2.1 eval valid tokens: `7810`
- v3 eval valid tokens: `6366`
- v3 valid-token ratio: `0.815109`
- v3 valid-token reduction: `18.49%`

## What Passed

- The deterministic slice avoided the prior eligible-window weakness: all `256` selected rows became mapper-eligible 8s windows.
- Both reports used the expected shared-runner contracts:
  - v2.1: `v2.1_sparse_lane_actions`
  - v3: `v3_event_groups`
- Both runs loaded the real d384 frozen-control checkpoint and used the existing d384 control-teacher cache.
- Both runs completed the required `3` MPS optimizer steps.
- Both final eval losses were finite.
- v3 preserved a lower target valid-token count on the real-config slice.
- Guard tests passed:
  - comparison/training guard: `18 passed`
  - inference/model guard: `26 passed`

## What Surfaced

- v3 still has higher per-token eval loss after the short run. This remains a watch item, not a quality conclusion, because v2.1 and v3 have different vocabularies and token semantics.
- The token reduction on this real-config slice was `18.49%`, below the prior broader tiny slice (`23.16%`) but still positive and close to the full-dataset representation audit direction.
- The actual split produced `56` eval windows from `256` selected windows. The report records the actual split rather than treating the requested `eval_size=32` as the observed eval size.
- This gate proves cache-backed real-config viability; it does not prove trained v3 chart quality, convergence, or session-runtime replacement readiness.

## Commands

The paired training used `run_mapper_v2_1_phase_b_training(...)` and `run_mapper_v3_phase_b_training(...)` with:

- `max_steps=3`
- `eval_every=3`
- `save_every=3`
- `batch_size=2`
- `device_name='mps'`
- `eval_fraction=0.25`
- `eval_size=32`
- `final_train_eval_size=16`
- d384/l4 mapper config from the existing v2.1/v3 MPS config files
- d384/l3 control config from the existing v2.1/v3 MPS config files
- `require_control_teacher_cache=True`

Guards:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Interpretation

Positive for the real-config hardening gate: v3 now has evidence beyond tiny CPU runs. It can execute the same d384/l4, global-context, cache-backed MPS training path as v2.1 on a multi-mapset eligible-window slice, while preserving a shorter teacher-forcing target stream.

Still not proven: trained mapper quality, convergence under longer real-config runs, full-song generated chart quality, and session-runtime/default replacement.

## Next Step

Run a longer cache-backed v3 real-config training/inference comparison, or begin a trained full-song inference/session-runtime gate using a v3 checkpoint from a longer run.
