# Target Grammar v3 Full-Pipeline Replacement Result Report

## Scope

This pass executed the existing `target_grammar_v3_full_pipeline_replacement_experiment_card.md`.

It tested the current v3 mapper surfaces as a full-pipeline smoke:

- v3 dataset/collate contract,
- v3 model forward and loss contract,
- v3 training entrypoint,
- checkpoint emission,
- runtime checkpoint loading through `ModelRuntime`,
- online v3 rollout and expansion to v2.1-compatible tokens.

This is not a chart-quality result and does not replace v2.1 defaults.

## Result

Decision: `TEST_NEXT`.

The bounded full-pipeline smoke passed, but the default full-index one-step command did not finish within the card runtime budget because it tried to build the full v3 mapper record cache.

## What Passed

- Focused v3/v2.1 verifier: `24 passed in 0.86s`.
- Bounded real-index v3 training smoke completed:
  - completed steps: `1`
  - train loss at step 1: `4.821866`
  - final eval loss: `3.974514`
  - final train loss: `3.571947`
  - checkpoint emitted: `artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run/checkpoint.pt`
  - report emitted: `artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run/report.json`
- Runtime-backed v3 inference smoke completed:
  - route: `TEST_NEXT`
  - windows: `1`
  - generated v3 tokens: `16`
  - expanded v2.1 tokens: `16`
  - completed: `true`
  - dead end: `false`
  - max tokens exceeded: `false`

## What Surfaced

The first one-step command used the full production v3 config and full index. It exceeded the 10-minute smoke budget before reaching training:

```bash
uv run python -m pulsefield_model.training.mapper_v3 \
  --config configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml \
  --output-dir artifacts/tmp/target_grammar_v3_full_pipeline_smoke \
  --run-name target_grammar_v3_full_pipeline_smoke \
  --max-steps 1 \
  --eval-every 1 \
  --save-every 1 \
  --log-every 1 \
  --batch-size 1 \
  --eval-size 1 \
  --final-train-eval-size 1 \
  --device cpu \
  --max-cached-maps 1 \
  --skip-first-eval-pass \
  --no-dataset-progress
```

Interrupt traceback showed it was still constructing `MapperV3WindowDataset` records and tokenizing the full source index. This is not a v3 model/loss failure; it is a smoke ergonomics and record-cache issue.

The bounded rerun used a temporary eight-row real index and temporary mapper record cache. That path completed quickly and exercised the same v3 training entrypoint.

## Commands

Focused verifier:

```bash
uv run --group dev pytest \
  tests/models/mapper/v3/test_event_token_smoke.py \
  tests/models/mapper/v3/test_data_windows.py \
  tests/models/mapper/v3/test_model.py \
  tests/evals/test_target_grammar_v3_event_smoke.py \
  tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
```

Observed:

```text
24 passed in 0.86s
```

Bounded training smoke:

```bash
uv run python -m pulsefield_model.training.mapper_v3 \
  --config configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml \
  --index-path artifacts/tmp/target_grammar_v3_full_pipeline_smoke/small_index.parquet \
  --output-dir artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run \
  --run-name target_grammar_v3_full_pipeline_smoke \
  --mapper-record-cache-path artifacts/tmp/target_grammar_v3_full_pipeline_smoke/mapper_v3_records.parquet \
  --max-steps 1 \
  --eval-every 1 \
  --save-every 1 \
  --log-every 1 \
  --batch-size 1 \
  --eval-size 1 \
  --final-train-eval-size 1 \
  --device cpu \
  --max-cached-maps 1 \
  --skip-first-eval-pass \
  --no-dataset-progress
```

Observed:

```text
mapper_v3_phase_b_progress step=1/1 loss=4.821866 elapsed_s=0.6 steps_per_s=1.782
mapper_v3_phase_b_eval step=1/1 loss=3.974514
mapper_v3_training_done steps=1 final_loss=3.974514 report=artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run/report.json checkpoint=artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run/checkpoint.pt
```

Runtime rollout smoke:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_trained_runtime_rollout \
  --mapper-checkpoint-path artifacts/tmp/target_grammar_v3_full_pipeline_smoke/run/checkpoint.pt \
  --control-checkpoint-path artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt \
  --summary-output artifacts/tmp/target_grammar_v3_full_pipeline_smoke/runtime_rollout_summary.json \
  --report-output artifacts/tmp/target_grammar_v3_full_pipeline_smoke/runtime_rollout_report.md \
  --device cpu \
  --chart-end-ms 1000 \
  --max-tokens-per-window 128 \
  --collect-logit-diagnostics \
  --logit-max-examples 5
```

Observed:

```text
mapper_v3_trained_runtime_rollout_done route=TEST_NEXT windows=1 tokens=16 timepoints=0
```

## Interpretation

The v3 full-pipeline smoke gate is now partially cleared:

- representation audit already passed full dataset reconstruction and token/bit reduction;
- unit/integration tests pass;
- a real-index v3 dataset/model/loss training smoke passes;
- a v3 checkpoint crosses the runtime boundary and executes online rollout.

The current blocker for broader training is not the v3 token contract. The immediate issue is operational: full production training expects a v3 mapper record cache or a long initial cache-build phase. A one-step smoke against the full index is not bounded unless that cache is prebuilt or the command uses a bounded index.

## Next Step

`TEST_NEXT`: run a bounded v3 training comparison with a prebuilt or explicitly bounded v3 mapper record cache, then compare trained v3 rollout behavior against the v2.1 baseline.

Do not replace v2.1 defaults yet. Full replacement still requires a trained v3 checkpoint with rollout quality evidence on a broader slice.
