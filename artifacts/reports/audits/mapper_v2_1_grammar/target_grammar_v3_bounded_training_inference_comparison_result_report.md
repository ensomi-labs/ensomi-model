# Target Grammar v3 Bounded Training/Inference Comparison Result Report

## Scope

This pass adds and verifies the parallel v3 Phase B training surface. It tests trainability, checkpoint/report plumbing, and compatibility with the existing v3 inference/model smoke path. It does not switch session defaults and does not claim trained v3 quality.

## Experiment Card

- Source card: `target_grammar_v3_bounded_training_inference_comparison_experiment_card.md`
- Selected variant: parallel v3 training CLI/function using the shared mapper runner.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- v3 training function/CLI added: `src/pulsefield_model/training/mapper_v3.py`
- v3 training config added: `configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml`
- focused v3 training tests added: `tests/training/test_mapper_v3.py`
- exact v3 card verification: `14 passed`
- exact v2.1 comparator verification: `26 passed`
- broader v3 smoke guard: `21 passed`
- broader v2/v2.1 guard: `31 passed` and `17 passed`
- real tiny v3 training smoke: `completed_steps=1`
- tiny smoke train loss: `4.683746814727783`
- tiny smoke eval loss: `4.679501953125`
- tiny smoke valid target tokens: `125`
- tiny smoke checkpoint: `artifacts/tmp/mapper_v3_training_smoke/run/checkpoint.pt`
- tiny smoke report: `artifacts/tmp/mapper_v3_training_smoke/run/report.json`

## What Passed

- `mapper_v3` now has a public `run_mapper_v3_phase_b_training(...)` function.
- `python -m pulsefield_model.training.mapper_v3` now exposes a v3 Phase B CLI.
- The v3 CLI loads a v3-specific config and forwards legal Phase B options.
- The v3 training path uses `MapperV3WindowDataset` and `collate_mapper_v3_windows`.
- The v3 training path reuses the shared mapper runner instead of adding a bespoke loop.
- The generated training config/report records `mapper_token_contract: v3_event_groups`.
- A one-step CPU smoke on a tiny real local index wrote a checkpoint and report.
- v3 model/inference tests and v2.1 comparator tests remained green.

## What Surfaced

- Resume is still intentionally unsupported for v3 and is rejected before dataset construction.
- A full 4K v3 training run should use the new v3 mapper-record cache path; the tiny smoke avoided full-cache build by writing a tiny index.
- The first tiny smoke attempt exposed two run-configuration constraints rather than code defects:
  - dataset root must remain `dataset` because index rows include shard `0`,
  - smoke `max_seq_len` must exceed observed v3 target length; the tiny sample required at least `125`.
- This pass proves runner/checkpoint/report wiring, not trained mapper quality or session-runtime replacement.

## Commands

```bash
uv run python -m py_compile src/pulsefield_model/training/mapper_v3.py tests/training/test_mapper_v3.py
uv run --group dev pytest tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py tests/inference/test_mapper_v2_1_rollout.py -q
uv run --group dev pytest tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py tests/models/mapper/v3/test_data_windows.py tests/models/mapper/v3/test_event_token_smoke.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/inference/test_mapper_v2_1_rollout.py -q
uv run --group dev pytest tests/models/mapper/v2/test_model.py tests/evals/test_target_grammar_v3_event_smoke.py tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
```

Tiny real training smoke:

```bash
uv run python - <<'PY'
# Builds artifacts/tmp/mapper_v3_training_smoke/tiny_index.parquet from two local index rows,
# then calls run_mapper_v3_phase_b_training(... max_steps=1, device_name='cpu',
# tiny d_model=16 configs, output_dir=artifacts/tmp/mapper_v3_training_smoke/run).
PY
```

## Interpretation

Positive for the bounded training surface: v3 can now be trained through the shared runner and can emit a checkpoint/report under the `v3_event_groups` contract. This satisfies the missing training-run smoke layer from the full-pipeline replacement card.

Still not proven: v3 trained quality, convergence, full 4K training runtime, trained full-song inference quality, and session-runtime/default replacement.

## Next Step

Run a real bounded v3-vs-v2.1 trained comparison with comparable configs/checkpoints, then wire a session-runtime v3 route only if the trained evidence is acceptable.
