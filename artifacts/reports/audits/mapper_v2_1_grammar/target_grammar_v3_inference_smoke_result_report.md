# Target Grammar v3 Inference Smoke Result Report

## Scope

This smoke pass implements the pipeline-facing v3 inference helper after the v3 dataset/model/training and incremental decode gates. It tests the parallel v3 path only; v2.1 remains unchanged and no runtime default is switched.

## Experiment Card

- Source card: `target_grammar_v3_full_pipeline_replacement_experiment_card.md`
- Subgoal covered: v3 inference decode/expansion.
- Selected variant: parallel v3 mapper contract.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- v3 inference helper added: `src/pulsefield_model/inference/mapper_v3_rollout.py`
- lazy inference exports added: `src/pulsefield_model/inference/__init__.py`
- focused tests added: `tests/inference/test_mapper_v3_rollout.py`
- focused v3 inference tests: `5 passed`
- v3 representation/model guard: `23 passed`
- v2/v2.1 compatibility guard: `41 passed`

## What Passed

- Grammar-constrained v3 window generation emits event-group tokens and expands them to canonical timepoints.
- Generated v3 event tokens expand back to v2.1 sparse lane-action tokens for compatibility checks.
- Non-initial window generation requires a left-context token instead of silently starting from BOS.
- Prefix state tensors are rebuilt from already generated v3 tokens only.
- Incremental logits decode appends only the new prefix token on repeated calls.
- The v3 incremental call path passes `chart_end_ms` and does not pass v2.1 sparse state fields such as `emitted_lane_mask` or `last_lane_index`.
- Existing v3 and v2.1 guard suites still pass.

## What Surfaced

- This closes the first inference smoke gap in the v3 full-pipeline card, but it is still an integration smoke, not a trained-quality result.
- Full-song rollout mirrors the existing v2.1 rollout shape and keeps v3 parallel/disabled by default.
- Session runtime/default replacement remains open because no trained v3 checkpoint or end-to-end session route has been validated yet.

## Commands

```bash
uv run python -m py_compile src/pulsefield_model/inference/mapper_v3_rollout.py tests/inference/test_mapper_v3_rollout.py
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_event_token_smoke.py tests/models/mapper/v3/test_data_windows.py tests/models/mapper/v3/test_model.py tests/evals/test_target_grammar_v3_event_smoke.py tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
uv run --group dev pytest tests/models/mapper/v2/test_model.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py tests/inference/test_mapper_v2_1_rollout.py -q
```

## Interpretation

Positive for the full-pipeline replacement smoke: v3 now has a bounded inference helper that can decode online from generated-prefix replay state and convert generated event tokens back to beatmap-facing events.

Remaining evidence needed before default replacement: bounded v3 training, trained v3 inference comparison, and a session-runtime route using a real v3 checkpoint.

## Next Step

Run a bounded v3 mapper training/inference comparison against v2.1.
