# Target Grammar v3 Bounded Fixed-Split Comparison Result Report

## Scope

This pass compares v2.1 and v3 after repeated optimizer steps on the same fixed local slice. It is a bounded train/report gate, not a final mapper-quality result and not a session-runtime replacement gate.

## Experiment Card

- Source card: `target_grammar_v3_bounded_fixed_split_comparison_experiment_card.md`
- Selected variant: 16-window fixed slice, 5 CPU steps, tiny d_model=16 configs.
- Date: `2026-06-17`

## Result

Decision: `TEST_NEXT`.

- Fixed index slice: first 16 rows from `stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`
- v2.1 report: `artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v21/run/report.json`
- v3 report: `artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v3/run/report.json`
- v2.1 completed steps: `5`
- v3 completed steps: `5`
- v2.1 final eval loss: `3.433933`
- v3 final eval loss: `4.878971`
- v3 eval loss delta: `1.445038`
- v2.1 eval valid tokens: `808`
- v3 eval valid tokens: `701`
- v3 valid-token ratio: `0.867574`
- v3 valid-token reduction: `13.24%`

## What Passed

- Both reports use the expected shared-runner contracts:
  - v2.1: `v2.1_sparse_lane_actions`
  - v3: `v3_event_groups`
- Both runs completed the required 5 steps.
- Both final eval losses are finite.
- v3 preserved a lower valid-token count after the training runner boundary.
- The comparison harness returned `TEST_NEXT`.
- Guard tests for report comparison, v3/v2.1 training, inference rollout, and mapper model surfaces passed.

## What Surfaced

- v3 still has higher per-token eval loss in this tiny run. This is not a quality conclusion because vocabulary sizes and token semantics differ, but it is a signal to watch in the longer comparison.
- The slice is narrow: only 16 adjacent windows from one early local index region, with 4 effective train/eval windows after mapper eligibility/split.
- The result supports a broader bounded run; it does not justify default replacement.

## Commands

Paired training used `run_mapper_v2_1_phase_b_training(...)` and `run_mapper_v3_phase_b_training(...)` with:

- `max_steps=5`
- `eval_every=5`
- `batch_size=1`
- `device_name='cpu'`
- tiny `d_model=16`, `layers=1`, `heads=4`
- `eval_fraction=0.25`, `eval_size=4`, `final_train_eval_size=4`

Comparison:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_training_comparison \
  --v2-1-report artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v21/run/report.json \
  --v3-report artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v3/run/report.json \
  --summary-output artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/comparison_summary.json \
  --report-output artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/comparison_report.md \
  --min-completed-steps 5
```

Guards:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Interpretation

Positive for the bounded fixed-split gate: v3 remains trainable through repeated shared-runner updates and keeps a target-token reduction in report metrics. The right next step is a broader bounded comparison using a larger fixed split and, if feasible, the real d384 config/cache path.

Still not proven: trained v3 mapper quality, convergence, full 4K runtime, trained full-song inference quality, and session-runtime/default replacement.

## Next Step

Run a broader bounded v3-vs-v2.1 trained comparison with a fixed multi-song split and comparable compute.
