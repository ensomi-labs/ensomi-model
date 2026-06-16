# Target Grammar v3 Broader Multi-Song Comparison Result Report

## Scope

This pass compares v2.1 and v3 on a deterministic multi-song local slice with repeated optimizer steps. It is a broader bounded train/report gate, not a final mapper-quality result and not a session-runtime replacement gate.

## Experiment Card

- Source card: `target_grammar_v3_broader_multisong_comparison_experiment_card.md`
- Selected variant: 4 beatmaps x 16 index rows, 10 CPU steps, tiny d_model=16 configs.
- Date: `2026-06-17`

## Selected Beatmaps

- `1000065/Billx - Punishment (Dustvoxx Remix) (_underjoy) [4K Equalizer].osu`
- `1000722/Ohara Yuiko - Zero Centimeters (TV Size) (-Mikan) [Asha's HD].osu`
- `1000722/Ohara Yuiko - Zero Centimeters (TV Size) (-Mikan) [Shini's NM].osu`
- `1001794/Phantom Sage - Holystone (UnluckyCroco) [At the Point of No Return].osu`

## Result

Decision: `TEST_NEXT`.

- Fixed index slice: `64` rows across `4` beatmaps.
- Mapper-eligible source windows: `16`
- Train windows after split: `4`
- Eval windows after split: `12`
- v2.1 report: `artifacts/tmp/mapper_v3_broader_multisong_comparison/v21/run/report.json`
- v3 report: `artifacts/tmp/mapper_v3_broader_multisong_comparison/v3/run/report.json`
- v2.1 completed steps: `10`
- v3 completed steps: `10`
- v2.1 final eval loss: `3.353752`
- v3 final eval loss: `4.564607`
- v3 eval loss delta: `1.210855`
- v2.1 eval valid tokens: `1533`
- v3 eval valid tokens: `1178`
- v3 valid-token ratio: `0.768428`
- v3 valid-token reduction: `23.16%`

## What Passed

- Both reports use the expected shared-runner contracts:
  - v2.1: `v2.1_sparse_lane_actions`
  - v3: `v3_event_groups`
- Both runs completed the required 10 steps.
- Both final eval losses are finite.
- v3 preserved a lower valid-token count on a multi-song slice.
- The comparison harness returned `TEST_NEXT`.

## What Surfaced

- The target-length advantage strengthens on this broader slice versus the prior fixed adjacent slice: `23.16%` eval valid-token reduction here versus `13.24%` there.
- v3 still has higher per-token eval loss. This remains a watch item, not a quality conclusion, because v3 and v2.1 have different vocabulary sizes and token semantics.
- The selected 64 index rows produced 16 mapper-eligible source windows after mapper filtering/splitting. A larger run should intentionally target more eligible windows and multiple mapsets.

## Commands

Paired training used `run_mapper_v2_1_phase_b_training(...)` and `run_mapper_v3_phase_b_training(...)` with:

- `max_steps=10`
- `eval_every=10`
- `batch_size=1`
- `device_name='cpu'`
- tiny `d_model=16`, `layers=1`, `heads=4`
- `eval_fraction=0.25`, `eval_size=16`, `final_train_eval_size=16`

Comparison:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_training_comparison \
  --v2-1-report artifacts/tmp/mapper_v3_broader_multisong_comparison/v21/run/report.json \
  --v3-report artifacts/tmp/mapper_v3_broader_multisong_comparison/v3/run/report.json \
  --summary-output artifacts/tmp/mapper_v3_broader_multisong_comparison/comparison_summary.json \
  --report-output artifacts/tmp/mapper_v3_broader_multisong_comparison/comparison_report.md \
  --min-completed-steps 10
```

Guards:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

- comparison/training guard: `18 passed`
- inference/model guard: `26 passed`

## Interpretation

Positive for the broader bounded gate: v3 remains trainable through repeated shared-runner updates and keeps a stronger target-token reduction on a multi-song local slice. The next evidence step should increase eligible-window and mapset diversity, and preferably use cache-backed real d384 configs if runtime allows.

Still not proven: trained v3 mapper quality, convergence, full 4K runtime, trained full-song inference quality, and session-runtime/default replacement.

## Next Step

Run a larger eligible-window comparison across more mapsets or move to a short real-config cache-backed comparison.
