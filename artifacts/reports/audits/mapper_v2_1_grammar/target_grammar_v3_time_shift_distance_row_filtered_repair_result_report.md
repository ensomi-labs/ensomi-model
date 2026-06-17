# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Result

## Mode

- Mode: executor
- Experiment Card: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_experiment_card.md`
- Date: 2026-06-17
- Route entering executor: `TEST`
- Stopped and returned to planner mode: no
- Source snapshot / evidence grade: strong local evidence from the failed tiny-gate config plus matched 80-step rerun.

## Experiment

- Root objective: keep the C3/v3 target grammar path trainable by repairing the time-shift distance auxiliary without changing tokenizer, grammar masks, replay, model architecture, or online inference assumptions.
- Selected variant: row-filter `time_shift_distance_loss` to target time-shift rows before softmax.
- Candidate variants rejected before execution: pre-mask auxiliary logits, finite sentinel clamping, disabling the auxiliary.
- Dataset slice: fixed 32-song / 256-window cache-backed tiny-gate slice.
- Baseline / comparator: commit `065ce8f` diagnostic showed full masked loss `[nan]` and row-filtered masked loss `0.07376536726951599`.
- Files changed: `src/pulsefield_model/models/mapper/shared/loss.py`, focused tests, diagnostic decision/reporting, and audit artifacts.

## Result

- Diagnostic route: `PASS_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR`
- Tiny training route: `TEST_ROLLOUT_GATE`
- Primary metric value: enabled `loss/time_shift_distance = 0.018967516809448838`, finite and positive.
- Secondary metric value: enabled completed `80/80` steps; final eval total loss `2.5338356302999174`; final eval token loss `2.448784779948032`.
- Baseline comparator: baseline completed `80/80` steps; final eval total loss `2.5338865707935243`; final eval token loss `2.448845825890419`.
- Token-loss delta: `-0.00006104594238731664`
- Total-loss delta: `-0.00005094049360687691`
- Positive signal observed: yes.
- Negative signal observed: no.
- Kill criteria triggered: no.

## Diagnostic Evidence

- target time-shift rows: `185`
- target all-nonfinite masked rows: `0`
- any-row all-nonfinite masked rows: `71`
- invalid all-nonfinite masked rows: `71`
- masked loss values after repair: `[0.07376536726951599]`
- row-filtered masked gradient abs sums: `[0.17154835164546967]`
- row-filtered masked gradient all finite: `True`

## Training Gate Evidence

All training checks passed:

- baseline and enabled mapper contracts are `v3_event_groups`
- baseline and enabled dataset contracts are `v3_event_groups`
- baseline and enabled complete flags are true
- baseline and enabled total/token losses are finite
- enabled lambda is positive
- enabled time-shift distance loss is finite and positive
- token and total losses are not materially worse
- valid-token counts match at `6366`

No rollout-pair file was supplied, so the correct route is `TEST_ROLLOUT_GATE`, not full32 escalation.

## Commands

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_time_shift_distance_nan_diagnostic.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py tests/models/mapper/v3/test_model.py -q
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_nan_diagnostic --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml --batch-limit 1 --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_diagnostic_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_diagnostic_report.md
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate --baseline-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json --enabled-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_training_gate_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_training_gate_report.md
uv run --group dev pytest tests/evals/test_mapper_v3_time_shift_distance_nan_diagnostic.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py -q
```

## Command Results

- focused guard: `34 passed`
- broader guard: `40 passed`
- baseline training: `mapper_v3_training_done steps=80 final_loss=2.533887`
- enabled training: `mapper_v3_training_done steps=80 final_loss=2.533836`
- comparator: `mapper_v3_time_shift_distance_tiny_training_gate_done route=TEST_ROLLOUT_GATE token_loss_delta=-0.000061`

## Verification / Failure Modes

- Checks performed: unit/diagnostic/model/training tests, real-batch diagnostic, matched baseline/enabled 80-step training, report comparator.
- Failed checks: none after updating tests for the repaired behavior.
- Expected failure mode avoided: padded all-`-inf` time-shift rows no longer poison the auxiliary softmax.
- Evidence gap: no rollout pairs and no full-dataset audit in this card.

## Interpretation

The result supports keeping the row-filtered implementation of `time_shift_distance_loss`. It does not prove generated quality, full32 readiness, or full-dataset replacement readiness. The next-loop action is `TEST`: run a 2-4 case rollout-pair diagnostic with full timepoint previews before considering broader C3/v3 mapper escalation.
