# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Experiment Card

## Hypothesis

The v3 time-shift distance auxiliary failed because `time_shift_distance_loss` softmaxes every `[B,T]` row before applying `target_shift_mask`. Grammar-masked padded rows can have all `-inf` time-shift logits, so their softmax becomes NaN and `NaN * 0` contaminates the reduced loss. If the loss filters to valid target time-shift rows before softmax, the same masked candidate surface should produce finite nonzero gradients and the tiny training gate should no longer NaN.

## Minimal Code Change

In `src/pulsefield_model/models/mapper/shared/loss.py`, change `time_shift_distance_loss` to gather only rows where `target_shift_mask` is true before computing `shift_logits`, `softmax`, expected shift, and Smooth L1. Preserve the existing zero-loss path when no target time-shift rows exist. Do not change tokenizer, grammar masks, model logits, or training config.

## Dataset Slice

Use the same tiny-gate config that reproduced the failure:

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_baseline.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml`

Run the diagnostic slice first with `batch_limit=1`, then rerun the 80-step baseline/enabled tiny training gate.

## Metric

Primary:

- enabled run `loss/time_shift_distance` remains finite for all logged eval points
- enabled run `loss/total` remains finite for all logged eval points

Secondary:

- diagnostic row-filtered masked loss remains finite
- diagnostic row-filtered masked gradient is finite and nonzero
- enabled run completes the same number of steps as baseline

## Positive Signal

The repair passes if:

- the diagnostic still reports `target all-nonfinite masked rows = 0`
- row-filtered masked loss is finite with finite nonzero gradients
- enabled 80-step tiny gate completes without NaN or Inf losses
- enabled token loss is not obviously worse than the failed pre-repair run because of a new numerical issue

## Negative Signal

The repair fails if:

- `loss/time_shift_distance` is still NaN or Inf after row filtering
- gradients are zero or non-finite on target time-shift rows
- the enabled run crashes, diverges immediately, or produces non-finite total loss
- the fix requires changing grammar masks, tokenizer semantics, or training data to pass

## Kill Criteria

Kill the current time-shift distance auxiliary if a row-filtered implementation still fails the enabled tiny training gate with non-finite losses, or if the finite diagnostic cannot be reproduced on the failed config.

## Expected Runtime

Less than 5 minutes for focused unit tests, diagnostic rerun, and the 80-step tiny training gate on local CPU/MPS.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `tests/models/mapper/v3/test_model.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_summary.json`

## Current Evidence

Pre-repair diagnostic result:

- route: `TEST_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR`
- target time-shift rows: `185`
- target all-nonfinite masked rows: `0`
- any-row all-nonfinite masked rows: `71`
- invalid all-nonfinite masked rows: `71`
- masked loss values: `[nan]`
- row-filtered masked loss values: `[0.07376536726951599]`
- row-filtered masked gradient abs sums: `[0.17154835164546967]`

This supports a local loss repair over tokenizer replacement, pre-mask objective routing, or chart-level filtering.
