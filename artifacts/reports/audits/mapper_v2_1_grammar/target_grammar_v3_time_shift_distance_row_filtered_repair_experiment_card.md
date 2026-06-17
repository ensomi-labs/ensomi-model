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

## Mode

- Mode: executor
- Route: `TEST`
- Source idea: repair the C3/v3 time-shift distance auxiliary after the tiny training gate produced NaN from the first logged step.
- Acceptance source: pre-repair masked-logit NaN diagnostic committed in `065ce8f`.
- Source snapshot / evidence grade: strong local evidence from real failed config, one batch, no optimizer step.

## Root Objective

Keep the target grammar v3/C3 path moving toward full-pipeline usability by making the already-selected time-shift distance auxiliary numerically trainable without changing tokenization, grammar masks, replay assumptions, or online inference constraints.

## Goal Decomposition

- Subgoal 1: prove the NaN source is loss-row handling, not target token infeasibility.
- Subgoal 2: preserve masked grammar semantics and teacher-forcing target semantics.
- Subgoal 3: rerun the failed tiny training gate and require finite enabled loss traces.

## Candidate Variants

- Variant A: row-filter `time_shift_distance_loss` to target time-shift rows before softmax.
- Variant B: feed pre-mask logits to the auxiliary while keeping token CE on masked logits.
- Variant C: clamp all-`-inf` shift rows to finite sentinels before softmax.
- Variant D: disable the auxiliary and return to tokenizer/grammar changes.

## Local Verification Matrix

- Variant A: pass if row-filtered masked loss is finite with nonzero finite gradients and the enabled tiny gate has finite losses.
- Variant B: pass only if target masked rows are truly infeasible; diagnostic shows they are not, so this is larger than needed.
- Variant C: rejected unless A fails because it changes non-target probability surfaces instead of removing irrelevant rows.
- Variant D: rejected unless the local repair still NaNs or gives zero/non-finite gradients.

## Selected Variant

- Selected: Variant A, row-filter inside `time_shift_distance_loss`.
- Rejected: B, C, and D for being broader than the observed fault.
- Why this is the smallest useful test: it only changes the auxiliary loss computation order on rows that already define the metric.

## Selection Pressure

- Primary pressure: finite `loss/time_shift_distance` and `loss/total` in the enabled tiny training gate.
- Guard pressure: existing v3 model/loss tests and diagnostic tests remain green.
- Runtime pressure: finish diagnostic plus 80-step tiny gate locally in minutes.
- Kill pressure: if row filtering still NaNs, stop this auxiliary rather than adding larger repairs.

## Research Question

Can the current C3/v3 grammar-masked target surface support a time-shift distance auxiliary if the auxiliary softmax is restricted to the rows where that auxiliary is defined?

## Closest Analogies / Novelty Layer

- Closest analogies: masked sequence loss row filtering; ignore-index cross entropy; auxiliary loss masking before normalization.
- Relevant taxonomy bucket: representation-to-training-objective interface audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering variation in the loss implementation, not representation novelty.

## Read-Only Context Files

- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_nan_diagnostic.py`
- `tests/evals/test_mapper_v3_time_shift_distance_nan_diagnostic.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_masked_logit_nan_diagnostic_result_report.md`

## Baseline / Comparator

Baseline is commit `065ce8f`, where the diagnostic reports full masked loss `[nan]` and row-filtered masked loss `[0.07376536726951599]` on the failed enabled tiny-gate config.

## Verify Command / Evaluation Procedure

1. Run focused loss/model/diagnostic tests.
2. Rerun masked-logit NaN diagnostic on the failed enabled config.
3. Rerun the baseline/enabled 80-step tiny training gate.
4. Write a result report and summary JSON.

## Guard Check

The guard is that no tokenizer, grammar mask, model architecture, or dataset code changes are required for the repair to pass.

## Qualitative Check

Inspect the result report for a clean causal story: target time-shift rows remain finite, padded rows no longer poison the auxiliary, and the enabled tiny gate finishes.

## Expected Failure Modes

- The row-filtered loss is finite in isolation but training still NaNs from another loss component.
- The row-filtered loss has zero gradients because selected rows or shift values are mishandled.
- MPS/device behavior differs from CPU test behavior.

## Confounders

- Tiny-gate loss comparisons are not a quality result; they only prove numerical viability.
- One-batch diagnostic does not prove full-dataset stability.
- The auxiliary can be numerically stable while still not improving mapper quality.

## Result Interpretation Plan

- Positive result would suggest: keep the local row-filtered repair and move to broader C3/v3 training gates.
- Negative result would suggest: kill or reformulate this auxiliary before spending more training time.
- Ambiguous result would require: isolate whether a different loss component or device path causes the remaining NaN.
- Human owner decides: whether the auxiliary remains part of the C3/v3 path.
- Next-loop action if positive: run a broader v3 mapper/planner gate using the repaired loss.
- Next-loop action if negative: return `KILL` or `MUTATE` for this auxiliary.
- Next-loop action if ambiguous: add a narrower diagnostic for the new failure mode.

## Result Log Template

- Experiment: Target grammar v3 time-shift distance row-filtered repair
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: no
- Remaining ambiguity: full-dataset and downstream quality are out of scope for this repair card.

## Next-Loop Action

- If positive: `TEST` a broader repaired-loss v3 mapper gate.
- If negative: `KILL` the current distance auxiliary or `MUTATE` to a different objective.
- If ambiguous: `MUTATE` into a narrower diagnostic card.

## Novelty Notes

- Closest analogies: ignore-index sequence losses and row-filtered masked auxiliary losses.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation only.
