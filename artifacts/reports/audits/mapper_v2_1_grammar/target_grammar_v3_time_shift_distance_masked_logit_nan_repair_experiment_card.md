# Target Grammar v3 Time-Shift Distance Masked-Logit NaN Repair Experiment Card

## Hypothesis

The tiny training gate failed because `time_shift_distance_loss` reads `logits_final`, which includes the v3 grammar mask. On some target time-shift rows, all time-shift logits can be `-inf`, so the auxiliary softmax returns NaN and poisons the first optimizer step. A bounded diagnostic can confirm this and test a finite-logit repair before another training run.

## Root Objective

Repair or kill the current time-shift distance auxiliary objective after the 80-step enabled run produced NaN from the first logged step.

## Goal Decomposition

- Confirm whether real v3 training batches contain target time-shift rows where `logits_final[..., time_shift_ids]` is all non-finite.
- Compare `logits_final` against finite pre-mask logits available on the model output, such as `base_logits` plus relevant adapter biases.
- Decide whether the auxiliary should use unmasked finite logits, skip all-nonfinite rows, or be killed as mismatched with grammar-constrained decoding.

## Candidate Variants

- Variant A: diagnostic-only real-batch finite-mask audit.
- Variant B: change the auxiliary to use finite pre-mask logits for time-shift distance while keeping CE on `logits_final`.
- Variant C: keep `logits_final` but skip target rows where no finite time-shift candidate exists.
- Variant D: lower lambda or change scale only.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
| --- | --- | --- |
| A | count all-nonfinite time-shift candidate rows on one cached real batch | Required before repair. If count is zero, the NaN source is elsewhere. |
| B | synthetic and one-real-batch loss finite, gradients finite, CE path unchanged | Preferred if `logits_final` masking is the source. |
| C | finite loss but skipped-row share is reported and low | Accept only if skipped share is small and interpretable. |
| D | rerun tiny training with lower lambda/scale | Rejected for now because NaN appears before magnitude matters. |

## Selected Variant

Selected next: Variant A. Only if A confirms masked-logit all-nonfinite rows should B or C be implemented.

## Selection Pressure

The previous Stage 1 synthetic gate passed because it used finite synthetic logits. The real training gate failed immediately, so the next step must inspect the real masked-logit surface before changing lambda, scale, or rollout scope.

## Minimal Change

Add a diagnostic that loads one or a few cached v3 training batches, runs the model forward without optimizer updates, and reports:

- target time-shift row count;
- all-nonfinite `logits_final` time-shift row count;
- finite `base_logits` time-shift row count;
- gold target validity under grammar mask;
- candidate finite loss if using pre-mask logits.

No training or rollout in this repair card.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_nan_diagnostic.py`
- `tests/evals/test_mapper_v3_time_shift_distance_nan_diagnostic.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_masked_logit_nan_repair_experiment_card.md`
- diagnostic result summary/report under the same audit directory

## Dataset Slice

Use the same fixed 32-song cached v3 window records from the failed tiny gate, with `batch_limit=1` first and optional `batch_limit=8` if the first batch is inconclusive.

## Baseline / Comparator

Baseline is the failed enabled tiny gate:

- completed steps: `80`
- final eval `loss/total`: NaN
- final eval `loss/token`: NaN
- final eval `loss/time_shift_distance`: NaN
- route: `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE`

Comparator surfaces:

- `logits_final` time-shift logits;
- finite pre-mask logits;
- optional skip-all-nonfinite-row loss.

## Primary Metric

All-nonfinite target time-shift row count for `logits_final[..., time_shift_ids]`.

## Secondary Metric

- target time-shift row count;
- all-nonfinite row share;
- gold-target finite share;
- finite loss value under candidate pre-mask/skip variants;
- gradient finite check on the candidate loss.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_nan_diagnostic \
  --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml \
  --batch-limit 1 \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_masked_logit_nan_diagnostic_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_masked_logit_nan_diagnostic_result_report.md
uv run --group dev pytest tests/evals/test_mapper_v3_time_shift_distance_nan_diagnostic.py tests/models/mapper/v3/test_model.py -q
```

## Guard Check

- No optimizer step.
- No checkpoint writes.
- No tokenizer/default/decode-policy changes.
- Existing v3 loss tests still pass.

## Qualitative Check

Inspect examples of all-nonfinite rows: current ms, target token, valid time-shift ids, and whether the gold target token itself is grammar-valid.

## Positive Signal

Route `TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR` if the NaN is explained by masked logits and a finite pre-mask candidate loss has finite gradients.

## Negative Signal

Route `KILL_TIME_SHIFT_DISTANCE_DISTANCE_LOSS` if all-nonfinite rows are not the issue or every finite candidate requires skipping a large share of target time-shift rows.

## Kill Criteria

- diagnostic cannot reproduce any target time-shift rows on the real slice;
- no all-nonfinite masked rows but loss is still NaN;
- pre-mask candidate loss has non-finite gradients;
- skip-row share is too high to trust the auxiliary;
- repair requires target leakage or future context.

## Expected Failure Modes

- The real NaN source is not grammar masking but model parameters after the first NaN step.
- Pre-mask logits are finite but misaligned with legal decode candidates.
- Skipping rows hides the exact hard rows that matter for timing calibration.

## Expected Runtime / Runtime Budget

Under five minutes for `batch_limit=1`; stop before training.

## Confounders

- A single batch may not contain the same failure rows as the training run.
- Random initialization plus MPS numerical behavior may affect finite checks.
- `logits_final` is the deployed decode surface, while pre-mask logits are only a training auxiliary surface.

## Result Interpretation Plan

- `TEST_PREMASK_TIME_SHIFT_DISTANCE_REPAIR`: implement the smallest finite-logit repair and rerun the tiny training gate.
- `MUTATE_TIME_SHIFT_DISTANCE_DIAGNOSTIC`: broaden the batch slice or inspect another candidate surface.
- `KILL_TIME_SHIFT_DISTANCE_DISTANCE_LOSS`: stop this objective and return to v3 grammar repair or v2.1 grammar improvements.

## Result Log Template

```markdown
# Target Grammar v3 Time-Shift Distance Masked-Logit NaN Diagnostic Result

- batch limit:
- target time-shift rows:
- all-nonfinite masked rows:
- all-nonfinite share:
- finite pre-mask candidate loss:
- finite gradient:
- decision:
- next step:
```

## Next-Loop Action

Run the diagnostic before any further time-shift distance training or rollout.

## Closest Analogies And Novelty Layer

Closest analogies: masked-logit auxiliary loss diagnostics, invalid-action masking with auxiliary objectives, ordinal-distance loss over discrete timing buckets.

Novelty is not claimed. This is numerical/semantic hardening of a v3 training objective.
