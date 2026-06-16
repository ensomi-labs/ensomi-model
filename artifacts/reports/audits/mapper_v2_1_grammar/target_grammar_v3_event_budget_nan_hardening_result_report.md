# Target Grammar v3 Event-Budget NaN Hardening Result Report

## Scope

This report records the numerical hardening required before the event-budget training gate could run. It is a loss-stability result, not a mapper-quality result.

## Reproduction

- Interrupted run report: `artifacts/tmp/mapper_v3_event_budget_training_gate/train/run/report.json`
- Step before stop: `100`
- Last train total loss: `nan`
- Last train event-budget loss: `nan`
- Raw softmax NaN rows: `71`
- Raw softmax NaN valid rows: `0`
- Raw softmax NaN invalid rows: `71`

## Patch

- `event_budget_by_half_from_logits` now replaces invalid target rows with finite zeros before `softmax`.
- Valid rows are left untouched, so a valid all-invalid grammar row still surfaces as a real problem.
- Added a regression test for a padded all-`-inf` row.

## Verification

- Focused v3/training tests: `13 passed in 0.94s`
- Real cached batch event-budget loss: `29.321352`
- Real cached batch total loss: `6.291800`
- Post-fix 500-step final eval event-budget loss: `0.219440`

## Decision

Route: `TEST_NEXT` for numerical hardening. The objective is now trainable, so the separate event-budget training gate can be interpreted as a rollout-quality result.
