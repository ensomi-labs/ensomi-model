# Target Grammar v3 Event-Budget NaN Hardening Experiment Card

## Hypothesis

The event-budget training gate failed before rollout because padded invalid decoder rows contain all `-inf` logits. `torch.softmax` over those rows returns NaN, and masking after softmax does not remove the NaN. Masking invalid rows before softmax should make the event-budget objective finite without changing valid-row behavior.

## Root Objective

Harden the v3 event-budget objective so it can be tested in the mapper training pipeline and continue the v3 replacement path with reliable evidence.

## Goal Decomposition

1. Reproduce the NaN on a real cached v3 batch.
2. Isolate whether NaNs occur only on invalid/padded rows or valid teacher-forced rows.
3. Patch event-budget loss to sanitize invalid rows before softmax.
4. Add a unit regression test with an invalid all-`-inf` row.
5. Verify the real cached batch reports finite `loss/event_budget`.
6. Rerun the original event-budget training gate from scratch only after the loss is finite.

## Candidate Variants

### A. Pre-Softmax Invalid-Row Sanitization

Replace logits for invalid target rows with finite zeros before softmax, then apply the existing half-window masks.

### B. Post-Softmax NaN Cleanup

Run `torch.nan_to_num` on probabilities after softmax.

### C. Exclude Invalid Rows By Compacting

Gather valid rows only, compute event mass, and scatter back.

## Local Verification Matrix

| Variant | Smallest Check | Pass Condition | Fail Condition |
| --- | --- | --- | --- |
| A | Synthetic invalid all-`-inf` row plus real cached batch | Finite event-budget loss and unchanged valid-row semantics | Valid rows change or real batch still NaNs |
| B | Same check | Finite loss | Silently hides valid-row NaNs |
| C | Same check | Finite loss | Adds unnecessary shape/scatter complexity |

## Selected Variant

Variant A: pre-softmax invalid-row sanitization.

## Selection Pressure

Variant A is explicit and preserves observability: valid rows still use the original logits, while invalid rows are made finite before softmax because their probability mass will be masked out. Variant B hides NaNs after the fact, and Variant C adds complexity for the same result.

## Minimal Change

- In `event_budget_by_half_from_logits`, compute the existing `valid` mask first.
- Before `torch.softmax`, replace logits on invalid rows with finite zeros.
- Keep the half-window and target-budget logic unchanged.
- Add a regression test where one padded invalid row is all `-inf`.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `tests/models/mapper/v3/test_model.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_nan_hardening_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_nan_hardening_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_nan_hardening_summary.json`

## Dataset Slice

- Synthetic unit batch with one valid event row and one invalid all-`-inf` padded row.
- Real cached first batch from the fixed 32-song/256-window v3 training slice.

## Baseline / Comparator

Current behavior from the interrupted event-budget training gate:

- `loss/event_budget=nan`
- `loss/total=nan`
- NaN probability rows: `71`
- NaN valid rows: `0`
- NaN invalid rows: `71`

## Primary Metric

`event_budget_auxiliary_loss` is finite on both the synthetic padded-row case and the real cached batch.

## Secondary Metric

Existing v3/model/training tests still pass, and the controlled matching-vs-suppressed event-budget test still prefers matching event mass.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python - <<'PY'
# real cached batch probe records finite loss/event_budget
PY
```

## Guard Check

- Do not change tokenizer, grammar, replay, runtime, decode policy, or default training config.
- Do not hide NaNs on valid rows.
- Do not change event-budget target counts.

## Qualitative Check

Inspect the real cached batch diagnostics after the patch: invalid rows may still be all-`-inf` in model logits, but event-budget loss must treat them as zero-contribution rows.

## Positive Signal

`TEST_NEXT` if the hardening test and real cached-batch probe are finite, allowing the original 500-step event-budget training gate to be rerun.

## Negative Signal

`MUTATE` if valid rows are the NaN source or if the hardening changes event-budget behavior on valid synthetic rows.

## Kill Criteria

Kill this patch if the real cached batch still produces non-finite `loss/event_budget` or if existing event-budget directionality tests fail.

## Expected Failure Modes

- A valid row can still be all-`-inf`; that would indicate a grammar/model issue outside this hardening patch.
- The loss becomes finite but the 500-step objective still fails the 32-case rollout gate.

## Expected Runtime / Runtime Budget

Expected runtime is under one minute for tests and the real cached-batch probe.

## Confounders

This hardening only fixes objective numerics. It does not prove that `lambda_event_budget=0.05` improves rollout behavior.

## Result Interpretation Plan

If hardening passes, rerun the original event-budget training gate from scratch. If hardening fails, return to planner mode and do not train.

## Result Log Template

```markdown
# Target Grammar v3 Event-Budget NaN Hardening Result Report

## Scope

## Reproduction

## Patch

## Verification

## Decision
```

## Next-Loop Action

Rerun `target_grammar_v3_event_budget_training_gate` only if this card passes.

## Closest Analogies And Novelty Layer

This is standard numerical masking for sequence losses with invalid decoder rows. It is implementation hardening, not representation or research novelty.
