# Target Grammar v3 Event-Budget Objective Result Report

## Scope

This pass executes `target_grammar_v3_event_budget_objective_experiment_card.md`. It adds a default-off event-budget objective that matches expected event-token mass to teacher-forced event-token counts in the first and second half of each v3 write window.

No tokenizer, grammar, replay, rollout, dataset cache schema, or default training config was changed.

## Result

Decision: `TEST_NEXT`.

Selected variant result: `PASS_LOSS_PLUMBING_GATE`.

Reason: the objective is finite, default-off, accepted by v3 config plumbing, and directional on a controlled v3 batch.

## Controlled Values

| Metric | Value |
| --- | ---: |
| Matching event-budget loss | `0.000145` |
| Suppressed event-budget loss | `0.471844` |
| Suppressed minus matching | `0.471700` |
| Default-off event-budget loss | `0.000000` |
| Enabled event-budget loss | `0.784758` |
| Enabled minus default total loss | `0.392379` |

Target budget by half: `[[1.0, 1.0]]`

Matching predicted budget by half: `[[1.0217421054840088, 1.0103100538253784]]`

Suppressed predicted budget by half: `[[0.034296486526727676, 0.02286432310938835]]`

## What Passed

- The loss is default-off and reports `phase/lambda_event_budget` plus `loss/event_budget`.
- Matching event-token mass costs less than suppressing event tokens.
- `load_run_config` accepts `lambda_event_budget`.
- v2.1/shared mapper loss wrappers still pass with the extended loss output.

## What Surfaced

This does not prove free-running rollout quality. It only establishes that the next training gate is technically valid. The follow-up must train a fresh checkpoint and rerun the fixed 32-case rollout with strict max-token, duplicate-event, starvation, and overgeneration guards.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/models/mapper/shared/test_loss_contract.py tests/models/mapper/v2_1/test_model.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_event_token_smoke.py tests/evals/test_target_grammar_v3_event_smoke.py tests/evals/test_target_grammar_v3_pressure.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_objective_summary.json >/dev/null
```

Observed:

- focused v3/config guard: `12 passed in 0.80s`
- broad mapper loss/model guard: `31 passed in 0.84s`
- v3 grammar/rollout smoke guard: `21 passed in 0.66s`
- summary JSON validation: passed

## Next Step

Create `target_grammar_v3_event_budget_training_gate`: train a fresh 500-step v3 checkpoint with a modest `lambda_event_budget` and rerun the same fixed 32-case rollout gate used by the density calibration experiment.
