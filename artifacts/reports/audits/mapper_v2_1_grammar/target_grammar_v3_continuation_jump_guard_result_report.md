# Target Grammar v3 Continuation-Jump Guard Result Report

## Scope

This pass implements the local gate from `target_grammar_v3_terminal_jump_guard_experiment_card.md`. It adds default-off loss plumbing for a continuation-jump guard and verifies the loss locally. It does not run the 500-step fixed-slice training gate and does not change v3 tokenization, replay, grammar, rollout, or defaults.

## Code Change

- Added `lambda_continuation_jump` and `continuation_jump_tolerance_ms` to the shared mapper loss config.
- Added `continuation_jump_guard_loss(...)`, which penalizes probability mass on time-shift tokens that skip past the next gold event by more than the configured tolerance.
- Added the loss to `MapperTupleModelLoss` only when `lambda_continuation_jump > 0`.
- Carried the new loss scalar through the v2.1 loss wrapper.
- Added v3 tests for directional behavior and default-off behavior.

## Local Loss Gate

Decision: `TEST_NEXT`.

- Directional synthetic gate: passed. Skip-over logits produce higher continuation-jump loss than local-continuation logits.
- Default-off gate: passed. `lambda_continuation_jump=0.0` leaves the new loss at zero.
- Enabled metric gate: passed. `lambda_continuation_jump=0.5` reports a positive `loss/continuation_jump` and increases total loss.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/models/mapper/shared/test_loss_contract.py -q
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/models/mapper/v3/test_event_token_smoke.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_same_ms_event_guard_summary.json >/dev/null
git diff --check
```

Observed:

- focused v3/shared loss suite: `18 passed`
- v2.1/v3 training/model regression suite: `25 passed`
- v3 replay/rollout/eval suite: `14 passed`
- JSON validation and diff whitespace checks passed.

## Passed

- The mutation is local and default-off.
- The loss uses future target information only as teacher-forced supervision, not as an inference input.
- Existing v2.1 and v3 mapper tests still pass.
- No vocabulary, grammar, replay, rollout, or tokenization changes were made.

## Surfaced

- This is only a local implementation gate. It does not prove the loss transfers to free-running rollout.
- The previous post-guard failure remains the authoritative trained-quality blocker: 32-case same-ms-guard rollout still has `22` starved cases, `9` rigid cases, and median event ratio `0.585366`.
- The next check must be a bounded 500-step fixed-slice training gate; local tests are not replacement evidence.

## Decision

Route: `TEST_NEXT`.

Run a bounded `lambda_continuation_jump` 500-step fixed-slice training gate and compare against:

- same-ms guard full32 gate: starved `22`, rigid `9`, median event ratio `0.585366`;
- 500-step baseline wide audit: starved `11`, rigid `7`, median event ratio `1.0`.

Kill this mutation if it only improves continuation by reintroducing overgeneration, max-token loops, or duplicate boundary events.
