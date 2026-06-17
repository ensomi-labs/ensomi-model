# Target Grammar v3 Event-Token CE Weight Result Report

## Scope

This pass implements `target_grammar_v3_event_token_ce_weight_experiment_card.md`: a default-off event-token cross-entropy reweighting objective for shared mapper losses, intended first for v3 training gates.

It does not change v3 tokenization, grammar, replay, rollout, dataset schemas, model architecture, or production training configs.

## Result

Decision: `TEST_NEXT`.

Selected variant result: `PASS_OBJECTIVE_PLUMBING_GATE`.

Reason: the objective is default-off, finite, directional on a controlled v3 batch, accepted by v3 config loading, and covered by focused mapper tests.

## Implemented Contract

Added to `MapperTupleLossConfig`:

- `event_token_loss_weight: float = 1.0`

When the value is `1.0`, token cross entropy follows the prior unweighted path. When the value is greater than `1.0`, token ids decodable by `decode_event` receive that class weight in token CE.

The metric output now records:

- `phase/event_token_loss_weight`

Invalid nonpositive weights are rejected.

## Controlled Directional Check

The controlled batch has one underpredicted v3 event-token target and one correctly predicted time-shift target.

| event_token_loss_weight | token loss | total loss |
| ---: | ---: | ---: |
| `1.0` | `5.012589` | `5.012589` |
| `2.0` | `6.679255` | `6.679255` |
| `4.0` | `8.012589` | `8.012589` |

This passes the local direction gate: event-token mistakes receive more pressure without adding an auxiliary loss term.

## What Passed

- Default value remains `1.0`.
- Weighted CE is only constructed when the configured weight differs from `1.0` and event token ids are available.
- The v3 config loader accepts `event_token_loss_weight`.
- `MapperV3ModelLoss` reports `phase/event_token_loss_weight`.
- Nonpositive weights are rejected.
- Existing focused mapper shared/v3/training tests pass.

## What Surfaced

This is objective plumbing only. It does not prove v3 rollout quality. Because previous event-budget, density, continuation-jump, and decode-only variants caused starvation, overgeneration, or legality regressions, the follow-up training gate must be conservative and rollout-backed.

The likely failure mode is overproducing event tokens or creating boundary duplicates. The same-ms event guard should remain in place, but the gate must still check max-token cases, duplicate/nonincreasing spacing, second-window starvation, event-count ratio, and F1.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/models/mapper/shared/test_loss_contract.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python - <<'PY'
...
PY
```

Observed:

- focused verifier suite: `26 passed in 0.82s`
- controlled batch losses: `5.012589 -> 6.679255 -> 8.012589` as event-token weight increased from `1.0` to `2.0` to `4.0`

## Next Step

Create and run `target_grammar_v3_event_token_ce_weight_training_gate`:

- baseline: current v3 500-step setup with `event_token_loss_weight=1.0`;
- variant: modest event-token CE weight, starting at `2.0`;
- dataset: same fixed 32-case rollout gate used by recent v3 audits;
- guards: all rollouts legal, no max-token cases, no duplicate/nonincreasing regression, starved cases below baseline `11`, overgeneration no worse than baseline `2`, mean F1 no more than `0.03` below baseline.

Do not replace v2.1 defaults before that rollout-backed gate passes.
