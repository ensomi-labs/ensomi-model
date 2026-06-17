# Target Grammar v3 Event-Token CE Weight Experiment Card

## Hypothesis

v3 rollout failures repeatedly show event tokens under-ranked against time-shift tokens in generated state. A default-off event-token cross-entropy reweighting objective should provide a smaller, more direct training lever than the failed density, event-budget, continuation-jump, and decode-only calibration paths.

## Root Objective

Move v3 toward replacement readiness by adding one bounded training objective surface that directly targets event-token under-ranking while preserving the existing reversible v3 grammar and mapper defaults.

## Idea Quality

Medium. The failure diagnostics show a real event-vs-time-shift ranking problem, but previous stronger scalar calibration attempts produced overgeneration, starvation, or legality regressions. This card should only establish the objective plumbing and controlled local behavior before any rollout claims.

## Related Work / Analogies

- Class-weighted cross entropy for imbalanced categorical targets.
- Event/action reweighting in sequence models where non-event or delay tokens dominate.
- Cost-sensitive training as a smaller alternative to decode-time forcing.

Novelty is not claimed. This is an engineering probe over the v3 target grammar training objective.

## Goal Decomposition

- Add a default-off loss config field for event-token target weighting.
- Keep ordinary token CE exactly unchanged when the weight is `1.0`.
- Verify the weighted CE increases pressure on event-token mistakes relative to non-event targets.
- Keep v2.1/v3 model, grammar, replay, dataset, and rollout semantics unchanged.
- Produce a result report deciding whether a fresh v3 training gate is justified.

## Candidate Variants

- Variant A: global decode event bonus. Rejected because the existing margin audit killed global bonus as unsafe.
- Variant B: event-budget auxiliary loss. Already implemented and trained; the training gate worsened starvation and max-token behavior.
- Variant C: continuation-jump loss. Already trained; free-running rollout worsened rigidity and introduced dead ends.
- Variant D: event-token CE reweighting. Selected because it targets teacher-forced event-vs-time-shift ranking directly and can be tested as a default-off objective gate.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing margin report | Small safe global bias can move starved cases | Required bias too large; already killed |
| B | Existing training gate | Starvation falls without overgeneration | Starvation and max-token failures worsen |
| C | Existing training gate | Continuation improves without dead ends | Dead ends and rigid collapse worsen |
| D | Loss plumbing gate | Weighted CE is default-off, finite, directional, config-loadable | Defaults change or weighted event mistakes are not penalized more |

## Selected Variant

Variant D: add `event_token_loss_weight` to the shared mapper loss config, defaulting to `1.0`.

## Selection Pressure

This is the smallest untested v3 training lever that directly addresses event-token under-ranking. It must pass local objective behavior before spending a 500-step rollout gate.

## Minimal Code Change

Extend shared token cross entropy to accept optional class weights. In `MapperTupleModelLoss`, apply the event-token weight only to token ids decodable by `decode_event`, and only when `event_token_loss_weight != 1.0`.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `tests/models/mapper/shared/test_loss_contract.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/training/test_mapper_v3.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_summary.json`

## Dataset Slice

Synthetic v3 unit-test batches only. No real training or rollout is part of this card.

## Baseline / Comparator

Baseline is `event_token_loss_weight=1.0`, which must match the current token cross entropy behavior.

Comparator is `event_token_loss_weight>1.0` on a controlled batch where event-token logits are deliberately worse than non-event logits.

## Primary Metric

Weighted CE must be greater than default CE when event-token targets are selectively underpredicted and non-event targets are predicted correctly.

## Secondary Metric

- Loss config accepts the new field through v3 config loading.
- The loss report exposes `phase/event_token_loss_weight`.
- Existing mapper shared/v3 tests pass.
- Grammar/replay behavior is untouched.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/shared/test_loss_contract.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_summary.json >/dev/null
```

## Guard Check

- Defaults remain unchanged: `event_token_loss_weight=1.0`.
- Invalid nonpositive or nonfinite weights are rejected.
- No mapper dataset, model architecture, grammar, replay, or rollout code changes.
- No full replacement or quality claim is made from this plumbing gate.

## Qualitative Check

The result report must explicitly classify this as objective plumbing only and point the next step to a fresh v3 training gate with strict rollout guards.

## Positive Signal

- Default-off behavior passes tests.
- Weighted CE behaves directionally on a controlled batch.
- Config parsing accepts the field.
- The focused verifier suite passes.

## Negative Signal

- Default token loss changes at weight `1.0`.
- Weighted CE does not increase pressure on event-token mistakes.
- The change affects v2.1/C3 auxiliary behavior unexpectedly.

## Kill Criteria

Kill this objective if it requires grammar/model/dataset changes, if it cannot be kept default-off, or if local weighted-CE behavior is not directional.

## Expected Failure Modes

- PyTorch class weighting normalizes by weighted target mass, so tests must use mixed event and non-event targets.
- v2.1 may expose event-like tokens through shared helpers; the default value must keep behavior unchanged.
- A positive plumbing result may still fail free-running rollout by overproducing events.

## Expected Runtime

Focused tests should finish in seconds.

## Result Interpretation Plan

- If positive: create and run a v3 event-token CE reweight training gate, starting with a modest weight and 32-case rollout guards.
- If negative: do not pursue CE weighting; return to grammar-level continuation or v2.1 improvement.
- If ambiguous: add a controlled synthetic training batch before real v3 rollout work.
