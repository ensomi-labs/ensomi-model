# Target Grammar v3 Event-Budget Objective Experiment Card

## Hypothesis

The remaining v3 mapper failure is not solved by scalar density loss because density supervises a framewise lane-onset proxy, while the rollout failures are event-group count and first/second-window continuity failures. A default-off event-budget objective that matches expected event-token mass to teacher-forced event-token counts in each half-window should give the trainer a direct signal for undergeneration, overgeneration, and second-window starvation without changing the v3 grammar.

## Root Objective

Move target grammar v3 toward full-pipeline readiness while preserving the already-passed representation requirements: reversible beatmap event reconstruction, lower-bit teacher-forcing target than v2.1, no complex cross-window replay, and online inference without future target-derived inputs.

## Goal Decomposition

- Preserve the v3 tokenizer/grammar/replay contract.
- Add a direct training-time event-count/continuity signal that does not require new dataset labels.
- Keep the new objective default-off until a bounded training/rollout comparison proves it useful.
- Verify the objective behaves on controlled batches before spending training runtime.

## Candidate Variants

- A: Increase `lambda_density` again. Rejected because `lambda_density=0.20` worsened legality, starvation, and overgeneration.
- B: Add a per-half-window event-budget loss over expected event-token probability mass. Selected because it directly targets event-group count and second-window continuity while using existing teacher-forced tokens/states.
- C: Decode-time event-count clamp. Rejected for this turn because it can mask model weakness and risks adding hand-coded rollout behavior before training signal is clear.
- D: Planner-side event-budget target. Deferred because it requires planner/control target design; first verify whether a local mapper loss can learn the target.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing density=0.20 report | Starvation improves without legality loss | Already regressed |
| B | Unit loss on synthetic v3 batch | Loss is finite, default-off unchanged, wrong event budget costs more than matching budget | Shape/gradient failure or default regression |
| C | Replay-only decode clamp smoke | Rollouts legal and count improves | Quality only improves through hard clamp |
| D | Target schema card | Planner target can be computed online | Requires broader pipeline design |

## Selected Variant

B: default-off per-half event-budget loss.

## Selection Pressure

B is the smallest mutation aligned with the failure clusters. It changes only training loss plumbing and tests. It does not change tokenizer, grammar, inference, dataset cache schema, or default training configs. If it fails controlled tests, no rollout runtime is wasted.

## Minimal Change

Add `lambda_event_budget` to the shared mapper loss config and implement an `event_budget_loss` helper:

- predicted budget: sum softmax probability mass over event tokens in first half and second half of the write window;
- target budget: count teacher-forced event tokens by `current_ms` in the same halves;
- loss: Smooth L1 over the two half-window counts;
- mask: respect `target_fragment_mask`/PAD;
- default: `lambda_event_budget=0.0`.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/training/test_mapper_v3.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_objective_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_budget_objective_summary.json`

## Dataset Slice

No dataset slice for this card. Use synthetic v3 batches and existing test fixtures only. Training and 32-case rollout are the next card if this loss passes.

## Baseline / Comparator

Baseline is current default `MapperV3LossConfig` with no event-budget term. Existing failure comparator is `target_grammar_v3_event_distribution_calibration`: scalar density `0.20` failed with `all_legal=false`, starved cases `17`, overgeneration `5`, and median second-window share `0.045`.

## Primary Metric

Unit-level objective behavior:

- default-off total loss equals previous composition;
- `lambda_event_budget > 0` contributes a finite positive term;
- logits favoring the correct number of event tokens have lower event-budget loss than logits suppressing event tokens.

## Secondary Metric

Config plumbing:

- YAML/config loader accepts `lambda_event_budget`;
- metrics include `loss/event_budget` and `phase/lambda_event_budget`.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/models/mapper/shared/test_loss_contract.py tests/models/mapper/v2_1/test_model.py -q
```

## Guard Check

- Existing v3 model/loss tests still pass.
- Existing v2.1 loss/model tests still pass with default `lambda_event_budget=0.0`.
- No tokenizer, grammar, replay, rollout, or training config default changes.

## Qualitative Check

The result report must state that this is only a loss-plumbing gate and does not prove rollout readiness until a bounded training/rollout comparison is run.

## Positive Signal

All guards pass and controlled logits show lower event-budget loss when event-token mass matches the teacher-forced half-window counts.

## Negative Signal

The loss cannot distinguish matching from suppressed event budgets, destabilizes default loss behavior, or requires future/inference-only information.

## Kill Criteria

- Default mapper losses change when `lambda_event_budget=0.0`.
- Loss requires new target-derived inference inputs.
- Loss cannot be computed from existing batch tensors.
- Existing v3/v2.1 tests regress.

## Expected Failure Modes

- Event-token probability mass is too diffuse early in training.
- Half-window counts may be too coarse for timing quality.
- Matching event count may not prevent rigid 160ms grid collapse.

## Expected Runtime / Runtime Budget

Expected runtime: under 2 minutes for focused tests. Stop if focused tests fail.

## Confounders

This loss uses teacher-forced target states and tokens, so it is a training target only. It does not add inference conditioning and does not guarantee free-running event distribution improvement without a follow-up training/rollout card.

## Result Interpretation Plan

- `TEST_NEXT`: loss plumbing passes and controlled behavior is correct; run a bounded 500-step training/32-case rollout comparison with modest `lambda_event_budget`.
- `MUTATE`: loss plumbing works but controlled behavior is weak; refine count bins or add explicit first/second-window ratio target.
- `KILL`: default behavior regresses or the loss cannot be computed from existing tensors.

## Result Log Template

- command:
- default-off guard:
- positive loss value:
- matching-vs-suppressed loss delta:
- config loader result:
- test results:
- decision:
- next-loop action:

## Next-Loop Action

If positive, create `target_grammar_v3_event_budget_training_gate` to train a fresh 500-step checkpoint with a small event-budget weight and rerun the fixed 32-case rollout gate.

## Closest Analogies And Novelty Layer

Closest analogies: length/count auxiliary losses, coverage penalties, monotonic/segmented sequence budget supervision. The novelty layer here is engineering fit for Pulsefield's v3 event-group grammar, not a new sequence modeling method.
