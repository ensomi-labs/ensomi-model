# Target Grammar v3 Continuation-Jump Guard Experiment Card

## Hypothesis

The post same-ms-guard v3 failure is no longer a legality problem. The main remaining free-running failure is a continuation collapse: many rollouts emit a rigid 150/160ms grid through the first half of the 16s prefix, then assign high probability to a large time shift into the terminal boundary while reference events continue in the second half. A default-off continuation-jump guard loss should reduce this specific boundary-skip tendency without changing the v3 token representation, replay grammar, or online inference inputs.

## Root Objective

Move v3 from representation-ready toward full-pipeline replacement readiness by targeting the fixed-slice starvation/rigid-grid failure that survived the same-ms event guard.

## Idea Quality

Moderate. The representation gate is strong: v3 is reversible, shorter than v2.1, and online-decodable. The trained rollout gate is weak: same-ms legality is fixed, but the 32-case gate still reports `22` second-window-starved cases, `9` rigid cases, and median event ratio `0.585366`. A continuation-jump loss is a bounded mutation that directly targets the observed 8s-to-boundary skip, but it is still teacher-forced auxiliary pressure and may not transfer to free-running generation.

## Related Work / Analogies

- Scheduled sequence calibration and continuation losses in autoregressive decoding.
- Monotonic sequence models with auxiliary duration or skip penalties.
- Constrained decoding plus training-time regularizers for local failure modes.

This is engineering variation on the v3 target grammar, not representation novelty. The representation novelty remains the event-group grammar itself.

## Goal Decomposition

1. Preserve the v3 representation contract: reversible event reconstruction, lower-bit/shorter target than v2.1, no C3-style cross-window backreference, and no future target input at inference.
2. Add only default-off loss plumbing that can be trained and ablated.
3. Prove local behavior on synthetic logits: boundary-skipping time shifts should be penalized when continuation events remain.
4. Run focused model/loss tests before any 500-step fixed-slice gate.
5. If local gates pass, run a fixed-slice training gate against the same 32-case manifest and compare to the same-ms-guard and 500-step baselines.

## Candidate Variants

- Variant A: continuation-jump guard loss. Penalize probability mass on time-shift tokens that skip past the next gold event by more than a small tolerance when the gold target still contains continuation events after the current time.
- Variant B: lower global `lambda_event_budget`. Cheap, but already too indirect; it does not distinguish starvation from overgeneration or boundary jumps.
- Variant C: stronger density loss. Killed in the previous `lambda_density=0.20` gate because legality and distribution regressed.
- Variant D: decode-policy penalty. Killed by the continuation sweep because deterministic policies did not fix rigid grids before failing legality.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | Synthetic logits with a continuation target | Loss is higher for skip-over logits than for local-continuation logits | Loss cannot distinguish terminal jumps or produces NaN |
| B | 500-step retrain with smaller event-budget scalar | Fewer starved cases without overgeneration | Another scalar sweep with no causal signal |
| C | Stronger density weight | Already failed | Killed |
| D | Decode-policy sweep | Already failed | Killed |

## Selected Variant

Variant A: default-off continuation-jump guard loss.

## Selection Pressure

Variant A is the smallest mutation that targets the current observed failure rather than an already failed broad knob. It keeps inference legal because the loss only uses teacher-forced gold continuation during training; online rollout still uses only generated prefix state, audio/control context, and the v3 grammar mask.

## Minimal Change

- Add `lambda_continuation_jump` and `continuation_jump_tolerance_ms` to the shared mapper loss config, defaulting to `0.0` and a conservative timing tolerance.
- Add `continuation_jump_guard_loss(...)` over v3-style vocabularies with event and time-shift tokens.
- Include the loss in `MapperTupleModelLoss` only when enabled.
- Add v3 tests proving the loss is finite, default-off, and directional.
- Do not change tokenization, vocab, replay, grammar, model architecture, rollout, or defaults.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/shared/loss.py`
- `tests/models/mapper/v3/test_model.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_terminal_jump_guard_experiment_card.md`
- Later, if local gates pass:
  - training config under `artifacts/tmp/mapper_v3_terminal_jump_guard/`
  - result report and summary JSON under `artifacts/reports/audits/mapper_v2_1_grammar/`

## Dataset Slice

Local smoke uses synthetic v3 windows. The next runtime gate, if enabled, should reuse the fixed 32-case 16s manifest from `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`.

## Baseline / Comparator

- Same-ms guard full32 gate: legal `true`, max-token `0`, starved `22`, rigid `9`, median event ratio `0.585366`, mean F1 `0.556877`.
- 500-step baseline wide audit: legal `true`, max-token `0`, starved `11`, rigid `7`, median event ratio `1.0`, mean F1 `0.639566`.
- Event-budget 0.05 pre-guard: `MUTATE`, max-token `4`, starved `18`.

## Primary Metric

Local pass: synthetic skip-over logits have higher continuation-jump guard loss than local-continuation logits, and enabling the loss changes total loss while leaving default-off behavior unchanged.

## Secondary Metric

If trained:

- no max-token cases;
- all 32 rollouts legal;
- second-window starved cases below `11`;
- rigid cases no worse than `7`;
- median event-count ratio in `[0.80, 1.25]`;
- mean F1 no worse than the 500-step baseline by more than `0.03`.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/models/mapper/shared/test_loss_contract.py -q
```

If local tests pass, run a bounded 500-step fixed-slice train/rollout gate with `lambda_continuation_jump > 0` and compare against the same 32-case summaries.

## Guard Check

- Existing v3 tokenizer/replay/grammar tests must pass.
- Existing v2.1 loss-contract tests must pass because the loss config is shared.
- Defaults must keep the new loss disabled.
- No source path may require future target information at inference.
- No grammar or vocabulary changes are allowed in this card.

## Qualitative Check

Inspect starved cases where generated times previously ended like `[... 7700, 7860, 7990, 15950, 15990]`. A positive trained result should remove the first-half-to-boundary jump without replacing it with same-ms loops or uncontrolled overgeneration.

## Positive Signal

`TEST_NEXT` if the local loss gate passes and the 500-step gate improves second-window continuation without reintroducing max-token or duplicate boundary failures.

## Negative Signal

`MUTATE` if the loss is locally valid but training still starves, or if it improves continuation only by overgenerating easier charts.

## Kill Criteria

Kill this mutation if it breaks shared loss contracts, needs inference-time future targets, reintroduces same-time duplicate loops, or improves starvation only by exceeding the prior overgeneration/max-token guards.

## Expected Failure Modes

- The loss is too narrow and only affects teacher-forced logits, not free-running state.
- Boundary-band suppression pushes the model to dense grids just before the band.
- The guard conflicts with legitimate sparse terminal windows.
- A weight that fixes hard cases overgenerates easy charts.

## Expected Runtime / Runtime Budget

Local tests should finish under one minute. A 500-step fixed-slice training/rollout gate should be treated as the next runtime card and stopped if local tests fail.

## Confounders

The 32-case gate is small and intentionally adversarial. A pass there would justify a wider audit, not full replacement. The loss uses target continuation during teacher forcing, but that is legal only as supervision; it must not become a generation-time input.

## Result Interpretation Plan

- Local pass, no training yet: implementation gate passed; run bounded training.
- Training improves starvation with stable legality: keep the loss and run a broader fixed-slice audit.
- Training does not improve: pause scalar/auxiliary loss mutations and revisit v2.1/v3 grammar or planner-side event-budget targets.
- Training overgenerates: mutate to difficulty-conditioned or zero-reference-gated pressure.

## Result Log Template

```markdown
# Target Grammar v3 Continuation-Jump Guard Result Report

## Scope
## Code Change
## Local Loss Gate
## Fixed-Slice Training Gate
## Passed
## Surfaced
## Decision
```

## Next-Loop Action

If local tests pass, run the bounded 500-step fixed-slice training gate. If that fails, stop v3 scalar-loss calibration and compare a v2.1 grammar-improvement card against the remaining v3 failure modes.

## Closest Analogies And Novelty Layer

Closest analogies are duration/skip regularizers and continuation losses for autoregressive sequence models. This is not a novelty claim; it is a targeted engineering mutation to test whether the v3 representation can be trained out of a known free-running failure.
