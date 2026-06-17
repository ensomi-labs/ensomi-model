# Target Grammar v3 CE Anti-Rigid Decode Stress Experiment Card

## Hypothesis

The event-token CE-weight checkpoint improved continuation but increased rigid-grid cases because greedy decode repeatedly selects the next canonical time-shift piece after a repeated event-spacing run. A narrow v3 anti-rigid logits transform can reduce rigid-spacing stress cases without retraining, without target leakage, and without changing the reversible v3 grammar.

## Root Objective

Move target grammar v3 closer to full-pipeline readiness by testing whether the CE-weight continuation gain can be kept while suppressing the rigid-spacing attractor that failed the previous gate.

## Idea Quality

Medium. The previous CE-weight gate gave a useful partial positive: all 32 rollouts were legal, starvation improved from `11` to `5`, and mean F1 improved, but rigid cases worsened from `7` to `11`. A decode-only stress probe is a cheap way to decide whether the issue is mostly greedy selection of canonical time shifts or requires a deeper grammar/training mutation.

## Related Work / Analogies

- Repetition penalties in autoregressive decoding.
- Constrained decoding with local token suppression.
- The existing v2.1 anti-rigid spacing transform in this repository.

No novelty is claimed. The engineering layer is v3-specific application of a known local decode-control idea to a reversible event/time-shift grammar.

## Goal Decomposition

1. Keep the CE-weight checkpoint, v3 grammar, tokenizer, training data, control checkpoint, and real-audio runtime fixed.
2. Identify only the rigid cases from the committed CE-weight 32-case result.
3. Add an opt-in v3 logits transform that suppresses or penalizes the next time-shift token that would continue a repeated event-spacing run.
4. Rerun only the rigid stress cases first.
5. Compare against the committed CE-weight rows case-by-case.

## Candidate Variants

### A. Hard-Block Anti-Rigid Transform

Block the next canonical time-shift piece after `4` identical recent generated event spacings in `[40, 400]` ms, requiring another valid time-shift alternative.

### B. Finite-Penalty Anti-Rigid Transform

Subtract a finite penalty from that same time-shift token instead of hard-blocking it.

### C. Tap-Only Anti-Rigid Transform

Apply the hard block only when the repeated run is tap-only.

### D. No Decode Mutation

Do not pursue decode-time rigid suppression; pivot to grammar-level timing diversity or v2.1 grammar improvement.

## Local Verification Matrix

| Variant | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Rigid-case stress reroll | Rigid count drops with no new dead-end/max-token/starved stress failures | Starvation or dead-end appears, or rigid count does not drop |
| B | Same stress reroll with finite penalty | Softer behavior preserves continuation better than A | Penalty too weak to change rigid cases |
| C | Same stress reroll with tap-only requirement | Avoids LN/chord damage | Does not activate on enough rigid cases |
| D | Artifact route audit | Avoids spending on decode patches | Leaves the CE-weight partial positive unactioned |

## Selected Variant

Variant A: hard-block anti-rigid transform on the CE-weight checkpoint's rigid stress cases.

## Selection Pressure

Hard block is selected for the first stress probe because the question is whether local greedy selection is the binding failure mode at all. If hard block cannot reduce rigid cases safely on the stress subset, weaker penalties are unlikely to justify another full 32-case run. If hard block reduces rigidity but creates starvation, the next mutation can be finite penalty or tap-only.

## Minimal Change

Add an opt-in v3 anti-rigid logits transform and one stress-probe evaluator. Do not change mapper defaults, tokenizer, grammar, training config, or the full runtime path unless the transform is explicitly passed to the evaluator.

## Files Likely To Change

- `src/pulsefield_model/inference/mapper_v3_rollout.py`
- `src/pulsefield_model/evals/mapper_v3_ce_antirigid_decode_stress.py`
- `tests/inference/test_mapper_v3_rollout.py`
- `tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_decode_stress_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_decode_stress_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_decode_stress_summary.json`

## Dataset Slice

Stress subset from the committed CE-weight gate:

- Source summary: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- Cases: only rows with `dominant_spacing_ratio >= 0.95` under the CE-weight checkpoint
- Expected case count: `11`
- Prefix: `0-16000ms`
- Decode: greedy, `temperature=0.0`, no top-p, no time-shift penalties, v3 anti-rigid transform enabled
- Max tokens per window: `512`

## Baseline / Comparator

Primary comparator is each matching CE-weight rollout row from `target_grammar_v3_event_token_ce_weight_training_gate_summary.json`.

Stress baseline aggregate:

- rigid cases: `11`
- starved cases: measured on the stress subset before rerun
- all legal: `true`
- dead-end cases: `0`
- max-token cases: `0`

## Primary Metric

Stress-subset rigid case count versus the CE-weight baseline stress rows.

## Secondary Metric

Starved count, mean F1@100ms, median event-count ratio, mean second-window share, transform block count, max boundary-event ratio, and any dead-end or max-token failure.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_decode_stress_summary.json >/dev/null
```

## Guard Check

- No source-default behavior changes.
- Stress summary must include every CE-weight rigid case.
- Any dead-end or max-token rollout is a hard failure.
- Starved stress cases must not increase.
- The transform must only use generated-token history and current valid-token masks, not reference targets or future context.

## Qualitative Check

Inspect the highest-rigid cases and transform examples. Improvement is weak if the transform only reduces the dominant-spacing ratio by deleting most events, shifting failures into starvation, or creating boundary artifacts.

## Positive Signal

`TEST_NEXT` if hard block reduces rigid stress cases by at least `4` with no dead-end, no max-token, and no starved-count increase.

## Negative Signal

`MUTATE` if rigidity improves but starved cases increase or F1/event count regresses materially.

## Kill Criteria

Kill hard-block anti-rigid v3 decode if it creates any dead-end/max-token case, does not reduce rigid cases by at least `2`, increases starved stress cases, or reduces mean F1 by more than `0.03` on the stress subset.

## Expected Failure Modes

- Blocking the canonical time shift may force shorter/longer shifts that still create a rigid grid.
- It may suppress continuation and revive the starvation problem CE weighting fixed.
- It may not activate because the rigid pattern is produced through compound time-shift pieces rather than the first canonical piece.
- It may reduce the stress subset but fail on the full 32-case gate.

## Expected Runtime / Runtime Budget

Expected runtime is one 11-case real-audio rollout pass plus focused tests. Stop if any of the first three stress rollouts hits dead-end or max-token, because that would reject the hard-block variant quickly.

## Confounders

The stress subset is selected from the same CE-weight training-slice result, so a positive result is only a decode-control signal. It does not prove full 4K readiness, held-out generalization, or replacement readiness.

## Result Interpretation Plan

- If positive, run the same transform on the full 32-case CE-weight fixed slice.
- If rigidity improves but starvation returns, mutate to finite penalty or tap-only mode.
- If no meaningful rigidity reduction occurs, stop decode-time anti-rigid suppression and move to grammar-level timing-diversity training or v2.1 grammar improvement.

## Result Log Template

- Command and commit.
- Stress case count and case IDs.
- Transform config and block counts.
- Baseline versus transformed aggregate.
- Per-case rigid/starved/F1/event-ratio deltas.
- Guard results.
- Decision.
- Next-loop action.

## Next-Loop Action

Use the stress result to decide whether v3 gets a full 32-case decode-control audit, a finite/tap-only mutation, or a pivot away from decode suppression toward grammar/training changes.
