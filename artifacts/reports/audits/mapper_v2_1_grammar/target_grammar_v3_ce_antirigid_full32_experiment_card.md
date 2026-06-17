# Target Grammar v3 CE Anti-Rigid Full-32 Decode Gate Experiment Card

## Hypothesis

The v3 anti-rigid hard-block transform that passed the CE rigid-stress probe can generalize from the 11 rigid stress cases to the full 32-case CE-weight fixed slice. If true, it should keep the CE-weight continuation gains while reducing rigid-grid failures enough to pass the original fixed-slice rollout gate.

## Root Objective

Move target grammar v3 closer to full-pipeline replacement readiness by testing the smallest decode-control mutation that addresses the latest blocker: CE weighting improved starvation and F1, but worsened rigid cases.

## Goal Decomposition

1. Keep the CE-weight checkpoint, v3 grammar, tokenizer, control checkpoint, real-audio runtime, and fixed 32-case manifest unchanged.
2. Enable only the opt-in `MapperV3AntiRigidSpacingLogitsTransform`.
3. Rerun all 32 fixed-slice 16s real-audio cases.
4. Compare against both the committed CE-weight result and the original 500-step v3 baseline.
5. Decide whether decode-control v3 is worth a broader training/inference gate or must mutate again.

## Candidate Variants

### A. Full-32 Hard-Block Anti-Rigid

Use the same transform settings that passed the stress probe: hard block, `min_repeated_spacings=4`, spacing range `[40, 400]` ms, require a valid alternative time-shift.

### B. Full-32 Finite Penalty

Use the same detector but subtract a finite penalty instead of blocking.

### C. Tap-Only Full-32 Hard Block

Apply the hard block only when the repeated run is tap-only.

### D. Stop Decode Control

Treat the stress pass as overfit to selected rigid cases and pivot to grammar-level timing diversity or v2.1 grammar improvement.

## Local Verification Matrix

| Variant | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Full 32-case rerun | Original gate passes: all legal, starved below 11, rigid no worse than 7, F1 not regressed | New starvation, dead-end, max-token, or overgeneration |
| B | Full 32-case finite penalty | Similar rigid reduction with lower event-count inflation | Too weak to reduce rigid cases |
| C | Full 32-case tap-only hard block | Reduces tap grids without damaging LN/chord cases | Activates too rarely or leaves rigid cases |
| D | Route audit only | Avoids decode patch complexity | Leaves a positive stress result untested |

## Selected Variant

Variant A: full 32-case hard-block anti-rigid decode.

## Selection Pressure

Variant A is selected because the stress probe passed cleanly: rigid cases dropped `11 -> 0`, starvation dropped `2 -> 1`, mean F1 improved, and no dead-end/max-token failures occurred. The full 32-case gate is the required next falsification step before considering finite/tap-only mutations or any default change.

## Minimal Change

Extend the existing stress evaluator with an `all_cases` mode and write full-32 summary/report artifacts. No model weights, tokenization, grammar, training config, or default runtime behavior should change.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_ce_antirigid_decode_stress.py`
- `tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_summary.json`

## Dataset Slice

- Source summary: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- Cases: all 32 CE-weight fixed-slice rows
- Prefix: `0-16000ms`
- Decode: greedy, `temperature=0.0`, no top-p, no time-shift penalties, v3 anti-rigid transform enabled
- Max tokens per window: `512`

## Baseline / Comparator

Primary comparator: committed CE-weight gate summary.

- all legal: `true`
- starved cases: `5`
- rigid cases: `11`
- mean F1@100ms: `0.691899`
- median event-count ratio: `1.156108`
- max boundary-event ratio: `0.041667`

Secondary comparator: original 500-step v3 baseline.

- starved cases: `11`
- rigid cases: `7`
- mean F1@100ms: `0.639566`
- median event-count ratio: `1.000000`
- max boundary-event ratio: `0.200000`

## Primary Metric

Original fixed-slice gate after transform:

- all 32 rollouts legal;
- zero dead-end cases;
- zero max-token cases;
- starved cases below original baseline `11`;
- rigid cases no worse than original baseline `7`;
- median event-count ratio in `[0.80, 1.25]`;
- max boundary-event ratio no worse than original baseline `0.200`;
- mean F1 no more than `0.03` below original baseline.

## Secondary Metric

Compare against CE-weight baseline:

- starved cases must not increase by more than `2`;
- rigid cases should drop by at least `4`;
- mean F1 should not regress by more than `0.03`;
- max event-count ratio should not exceed the CE baseline by more than `0.5`;
- transform candidate/block count and worst-case examples must be recorded.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress --all-cases --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_result_report.md --output-dir artifacts/tmp/mapper_v3_ce_antirigid_full32
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_summary.json >/dev/null
```

## Guard Check

- No source-default behavior changes.
- Summary must cover all 32 CE-weight cases.
- Any dead-end or max-token rollout is a hard failure.
- Do not claim full v3 replacement readiness from this fixed-slice gate.
- The transform must use only generated history and valid-token masks.

## Qualitative Check

Inspect newly starved cases, overgenerated cases, and the highest remaining rigid cases. Improvement is weak if rigidity drops only by event inflation or if non-rigid cases are damaged.

## Positive Signal

`TEST_NEXT` if the transformed full-32 run passes the original gate and also reduces rigid cases by at least `4` versus the CE-weight baseline without losing the CE starvation improvement.

## Negative Signal

`MUTATE` if the transform preserves legality and partly improves rigidity but introduces moderate event inflation, F1 regression, or new starvation.

## Kill Criteria

Kill full-32 hard-block anti-rigid decode if any rollout is illegal, any case hits dead-end/max-token, starved cases return to at least `11`, rigid cases remain above `7`, mean F1 regresses below `0.609566`, or median event-count ratio leaves `[0.80, 1.25]`.

## Expected Failure Modes

- The transform may pass selected rigid cases but overgenerate on non-rigid cases.
- It may shift rigid failures into starvation.
- It may reduce dominant spacing ratio while reducing timing F1.
- It may reveal that decode-control is useful only as a diagnostic, not a stable inference policy.

## Expected Runtime / Runtime Budget

Expected runtime is one 32-case real-audio rollout pass plus focused tests. Stop if the first three cases include a dead-end or max-token failure, because that would reject the hard-block variant quickly.

## Confounders

The fixed 32-case slice is still training-slice evidence. A positive result supports a broader v3 gate, not full 4K replacement readiness.

## Result Interpretation Plan

- If positive, run a broader v3 full-pipeline gate with CE weighting plus explicit anti-rigid decode-control reporting.
- If partially positive, mutate to finite penalty or tap-only mode.
- If negative, stop decode-time anti-rigid suppression and pivot to grammar/training timing diversity or v2.1 grammar improvement.

## Result Log Template

- Command and commit.
- Case count and manifest coverage.
- Baseline, CE-weight, and transformed aggregate table.
- Per-case worst failures.
- Transform block/candidate counts.
- Guard results.
- Decision.
- Next-loop action.

## Next-Loop Action

Use the full-32 result to decide whether v3 continues with this decode-control path, mutates the anti-rigid policy, or pivots away from decode suppression.

## Closest Analogies And Novelty Layer

Closest analogies are repetition penalties and constrained decoding. The novelty layer, if any, is not algorithmic novelty; it is whether this local decode-control policy is a useful engineering bridge for Pulsefield's reversible v3 event/time-shift grammar.
