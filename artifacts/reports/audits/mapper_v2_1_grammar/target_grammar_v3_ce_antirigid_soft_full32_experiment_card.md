# Target Grammar v3 CE Anti-Rigid Soft Full-32 Decode Gate Experiment Card

## Hypothesis

The hard-block v3 anti-rigid transform failed full-32 because it forced alternative time shifts too aggressively, raising median event-count ratio to `1.412331`. A finite penalty on the same repeated-spacing token can reduce rigid-grid cases while avoiding hard-block event inflation.

## Root Objective

Move target grammar v3 closer to full-pipeline replacement readiness by testing whether the CE continuation gain and anti-rigid signal can coexist inside the original fixed-slice rollout envelope.

## Goal Decomposition

1. Keep the CE-weight checkpoint, v3 grammar, tokenizer, training data, control checkpoint, manifest, and real-audio runtime fixed.
2. Replace hard-block anti-rigid decode with a finite penalty using the same detector.
3. Rerun all 32 fixed-slice cases.
4. Compare against CE-weight baseline and hard-block full32 result.
5. Decide whether decode-control remains viable or should be deprioritized in favor of grammar/training timing diversity or v2.1 grammar work.

## Candidate Variants

### A. Soft Penalty 4.0

Subtract `4.0` from the repeated-spacing canonical time-shift token after `4` identical generated event spacings in `[40, 400]` ms.

### B. Soft Penalty 2.0

Weaker penalty; less risk of event inflation, higher risk of no rigid improvement.

### C. Soft Penalty 6.0

Stronger penalty; more likely to reduce rigid grids, closer to hard-block overproduction risk.

### D. Tap-Only Hard Block

Restrict the hard block to tap-only repeated runs.

## Local Verification Matrix

| Variant | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Full 32-case rerun | Rigid cases drop while median event ratio returns to `[0.80, 1.25]` | Rigid remains high or event ratio still exceeds gate |
| B | Full 32-case rerun | Preserves event ratio with some rigid reduction | Too weak to move the failure |
| C | Full 32-case rerun | Stronger rigid reduction without hard-block inflation | Repeats hard-block overgeneration |
| D | Full 32-case rerun | Avoids LN/chord damage | Activates too narrowly |

## Selected Variant

Variant A: soft penalty `4.0`.

## Selection Pressure

Penalty `4.0` is selected as the middle finite-pressure test. Hard block already proved the detector can remove rigid grids but overproduces. A weak penalty may simply reproduce CE baseline rigidity. The middle value is the fastest falsification of whether finite decode pressure can thread the gate before spending another loop on a sweep.

## Minimal Change

No source-default behavior change. Reuse the existing opt-in v3 anti-rigid transform in soft-penalty mode and write result artifacts for the full32 gate.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_ce_antirigid_decode_stress.py`
- `tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_summary.json`

## Dataset Slice

- Source summary: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- Cases: all 32 CE-weight fixed-slice rows
- Prefix: `0-16000ms`
- Decode: greedy, `temperature=0.0`, no top-p, no time-shift penalties, v3 anti-rigid finite penalty enabled
- Penalty: `4.0`
- Max tokens per window: `512`

## Baseline / Comparator

Primary comparator: CE-weight gate.

- starved cases: `5`
- rigid cases: `11`
- mean F1@100ms: `0.691899`
- median event-count ratio: `1.156108`

Negative comparator: hard-block full32.

- starved cases: `1`
- rigid cases: `0`
- mean F1@100ms: `0.703179`
- median event-count ratio: `1.412331`
- decision: `KILL`

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

Compared with CE baseline:

- rigid cases reduced by at least `4`;
- starved cases do not increase by more than `2`;
- mean F1 does not regress by more than `0.03`;
- median event-count ratio does not exceed `1.25`;
- transform candidate/penalty count recorded.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress --all-cases --soft-penalty --penalty 4.0 --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_result_report.md --output-dir artifacts/tmp/mapper_v3_ce_antirigid_soft_full32 --experiment-card artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_experiment_card.md
uv run --group dev pytest tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_summary.json >/dev/null
```

## Guard Check

- No default decode behavior changes.
- Summary must cover all 32 CE-weight cases.
- Any dead-end or max-token rollout is a hard failure.
- Do not claim full v3 replacement readiness from this fixed-slice gate.

## Qualitative Check

Inspect highest remaining rigid cases and event-ratio outliers. A positive-looking rigid reduction is weak if it comes from dense event overgeneration.

## Positive Signal

`TEST_NEXT` if the soft full32 run passes the original gate and reduces CE rigid cases by at least `4`.

## Negative Signal

`MUTATE` if it improves rigidity but misses one non-hard gate without dead-end/max-token failure.

## Kill Criteria

Kill soft penalty `4.0` if rigid cases stay above `7`, median event-count ratio leaves `[0.80, 1.25]`, starved cases reach at least `11`, mean F1 drops below `0.609566`, or any rollout is illegal/dead-end/max-token.

## Expected Failure Modes

- Penalty too weak: CE baseline rigidity remains.
- Penalty still too strong: hard-block-like overproduction remains.
- Penalty creates mixed behavior: fewer rigid cases but F1 or event ratio fails.

## Expected Runtime / Runtime Budget

Expected runtime is one 32-case real-audio rollout pass plus focused tests. Stop if the first three cases include dead-end or max-token failure.

## Confounders

This is fixed-slice training evidence, not held-out or full 4K readiness. Passing this card would only justify a broader v3 gate.

## Result Interpretation Plan

- If positive, run a broader v3 inference gate with CE weighting plus finite anti-rigid decode.
- If too weak, consider penalty `6.0` only if event ratio remains safe.
- If too strong, consider penalty `2.0` or tap-only hard block.
- If no finite penalty can thread the gate, stop decode-control and pivot to grammar/training timing diversity or v2.1 grammar improvement.

## Result Log Template

- Command and commit.
- Full32 aggregate table versus CE and hard-block.
- Guard results.
- Transform count.
- Worst cases.
- Decision.
- Next-loop action.

## Next-Loop Action

Use the result to decide whether decode-control remains viable or should be killed/deprioritized.

## Closest Analogies And Novelty Layer

Closest analogies are repetition penalties and constrained decoding. No algorithmic novelty is claimed; this tests an engineering decode-control layer for the v3 event/time-shift grammar.
