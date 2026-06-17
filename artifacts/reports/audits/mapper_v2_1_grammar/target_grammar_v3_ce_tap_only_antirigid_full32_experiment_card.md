# Target Grammar v3 CE Tap-Only Anti-Rigid Full-32 Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: CE hard-block anti-rigid decode reduced rigidity and starvation on the full 32-case slice, but was killed because median event-count ratio inflated to `1.412331`.
- Acceptance source, if any: `target_grammar_v3_ce_antirigid_full32_result_report.md` recommended inspecting failures or mutating to finite/tap-only decoding.
- Source snapshot / evidence grade: strong evidence that broad anti-rigid blocking fixes rigid/starved symptoms; strong evidence that the same policy overgenerates; missing evidence on whether tap-only blocking narrows the policy enough.

## Hypothesis

If the event-count inflation comes from blocking repeated time-shifts in mixed/LN contexts, then requiring the repeated run to be tap-only should reduce overgeneration while preserving enough rigidity reduction. If tap-only blocking still inflates event count or fails to reduce rigidity, then the CE anti-rigid decode family should stay killed and the next work should move to training or target-grammar repair.

## Root Objective

Decide whether the only surviving CE anti-rigid decode mutation can be narrowed enough to remain useful, before spending effort on another v3 decode policy or default change.

## Goal Decomposition

- Subgoal 1: Reuse the committed CE-weight checkpoint and 32-case fixed-slice baseline.
- Subgoal 2: Run the existing anti-rigid logits transform with `require_tap_only_run=true`.
- Subgoal 3: Measure the original full-32 gate: legality, starvation, rigidity, mean F1, median event-count ratio, and boundary ratio.
- Subgoal 4: Compare against CE baseline and the killed broad anti-rigid full-32 run.
- Subgoal 5: Route to keep, mutate, or kill this decode family.

## Candidate Variants

- Variant A: Broad hard-block anti-rigid full32. Already tested and killed because median event-count ratio was too high.
- Variant B: Broad soft-penalty anti-rigid full32. Already tested and killed with the same aggregate behavior.
- Variant C: Tap-only hard-block anti-rigid full32. Selected because it is the smallest policy narrowing already supported by the runtime transform.
- Variant D: Move directly to training/target-grammar repair. Deferred until the tap-only narrowing is checked.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing full32 report | Rigid/starved improve | Event-count inflation killed it |
| B | Existing full32 report | Softer policy avoids overgen | Same aggregate failure as hard block |
| C | Existing evaluator with `--require-tap-only-run --all-cases` | Original full32 gate passes and rigidity falls | Event ratio still high or rigidity no longer improves |
| D | Synthesis-only pivot | Decode family exhausted | Tap-only branch still untested |

## Selected Variant

- Selected: Variant C, tap-only hard-block anti-rigid full32.
- Rejected: A and B because current evidence already killed them.
- Deferred: D until Variant C resolves the last recommended decode-policy mutation.
- Why this is the smallest useful test: no model, tokenizer, grammar, or training code changes are required; it only reruns the existing full32 evaluator with one stricter transform flag.

## Selection Pressure

- Primary pressure: median event-count ratio must return to the original full32 gate range `0.80..1.25`.
- Guard pressure: all 32 rollouts legal; no dead-end or max-token cases.
- Runtime pressure: one 32-case real-audio fixed-slice rerun using the existing CE checkpoint.
- Kill pressure: if tap-only fails either event-ratio or rigidity gates, stop CE anti-rigid decode policy work.

## Research Question

Can CE anti-rigid decoding be made selective enough by a tap-only repeated-run condition, or is the anti-rigid decode family intrinsically an overgeneration tradeoff on the full32 slice?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding ablation, anti-repetition decoding, finite-state decode-policy guard.
- Relevant taxonomy bucket: local verification before bounded mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: decode-policy engineering evidence for the v3 representation.

## Minimal Change

No behavior code change is required. Run the existing `mapper_v3_ce_antirigid_decode_stress` evaluator with:

- `--all-cases`;
- `--require-tap-only-run`;
- separate output directory and report paths.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_full32_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_antirigid_soft_full32_summary.json`

## Dataset Slice

All 32 cases from the committed CE-weight v3 fixed-slice gate.

## Baseline / Comparator

Baseline is CE-weight full32:

- rigid cases: `11`;
- starved cases: `5`;
- mean F1@100ms: `0.691899`;
- median event-count ratio: `1.156108`.

Rejected comparator is broad anti-rigid full32:

- rigid cases: `0`;
- starved cases: `1`;
- mean F1@100ms: `0.703179`;
- median event-count ratio: `1.412331`, which failed the gate.

## Primary Metric

Original full32 gate route from the existing evaluator:

- `TEST_NEXT` is positive;
- `KILL` or `MUTATE` is negative for this decode-policy family.

## Secondary Metric

- Median event-count ratio.
- Rigid case count.
- Starved case count.
- Mean F1@100ms.
- Max boundary ratio.
- Total transform candidates/blocks.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_ce_antirigid_decode_stress \
  --baseline-summary artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json \
  --output-dir artifacts/tmp/mapper_v3_ce_tap_only_antirigid_full32 \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_result_report.md \
  --device mps \
  --all-cases \
  --require-tap-only-run \
  --experiment-card artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_experiment_card.md
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_ce_antirigid_decode_stress.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_ce_tap_only_antirigid_full32_summary.json >/dev/null
git diff --check
```

## Qualitative Check

The result report must say whether tap-only narrowing reduced event-count inflation enough to make anti-rigid decode worth another held-out gate.

## Positive Signal

- Full32 decision route is `TEST_NEXT`.
- Median event-count ratio is within `0.80..1.25`.
- Rigid cases are no worse than original v3 baseline and preferably below CE baseline.
- Starved cases stay below the original v3 baseline.
- No legality failures.

## Negative Signal

- Median event-count ratio remains above `1.25`.
- Tap-only condition removes most blocks and fails to reduce rigidity.
- Any dead-end/max-token case appears.
- Mean F1 drops below the original floor.

## Kill Criteria

Kill this CE anti-rigid decode family if tap-only full32 fails the event-ratio gate or cannot reduce rigidity by at least four CE cases without legality failures.

## Expected Failure Modes

- Tap-only filter may produce zero or few transform candidates if repeated grids include LN/chord events.
- It may still overgenerate because blocking time shifts forces event alternatives in easy charts.
- It may improve aggregate F1 while worsening pass-like high-F1 charts.
- MPS rerun can shift exact counts slightly, but gate-level route should be stable.

## Confounders

- This is still a decode-time policy, not a learned fix.
- The transform uses generated history, so early decode errors can change later candidate eligibility.
- Full32 is not a full-dataset held-out gate.
- Event-count ratio uses reference counts and is not available as an online policy input.

## Expected Runtime / Runtime Budget

Expected runtime: under 15 minutes. Stop if the first three cases produce legality failures, matching the evaluator’s existing safety behavior.

## Result Interpretation Plan

- Positive result would suggest: one held-out/multi-slice tap-only anti-rigid gate before any default change.
- Negative result would suggest: kill CE anti-rigid decode policy and pivot to training/target-grammar repair.
- Ambiguous result would require: case-level failure inspection only if the route is `MUTATE`, not `KILL`.
- Human owner decides: whether decode policy work should continue after this last narrowing.
- Next-loop action if positive: `TEST_HELDOUT_TAP_ONLY_ANTIRIGID`.
- Next-loop action if negative: `KILL_CE_ANTIRIGID_DECODE_FAMILY`.
- Next-loop action if ambiguous: `DIAGNOSE_TAP_ONLY_CASE_FAILURES`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: whether tap-only narrowing preserves enough anti-rigid effect while fixing event-count inflation.

## Next-Loop Action

- If positive: create a held-out tap-only anti-rigid gate.
- If negative: pivot to training/target-grammar repair.
- If ambiguous: inspect case-level failures before another decode mutation.

## Novelty Notes

- Closest analogies: anti-repetition decoding and finite-state constrained decoding.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: decode-policy ablation only.
