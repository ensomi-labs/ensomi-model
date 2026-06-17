# Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_delta_event_head_smoke` passed default-off model/loss plumbing and recommended a bounded tiny-training auxiliary objective gate.
- Acceptance source, if any: `target_grammar_v3_delta_event_head_smoke_result_report.md`
- Source snapshot / evidence grade: strong local smoke evidence; no trainability, rollout, or full-dataset evidence yet.

## Hypothesis

The default-off v3 delta-event auxiliary objective can optimize on a tiny teacher-forced batch: event-delta, event-signature, and end-gap losses should be finite and should decrease under real optimizer steps, while preserving default-off behavior and without touching tokenizer, dataset schema, rollout, or mapper defaults.

## Root Objective

Move v3 toward a factorized target grammar that is reversible, lower-bit, teacher-forcing friendly, online-local, and not dependent on C3-style cross-window replay. This card tests whether the new auxiliary factorization is trainable before any real-data or rollout escalation.

## Goal Decomposition

- Subgoal 1: Verify default-off behavior still produces no auxiliary logits and zero auxiliary phase weight.
- Subgoal 2: Train a small enabled v3 model on a synthetic teacher-forced v3 batch with event, LN, and terminal end-gap rows.
- Subgoal 3: Confirm finite losses and nonzero label counts for event rows and end-gap rows.
- Subgoal 4: Confirm auxiliary loss, token loss, and total loss decrease over optimizer steps.

## Candidate Variants

- Variant A: self-contained synthetic overfit gate using the current v3 model/loss and a tiny fixed teacher-forced batch.
- Variant B: compare full mapper training reports, as in the time-shift-distance tiny-training gate.
- Variant C: run a real fixed-slice training job immediately.
- Variant D: skip trainability and move directly to rollout.

## Local Verification Matrix

- Variant A: pass if default-off behavior is clean, enabled loss is finite, event/end labels are positive, gradients are nonzero, and auxiliary loss decreases materially after optimizer steps.
- Variant B: reject for this card because no delta-event auxiliary training reports exist yet.
- Variant C: reject for this card because it combines trainability, dataset coverage, and runtime/cache risk before a cheap overfit check.
- Variant D: reject because rollout cannot interpret an objective that has not first shown trainability.

## Selected Variant

- Selected: Variant A.
- Rejected: B needs training artifacts that do not exist; C is too broad; D skips the required trainability gate.
- Why this is the smallest useful test: it uses current v3 code paths, performs real optimizer steps, writes reusable evidence artifacts, and runs in seconds.

## Selection Pressure

- Primary pressure: enabled delta-event auxiliary loss must decrease to at most 25% of its initial value.
- Guard pressure: default-off behavior unchanged; event/end label counts positive; losses finite; no tokenizer, dataset schema, rollout, C3 replay, or future target lookup.
- Runtime pressure: under 2 minutes for the evaluator and focused tests.
- Kill pressure: non-finite loss, no event/end labels, no auxiliary loss decrease, missing gradients, or default-off behavior regression.

## Minimal Change

Add a bounded evaluator that builds a synthetic v3 teacher-forced batch, runs a small enabled model for a fixed number of optimizer steps, records loss curves and guard checks, and writes JSON/Markdown artifacts.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_training_gate.py`
- `tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_head_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_head_smoke_result_report.md`
- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/loss.py`
- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_tiny_training_gate.py`

## Dataset Slice

Tiny synthetic teacher-forced v3 window only. No real dataset, cache, full32, or 4k run in this card.

## Baseline / Comparator

- Default-off v3 model/loss path.
- Initial enabled loss at optimizer step 0.

## Primary Metric

`final_delta_event_auxiliary_loss / initial_delta_event_auxiliary_loss`.

Pass threshold: ratio <= `0.25`.

## Secondary Metric

Total-loss ratio, token-loss ratio, nonzero event/end label counts, nonzero first-step gradients, and default-off auxiliary-logit absence.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_training_gate
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json >/dev/null
git diff --check
```

## Guard Check

No tokenizer/default/runtime rollout changes. The evaluator must report `no_rollout=true`, `dataset_schema_changed=false`, `tokenizer_changed=false`, `default_behavior_changed=false`, `uses_c3_backreference=false`, and `uses_future_lookup=false`.

## Qualitative Check

The report must state that this is a synthetic overfit/trainability gate only, not full-dataset learnability, rollout quality, or target replacement readiness.

## Positive Signal

Auxiliary loss decreases materially under optimizer steps; total/token losses also decrease or at least do not worsen; default-off behavior remains unchanged.

## Negative Signal

Loss is non-finite, auxiliary labels are missing, gradients are zero, default-off output changes, or the auxiliary loss fails to decrease.

## Kill Criteria

- `event_label_count == 0` or `end_gap_label_count == 0`.
- Any non-finite loss during the run.
- Final auxiliary-loss ratio > `0.25`.
- Default-off auxiliary logits are present.
- Evaluator requires future target lookup beyond the existing teacher-forced batch.

## Expected Failure Modes

- End-gap labels attach to the wrong terminal row.
- Delta/end-gap class ranges are too small.
- Loss decreases only by overfitting token CE while auxiliary heads do not learn.
- Auxiliary optimization conflicts with token CE even on the tiny batch.

## Expected Runtime / Runtime Budget

Under 2 minutes. Stop before any real dataset training or rollout.

## Confounders

Synthetic overfit does not prove real-data trainability, rollout quality, full-dataset label coverage, or final v3 target grammar replacement readiness.

## Result Interpretation Plan

- Positive result: create a fixed-slice label coverage or tiny real-data training gate.
- Negative result: mutate the auxiliary label extraction/head formulation before any dataset run.
- Ambiguous result: tighten diagnostics around per-head gradients and label placement.
- Human owner decides whether this auxiliary bridge remains the right route toward a real factorized target grammar.

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
- Remaining ambiguity: a passing synthetic overfit gate does not justify target replacement, rollout, or full-pipeline v3 promotion.

## Next-Loop Action

- If positive: create a fixed-slice label coverage or tiny real-data training gate.
- If negative: mutate label extraction/head formulation.
- If ambiguous: add per-head gradient/label-placement diagnostics.

## Closest Analogies and Novelty Layer

- Closest analogies: auxiliary decoder-head overfit tests and factorized timing/event symbolic music objectives.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is Pulsefield-specific trainability plumbing, not a broad novelty claim.
