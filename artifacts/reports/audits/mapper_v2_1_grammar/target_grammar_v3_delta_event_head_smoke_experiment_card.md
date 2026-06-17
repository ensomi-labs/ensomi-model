# Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_delta_event_proxy_audit` passed representation gates and recommended a bounded factorized delta-event model/loss plumbing card before training-scale work.
- Acceptance source, if any: `target_grammar_v3_delta_event_proxy_audit_result_report.md`
- Source snapshot / evidence grade: strong local artifact evidence on the fixed 32-song / 256-window v3 slice; no training or rollout evidence yet.

## Hypothesis

A default-off v3 auxiliary delta-event head can be plumbed from existing decoder hidden states, with labels derived from current v3 teacher-forced target fragments, without changing current v3 token logits, tokenizer behavior, rollout behavior, or training defaults. If the auxiliary head cannot produce finite losses and gradients on a tiny teacher-forced smoke batch, stop before target replacement.

## Root Objective

Move v3 toward a factorized target grammar where each event-bearing step predicts `delta_ms_to_event` and event signature directly, while preserving reversibility, teacher-forcing compatibility, online inference locality, and no complex cross-window replay.

## Goal Decomposition

- Subgoal 1: Add a model surface that can emit delta-event logits from current v3 decoder hidden states without affecting standard token logits by default.
- Subgoal 2: Derive delta-event labels from existing v3 target fragments and replay state, using only teacher-forced current target rows.
- Subgoal 3: Verify finite auxiliary loss and gradient flow through delta/event/end heads on a tiny v3 batch.
- Subgoal 4: Verify default-off behavior: current v3 model/loss output remains unchanged unless the new config is enabled.

## Candidate Variants

- Variant A: auxiliary factorized heads on current v3 decoder positions: event-delta class, event-signature class, and end-gap class, enabled by model/loss config flags.
- Variant B: replace `target_fragment_tokens` with a new delta-event target tensor immediately.
- Variant C: flat `(delta,event_signature)` auxiliary head plus end-gap head.
- Variant D: continue using current v3 token logits with time-shift-distance/event-budget losses only.

## Local Verification Matrix

- Variant A: pass if default-off logits are absent, enabled heads produce expected shapes, loss is finite, metrics report label counts, and gradients reach auxiliary heads and shared decoder parameters.
- Variant B: fail for this card because it changes the target contract before a smoke can isolate plumbing risk.
- Variant C: fail for this card because the proxy audit already showed the factorized route is safer and easier to extend than a flat target grammar.
- Variant D: fail as next step because prior decode-local repairs did not remove the standalone time-shift generation surface.

## Selected Variant

- Selected: Variant A.
- Rejected: Variant B is too broad; Variant C tests the wrong factorization first; Variant D repeats a path with negative rollout-side evidence.
- Why this is the smallest useful test: it adds no dataset schema change, no rollout change, no default behavior change, and can be verified on existing unit-test windows.

## Selection Pressure

- Primary pressure: finite factorized auxiliary loss with nonzero gradients to the new heads and existing decoder path.
- Guard pressure: default-off behavior unchanged, no tokenizer/default/runtime rollout change, no C3 backreference or future lookup.
- Runtime pressure: focused unit tests should run under 2 minutes.
- Kill pressure: missing/ambiguous labels, non-finite loss, no gradient flow, default-off output changes, or training config cannot expose the new loss weight.

## Minimal Change

Add v3-only default-off model heads and loss helpers for a teacher-forced delta-event auxiliary objective. The heads predict:

- event delta class for rows where the target token is an event;
- event signature class for rows where the target token is an event;
- terminal end-gap class for rows where the target token is `EOS`;
- no auxiliary target for pure standalone time-shift rows.

## Files Likely to Change

- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/loss.py`
- `src/pulsefield_model/training/mapper_v3.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/training/test_mapper_v3.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_head_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_head_smoke_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_proxy_audit_result_report.md`
- `src/pulsefield_model/evals/target_grammar_v3_delta_event_proxy_audit.py`
- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `src/pulsefield_model/models/mapper/v2_1/model.py`

## Dataset Slice

Tiny synthetic v3 teacher-forced unit-test windows only. No full dataset run in this card.

## Baseline / Comparator

Current v3 model/loss with `use_delta_event_auxiliary_target=False` and `lambda_delta_event_auxiliary=0.0`.

## Primary Metric

Focused smoke pass/fail:

- auxiliary loss finite and positive when enabled;
- auxiliary label count positive;
- gradients nonzero for delta/event/end heads and shared token embedding.

## Secondary Metric

Default-off output equality / absence of auxiliary logits, training config accepts new v3-only loss/model fields, and invalid config values are rejected.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/model.py src/pulsefield_model/models/mapper/v3/loss.py src/pulsefield_model/training/mapper_v3.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_head_smoke_summary.json >/dev/null
git diff --check
```

## Guard Check

The new model/loss config must be disabled by default. Existing v3 token logits, grammar masks, tokenizer outputs, and incremental rollout behavior must not change when disabled.

## Qualitative Check

The result report must state that this is an auxiliary teacher-forced plumbing smoke, not a trained target-grammar replacement or rollout result.

## Positive Signal

The enabled auxiliary path produces finite delta/event/end losses, reports nonzero event/end label counts, and backpropagates through the new heads and shared decoder path without changing default behavior.

## Negative Signal

No event/end labels can be derived, loss is non-finite, gradients do not reach heads/shared decoder, default-off behavior changes, or the training config cannot expose the new knobs.

## Kill Criteria

- Default-off v3 model output changes.
- New loss requires future target lookup outside the existing teacher-forced target row.
- New loss requires C3 backreference or cross-window replay.
- Auxiliary heads cannot receive gradients on a tiny synthetic batch.
- Loss is non-finite or silently ignores all event/end rows.

## Expected Failure Modes

- End-gap labels are ambiguous at padded rows.
- Delta labels exceed the bounded head vocabulary.
- Event-signature mapping mismatches the v3 event vocabulary order.
- The auxiliary head adds config fields but training config validation rejects them.

## Expected Runtime / Runtime Budget

Under 2 minutes for focused tests. Stop before any training run.

## Confounders

This card proves only model/loss plumbing. It does not prove learnability, decode quality, full-dataset coverage, or superiority over current v3 token CE.

## Result Interpretation Plan

- Positive result: create a bounded tiny-training gate for the auxiliary objective, still default-off.
- Negative result: mutate label extraction or keep the delta-event proxy at representation-audit status only.
- Ambiguous result: add a label-coverage audit over the fixed 256-window slice before any training work.
- Human owner decides whether this auxiliary head is the right bridge toward a real factorized target grammar.

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
- Remaining ambiguity: a passing smoke does not justify target replacement or pipeline training.

## Next-Loop Action

- If positive: create a tiny-training auxiliary objective gate.
- If negative: mutate label extraction/head shape before any training-scale run.
- If ambiguous: run a label coverage audit on the fixed 256-window slice.

## Closest Analogies and Novelty Layer

- Closest analogies: factorized timing/event symbolic music heads and auxiliary prediction heads on decoder hidden states.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is a Pulsefield v3 target-grammar plumbing experiment, not a broad novelty claim.
