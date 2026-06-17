# Target Grammar v3 Factorized Event-Signature Proxy Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the conditioned event-distribution short rollout gate was killed by event-count calibration: it improved starvation on the selected subset but still failed the median event-count ratio gate.
- Acceptance source, if any: `target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json`.
- Source snapshot / evidence grade: strong representation evidence for v3; strong negative evidence against simple decode and conditioned scalar objective repairs; medium evidence that the remaining issue may be target-side event-token competition.

## Hypothesis

If v3's trained event-count failure is caused partly by one high-cardinality event token competing against time-shift tokens, then a local factorization of event signatures should reduce event-symbol entropy while preserving reversibility, online legality, and lower total-bit cost than v2.1. If factorization increases total bits or does not concentrate event entropy, target-side repair should move away from event-token factorization.

## Root Objective

Decide whether the next v3 target-grammar repair should factor the current 4-lane event token before implementing a new tokenizer or model head.

## Goal Decomposition

- Subgoal 1: Estimate whether v3 event signatures have enough internal structure to justify a factorized target.
- Subgoal 2: Keep the candidate local: no cross-window replay, no future target dependency, no decode selector.
- Subgoal 3: Preserve exact reconstruction of current v3 event signatures.
- Subgoal 4: Compare total-bit proxy against current v3 and v2.1 before changing the grammar.

## Candidate Variants

- Variant A: Retune conditioned event-distribution objective weights.
- Variant B: Add another decode selector or anti-rigid decode policy.
- Variant C: Audit a factorized event-signature representation proxy.
- Variant D: Add future event-count anchor/control tokens.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Short rollout result | Safe starvation and event-ratio improvement | Route is `KILL`; median event ratio still out of range |
| B | Post-CE route synthesis and anti-rigid reports | One decode family remains open | Simple decode branches already killed or too narrow |
| C | Artifact-only factorization proxy over existing target streams | Event entropy falls and total-bit proxy remains target-competitive | Total bits worsen or reconstruction cannot be exact |
| D | Online inference contract check | Local and inference-safe | Future event-count anchors would depend on future target events |

## Selected Variant

- Selected: Variant C, factorized event-signature proxy audit.
- Rejected: Variant A because the conditioned objective short gate failed a hard event-count guard.
- Rejected: Variant B because post-CE synthesis killed simple decode families.
- Rejected: Variant D because future event anchors would violate online inference constraints.
- Why this is the smallest useful test: it is representation-side and artifact-only; it can kill or justify a grammar mutation before changing tokenizer, replay, model heads, or inference.

## Selection Pressure

- Primary pressure: reduce event-symbol entropy without giving back v3's lower-bit advantage.
- Guard pressure: exact reconstruction, local online inference, no future dependency, no C3-style side stream.
- Runtime pressure: no training; run on existing target streams and summaries.
- Kill pressure: if proxy bits are not competitive, do not implement factorized grammar.

## Research Question

Is v3's remaining quality failure better addressed by factorizing the event signature target, or is the single-token event grammar still the right target and the issue lies elsewhere?

## Closest Analogies / Novelty Layer

- Closest analogies: compound-token factorization, class-factorized categorical outputs, lane-mask/action decomposition, product-code tokenization.
- Relevant taxonomy bucket: representation repair after objective and decode ablations.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: possible representation variation, but this card only audits a proxy.

## Minimal Change

Add an artifact-only evaluator that reads existing v2.1/v3 target streams or reconstructs them through current tokenizers, then computes a reversible proxy for factorized v3 event signatures:

- current v3 event token: one token per non-empty 4-lane event signature;
- proxy factors: lane-count bucket, lane-mask bucket, and active-lane action symbols;
- reconstruction check: factors reconstruct the exact current v3 event signature;
- bit proxy: compare current v3 unigram bits, factorized-event bits plus unchanged time-shift bits, and v2.1 bit baseline.

This evaluator must not add a new runtime tokenizer or model head.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_factorized_event_signature_proxy.py`
- `tests/evals/test_mapper_v3_factorized_event_signature_proxy.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_conditioned_event_distribution_short_rollout_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- current v3 tokenizer/vocab/replay files.

## Dataset Slice

Use the largest existing v3 audit-accessible target stream first. If the full 4K token stream is not directly reusable, run the same fixed full-dataset audit input path used by the v3 full-dataset audit. Stop at a bounded subset only if the full input path is unavailable, and record the fallback.

## Baseline / Comparator

Comparators:

- current v3 full-dataset audit: exact reconstruction and lower total-bit proxy than v2.1;
- conditioned short rollout: `KILL` due `median_event_count_ratio_in_range=false`;
- CE-weight full32 gate: starvation improved but rigid cases worsened.

## Primary Metric

Route decision:

- `TEST_FACTORIZE_V3_EVENT_SIGNATURE_GRAMMAR` if the proxy is exactly reversible and total-bit proxy stays below v2.1 while event-symbol entropy meaningfully drops versus current v3 event tokens.
- `KILL_FACTORIZE_EVENT_SIGNATURE` if total bits worsen beyond v2.1 or reconstruction is not exact.
- `MUTATE_TARGET_REPAIR` if factorization helps entropy but not enough to justify a grammar change.

## Secondary Metric

- Event-token entropy before and after factorization.
- Total proxy bits versus v3 and v2.1.
- Token expansion ratio.
- Factor vocabulary sizes.
- Reconstruction mismatch count.
- Per-bucket concentration for lane masks and action symbols.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_factorized_event_signature_proxy.py -q

uv run python -m pulsefield_model.evals.mapper_v3_factorized_event_signature_proxy \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_result_report.md
```

## Guard Check

```bash
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_summary.json >/dev/null
git diff --check
```

The evaluator must also assert:

- current v3 grammar/tokenizer defaults are unchanged;
- factorization reconstructs current v3 event signatures exactly;
- no future target event is needed to emit a factor;
- no cross-window side stream is introduced.

## Qualitative Check

The result report must explain whether factorization would plausibly help trained event calibration, not only whether it compresses. If it lowers event entropy by shifting complexity into more tokens, that tradeoff must be explicit.

## Positive Signal

- Reconstruction mismatches are zero.
- Factorized event entropy is meaningfully lower than current event-token entropy.
- Total-bit proxy remains below v2.1 and is close to or better than current v3.
- Token expansion is bounded enough to remain teacher-forcing practical.

## Negative Signal

- Factorization loses exact event signature reconstruction.
- Total-bit proxy is worse than v2.1.
- Event entropy improvement is trivial after accounting for added factor tokens.
- The factorization requires future event information or cross-window replay.

## Kill Criteria

- Any reconstruction mismatch.
- Total-bit proxy not below v2.1.
- Factorized proxy bits exceed current v3 by more than a small bounded margin without a large event-entropy reduction.
- Any factor cannot be emitted online from current state.

## Expected Failure Modes

- Single-lane tap events dominate so factorization adds overhead without enough entropy gain.
- Lane-mask factors help chords but hurt sparse taps.
- LN start/end symbols make action factors too diffuse.
- Available audit artifacts do not expose enough token-level detail and require regenerating target streams.

## Confounders

- Bit proxy is not trained model quality.
- Lower entropy does not prove better rollout calibration.
- A factorized grammar may need model-head changes that are not measured here.
- Full replacement still requires full 4K audit and planner/mapper training/inference.

## Expected Runtime / Runtime Budget

Expected runtime: under a few minutes if existing audit inputs are available; otherwise stop after confirming missing inputs and write `MUTATE_INPUTS`.

No training and no real-audio rollout in this card.

## Result Interpretation Plan

- Positive result would suggest: create a bounded v3.1 factorized-event grammar implementation card.
- Negative result would suggest: kill factorized event-signature repair and move to another target-side or v2.1 grammar repair candidate.
- Ambiguous result would require: one bucket-level diagnostic, not training.
- Human owner decides: whether the entropy/bit tradeoff is worth a grammar mutation.
- Next-loop action if positive: `TEST_FACTORIZE_V3_EVENT_SIGNATURE_GRAMMAR`.
- Next-loop action if negative: `KILL_FACTORIZE_EVENT_SIGNATURE`.
- Next-loop action if ambiguous: `TEST_FACTOR_BUCKET_DIAGNOSTIC`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary route:
- Reconstruction mismatches:
- Current v3 event entropy:
- Factorized event entropy:
- Current v3 total bits:
- Factorized total bits:
- v2.1 total bits:
- Token expansion ratio:
- Factor vocabulary sizes:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
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
- Remaining ambiguity: exact factor definitions may be adjusted to match existing v3 signature helpers, but the proxy must remain exactly reversible.

## Next-Loop Action

- If positive: implement a bounded v3.1 factorized-event grammar card.
- If negative: kill event-signature factorization and move to another target repair or v2.1 grammar route.
- If ambiguous: bucket-level diagnostic only.

## Novelty Notes

- Closest analogies: compound-token factorization, lane-mask/action decomposition, class-factorized categorical targets.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is a representation proxy audit, not a novelty claim.
