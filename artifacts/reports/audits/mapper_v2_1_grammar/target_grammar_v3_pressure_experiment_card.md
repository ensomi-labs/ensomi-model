# Target Grammar v3 Pressure Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: C3 is a strong codec-side result, but P20-P22 show diminishing marginal returns for C3 as a mapper-side auxiliary target. The updated objective asks for a new target grammar v3 if C3 stalls.
- Acceptance source, if any: active full-pipeline tokenization goal updated with the v3 requirements.
- Source snapshot / evidence grade: strong evidence for C3 codec usefulness; medium/negative evidence for C3 auxiliary promotion; unmeasured evidence for whether v2.1 target grammar can be compressed into a reversible v3 teacher-forcing grammar.

## Hypothesis

A v3 event-group target grammar can encode the same beatmap event stream as v2.1 with fewer estimated bits by replacing same-time sparse lane-action runs with learned group tokens plus sparse fallback, while keeping time shifts ordered and avoiding C3-style cross-window references.

## Root Objective

Identify whether a target grammar v3 is worth implementing for the full pipeline:

- reversible reconstruction of beatmap events,
- lower estimated bits than v2.1,
- no complex cross-window replay dependency,
- direct teacher-forcing target,
- online inference does not require future target tokens.

## Goal Decomposition

- Subgoal 1: Extract legal v2.1 target token streams from the current mapper dataset.
- Subgoal 2: Convert same-time lane-action runs into reversible event-group signatures.
- Subgoal 3: Simulate top-K v3 group-token dictionaries with sparse fallback.
- Subgoal 4: Compare v3 candidates against v2.1 on token count, unigram NLL bits, group coverage, and reconstruction guard.

## Candidate Variants

- Variant A: Top-K event-group dictionary with sparse v2.1 lane-action fallback.
- Variant B: Full observed group-token vocabulary with no sparse fallback.
- Variant C: Direct factorized mask grammar with separate tap/start/end mask tokens.
- Variant D: Continue C3 RAW/factor auxiliary probing.

## Local Verification Matrix

- Variant A: Passes if it reduces bits/tokens while preserving exact v2.1 token reconstruction through fallback; bounded and reversible.
- Variant B: Likely lower tokens but risks open vocabulary and weak train/eval coverage.
- Variant C: Promising but requires a larger grammar design before proving group-token pressure.
- Variant D: Rejected for this turn because P20-P22 already show diminishing returns on C3 auxiliary targets.

## Selected Variant

- Selected: Variant A, top-K event-group dictionary with sparse fallback.
- Rejected: B/C/D for scope or because they require more design before this pressure test.
- Why this is the smallest useful test: it tests the v3 requirements without changing tokenizer, replay, grammar masks, training, or inference code.

## Selection Pressure

- Primary pressure: top-128 v3 candidate should reduce eval estimated unigram NLL bits by at least 5% or total target tokens by at least 10%.
- Guard pressure: candidate stream must reconstruct the exact v2.1 sparse token stream with zero mismatches.
- Runtime pressure: use first `4096` train windows and full P19 eval split; expected under 2 minutes.
- Kill pressure: if top-128 gives below 3% bit reduction and below 5% token reduction, do not implement group-token v3.

## Research Question

Can a reversible event-group grammar v3 dominate v2.1 sparse lane-action targets under a cheap train-derived dictionary and fallback audit?

## Closest Analogies / Novelty Layer

- Closest analogies: chord/event token dictionaries, byte-pair-style phrase tokens, literal fallback codecs, teacher-forced event grammars.
- Relevant taxonomy bucket: target representation and grammar audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation unless later full-pipeline results show a new modeling advantage.

## Minimal Change

Add a data-only evaluator that:

- loads the existing v2.1 mapper dataset from a config,
- extracts `target_fragment_tokens`,
- parses same-time lane-action runs into 4-lane signatures using `.`/`T`/`S`/`E`,
- builds top-K group dictionaries from a bounded train slice,
- converts eval streams into v3 candidates: `GROUP|signature` for dictionary hits, original lane tokens for fallback, time-shift/BOS/EOS unchanged,
- verifies every v3 candidate reconstructs the original v2.1 token stream exactly,
- scores train-unigram add-alpha NLL bits for v2.1 and each v3 candidate,
- writes JSON and Markdown reports.

## Files Likely to Change

- `src/pulsefield_model/evals/target_grammar_v3_pressure.py`
- `tests/evals/test_target_grammar_v3_pressure.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_pressure_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_pressure_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_pressure_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/models/mapper/v2_1/vocab.py`
- `src/pulsefield_model/models/mapper/v2_1/tokenizer.py`
- `src/pulsefield_model/models/mapper/v2_1/replay.py`
- `src/pulsefield_model/models/mapper/v2_1/grammar.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_raw_factor_probe_result_report.md`

## Dataset Slice

Use the P19/P21 mapper config as the current v2.1 dataset source:

- config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- train dictionary slice: first `4096` train windows
- eval scoring slice: full eval split, expected `142` windows

## Baseline / Comparator

Baseline is current v2.1 sparse target grammar:

- `TS_*` time-shift tokens,
- one token per present lane action,
- no token for NONE lanes,
- BOS/EOS unchanged.

## Primary Metric

Eval estimated unigram NLL bit reduction for top-128 v3 candidate versus v2.1 baseline.

## Secondary Metric

- Total target-token reduction.
- Lane-action token reduction.
- Group-token coverage.
- Fallback group rate.
- Reconstruction mismatch count.
- Train/eval observed group vocabulary sizes.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.target_grammar_v3_pressure \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_pressure_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_pressure_result_report.md \
  --train-limit 4096
```

## Guard Check

```bash
uv run --group dev pytest tests/evals/test_target_grammar_v3_pressure.py tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
```

## Qualitative Check

Inspect top group signatures. A useful v3 candidate should cover multi-lane/chord/LN mixed groups, not only single-lane actions that do not reduce sparse token count.

## Positive Signal

Top-128 candidate has zero reconstruction mismatches and either:

- eval NLL bits at least 5% lower than v2.1, or
- total target tokens at least 10% lower than v2.1.

## Negative Signal

Top-128 candidate has low group coverage, little token reduction, or does not reduce estimated bits after dictionary/fallback effects.

## Kill Criteria

Kill immediate group-token v3 implementation if top-128 has reconstruction mismatches, less than 3% bit reduction, and less than 5% token reduction.

## Expected Failure Modes

- Train slice may underfit group dictionary.
- Unigram NLL is a weak proxy for trained mapper loss.
- Group tokens may increase grammar-mask complexity.
- Token savings may be dominated by rare groups if K is too large.

## Confounders

- v2.1 already omits NONE lanes, so single-lane groups do not save tokens.
- Time-shift decomposition remains unchanged in this candidate.
- Full grammar implementation would need dedicated replay/mask support even if this audit is positive.
- Eval split is small.

## Expected Runtime / Runtime Budget

Expected under 2 minutes on CPU. Stop if config/dataset loading fails.

## Result Interpretation Plan

- Positive result would suggest: create a target grammar v3 smoke implementation card for vocab/tokenizer/replay/mask.
- Negative result would suggest: audit another v3 family, such as factorized tap/start/end masks or time-shift compression.
- Ambiguous result would require: repeat on larger train/eval slice.
- Human owner decides: whether to prioritize v3 implementation or another representation family.
- Next-loop action if positive: v3 group-token grammar smoke card.
- Next-loop action if negative: v3 factorized-mask or time-shift pressure card.
- Next-loop action if ambiguous: larger v3 pressure audit.

## Result Log Template

- Experiment: Target grammar v3 pressure audit
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
- Remaining ambiguity: a positive pressure audit proves representation pressure, not trained mapper quality.

## Next-Loop Action

- If positive: write a v3 group-token grammar smoke implementation card.
- If negative: test factorized mask or time-shift v3 families.
- If ambiguous: repeat on a larger slice.

## Novelty Notes

- Closest analogies: event/chord token dictionaries, phrase-token compression, literal fallback codecs.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation for target grammar selection.
