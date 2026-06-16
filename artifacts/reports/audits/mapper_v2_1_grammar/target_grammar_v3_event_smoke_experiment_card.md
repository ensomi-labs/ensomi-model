# Target Grammar v3 Event-Token Smoke Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: C3 mapper-side auxiliary targets show diminishing returns, while the v3 pressure audit shows that complete same-time event/group tokens can shorten v2.1 target streams and improve the simple total-bit proxy at K=256.
- Acceptance source, if any: active target grammar v3 goal.
- Source snapshot / evidence grade: strong local evidence for reversible group-token pressure; medium evidence for full event-token v3 because the previous audit was data-only.

## Hypothesis

A v3 event-token grammar using one local 4-lane event token per non-empty timepoint can preserve exact beatmap-event reconstruction, reduce target length versus v2.1 sparse lane-action runs, avoid C3-style cross-window references, and remain teacher-forcing/online-inference friendly.

## Root Objective

Move from pressure evidence toward a full-pipeline target grammar v3 that:

- reversibly reconstructs beatmap events,
- is lower-complexity than v2.1 sparse lane-action runs,
- does not need C3-style cross-window replay or future targets,
- can be used directly as teacher-forcing targets.

## Goal Decomposition

- Subgoal 1: Define a concrete v3 event-token contract.
- Subgoal 2: Convert v2.1 sparse target streams into v3 event tokens and back exactly.
- Subgoal 3: Verify v3 replay legality for LN carry, chart-end-inside-window EOS, and online left-to-right decoding.
- Subgoal 4: Compare v3 token/bit pressure against v2.1 on the same bounded dataset slice.

## Candidate Variants

- Variant A: Full 255-token 4-lane event vocabulary, no fallback.
- Variant B: Top-128 event dictionary plus sparse v2.1 fallback.
- Variant C: Factorized tap/start/end mask grammar.
- Variant D: Continue C3 RAW/factor auxiliary work.

## Local Verification Matrix

- Variant A: Passes if it reconstructs v2.1 event streams exactly, validates replay/mask behavior, and preserves the positive K=256 pressure signal.
- Variant B: Legal and shorter, but previous audit showed top-128 worsened the unigram bit proxy.
- Variant C: May reduce vocabulary size but needs a separate factorized-mask design before implementation.
- Variant D: Rejected for this loop because P20-P22 showed mapper-side C3 auxiliary diminishing returns.

## Selected Variant

- Selected: Variant A, full 4-lane event-token vocabulary.
- Why this is the smallest useful test: the previous audit already showed K=256 was the first candidate with both token reduction and total-bit reduction; full event tokens also remove fallback/dictionary lookup from online inference.

## Selection Pressure

- Primary pressure: v3 event streams must have zero reconstruction mismatches against v2.1 sparse streams and improve eval total unigram bits by at least 3%.
- Secondary pressure: v3 should reduce eval token count by at least 10%.
- Guard pressure: v3 replay and grammar masks must pass focused unit tests for taps, chords, LN start/end, cross-window carry, and terminal padded windows.
- Runtime pressure: bounded CPU audit under 2 minutes.

## Minimal Change

Add a bounded v3 event-token smoke surface:

- v3 event vocab wrapper over the existing 4-lane event vocabulary shape,
- conversion helpers between v2.1 sparse same-time runs and v3 event tokens,
- replay wrappers with v2.1-style `chart_end_ms` handling,
- focused tests and a data-only audit/report.

Do not change production training defaults or checkpoint schemas in this experiment.

## Files Likely to Change

- `src/pulsefield_model/models/mapper/v3/__init__.py`
- `src/pulsefield_model/models/mapper/v3/vocab.py`
- `src/pulsefield_model/models/mapper/v3/replay.py`
- `src/pulsefield_model/models/mapper/v3/conversion.py`
- `src/pulsefield_model/evals/target_grammar_v3_event_smoke.py`
- `tests/models/mapper/v3/test_event_token_smoke.py`
- `tests/evals/test_target_grammar_v3_event_smoke.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_smoke_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_smoke_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/evals/target_grammar_v3_pressure.py`
- `src/pulsefield_model/models/mapper/shared/vocab.py`
- `src/pulsefield_model/models/mapper/shared/replay.py`
- `src/pulsefield_model/models/mapper/v2_1/vocab.py`
- `src/pulsefield_model/models/mapper/v2_1/tokenizer.py`
- `src/pulsefield_model/models/mapper/v2_1/replay.py`

## Dataset Slice

- Config: `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`
- Train scoring slice: first `4096` train windows
- Eval scoring slice: full eval split, expected `142` windows

## Baseline / Comparator

Baseline is current v2.1 sparse target grammar:

- `TS_*` time-shift tokens,
- one lane-action token per non-empty lane,
- no NONE lane tokens,
- EOS on full-chart terminal fragments.

## Primary Metric

Eval total unigram bit reduction versus v2.1 sparse target streams.

## Secondary Metric

- Eval token-count reduction.
- Reconstruction mismatch count.
- v3 event vocabulary size.
- Eval event-token count and chord/multi-lane event count.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.target_grammar_v3_event_smoke \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_smoke_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_smoke_result_report.md \
  --train-limit 4096
```

## Guard Check

```bash
uv run --group dev pytest \
  tests/models/mapper/v3/test_event_token_smoke.py \
  tests/evals/test_target_grammar_v3_event_smoke.py \
  tests/evals/test_target_grammar_v3_pressure.py \
  tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
```

## Qualitative Check

Inspect top v3 event tokens and confirm the compression signal is not only single-lane taps; chords and LN mixed events must be represented directly as current-time event tokens.

## Positive Signal

- zero reconstruction mismatches,
- total-bit reduction at least 3%,
- token-count reduction at least 10%,
- focused replay/grammar tests pass.

## Negative Signal

- any reconstruction mismatch,
- v3 cannot represent terminal padded windows or cross-window LN carry,
- bit reduction below 0% and token reduction below 10%.

## Kill Criteria

Kill immediate v3 event-token implementation if reconstruction mismatches are nonzero or replay cannot support v2.1 terminal/carry semantics without adding cross-window lookahead.

## Expected Failure Modes

- Existing tuple replay may need wrapper semantics for `chart_end_ms`.
- Event vocabulary size may increase per-token entropy despite shorter streams.
- Full event tokens may still need trained mapper validation before promotion.
- Conversion could hide illegal same-lane collisions if it trusts malformed v2.1 runs.

## Expected Runtime / Runtime Budget

Expected under 2 minutes for the data audit and under 10 seconds for unit tests.

## Confounders

- The unigram bit proxy is not trained model loss.
- V2.1 sparse vocabulary is much smaller, so per-token bits can rise even when total bits improve.
- Existing shared tuple code is an analogy and reusable implementation family, not proof that v3 is already production-ready.

## Result Interpretation Plan

- Positive: implement the dataset/model integration card for v3 event-token full-pipeline smoke.
- Negative: mutate to factorized mask grammar or time-shift compression.
- Ambiguous: repeat pressure on a larger eval slice or add trained-loss smoke.

## Result Log Template

- Experiment: Target grammar v3 event-token smoke
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

## Next-Loop Action

- If positive: create v3 full-pipeline dataset/model smoke card.
- If negative: test factorized event-mask grammar.
- If ambiguous: repeat with trained-loss or larger eval.

## Closest Analogies and Novelty Layer

- Closest analogies: existing tuple/event-token mapper grammar, chord/event token dictionaries, phrase-token compression.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation unless later full-pipeline experiments show a modeling advantage.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: a positive smoke test proves local representation legality and pressure, not final trained mapper quality.
