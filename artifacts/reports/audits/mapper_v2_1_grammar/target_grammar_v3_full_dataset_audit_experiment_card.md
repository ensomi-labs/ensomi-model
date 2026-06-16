# Target Grammar v3 Full 4K Dataset Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: bounded v3 event-token smoke passed, but the updated goal requires full 4K dataset audit before v3 can replace the mapper pipeline.
- Acceptance source, if any: active goal update requiring v3 final state before full training/inference replacement.
- Source snapshot / evidence grade: strong bounded-slice local evidence; missing full-dataset evidence.

## Hypothesis

The v3 event-token grammar will preserve exact v2.1 reconstruction and maintain positive token/bit pressure on the full 4K mapper dataset, not only on the bounded 4096-window train slice plus 142-window eval split.

## Root Objective

Prove or reject v3 event-token grammar as a final target representation candidate before any full-pipeline replacement.

## Goal Decomposition

- Subgoal 1: Run v3 conversion on the full available 4K mapper dataset.
- Subgoal 2: Verify zero reconstruction mismatches across all audited windows.
- Subgoal 3: Measure token reduction, total-bit reduction, vocabulary usage, and terminal/cross-window LN cases.
- Subgoal 4: Produce a replacement gate decision for dataset/model/training/inference integration.

## Candidate Variants

- Variant A: Full v3 event-token grammar over all available mapper windows.
- Variant B: Bounded smoke only.
- Variant C: Factorized event-mask v3.
- Variant D: Return to C3 side-stream integration.

## Local Verification Matrix

- Variant A: Required by updated goal; selected if runtime is feasible and exact reconstruction can be checked globally.
- Variant B: Rejected as insufficient for replacement.
- Variant C: Deferred unless Variant A fails.
- Variant D: Rejected for now because C3 mapper-side auxiliary showed diminishing returns.

## Selected Variant

- Selected: Variant A, full v3 event-token audit.

## Selection Pressure

- Must have zero reconstruction mismatches.
- Must reduce total target tokens versus v2.1.
- Should improve total unigram bits or provide a defensible trained-loss follow-up if the bit proxy weakens.
- Must summarize terminal chart-end windows and cross-window LN carry windows.

## Minimal Change

Extend the smoke evaluator only as needed to support full-dataset audit controls:

- `--train-limit all` or equivalent,
- optional `--eval-limit`,
- window-class counters for terminal fragments and LN carry,
- resumable/progress-safe output if runtime is long.

Do not change production training defaults in this experiment.

## Files Likely to Change

- `src/pulsefield_model/evals/target_grammar_v3_event_smoke.py`
- `tests/evals/test_target_grammar_v3_event_smoke.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_result_report.md`

## Dataset Slice

- Full available 4K mapper dataset from the current v2.1 config/index.
- No bounded train-window cap for the main audit.
- If runtime is excessive, stop and report runtime blocker with partial counts; do not call it final v3 evidence.

## Baseline / Comparator

Current v2.1 sparse target grammar over the same windows.

## Primary Metric

Full-dataset reconstruction mismatch count.

## Secondary Metric

- Total token reduction.
- Total unigram bit reduction.
- Terminal chart-end window count.
- Cross-window LN carry window count.
- Event vocabulary coverage and top event signatures.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.target_grammar_v3_event_smoke \
  --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_result_report.md \
  --train-limit all
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

Inspect examples or counters for:

- terminal chart-end events,
- cross-window LN carry,
- LN start/end mixed chords,
- rare event signatures.

## Positive Signal

- zero reconstruction mismatches on the full audit,
- token reduction remains positive,
- bit proxy remains positive or at least not strongly negative,
- terminal/carry cases are covered.

## Negative Signal

- any reconstruction mismatch,
- unhandled terminal/carry class,
- positive bounded-slice result disappears on full dataset.

## Kill Criteria

Kill v3 event-token replacement if full-dataset reconstruction has any mismatch that is not explained by invalid input data, or if terminal/carry cases require future target lookahead.

## Expected Failure Modes

- Runtime may be long because the current evaluator tokenizes windows on demand.
- Full dataset may expose rare same-lane collisions or chart-end boundary cases.
- Unigram bit proxy may differ from trained loss.

## Expected Runtime / Runtime Budget

Expected minutes to tens of minutes on CPU depending on cache state. Stop if runtime or memory becomes impractical and record partial evidence; do not treat partial evidence as final.

## Confounders

- Full train scoring may use a much larger token count than previous bounded runs.
- The eval split in the current config is small; full-dataset audit should report both train/all-window and eval metrics.
- Passing this audit still does not prove trained mapper quality.

## Result Interpretation Plan

- Positive: create the v3 full-pipeline replacement card covering dataset, model config, training, inference, and compatibility migration.
- Negative: mutate to factorized-mask v3 or repair event-token semantics before replacement.
- Ambiguous/runtime-limited: optimize/resume the evaluator before making a representation decision.

## Result Log Template

- Experiment: Target grammar v3 full 4K dataset audit
- Date:
- Commit / run id:
- Dataset scope:
- Runtime:
- Reconstruction mismatches:
- Token reduction:
- Bit reduction:
- Terminal/carry coverage:
- Guard command / result:
- Failed checks:
- Confounders:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: v3 planner/mapper full-pipeline replacement card.
- If negative: v3 grammar mutation card.
- If runtime-blocked: evaluator hardening card.

## Closest Analogies and Novelty Layer

- Closest analogies: existing tuple/event mapper grammar, chord/event token dictionaries, teacher-forced event grammars.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation until full-pipeline training/inference evidence exists.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: this audit can approve representation replacement work, but does not itself train or validate the final mapper.
