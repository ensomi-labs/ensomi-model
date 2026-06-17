# Target Grammar Shared Timing-Residue Structure Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 is reversible and lower-bit than v2.1, but trained rollouts collapse to rigid `160ms`/`200ms` grids and sparse/dense second-window starvation; matched v2.1 also shows rigid-grid collapse, while C3-as-target and event-signature factorization are not competitive target replacements.
- Acceptance source, if any: `target_grammar_v3_generated_state_continuation_diagnostic_summary.json`, `target_grammar_v3_decode_policy_continuation_sweep_summary.json`, `target_grammar_v3_matched_v21_timing_baseline_summary.json`, and `target_grammar_v3_factorized_event_signature_proxy_summary.json`.
- Source snapshot / evidence grade: strong representation evidence for v3, strong negative evidence against C3 as a direct target sequence and factorized event signatures, medium generated-quality evidence that the remaining failure is shared timing calibration rather than v3 event legality.

## Hypothesis

If the teacher-forcing target streams contain substantially more timing-residue diversity than the generated v2.1/v3 rigid-grid rollouts, then the next bounded repair should target timing loss, timing embeddings, or decode calibration rather than event-token grammar. If the teacher-forcing target streams themselves are dominated by the same `160ms`/`200ms` grid structure, then a timing grammar or v2.1 grammar mutation may be required before more training.

## Root Objective

Decide whether the current v3/v2.1 timing failure is primarily generated-state calibration against a rich target timing distribution, or whether the target grammar itself under-exposes beatmap timing residue.

## Goal Decomposition

- Subgoal 1: Measure time-shift and event-interval structure in current v2.1 teacher-forcing target streams.
- Subgoal 2: Measure the same structure after v3 event-token conversion, confirming v3 does not erase timing-residue information.
- Subgoal 3: Compare teacher-forcing target timing diversity against committed generated-rollout rigid-grid summaries.
- Subgoal 4: Route the next experiment toward timing loss/embedding calibration, timing grammar mutation, or v2.1 grammar repair.

## Candidate Variants

- Variant A: Run another decode-only timing policy sweep.
- Variant B: Add another event-count or event-distribution objective.
- Variant C: Audit shared teacher-forcing timing-residue structure for v2.1 and v3.
- Variant D: Implement beat-phase/time-residue grammar tokens immediately.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Existing decode-policy continuation sweep | A policy passes rigid/continuation guards | Existing sweep route is `KILL`; deterministic knobs are weak and stochastic is unstable |
| B | Conditioned event-distribution short gate | Starvation and event-ratio guards pass | Existing gate route is `KILL`; median event ratio remains out of range |
| C | Artifact-only target-stream timing audit | Target timing is richer than generated rigid grids and v3 preserves timing tokens | Target streams are themselves grid-dominated or v3 conversion loses timing information |
| D | New grammar/tokenizer implementation | Could expose timing residue directly | Too large before checking whether current targets already contain the needed signal |

## Selected Variant

- Selected: Variant C, shared timing-residue structure audit.
- Rejected: Variant A because the decode-only continuation sweep is already killed.
- Rejected: Variant B because conditioned event-distribution training failed a hard calibration guard.
- Rejected: Variant D because a grammar mutation is premature before measuring target timing residue.
- Why this is the smallest useful test: it is artifact-only, uses current dataset/tokenizer paths, and can fail quickly before another training run or grammar change.

## Selection Pressure

- Primary pressure: determine whether generated rigid-grid collapse is contradicted by richer teacher-forcing timing structure.
- Guard pressure: preserve v3 reversibility, no future target lookup, no tokenizer/default changes.
- Runtime pressure: bounded target-stream audit first; no training or rollout.
- Kill pressure: if targets are already grid-dominated, do not spend the next loop on loss-only calibration.

## Research Question

Does the current v2.1/v3 teacher-forcing target stream already encode the timing-residue diversity needed to escape generated rigid grids, or does the grammar need a timing-specific repair?

## Closest Analogies / Novelty Layer

- Closest analogies: teacher-forcing distribution audit, exposure-bias diagnostic, rhythm-token/time-shift entropy audit, timing-residue tokenization probe.
- Relevant taxonomy bucket: representation and calibration diagnosis before grammar mutation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is an observability audit, not a new representation claim.

## Minimal Change

Add an artifact-only evaluator that reads current v2.1 target streams, converts them to v3 with the existing conversion, and reports timing structure:

- time-shift token histograms for v2.1 and v3;
- top time-shift share, `200ms` time-shift share, entropy, and effective vocabulary;
- event-interval histograms reconstructed from time shifts;
- `160ms`/`200ms` event-interval share;
- per-window dominant event-interval ratio and rigid-window share;
- v2.1/v3 timing parity checks;
- comparison against generated rollout summaries that already report `160ms`/`200ms` dominant-grid collapse.

This evaluator must not add a runtime tokenizer, model head, loss, or decode policy.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_experiment_card.md`
- `src/pulsefield_model/evals/target_grammar_shared_timing_residue_structure_audit.py`
- `tests/evals/test_target_grammar_shared_timing_residue_structure_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_multicase_timing_quality_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_matched_v21_timing_baseline_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_factorized_event_signature_proxy_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_target_complexity_comparison_summary.json`
- current v2.1/v3 tokenizer and conversion files.

## Dataset Slice

Default bounded slice: `4096` train windows and all eval windows from the same config used by the v3 full-dataset audit, `artifacts/reports/audits/context_adaptive_fallback_codec/c3_kind_heads_1000step_enabled.yaml`.

Allow `--train-limit all --eval-limit all` for a full rerun, but do not require full generation for this first audit.

## Baseline / Comparator

- v3 full-dataset representation audit: reconstruction mismatches `0`, eval token reduction `19.96%`, eval total-bit reduction `5.43%`.
- v3 multicase/generated-state audits: rigid-grid generated timing and sparse/dense second-window starvation.
- matched v2.1 timing baseline: also rigid-grid generated timing, but less v3-specific second-window starvation.
- C3 target complexity audit: C3 is reconstructive but not target-sequence competitive.
- factorized event proxy: exact reconstruction but killed by total-bit and event-cost guards.

## Primary Metric

Route decision:

- `TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION` if teacher-forcing target streams show nontrivial timing-residue diversity, v3 preserves timing parity, and generated summaries remain rigid-grid dominated.
- `MUTATE_TO_TIMING_GRAMMAR_REPAIR` if teacher-forcing targets are themselves dominated by `160ms`/`200ms` or low effective time-shift vocabulary.
- `KILL_TIMING_RESIDUE_AUDIT_INPUTS` if the evaluator cannot access target streams or comparator summaries.

## Secondary Metric

- Time-shift entropy and effective vocabulary.
- Top-1/top-3 time-shift share.
- `200ms` time-shift share and `160ms`/`200ms` event-interval share.
- Event-interval entropy and effective vocabulary.
- Per-window dominant event-interval ratio.
- Rigid-window share at thresholds `0.80`, `0.90`, and `0.95`.
- v2.1/v3 time-shift parity mismatch count.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_target_grammar_shared_timing_residue_structure_audit.py -q

uv run python -m pulsefield_model.evals.target_grammar_shared_timing_residue_structure_audit \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_result_report.md
```

## Guard Check

```bash
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_summary.json >/dev/null
git diff --check
```

The evaluator must also assert:

- v2.1 and v3 time-shift sequences match after conversion;
- v3 reconstruction mismatches remain zero on the audited slice;
- no target grammar/defaults are changed;
- generated-summary comparisons are treated as context, not proof of trained quality.

## Qualitative Check

The result report must explain whether rigid generated timing looks like exposure/calibration failure against richer targets, or a target-grammar limitation. It must not claim full replacement readiness from target-stream statistics alone.

## Positive Signal

- v2.1/v3 time-shift parity mismatches are zero.
- Target time-shift/event-interval effective vocabulary is greater than a few grid values.
- Target rigid-window share is materially lower than generated rigid-grid reports.
- Generated summaries remain rigid under v3 and matched v2.1, supporting a shared calibration interpretation.

## Negative Signal

- Teacher-forcing target streams are dominated by `160ms`/`200ms` shifts.
- Most target windows have dominant interval ratio above `0.95`.
- v3 conversion changes timing-shift sequences.
- Comparator summaries are missing or inconsistent.

## Kill Criteria

- Any v2.1/v3 time-shift parity mismatch.
- Any v3 reconstruction mismatch on the audited slice.
- Target streams are as rigid as generated rollouts by the selected thresholds.
- Missing target stream access or missing generated comparator summaries.

## Expected Failure Modes

- Bounded train slice may overrepresent simple maps.
- Event intervals reconstructed from time shifts do not capture lane/action complexity.
- Timing entropy can look high because of sparse long jumps while local event timing remains gridlike.
- Generated summaries use small fixed slices and cannot prove full-distribution behavior.

## Confounders

- This is not trained model quality.
- Timing F1 is an imperfect quality proxy.
- v2.1 and v3 share time-shift tokens, so parity alone cannot prove better embeddings.
- A positive audit may still require model/loss changes and broader rollout validation.

## Expected Runtime / Runtime Budget

Expected runtime: under two minutes for the bounded default slice and all eval windows. Stop on missing config, inaccessible target streams, or failed parity guards.

No training and no real-audio rollout in this card.

## Result Interpretation Plan

- Positive result would suggest: create a bounded timing-loss or timing-embedding calibration card with rigid-grid and second-window guards.
- Negative result would suggest: mutate toward timing grammar repair or v2.1 grammar repair before more v3 training.
- Ambiguous result would require: one bucket-level audit by difficulty/density, not training.
- Human owner decides: whether timing loss/embedding calibration is worth the next training run.
- Next-loop action if positive: `TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION`.
- Next-loop action if negative: `MUTATE_TO_TIMING_GRAMMAR_REPAIR`.
- Next-loop action if ambiguous: `TEST_TIMING_RESIDUE_BUCKET_DIAGNOSTIC`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary route:
- v2.1/v3 time-shift parity mismatches:
- v3 reconstruction mismatches:
- Time-shift entropy:
- Time-shift effective vocabulary:
- Top time-shift share:
- `160ms`/`200ms` share:
- Event-interval entropy:
- Event-interval effective vocabulary:
- Target rigid-window share:
- Generated rigid-grid comparator:
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
- Remaining ambiguity: exact richness thresholds may be tuned in the evaluator, but the guard must keep timing parity and reconstruction exact.

## Next-Loop Action

- If positive: write a bounded timing loss/embedding calibration Experiment Card.
- If negative: mutate to timing grammar repair or v2.1 grammar repair.
- If ambiguous: run one timing-residue bucket diagnostic.

## Novelty Notes

- Closest analogies: teacher-forcing distribution audit, exposure-bias diagnostic, rhythm/time-shift entropy probe.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is a target-stream observability audit, not a novelty claim.
