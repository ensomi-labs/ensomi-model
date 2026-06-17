# Target Grammar v3 Teacher-Forced Time-Shift Logit Audit Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: after C3 mapper-side diminishing returns and failed time-shift full32 rollout, test whether v3 timing collapse is visible under teacher forcing before adding another loss or grammar mutation.
- Acceptance source, if any: `target_grammar_v3_c3_diminishing_returns_route_synthesis_result_report.md`.
- Source snapshot / evidence grade: medium-high. v3 representation and timing-residue audits passed; generated rollouts fail timing/continuation; current time-shift objective failed full32; C3 mapper-side follow-ups are mostly `MUTATE`.

## Hypothesis

If the current 500-step v3 checkpoint's teacher-forced time-shift logits are already collapsed toward a small rigid-grid prior, then v3 needs timing embedding/logit calibration or a different timing objective before more rollout scaling. If teacher-forced time-shift ranks are good while free-running rollouts collapse, the next fix should target exposure/decode-state calibration instead of another teacher-forced scalar loss.

## Root Objective

Move toward a full-pipeline v3 replacement by diagnosing the timing bottleneck after C3 mapper-side routes showed diminishing returns and before spending more runtime on training or C3 target-grammar integration.

## Goal Decomposition

- Subgoal 1: load the existing 500-step v3 checkpoint and fixed-slice eval/windows without retraining.
- Subgoal 2: run teacher-forced forward passes and collect time-shift target ranks, top-k recall, posterior mass, and argmax shift distribution.
- Subgoal 3: compare predicted time-shift concentration against target timing residue and generated rollout rigid-grid symptoms.
- Subgoal 4: route the next mutation to timing embedding/loss, decode-state calibration, or target grammar repair.

## Candidate Variants

- Variant A: C3 RAW split/factor-head training. This follows P20-P23 but spends new model runtime on a weak RAW signal.
- Variant B: ordered RAW field grammar refinement. This preserves C3 reconstruction but starts from a 3.185742x sequence expansion and worse RAW bits/token.
- Variant C: teacher-forced v3 time-shift logit audit. This is artifact-only and directly tests the current rollout bottleneck.
- Variant D: full32/full4k v3 training escalation. This is too expensive before identifying whether timing failure is teacher-forced or exposure-driven.

## Local Verification Matrix

- Variant A: reject unless RAW factor/context probes show broad lift over unigram; current evidence does not.
- Variant B: reject until sequence and bit budgets improve; current ordered RAW probe failed both gates.
- Variant C: pass local planning if it can use existing checkpoint/data and produce rank/concentration metrics without changing defaults.
- Variant D: reject until a smaller diagnostic identifies the next mechanism.

## Selected Variant

- Selected: Variant C, teacher-forced v3 time-shift logit audit.
- Rejected: A, B, and D.
- Why this is the smallest useful test: it uses existing artifacts, directly probes the v3 timing-collapse failure, and can route the next mutation without retraining.

## Selection Pressure

- Primary pressure: distinguish teacher-forced timing-logit collapse from free-running exposure/decode collapse.
- Guard pressure: no training, no rollout rerun, no source behavior changes, no default changes.
- Runtime pressure: run on the fixed-slice eval windows or a bounded subset first; stop if checkpoint/data loading fails.
- Kill pressure: if the audit cannot produce target-rank and shift-concentration metrics, do not implement a new timing loss.

## Research Question

Does the existing v3 checkpoint rank the correct teacher-forced time-shift tokens well, or does it already prefer rigid 160/200ms-style shifts under teacher forcing?

## Closest Analogies / Novelty Layer

- Closest analogies: calibration audit, teacher-forced versus free-running diagnosis, posterior-rank analysis, exposure-bias triage.
- Relevant taxonomy bucket: model diagnostics and representation-route selection.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for the v3 target grammar path.

## Minimal Change

Add one artifact-only eval script/report, or run an equivalent notebook-free script, that loads the existing v3 checkpoint and fixed-slice eval records, computes teacher-forced logits, and writes rank/concentration summaries.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_teacher_forced_time_shift_logit_audit.py`
- `tests/evals/test_mapper_v3_teacher_forced_time_shift_logit_audit.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_result_report.md`

## Read-Only Context Files

- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`
- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json`
- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_summary.json`
- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/loss.py`
- `src/pulsefield_model/training/mapper_v3.py`

## Dataset Slice

Start with the existing fixed 32-song / 256-window v3 slice:

- eval windows from `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json`;
- fixed record cache `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`;
- control teacher cache `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`.

If full fixed-slice evaluation is slow, use the report eval split first and cap batches while preserving per-target-shift counts.

## Baseline / Comparator

Primary comparator:

- target time-shift distribution from teacher-forced v3 records;
- current 500-step v3 checkpoint time-shift posterior under teacher forcing.

Context comparators:

- generated rollout dominant-spacing ratios from the 32-case fixed-slice wide audit;
- target timing-residue audit interval distribution.

## Primary Metric

Teacher-forced target time-shift rank and recall:

- target time-shift recall@1, recall@3, recall@5;
- median and p90 target time-shift rank;
- target time-shift negative log probability on time-shift rows only.

## Secondary Metric

- predicted time-shift argmax distribution;
- predicted top-1 share for 160ms and 200ms shifts;
- JS or KL divergence between target shift distribution and predicted argmax distribution;
- event-token versus time-shift valid-step calibration on rows following events;
- per-difficulty or per-density timing-rank buckets if available.

## Verify Command / Evaluation Procedure

Planned command:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_teacher_forced_time_shift_logit_audit \
  --checkpoint artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt \
  --training-report artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/report.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_result_report.md \
  --device cpu \
  --batch-size 2
```

Guard command:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_teacher_forced_time_shift_logit_audit.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_teacher_forced_time_shift_logit_audit_summary.json >/tmp/teacher_forced_time_shift_logit_audit.valid.json
```

## Guard Check

- checkpoint loads as `MapperV3Model`;
- dataset contract is `v3_event_groups`;
- only target time-shift rows are scored for time-shift metrics;
- no training or rollout is run;
- summary JSON validates;
- existing v3 model/training tests pass.

## Qualitative Check

Inspect top overselected time-shift buckets and examples where the target shift rank is poor. Separate true teacher-forced logit collapse from cases where target shifts are ranked near top-k but free-running decode still collapses.

## Positive Signal

- The audit produces finite metrics over a nonzero number of target time-shift rows.
- If recall@5 is weak and argmax concentration is high around a few shifts, route to timing embedding/logit calibration.
- If recall@5 is strong but free-running remains rigid, route to exposure/decode-state calibration.

## Negative Signal

- The audit cannot load the checkpoint/data.
- Metrics are dominated by non-time-shift rows due to a masking bug.
- Results are too sparse to distinguish teacher-forced collapse from exposure collapse.
- The audit simply reproduces rollout symptoms without new rank/concentration evidence.

## Kill Criteria

Kill this diagnostic path if the fixed-slice checkpoint/data cannot be loaded reproducibly, if target-row masking cannot be verified, or if the audit cannot report target time-shift rank and predicted shift concentration.

## Expected Failure Modes

- The eval split is too small for rare shift buckets.
- CPU checkpoint evaluation is slow.
- Dataset reconstruction from training report requires config normalization.
- Teacher-forced logits look healthy, shifting the problem to free-running state distribution rather than loss/embedding.

## Confounders

- Fixed-slice evidence is not full 4k.
- Teacher forcing may overestimate generation-time calibration.
- Greedy decode may amplify small posterior biases.
- Existing rollout summaries use real-audio session preparation, while teacher-forced eval uses cached records.

## Expected Runtime / Runtime Budget

Expected runtime: less than 20 minutes for the fixed eval split on CPU; less than one hour if expanded to the full fixed 256-window slice. Stop early on checkpoint/dataset load failure or non-finite logits.

## Result Interpretation Plan

- Positive result would suggest: implement a timing embedding/logit calibration or loss mutation targeted at the observed collapse.
- Negative result would suggest: teacher-forced logits are not the blocker; focus on decode-state/exposure calibration or grammar-level continuation.
- Ambiguous result would require: a smaller generated-state paired diagnostic on high-leverage cases.
- Human owner decides: whether to invest in v3 timing calibration or return to v2.1 grammar.
- Next-loop action if positive: create a timing calibration implementation card.
- Next-loop action if negative: create a decode-state exposure diagnostic card.
- Next-loop action if ambiguous: narrow to 6 high-leverage fixed-slice cases.

## Result Log Template

- Experiment: target_grammar_v3_teacher_forced_time_shift_logit_audit
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
- Remaining ambiguity: exact dataset loader implementation may need to reuse training config normalization from `pulsefield_model.training.mapper_v3`.

## Next-Loop Action

- If positive: implement the artifact-only teacher-forced audit and route based on observed rank/concentration.
- If negative: stop this diagnostic and inspect decode-state exposure.
- If ambiguous: restrict to high-leverage cases and compare teacher-forced versus generated states directly.

## Novelty Notes

- Closest analogies: posterior calibration audits, exposure-bias triage, teacher-forced/free-running mismatch analysis.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering diagnostic for an already-defined v3 representation.
