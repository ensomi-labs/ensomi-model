# Target Grammar v3 Bounded Fixed-Split Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the tiny trained comparison passed all report-contract gates and showed lower v3 valid-token count, but one-step evidence is not enough before longer training or session-runtime replacement.
- Acceptance source, if any: active goal requires v3 mapper full pipeline from training to inference before replacement.
- Source snapshot / evidence grade: strong representation evidence; medium pipeline smoke evidence; weak trained-quality evidence.

## Hypothesis

On a fixed local slice with repeated optimizer steps, v3 and v2.1 can both train through the shared runner, and v3 will preserve a lower valid-token count while producing finite train/eval losses and valid reports.

## Root Objective

Move from one-step tiny comparability toward a bounded trained comparison that is still cheap enough to fail quickly.

## Goal Decomposition

- Subgoal 1: Build one fixed local index slice shared by v2.1 and v3.
- Subgoal 2: Run both mappers with comparable tiny CPU configs and the same train/eval split settings.
- Subgoal 3: Compare shared-runner reports with the existing report comparison harness.
- Subgoal 4: Record target-token ratio, finite losses, contract checks, and limitations.
- Subgoal 5: Keep mapper v2.1/v3 guard tests green.

## Candidate Variants

- Variant A: 16-window fixed slice, 5 CPU steps, tiny d_model=16 configs.
- Variant B: full 4K short training run.
- Variant C: only rerun the existing two-row one-step comparison.
- Variant D: session-runtime v3 route before bounded trained comparison.

## Local Verification Matrix

- Variant A: Selected. It is larger than the tiny gate, uses repeated updates, and stays low-risk.
- Variant B: Rejected for this turn because it is too expensive before bounded results are inspected.
- Variant C: Rejected because it adds no evidence beyond the previous gate.
- Variant D: Rejected because trained comparison evidence is still insufficient.

## Selected Variant

- Selected: Variant A, 16-window fixed slice with 5 CPU training steps per mapper.

## Selection Pressure

- Must use the same fixed index slice for v2.1 and v3.
- Must use comparable model/control/loss overrides.
- Must validate contracts and finite metrics with the existing comparison harness.
- Must not claim mapper quality or switch defaults.

## Minimal Change

No production code change is required for the selected variant. Run the existing v2.1/v3 training functions and comparison harness, then add result artifacts.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_bounded_fixed_split_comparison_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_bounded_fixed_split_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_bounded_fixed_split_comparison_summary.json`

## Dataset Slice

- First 16 rows from `artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet`.
- Local dataset root: `dataset`.
- Fixed output root: `artifacts/tmp/mapper_v3_bounded_fixed_split_comparison`.

## Baseline / Comparator

- v2.1 sparse lane-action mapper trained with the same tiny CPU config.
- v3 event-group mapper trained with the same tiny CPU config.

## Primary Metric

Comparison harness route must be `TEST_NEXT`.

## Secondary Metric

- v3 eval valid-token count and token ratio versus v2.1.
- final eval loss/total for both reports.
- completed steps for both reports.

## Verify Command / Evaluation Procedure

Run bounded paired training:

```bash
uv run python - <<'PY'
# Build fixed 16-row local index and run both run_mapper_v2_1_phase_b_training(...)
# and run_mapper_v3_phase_b_training(...) for max_steps=5 on CPU tiny configs.
PY
```

Run comparison:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_training_comparison \
  --v2-1-report artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v21/run/report.json \
  --v3-report artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/v3/run/report.json \
  --summary-output artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/comparison_summary.json \
  --report-output artifacts/tmp/mapper_v3_bounded_fixed_split_comparison/comparison_report.md
```

Guard tests:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Guard Check

- No code/default replacement in this experiment.
- No v3 resume claim.
- No quality claim from 5 CPU steps.
- v2.1 comparator remains unchanged.

## Qualitative Check

Inspect report artifacts and confirm they state:

- this is a bounded smoke/trend gate,
- v3 target-token reduction is the main positive signal,
- final loss is not directly comparable as quality because vocabularies differ,
- next step is a longer bounded comparison only if this passes.

## Positive Signal

- Both runs complete 5 steps.
- Both reports validate expected contracts.
- Both final eval losses are finite.
- v3 eval valid-token count is lower than v2.1.

## Negative Signal

- Either run fails.
- Report contract validation fails.
- v3 no longer has lower valid-token count.
- Any v2.1/v3 guard test regresses.

## Kill Criteria

Do not escalate to longer training if the bounded comparison cannot complete or if v3 loses the target-token reduction under the shared runner metrics.

## Expected Failure Modes

- Some slice rows may have longer target streams requiring `max_seq_len=1024`.
- Tiny CPU loss is noisy and may not reflect quality.
- The first 16 rows may be rhythmically narrow.

## Expected Runtime / Runtime Budget

Paired training plus comparison should run under 10 minutes on CPU.

## Confounders

- Tiny configs are capacity-limited.
- Different vocabularies make per-token loss not a direct quality metric.
- First-rows slice may not represent the full 4K distribution.

## Result Interpretation Plan

- Positive: schedule a larger bounded comparison using the real v3 config/cache path.
- Negative: repair training/report issues or inspect whether v3 compression does not survive runner metrics.
- Ambiguous: rerun with a slightly broader slice before changing architecture.

## Result Log Template

- Experiment: Target grammar v3 bounded fixed-split comparison
- Date:
- Commit / run id:
- Slice:
- Steps:
- v2.1 report:
- v3 report:
- Comparison route:
- Completed steps:
- Final eval losses:
- Valid-token counts:
- Token ratio:
- Guard tests:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run a longer bounded v3-vs-v2.1 comparison with a broader fixed split and possibly real d384 config.
- If negative: diagnose the failing runner/report/token-count layer.

## Closest Analogies and Novelty Layer

- Closest analogies: mapper v2.1 Phase B smoke training, shared-runner report comparison.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering validation for v3.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: real mapper quality still requires a larger run.
