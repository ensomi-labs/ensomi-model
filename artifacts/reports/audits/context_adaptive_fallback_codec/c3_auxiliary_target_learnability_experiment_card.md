# C3 Auxiliary-Target Learnability Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P12 proved the disabled-by-default C3 auxiliary-target path is wired and finite on a 2-step smoke.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_auxiliary_target_smoke_result_report.md`.
- Source snapshot / evidence grade: strong wiring evidence; weak learnability evidence.

## Hypothesis

Over a longer fixed-seed small run, C3 auxiliary target loss will remain finite and show a weak decreasing trend without materially degrading the main mapper token loss versus a matched baseline.

## Root Objective

Determine whether the legal C3 target-side auxiliary path is worth further training, or whether it should mutate toward top-K/kind-aware labels or full target-grammar decomposition.

## Goal Decomposition

- Subgoal 1: run a 100-step baseline with exact C3 tensors loaded but unused.
- Subgoal 2: run a matched 100-step C3 auxiliary-target comparison.
- Subgoal 3: compare main mapper loss and C3 auxiliary loss trajectory.

## Candidate Variants

- Variant A: repeat the 2-step smoke.
- Variant B: 100-step fixed-seed CPU comparison using full-vocab C3 multi-label bag loss.
- Variant C: top-K plus `OTHER` C3 target before longer training.
- Variant D: skip auxiliary and start target grammar decomposition.

## Local Verification Matrix

- Variant A: already done; insufficient to measure trend.
- Variant B: smallest run that reuses existing implementation and gives repeated eval points.
- Variant C: plausible mutation, but premature before checking full-vocab smoke trend.
- Variant D: important later, but broader than needed for the first learnability gate.

## Selected Variant

- Selected: Variant B, 100-step fixed-seed controlled comparison.
- Rejected: A is redundant; C/D are mutations for negative or ambiguous P13 results.
- Why this is the smallest useful test: it adds repeated eval points without changing the implementation.

## Selection Pressure

- Primary pressure: C3 auxiliary eval loss is finite and does not increase from first eval to final eval.
- Guard pressure: main eval `loss/token` is not worse than baseline by more than 2%.
- Runtime pressure: CPU, batch size 1, small model, eval every 20 steps.
- Kill pressure: stop or reject if the run fails, auxiliary loss is non-finite, or main token loss regresses by more than 2%.

## Research Question

Does full-vocab C3 bag supervision show a short-run learnability signal without hurting the main mapper objective?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary loss ablation, multi-label representation probe, target-side regularization check.
- Relevant taxonomy bucket: representation integration validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is an engineering validation of a legal C3 target role.

## Minimal Change

Add two reproducible YAML configs and run the existing mapper trainer:

- baseline: C3 sidecar tensors loaded, C3 conditioning disabled, auxiliary target disabled.
- auxiliary: same data path, C3 conditioning disabled, auxiliary target enabled.

No code changes are planned in this card.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_baseline.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_result_report.md`

## Read-Only Context Files

- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_target_role_feasibility_result_report.md`

## Dataset Slice

Use the exact P5 C3 sidecar and same mapper cache family:

- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- mapper record cache: `artifacts/cache/stage2_mapper_v2_1/window_records/stage2_mapper_v2_1_phase_b_sparse_global_mps_plus_end.parquet`
- control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- max steps: 100
- eval every: 20
- eval size: 8
- final train eval size: 16
- batch size: 1
- device: CPU

## Baseline / Comparator

Baseline:

- `include_c3_side_stream_token_tensors=true`
- `use_c3_side_stream_conditioning=false`
- `use_c3_auxiliary_target=false`
- `lambda_c3_auxiliary=0.0`

Comparator:

- same exact sidecar tensors
- `use_c3_side_stream_conditioning=false`
- `use_c3_auxiliary_target=true`
- `c3_auxiliary_vocab_size=14294`
- `lambda_c3_auxiliary=0.05`

## Primary Metric

- C3 auxiliary eval `loss/c3_auxiliary` trend from first eval to final eval.

## Secondary Metric

- Main eval `loss/token` delta between auxiliary and baseline.
- Final train-eval `loss/token` delta.
- Total loss accounting.
- Completion status and finite metrics.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_learnability_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Do not enable C3 conditioning.
- Do not change rollout/incremental decode.
- Do not commit checkpoints.
- Do not interpret auxiliary loss as generation quality.

## Qualitative Check

Inspect whether C3 auxiliary eval loss is flat, decreasing, or increasing and whether main token loss is comparable to baseline.

## Positive Signal

- Both runs complete.
- Auxiliary eval loss is finite and does not increase from first eval to final.
- Main eval token loss is within 2% of baseline.

## Negative Signal

- Auxiliary loss is non-finite or increases.
- Main token loss regresses by more than 2%.
- Reports do not prove exact C3 labels were loaded.

## Kill Criteria

Kill or mutate full-vocab bag auxiliary supervision if the run cannot complete, auxiliary loss is non-finite/increasing, or main token loss regresses by more than 2%.

## Expected Failure Modes

- Full-vocab BCE may be too blunt and mostly learn negatives.
- C3 auxiliary loss may move slowly over 100 steps.
- Bag target discards C3 order and reference semantics.

## Confounders

- The auxiliary model has extra parameters.
- Same seed controls data order and base model init, but the extra head changes optimization slightly.
- C3 labels are target-derived but used only as supervision.

## Expected Runtime / Runtime Budget

Expected under 20 minutes total. Stop if either run fails or clearly hangs.

## Result Interpretation Plan

- Positive result would suggest: longer auxiliary-target comparison or top-K/kind-aware diagnostics.
- Negative result would suggest: mutate to top-K/kind-aware labels or target grammar decomposition.
- Ambiguous result would require: stronger metrics, longer run, or sidecar token-kind targets.
- Human owner decides: whether auxiliary regularization is worth production-scale training.
- Next-loop action if positive: longer or richer auxiliary-target experiment.
- Next-loop action if negative: top-K/kind-aware mutation card.
- Next-loop action if ambiguous: diagnostics card.

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
- Remaining ambiguity: quality and generation-readiness remain out of scope.

## Next-Loop Action

- If positive: longer or richer auxiliary-target experiment.
- If negative: top-K/kind-aware mutation card.
- If ambiguous: diagnostics card.

## Novelty Notes

- Closest analogies: auxiliary loss ablation and representation-target probe.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this measures whether C3 can be learned as a target-side signal.
