# C3 Auxiliary-Target Smoke Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P11 selected auxiliary target as the smallest legal C3 integration family after P10 killed target-derived input conditioning.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_target_role_feasibility_result_report.md`.
- Source snapshot / evidence grade: strong sidecar shape evidence; no implementation evidence yet for C3 auxiliary-target learnability.

## Hypothesis

A disabled-by-default C3 auxiliary target can be added to Mapper V2.1 so that exact C3 sidecar labels are predicted as target supervision, without changing mapper outputs, rollout behavior, or default training configs.

## Root Objective

Move C3 toward a legal full-pipeline role by testing whether target-derived C3 labels can be supervised as outputs rather than consumed as inputs.

## Goal Decomposition

- Subgoal 1: add a disabled-by-default auxiliary C3 bag head to Mapper V2.1.
- Subgoal 2: add a loss term that treats existing `c3_side_stream_tokens` tensors as target labels, not conditioning input.
- Subgoal 3: verify default behavior is unchanged and a tiny C3 auxiliary smoke produces finite loss/gradients.

## Candidate Variants

- Variant A: multi-label bag target over the full C3 sidecar token vocabulary from pooled decoder hidden state.
- Variant B: top-K plus `OTHER` C3 target.
- Variant C: ordered sequence decoder for C3 side-stream tokens.
- Variant D: full target grammar replacement with `RAW`/`REF`/`RES` decode semantics.

## Local Verification Matrix

- Variant A: smallest implementation; reuses existing tensors; tests learnability without sequence grammar.
- Variant B: cheaper output head but requires sidecar vocabulary remapping/top-K artifact before training.
- Variant C: preserves order but adds a second decoder and teacher-forced C3 sequence path.
- Variant D: final-tokenization oriented but requires cross-window reference state and replay semantics.

## Selected Variant

- Selected: Variant A, full-vocab multi-label bag target.
- Rejected: B requires a new vocab remap artifact; C/D are too broad before proving any C3 target learnability.
- Why this is the smallest useful test: it adds one optional head and one optional loss term, uses existing sidecar tensors, and leaves rollout unchanged.

## Selection Pressure

- Primary pressure: auxiliary loss is finite, gradients reach the auxiliary head, and default model/loss behavior remains unchanged when disabled.
- Guard pressure: C3 labels must not be used as model conditioning in this smoke.
- Runtime pressure: unit tests plus a tiny CPU training smoke only.
- Kill pressure: if the auxiliary head cannot be wired without changing default mapper behavior or if finite gradients fail, mutate before training.

## Research Question

Can Mapper V2.1 legally learn a target-side C3 signal without using C3 labels as generation-time input?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary target regularization, multi-label bag supervision, target-side representation probe.
- Relevant taxonomy bucket: representation integration validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is an engineering smoke for a previously validated C3 representation.

## Minimal Change

Add disabled-by-default config fields:

- model: `use_c3_auxiliary_target`, `c3_auxiliary_vocab_size`
- loss: `lambda_c3_auxiliary`

When enabled:

- model pools decoder hidden state over the target fragment and emits `c3_auxiliary_logits`;
- loss converts `c3_side_stream_tokens` to a multi-hot target and computes BCE;
- training metrics include `loss/c3_auxiliary`, target counts, and availability counts.

## Files Likely to Change

- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `src/pulsefield_model/training/mapper_runner.py`
- `tests/models/mapper/v2_1/test_model.py`
- `tests/training/test_mapper_v2_1.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_target_role_feasibility_result_report.md`
- `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`

## Dataset Slice

Use the exact P5 C3 sidecar and the same real-cache mapper slice used by P9:

- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`
- mapper record cache: `artifacts/cache/stage2_mapper_v2_1/window_records/stage2_mapper_v2_1_phase_b_sparse_global_mps_plus_end.parquet`
- control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- max steps: 2
- eval size: 2
- batch size: 1
- device: CPU

## Baseline / Comparator

Baseline is current Mapper V2.1 with C3 auxiliary target disabled.

Comparator is the same mapper with:

- C3 sidecar tensors loaded as target labels;
- `use_c3_auxiliary_target=true`;
- `c3_auxiliary_vocab_size=14294`;
- `lambda_c3_auxiliary` small enough for smoke.

## Primary Metric

- `loss/c3_auxiliary` finite in train/eval metrics.

## Secondary Metric

- auxiliary-head gradient nonzero in unit test;
- main mapper tests still pass;
- `loss/total` includes `lambda_c3_auxiliary * loss/c3_auxiliary` only when enabled;
- default disabled path has no `c3_auxiliary_logits` and unchanged logits.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_auxiliary_target_smoke.yaml
```

## Guard Check

- Do not enable C3 auxiliary target by default.
- Do not change rollout/incremental decode.
- Do not use C3 side-stream tensors as conditioning in the auxiliary smoke.
- Do not commit checkpoints.

## Qualitative Check

Confirm the smoke report records C3 sidecar labels loaded and C3 auxiliary metrics present.

## Positive Signal

- Unit tests pass.
- Smoke run completes.
- C3 auxiliary loss is finite.
- Auxiliary-head gradients are nonzero.
- Default disabled behavior is unchanged.

## Negative Signal

- Default mapper behavior changes when auxiliary target is disabled.
- Auxiliary loss is non-finite.
- C3 labels are accidentally consumed as conditioning input.
- Training report does not expose C3 auxiliary metrics.

## Kill Criteria

Kill this auxiliary-target implementation if it cannot be made disabled-by-default, if it changes rollout/default logits, or if finite auxiliary loss/gradients fail in a tiny smoke.

## Expected Failure Modes

- BCE over a full 14,294-way bag may be weak or dominated by negatives.
- Bag supervision discards order and reference semantics.
- Two steps cannot prove learnability; it only proves wiring.
- Future useful metrics may need token-kind metadata.

## Confounders

- Target labels are still target-derived, but they are used as supervision rather than inference input.
- Positive smoke does not prove generation quality.
- Full C3 tokenization still needs grammar/replay design.

## Expected Runtime / Runtime Budget

Expected under 5 minutes. Stop if focused tests fail.

## Result Interpretation Plan

- Positive result would suggest: run a longer auxiliary-target learnability comparison.
- Negative result would suggest: mutate to top-K labels or sidecar metadata enrichment before another model smoke.
- Ambiguous result would require: add richer C3 label diagnostics.
- Human owner decides: whether auxiliary target is worth further training after smoke.
- Next-loop action if positive: longer auxiliary-target comparison card.
- Next-loop action if negative: top-K/metadata mutation card.
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
- Remaining ambiguity: this smoke only proves wiring, not C3 quality.

## Next-Loop Action

- If positive: longer auxiliary-target comparison card.
- If negative: top-K/metadata mutation card.
- If ambiguous: diagnostics card.

## Novelty Notes

- Closest analogies: auxiliary loss, multi-label target probe.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is a legal integration probe for C3, not a new tokenizer result.
