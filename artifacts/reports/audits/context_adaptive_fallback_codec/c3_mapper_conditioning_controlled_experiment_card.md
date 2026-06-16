# C3 Mapper Conditioning Controlled Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P7 tiny smoke completed with exact C3 sidecar and showed no immediate loss regression; the next step is a longer fixed-seed controlled comparison.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_mapper_conditioning_smoke_result_report.md`.
- Source snapshot / evidence grade: strong local runtime evidence from a 2-step smoke; weak evidence for training signal.

## Hypothesis

On a small fixed real-data training run, pooled exact-C3 side-stream conditioning will remain stable and produce a final eval loss comparable to the same model carrying C3 tensors with conditioning disabled.

## Root Objective

Establish whether C3 is ready for a longer controlled training comparison as a full-pipeline mapper input, rather than only a runtime smoke.

## Goal Decomposition

- Subgoal 1: run a baseline 20-step mapper training comparison with exact C3 tensors carried but unused.
- Subgoal 2: run a matched 20-step comparison with pooled C3 conditioning enabled.
- Subgoal 3: compare eval loss curves, final train-eval loss, run stability, and C3 dataset/model metadata.

## Candidate Variants

- Variant A: repeat the 2-step smoke.
- Variant B: 20-step fixed-seed CPU comparison using the small mapper config and exact sidecar.
- Variant C: production-size MPS comparison.
- Variant D: change the C3 conditioning architecture before more training.

## Local Verification Matrix

- Variant A: already done; insufficient for trend evidence.
- Variant B: bounded, reproducible, and exercises real training over multiple eval points.
- Variant C: premature before a small trend check.
- Variant D: premature because current pooled conditioning has not yet shown a controlled negative result.

## Selected Variant

- Selected: Variant B, 20-step fixed-seed controlled comparison.
- Rejected: A is redundant; C is too expensive; D lacks negative evidence.
- Why this is the smallest useful test: it adds repeated eval points and a final train-eval slice while keeping runtime and model size small.

## Selection Pressure

- Primary pressure: C3-enabled final eval loss is within 5% of baseline and finite.
- Guard pressure: both runs complete 20 steps, write reports, and focused verifier tests pass.
- Runtime pressure: CPU, batch size 1, small model, eval every 5 steps.
- Kill pressure: stop if either run fails, produces non-finite loss, or C3-enabled final eval loss is more than 5% worse than baseline.

## Research Question

Does pooled exact-C3 conditioning remain training-stable and loss-comparable over a small fixed real-data run?

## Closest Analogies / Novelty Layer

- Closest analogies: small ablation run, side-channel feature comparison, representation integration canary.
- Relevant taxonomy bucket: representation engineering validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is controlled integration evidence for C3, not a new modeling result.

## Minimal Change

Add two reproducible YAML configs and run the existing `pulsefield_model.training.mapper_v2_1` entrypoint:

- baseline: C3 tensors included, conditioning disabled,
- enabled: same config plus C3 conditioning enabled.

No model, dataset, or production training config changes are planned.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_baseline.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `src/pulsefield_model/training/mapper_v2_1.py`

## Dataset Slice

Use the same real-cache data path as P7:

- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`,
- mapper record cache: `artifacts/cache/stage2_mapper_v2_1/window_records/stage2_mapper_v2_1_phase_b_sparse_global_mps_plus_end.parquet`,
- control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`,
- max steps: 20,
- eval every: 5,
- final train eval size: 8,
- batch size: 1,
- device: CPU.

## Baseline / Comparator

Baseline carries exact C3 tensors but sets `use_c3_side_stream_conditioning=false`.

Comparator enables pooled C3 conditioning with:

- `c3_side_stream_vocab_size=14294`,
- `c3_side_stream_embedding_dim=16`,
- `c3_side_stream_scale_init=0.03`.

## Primary Metric

- Final eval `loss/total` delta: `(enabled - baseline) / baseline`.

## Secondary Metric

- Eval loss trace at steps 5, 10, 15, and 20.
- Final train-eval `loss/total`.
- Last train `loss/total`.
- Reported C3 dataset and model metadata.
- Runtime and completion status.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Do not edit production configs.
- Do not commit checkpoints.
- Both reports must show C3 side-stream tensors loaded from the exact P5 sidecar.
- Treat this as a small training trend check, not a quality claim.

## Qualitative Check

Inspect the loss curves. The enabled run should not diverge or show a large immediate eval-loss penalty.

## Positive Signal

- Both runs complete.
- Enabled final eval loss is within 5% of baseline.
- Eval curves are broadly comparable.

## Negative Signal

- Enabled run fails, produces non-finite loss, or is more than 5% worse on final eval.
- Reports do not prove exact C3 sidecar tensor use.
- Training path silently disables C3 despite enabled config.

## Kill Criteria

Kill the pooled-conditioning path if C3-enabled final eval loss is more than 5% worse than baseline, non-finite, or if the run cannot complete.

## Expected Failure Modes

- Tiny run remains noisy.
- CPU runtime may be dominated by eval/report overhead.
- Sidecar JSON load can dominate startup.
- The small model may not reflect production-size behavior.

## Confounders

- Twenty steps is still short and cannot prove quality.
- Same seed controls data order and base model init, but enabled run has extra C3 parameters.
- Full-song global context is disabled to keep the run small.

## Expected Runtime / Runtime Budget

Expected under 15 minutes total. Stop if either run clearly hangs or fails.

## Result Interpretation Plan

- Positive result would suggest: prepare a longer controlled C3/no-C3 training comparison.
- Negative result would suggest: mutate pooled conditioning or inspect sidecar token distribution in train windows.
- Ambiguous result would require: repeat with a larger fixed eval slice or more seeds.
- Human owner decides: whether to spend longer training runtime.
- Next-loop action if positive: longer controlled comparison card.
- Next-loop action if negative: mutate C3 conditioning.
- Next-loop action if ambiguous: strengthen controls.

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
- Remaining ambiguity: model-quality claims require longer runs.

## Next-Loop Action

- If positive: longer controlled C3/no-C3 comparison.
- If negative: mutate conditioning path.
- If ambiguous: repeat with stronger controls.

## Novelty Notes

- Closest analogies: small ablation run and side-channel feature canary.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering validation for C3 as full-pipeline input.
