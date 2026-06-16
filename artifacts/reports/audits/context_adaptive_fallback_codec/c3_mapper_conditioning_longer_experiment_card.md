# C3 Mapper Conditioning Longer Controlled Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P8 showed pooled exact-C3 conditioning is stable for 20 steps but not better than baseline.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_mapper_conditioning_controlled_result_report.md`.
- Source snapshot / evidence grade: strong local codec and sidecar evidence; moderate runtime evidence; weak model-quality evidence.

## Hypothesis

On a longer fixed real-data run, pooled exact-C3 side-stream conditioning will remain stable and final eval loss will stay comparable to the same run with C3 tensors loaded but conditioning disabled.

## Root Objective

Strengthen mapper-side evidence before deciding whether C3 should move from optional side-channel probe toward a full-pipeline training input.

## Goal Decomposition

- Subgoal 1: run a 100-step baseline with exact C3 tensors loaded and batched but unused.
- Subgoal 2: run a matched 100-step C3-enabled comparison using the same seed, dataset slice, eval slice, and small model shape.
- Subgoal 3: compare eval curves, final train-eval loss, run stability, and C3 metadata against P8.

## Candidate Variants

- Variant A: repeat the 20-step P8 controlled comparison.
- Variant B: 100-step CPU comparison with eval every 20 steps and eval size 8.
- Variant C: production-size MPS comparison.
- Variant D: change the C3 conditioning architecture before longer training.

## Local Verification Matrix

- Variant A: already passed; too weak to answer the next question.
- Variant B: increases training and eval evidence while preserving bounded runtime and existing code paths.
- Variant C: useful later, but premature before the pooled side-channel survives a stronger small run.
- Variant D: premature because current pooled conditioning has not yet failed the stability or loss-comparability gates.

## Selected Variant

- Selected: Variant B, 100-step fixed-seed CPU comparison.
- Rejected: A is redundant; C is too expensive for this gate; D changes the question before the current candidate is adequately measured.
- Why this is the smallest useful test: it increases steps by 5x and eval slice by 4x relative to P8 without changing the implementation.

## Selection Pressure

- Primary pressure: C3-enabled final eval `loss/total` is finite and within 2% of baseline.
- Guard pressure: both reports prove exact C3 sidecar tensor use and focused mapper tests pass.
- Runtime pressure: CPU, batch size 1, small model, fixed eval size 8, final train-eval size 16.
- Kill pressure: stop or reject if either run fails, produces non-finite loss, silently disables C3, or C3-enabled final eval loss is more than 2% worse than baseline.

## Research Question

Does pooled exact-C3 conditioning remain loss-comparable over a longer small fixed real-data run?

## Closest Analogies / Novelty Layer

- Closest analogies: small ablation run, side-channel feature comparison, representation integration canary.
- Relevant taxonomy bucket: representation engineering validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is integration evidence for an existing C3 side-stream representation, not a new modeling claim.

## Minimal Change

Add two reproducible YAML configs and run the existing `pulsefield_model.training.mapper_v2_1` entrypoint:

- baseline: exact C3 tensors included, conditioning disabled,
- enabled: same config plus C3 conditioning enabled.

No dataset, tokenizer, model code, or production training defaults are changed.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_baseline.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_summary.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_controlled_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `src/pulsefield_model/training/mapper_v2_1.py`
- `src/pulsefield_model/models/mapper/v2_1/model.py`

## Dataset Slice

Use the same real-cache data path as P8:

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

Baseline carries exact C3 tensors but sets `use_c3_side_stream_conditioning=false`.

Comparator enables pooled C3 conditioning with:

- `c3_side_stream_vocab_size=14294`
- `c3_side_stream_embedding_dim=16`
- `c3_side_stream_scale_init=0.03`

## Primary Metric

- Final eval `loss/total` delta: `(enabled - baseline) / baseline`.

## Secondary Metric

- Eval loss trace at steps 20, 40, 60, 80, and 100.
- Final train-eval `loss/total`.
- Last train `loss/total`.
- C3 model metadata and dataset metadata.
- Completion status.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_longer_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Do not edit production configs.
- Do not commit checkpoints.
- Both reports must show C3 side-stream tensors loaded from the exact P5 sidecar.
- Treat this as a longer small training comparison, not a quality or inference-readiness claim.

## Qualitative Check

Inspect whether the C3-enabled loss offset shrinks, grows, or remains flat across eval checkpoints.

## Positive Signal

- Both runs complete.
- Enabled final eval loss is within 2% of baseline.
- Eval curves remain broadly comparable and non-divergent.

## Negative Signal

- Enabled run fails, produces non-finite loss, or is more than 2% worse on final eval.
- Reports do not prove exact C3 sidecar tensor use.
- Training path silently disables C3 despite enabled config.

## Kill Criteria

Kill or mutate pooled conditioning if C3-enabled final eval loss is more than 2% worse than baseline, non-finite, incomplete, or if exact sidecar tensor use cannot be proven.

## Expected Failure Modes

- The run may still be too short to show quality benefit.
- The small model may under-use the side-channel.
- CPU runtime may be dominated by sidecar loading and repeated evaluation.
- Extra C3 parameters make exact initialization parity impossible.

## Confounders

- Same seed controls data order and base model init, but enabled run has additional C3 parameters.
- Full-song global context is disabled to keep the run small.
- This pooled architecture compresses the side stream into one vector per window, so it may be too coarse.
- Incremental decode currently rejects C3 conditioning, so training stability does not prove inference readiness.

## Expected Runtime / Runtime Budget

Expected under 45 minutes total. Stop if either run clearly hangs, fails, or exceeds practical local runtime.

## Result Interpretation Plan

- Positive result would suggest: run a production-shaped MPS comparison or implement an inference-side C3 conditioning design card.
- Negative result would suggest: mutate the conditioning architecture before longer training.
- Ambiguous result would require: repeat with stronger eval controls or multiple seeds.
- Human owner decides: whether neutral stability is enough to spend production-scale training time.
- Next-loop action if positive: production-shaped controlled comparison card.
- Next-loop action if negative: C3 conditioning architecture mutation card.
- Next-loop action if ambiguous: stronger eval-slice or multi-seed control card.

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
- Remaining ambiguity: model quality and inference readiness require later larger runs or architecture work.

## Next-Loop Action

- If positive: production-shaped controlled comparison card.
- If negative: C3 conditioning architecture mutation card.
- If ambiguous: stronger eval-slice or multi-seed control card.

## Novelty Notes

- Closest analogies: controlled ablation, side-channel conditioning probe.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this validates the existing C3 representation as a mapper input candidate; it does not claim a new model contribution.
