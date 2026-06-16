# C3 Mapper Conditioning Training Smoke Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: P6 proved C3 can condition Mapper v2.1 logits and receive gradients; the next evidence needed is a tiny real training smoke with the exact P5 sidecar.
- Acceptance source, if any: active C3 full-pipeline goal plus `c3_mapper_conditioning_probe_result_report.md`.
- Source snapshot / evidence grade: strong local integration evidence; no training-run evidence yet.

## Hypothesis

On a tiny real mapper training slice, exact C3 sidecar tensors can be loaded, batched, and used by Mapper v2.1 conditioning without runtime/config failure, and the short-run loss trace will be at least comparable to the no-conditioning baseline.

## Root Objective

Move C3 from model-unit integration into a real training loop smoke so the new tokenization is closer to full-pipeline use.

## Goal Decomposition

- Subgoal 1: run a baseline tiny mapper training smoke with exact C3 tensors carried but C3 conditioning disabled.
- Subgoal 2: run a matching tiny mapper training smoke with pooled C3 conditioning enabled.
- Subgoal 3: compare completion, final eval loss, train/eval history, sidecar coverage settings, and runtime stability.

## Candidate Variants

- Variant A: unit-test-only synthetic training loss probe.
- Variant B: tiny real training smoke using existing cached mapper windows, control-teacher tensors, and exact C3 sidecar.
- Variant C: longer controlled training comparison.
- Variant D: full production config training run.

## Local Verification Matrix

- Variant A: already covered by P6; not enough after model integration.
- Variant B: smallest run that exercises real data, sidecar loading, batching, model forward, loss, optimizer, checkpoint/report writing.
- Variant C: useful later, but premature before smoke stability is known.
- Variant D: too expensive and not justified before a tiny smoke passes.

## Selected Variant

- Selected: Variant B, two-run tiny real training smoke.
- Rejected: A is insufficient; C and D spend too much runtime before smoke evidence.
- Why this is the smallest useful test: it uses the real exact sidecar and real mapper training loop while keeping model size, steps, eval size, and device small.

## Selection Pressure

- Primary pressure: both baseline and C3-conditioned runs complete and write reports/checkpoints.
- Guard pressure: no default production config changes; existing verifier tests still pass.
- Runtime pressure: small CPU config, 2 steps, batch size 1, small eval/train-eval slices.
- Kill pressure: stop if sidecar loading, C3 token ids, training config, or optimizer path fails.

## Research Question

Can the exact C3 sidecar be used in an actual Mapper v2.1 training loop without instability, and does the tiny loss trace avoid an immediate negative signal versus the no-conditioning baseline?

## Closest Analogies / Novelty Layer

- Closest analogies: auxiliary feature training smoke, side-channel embedding ablation, small-scale integration canary.
- Relevant taxonomy bucket: representation engineering validation.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: this is integration/runtime evidence for C3, not a modeling contribution.

## Minimal Change

Add reproducible smoke YAML configs and run the existing `pulsefield_model.training.mapper_v2_1` entrypoint twice:

- baseline: `include_c3_side_stream_token_tensors=true`, `use_c3_side_stream_conditioning=false`,
- variant: same data path, `use_c3_side_stream_conditioning=true`.

No model or dataset code changes are planned in this card.

## Files Likely to Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_baseline.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_enabled.yaml`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_result_report.md`
- optional JSON summary under the same audit directory.

## Read-Only Context Files

- `src/pulsefield_model/training/mapper_v2_1.py`
- `src/pulsefield_model/models/mapper/v2_1/model.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_probe_result_report.md`

## Dataset Slice

Use the existing LE<=3 mapper cache and control-teacher cache:

- dataset root: `dataset`,
- mapper record cache: `artifacts/cache/stage2_mapper_v2_1/window_records/stage2_mapper_v2_1_phase_b_sparse_global_mps_plus_end.parquet`,
- control teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`,
- exact sidecar: `artifacts/cache/c3_mapper_window_sidecar/c3_exact_mapper_window_sidecar_le3.json`,
- max steps: 2,
- eval size: 2,
- final train eval size: 2,
- batch size: 1,
- device: CPU.

## Baseline / Comparator

Baseline carries C3 tensors through data/training but keeps `use_c3_side_stream_conditioning=false`.

Comparator enables pooled C3 conditioning with:

- `c3_side_stream_vocab_size=14294`,
- `c3_side_stream_embedding_dim=16`,
- `c3_side_stream_scale_init=0.03`.

## Primary Metric

- Both runs complete 2/2 training steps and write `report.json`.
- Compare final eval `loss/total` and final train-eval `loss/total`.

## Secondary Metric

- Runtime and final train step loss.
- Dataset report confirms C3 sidecar path, max tokens, and tensor inclusion.
- Model config confirms conditioning disabled vs enabled.
- History contains eval entries.

## Verify Command / Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v2_1 --config artifacts/reports/audits/context_adaptive_fallback_codec/c3_mapper_conditioning_smoke_enabled.yaml
uv run --group dev pytest tests/models/mapper/v2_1/test_model.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Do not edit production training configs.
- Do not commit generated checkpoints.
- Sidecar artifact remains ignored; commit reports/configs only.
- No claim of model quality improvement from a 2-step smoke.

## Qualitative Check

Inspect both training reports. They should differ only in output/run names and model C3 conditioning fields, with the exact sidecar configured in dataset metadata.

## Positive Signal

- Both runs complete.
- C3-enabled run does not immediately fail or produce non-finite loss.
- C3-enabled final eval loss is not dramatically worse than the disabled baseline.

## Negative Signal

- Sidecar loading makes the run fail or too slow.
- Training rejects C3 model config.
- C3-enabled loss is non-finite or far worse on the same tiny setup.
- Reports do not prove C3 tensors were included.

## Kill Criteria

Kill this path if the C3-enabled run cannot complete 2 steps, produces non-finite loss, or is more than 25% worse than baseline final eval loss on this tiny slice.

## Expected Failure Modes

- Control-teacher cache missing for selected records.
- Sidecar lookup misses many windows.
- CPU runtime slower than expected because sidecar JSON load is large.
- Tiny 2-step loss is noisy and should not be overinterpreted.

## Confounders

- Two-step loss is a stability smoke only, not a quality metric.
- Baseline and variant may see different random batches unless seed and configs are matched.
- A tiny CPU model differs from the production mapper model.

## Expected Runtime / Runtime Budget

Expected under 10 minutes total. Stop and report if either run clearly hangs or fails.

## Result Interpretation Plan

- Positive result would suggest: run a longer controlled C3/no-C3 comparison on a small fixed slice.
- Negative result would suggest: inspect sidecar coverage/config or mutate conditioning.
- Ambiguous result would require: repeat with a slightly larger fixed eval slice or controlled synthetic signal.
- Human owner decides: whether the tiny smoke is enough to spend longer training runtime.
- Next-loop action if positive: create small controlled training comparison card.
- Next-loop action if negative: repair runtime/config/conditioning path.
- Next-loop action if ambiguous: repeat with stronger controls.

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
- Remaining ambiguity: model quality still needs a longer controlled run.

## Next-Loop Action

- If positive: run small controlled C3/no-C3 training comparison.
- If negative: repair or mutate conditioning path.
- If ambiguous: strengthen smoke controls.

## Novelty Notes

- Closest analogies: tiny ablation canaries and side-channel feature smokes.
- Novelty layer, if any: not claimed.
- Representation novelty vs engineering variation: engineering validation for full-pipeline C3 use.
