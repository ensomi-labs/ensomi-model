# Target Grammar v3 Decode-Policy Continuation Sweep Experiment Card

## Hypothesis

The 200-step v3 checkpoint's rigid-grid timing and sparse/dense second-window starvation may be partly controlled by decode policy rather than by target grammar or training alone. A small sweep over existing deterministic and stochastic decode knobs should determine whether decode-only calibration is worth pursuing before changing loss or training.

## Root Objective

Decide whether target grammar v3 can make measurable progress toward full-pipeline replacement with existing online decode knobs, or whether the next mutation must move to mapper training/loss calibration.

## Goal Decomposition

- Separate shared rigid-grid collapse from v3-specific second-window starvation.
- Test existing runtime knobs without changing tokenizer, grammar, model, or training.
- Measure whether a decode policy can reduce dominant-spacing collapse while preserving legal generation and second-window continuity.
- Record kill criteria for decode-only calibration if it cannot improve the proven failure modes.

## Candidate Variants

- Variant A: deterministic policy sweep over flat and delta-scaled time-shift penalties on the three 16s multicase rollouts.
- Variant B: stochastic sampling sweep with temperature/top-p and fixed seeds on the same 16s cases.
- Variant C: train longer or change loss weights before any decode sweep.
- Variant D: change v3 grammar/token semantics to encode timing differently.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | Existing checkpoint, three 16s cases, four deterministic policies | Lower rigid-grid ratio and no sparse/dense starvation | Same 160ms grid, boundary duplication, or worse legality |
| B | Same cases, one stochastic policy and two seeds | Stable non-rigid timing without legality failures | Seed-sensitive artifacts or no structural improvement |
| C | New training run | Could address calibration directly | Too expensive before proving decode knobs are insufficient |
| D | Grammar mutation | Could reduce timing prior | Too large before decode-only path is killed |

## Selected Variant

Variant A plus one minimal stochastic probe from Variant B.

## Selection Pressure

The selected variant is the smallest experiment that can falsify decode-only calibration. Deterministic policies test existing production-shaped controls; the stochastic probe checks whether the rigid grid is only a greedy argmax artifact. If both fail, training/loss calibration becomes the better next card.

## Minimal Change

No model, tokenizer, grammar, or training-code changes. Run the existing `mapper_v3_trained_runtime_rollout` evaluator on the 200-step v3 checkpoint and aggregate the same timing-quality metrics used by the multicase audit.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json`

## Dataset Slice

- v3 checkpoint: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/checkpoint.pt`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Cases: the three `16000ms` real-audio cases from `artifacts/tmp/mapper_v3_multicase_timing_quality/manifest.json`.
- Difficulties: selected-map normalized difficulties from that manifest.

## Baseline / Comparator

- v3 multicase 16s baseline at alpha `0.00`: sparse/dense second-window share `0.02`, moderate second-window share `0.494`, mean dominant-spacing ratio at least `0.98`.
- Matched v2.1 baseline: same rigid-grid collapse but no 16s second-window starvation.

## Primary Metric

Policy pass/fail on both:

- Mean dominant-spacing ratio across 16s cases.
- Sparse/dense 16s second-window event share.

## Secondary Metric

- Timing F1 @100ms.
- Generated/reference event-count ratio.
- Boundary-event ratio.
- Completion/dead-end/max-token flags.
- Event top-1/top-k logit diagnostics.
- First/last generated timing previews.

## Verify Command Or Evaluation Procedure

1. Run three 16s cases for deterministic policies:
   - baseline: `length_alpha=0.00`, `delta_alpha=0.00`
   - flat mild: `length_alpha=0.05`, `delta_alpha=0.00`
   - delta mild: `length_alpha=0.00`, `delta_alpha=0.50`
   - combined mild: `length_alpha=0.05`, `delta_alpha=0.50`
2. Run one stochastic probe:
   - `temperature=0.80`, `top_p=0.95`, `length_alpha=0.00`, `delta_alpha=0.00`, fixed seed.
3. Aggregate metrics against the selected reference beatmaps.
4. Compare each policy to the v3 multicase baseline and matched v2.1 baseline.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/models/mapper/v3/test_model.py -q`
- Validate the result summary JSON with `uv run python -m json.tool`.

## Qualitative Check

Inspect whether policy changes create plausible timing variation or merely introduce boundary duplicates, denser fixed grids, repeated same-time events, or unstable stochastic artifacts.

## Positive Signal

At least one policy:

- keeps all three rollouts legal,
- reduces mean dominant-spacing ratio below `0.85`,
- gives sparse and dense 16s second-window share at least `0.25`,
- keeps boundary-event ratio at or below `0.05`,
- and does not reduce mean F1 @100ms by more than `0.05` versus alpha `0.00`.

## Negative Signal

- Deterministic policies keep dominant-spacing ratio above `0.95`.
- Sparse/dense second-window share remains below `0.10`.
- Penalties mainly create events at `7990/15980/16000` boundaries.
- Stochastic sampling is seed-sensitive or illegal, or lowers rigid-grid ratio only by producing incoherent timing.

## Kill Criteria

If no policy satisfies the positive signal, kill decode-only calibration as the next replacement path for v3 and move to a bounded training/loss calibration experiment with these metrics as guards.

## Expected Failure Modes

- Flat penalty may add boundary duplicates rather than structural events.
- Delta penalty may prefer shorter shifts and make the grid denser.
- Sampling may break the grid without learning music-conditioned timing.
- F1 may look acceptable because dense grids hit nearby reference events.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes for 15 rollouts, aggregation, and guards. Stop on missing checkpoint, non-legal baseline, or repeated runtime failure.

## Confounders

- The checkpoint is still undertrained at 200 steps.
- The three-case slice is diagnostic, not a full dataset audit.
- Timing F1 is only a proxy for chart quality.
- Stochastic generation with one seed cannot prove stable production behavior.

## Result Interpretation Plan

- If deterministic policy passes, promote it to a wider multicase audit and consider runtime default implications.
- If only stochastic policy improves metrics, mutate into a stability/seed-sensitivity card before using it.
- If all policies fail, declare decode-only calibration insufficient and create a training/loss calibration card.

## Result Log Template

- Policy table:
- Best policy:
- Baseline comparison:
- v2.1 comparison:
- What passed:
- What surfaced:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the result to choose between a wider decode policy audit, stochastic stability probe, or a bounded v3 training/loss calibration experiment.

## Closest Analogies And Novelty Layer

Closest analogies are autoregressive decoding calibration, exposure-bias diagnostics, and greedy-versus-sampling ablation. There is no novelty claim; this is an engineering verification step for the target grammar v3 replacement path.
