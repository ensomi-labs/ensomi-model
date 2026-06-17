# Mapper v2.1/v3 Fixed-Slice Decode Comparison Experiment Card

## Hypothesis

The six-case v2.1 anti-rigid spacing guard result may generalize across the fixed 32-map slice. If it does, v2.1 plus the opt-in hard-block guard should reduce rigid-grid collapse versus matched v2.1 baseline without introducing starvation or legality failures, and it should compare favorably against the existing v3 500-step fixed-slice wide audit on continuation and timing proxies.

## Root Objective

Decide whether the next grammar-improvement loop should continue scaling the v2.1 anti-rigid decode/grammar path, or treat the six-case signal as too local and return to v3/C3-side representation or training diagnostics.

## Goal Decomposition

- Reuse the exact 32-map, 16s case universe from the v3 500-step fixed-slice wide audit.
- Run v2.1 baseline and v2.1 anti-rigid guard with the same v2.1 checkpoint, control checkpoint, real audio, and selected-map normalized difficulty.
- Compare v2.1 baseline, v2.1 guard, and existing v3 500-step metrics on legality, second-window starvation, rigid-grid dominance, event-count ratio, and timing F1.
- Decide whether anti-rigid guard should advance to a larger decode comparison, mutate into a softer/richer grammar constraint, or be killed as a local artifact.

## Candidate Variants

- Variant A: rerun v2.1 baseline and v2.1 anti-rigid guard on all 32 v3-wide-audit maps at `16000ms`, then compare against the existing v3 500-step summary.
- Variant B: train a matched 500-step v2.1 checkpoint before the same 32-map comparison.
- Variant C: rerun v3 500-step and v2.1 in one fresh combined evaluator.
- Variant D: jump directly to held-out or full-4K v2.1/v3 decode comparison.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | 64 v2.1 rollouts, using the existing v3 32-case summary as read-only comparator | Guard reduces v2.1 rigidity on most cases, avoids new starvation, and is competitive with v3 continuation | Guard only helps handpicked cases or regresses timing/continuation |
| B | New 500-step v2.1 training plus 64 rollouts | Removes training-step confound | Adds runtime and training variance before knowing if the guard generalizes |
| C | 96 fresh rollouts | Perfectly fresh comparison artifact | Repeats an already committed v3 wide audit without answering a new question |
| D | Large decode audit | Closer to replacement evidence | Too expensive before fixed-slice guard generalization is known |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is the smallest test that can refute the six-case anti-rigid result as selection bias. It keeps the v3 comparator fixed, avoids another training run, and expands v2.1 evidence from six selected cases to the complete fixed-slice map universe.

## Minimal Change

- Add a bounded evaluator that reads `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json` for the 32-case universe and v3 comparator metrics.
- Run `run_trained_v21_runtime_rollout_smoke` twice per case: baseline and opt-in `MapperV21AntiRigidSpacingLogitsTransform`.
- Reuse the existing generated/reference timing metrics from the v2.1 anti-rigid gate.
- Write one summary JSON and one result report.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v21_v3_fixed_slice_decode_comparison.py`
- `tests/evals/test_mapper_v21_v3_fixed_slice_decode_comparison.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`

## Dataset Slice

- Case universe: the `runs` list in `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`.
- Unique maps: `32`.
- Prefix: `0-16000ms`.
- Audio and beatmap paths: the paths recorded per v3 wide-audit run.
- Difficulty: each run's recorded `normalized_difficulty`.

## Baseline / Comparator

- v2.1 baseline: `artifacts/tmp/mapper_v21_200step_timing_baseline/train/run/checkpoint.pt`, no anti-rigid transform.
- v2.1 guard: the same checkpoint with `MapperV21AntiRigidSpacingLogitsTransform`.
- v3 comparator: committed `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`, checkpoint `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`.

## Primary Metric

- Guard generalization: number of 32 cases where v2.1 guard lowers dominant-spacing ratio relative to v2.1 baseline.
- Continuation safety: new v2.1 guard starved cases relative to v2.1 baseline.
- Cross-grammar comparator: v2.1 guard starved count and mean F1 versus existing v3 500-step wide audit.

## Secondary Metric

- All-rollout legality.
- Mean and median timing F1 @100ms.
- Mean and median event-count ratio.
- Mean and median dominant-spacing ratio.
- Rigid-case count using dominant-spacing ratio `>= 0.95`.
- Boundary-event ratio and duplicate/non-increasing spacing ratio.
- Worst cases by low F1, high rigidity, and starvation.

## Verify Command Or Evaluation Procedure

1. Read the v3 wide-audit summary and validate it has 32 runs.
2. Load the v2.1 and control checkpoints recorded by the matched v2.1 timing baseline.
3. Run baseline and guarded v2.1 rollouts for every v3 case with `real_audio=True`, `temperature=0.0`, and `max_tokens_per_window=512`.
4. Aggregate metrics and compare to the existing v3 wide audit.
5. Validate the summary JSON.

## Guard Check

- `uv run --group dev pytest tests/evals/test_mapper_v21_v3_fixed_slice_decode_comparison.py tests/evals/test_mapper_v2_1_anti_rigid_spacing_guard.py tests/evals/test_mapper_v2_1_trained_runtime_rollout.py tests/inference/test_mapper_v2_1_rollout.py -q`
- `uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_v3_fixed_slice_decode_comparison_summary.json`

## Qualitative Check

Inspect whether wins or failures cluster by audio family, difficulty band, sparse reference support, or zero-reference prefixes. Treat a high F1 caused by dense regular grids as weak positive evidence unless rigidity also improves.

## Positive Signal

- All v2.1 baseline and guard rollouts are legal.
- Guard improves dominant-spacing ratio in at least `16/32` cases.
- Mean guard dominant-spacing ratio improves over v2.1 baseline by at least `0.10`.
- Guard mean F1 does not regress from v2.1 baseline by more than `0.05`.
- Guard introduces no more than `2` new starved cases.
- Guard has fewer starved cases than the v3 500-step wide audit and mean F1 at least `v3_mean_f1 - 0.05`.

## Negative Signal

- Any v2.1 guarded rollout is illegal.
- Guard improves rigidity in fewer than `8/32` cases.
- Guard mean F1 regresses by more than `0.05`.
- Guard introduces more than `2` new starved cases.
- Guard still looks like a mostly fixed grid across the 32-map slice.

## Kill Criteria

Kill hard-block scaling if legality fails, starvation regresses materially, or the guard does not improve rigidity outside the six selected examples. If rigidity improves but event-count or F1 behavior degrades, mutate toward a softer penalty or richer timing-state grammar instead of changing defaults.

## Expected Failure Modes

- The 200-step v2.1 checkpoint and 500-step v3 checkpoint are not training-budget matched, so cross-grammar superiority remains suggestive rather than final.
- Timing F1 can reward regular grids in dense sections.
- Some prefixes have sparse or zero reference events, making starvation/F1 interpretation weaker.
- Real-audio runtime may be slow because the evaluator runs 64 v2.1 rollouts.

## Expected Runtime / Runtime Budget

Expected runtime is under 45 minutes on local `auto` device. Stop if checkpoint loading fails, if more than one repeated runtime exception occurs, or if the run exceeds 90 minutes.

## Confounders

- The fixed slice is training-slice evidence, not held-out or full-4K evidence.
- The comparison tests decode behavior, not teacher-forcing loss or replacement readiness.
- The anti-rigid guard is opt-in and does not change model weights or tokenizer bits.
- v3's lower target-bit property is not retested here; this is runtime decode quality only.

## Result Interpretation Plan

- If positive, advance to a wider held-out or larger fixed-slice decode comparison before considering defaults.
- If mixed, identify whether hard blocking should mutate into a finite penalty or state-aware grammar change.
- If negative, treat the six-case result as local and refocus on v3/C3 diagnostics or a different v2.1 grammar improvement.

## Result Log Template

- v3 comparator summary:
- v2.1 baseline aggregate:
- v2.1 guard aggregate:
- Guard-vs-baseline deltas:
- Guard-vs-v3 deltas:
- Worst cases:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the result to choose between a larger v2.1 anti-rigid decode audit, a softer/richer v2.1 grammar mutation, or returning to v3/C3-side representation/training diagnostics.

## Closest Analogies And Novelty Layer

Closest analogies are constrained decoding, repetition blocking, timing-grid regularization, and fixed-slice model selection. There is no novelty claim; this is an engineering verification step for runtime grammar behavior.
