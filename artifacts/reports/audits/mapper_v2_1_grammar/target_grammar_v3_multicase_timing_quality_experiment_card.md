# Target Grammar v3 Multicase Timing-Quality Experiment Card

## Hypothesis

The 200-step v3 checkpoint has moved past zero-event collapse, but its free-running outputs may still rely on a crude timing prior rather than audio-conditioned beatmap timing. A small multicase real-audio diagnostic should reveal whether the rigid-grid/second-window artifact is local to one song/difficulty or systematic across the fixed training slice.

## Root Objective

Harden the v3 path toward full-pipeline replacement by measuring timing quality and window-continuation behavior across multiple audio/map cases before changing loss, decode policy, or training scale.

## Goal Decomposition

- Evaluate the existing 200-step v3 checkpoint without changing representation or training.
- Use real audio from the same fixed 32-beatmap/256-window slice.
- Pass normalized difficulty from the selected reference beatmap rather than using the previous default.
- Measure generated event count, timing-match F1, rigid-grid ratio, boundary-event ratio, and second-window event share.

## Candidate Variants

- Variant A: run alpha `0.00` real-audio rollouts for three selected audio/map cases at `8000` ms and `16000` ms, then compare generated event timing against the selected reference beatmap.
- Variant B: run a new longer training job before additional timing diagnostics.
- Variant C: add loss/decode calibration before measuring whether the timing artifact is systematic.
- Variant D: run a full 15-audio fixed-slice timing audit immediately.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | 3 cases x 2 prefixes with existing checkpoint | Stable legality plus timing-quality metrics across sparse/moderate/dense references | Runtime fails or metrics do not expose timing/window artifact |
| B | Longer training | Might improve quality | Expensive before knowing whether artifact is systematic |
| C | Loss/decode mutation | Might reduce rigid grid | Changes mechanism before measuring baseline failure surface |
| D | Full fixed-slice audit | More representative | Too broad before proving metric utility |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is the smallest diagnostic that directly tests the failure surfaced by the 200-step gate. It keeps the checkpoint, alpha, runtime path, and target grammar fixed while broadening the evaluation from one prefix to three audio/map cases and two prefix lengths.

## Minimal Change

- No tokenizer, grammar, model, loss, or training changes.
- Run existing `mapper_v3_trained_runtime_rollout` diagnostics with `alpha=0.00`.
- Use `--normalized-difficulty` computed from each selected reference beatmap difficulty.
- Aggregate timing-quality metrics into a tracked result report and summary.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_multicase_timing_quality_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_multicase_timing_quality_summary.json`
- This experiment card.

## Dataset Slice

Use the existing fixed-slice cases:

| Case | Audio | Reference beatmap | Difficulty |
| --- | --- | --- | ---: |
| sparse | `dataset/0/1000722/audio.mp3` | `dataset/0/1000722/Ohara Yuiko - Zero Centimeters (TV Size) (-Mikan) [Asha's HD].osu` | `3.09` |
| moderate | `dataset/0/1004416/audio.mp3` | `dataset/0/1004416/namirin - Kanzen ShouriEsper Girl (tailsdk) [Ash's Insane].osu` | `3.84` |
| dense | `dataset/0/1008095/Knight Rider.mp3` | `dataset/0/1008095/USAO - Knight Rider (Kuo Kyoka) [CS' Lone Sonorous].osu` | `4.65` |

Prefixes: `8000` ms and `16000` ms.

## Baseline / Comparator

The 200-step single-song diagnostic:

- Alpha `0.00` generated `50` events over `0-16000` ms.
- Timing pattern was mostly a rigid `160ms` grid from `160` to `7840` ms plus one late event at `15980` ms.
- Timing F1 at `100ms` was about `0.51-0.52` against two reference maps in the same beatmap set.

## Primary Metric

Timing-match F1 at `100ms` tolerance against the selected reference beatmap.

## Secondary Metric

- Generated/reference event-count ratio.
- Rigid-grid event ratio: share of adjacent generated spacings equal to the dominant spacing.
- Boundary-event ratio within `100ms` of an 8s window boundary.
- Second-window event share for the `16000` ms rollout.
- Event top-1/top-5 counts and best-event margin.
- Completion, dead-end, and max-token flags.

## Verify Command Or Evaluation Procedure

Run the existing v3 real-audio rollout diagnostic for each case/prefix:

- checkpoint: `artifacts/tmp/mapper_v3_200step_decode_policy/train/run/checkpoint.pt`
- control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- alpha: `0.00`
- temperature: `0.0`
- logit diagnostics enabled
- timepoint preview limit: `1024`

Then aggregate the six JSON outputs against parsed reference beatmap timepoints.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q`

## Qualitative Check

Inspect whether generated timing follows reference-like sparse/dense structure or collapses to repeated fixed spacing, boundary spikes, or window starvation.

## Positive Signal

- Timing F1 improves beyond the previous `~0.52` single-song baseline on multiple cases.
- Generated/reference event-count ratio is not extreme.
- Second-window event share is nonzero and not dominated by boundary spikes.
- No legality failures.

## Negative Signal

- Rigid-grid ratio is high across cases.
- Generated outputs ignore difficulty/density differences.
- `16000` ms rollouts put most events in the first 8s or at boundaries.
- Timing F1 stays weak across cases.

## Kill Criteria

If timing artifacts are systematic across cases, do not escalate v3 replacement training yet. Mutate to timing/loss/decode calibration or compare against v2.1 generated timing quality before broader v3 runs.

## Expected Failure Modes

- BeatThis real-audio preparation dominates runtime.
- Some selected maps may have chart timing offsets that make direct event-time matching imperfect.
- A three-case sample cannot prove full 4K quality.

## Expected Runtime / Runtime Budget

Expected runtime is under 30 minutes for six real-audio rollouts plus aggregation. Stop if any rollout fails legality or if the checkpoint is missing.

## Confounders

- The checkpoint was trained only 200 steps.
- The selected cases come from the same fixed training slice.
- Reference beatmap timing is a proxy for quality, not a complete music-to-chart metric.
- BeatThis timing fit differences may affect conditioning.

## Result Interpretation Plan

- If timing artifacts are systematic, v3 needs timing-aware calibration before broader replacement.
- If only one case fails, inspect audio/timing/reference alignment for that case.
- If multicase timing looks acceptable, proceed to a matched v2.1-v3 generated timing comparison.

## Result Log Template

- Cases:
- Prefixes:
- Rollout legality:
- Timing F1 table:
- Rigid-grid ratio:
- Boundary ratio:
- Second-window share:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use this audit to choose between v3 timing calibration, a matched v2.1-v3 generated-quality comparison, or broader v3 training.

## Closest Analogies And Novelty Layer

Closest analogies are sequence-generation timing audits, free-running exposure diagnostics, and music-conditioned event alignment metrics. There is no novelty claim here; this is an evaluation-hardening pass for the v3 target representation.
