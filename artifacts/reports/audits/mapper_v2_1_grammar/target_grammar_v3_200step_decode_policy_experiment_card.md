# Target Grammar v3 200-Step Decode-Policy Experiment Card

## Hypothesis

If the v3 zero-event real-audio rollout is mostly caused by undertraining and greedy time-shift preference, then a 200-step cache-backed v3 run on the existing fixed slice should improve token loss and increase event emission or event logit rank on the same 16s real-audio prefix without changing grammar semantics.

## Root Objective

Harden the v3 path toward full-pipeline replacement by testing whether more training alone improves free-running event generation enough to justify the next larger training/inference pass.

## Goal Decomposition

- Keep the representation fixed: v3 event groups remain reversible, lower-token teacher-forcing targets.
- Isolate training horizon from dataset/config changes.
- Compare the 200-step checkpoint against the 20-step real-audio decode diagnostic.
- Measure legality, event count, boundary artifact ratio, and logit-category calibration under the same decode-policy sweep.

## Candidate Variants

- Variant A: train v3 for 200 MPS steps on the same fixed 32-beatmap/256-window cache-backed slice, then run real-audio alpha sweep `0.00`, `0.05`, `0.10`, `0.20`.
- Variant B: change decode policy only, without more training.
- Variant C: change loss/event weighting before another training run.
- Variant D: expand to a larger dataset slice immediately.

## Local Verification Matrix

| Variant | Smallest check | Pass evidence | Reject evidence |
| --- | --- | --- | --- |
| A | 200-step training plus same real-audio alpha sweep | Loss improves and event emission/logit rank improves without legality failure | Still zero or boundary-only events with poor margins |
| B | Sweep current 20-step checkpoint only | Already done; showed decode sensitivity but boundary artifacts | Cannot answer whether training horizon helps |
| C | Loss-weighted short run | Could increase events | Changes two variables before proving training horizon limits |
| D | Larger slice run | More representative | Costs more and hides whether current small-slice training can learn |

## Selected Variant

Variant A.

## Selection Pressure

Variant A is the smallest training escalation that directly follows the previous diagnostic. It keeps the dataset, config family, cache, audio, and decode metrics stable while increasing the training signal by 10x over the prior 20-step checkpoint.

## Minimal Change

- No tokenizer, grammar, model architecture, or loss changes.
- Train a new v3 checkpoint for `200` steps using the existing fixed slice and cache-backed setup.
- Run the existing real-audio v3 rollout diagnostic with policy alphas `0.00`, `0.05`, `0.10`, `0.20`.
- Write a result report and summary comparing against the 20-step diagnostic baseline.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_200step_decode_policy_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_200step_decode_policy_summary.json`
- This experiment card.

## Dataset Slice

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Train/eval split: same deterministic split style as the 20-step run, `eval_fraction=0.25`, `eval_size=32`, `final_train_eval_size=16`.
- Real audio: `dataset/0/1000722/audio.mp3`, prefix `0-16000` ms.

## Baseline / Comparator

The committed 20-step checkpoint diagnostic:

- Eval token loss: `3.120049`.
- Greedy alpha `0.00`: `0` event timepoints.
- Alpha `0.05+`: `2` event timepoints at `7950` ms and `15950` ms, both boundary artifacts.
- Baseline event-valid steps: `214/215`; event top-5: `108/215`; event top-1: `0/215`.

## Primary Metric

Real-audio generated event timepoint count at alpha `0.00`.

## Secondary Metric

- Eval token loss and train/eval loss trend.
- Event count by alpha.
- Boundary event ratio within `100` ms of an 8s window end.
- Event-valid step ratio, event top-1/top-5 counts, best-event rank, and best-event margin.
- Completion, dead-end, and max-token flags.

## Verify Command Or Evaluation Procedure

1. Train v3 for `200` MPS steps using the existing fixed slice and control-teacher cache.
2. Run real-audio rollout diagnostics for alphas `0.00`, `0.05`, `0.10`, `0.20`.
3. Aggregate training metrics and rollout diagnostics into tracked Markdown/JSON reports.

## Guard Check

- `uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q`
- `uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q`
- `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q`

## Qualitative Check

Inspect whether generated events move away from exact window-boundary artifacts. Non-empty but boundary-only output remains a failure for quality readiness.

## Positive Signal

- Alpha `0.00` produces nonzero events, or event top-1/top-5 and margins improve materially over the 20-step baseline.
- Events are not only within `100` ms of `8000`/`16000` ms boundaries.
- No rollout dead-end or max-token failure occurs.

## Negative Signal

- Alpha `0.00` remains zero-event and penalty outputs remain boundary-only.
- Eval loss improves but free-running event behavior does not, indicating exposure/decode/loss calibration rather than plain undertraining.
- Any policy setting causes dead-end or max-token failure.

## Kill Criteria

If the 200-step run still produces zero or boundary-only events with weak event margins, do not escalate immediately to full replacement. Mutate to loss/decode calibration or a larger matched v2.1-v3 quality comparison before broader training.

## Expected Failure Modes

- MPS runtime is longer than expected.
- Training remains too short for meaningful quality despite loss improvements.
- Real-audio prefix is not representative.
- Stronger event logits may still appear only at window boundaries.

## Expected Runtime / Runtime Budget

Expected runtime is under 90 minutes on local MPS for training plus four real-audio rollouts. Stop if training fails, produces non-finite metrics, or any rollout fails legality.

## Confounders

- The slice is small and fixed.
- The real-audio check uses one 16s prefix.
- v3 and v2.1 token losses are not directly comparable across vocabularies.
- Decode-policy penalties are diagnostic, not final generation policy.

## Result Interpretation Plan

- If 200 steps produces non-boundary events under alpha `0.00`, continue to a longer matched v2.1-v3 real-song inference comparison.
- If only penalty settings produce events and those remain boundary artifacts, mutate toward decode/loss calibration.
- If loss improves but event behavior does not, treat this as evidence that training horizon alone has diminishing marginal return.
- If legality regresses, repair runtime/grammar before additional training.

## Result Log Template

- Training command:
- Completed steps:
- Eval token loss:
- Train token loss:
- Alpha sweep table:
- Boundary event ratio:
- Guards:
- Decision: `TEST_NEXT` / `MUTATE` / `KILL`
- Next-loop action:

## Next-Loop Action

Use the result to choose between longer matched training/inference, decode/loss calibration, or v2.1 grammar improvement if v3 free-running quality remains boundary-only.

## Closest Analogies And Novelty Layer

Closest analogies are constrained autoregressive sequence generation, teacher-forcing versus free-running exposure diagnostics, and length-penalty calibration. There is no novelty claim here; this is evidence hardening for the v3 event-group target representation.
