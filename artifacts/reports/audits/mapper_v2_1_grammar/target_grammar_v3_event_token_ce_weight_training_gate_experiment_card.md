# Target Grammar v3 Event-Token CE Weight Training Gate Experiment Card

## Hypothesis

The v3 fixed-slice rollout failure is partly caused by event tokens being under-ranked against time-shift tokens during teacher-forced training. A modest event-token CE weight of `2.0` should improve generated-state continuation or event-token ranking without the overpressure failures seen in density, event-budget, continuation-jump, and decode-only variants.

## Root Objective

Move v3 toward full-pipeline replacement readiness by testing whether the newly added event-token CE weighting objective transfers from local loss behavior into free-running 32-case rollout behavior.

## Idea Quality

Medium. The local objective gate passed and the failure diagnostics show event-token ranking pressure, but previous scalar event-distribution fixes failed rollout gates. This is worth one bounded training gate, not a broad sweep.

## Related Work / Analogies

- Class-weighted CE for imbalanced categorical sequence targets.
- Action-token reweighting in autoregressive event/delay grammars.
- Teacher-forced event-vs-delay calibration as a smaller alternative to decode-time event forcing.

Novelty is not claimed. This is engineering calibration for the v3 target grammar.

## Goal Decomposition

1. Keep v3 tokenizer, grammar, replay, model shape, data slice, control checkpoint, teacher cache, decode policy, and rollout manifest fixed.
2. Change only one training objective field: `loss.event_token_loss_weight=2.0`.
3. Train a fresh 500-step v3 checkpoint on the fixed 32-song/256-window slice.
4. Run the same 32-case 16s real-audio greedy rollout gate.
5. Compare against the committed 500-step baseline and killed event-pressure variants.

## Candidate Variants

### A. CE Weight 2.0

Train a fresh 500-step checkpoint with `event_token_loss_weight=2.0`, keeping `lambda_density=0.05`, `lambda_event_budget=0.0`, and `lambda_continuation_jump=0.0`.

### B. CE Weight 4.0

Stronger event-token pressure based on the local directional check.

### C. Combine CE Weight With Event Budget

Use event CE weighting plus `lambda_event_budget=0.05`.

### D. Skip Training And Return To v2.1

Treat all v3 scalar pressure as exhausted.

## Local Verification Matrix

| Variant | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | 500-step train plus 32-case rollout | Starvation or event ranking improves without legality/overgeneration regression | Same or worse starvation, overproduction, or boundary artifacts |
| B | Same gate with higher weight | Stronger event transfer than A | Repeats overpressure from density/event-budget failures |
| C | Combined loss gate | Event count and continuation improve together | Hard to attribute failures; prior event-budget already regressed |
| D | Route audit only | Avoids wasted runtime | Leaves a passed objective gate untested in rollout |

## Selected Variant

Variant A: CE weight `2.0`.

## Selection Pressure

CE `2.0` is selected because it tests the new objective with moderate pressure. CE `4.0` and combined event-budget pressure are rejected for this loop because the dominant risk is overproduction or boundary duplicate collapse.

## Minimal Code Change

No source change is planned. Add a committed audit config and run the existing `pulsefield_model.training.mapper_v3` entrypoint plus the existing runtime rollout evaluator.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_enabled.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- temporary ignored runtime artifacts under `artifacts/tmp/mapper_v3_event_token_ce_weight_training_gate/`

## Dataset Slice

Training:

- Index: `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- Mapper record cache: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`
- Timeseries: `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`
- Control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- Control-teacher cache: `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- Eval settings: `eval_fraction=0.25`, `eval_size=32`, `final_train_eval_size=16`
- Horizon: `500` steps

Rollout:

- Manifest: `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- Cases: all 32 fixed-slice beatmaps
- Prefix: `0-16000ms`
- Decode: greedy, `temperature=0.0`, no top-p, no time-shift penalties
- Max tokens per window: `512`

## Baseline / Comparator

Primary baseline: committed `target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`.

- all legal: `true`
- starved cases: `11`
- rigid cases: `7`
- median event-count ratio: `1.000`
- mean F1 @100ms: `0.640`
- max boundary-event ratio: `0.200`

Negative comparators:

- `lambda_density=0.20`: worsened starvation and legality.
- `lambda_event_budget=0.05`: max-token cases and starvation worsened.
- `lambda_continuation_jump=0.05`: dead ends and rigid cases worsened.
- decode-only sweep: killed because deterministic policies did not move the pattern and stochastic probing overproduced.

## Primary Metric

Fixed-slice rollout gate:

- all 32 rollouts legal;
- no max-token cases;
- starved cases below baseline `11`;
- rigid cases no worse than baseline `7`;
- median event-count ratio in `[0.80, 1.25]`;
- max boundary-event ratio no worse than baseline `0.200`;
- mean F1 no more than `0.03` below baseline.

## Secondary Metric

- Mean and median second-window share.
- Under/overgeneration counts.
- Dominant-spacing ratio.
- Event top-1/top-k diagnostic ratios.
- Teacher-forced final eval token loss and `phase/event_token_loss_weight`.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_enabled.yaml
# then run mapper_v3_trained_runtime_rollout over all 32 manifest rows and aggregate the same metrics as the wide audit
uv run --group dev pytest tests/models/mapper/shared/test_loss_contract.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py -q
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json >/dev/null
```

## Guard Check

- No source code changes.
- Summary must cover all 32 manifest cases.
- Any dead-end or max-token rollout is an automatic gate failure.
- Do not claim replacement readiness from this fixed-slice gate.

## Qualitative Check

Inspect worst starvation, overgeneration, and zero-reference/boundary cases. Improvement is weak if event count rises only through dense grids or boundary duplicates.

## Positive Signal

`TEST_NEXT` if CE weighting lowers starvation or improves second-window continuation while preserving all legality and event-count guards.

## Negative Signal

`MUTATE` if the run is legal but does not improve starvation or shifts the failure into overgeneration/rigidity.

## Kill Criteria

Kill CE weighting if the run has any dead-end, max-token failure, non-finite training metric, starved cases at least baseline `11`, or overgeneration/boundary artifacts worse than the baseline envelope.

## Expected Failure Modes

- Event CE weighting may improve teacher-forced event loss but not greedy rollout.
- Event pressure may overproduce events on easier charts.
- It may increase event tokens but keep the same 150/160ms rigid-grid attractor.
- Boundary duplicate failures may reappear despite same-ms event guarding.

## Expected Runtime / Runtime Budget

Expected runtime is one bounded 500-step mapper training run plus 32 real-audio runtime rollouts. Stop if training fails to finish, the checkpoint cannot load, or repeated early rollouts hit max tokens or dead-end.

## Confounders

- Fixed-slice training is not held-out evidence.
- F1 can reward regular grids.
- The event CE class weight is teacher-forced and may not transfer under generated-state exposure.
- The baseline checkpoint is a separate fresh run, not the same checkpoint continued.

## Result Interpretation Plan

- If positive, run a broader v3 training/rollout audit with event CE weighting.
- If neutral but stable, compare CE `2.0` against a longer horizon or stronger ordered timing objective.
- If negative, kill simple event CE pressure and pivot to grammar-level continuation or v2.1 improvement.

## Result Log Template

- Training command and final losses.
- Rollout command pattern.
- Aggregate 32-case metrics.
- Baseline comparison.
- Worst cases.
- Guard results.
- Decision.
- Next-loop action.

## Next-Loop Action

Use the result to decide whether v3 continues with event CE weighting, mutates to grammar-level continuation, or pauses in favor of v2.1 grammar improvement.
