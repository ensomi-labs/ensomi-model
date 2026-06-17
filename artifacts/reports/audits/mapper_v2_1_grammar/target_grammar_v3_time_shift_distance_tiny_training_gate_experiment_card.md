# Target Grammar v3 Time-Shift Distance Tiny Training Gate Experiment Card

## Hypothesis

Adding a disabled-by-default time-shift distance auxiliary loss to the current partial-positive v3 training recipe can improve timing-bucket calibration without losing the event-continuation gains from event-token CE weighting. If the term only adds loss but does not survive training or worsens rigid/second-window rollout diagnostics, kill or mutate it before any full32 run.

## Root Objective

Decide whether `lambda_time_shift_distance` deserves escalation from synthetic Stage 1 plumbing into a bounded v3 training and rollout gate.

## Goal Decomposition

- Verify the new loss is visible and finite in a real v3 training report.
- Compare a matched v3 baseline against the enabled auxiliary on the same fixed 32-song cache slice.
- If training passes, compare a small real-audio rollout slice for legality, rigid-spacing ratio, second-window starvation, and event-count ratio.
- Route to full32 only if both training and rollout guards pass.

## Candidate Variants

- Variant A: report-only tiny training comparison, baseline CE-weight recipe versus CE-weight plus time-shift distance loss.
- Variant B: tiny training comparison plus a small rollout-pair diagnostic on selected high-risk cases.
- Variant C: full 32-case 500-step run immediately.
- Variant D: replace decode policy with a hand-written time-shift penalty instead of training the auxiliary.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
| --- | --- | --- |
| A | both runs complete, enabled loss metric finite and positive, token loss not materially worse | Required before any rollout. If it fails, kill/mutate the objective. |
| B | A passes and enabled rollouts stay legal without new rigid/starved regressions | Smallest useful escalation after Stage 1. |
| C | full32 behavior improves | Rejected for now; too expensive before A/B. |
| D | decode-only timing penalty reduces rigidity | Rejected for this card; it does not test the new training objective. |

## Selected Variant

Selected: Variant B, with report-only routing allowed when rollout pair summaries have not been produced yet.

## Selection Pressure

Variant B is selected because the previous event-token CE weight gate improved starvation and F1 but worsened rigid-grid cases. The new objective should first prove it can coexist with that CE-weight recipe and then show no early rigid/second-window regression on a tiny rollout slice.

## Minimal Change

Add reproducible baseline/enabled configs and an evaluator that compares v3 training reports plus optional rollout pairs. Do not change tokenizer behavior, model defaults, decode policy, inference defaults, or the full32 evaluator.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_baseline.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml`
- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_tiny_training_gate.py`
- `tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py`
- result summary/report under the same audit directory after execution

## Dataset Slice

Use the existing fixed 32-song/256-window eligible index and cached mapper records:

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/fixed_32song_256_v3_window_records.parquet`

The tiny training config uses a bounded eval size and cached control-teacher tensors. The rollout slice should start with 2-4 known high-risk fixed-slice cases only after both checkpoints exist.

## Baseline / Comparator

Baseline is the previous partial-positive v3 recipe:

- `event_token_loss_weight=2.0`
- `lambda_time_shift_distance=0.0`
- unchanged density/LN-close/adaptor weights

Candidate keeps the same recipe and sets:

- `lambda_time_shift_distance=0.5`
- `time_shift_distance_scale_ms=1000.0`

## Primary Metric

Training gate:

- enabled `loss/time_shift_distance` is finite and positive on eval;
- enabled token loss does not regress by more than `10%` relative to baseline.

Rollout gate:

- no illegal enabled rollouts;
- no new second-window-starved cases versus baseline;
- enabled mean dominant-spacing ratio does not exceed baseline by more than `0.02`.

## Secondary Metric

- completed steps;
- final eval total loss and token loss;
- eval valid-token count;
- event-count ratio;
- mean/median second-window event share;
- mean F1@100ms on the small rollout slice;
- rigid-case count under dominant-spacing ratio `>=0.95`.

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_enabled.yaml
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate \
  --baseline-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json \
  --enabled-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_tiny_training_gate_result_report.md
uv run --group dev pytest tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py -q
```

If the report-only route is `TEST_ROLLOUT_GATE`, run a 2-4 case rollout-pair file and rerun the evaluator with `--rollout-pairs-json`.

## Guard Check

- Existing v3 loss/model/training tests must pass.
- Existing v3 config loading must continue to accept the new loss fields.
- Candidate configs must not change mapper defaults.
- Rollout summaries must include full timepoint previews when used for timing metrics.

## Qualitative Check

Inspect worst rollout cases for repeated `160ms`/`320ms` spacing and second-window collapse. The objective should not simply increase event count while preserving a rigid grid.

## Positive Signal

Route `TEST_ROLLOUT_GATE` if training passes without rollout pairs. Route `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE` only if training passes and the tiny rollout pair comparison has no legality, new-starvation, or rigid-spacing regression.

## Negative Signal

Route `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE` if the enabled metric is missing/non-finite, token loss regresses beyond the gate, rollout legality fails, new second-window starvation appears, or rigid-spacing ratio worsens.

## Kill Criteria

- enabled `loss/time_shift_distance` is zero, missing, non-finite, or not controlled by the lambda;
- enabled token loss regresses more than `10%` on the matched tiny run;
- enabled rollout has any dead end or max-token case on the tiny slice;
- enabled rollout introduces any new starved case;
- enabled mean dominant-spacing ratio increases by more than `0.02`;
- the result depends on target-derived C3 sidecar input or future context.

## Expected Failure Modes

- The auxiliary is too small relative to CE and has no behavioral impact.
- The expected-value distance loss smooths over multimodal timing buckets.
- The loss improves time-shift logits in teacher forcing but decode still falls into rigid greedy attractors.
- Tiny-run variance dominates the report-only signal.

## Expected Runtime / Runtime Budget

Training-only gate should complete in under one hour on the local MPS setup. Stop after report-only comparison if the route is `MUTATE`; do not run rollouts.

## Confounders

- Small training horizon may not expose timing-calibration effects.
- Greedy rollout can mask teacher-forced calibration gains.
- Event-token CE weight `2.0` is a partial-positive but not a passed full32 recipe.
- The fixed 32-song slice is useful for continuity with prior gates but not a full-dataset proof.

## Result Interpretation Plan

- `TEST_ROLLOUT_GATE`: training plumbing and loss signal passed; run a tiny rollout-pair diagnostic next.
- `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE`: tiny training plus rollout behavior passed; escalate to a full32 500-step comparison.
- `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE`: do not scale as-is; consider different scale/lambda, a distributional time-shift target, or a grammar-side timing repair.

## Result Log Template

```markdown
# Target Grammar v3 Time-Shift Distance Tiny Training Gate Result

- baseline report:
- enabled report:
- baseline steps / enabled steps:
- enabled eval `loss/time_shift_distance`:
- token-loss delta:
- rollout pair count:
- enabled illegal cases:
- new starved cases:
- mean rigid delta:
- decision:
- next step:
```

## Next-Loop Action

If report-only training passes, run a 2-4 case rollout-pair diagnostic. If that passes, create a full32 500-step time-shift distance training gate. If it fails, mutate the timing objective before spending more runtime.

## Closest Analogies And Novelty Layer

Closest analogies: auxiliary regression loss over ordinal/discrete timing classes, label-distance smoothing, teacher-forced calibration probes, small-run rollout safety gates.

Novelty is not claimed. This is an engineering/research validation step for a v3 grammar objective, not a new tokenizer or model family.
