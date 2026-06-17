# Target Grammar v3 Time-Shift Distance Loss Stage 1 Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_shared_timing_residue_structure_audit_summary.json` routed to `TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION`: v2.1/v3 target streams preserve rich timing residue while generated v2.1/v3 rollouts collapse to rigid grids.
- Acceptance source, if any: `target_grammar_shared_timing_residue_structure_audit_summary.json`.
- Source snapshot / evidence grade: strong target-stream parity evidence, medium generated-state evidence, strong negative evidence against event-signature factorization and decode-only timing sweeps.

## Hypothesis

If rigid generated timing is partly caused by weak calibration among time-shift tokens, then a disabled-by-default time-shift distance auxiliary loss should produce a finite local signal that penalizes rigid wrong-shift logits more than matching target time-shift logits, while preserving existing behavior at zero lambda. If this cannot be proven synthetically, do not run a training gate.

## Root Objective

Test the smallest loss-side timing calibration surface before changing v3 grammar, mapper embeddings, or running another training/rollout gate.

## Goal Decomposition

- Subgoal 1: Add a local auxiliary loss that only uses teacher-forced logits, target tokens, and the existing shared time-shift vocabulary.
- Subgoal 2: Prove disabled-default compatibility for current v3/v2.1 training configs.
- Subgoal 3: Prove the loss distinguishes matching target timing from rigid wrong-shift timing logits.
- Subgoal 4: Route the next loop to a tiny training/rollout gate only if Stage 1 passes.

## Candidate Variants

- Variant A: Uniformly increase all time-shift token CE weight.
- Variant B: Add expected time-shift distance loss at gold time-shift positions.
- Variant C: Add beat-phase or time-residue embeddings to the mapper model.
- Variant D: Implement a timing grammar mutation immediately.

## Local Verification Matrix

| Candidate | Smallest Check | Pass Signal | Fail Signal |
| --- | --- | --- | --- |
| A | Synthetic CE-weight test | Time-shift errors get larger CE | Too blunt; does not distinguish wrong timing distance or grid collapse |
| B | Synthetic expected-distance loss | Matching logits score lower than rigid wrong-shift logits; zero lambda preserves behavior | No finite signal, no gradient, or no separation |
| C | Model-forward smoke | Could add timing capacity | Too broad before proving loss-side signal |
| D | New grammar/tokenizer | Could expose timing residue directly | Premature because current targets already contain rich timing residue |

## Selected Variant

- Selected: Variant B, time-shift distance auxiliary Stage 1.
- Rejected: Variant A because uniform class weighting does not directly target timing value error.
- Deferred: Variant C and D until a local loss-side signal is killed.
- Why this is the smallest useful test: it changes no tokenizer, no inference path, no decode policy, and no model architecture; it only adds disabled-by-default loss plumbing and a synthetic gate.

## Selection Pressure

- Primary pressure: separate matching target time-shift logits from rigid wrong-shift logits.
- Guard pressure: default lambda zero, no future target inference dependency, no grammar/tokenizer change.
- Runtime pressure: Stage 1 only; no training or rollout.
- Kill pressure: if the loss cannot produce a clean finite local signal, stop before training.

## Research Question

Can a local time-shift value calibration loss provide a measurable training signal that current CE does not make explicit enough for rigid-grid failure, without changing the v3 target grammar?

## Closest Analogies / Novelty Layer

- Closest analogies: regression-augmented classification, ordinal token-distance losses, expected-value calibration over discrete bins.
- Relevant taxonomy bucket: training objective calibration after representation audit.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is an objective-side engineering variation; v3 representation is unchanged.

## Minimal Change

Add disabled-by-default shared mapper tuple loss plumbing:

- config fields: `lambda_time_shift_distance`, `time_shift_distance_scale_ms`;
- loss function: at gold time-shift target positions, compute the expected time-shift value from the model distribution over existing time-shift tokens, then apply Smooth L1 against the gold shift value after scaling by `time_shift_distance_scale_ms`;
- metrics: `loss/time_shift_distance`, `phase/lambda_time_shift_distance`, and denominators for aggregation;
- defaults preserve current behavior exactly.

Add an artifact-only Stage 1 gate that runs synthetic probes:

- matching target timing logits;
- rigid wrong-shift logits;
- non-time-shift target rows ignored;
- zero lambda disabled-default behavior;
- config exposure through `MapperV3LossConfig`.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_experiment_card.md`
- `src/pulsefield_model/models/mapper/shared/loss.py`
- `src/pulsefield_model/models/mapper/v2_1/loss.py`
- `src/pulsefield_model/evals/mapper_v3_time_shift_distance_loss_stage1_gate.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/training/test_mapper_v3.py`
- `tests/evals/test_mapper_v3_time_shift_distance_loss_stage1_gate.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_result_report.md`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_shared_timing_residue_structure_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_decode_policy_continuation_sweep_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_continuation_jump_training_gate_summary.json`
- current v3 tokenizer/vocab/conversion files.

## Dataset Slice

Stage 1 uses synthetic logits only:

- one row with target sequence `TS_80, TS_100, TS_60, event`;
- matching logits put mass on each target shift;
- rigid wrong-shift logits put mass on `TS_100` at each time-shift step;
- event target row verifies non-time-shift rows are ignored.

No dataset generation, no training, and no rollout in Stage 1.

## Baseline / Comparator

- Current CE-only loss path at `lambda_time_shift_distance=0.0`.
- Timing-residue audit: target interval effective vocab `22.690931`, target `160/200ms` interval share `9.65%`, v2.1/v3 time-shift parity mismatches `0`.
- Decode-policy continuation sweep: route `KILL`.
- Continuation-jump training gate: route `MUTATE`.

## Primary Metric

Route decision:

- `TEST_TIME_SHIFT_DISTANCE_TINY_TRAINING_GATE` if synthetic checks pass and disabled-default behavior is exact.
- `KILL_TIME_SHIFT_DISTANCE_PLUMBING` if finite loss, config, gradient, or separation checks fail.

## Secondary Metric

- Matching loss value.
- Rigid wrong-shift loss value.
- Rigid-minus-matching gap.
- Gradient finite/nonzero check.
- Disabled-default total loss parity.
- Enabled metric positivity.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest \
  tests/models/mapper/v3/test_model.py \
  tests/training/test_mapper_v3.py \
  tests/evals/test_mapper_v3_time_shift_distance_loss_stage1_gate.py -q

uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_loss_stage1_gate \
  --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_summary.json \
  --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_result_report.md
```

## Guard Check

```bash
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_loss_stage1_summary.json >/dev/null
git diff --check
```

The gate must also assert:

- no tokenizer, grammar, replay, rollout, or inference files are changed;
- existing configs remain behavior-preserving at zero lambda;
- the loss uses only current-step teacher-forced target tokens for training and adds no inference-time dependency.

## Qualitative Check

The report must explain that Stage 1 proves only a local loss surface, not trained rollout improvement. A positive Stage 1 can justify a tiny training gate, not replacement.

## Positive Signal

- Matching time-shift logits have lower distance loss than rigid wrong-shift logits.
- Loss is finite and has nonzero finite gradients.
- Non-time-shift target positions are ignored by the auxiliary term.
- `lambda_time_shift_distance=0.0` preserves current total loss behavior.
- Config fields are accepted by `MapperV3LossConfig` and training config loading.

## Negative Signal

- Matching and rigid wrong-shift logits are not separated.
- Loss is unstable or gradientless.
- Default behavior changes.
- The implementation requires future target events or generated rollout state.

## Kill Criteria

- Any disabled-default behavior change.
- Any nonfinite loss or gradient.
- Rigid wrong-shift loss is not greater than matching loss.
- Config exposure fails.

## Expected Failure Modes

- Expected-value loss can be weak for multimodal time-shift distributions.
- It may duplicate CE without improving generated-state behavior.
- It may improve teacher-forced token loss but not free-running timing.
- It may need a later bucketed diagnostic by shift value.

## Confounders

- Synthetic logits do not prove learned rollout quality.
- Time-shift tokens encode event intervals indirectly; `160ms` intervals are often `TS_100 + TS_60`.
- Existing time features already expose current position; the failure may still be exposure bias.
- Full replacement still requires trained v3 quality on broader slices.

## Expected Runtime / Runtime Budget

Expected runtime: under one minute for Stage 1 gate and focused tests.

No training and no real-audio rollout in this card.

## Result Interpretation Plan

- Positive result would suggest: create a tiny v3 training/rollout gate with `lambda_time_shift_distance` and rigid-grid/second-window guards.
- Negative result would suggest: kill this loss plumbing and mutate toward timing embeddings or grammar repair.
- Ambiguous result would require: one synthetic bucket diagnostic by shift value.
- Human owner decides: whether to spend a training run on this objective.
- Next-loop action if positive: `TEST_TIME_SHIFT_DISTANCE_TINY_TRAINING_GATE`.
- Next-loop action if negative: `MUTATE_TIMING_EMBEDDING_OR_GRAMMAR`.
- Next-loop action if ambiguous: `TEST_TIME_SHIFT_DISTANCE_BUCKET_DIAGNOSTIC`.

## Result Log Template

- Experiment:
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary route:
- Matching loss:
- Rigid wrong-shift loss:
- Rigid-minus-matching gap:
- Gradient check:
- Disabled-default check:
- Config exposure:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
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
- Remaining ambiguity: exact lambda for a later training gate is not selected in Stage 1.

## Next-Loop Action

- If positive: write a tiny v3 training/rollout gate card for `lambda_time_shift_distance`.
- If negative: mutate to timing embedding or grammar repair.
- If ambiguous: run one synthetic bucket diagnostic.

## Novelty Notes

- Closest analogies: ordinal classification losses, regression-augmented CE, expected-value calibration over discrete bins.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: objective-side engineering variation only.
