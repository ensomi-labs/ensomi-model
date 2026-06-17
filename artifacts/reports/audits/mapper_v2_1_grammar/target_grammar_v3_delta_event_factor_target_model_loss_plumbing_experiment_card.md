# Target Grammar v3 Delta-Event Factor Target Model/Loss Plumbing Experiment Card

## Hypothesis

The default-off factorized delta-event row target can be consumed by the production v3 model/loss surface without changing current v3 token-stream defaults: an optional row-level causal factor branch can read shifted factor rows, emit kind/delta/signature/end-gap logits, compute finite supervised loss on real fixed-slice batches, and backpropagate through all factor heads while the existing default model/loss/training behavior remains unchanged.

## Root Objective

Move the audited v3 factorized target from data-contract readiness toward a full replacement-capable mapper pipeline. Full4k representation and bit-proxy gates proved the target is reversible and lower-bit than current v3/v2.1 comparators; the data-contract gate proved default-off production batch plumbing. This card tests the next necessary boundary: model output and loss plumbing.

## Goal Decomposition

- Subgoal 1: Add explicit optional model outputs for factorized row targets, separate from the older token-position delta-event auxiliary heads.
- Subgoal 2: Add loss terms that consume `batch["delta_event_factor_target"]` directly and report label counts/loss metrics.
- Subgoal 3: Keep all new behavior default-off and require explicit config/loss enablement.
- Subgoal 4: Wire training dataset construction so enabled factor-target model/loss configs request factor-target batch fields, while default configs do not.
- Subgoal 5: Verify on the fixed real-data slice that output shapes align with factor rows, losses are finite, and gradients reach all factor heads.

## Candidate Variants

- Variant A: Add an optional row-level factor branch inside `MapperV3Model`, conditioned on projected control memory and shifted factor-row inputs, with `MapperV3ModelLoss` consuming nested factor labels.
- Variant B: Reuse the existing token-position `delta_event_auxiliary` heads and reinterpret their labels as the final factorized target.
- Variant C: Replace the main token decoder immediately with factor rows.
- Variant D: Keep model/loss eval-only and jump directly to a separate training script.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Default-off outputs absent; enabled real batch emits `[B,R,*]` factor logits; factor loss finite; kind/delta/signature/end-gap gradients nonzero; training dataset requests factor fields only when enabled | Shape mismatch, missing labels, finite-loss failure, zero gradients, default contract regression |
| B | Less code | Conflates token-position labels with the row-level target and does not test the actual replacement surface |
| C | Direct replacement | Too much blast radius before proving model/loss contract |
| D | No production risk | Does not advance production pipeline readiness |

## Selected Variant

Variant A: optional row-level factor target branch and loss in the production v3 model/loss surface, enabled explicitly and verified on real fixed-slice batches.

## Selection Pressure

Variant A is the smallest change that makes the requested final state more true: it tests the actual factor-row target surface in production modules without replacing training or inference defaults. Variant B is rejected because it preserves the wrong target alignment. Variant C is rejected until model/loss plumbing passes. Variant D stalls after data plumbing and cannot validate gradients through production modules.

## Minimal Change

- Add factor-target config flags and row-level output tensors to `MapperV3Model`.
- Add factor-target loss config, metrics, validation, and helper loss in `MapperV3ModelLoss`.
- Request `include_delta_event_factor_target=True` in v3 training datasets only when factor-target model/loss config is enabled.
- Add a bounded evaluator that runs enabled model/loss forward/backward on fixed-slice real batches.
- Add focused unit tests for default-off behavior, enabled shapes, loss metrics, gradients, and training dataset flag selection.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/loss.py`
- `src/pulsefield_model/training/mapper_v3.py`
- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_model_loss_plumbing.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/training/test_mapper_v3.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_model_loss_plumbing.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_result_report.md`

## Read-Only Context

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`

## Dataset Slice

Use the existing fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

The evaluator audits the first `32` real windows for row-count parity and runs forward/backward on the first enabled collated batch.

## Baseline / Comparator

- Data-contract route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD`.
- Tiny factor-target model route: `TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD`.
- Data-contract fixed-slice counts: `1479` rows, `1447` event rows, `32` end rows, `0` reconstruction mismatches.
- Current default v3 model/loss has no row-level factor target outputs or losses.

## Primary Metric

Enabled model/loss plumbing passes on a real batch:

- factor output tensors align with `[batch, row, classes]`
- factor loss is finite and positive
- kind, delta, signature, and end-gap head gradients are nonzero
- audited fixed-slice row counts match the data-contract gate

## Secondary Metrics

- default-off factor outputs remain `None`
- default-off loss metrics report `lambda_delta_event_factor_target = 0`
- event/signature/end-gap label counts
- row mask valid count
- enabled training dataset kwargs include factor-target fields only when enabled

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_model_loss_plumbing
uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py tests/evals/test_mapper_v3_delta_event_factor_target_model_loss_plumbing.py -q
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/model.py src/pulsefield_model/models/mapper/v3/loss.py src/pulsefield_model/training/mapper_v3.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_model_loss_plumbing.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_model_loss_plumbing_summary.json >/dev/null
git diff --check
```

## Guard Check

- Default `MapperV3Config` keeps factor target disabled.
- Default `MapperV3LossConfig` keeps factor target loss disabled.
- Default v3 dataset/training config does not emit `delta_event_factor_target`.
- No inference or incremental decode behavior changes.
- No C3 backreference, future lookup, or target-derived input conditioning.
- Existing token-stream loss remains available and unchanged when factor target is disabled.

## Qualitative Check

The result report must state this is a model/loss plumbing gate only. It must not claim full training stability, rollout quality, inference readiness, or complete v3 replacement.

## Positive Signal

Real fixed-slice enabled batches produce finite factor losses, nonzero factor-head gradients, exact fixed-slice row-count parity, and no observable default-off regression.

## Negative Signal

Missing nested factor labels, output/label shape mismatch, non-finite loss, zero factor-head gradients, default-off metric/output regression, or fixed-slice row count mismatch.

## Kill Criteria

- Factor target loss cannot be computed on the production batch mapping.
- Any required factor head receives zero gradient under the enabled loss.
- Default-off model/loss/training data contract changes.

## Expected Failure Modes

- Row-level branch may accidentally align to token sequence length instead of factor row length.
- Training dataset enablement may fail to request `delta_event_factor_target` when loss is enabled.
- Loss metrics may count padded rows or ignore labels incorrectly.
- End-gap labels may be weighted or counted incorrectly.

## Expected Runtime / Runtime Budget

Expected runtime: under 2 minutes with cached fixed-slice control teacher tensors.

Stop condition: fail fast on default-off regression, missing fixed-slice cache, shape mismatch, non-finite loss, or zero factor-head gradient.

## Confounders

- This does not train a full production model.
- This does not implement autoregressive factor-row inference.
- This does not remove current `target_fragment_tokens`.
- This does not prove rollout quality or final replacement readiness.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_TRAINING_SMOKE_CARD`.
- If default-off regresses: route to `KILL_FACTOR_TARGET_MODEL_LOSS_DEFAULT_REGRESSION`.
- If factor loss/gradient plumbing fails: route to `MUTATE_FACTOR_TARGET_MODEL_LOSS_PLUMBING`.

## Result Log Template

```text
route:
audited_windows:
row_count:
event_row_count:
end_row_count:
default_off_outputs:
enabled_output_shapes:
factor_loss:
label_counts:
gradient_abs:
training_dataset_flag:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to run a short production training smoke with factor-target loss enabled or repair the factor model/loss plumbing first.

## Closest Analogies And Novelty Layer

Closest analogies are multi-head autoregressive decoders, auxiliary/factorized target heads, and staged representation migrations. This is engineering plumbing for an already-audited representation, not a novelty claim. The tested layer is production model/loss compatibility with the row-level factor target.
