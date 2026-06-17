# Target Grammar v3 Delta-Event Auxiliary Tiny Real-Data Training Experiment Card

## Hypothesis

The default-off v3 delta-event auxiliary objective can optimize on real fixed-slice v3 batches without destabilizing the token objective: enabled auxiliary heads receive non-zero gradients, event/signature/end-gap labels are present on every training batch, losses remain finite, auxiliary loss decreases, and token loss is not materially worse than its starting value.

## Root Objective

Move v3 toward a factorized target grammar that is reversible, lower-bit than v2.1, teacher-forcing friendly, online-local, and does not depend on C3-style cross-window replay. This card tests whether the auxiliary factorization is trainable on real fixed-slice target fragments after synthetic trainability and fixed-slice label coverage have passed.

## Goal Decomposition

- Subgoal 1: Use the fixed 32-song / 256-window v3 comparison index already used by the proxy and label-coverage audits.
- Subgoal 2: Train a small v3 model with `use_delta_event_auxiliary_target=true` and `lambda_delta_event_auxiliary>0` on real collated v3 batches.
- Subgoal 3: Verify auxiliary labels and gradients reach the delta, event-signature, end-gap heads, and shared token embedding.
- Subgoal 4: Verify total, token, and auxiliary losses are finite and improve enough for the next bounded gate.
- Subgoal 5: Keep tokenizer, dataset schema, mapper defaults, C3 replay, future lookup, and rollout unchanged.

## Candidate Variants

- Variant A: Direct DataLoader tiny optimizer over the fixed-slice real v3 dataset, using cached control-teacher tensors and a small non-global model.
- Variant B: Full `run_mapper_v3_phase_b_training` baseline-vs-enabled pair for 20-40 steps, then compare training reports.
- Variant C: Full4k delta-event label coverage before any real-data optimization.
- Variant D: Autoregressive rollout with enabled auxiliary heads immediately after the fixed-slice coverage pass.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Finite real-batch losses, positive labels, non-zero head/shared gradients, auxiliary loss decreases, token loss not worse than threshold | Missing labels, non-finite losses, zero gradients, auxiliary loss fails to decrease, token loss regression |
| B | Training-runner reports complete and enabled run has finite auxiliary metrics with bounded token regression | Runtime overhead dominates, report comparability fails, checkpoint plumbing masks the loss question |
| C | Full4k range/coverage passes and resolves the 8000ms end-gap boundary concern | Does not test trainability; delays the immediate fixed-slice training question |
| D | Legal rollout emits plausible events | Premature: free-running decode confounds representation, training, and policy before real-batch optimization is known |

## Selected Variant

Variant A: direct fixed-slice real-data tiny training gate.

## Selection Pressure

Variant A is the smallest experiment that tests the next missing claim from the previous route: real-batch optimization of the enabled auxiliary objective. It is cheaper and more diagnostic than full-runner training, and it avoids rollout-policy confounds. Variant C remains required before replacement, but it does not answer whether the auxiliary objective trains on real batches. Variant D is too broad for the current evidence.

## Minimal Change

- Add one evaluator that loads the fixed index, builds real v3 batches, trains a tiny enabled model for a bounded number of optimizer steps, writes JSON and Markdown artifacts, and returns a route.
- Add focused unit tests for route selection, batch label accounting, and invalid run config.
- Do not change tokenizer, mapper defaults, dataset schema, inference, rollout, or training-runner defaults.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_real_data_training.py`
- `tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_result_report.md`

## Read-Only Context

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json`
- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`
- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/loss.py`

## Dataset Slice

Fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

The default run may train on a bounded prefix of this index, but the summary must record the source window count, consumed windows, batch size, and step count.

## Baseline / Comparator

- Current enabled auxiliary objective passed synthetic overfit with auxiliary loss ratio `0.014565`.
- Fixed-slice label coverage passed with `11983` event labels, `11983` signature labels, `256` end-gap labels, and zero range/label errors.
- Comparator for this card is the initial real-batch enabled-objective loss on the same model and batches before optimizer updates.

## Primary Metric

Auxiliary loss ratio:

`final_delta_event_auxiliary_loss / initial_delta_event_auxiliary_loss`

Pass threshold: ratio <= `0.75` with finite losses.

## Secondary Metrics

- total loss ratio
- token loss ratio
- event label count
- signature label count
- end-gap label count
- first-step gradient sum for delta head, signature head, end-gap head, and token embedding
- number of consumed windows
- max sequence length in the real batches

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_tiny_real_data_training
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_tiny_real_data_training.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_tiny_real_data_training_summary.json >/dev/null
git diff --check
```

## Guard Check

- tokenizer changed: false
- dataset schema changed: false
- mapper default behavior changed: false
- rollout run: false
- C3 backreference used: false
- future lookup used: false
- fixed-slice label-coverage route remains `TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING`

## Qualitative Check

The report must state that this is a real-batch optimization gate only. It must not claim rollout quality, full4k label coverage, default replacement readiness, or mapper-pipeline completion.

## Positive Signal

Real fixed-slice batches produce valid auxiliary labels; all losses are finite; auxiliary loss decreases below the ratio threshold; total loss decreases; token loss does not materially regress; all auxiliary heads and the shared embedding receive non-zero gradients.

## Negative Signal

Missing labels, non-finite loss, zero gradients, auxiliary loss fails to decrease, total loss fails to decrease, or token loss materially regresses.

## Kill Criteria

- Any real batch lacks event labels or end-gap labels.
- Any auxiliary loss component is non-finite.
- Any auxiliary head has zero first-step gradient.
- Label derivation throws on the fixed slice under current head ranges.

## Expected Failure Modes

- The auxiliary heads optimize while token loss regresses, implying lambda or weighting needs mutation.
- Real-batch sequence padding masks end-gap labels incorrectly.
- Small model capacity or learning rate makes the auxiliary loss noisy within the runtime budget.
- The fixed slice under-represents the full4k end-gap boundary issue surfaced by label coverage.

## Expected Runtime / Runtime Budget

Expected runtime: under 5 minutes on local CPU/MPS with cached control-teacher tensors.

Stop condition: finish the configured optimizer steps or fail immediately on non-finite loss, missing labels, range error, or missing cache/data.

## Confounders

- A tiny model and short run measure local trainability, not final mapper quality.
- Fixed-slice real batches do not prove full4k coverage.
- Token and auxiliary losses are optimized jointly, so a pass does not prove the factorized representation should replace the current token stream.
- Cached control-teacher tensors avoid encoder cost but do not test fresh control feature generation.

## Result Interpretation Plan

- If all checks pass: route to `TEST_DELTA_EVENT_AUXILIARY_FULL_SLICE_OR_FULL4K_LABEL_COVERAGE`, with full4k end-gap range stress required before any default grammar change.
- If optimization is finite but token loss regresses: route to `MUTATE_DELTA_EVENT_AUXILIARY_WEIGHTING`.
- If labels, gradients, or finite-loss checks fail: route to `KILL_DELTA_EVENT_AUXILIARY_REAL_DATA_TRAINING`.

## Result Log Template

```text
route:
steps:
consumed_windows:
initial_total_loss:
final_total_loss:
total_loss_ratio:
initial_token_loss:
final_token_loss:
token_loss_ratio:
initial_auxiliary_loss:
final_auxiliary_loss:
auxiliary_loss_ratio:
event_label_count:
signature_label_count:
end_gap_label_count:
first_step_gradients:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide between full-slice/full4k label coverage stress, auxiliary weighting mutation, or killing the delta-event auxiliary training path before spending rollout runtime.

## Closest Analogies And Novelty Layer

Closest analogies are auxiliary multi-task heads for structured sequence targets, factorized event-time prediction, and teacher-forced local target decomposition. There is no novelty claim here. The layer being tested is an engineering variation of the v3 target/loss surface, not a new representation result by itself.
