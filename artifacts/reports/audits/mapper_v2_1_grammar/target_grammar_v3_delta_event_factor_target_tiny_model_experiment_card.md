# Target Grammar v3 Delta-Event Factor Target Tiny Model Experiment Card

## Hypothesis

A true factorized delta-event target surface is trainable as a teacher-forced generated target on real fixed-slice data: rows are `EVENT(delta_ms, event_signature)` plus one terminal `END(end_gap_ms)` row, decoder inputs are shifted previous rows with a BOS row, reconstruction is exact, losses are finite, all factor heads receive gradients, and the factorized loss decreases without using current v3 time-shift tokens as generated targets.

## Root Objective

Move from representation-only delta-event evidence toward a replacement-ready v3 target grammar that is reversible, lower-bit than v2.1/current v3, teacher-forcing friendly, online-local, and does not require complex C3 cross-window replay. Full4k evidence already rejected the flat delta/event token family and routed to a factorized target/model card; this card tests the smallest real model surface for that route.

## Goal Decomposition

- Subgoal 1: Convert current v3 fixed-slice target fragments into factorized delta-event rows.
- Subgoal 2: Verify factorized rows reconstruct event times/signatures and terminal target end.
- Subgoal 3: Build shifted teacher-forcing decoder inputs from previous factorized rows only.
- Subgoal 4: Train a tiny causal factorized decoder on real fixed-slice batches.
- Subgoal 5: Verify kind, delta, signature, and end-gap heads have labels, finite losses, non-zero gradients, and decreasing loss.
- Subgoal 6: Avoid tokenizer/default/dataset schema/training runner/inference changes.

## Candidate Variants

- Variant A: Research-only tiny causal decoder over factorized rows, trained on 32 real fixed-slice windows using cached control-teacher vectors as context.
- Variant B: Immediately modify `MapperV3Model`/dataset/training runner to make factorized rows a production target.
- Variant C: Continue with the existing v3 token stream and treat delta-event only as an auxiliary objective.
- Variant D: Implement flat delta/event row tokens despite full4k pair cardinality failure.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Exact row reconstruction, positive labels for all heads, finite losses, non-zero gradients, factor loss decreases on real data | Missing labels, non-finite loss, zero gradients, loss fails to decrease |
| B | Production path compiles and trains | Too much blast radius before factorized target trainability is established |
| C | Lower implementation cost | Does not move toward replacing current v3 generated target rows |
| D | Simpler single softmax | Rejected by full4k: `29453` flat pairs exceed `4096` tractability guard |

## Selected Variant

Variant A: research-only tiny causal factorized decoder over real fixed-slice factorized rows.

## Selection Pressure

Variant A is the smallest experiment that directly tests the next missing claim: factorized rows can be used as a generated teacher-forcing target, not just as auxiliary labels attached to current v3 tokens. It keeps production defaults untouched and can fail quickly before a broader mapper replacement.

## Minimal Change

- Add one evaluator that builds factorized target rows from the fixed 32-song / 256-window index, trains a tiny causal decoder for a bounded number of steps, writes JSON and Markdown artifacts, and returns a route.
- Add focused tests for row construction/reconstruction, shifted teacher forcing, and route decisions.
- Do not change production tokenizer, dataset schema, v3 model defaults, training runner, inference, or rollout.

## Files Likely To Change

- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_tiny_model.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_tiny_model.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_result_report.md`

## Read-Only Context

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json`
- `src/pulsefield_model/evals/target_grammar_v3_delta_event_proxy_audit.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`

## Dataset Slice

Fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

Default training gate uses a bounded prefix of `32` windows with batch size `4`. The summary must record consumed windows, row counts, max row length, and label counts.

## Baseline / Comparator

- Full4k bit proxy route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`.
- Full4k factorized row ratio: `0.403157`.
- Full4k factorized bit ratio versus current v3: `0.792753`.
- Flat pair route rejected: `29453` unique flat pairs, not tractable.
- Existing auxiliary real-data gate passed, but it did not train a true factorized generated target.

## Primary Metric

Factorized total loss ratio:

`final_total_loss / initial_total_loss`

Pass threshold: ratio < `0.75` with finite losses.

## Secondary Metrics

- kind loss ratio
- delta loss ratio
- signature loss ratio
- end-gap loss ratio
- event row count
- end row count
- max row length
- first-step gradient sums for shared row embeddings and all four output heads
- reconstruction mismatch count

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_tiny_model
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_factor_target_tiny_model.py tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_tiny_model.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json >/dev/null
git diff --check
```

## Guard Check

- no production tokenizer change
- no dataset schema change
- no mapper default change
- no training runner change
- no rollout or inference change
- no C3 backreference
- no future lookup
- full4k bit-proxy route remains `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`

## Qualitative Check

The report must state that this is a research-only tiny model gate for target-surface trainability. It must not claim replacement readiness, full runner integration, rollout quality, or final v3 completion.

## Positive Signal

Rows reconstruct exactly, every batch has event and end labels, all factor heads receive gradients, all losses are finite, and total factorized loss decreases below threshold on real fixed-slice data.

## Negative Signal

Any reconstruction mismatch, missing labels, non-finite loss, zero head gradients, or loss ratio above threshold.

## Kill Criteria

- Factorized row reconstruction fails.
- Any factor head has no labels on real fixed-slice batches.
- Any factor head has zero first-step gradient.
- Loss is non-finite.

## Expected Failure Modes

- End-gap head is sparse and may train noisily.
- A tiny model may not learn event signatures quickly enough without more steps.
- Delta/end-gap class ranges are large (`801` classes each), so learning rate or model size may need mutation.
- This model does not include full v3 replay/grammar state, so it only tests the target surface, not production inference.

## Expected Runtime / Runtime Budget

Expected runtime: under 5 minutes on CPU/MPS with cached control-teacher tensors.

Stop condition: finish configured optimizer steps or fail immediately on reconstruction, labels, non-finite loss, or missing cache/data.

## Confounders

- Tiny model trainability does not prove full mapper quality.
- Fixed-slice training does not prove full4k training stability.
- Causal decoder uses simple context and does not represent final production architecture.
- Passing this gate still leaves production dataset/model/training/inference integration and rollout validation.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD`.
- If only loss decrease fails: route to `MUTATE_DELTA_EVENT_FACTOR_TARGET_MODEL_SCALE_OR_LR`.
- If labels/reconstruction/gradients fail: route to `KILL_DELTA_EVENT_FACTOR_TARGET_MODEL_SURFACE`.

## Result Log Template

```text
route:
consumed_windows:
row_count:
event_row_count:
end_row_count:
max_row_len:
initial_total_loss:
final_total_loss:
total_loss_ratio:
kind_loss_ratio:
delta_loss_ratio:
signature_loss_ratio:
end_gap_loss_ratio:
first_step_gradients:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to create the production plumbing card for factorized v3 target rows, mutate tiny model scale/loss, or stop the delta-event factor target path.

## Closest Analogies And Novelty Layer

Closest analogies are factorized autoregressive sequence targets, duration-conditioned event generation, and multi-head teacher-forced decoders. There is no novelty claim here. The tested layer is engineering feasibility of the factorized target/model surface after full4k representation evidence.
