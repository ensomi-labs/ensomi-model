# Target Grammar v3 Delta-Event Factor Target Data Contract Result Report

## Scope

This gate verifies default-off production data plumbing for factorized delta-event target rows. It does not change production model/loss, training runner, inference, rollout, or mapper defaults.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD`.

- Reason: default-off v3 contract is preserved and default-on factor target rows collate with exact fixed-slice counts
- Recommended next step: Create a bounded model/loss plumbing card for factorized v3 target rows.

## Metrics

| Metric | Value |
| --- | ---: |
| audited windows | `32` |
| row count | `1479` |
| event rows | `1447` |
| end rows | `32` |
| max row len | `97` |
| row mask valid count | `1479` |
| kind label count | `1479` |
| delta label count | `1447` |
| signature label count | `1447` |
| end-gap label count | `32` |
| reconstruction mismatches | `0` |

## Checks

| Check | Value |
| --- | ---: |
| default_off_factor_target_absent | `True` |
| tiny_model_route_positive | `True` |
| audited_windows_positive | `True` |
| row_count_positive | `True` |
| event_rows_positive | `True` |
| end_rows_match_windows | `True` |
| row_mask_count_matches_rows | `True` |
| kind_labels_match_rows | `True` |
| delta_labels_match_events | `True` |
| signature_labels_match_events | `True` |
| end_gap_labels_match_windows | `True` |
| reconstruction_mismatches_zero | `True` |
| row_count_matches_tiny_model | `True` |
| event_rows_match_tiny_model | `True` |
| end_rows_match_tiny_model | `True` |
| no_model_loss_runner_inference_change | `True` |
| no_c3_backreference_or_future_lookup | `True` |

## What Passed

- Default-off v3 batches do not expose factor-target fields.
- Default-on batches collate shifted factor-target inputs and labels.
- Fixed-slice row counts match the accepted tiny model gate.
- Reconstruction mismatch count remains zero.

## What Surfaced

The factorized target has crossed from eval-only evidence into a default-off production data contract.

## What Is Not Proved

- This does not train a production factorized model.
- This does not implement factorized inference or rollout.
- This does not make v3 replacement ready.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_data_contract
uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/evals/test_mapper_v3_delta_event_factor_target_data_contract.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/factor_target.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_data_contract.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_summary.json >/dev/null
git diff --check
```

## Next Step

Create a bounded model/loss plumbing card for factorized v3 target rows.
