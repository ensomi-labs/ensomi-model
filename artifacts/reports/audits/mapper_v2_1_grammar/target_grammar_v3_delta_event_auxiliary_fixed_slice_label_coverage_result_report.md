# Target Grammar v3 Delta-Event Auxiliary Fixed-Slice Label Coverage Result Report

## Scope

This fixed-slice audit derives delta-event auxiliary labels from current teacher-forced v3 target fragments. It does not train, roll out, change tokenizer behavior, change dataset schema, or change mapper defaults.

## Decision

Route: `TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING`.

- Reason: fixed-slice label coverage passed for current delta-event auxiliary head ranges
- Recommended next step: Create a tiny real-data training gate for the default-off delta-event auxiliary objective.

## Metrics

| Metric | Value |
| --- | ---: |
| audited windows | `256` |
| v3 token count | `31640` |
| time-shift token count | `19657` |
| event token count | `11983` |
| event label count | `11983` |
| signature label count | `11983` |
| end-gap label count | `256` |
| label error count | `0` |
| out-of-range delta count | `0` |
| out-of-range end-gap count | `0` |
| negative delta count | `0` |
| negative end-gap count | `0` |
| max event delta ms | `5790` |
| max end-gap ms | `8000` |
| unique event deltas | `88` |
| unique end gaps | `33` |

## Checks

| Check | Value |
| --- | ---: |
| no_training | `True` |
| no_rollout | `True` |
| no_tokenizer_dataset_default_change | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| audited_windows_positive | `True` |
| label_errors_zero | `True` |
| negative_deltas_zero | `True` |
| negative_end_gaps_zero | `True` |
| delta_out_of_range_zero | `True` |
| end_gap_out_of_range_zero | `True` |
| event_labels_match_event_tokens | `True` |
| signature_labels_match_event_tokens | `True` |
| end_gap_labels_match_windows | `True` |
| event_labels_positive | `True` |
| end_gap_labels_positive | `True` |

## Top Event Deltas

| delta ms | count |
| ---: | ---: |
| `160` | `2108` |
| `80` | `1912` |
| `150` | `1733` |
| `170` | `812` |
| `200` | `804` |
| `70` | `535` |
| `90` | `441` |
| `180` | `382` |
| `190` | `350` |
| `100` | `306` |
| `110` | `274` |
| `230` | `254` |

## Top End Gaps

| end gap ms | count |
| ---: | ---: |
| `100` | `34` |
| `40` | `25` |
| `30` | `22` |
| `80` | `16` |
| `130` | `16` |
| `60` | `14` |
| `10` | `12` |
| `70` | `9` |
| `50` | `9` |
| `150` | `9` |
| `120` | `9` |
| `110` | `8` |

## What Passed

- Real fixed-slice v3 fragments produced event-delta, event-signature, and end-gap labels under the current 8s head ranges.
- Event/signature label counts matched the current v3 event-token count.
- End-gap label count matched the audited window count.
- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used.

## What Surfaced

Fixed-slice real v3 fragments fit the current delta-event auxiliary label surface. This supports a tiny real-data training gate, not rollout or replacement.

The max end-gap label hit the current head boundary exactly at `8000ms`. This is still a pass for the fixed slice, but the future full4k label coverage audit should explicitly stress the end-gap range before any default grammar change.

## What Is Not Proved

- No real training run was performed.
- No rollout quality or generated-chart legality was measured.
- This fixed 256-window slice is not full4k label coverage.
- This does not make v3 target replacement ready.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_training_gate.py tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=`TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING`, event labels=`11983`, end-gap labels=`256`
- `40 passed in 1.37s`
- py_compile passed
- JSON validation passed
- diff check passed

## Next Step

Create a tiny real-data training gate for the default-off delta-event auxiliary objective.
