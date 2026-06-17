# Target Grammar v3 Delta-Event Auxiliary Full4K Label Coverage Result Report

## Scope

This one-pass full4k audit derives delta-event auxiliary labels from current v3 target fragments. It does not train, roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.

## Decision

Route: `TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD`.

- Reason: full4k delta-event auxiliary label coverage passed under current head ranges
- Recommended next step: Create a full4k delta-event bit-proxy or factor-target card before changing mapper defaults.

## Metrics

| Metric | Value |
| --- | ---: |
| source windows | `707767` |
| candidate windows | `180696` |
| audited windows | `174515` |
| unsupported skipped windows | `6181` |
| label error count | `0` |
| v3 token count | `24542731` |
| time-shift token count | `14813309` |
| event token count | `9720062` |
| event label count | `9720062` |
| signature label count | `9720062` |
| end-gap label count | `174515` |
| out-of-range delta count | `0` |
| out-of-range end-gap count | `0` |
| negative delta count | `0` |
| negative end-gap count | `0` |
| max event delta ms | `7980` |
| max end-gap ms | `8000` |
| unique event deltas | `680` |
| unique end gaps | `475` |

## Checks

| Check | Value |
| --- | ---: |
| no_training | `True` |
| no_rollout | `True` |
| no_tokenizer_dataset_default_change | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| full_scope | `True` |
| full_v3_representation_audit_positive | `True` |
| fixed_slice_label_coverage_positive | `True` |
| tiny_real_data_training_positive | `True` |
| audited_windows_match_full_v3_audit | `True` |
| candidate_windows_positive | `True` |
| audited_windows_positive | `True` |
| label_errors_zero | `True` |
| negative_deltas_zero | `True` |
| negative_end_gaps_zero | `True` |
| delta_out_of_range_zero | `True` |
| end_gap_out_of_range_zero | `True` |
| event_labels_match_event_tokens | `True` |
| signature_labels_match_event_tokens | `True` |
| end_gap_labels_match_audited_windows | `True` |
| event_labels_positive | `True` |
| end_gap_labels_positive | `True` |

## Top Event Deltas

| delta ms | count |
| ---: | ---: |
| `80` | `1188664` |
| `90` | `883375` |
| `70` | `793644` |
| `160` | `630064` |
| `150` | `626043` |
| `170` | `622911` |
| `100` | `577951` |
| `60` | `448828` |
| `110` | `448741` |
| `120` | `360946` |
| `140` | `355143` |
| `180` | `334747` |

## Top End Gaps

| end gap ms | count |
| ---: | ---: |
| `20` | `15186` |
| `10` | `11804` |
| `60` | `11284` |
| `50` | `11073` |
| `30` | `11003` |
| `40` | `10731` |
| `70` | `10583` |
| `80` | `9537` |
| `0` | `9360` |
| `90` | `8083` |
| `100` | `6563` |
| `120` | `6349` |

## What Passed

- Full4k v3 fragments produced event-delta, event-signature, and end-gap labels under the current head ranges.
- Event/signature label counts matched event-token count.
- End-gap label count matched audited windows.
- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used.

## What Surfaced

Full4k coverage still reaches the current end-gap boundary at `8000ms` but does not exceed it. This supports a follow-up full4k bit-proxy/factor-target card, while keeping the boundary visible as a range-risk.

## What Is Not Proved

- This does not prove trained mapper quality.
- This does not prove autoregressive rollout quality.
- This does not by itself make delta-event factorization the generated target representation.
- This does not replace the full v2.1/v3 training and inference comparison.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_auxiliary_full4k_label_coverage --progress-interval 5000
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage.py tests/evals/test_mapper_v3_delta_event_auxiliary_tiny_real_data_training.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_auxiliary_full4k_label_coverage.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_auxiliary_full4k_label_coverage_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=`TEST_DELTA_EVENT_FULL4K_BIT_PROXY_OR_FACTOR_TARGET_CARD`, audited windows=`174515`, label errors=`0`, max end-gap=`8000ms`
- 33 passed in 0.87s
- passed
- passed
- passed

## Next Step

Create a full4k delta-event bit-proxy or factor-target card before changing mapper defaults.
