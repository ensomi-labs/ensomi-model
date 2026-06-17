# Target Grammar v3 Delta-Event Full4K Bit Proxy Result Report

## Scope

This full4k representation audit converts current v3 target fragments into a factorized delta-event proxy and computes sequence/bit proxies. It does not train, roll out, change tokenizer behavior, change dataset schema, change mapper defaults, or use C3/future context.

## Decision

Route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`.

- Reason: full4k delta-event factorized proxy passed reconstruction, sequence, and bit gates; flat delta-event pairs are too numerous for the flat-token variant
- Recommended next step: Create a bounded factorized target/model card; keep flat delta-event rows as rejected baseline.

## Guard Results

| Guard | Value |
| --- | ---: |
| no_training | `True` |
| no_rollout | `True` |
| no_tokenizer_dataset_default_change | `True` |
| no_c3_backreference_or_future_lookup | `True` |
| full_scope | `True` |
| full_v3_representation_audit_positive | `True` |
| full4k_label_coverage_positive | `True` |
| audited_windows_match_full_v3_audit | `True` |
| audited_windows_match_label_coverage | `True` |
| event_rows_match_label_coverage | `True` |
| label_errors_zero | `True` |
| event_reconstruction_zero | `True` |
| end_reconstruction_zero | `True` |
| sequence_shorter_than_v3 | `True` |
| factorized_bits_not_worse_than_v3 | `True` |
| proxy_rows_positive | `True` |
| v3_tokens_positive | `True` |

## Metrics

| Metric | Value |
| --- | ---: |
| audited windows | `174515` |
| candidate windows | `180696` |
| unsupported skipped windows | `6181` |
| v3 token count | `24542731` |
| proxy row count | `9894577` |
| sequence ratio vs v3 | `0.403157` |
| v3 unigram total bits | `124283646.403187` |
| proxy factorized total bits | `98526239.448001` |
| proxy bit ratio vs v3 | `0.792753` |
| flat pair bit ratio vs v3 | `0.785586` |
| event reconstruction mismatches | `0` |
| end reconstruction mismatches | `0` |
| unique event deltas | `680` |
| unique end gaps | `475` |
| unique event signatures | `255` |
| unique flat delta-event pairs | `29453` |
| flat pair tractable | `False` |
| v3 time-shift token count | `14813309` |
| v3 event token count | `9720062` |
| max event delta ms | `7980` |
| max end-gap ms | `8000` |

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

- Full4k delta-event proxy reconstruction was exact.
- Proxy rows were fewer than current v3 tokens.
- Factorized proxy bits were not worse than current v3 under the full-scope self-unigram proxy.
- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used.

## What Surfaced

Full4k supports the factorized delta-event route, but `29453` flat delta/event pairs exceed the `4096` guard, so the flat-token variant should remain a rejected baseline.

## What Is Not Proved

- This does not prove trained mapper quality.
- This does not prove autoregressive rollout quality.
- This does not by itself make delta-event factorization the generated target representation.
- This does not replace the full v2.1/v3 training and inference comparison.

## Verification

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_full4k_bit_proxy --progress-interval 5000
uv run --group dev pytest tests/evals/test_mapper_v3_delta_event_full4k_bit_proxy.py tests/evals/test_mapper_v3_delta_event_auxiliary_full4k_label_coverage.py tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/evals/mapper_v3_delta_event_full4k_bit_proxy.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json >/dev/null
git diff --check
```

Results:
- passed; route=`TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`, row ratio=`0.403157`, bit ratio=`0.792753`
- 35 passed in 0.87s
- passed
- passed
- passed

## Next Step

Create a bounded factorized target/model card; keep flat delta-event rows as rejected baseline.
