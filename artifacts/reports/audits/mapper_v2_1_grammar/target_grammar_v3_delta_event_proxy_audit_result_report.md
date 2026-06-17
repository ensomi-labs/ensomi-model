# Target Grammar v3 Delta-Event Proxy Audit Result Report

## Scope

This bounded representation audit converts current v3 target fragments into a factorized delta-event proxy: event rows carry `delta_ms` plus event signature, and each window carries an end-gap marker. It does not train, rerun rollout, change tokenizer behavior, or change defaults.

## Decision

Route: `TEST_DELTA_EVENT_PROXY_MODEL_HEAD`.

- Reason: delta-event proxy passed reconstruction, sequence, and factorized-bit gates on the bounded slice (row ratio=0.386820, bit ratio=0.747341)
- Recommended next step: Create a bounded factorized delta-event model/loss plumbing card before any training-scale run.

## Guard Results

| Guard | Value |
| --- | ---: |
| no_training | `True` |
| no_rollout | `True` |
| no_tokenizer_or_default_change | `True` |
| no_c3_backreference | `True` |
| no_future_lookup | `True` |
| event_reconstruction_zero | `True` |
| end_reconstruction_zero | `True` |
| sequence_shorter_than_v3 | `True` |
| factorized_bits_not_worse_than_v3 | `True` |
| flat_pair_vocab_tractable | `True` |

## Metrics

| Metric | Value |
| --- | ---: |
| audited windows | `256` |
| v3 token count | `31640` |
| proxy row count | `12239` |
| sequence ratio vs v3 | `0.386820` |
| v3 unigram total bits | `157924.403184` |
| proxy factorized total bits | `118023.418097` |
| proxy bit ratio vs v3 | `0.747341` |
| flat pair bit ratio vs v3 | `0.703330` |
| event reconstruction mismatches | `0` |
| end reconstruction mismatches | `0` |
| unique event deltas | `88` |
| unique event signatures | `227` |
| unique flat delta-event pairs | `2026` |
| v3 time-shift token count | `19657` |
| v3 event token count | `11983` |

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

- The audit stayed representation-only: no training, rollout, tokenizer/default change, C3 input, or future lookup.
- Reconstruction passed: `0` event mismatches and `0` terminal end mismatches.
- Sequence pressure passed: `12239` proxy rows versus `31640` v3 target tokens (`0.386820` ratio).
- Bit proxy passed: factorized proxy bits are `0.747341` of current v3 unigram bits.
- Flat pair cardinality stayed tractable on this slice: `2026` unique delta/event pairs under the `4096` guard.

## What Surfaced

The proxy is viable enough for the next bounded model-head card, but it is not trained rollout evidence or replacement readiness.

The audit also surfaced that current v3 spends much of its target surface on standalone time-shift tokens: `19657` time-shift tokens versus `11983` event tokens in the audited fragments. Binding each event to its preceding delta removes that generated-prefix decision surface in the representation, but this remains a unigram proxy and does not prove learnability.

The timing/event space is concentrated but not trivial: `88` event-delta values, `227` event signatures, and `33` terminal end-gap values. That supports a factorized head follow-up before considering a flat target grammar.

## What Is Not Proved

- No mapper training run was performed.
- No autoregressive rollout or generated-chart quality improvement was measured.
- The bit estimate is a target-stream unigram proxy, not a conditional model likelihood.
- This does not replace the C3 local-backreference result; it only tests a separate v3-family grammar mutation without C3 replay.

## Next Step

Create a bounded factorized delta-event model/loss plumbing card before any training-scale run.
