# C3/v3 Target Complexity Comparison Result Report

## Scope

This artifact-only pass compares the current C3 side-stream/ordered target evidence against v3 and the v2.1 sparse target baseline. It does not retrain, retokenize the dataset, change rollout behavior, or change mapper defaults.

## Decision

- route: `MUTATE_TO_V3_GRAMMAR_REPAIR`
- reason: C3 remains reconstructive but side-stream/state costs are not target-grammar competitive with v3
- next step: Focus the next bounded experiment on v3/v2.1 grammar repair and trained rollout failure clusters.

## Target Complexity Snapshot

| Candidate | Key property | Value |
| --- | --- | ---: |
| v2.1 | eval tokens | `28073` |
| v2.1 | vocab size | `37` |
| v3 | full token reduction | `20.544%` |
| v3 | eval total-bit reduction | `5.431%` |
| v3 | reconstruction mismatches | `0` |
| v3 | vocab size | `280` |
| C3 | codec delta bits/event | `-0.384910` |
| C3 | combined main+side/baseline ratio | `1.566805` |
| C3 | ordered/exact sequence ratio | `3.185742` |
| C3 | ordered bits/original token | `19.659257` |
| C3 | exact C3 bits/token | `10.877115` |
| C3 | reference cross-window span rate | `10.351%` |
| C3 | production input legal | `false` |

## Guard Results

- v3_reconstruction_zero: `true`
- v3_eval_total_bit_reduction_positive: `true`
- v3_full_token_reduction_positive: `true`
- v3_no_future_lookup: `true`
- c3_codec_reconstruction_zero: `true`
- c3_exact_sidecar_generation_pass: `true`
- c3_combined_sequence_competitive: `false`
- c3_ordered_sequence_bounded: `false`
- c3_ordered_bits_not_worse: `false`
- c3_cross_window_reference_low: `false`
- c3_production_input_legal: `false`

## What Passed

- C3 still has strong codec-side evidence: exact reconstruction and negative charged bits/event delta.
- The exact C3 mapper-window sidecar remains loader-compatible and cap-tractable.
- v3 remains exactly reconstructive on the full dataset and reduces target tokens/bits versus v2.1.

## What Surfaced

- C3 is not target-sequence competitive as an emitted mapper grammar: combined main+side tokens exceed the baseline stream.
- Ordered RAW splitting is lossless but expands the sequence and worsens charged bits versus exact C3 tokens.
- C3 reference semantics remain cross-window/stateful, and target-derived C3 labels are not legal production inputs.
- v3 is the simpler local teacher-forcing target, but this report does not prove v3 rollout replacement readiness.

## Interpretation

The C3 result should be kept as strong tokenizer-side/codec evidence, but the next mapper-facing work should focus on v3/v2.1 grammar repair rather than another C3 target or input-conditioning run.
