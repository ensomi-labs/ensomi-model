# CASF v2 Context-Adaptive Fallback Codec Result Log

## Summary

- Selected variant: `c3_lz_active_span_backref`
- Recommendation: TEST_NEXT: c3_lz_active_span_backref passed charged CASF gates; refine the winning fallback entropy model before mapper-side work.
- Research pass: True
- Runtime seconds: 474.345997
- Limited: False
- Code dirty: True
- Reconstruction pass: True
- Reconstruction mismatches: 0
- Invalid active-hold transitions: 1789
- Segment-end active holds: 1788
- Baseline charged test bits/event: 4.421714207191301
- Selected charged test bits/event: 4.198613101989439
- Selected test delta bits/event: -0.2231011052018621
- Same-song filtered delta: -0.22353525270533403
- Bootstrap delta mean: -0.22282466691710426
- Bootstrap delta 95% interval: [-0.2580919949410291, -0.18675671602064908]
- Free-oracle test gain bits/event: 0.6686365689768934

## Variant Comparison

| variant | legal | valid charged | test charged | test delta | fallback payload delta | table/model bits |
|---|---:|---:|---:|---:|---:|---:|
| b0_r0_delta | True | 4.536986 | 4.421714 | NA | NA | 0.000000 |
| c1a_frozen_bitplane_all_fallback | True | 5.445753 | 5.439390 | 1.017676 | -1.731425 | 2683934.008913 |
| c1b_online_bitplane_all_fallback | True | 5.441536 | 5.436959 | 1.015245 | -1.738835 | 2683934.008913 |
| c2a_ppm_symbol_all_fallback | True | 227.015946 | 232.096720 | 227.675006 | 9.111698 | 380332350.245508 |
| c2b_ctw_mixture_all_fallback | True | 224.253606 | 229.719261 | 225.297546 | 1.863180 | 380332350.245508 |
| c3_lz_active_span_backref | True | 4.257266 | 4.198613 | -0.223101 | -0.680202 | 0.000000 |
| c4_signaled_active_patch_c1 | True | 5.754535 | 5.731702 | 1.309988 | -0.840208 | 2683934.008913 |
| s1_explicit_per_fallback_selector | True | 224.715346 | 230.364737 | 225.943023 | -1.003025 | 383016284.254421 |
| f0_free_oracle_upper_bound | False | 3.767522 | 3.753078 | -0.668637 | -2.038573 | 0.000000 |

## Gates

- selected_variant: c3_lz_active_span_backref
- baseline_consistency_pass: True
- expected_r0_test_charged_bits_per_event: 4.421714207191301
- observed_r0_test_charged_bits_per_event: 4.421714207191301
- valid_selected_bits_per_event_with_dictionary: 4.257266005220513
- test_selected_bits_per_event_with_dictionary: 4.198613101989439
- test_baseline_bits_per_event_with_dictionary: 4.421714207191301
- test_delta_bits_per_event_with_dictionary: -0.2231011052018621
- global_promotion_pass: True
- global_continuation_pass: True
- active_hold_fallback_gain_bits_per_event: 1.1959314528697924
- same_song_filtered_delta_bits_per_event: -0.22353525270533403
- same_song_filtered_pass: True
- bootstrap_mostly_below_zero_pass: True
- reconstruction_pass: True
- research_pass: True

## Interpretation

TEST_NEXT: c3_lz_active_span_backref passed charged CASF gates; refine the winning fallback entropy model before mapper-side work.

