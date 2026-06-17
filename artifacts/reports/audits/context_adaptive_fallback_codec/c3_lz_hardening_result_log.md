# C3 LZ Fallback-Substream Hardening Result Log

## Summary

- Selected variant: `a2_skeleton_residual_all_fallback_w256`
- Recommendation: TEST_NEXT: a2_skeleton_residual_all_fallback_w256 passed C3 hardening gates; build an optional pipeline-facing fallback side-stream tokenization artifact next.
- Research promotion pass: True
- Engineering promotion pass: True
- Runtime seconds: 819.045022
- Limited: False
- Code dirty: True
- Hard reconstruction pass: True
- Hard reconstruction mismatches: 0
- Baseline charged test bits/event: 4.421714207191301
- Selected charged test bits/event: 4.0368045441872065
- Selected test delta bits/event: -0.3849096630040947
- Same-song filtered delta: -0.38592890137466185
- Clean-trace test delta: -0.3809793375551669
- Dirty-trace test delta: -0.43255187244188775
- Bootstrap delta 95% interval: [-0.4235772949889358, -0.34584630062559635]
- Selected spans: 417208
- Selected fallback literals: 1619653
- Selected noncontiguous main-stream spans: 199906

## Variant Comparison

| variant | valid charged | test charged | test delta | fallback payload delta | table/model bits |
|---|---:|---:|---:|---:|---:|
| c3_active_all_fixed_w256 | 4.233193 | 4.179146 | -0.242568 | -0.739555 | 0.000000 |
| a1_exact_only_all_fallback_w256 | 4.195274 | 4.143128 | -0.278586 | -0.849367 | 0.000000 |
| a2_skeleton_residual_all_fallback_w256 | 4.088714 | 4.036805 | -0.384910 | -1.173532 | 0.000000 |
| a3_mirror_only_all_fallback_w256 | 4.198796 | 4.131629 | -0.290085 | -0.884426 | 0.000000 |
| a5_non_active_all_fixed_w256 | 4.440926 | 4.329050 | -0.092664 | -0.282519 | 0.000000 |
| a6_active_all_fixed_w16 | 4.327292 | 4.250113 | -0.171601 | -0.523187 | 0.000000 |
| a6_active_all_fixed_w32 | 4.285379 | 4.219147 | -0.202567 | -0.617596 | 0.000000 |
| a6_active_all_fixed_w64 | 4.258669 | 4.195221 | -0.226493 | -0.690543 | 0.000000 |
| a6_active_all_fixed_w128 | 4.243071 | 4.183879 | -0.237835 | -0.725125 | 0.000000 |
| a6_active_all_fixed_w512 | 4.225509 | 4.177688 | -0.244026 | -0.743999 | 0.000000 |
| a6_active_all_fixed_w1024 | 4.226730 | 4.183298 | -0.238416 | -0.726894 | 0.000000 |
| a6_active_all_fixed_full_history | 4.191628 | 4.153449 | -0.268265 | -0.817901 | 0.000000 |
| a7_active_all_elias_gamma_w256 | 4.228882 | 4.173456 | -0.248259 | -0.756903 | 0.000000 |
| a7_active_all_elias_delta_w256 | 4.232622 | 4.176948 | -0.244767 | -0.746257 | 0.000000 |
| a7_active_all_power_bucket_w256 | 4.244156 | 4.186584 | -0.235131 | -0.716878 | 0.000000 |
| a7_active_all_train_fitted_w256 | 4.231263 | 4.178222 | -0.243492 | -0.757972 | 8661.363001 |
| a7_active_all_online_adaptive_w256 | 4.396233 | 4.323181 | -0.098533 | -0.300412 | 0.000000 |
| a8_active_all_fixed_same_offset_w256 | 4.286712 | 4.215014 | -0.206700 | -0.630199 | 0.000000 |
| a8_active_all_fixed_same_bar_phase_w256 | 4.271512 | 4.210032 | -0.211682 | -0.645388 | 0.000000 |

## Reconstruction Guards

- `c3_active_all_fixed_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=86451
- `a1_exact_only_all_fallback_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=82290
- `a2_skeleton_residual_all_fallback_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=199906
- `a3_mirror_only_all_fallback_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=82279
- `a5_non_active_all_fixed_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=62672
- `a6_active_all_fixed_w16` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=63744
- `a6_active_all_fixed_w32` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=74804
- `a6_active_all_fixed_w64` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=80400
- `a6_active_all_fixed_w128` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=83270
- `a6_active_all_fixed_w512` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=87891
- `a6_active_all_fixed_w1024` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=86724
- `a6_active_all_fixed_full_history` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=95609
- `a7_active_all_elias_gamma_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=86033
- `a7_active_all_elias_delta_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=86968
- `a7_active_all_power_bucket_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=82839
- `a7_active_all_train_fitted_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=96230
- `a7_active_all_online_adaptive_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=20532
- `a8_active_all_fixed_same_offset_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=75770
- `a8_active_all_fixed_same_bar_phase_w256` pass=True, fallback=0, stream=0, chart=0, boundary=0, transform=0, noncontiguous=73976

## Gates

- selected_variant: a2_skeleton_residual_all_fallback_w256
- baseline_consistency_pass: True
- expected_r0_test_charged_bits_per_event: 4.421714207191301
- observed_r0_test_charged_bits_per_event: 4.421714207191301
- valid_selected_bits_per_event_with_dictionary: 4.088714143051159
- test_selected_bits_per_event_with_dictionary: 4.0368045441872065
- test_baseline_bits_per_event_with_dictionary: 4.421714207191301
- test_delta_bits_per_event_with_dictionary: -0.3849096630040947
- same_song_filtered_delta_bits_per_event: -0.38592890137466185
- same_song_filtered_pass: True
- clean_trace_test_delta_bits_per_event: -0.3809793375551669
- dirty_trace_test_delta_bits_per_event: -0.43255187244188775
- clean_trace_test_pass: True
- bootstrap_95pct_below_zero_pass: True
- hard_reconstruction_pass: True
- exact_only_test_delta_bits_per_event: -0.2785859452434707
- simple_family_pass: True
- bounded_window_128_delta_bits_per_event: -0.23783549837040585
- bounded_window_256_delta_bits_per_event: -0.24256829527744017
- bounded_window_pass: True
- research_promotion_pass: True
- engineering_promotion_pass: True

## Interpretation

TEST_NEXT: a2_skeleton_residual_all_fallback_w256 passed C3 hardening gates; build an optional pipeline-facing fallback side-stream tokenization artifact next.
