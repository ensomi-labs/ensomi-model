# Target Grammar v3 500-Step Failure-Cluster Diagnostic Result Report

## Scope

This pass executes `target_grammar_v3_500step_failure_cluster_diagnostic_experiment_card.md`. It uses only the existing 32-case fixed-slice wide-audit artifacts and source beatmaps for reference timestamp distribution. It does not rerun training, rollout, or source-code changes.

## Procedure

- Wide-audit summary: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- Manifest: `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/manifest.json`
- Loaded all rollout summaries referenced by the wide-audit runs.
- Parsed source beatmaps to estimate reference first/second-window timing distribution.
- Kept wide-audit reference counts, F1, event-count ratio, rigidity, boundary, and logit metrics authoritative.
- Assigned rule-based failure clusters and checked all 32 cases are represented once.

## Guard Results

- Loaded cases: `32`
- Unique case IDs: `32`
- Training rerun: `false`
- Rollout rerun: `false`
- Source code changed: `false`

## Result

Decision: `TEST_NEXT`.

Reason: existing artifacts assign every non-pass-like case to concrete failure clusters. The next mutation should target event distribution and timing continuity before wider replacement scaling.

## Cluster Counts

| Cluster | Cases | Representative | Interpretation |
| --- | ---: | --- | --- |
| `pass_like_or_minor_residual` | 13 | `06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd` | No major rule-based failure label assigned. |
| `second_window_starvation` | 11 | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | Generation emits too little after 8s; primary continuity/count problem. |
| `reference_mismatch_second_window_drop` | 11 | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | Reference continues in the second window but generation does not. |
| `starved_160_grid` | 10 | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | Starvation is coupled to a 160ms repeated grid. |
| `undergeneration` | 9 | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | Generated event count is far below reference. |
| `event_logit_time_shift_bias` | 8 | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | Event tokens are rarely top-1 and median event rank is weak. |
| `semi_rigid_low_f1_grid` | 7 | `23_usao_knight_rider_kuo_kyoka_expert` | Moderately rigid grid plus weak alignment. |
| `boundary_or_duplicate_artifact` | 5 | `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | Boundary concentration or same-time/nonincreasing emissions. |
| `low_f1_misalignment` | 5 | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | Generated count may be plausible but timing placement is poor. |
| `rigid_high_f1_same_audio_grid` | 4 | `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | High local F1 but same-audio rigid timing pattern; quality/diversity risk. |
| `rigid_low_f1_grid_collapse` | 3 | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | Highly regular grid with poor reference alignment. |
| `overgeneration` | 2 | `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | Generated event count is far above reference. |
| `easier_chart_overproduction` | 2 | `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | Overgeneration happens on lower/mid difficulty charts. |
| `zero_reference_boundary_duplicate` | 1 | `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | No reference events in prefix, but model emits duplicate/boundary events. |

## Cluster Overlaps

| Cluster pair | Cases |
| --- | ---: |
| `reference_mismatch_second_window_drop` + `second_window_starvation` | 11 |
| `second_window_starvation` + `starved_160_grid` | 10 |
| `reference_mismatch_second_window_drop` + `starved_160_grid` | 10 |
| `second_window_starvation` + `undergeneration` | 9 |
| `reference_mismatch_second_window_drop` + `undergeneration` | 9 |
| `event_logit_time_shift_bias` + `second_window_starvation` | 8 |
| `event_logit_time_shift_bias` + `reference_mismatch_second_window_drop` | 8 |
| `starved_160_grid` + `undergeneration` | 8 |
| `event_logit_time_shift_bias` + `starved_160_grid` | 8 |
| `second_window_starvation` + `semi_rigid_low_f1_grid` | 7 |
| `reference_mismatch_second_window_drop` + `semi_rigid_low_f1_grid` | 7 |
| `semi_rigid_low_f1_grid` + `undergeneration` | 6 |

## Difficulty-Band Summary

| Band | Cases | Non-pass-like | Mean F1 | Mean gen 2nd | Mean ref 2nd | Top clusters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `high_>=4` | 10 | 6 | 0.514 | 22.2% | 65.0% | `second_window_starvation`:5, `reference_mismatch_second_window_drop`:5, `starved_160_grid`:5, `undergeneration`:4 |
| `low_<3` | 11 | 5 | 0.699 | 59.1% | 53.3% | `pass_like_or_minor_residual`:6, `rigid_high_f1_same_audio_grid`:2, `boundary_or_duplicate_artifact`:1, `overgeneration`:1 |
| `mid_3to4` | 11 | 8 | 0.694 | 29.4% | 55.4% | `second_window_starvation`:5, `reference_mismatch_second_window_drop`:5, `starved_160_grid`:5, `event_logit_time_shift_bias`:5 |

## Worst Non-Pass-Like Cases

| Case | Diff | Gen/Ref | F1 | Gen 2nd | Ref 2nd | Rigid | Dom | Boundary | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | 4.75 | 5/0 | 0.000 | 20.0% | n/a | 0.750 | 0 | 20.0% | `zero_reference_boundary_duplicate`, `boundary_or_duplicate_artifact` |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | 4.69 | 51/49 | 0.240 | 3.9% | 75.5% | 0.760 | 160 | 3.9% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact`, `event_logit_time_shift_bias` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 5.12 | 40/89 | 0.248 | 0.0% | 73.0% | 0.974 | 160 | 0.0% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `rigid_low_f1_grid_collapse`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | 4.49 | 47/136 | 0.426 | 2.1% | 67.6% | 0.957 | 160 | 2.1% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `rigid_low_f1_grid_collapse`, `starved_160_grid`, `low_f1_misalignment` |
| `23_usao_knight_rider_kuo_kyoka_expert` | 4.26 | 49/104 | 0.471 | 6.1% | 63.5% | 0.917 | 160 | 6.1% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | 3.47 | 51/77 | 0.484 | 9.8% | 62.3% | 0.880 | 160 | 9.8% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact`, `event_logit_time_shift_bias` |
| `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm` | 2.00 | 39/44 | 0.506 | 48.7% | 47.7% | 0.421 | 400 | 5.1% | `boundary_or_duplicate_artifact` |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | 2.81 | 38/91 | 0.512 | 5.3% | 52.7% | 0.865 | 210 | 2.6% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid` |
| `19_nekodex_circles_famoss_hard` | 3.55 | 48/90 | 0.522 | 4.2% | 61.1% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 4.19 | 48/93 | 0.582 | 4.2% | 54.8% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | 3.09 | 123/57 | 0.589 | 40.7% | 50.9% | 0.582 | 100 | 0.8% | `overgeneration`, `easier_chart_overproduction` |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | 3.44 | 48/91 | 0.604 | 4.2% | 54.9% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |

## Starvation Cases

| Case | Diff | Gen/Ref | F1 | Gen 2nd | Ref 2nd | Rigid | Dom | Boundary | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 5.12 | 40/89 | 0.248 | 0.0% | 73.0% | 0.974 | 160 | 0.0% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `rigid_low_f1_grid_collapse`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | 4.49 | 47/136 | 0.426 | 2.1% | 67.6% | 0.957 | 160 | 2.1% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `rigid_low_f1_grid_collapse`, `starved_160_grid`, `low_f1_misalignment` |
| `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | 3.29 | 47/88 | 0.622 | 2.1% | 50.0% | 0.957 | 160 | 2.1% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `rigid_low_f1_grid_collapse`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | 4.69 | 51/49 | 0.240 | 3.9% | 75.5% | 0.760 | 160 | 3.9% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact`, `event_logit_time_shift_bias` |
| `19_nekodex_circles_famoss_hard` | 3.55 | 48/90 | 0.522 | 4.2% | 61.1% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 4.19 | 48/93 | 0.582 | 4.2% | 54.8% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | 3.44 | 48/91 | 0.604 | 4.2% | 54.9% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `14_oomori_seiko_justadice_tv_size_remu_hard` | 3.19 | 48/87 | 0.607 | 4.2% | 54.0% | 0.936 | 160 | 4.2% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `event_logit_time_shift_bias` |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | 2.81 | 38/91 | 0.512 | 5.3% | 52.7% | 0.865 | 210 | 2.6% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid` |
| `23_usao_knight_rider_kuo_kyoka_expert` | 4.26 | 49/104 | 0.471 | 6.1% | 63.5% | 0.917 | 160 | 6.1% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | 3.47 | 51/77 | 0.484 | 9.8% | 62.3% | 0.880 | 160 | 9.8% | `second_window_starvation`, `reference_mismatch_second_window_drop`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact`, `event_logit_time_shift_bias` |

## Rigid High-F1 Same-Audio Grid Cases

| Case | Diff | Gen/Ref | F1 | Gen 2nd | Ref 2nd | Rigid | Dom | Boundary | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | 3.84 | 102/86 | 0.894 | 52.9% | 46.5% | 0.980 | 150 | 2.0% | `rigid_high_f1_same_audio_grid` |
| `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | 2.25 | 102/77 | 0.860 | 52.9% | 45.5% | 0.980 | 150 | 2.0% | `rigid_high_f1_same_audio_grid` |
| `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | 2.79 | 102/77 | 0.860 | 52.9% | 46.8% | 0.980 | 150 | 2.0% | `rigid_high_f1_same_audio_grid` |
| `20_namirin_kanzen_shouriesper_girl_tailsdk_insane` | 3.79 | 102/79 | 0.851 | 52.9% | 46.8% | 0.980 | 150 | 2.0% | `rigid_high_f1_same_audio_grid` |

## Overgeneration Cases

| Case | Diff | Gen/Ref | F1 | Gen 2nd | Ref 2nd | Rigid | Dom | Boundary | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | 3.09 | 123/57 | 0.589 | 40.7% | 50.9% | 0.582 | 100 | 0.8% | `overgeneration`, `easier_chart_overproduction` |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | 2.31 | 101/50 | 0.662 | 53.5% | 54.0% | 0.530 | 150 | 2.0% | `overgeneration`, `easier_chart_overproduction` |

## What Passed

- All 32 existing fixed-slice cases loaded and remained legal in the source audit.
- Assignment coverage for non-pass-like cases: `1.000`.
- The diagnostic explains the wide-audit symptoms without new training or rollout runtime.
- Rigid high-F1 cases are separated from low-F1 grid collapse, avoiding one overly broad rigid-grid label.

## What Surfaced

- 11 cases show second-window starvation; 10 are coupled to a 160ms grid.
- 4 high-F1 rigid same-audio grid cases remain a quality/diversity risk, not a legality failure.
- 2 cases overgenerate, mostly on easier/mid charts.
- 1 zero-reference prefix case emitted duplicate/boundary events and should be handled separately from ordinary F1.
- 8 cases show low event-token top-1 confidence relative to time-shift tokens.
- The dominant failure family is event distribution over time: second-window starvation and 160ms grid collapse in mid/high difficulty cases.
- A separate high-F1 rigid-grid family exists on the Namirin same-audio cases; this is not the same failure as starvation.
- Count calibration is asymmetric: some hard cases undergenerate while easier/mid cases overgenerate.
- Boundary/duplicate artifacts are concentrated but important because one zero-reference prefix still emits repeated events.

## Interpretation

This is not a v3 replacement pass. It is a targeted `TEST_NEXT` diagnostic after the wide-audit `MUTATE` result. The next bounded experiment should not be a full-dataset rollout. It should first test an event-distribution calibration on the fixed-slice bad clusters, with explicit gates for second-window share, event-count ratio, boundary duplicates, and dominant-spacing rigidity.

## Recommended Next Card

`target_grammar_v3_event_distribution_calibration`:

- dataset slice: the same 32 fixed-slice cases plus the six-case horizon gate;
- primary target: reduce second-window starvation from `11` cases while preserving legality;
- secondary targets: reduce overgeneration on easier charts and suppress zero-reference boundary duplicates;
- guard: no regression to v3 grammar reversibility or runtime legality;
- kill condition: calibration improves one cluster only by worsening another cluster or by reintroducing max-token/dead-end failures.
