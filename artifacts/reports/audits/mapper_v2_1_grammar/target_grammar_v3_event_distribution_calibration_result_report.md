# Target Grammar v3 Event-Distribution Calibration Result Report

## Scope

This pass executes `target_grammar_v3_event_distribution_calibration_experiment_card.md`. It trains a fresh 500-step v3 checkpoint on the same fixed slice with `lambda_density=0.20`, then runs the same 32-case 16s fixed-slice real-audio rollout gate used by the v3 500-step wide audit.

No tokenizer, grammar, model, runtime, or decode-policy source code was changed.

## Result

Decision: `MUTATE`.

Selected variant result: `KILL_SIMPLE_DENSITY_WEIGHT_0_20`.

Reason: higher density weight regressed legality and event distribution on the 32-case fixed-slice gate.

## Training

- Checkpoint: `artifacts/tmp/mapper_v3_event_distribution_calibration/train/run/checkpoint.pt`
- Completed steps: `500`
- Density weight: `0.2`
- Final eval total loss: `2.234775`
- Final eval token loss: `1.906426`
- Final eval density loss: `1.614330`
- Baseline final eval token loss: `1.940945`
- Baseline final eval density loss: `1.608767`

| Step | Eval total | Eval token | Eval density |
| ---: | ---: | ---: | ---: |
| 1 | 4.448114 | 4.131833 | 1.520292 |
| 100 | 2.465590 | 2.138917 | 1.605133 |
| 200 | 2.316813 | 1.991973 | 1.598042 |
| 300 | 2.393525 | 2.065960 | 1.613335 |
| 400 | 2.243557 | 1.915791 | 1.613441 |
| 500 | 2.234775 | 1.906426 | 1.614330 |

## Aggregate Comparison

| Metric | density=0.20 | density=0.05 baseline |
| --- | ---: | ---: |
| All legal | `False` | `True` |
| Mean F1 @100ms | `0.582` | `0.640` |
| Median F1 @100ms | `0.643` | `0.679` |
| Mean event-count ratio | `1.412` | `0.971` |
| Median event-count ratio | `1.088` | `1.000` |
| Max event-count ratio | `8.571` | `2.158` |
| Mean dominant-spacing ratio | `0.830` | `0.749` |
| Median second-window share | `0.045` | `0.491` |
| Starved cases | `17` | `11` |
| Starved-160-grid cases | `11` | `10` |
| Undergeneration cases | `10` | `9` |
| Overgeneration cases | `5` | `2` |
| Rigid cases | `9` | `7` |
| Max boundary ratio | `0.048` | `0.200` |
| Max duplicate/nonincreasing ratio | `0.893` | `0.750` |

## Cluster Comparison

| Cluster | density=0.20 | density=0.05 baseline |
| --- | ---: | ---: |
| `second_window_starvation` | 17 | 11 |
| `reference_mismatch_second_window_drop` | 16 | 11 |
| `starved_160_grid` | 11 | 10 |
| `undergeneration` | 10 | 9 |
| `overgeneration` | 5 | 2 |
| `easier_chart_overproduction` | 3 | 2 |
| `rigid_high_f1_same_audio_grid` | 5 | 4 |
| `rigid_low_f1_grid_collapse` | 4 | 3 |
| `semi_rigid_low_f1_grid` | 15 | 7 |
| `low_f1_misalignment` | 11 | 5 |
| `boundary_or_duplicate_artifact` | 4 | 5 |
| `zero_reference_boundary_duplicate` | 1 | 1 |
| `event_logit_time_shift_bias` | 11 | 8 |
| `pass_like_or_minor_residual` | 7 | 13 |

## Non-Legal Cases

| Case | Legal | F1 | Base F1 | Gen/Ref | Base Gen/Ref | 2nd share | Base 2nd | Rigid | Base rigid | Clusters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `False` | 0.159 | 0.825 | 413/102 | 121/102 | 0.0% | 44.6% | 0.881 | 0.542 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `easier_chart_overproduction`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `False` | 0.051 | 0.240 | 420/49 | 51/49 | 0.0% | 3.9% | 0.893 | 0.760 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | `False` | 0.000 | 0.000 | 420/0 | 5/0 | 0.0% | 20.0% | 0.893 | 0.750 | `zero_reference_boundary_duplicate`, `second_window_starvation`, `semi_rigid_low_f1_grid`, `boundary_or_duplicate_artifact` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `False` | 0.063 | 0.248 | 420/89 | 40/89 | 0.0% | 0.0% | 0.893 | 0.974 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |

## Worst Cases

| Case | Legal | F1 | Base F1 | Gen/Ref | Base Gen/Ref | 2nd share | Base 2nd | Rigid | Base rigid | Clusters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | `False` | 0.000 | 0.000 | 420/0 | 5/0 | 0.0% | 20.0% | 0.893 | 0.750 | `zero_reference_boundary_duplicate`, `second_window_starvation`, `semi_rigid_low_f1_grid`, `boundary_or_duplicate_artifact` |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `False` | 0.051 | 0.240 | 420/49 | 51/49 | 0.0% | 3.9% | 0.893 | 0.760 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `False` | 0.063 | 0.248 | 420/89 | 40/89 | 0.0% | 0.0% | 0.893 | 0.974 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `False` | 0.159 | 0.825 | 413/102 | 121/102 | 0.0% | 44.6% | 0.881 | 0.542 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `easier_chart_overproduction`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `01_hatsuki_yura_guren_yasha_a_m_d_normal` | `True` | 0.118 | 0.696 | 22/63 | 75/63 | 0.0% | 72.0% | 0.619 | 0.716 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `low_f1_misalignment` |
| `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane` | `True` | 0.323 | 0.734 | 48/76 | 93/76 | 4.2% | 50.5% | 0.936 | 0.500 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | `True` | 0.357 | 0.512 | 21/91 | 38/91 | 4.8% | 5.3% | 0.900 | 0.865 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment` |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | `True` | 0.435 | 0.426 | 48/136 | 47/136 | 4.2% | 2.1% | 0.936 | 0.957 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `27_billiummoto_four_veiled_stars_aries_insane` | `True` | 0.437 | 0.817 | 49/93 | 93/93 | 4.1% | 49.5% | 0.917 | 0.489 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `23_usao_knight_rider_kuo_kyoka_expert` | `True` | 0.484 | 0.471 | 49/104 | 49/104 | 4.1% | 6.1% | 0.917 | 0.917 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `True` | 0.484 | 0.484 | 51/77 | 51/77 | 3.9% | 9.8% | 0.880 | 0.880 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |
| `25_nekodex_circles_famoss_insane` | `True` | 0.493 | 0.821 | 48/102 | 93/102 | 4.2% | 50.5% | 0.936 | 0.500 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `undergeneration`, `semi_rigid_low_f1_grid`, `starved_160_grid`, `low_f1_misalignment`, `event_logit_time_shift_bias` |

## Overgeneration Cases

| Case | Legal | F1 | Base F1 | Gen/Ref | Base Gen/Ref | 2nd share | Base 2nd | Rigid | Base rigid | Clusters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `False` | 0.051 | 0.240 | 420/49 | 51/49 | 0.0% | 3.9% | 0.893 | 0.760 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `False` | 0.063 | 0.248 | 420/89 | 40/89 | 0.0% | 0.0% | 0.893 | 0.974 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `False` | 0.159 | 0.825 | 413/102 | 121/102 | 0.0% | 44.6% | 0.881 | 0.542 | `second_window_starvation`, `reference_mismatch_second_window_drop`, `overgeneration`, `easier_chart_overproduction`, `semi_rigid_low_f1_grid`, `low_f1_misalignment`, `boundary_or_duplicate_artifact` |
| `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm` | `True` | 0.604 | 0.506 | 95/44 | 39/44 | 51.6% | 48.7% | 0.979 | 0.421 | `overgeneration`, `easier_chart_overproduction`, `rigid_low_f1_grid_collapse` |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `True` | 0.695 | 0.662 | 91/50 | 101/50 | 53.8% | 53.5% | 0.933 | 0.530 | `overgeneration`, `easier_chart_overproduction`, `semi_rigid_low_f1_grid` |

## What Passed

- Training completed 500 steps and produced a valid checkpoint.
- The 32-case rollout driver completed and wrote all per-case summaries.
- Max boundary-event ratio improved versus the baseline wide audit, but this was not sufficient because legality and distribution regressed.

## What Surfaced

- The simple density-weight mutation failed the primary gate: starvation worsened from `11` to `17` cases.
- Four cases hit `max_tokens_exceeded` through same-time duplicate-event collapse: Hibiki/Muscle, Camellia Synergic Ascent, Goreshit, and Camellia Equatorial.
- Overgeneration worsened from `2` to `5` cases, with max event-count ratio rising from `2.158` to `8.571`.
- Median second-window share collapsed from `49.1%` to `4.5%` even though the training density weight was stronger.
- The density auxiliary path is not enough as a scalar weight knob; it can amplify degenerate duplicate-event behavior.

## Interpretation

This is a negative calibration result. It does not kill v3 as a target grammar, because v3 still has reversible reconstruction and a shorter teacher-forcing target. It does kill the simple `lambda_density=0.20` path as a route to replacement readiness.

The next v3 mutation should not be another scalar density-weight increase. The failure pattern points to a more explicit event-budget/window-continuity objective or planner-side event-budget target that can distinguish undergeneration from overgeneration and suppress same-time duplicate collapse.

## Verification

- Runtime guard: `13 passed in 1.21s`.
- Training guard: `18 passed in 0.63s`.
- v3 rollout/model guard: `9 passed in 0.82s`.
- Result summary JSON validation: passed.

## Next Step

Create a bounded `target_grammar_v3_event_budget_objective` card. It should test an explicit per-window event-count or first/second-window continuity target, with hard guards for max-token, duplicate same-time events, and zero-reference intros. If that also fails, pause v3 training calibration and return to v2.1 grammar improvement while preserving v3 grammar artifacts.
