# Mapper v2.1/v3 Fixed-Slice Decode Comparison Result Report

## Scope

This pass reruns v2.1 baseline and v2.1 anti-rigid guard on the 32-map, 16s case universe from the v3 500-step fixed-slice wide audit. The v3 wide audit is used as a read-only comparator.

## Decision

Decision: `KILL`.

- Reason: one or more v2.1 baseline/guard rollouts were illegal
- Cases: `32`
- v2.1 guard blocks: `1874`
- Guard rigid-improved cases: `30`
- Guard mean F1 delta vs v2.1 baseline: `-0.036039`
- Guard mean rigid delta vs v2.1 baseline: `-0.235288`
- Guard new starved cases vs v2.1 baseline: `2`
- Guard starved count vs v3: `3` vs `11`
- Guard mean F1 delta vs v3: `0.068474`

## Aggregate Table

| Variant | Legal | Mean F1 | Median F1 | Mean rigid | Rigid cases | Starved | Mean event ratio | Median event ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.1 baseline` | `False` | 0.744079 | 0.791425 | 0.956683 | 28 | 2 | 1.117148 | 1.117353 |
| `v2.1 guard` | `False` | 0.708040 | 0.777901 | 0.721395 | 0 | 3 | 1.073172 | 1.035610 |
| `v3 500 wide` | `True` | 0.639566 | 0.678952 | 0.749100 | 7 | 11 | 0.941075 | 0.955882 |

## What Passed

- The evaluator completed the full 32-map fixed-slice comparison: 64 v2.1 real-audio rollouts plus the read-only v3 comparator.
- The guard activated `1874` times and reduced dominant-spacing ratio in `30` of `32` cases.
- Rigid-case count dropped from `28` in v2.1 baseline to `0` under the guard.
- Guard starvation remained lower than v3: `3` vs `11` cases.
- Guard mean F1 remained above the v3 comparator by `0.068474`.

## What Surfaced

- Legality is not robust: `2` v2.1 rollout(s) dead-ended or failed completion.
- Guard mean F1 regressed by `-0.036039` versus v2.1 baseline despite reducing rigidity.
- Guard introduced `2` new starved case(s) versus v2.1 baseline.
- The hard-block guard can fix one dead-end pattern while creating another, so it should not be scaled as-is.

## Illegal Cases

| Case | Variant | Diff | Completed | Dead end | Max tokens | Terminal ms | Tokens | Timepoints |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `v2.1 baseline` | 2.300000 | False | True | False | 7990 | 64 | 12 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `v2.1 guard` | 2.310000 | False | True | False | 7990 | 98 | 32 |

## Case Table

| Case | Diff | Baseline F1/Rigid | Guard F1/Rigid | v3 F1/Rigid | Starved B/G/v3 | Blocks |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm` | 2.000000 | 0.699187/1.000000 | 0.582781/0.518868 | 0.506024/0.421053 | False/False/False | 40 |
| `01_hatsuki_yura_guren_yasha_a_m_d_normal` | 2.020000 | 0.788732/1.000000 | 0.760000/0.813953 | 0.695652/0.716216 | False/False/False | 48 |
| `02_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | 2.090000 | 0.886228/1.000000 | 0.857143/0.813953 | 0.765432/0.712329 | False/False/False | 64 |
| `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | 2.250000 | 0.820513/1.000000 | 0.829268/0.813953 | 0.860335/0.980198 | False/False/False | 64 |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | 2.300000 | 0.202532/0.909091 | 0.677686/0.584906 | 0.709220/0.712329 | True/False/False | 19 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | 2.310000 | 0.645161/0.671233 | 0.439024/0.548387 | 0.662252/0.530000 | False/True/False | 1 |
| `06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd` | 2.450000 | 0.680851/0.529412 | 0.723164/0.555556 | 0.657343/0.702703 | False/False/False | 68 |
| `07_nekodex_circles_famoss_normal` | 2.600000 | 0.750000/0.504132 | 0.728205/0.555556 | 0.737500/0.712329 | False/False/False | 66 |
| `08_moso_calibration_sakurairo_diary_tv_size_drum_hitnormal_hard` | 2.660000 | 0.906832/1.000000 | 0.887574/0.813953 | 0.722581/0.722222 | False/False/False | 64 |
| `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | 2.790000 | 0.870056/1.000000 | 0.817610/0.419753 | 0.860335/0.980198 | False/False/False | 60 |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | 2.810000 | 0.776471/1.000000 | 0.831461/0.813953 | 0.511628/0.864865 | False/False/True | 64 |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | 3.060000 | 0.851485/1.000000 | 0.820513/0.434783 | 0.825112/0.541667 | False/False/False | 68 |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | 3.090000 | 0.692308/1.000000 | 0.685714/0.804878 | 0.588889/0.581967 | False/False/False | 50 |
| `13_usao_knight_rider_kuo_kyoka_hard` | 3.150000 | 0.854054/1.000000 | 0.804598/0.413793 | 0.875676/0.530612 | False/False/False | 55 |
| `14_oomori_seiko_justadice_tv_size_remu_hard` | 3.190000 | 0.924731/1.000000 | 0.870588/0.804878 | 0.607407/0.936170 | False/False/True | 50 |
| `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | 3.290000 | 0.651515/1.000000 | 0.627737/0.812500 | 0.622222/0.956522 | True/True/True | 18 |
| `16_hatsuki_yura_guren_yasha_a_m_d_hard` | 3.290000 | 0.763006/1.000000 | 0.721519/0.809524 | 0.763006/0.525253 | False/False/False | 64 |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | 3.440000 | 0.900524/1.000000 | 0.806818/0.809524 | 0.604317/0.936170 | False/False/True | 64 |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | 3.470000 | 0.840909/1.000000 | 0.751592/0.810127 | 0.484375/0.880000 | False/False/True | 49 |
| `19_nekodex_circles_famoss_hard` | 3.550000 | 0.842105/1.000000 | 0.781609/0.807229 | 0.521739/0.936170 | False/False/True | 64 |
| `20_namirin_kanzen_shouriesper_girl_tailsdk_insane` | 3.790000 | 0.860335/1.000000 | 0.806283/0.792793 | 0.850829/0.980198 | False/False/False | 88 |
| `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | 3.840000 | 0.881720/1.000000 | 0.848485/0.792793 | 0.893617/0.980198 | False/False/False | 88 |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 4.190000 | 0.911917/1.000000 | 0.846561/0.800000 | 0.581560/0.936170 | False/False/True | 76 |
| `23_usao_knight_rider_kuo_kyoka_expert` | 4.260000 | 0.794118/1.000000 | 0.814815/0.792793 | 0.470588/0.916667 | False/False/True | 88 |
| `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane` | 4.280000 | 0.750000/1.000000 | 0.712766/0.792793 | 0.733728/0.500000 | False/False/False | 88 |
| `25_nekodex_circles_famoss_insane` | 4.430000 | 0.801980/1.000000 | 0.803738/0.792793 | 0.820513/0.500000 | False/False/False | 88 |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | 4.490000 | 0.754237/1.000000 | 0.774194/0.792793 | 0.426230/0.956522 | False/False/True | 88 |
| `27_billiummoto_four_veiled_stars_aries_insane` | 4.610000 | 0.818653/1.000000 | 0.783069/0.800000 | 0.817204/0.489130 | False/False/False | 75 |
| `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous` | 4.650000 | 0.680328/1.000000 | 0.731518/0.767857 | 0.802721/0.348993 | False/False/False | 88 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | 4.690000 | 0.617450/1.000000 | 0.236364/0.733333 | 0.240000/0.760000 | False/True/True | 23 |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | 4.750000 | 0.000000/1.000000 | 0.000000/0.800000 | 0.000000/0.750000 | False/False/False | 21 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 5.120000 | 0.592593/1.000000 | 0.294872/0.666667 | 0.248062/0.974359 | False/False/True | 23 |

## Interpretation

The hard-block guard should not be scaled from the six-case result. Either legality, starvation, F1, or fixed-slice rigidity failed the predefined floor.

## Next Step

Do not scale the anti-rigid hard block; inspect the illegal cases first.
