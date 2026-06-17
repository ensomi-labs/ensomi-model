# Mapper v2.1 Terminal LN-Start Guard Repair Result Report

## Scope

This eval reruns v2.1 fixed-slice real-audio cases with the existing grammar `min_ln_duration_ms` primitive exposed through runtime rollout. The guard is opt-in and mapper defaults remain unchanged.

## Decision

Decision: `TEST_NEXT`.

- Reason: terminal LN-start guard passed the full32 legality and quality gate
- Full32: `True`
- Runs: `64`
- Cases: `32`
- Primary case 04 baseline legal: `True`
- Primary case 05 anti-rigid legal: `True`
- All candidate legal: `True`
- Max-token count: `0`
- New starved count: `0`
- Mean F1 delta vs matching comparator: `0.008080`
- Anti-rigid blocked count: `1906`

## Aggregate Table

| Mode | Legal | Mean F1 | Starved | Mean rigid | Mean event ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline + terminal guard` | `32` / `32` | 0.755842 | 1 | 0.955060 | 1.151196 |
| `anti-rigid + terminal guard` | `32` / `32` | 0.712436 | 2 | 0.720063 | 1.108172 |

## What Passed

- The eval completed `64` real-audio guarded rollouts.
- The terminal guard used `min_ln_duration_ms=20` and remained opt-in.
- Primary case `04` baseline legality: `True`.
- Primary case `05` anti-rigid legality: `True`.

## What Surfaced

The terminal guard passed the widened fixed-slice gate. It remains an opt-in grammar hardening result until compared against v3 and v2.1 default behavior in a replacement decision.

## Case Table

| Case | Mode | Previous legal | Candidate legal | Previous F1/Rigid/Starved | Candidate F1/Rigid/Starved | F1 delta | Terminal ms | Tokens | Blocks |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm` | `baseline_guard` | True | True | 0.699187/1.000000/False | 0.699187/1.000000/False | 0.000000 | 16000 | 200 | 0 |
| `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm` | `anti_rigid_guard` | True | True | 0.582781/0.518868/False | 0.582781/0.518868/False | 0.000000 | 16000 | 273 | 40 |
| `01_hatsuki_yura_guren_yasha_a_m_d_normal` | `baseline_guard` | True | True | 0.788732/1.000000/False | 0.788732/1.000000/False | 0.000000 | 16000 | 202 | 0 |
| `01_hatsuki_yura_guren_yasha_a_m_d_normal` | `anti_rigid_guard` | True | True | 0.760000/0.813953/False | 0.760000/0.813953/False | 0.000000 | 16000 | 331 | 48 |
| `02_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `baseline_guard` | True | True | 0.886228/1.000000/False | 0.886228/1.000000/False | 0.000000 | 16000 | 245 | 0 |
| `02_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `anti_rigid_guard` | True | True | 0.857143/0.813953/False | 0.857143/0.813953/False | 0.000000 | 16000 | 411 | 64 |
| `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | `baseline_guard` | True | True | 0.820513/1.000000/False | 0.820513/1.000000/False | 0.000000 | 16000 | 345 | 0 |
| `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | `anti_rigid_guard` | True | True | 0.829268/0.813953/False | 0.829268/0.813953/False | 0.000000 | 16000 | 421 | 64 |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `baseline_guard` | False | True | 0.202532/0.909091/True | 0.578947/0.857143/False | 0.376416 | 16000 | 339 | 0 |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `anti_rigid_guard` | True | True | 0.677686/0.584906/False | 0.677686/0.584906/False | 0.000000 | 16000 | 199 | 19 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `baseline_guard` | True | True | 0.645161/0.671233/False | 0.645161/0.671233/False | 0.000000 | 16000 | 275 | 0 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `anti_rigid_guard` | False | True | 0.439024/0.548387/True | 0.579710/0.505747/False | 0.140686 | 16000 | 338 | 33 |
| `06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd` | `baseline_guard` | True | True | 0.680851/0.529412/False | 0.680851/0.529412/False | 0.000000 | 16000 | 375 | 0 |
| `06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd` | `anti_rigid_guard` | True | True | 0.723164/0.555556/False | 0.723164/0.555556/False | 0.000000 | 16000 | 402 | 68 |
| `07_nekodex_circles_famoss_normal` | `baseline_guard` | True | True | 0.750000/0.504132/False | 0.750000/0.504132/False | 0.000000 | 16000 | 374 | 0 |
| `07_nekodex_circles_famoss_normal` | `anti_rigid_guard` | True | True | 0.728205/0.555556/False | 0.728205/0.555556/False | 0.000000 | 16000 | 408 | 66 |
| `08_moso_calibration_sakurairo_diary_tv_size_drum_hitnormal_hard` | `baseline_guard` | True | True | 0.906832/1.000000/False | 0.906832/1.000000/False | 0.000000 | 16000 | 393 | 0 |
| `08_moso_calibration_sakurairo_diary_tv_size_drum_hitnormal_hard` | `anti_rigid_guard` | True | True | 0.887574/0.813953/False | 0.887574/0.813953/False | 0.000000 | 16000 | 433 | 64 |
| `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | `baseline_guard` | True | True | 0.870056/1.000000/False | 0.870056/1.000000/False | 0.000000 | 16000 | 403 | 0 |
| `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | `anti_rigid_guard` | True | True | 0.817610/0.419753/False | 0.817610/0.419753/False | 0.000000 | 16000 | 360 | 60 |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | `baseline_guard` | True | True | 0.776471/1.000000/False | 0.776471/1.000000/False | 0.000000 | 16000 | 395 | 0 |
| `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | `anti_rigid_guard` | True | True | 0.831461/0.813953/False | 0.831461/0.813953/False | 0.000000 | 16000 | 435 | 64 |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `baseline_guard` | True | True | 0.851485/1.000000/False | 0.851485/1.000000/False | 0.000000 | 16000 | 601 | 0 |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | `anti_rigid_guard` | True | True | 0.820513/0.434783/False | 0.820513/0.434783/False | 0.000000 | 16000 | 548 | 68 |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | `baseline_guard` | True | True | 0.692308/1.000000/False | 0.692308/1.000000/False | 0.000000 | 16000 | 350 | 0 |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | `anti_rigid_guard` | True | True | 0.685714/0.804878/False | 0.685714/0.804878/False | 0.000000 | 16000 | 320 | 50 |
| `13_usao_knight_rider_kuo_kyoka_hard` | `baseline_guard` | True | True | 0.854054/1.000000/False | 0.854054/1.000000/False | 0.000000 | 16000 | 450 | 0 |
| `13_usao_knight_rider_kuo_kyoka_hard` | `anti_rigid_guard` | True | True | 0.804598/0.413793/False | 0.804598/0.413793/False | 0.000000 | 16000 | 418 | 55 |
| `14_oomori_seiko_justadice_tv_size_remu_hard` | `baseline_guard` | True | True | 0.924731/1.000000/False | 0.924731/1.000000/False | 0.000000 | 16000 | 350 | 0 |
| `14_oomori_seiko_justadice_tv_size_remu_hard` | `anti_rigid_guard` | True | True | 0.870588/0.804878/False | 0.870588/0.804878/False | 0.000000 | 16000 | 320 | 50 |
| `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | `baseline_guard` | True | True | 0.651515/1.000000/True | 0.651515/1.000000/True | 0.000000 | 16000 | 235 | 0 |
| `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | `anti_rigid_guard` | True | True | 0.627737/0.812500/True | 0.627737/0.812500/True | 0.000000 | 16000 | 250 | 18 |
| `16_hatsuki_yura_guren_yasha_a_m_d_hard` | `baseline_guard` | True | True | 0.763006/1.000000/False | 0.763006/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `16_hatsuki_yura_guren_yasha_a_m_d_hard` | `anti_rigid_guard` | True | True | 0.721519/0.809524/False | 0.721519/0.809524/False | 0.000000 | 16000 | 442 | 64 |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | `baseline_guard` | True | True | 0.900524/1.000000/False | 0.900524/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | `anti_rigid_guard` | True | True | 0.806818/0.809524/False | 0.806818/0.809524/False | 0.000000 | 16000 | 442 | 64 |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `baseline_guard` | True | True | 0.840909/1.000000/False | 0.840909/1.000000/False | 0.000000 | 16000 | 352 | 0 |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `anti_rigid_guard` | True | True | 0.751592/0.810127/False | 0.751592/0.810127/False | 0.000000 | 16000 | 310 | 49 |
| `19_nekodex_circles_famoss_hard` | `baseline_guard` | True | True | 0.842105/1.000000/False | 0.842105/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `19_nekodex_circles_famoss_hard` | `anti_rigid_guard` | True | True | 0.781609/0.807229/False | 0.781609/0.807229/False | 0.000000 | 16000 | 438 | 64 |
| `20_namirin_kanzen_shouriesper_girl_tailsdk_insane` | `baseline_guard` | True | True | 0.860335/1.000000/False | 0.860335/1.000000/False | 0.000000 | 16000 | 601 | 0 |
| `20_namirin_kanzen_shouriesper_girl_tailsdk_insane` | `anti_rigid_guard` | True | True | 0.806283/0.792793/False | 0.806283/0.792793/False | 0.000000 | 16000 | 649 | 88 |
| `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | `baseline_guard` | True | True | 0.881720/1.000000/False | 0.881720/1.000000/False | 0.000000 | 16000 | 601 | 0 |
| `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | `anti_rigid_guard` | True | True | 0.848485/0.792793/False | 0.848485/0.792793/False | 0.000000 | 16000 | 649 | 88 |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | `baseline_guard` | True | True | 0.911917/1.000000/False | 0.911917/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | `anti_rigid_guard` | True | True | 0.846561/0.800000/False | 0.846561/0.800000/False | 0.000000 | 16000 | 501 | 76 |
| `23_usao_knight_rider_kuo_kyoka_expert` | `baseline_guard` | True | True | 0.794118/1.000000/False | 0.794118/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `23_usao_knight_rider_kuo_kyoka_expert` | `anti_rigid_guard` | True | True | 0.814815/0.792793/False | 0.814815/0.792793/False | 0.000000 | 16000 | 561 | 88 |
| `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane` | `baseline_guard` | True | True | 0.750000/1.000000/False | 0.750000/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane` | `anti_rigid_guard` | True | True | 0.712766/0.792793/False | 0.712766/0.792793/False | 0.000000 | 16000 | 561 | 88 |
| `25_nekodex_circles_famoss_insane` | `baseline_guard` | True | True | 0.801980/1.000000/False | 0.801980/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `25_nekodex_circles_famoss_insane` | `anti_rigid_guard` | True | True | 0.803738/0.792793/False | 0.803738/0.792793/False | 0.000000 | 16000 | 561 | 88 |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | `baseline_guard` | True | True | 0.754237/1.000000/False | 0.754237/1.000000/False | 0.000000 | 16000 | 513 | 0 |
| `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | `anti_rigid_guard` | True | True | 0.774194/0.792793/False | 0.774194/0.792793/False | 0.000000 | 16000 | 561 | 88 |
| `27_billiummoto_four_veiled_stars_aries_insane` | `baseline_guard` | True | True | 0.818653/1.000000/False | 0.818653/1.000000/False | 0.000000 | 16000 | 475 | 0 |
| `27_billiummoto_four_veiled_stars_aries_insane` | `anti_rigid_guard` | True | True | 0.783069/0.800000/False | 0.783069/0.800000/False | 0.000000 | 16000 | 496 | 75 |
| `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous` | `baseline_guard` | True | True | 0.680328/1.000000/False | 0.680328/1.000000/False | 0.000000 | 16000 | 501 | 0 |
| `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous` | `anti_rigid_guard` | True | True | 0.731518/0.767857/False | 0.731518/0.767857/False | 0.000000 | 16000 | 565 | 88 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `baseline_guard` | True | True | 0.617450/1.000000/False | 0.617450/1.000000/False | 0.000000 | 16000 | 351 | 0 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `anti_rigid_guard` | True | True | 0.236364/0.733333/True | 0.236364/0.733333/True | 0.000000 | 16000 | 274 | 23 |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | `baseline_guard` | True | True | 0.000000/1.000000/False | 0.000000/1.000000/False | 0.000000 | 16000 | 301 | 0 |
| `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | `anti_rigid_guard` | True | True | 0.000000/0.800000/False | 0.000000/0.800000/False | 0.000000 | 16000 | 268 | 21 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `baseline_guard` | True | True | 0.592593/1.000000/False | 0.592593/1.000000/False | 0.000000 | 16000 | 351 | 0 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `anti_rigid_guard` | True | True | 0.294872/0.666667/False | 0.294872/0.666667/False | 0.000000 | 16000 | 282 | 23 |

## Next Step

Compare this opt-in v2.1 repair against current v3 rollout failures before changing defaults.
