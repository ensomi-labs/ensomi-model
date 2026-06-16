# Target Grammar v3 500-Step Fixed-Slice Wide Audit Result Report

## Scope

This pass audits the 500-step v3 checkpoint across all 32 unique beatmaps in the fixed 256-window training slice at a 16s prefix. It does not change tokenizer, grammar, model, loss, training, or decode policy.

## Experiment Card

- Source card: `target_grammar_v3_500step_fixed_slice_wide_audit_experiment_card.md`
- Selected variant: all 32 unique fixed-slice maps, `16000ms`, greedy alpha `0.00`.
- Comparator: six-case 500-step horizon gate and 200-step v3 multicase audit.
- Date: `2026-06-17`

## Result

Decision: `MUTATE`.

- Reason: 500-step v3 remains too rigid or starved on the wider fixed-slice audit for immediate scaling
- Checkpoint: `artifacts/tmp/mapper_v3_500step_training_horizon_gate/train/run/checkpoint.pt`
- Cases: `32` maps across `15` audio families
- All legal: `True`
- Mean / median F1 @100ms: `0.640` / `0.679`
- Mean / median dominant-spacing ratio: `0.749` / `0.736`
- Starved cases: `11`
- Rigid cases: `7`
- Median event-count ratio: `1.000`
- Max boundary-event ratio: `0.200`
- Positive signal passed: `False`

## Aggregate Comparison

| Metric | 32-case v3 500 | 6-case v3 500 | 6-case v3 200 |
| --- | ---: | ---: | ---: |
| Mean F1 @100ms | `0.640` | `0.762` | `0.643` |
| Mean event-count ratio | `0.971` | `1.592` | `0.918` |
| Mean dominant-spacing ratio | `0.749` | `0.733` | `0.993` |
| Starved case count | `11` | `0` | `2` |

## Difficulty Bands

| Band | Cases | Mean F1 | Median event ratio | Mean rigid ratio | Rigid cases | Starved cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `high_>=4` | `10` | `0.514` | `0.912` | `0.713` | `2` | `5` |
| `low_<3` | `11` | `0.699` | `1.103` | `0.732` | `2` | `1` |
| `mid_3to4` | `11` | `0.694` | `1.151` | `0.799` | `3` | `5` |

## Worst Cases

| Category | Case | Diff | Gen/Ref | F1 | Rigid | 2nd share | Dominant spacing | First generated times |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lowest_f1` | `30_goreshit_one_way_to_hannover_cokiiplay_autophobia` | `4.75` | `5/0` | `0.000` | `0.750` | `0.200` | `0` | `[7980, 7980, 7980, 7980, 15920]` |
| `lowest_f1` | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `4.69` | `51/49` | `0.240` | `0.760` | `0.039` | `160` | `[1780, 1940, 2100, 2260, 2420, 2580]` |
| `lowest_f1` | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `5.12` | `40/89` | `0.248` | `0.974` | `0.000` | `160` | `[1780, 1940, 2100, 2260, 2420, 2580]` |
| `lowest_f1` | `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | `4.49` | `47/136` | `0.426` | `0.957` | `0.021` | `160` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `lowest_f1` | `23_usao_knight_rider_kuo_kyoka_expert` | `4.26` | `49/104` | `0.471` | `0.917` | `0.061` | `160` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `highest_rigid` | `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | `2.25` | `102/77` | `0.860` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `highest_rigid` | `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | `2.79` | `102/77` | `0.860` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `highest_rigid` | `20_namirin_kanzen_shouriesper_girl_tailsdk_insane` | `3.79` | `102/79` | `0.851` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `highest_rigid` | `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | `3.84` | `102/86` | `0.894` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `highest_rigid` | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `5.12` | `40/89` | `0.248` | `0.974` | `0.000` | `160` | `[1780, 1940, 2100, 2260, 2420, 2580]` |
| `lowest_second_share` | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `5.12` | `40/89` | `0.248` | `0.974` | `0.000` | `160` | `[1780, 1940, 2100, 2260, 2420, 2580]` |
| `lowest_second_share` | `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | `3.29` | `47/88` | `0.622` | `0.957` | `0.021` | `160` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `lowest_second_share` | `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | `4.49` | `47/136` | `0.426` | `0.957` | `0.021` | `160` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `lowest_second_share` | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `4.69` | `51/49` | `0.240` | `0.760` | `0.039` | `160` | `[1780, 1940, 2100, 2260, 2420, 2580]` |
| `lowest_second_share` | `14_oomori_seiko_justadice_tv_size_remu_hard` | `3.19` | `48/87` | `0.607` | `0.936` | `0.042` | `160` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `highest_event_ratio` | `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | `3.09` | `123/57` | `0.589` | `0.582` | `0.407` | `100` | `[410, 820, 920, 1020, 1120, 1220]` |
| `highest_event_ratio` | `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `2.31` | `101/50` | `0.662` | `0.530` | `0.535` | `150` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `highest_event_ratio` | `16_hatsuki_yura_guren_yasha_a_m_d_hard` | `3.29` | `100/73` | `0.763` | `0.525` | `0.540` | `150` | `[820, 980, 1140, 1300, 1460, 1620]` |
| `highest_event_ratio` | `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | `2.25` | `102/77` | `0.860` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |
| `highest_event_ratio` | `09_namirin_kanzen_shouriesper_girl_tailsdk_hard` | `2.79` | `102/77` | `0.860` | `0.980` | `0.529` | `150` | `[820, 970, 1120, 1270, 1420, 1570]` |

## What Passed

- All 32 runtime-backed rollouts completed legally with no dead-end or max-token failure.
- The audit covers the complete fixed training-slice beatmap universe, not only the six selected examples.
- The summary records difficulty-band, audio-family, and worst-case metrics for follow-up targeting.

## What Surfaced

- Positive signal failed under the predefined fixed-slice gate: `11` starved cases, max boundary ratio `0.200`, and mean F1 `0.640`.
- One case has zero reference events in the first 16s; the model still emitted boundary-duplicated events, so this is a tail quality failure rather than an ordinary F1 comparison.
- Failures cluster more in mid/high difficulty bands: high>=4 has `5` starved cases, mid 3-4 has `5`, low<3 has `1`.
- This is still training-slice evidence; it does not prove held-out, full 4K, or replacement readiness.

## Interpretation

This is a MUTATE result. The six-case 500-step pass did not generalize strongly enough across the fixed slice; inspect failure clusters before scaling v3.

## Verification

- Runtime/v3 diagnostic guard: `13 passed`.
- v3 rollout/model guard: `9 passed`.
- Summary JSON validation: passed.

## Next Step

Create a targeted failure-cluster diagnostic or loss/training calibration card before a larger v3 audit.
