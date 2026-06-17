# Target Grammar v3 CE Anti-Rigid Full-32 Decode Gate Result Report

## Scope

This pass reruns all 32 committed CE-weight v3 fixed-slice cases with an opt-in v3 anti-rigid logits transform. It does not change mapper defaults, model weights, tokenization, or training.

## Decision

- route: `KILL`
- reason: full-32 soft-penalty-4 anti-rigid decode failed an original fixed-slice gate
- transform: `soft-penalty-4`

## Aggregate

| Metric | CE baseline full-32 | Anti-rigid candidate | Delta |
| --- | ---: | ---: | ---: |
| rigid cases | `11` | `0` | `-11` |
| starved cases | `5` | `1` | `-4` |
| mean F1@100ms | `0.691899` | `0.703179` | `0.011280` |
| mean dominant spacing | `0.745180` | `0.535766` | `-0.209414` |
| mean event ratio | `1.059223` | `1.325593` | `n/a` |
| median event ratio | `1.156108` | `1.412331` | `n/a` |
| max boundary ratio | `0.041667` | `0.051724` | `n/a` |

## Gate Results

- case_coverage: `true`
- all_legal: `true`
- no_dead_end_cases: `true`
- no_max_token_cases: `true`
- starved_below_original_baseline: `true`
- rigid_no_worse_than_original_baseline: `true`
- mean_f1_within_original_floor: `true`
- median_event_count_ratio_in_range: `false`
- max_boundary_no_worse_than_original_baseline: `true`
- ce_rigid_reduction_at_least_four: `true`
- ce_starvation_not_materially_worse: `true`
- ce_mean_f1_not_materially_worse: `true`

## Transform

- total candidates: `592`
- total blocks: `592`
- dead-end cases: `0`
- max-token cases: `0`

## Worst Cases

Highest candidate rigid ratios:
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: rigid `0.980198` -> `0.800000`, f1 `0.860335` -> `0.819149`, blocks `21`
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: rigid `0.980198` -> `0.795918`, f1 `0.860335` -> `0.829545`, blocks `18`
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`: rigid `0.462366` -> `0.752475`, f1 `0.601399` -> `0.622517`, blocks `19`
- `23_usao_knight_rider_kuo_kyoka_expert`: rigid `0.530612` -> `0.750000`, f1 `0.827586` -> `0.747082`, blocks `29`
- `01_hatsuki_yura_guren_yasha_a_m_d_normal`: rigid `0.875000` -> `0.727273`, f1 `0.461538` -> `0.555556`, blocks `7`

## Next Step

Do not scale this soft-penalty-4 policy; inspect full-32 failures or mutate decode/training strategy.
