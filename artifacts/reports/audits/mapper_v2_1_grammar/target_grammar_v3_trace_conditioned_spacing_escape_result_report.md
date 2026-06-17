# Target Grammar v3 Trace-Conditioned Spacing Escape Smoke Result Report

## Scope

This pass runs an eval-only generated-prefix spacing/boundary escape policy on the six traced v3 failures plus four sentinel controls. It does not train, change tokenizer behavior, change grammar defaults, or use target labels during decode.

## Decision

- route: `KILL_OVERPRODUCTION`
- reason: trace-conditioned spacing escape failed overproduction guards: ['primary_event_ratio_guard', 'sentinel_median_event_ratio_guard']

## Aggregate

| Metric | Baseline primary | Candidate primary |
| --- | ---: | ---: |
| mean dominant spacing | `0.933173` | `0.513972` |
| mean second-window share | `0.044118` | `0.491164` |
| mean event ratio | `0.540072` | `1.222994` |
| mean F1@100ms | `0.507910` | `0.825051` |

- Primary rigidity-improved cases: `6` / `6`
- Primary second-window-improved cases: `6` / `6`
- Primary both-improved cases: `6` / `6`
- Sentinel median event ratio: `1.634997`
- Total activations: `158`
- Spacing activations: `144`
- Boundary activations: `14`

## Guard Results

- case_coverage: `true`
- no_training: `true`
- no_tokenizer_or_default_decode_change: `true`
- no_target_leakage: `true`
- all_legal: `true`
- no_dead_end_cases: `true`
- no_max_token_cases: `true`
- primary_event_ratio_guard: `false`
- sentinel_median_event_ratio_guard: `false`
- max_boundary_ratio_guard: `true`
- mean_f1_guard: `true`

## Cases

| group | case | event ratio | second share | dominant spacing | F1 delta | activations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `primary` | `14_oomori_seiko_justadice_tv_size_remu_hard` | `1.149425` | `0.540000` | `0.525253` | `0.323074` | `17` |
| `primary` | `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | `1.098901` | `0.540000` | `0.505051` | `0.285736` | `18` |
| `primary` | `19_nekodex_circles_famoss_hard` | `1.144444` | `0.553398` | `0.509804` | `0.338364` | `18` |
| `primary` | `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | `1.365591` | `0.370079` | `0.531746` | `0.200258` | `16` |
| `primary` | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `1.280899` | `0.403509` | `0.486726` | `0.392332` | `17` |
| `primary` | `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `1.298701` | `0.540000` | `0.525253` | `0.363083` | `17` |
| `sentinel` | `23_usao_knight_rider_kuo_kyoka_expert` | `1.682692` | `0.537143` | `0.879310` | `0.267763` | `17` |
| `sentinel` | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `2.326531` | `0.403509` | `0.486726` | `0.324417` | `18` |
| `sentinel` | `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous` | `1.347222` | `0.500000` | `0.689119` | `0.031598` | `9` |
| `sentinel` | `01_hatsuki_yura_guren_yasha_a_m_d_normal` | `1.587302` | `0.800000` | `0.797980` | `-0.094425` | `11` |

## Interpretation

The smoke repeated the known anti-rigid overproduction failure mode. This argues against local decode repair.

## Evidence Boundary

- Proved: whether this bounded eval-only local decode policy clears its six-case smoke and sentinel guards.
- Not proved: full32 quality, default-decode readiness, training objective value, or target-grammar replacement readiness.

## Next Step

Stop this local decode repair and pivot to training-side state/objective work or v2.1 grammar mutation.
