# Target Grammar v3 CE Anti-Rigid Decode Stress Result Report

## Scope

This pass reruns only the rigid cases from the committed CE-weight v3 gate with an opt-in v3 anti-rigid logits transform. It does not change mapper defaults, model weights, tokenization, or training.

## Decision

- route: `TEST_NEXT`
- reason: anti-rigid hard block reduced stress rigidity without reviving starvation

## Aggregate

| Metric | CE baseline stress | Anti-rigid candidate | Delta |
| --- | ---: | ---: | ---: |
| rigid cases | `11` | `0` | `-11` |
| starved cases | `2` | `1` | `-1` |
| mean F1@100ms | `0.681342` | `0.711130` | `0.029788` |
| mean dominant spacing | `0.981088` | `0.573290` | `-0.407798` |
| mean event ratio | `0.961089` | `1.192467` | `n/a` |

## Transform

- total candidates: `169`
- total blocks: `169`
- dead-end cases: `0`
- max-token cases: `0`

## Worst Cases

Highest candidate rigid ratios:
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: rigid `0.980198` -> `0.800000`, f1 `0.860335` -> `0.819149`, blocks `21`
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: rigid `0.980198` -> `0.795918`, f1 `0.860335` -> `0.829545`, blocks `18`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: rigid `0.977778` -> `0.686275`, f1 `0.266667` -> `0.255319`, blocks `9`
- `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane`: rigid `0.980198` -> `0.624000`, f1 `0.730337` -> `0.702970`, blocks `24`
- `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`: rigid `0.989474` -> `0.620879`, f1 `0.859701` -> `0.831804`, blocks `15`

## Next Step

Run the transform on the full 32-case CE-weight fixed slice before any default change.
