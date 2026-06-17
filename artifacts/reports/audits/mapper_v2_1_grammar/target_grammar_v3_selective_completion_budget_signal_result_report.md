# Target Grammar v3 Selective Completion-Budget Signal Result Report

## Scope

This runtime-backed smoke tests a default-off v3 logits transform that promotes rank-near event tokens only when generated-prefix events are below an online control-density budget. It does not change tokenizer, grammar, replay, model weights, training, or runtime defaults.

## Result

Decision: `KILL_SELECTIVE_COMPLETION_BUDGET_SIGNAL`.

- Reason: event-ratio cap failed: primary_over=2, sentinel_over=2
- Recommended next step: Pivot to v2.1 grammar improvement or a structural v3 grammar mutation card.
- Primary positives: `3` / `6`
- Primary event-ratio over-cap: `2`
- Sentinel event-ratio over-cap: `2`
- Sentinel median event ratio: `1.229701`
- Mean primary F1 delta: `0.257752`
- Total forced events: `11`

## Case Table

| Group | Case | legal | forced | budget | 2nd share base->candidate | event ratio base->candidate | F1 delta | positive | over cap |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| primary | `14_oomori_seiko_justadice_tv_size_remu_hard` | True | 2 | 53 | 0.041667->0.634921 | 0.551724->1.448276 | 0.209494 | False | True |
| primary | `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | True | 1 | 52 | 0.041667->0.540000 | 0.527473->1.098901 | 0.296207 | True | False |
| primary | `19_nekodex_circles_famoss_hard` | True | 1 | 52 | 0.041667->0.544554 | 0.533333->1.122222 | 0.347371 | True | False |
| primary | `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | True | 1 | 53 | 0.041667->0.540000 | 0.516129->1.075269 | 0.330357 | True | False |
| primary | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | True | 0 | 47 | 0.000000->0.000000 | 0.449438->0.449438 | 0.000000 | False | False |
| primary | `18_billiummoto_four_veiled_stars_aries_famoss_hard` | True | 1 | 54 | 0.098039->0.540000 | 0.662338->1.298701 | 0.363083 | False | True |
| sentinel | `23_usao_knight_rider_kuo_kyoka_expert` | True | 1 | 52 | 0.061224->0.684932 | 0.471154->1.403846 | 0.329412 | False | True |
| sentinel | `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | False | 0 | n/a | 0.039216->0.000000 | 1.040816->0.816327 | 0.029663 | False | False |
| sentinel | `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous` | True | 1 | 96 | 0.353333->0.361842 | 1.041667->1.055556 | 0.008090 | False | False |
| sentinel | `01_hatsuki_yura_guren_yasha_a_m_d_normal` | True | 3 | 25 | 0.720000->0.800000 | 1.190476->1.587302 | -0.094425 | False | True |

## Interpretation

The online budget-conditioned local decode repair is not strong enough as configured.

## Budget Trace Notes

The transform uses only generated-prefix event counts and `density_teacher_8s` from the runtime control batch. Reference beatmaps are used only after generation for evaluation metrics.

## Next Step

Pivot to v2.1 grammar improvement or a structural v3 grammar mutation card.
