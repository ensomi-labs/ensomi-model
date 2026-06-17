# Target Grammar v3 Conditioned Event-Distribution Short Rollout Gate Result Report

## Scope

This pass trains or uses one conditioned-objective v3 checkpoint, then rolls only the CE-sensitive fixed-slice subset. It does not change tokenizer, grammar, decode defaults, or run full32.

## Decision

- route: `KILL`
- reason: conditioned objective tripped a hard safety or event-count calibration guard: median_event_count_ratio_in_range
- next step: Do not run full32; inspect event-count calibration failure or move to target-grammar repair.

## Training

- mapper checkpoint: `artifacts/tmp/mapper_v3_conditioned_event_distribution_short_rollout/train/run/checkpoint.pt`
- control checkpoint: `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- training report: `artifacts/tmp/mapper_v3_conditioned_event_distribution_short_rollout/train/run/report.json`
- training config: `artifacts/tmp/mapper_v3_conditioned_event_distribution_short_rollout/train/conditioned_event_distribution_enabled.yaml`
- max steps: `500`
- loss overrides: `{'lambda_conditioned_event_distribution': 0.05, 'conditioned_event_under_weight': 3.0, 'conditioned_event_over_weight': 1.0, 'conditioned_event_zero_target_over_weight': 2.0, 'conditioned_event_high_difficulty_over_weight': 4.0, 'conditioned_event_high_difficulty_min': 0.75, 'event_token_loss_weight': 1.0}`

## Aggregate

| Metric | CE selected baseline | Conditioned candidate | Delta |
| --- | ---: | ---: | ---: |
| selected cases | `8` | `8` | `n/a` |
| starved cases | `5` | `3` | `-2` |
| rigid cases | `3` | `4` | `1` |
| mean F1@100ms | `0.556890` | `0.606935` | `0.050045` |
| median event ratio | `0.682416` | `0.689944` | `n/a` |
| max boundary ratio | `0.041667` | `0.051282` | `n/a` |

## Guard Results

- case_coverage: `true`
- all_legal: `true`
- no_dead_end_cases: `true`
- no_max_token_cases: `true`
- starved_below_ce_baseline_5: `true`
- rigid_no_worse_than_ce_baseline_11: `true`
- median_event_count_ratio_in_range: `false`
- starved_improved_vs_selected_baseline: `true`

## Starvation vs Flooding

| Case | Reasons | Starved | Event Ratio | Rigid | F1 Delta | Legal |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | overproduction_risk_control | `False` -> `False` | `2.000000` -> `1.860000` | `0.525253` -> `0.500000` | `0.018648` | `True` |
| `07_nekodex_circles_famoss_normal` | pass_like_control | `False` -> `False` | `1.162791` -> `0.453488` | `0.525253` -> `0.473684` | `-0.369462` | `True` |
| `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | rigid_risk_control | `False` -> `True` | `0.382353` -> `0.186275` | `1.000000` -> `1.000000` | `-0.227302` | `True` |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | ce_starved | `True` -> `False` | `0.842105` -> `1.561404` | `0.936170` -> `0.988636` | `0.184344` | `True` |
| `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | ce_starved | `True` -> `True` | `0.522727` -> `0.465909` | `0.977778` -> `0.950000` | `0.023719` | `True` |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | ce_starved | `True` -> `False` | `0.516129` -> `0.924731` | `0.936170` -> `0.976471` | `0.312294` | `True` |
| `27_billiummoto_four_veiled_stars_aries_insane` | ce_starved | `True` -> `False` | `0.516129` -> `0.913978` | `0.936170` -> `0.630952` | `0.459160` | `True` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | ce_starved | `True` -> `True` | `1.022472` -> `0.438202` | `0.977778` -> `0.947368` | `-0.001042` | `True` |

## What This Proves

- This proves only the short trained gate result for the selected CE-sensitive cases.
- It can justify a full32 card only if all guards pass.

## What Remains Unproven

- Full32 conditioned-objective behavior.
- Full-dataset/full-song v3 quality.
- Replacement readiness for mapper or planner.

## Next Step

Do not run full32; inspect event-count calibration failure or move to target-grammar repair.
