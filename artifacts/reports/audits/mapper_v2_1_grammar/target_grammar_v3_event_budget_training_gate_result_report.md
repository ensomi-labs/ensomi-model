# Target Grammar v3 Event-Budget Training Gate Result Report

## Scope

This report tests `lambda_event_budget=0.05` on the fixed 32-song/256-window v3 slice after hardening the event-budget loss against padded all-invalid rows. It is a calibration gate, not replacement approval.

## Training Result

- Training completed: `True` at `500` steps.
- Final eval total loss: `1.992164`.
- Final eval event-budget loss: `0.219440`.
- Loss config: `{'density_calibration_bias': 0.0, 'density_calibration_scale': 1.0, 'lambda_adapter_reg': 1e-05, 'lambda_density': 0.05, 'lambda_event_budget': 0.05, 'lambda_ln_close': 0.05, 'ln_close_focal_gamma': 1.5, 'ln_close_pos_weight': 1.0}`.

## 32-Case Rollout Result

Decision: `MUTATE`.

- Reason: failed checks: all_32_rollouts_legal, no_max_token_cases, starved_cases_below_baseline_11, rigid_cases_no_worse_than_baseline_7, median_event_ratio_in_0_80_1_25, max_boundary_ratio_no_worse_than_baseline_0_200
- All legal: `False`
- Max-token cases: `4`
- Second-window starved cases: `18`
- Rigid cases: `8`
- Median event-count ratio: `0.716418`
- Mean F1 @100ms: `0.513257`
- Median second-window share: `0.047619`
- Max boundary-event ratio: `1.000000`

## Baseline Comparison

| Metric | 500-step baseline | Event-budget 0.05 | Delta |
| --- | ---: | ---: | ---: |
| Max-token cases | `0` | `4` | `4` |
| Starved cases | `11` | `18` | `7` |
| Rigid cases | `7` | `8` | `1` |
| Median event ratio | `1.000000` | `0.716418` | `-0.283582` |
| Mean F1 @100ms | `0.639566` | `0.513257` | `-0.126308` |
| Median second-window share | `0.490902` | `0.047619` | `-0.443283` |

## Passed

- Event-budget loss is now finite on the real cached batch and through 500-step training.
- The 500-step checkpoint loads through the existing v3 runtime path.
- All 32 rollout attempts produced summaries; the failure is behavioral, not an evaluator crash.

## Surfaced

- `4` high-difficulty cases hit max tokens, which is an automatic gate failure.
- Starved cases worsened from `11` to `18`.
- Median event-count ratio fell to `0.716418`, below the pass band.
- Overgeneration cases reached `6` with max event-count ratio `4.774194`.
- The objective is too blunt as a global scalar weight: high-difficulty cases show both max-token overgeneration and starvation tails.

## Worst Cases

- Max-token cases: `23_usao_knight_rider_kuo_kyoka_expert, 25_nekodex_circles_famoss_insane, 26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer, 27_billiummoto_four_veiled_stars_aries_insane`
- Lowest second-window share: `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv=0.000, 15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return=0.000, 30_goreshit_one_way_to_hannover_cokiiplay_autophobia=0.000, 31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial=0.000, 12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd=0.021`
- Highest event ratio: `27_billiummoto_four_veiled_stars_aries_insane=4.774, 25_nekodex_circles_famoss_insane=4.363, 23_usao_knight_rider_kuo_kyoka_expert=4.269, 26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer=3.272, 00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm=2.159`

## Decision

Route: `MUTATE`. Do not scale `lambda_event_budget=0.05` as-is. Keep the NaN hardening, but mutate the objective toward lower weight, difficulty-conditioned/event-budget conditioning, or an ordered timing-continuity objective with explicit high-difficulty max-token guards.

## Next Step

Do not scale lambda_event_budget=0.05. Mutate to a lower or conditioned event-budget objective with high-difficulty overgeneration guards.
