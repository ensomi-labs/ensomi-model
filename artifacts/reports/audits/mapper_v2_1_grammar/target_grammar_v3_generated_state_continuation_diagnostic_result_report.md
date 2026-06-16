# Target Grammar v3 Generated-State Continuation Diagnostic Result Report

## Scope

This artifact-only diagnostic compares existing 32-case v3 rollout summaries. It does not retrain, change grammar, or change inference behavior. The goal is to choose the next bounded grammar/decode experiment after C3 mapper-side paths showed diminishing returns.

## Result

Decision: `TEST_DECODE_TIMING_CALIBRATION`.

- Reason: boundary/starvation persists while event tokens remain broadly valid
- Baseline: `baseline_500step`
- Matched cases: `32`
- Recommended next step: Create a bounded decode timing-calibration card before more training or grammar replacement.

## Failure-Class Table

| Variant | legal | starved | rigid | boundary clamp | late spill | dead end | max token | undergen | overgen | median event ratio | mean F1 | event top-k minus top-1 | event valid ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_500step | True | 11 | 12 | 10 | 9 | 0 | 0 | 10 | 6 | 1.000000 | 0.639566 | 0.207920 | 0.995887 |
| same_ms_guard | True | 22 | 26 | 22 | 18 | 0 | 0 | 19 | 4 | 0.585366 | 0.556877 | 0.180471 | 0.758776 |
| event_budget_lambda005 | False | 18 | 25 | 18 | 14 | 0 | 4 | 15 | 8 | 0.716418 | 0.513257 | 0.170810 | 0.996449 |
| continuation_jump_lambda005 | False | 20 | 24 | 20 | 14 | 3 | 0 | 17 | 7 | 0.631579 | 0.569581 | 0.156951 | 0.730265 |

## Shared Failure Overlap

- Cases starved in every variant: `7`
- Cases boundary-clamped in every variant: `6`
- Cases rigid in every variant: `10`
- Cases starved in baseline and latest (`continuation_jump_lambda005`): `10`
- Cases boundary-clamped in baseline and latest (`continuation_jump_lambda005`): `9`

## Highest-Leverage Cases

- `14_oomori_seiko_justadice_tv_size_remu_hard`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `19_nekodex_circles_famoss_hard`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `18_billiummoto_four_veiled_stars_aries_famoss_hard`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `20_namirin_kanzen_shouriesper_girl_tailsdk_insane`: starved=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane`: starved=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `23_usao_knight_rider_kuo_kyoka_expert`: starved=['baseline_500step', 'same_ms_guard', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`: starved=['baseline_500step', 'same_ms_guard', 'continuation_jump_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'continuation_jump_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: starved=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], boundary=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], rigid=['same_ms_guard', 'event_budget_lambda005', 'continuation_jump_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005
- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: starved=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005'], boundary=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005'], rigid=['baseline_500step', 'same_ms_guard', 'event_budget_lambda005'], variants=baseline_500step, same_ms_guard, event_budget_lambda005, continuation_jump_lambda005

## Interpretation

The failure is mostly generated-state timing/ranking collapse rather than basic grammar legality. The next small test should change decode/target timing pressure on existing checkpoints, with high-difficulty and boundary-clamp guards.

## Next Step

Create a bounded decode timing-calibration card before more training or grammar replacement.
