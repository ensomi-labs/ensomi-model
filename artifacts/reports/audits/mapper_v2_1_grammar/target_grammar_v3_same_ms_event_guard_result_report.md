# Target Grammar v3 Same-ms Event Guard Result Report

## Scope

This report evaluates a v3 replay/grammar guard that forbids more than one event group at the same `current_ms` during online generation. It uses existing checkpoints and does not retrain.

## Code Change

- Added `event_emitted_at_current_ms` to replay state.
- v3 time-shift transitions reset the flag; v3 event transitions set it.
- v3 valid-token masks reject event tokens when the flag is already true.
- Token vocabulary and target tokenization are unchanged.

## Four-Case Smoke

- Max-token cases: `0`
- Max duplicate/non-increasing spacing ratio: `0.000000`
- Max boundary-event ratio: `0.071429`
- Mean F1 @100ms: `0.461512`

## Full 32-Case Gate

Decision: `TEST_NEXT`.

- Reason: same-ms event guard removed max-token duplicate loops without dead ends
- All legal: `True`
- Max-token cases: `0`
- Starved cases: `22`
- Rigid cases: `9`
- Median event-count ratio: `0.585366`
- Mean F1 @100ms: `0.556877`
- Median second-window share: `0.047619`
- Max duplicate/non-increasing spacing ratio: `0.000000`
- Max boundary-event ratio: `1.000000`

## Pre-Guard Comparison

| Metric | Pre-guard event-budget | Same-ms guard | Delta |
| --- | ---: | ---: | ---: |
| Max-token cases | `4` | `0` | `-4` |
| Max duplicate ratio | `0.909910` | `0.000000` | `-0.909910` |
| Max boundary ratio | `1.000000` | `1.000000` | `0.000000` |
| Starved cases | `18` | `22` | `4` |
| Mean F1 @100ms | `0.513257` | `0.556877` | `0.043620` |

## Passed

- The repeated same-timestamp event wall is removed from all four formerly max-token cases.
- Full 32-case rollout is legal: no dead-end and no max-token failures.
- Existing v3 replay/rollout tests pass with the new state bit.

## Surfaced

- Starvation remains high: `22` cases.
- Rigid-grid cases remain high: `9` cases.
- Median event-count ratio is still low: `0.585366`.
- The guard fixes legality, not timing quality or event-budget calibration.

## Worst Cases

- Lowest second-window share: `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv=0.000, 15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return=0.000, 30_goreshit_one_way_to_hannover_cokiiplay_autophobia=0.000, 31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial=0.000, 12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd=0.021`
- Highest event ratio: `00_ohara_yuiko_zero_centimeters_tv_size_mikan_shini_s_nm=2.159, 05_usao_knight_rider_kuo_kyoka_dnm_s_normal=2.000, 06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd=1.456, 03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal=1.351, 28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous=1.243`

## Decision

Route: `TEST_NEXT` for grammar legality hardening. Keep the guard. The next mutation should target remaining timing-continuity and rigid-grid/starvation failures, not same-ms duplicate loops.
