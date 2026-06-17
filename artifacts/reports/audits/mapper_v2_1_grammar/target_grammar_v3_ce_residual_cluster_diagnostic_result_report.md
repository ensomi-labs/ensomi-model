# Target Grammar v3 Generated-State Continuation Diagnostic Result Report

## Scope

This artifact-only diagnostic compares existing 32-case v3 rollout summaries. It does not retrain, change grammar, or change inference behavior. The goal is to choose the next bounded grammar/decode experiment after C3 mapper-side paths showed diminishing returns.

## Result

Decision: `TEST_EVENT_RANKING_CALIBRATION`.

- Reason: event candidates are often rank-near but underselected
- Baseline: `baseline_500step`
- Matched cases: `32`
- Recommended next step: Create a bounded event-ranking calibration card on the fixed 32-case slice.

## Failure-Class Table

| Variant | legal | starved | rigid | boundary clamp | late spill | dead end | max token | undergen | overgen | median event ratio | mean F1 | event top-k minus top-1 | event valid ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_500step | True | 11 | 12 | 10 | 9 | 0 | 0 | 10 | 6 | 1.000000 | 0.639566 | 0.207920 | 0.995887 |
| ce_weight2 | True | 5 | 14 | 5 | 4 | 0 | 0 | 9 | 12 | 1.156108 | 0.691899 | 0.100976 | 0.660418 |

## Shared Failure Overlap

- Cases starved in every variant: `3`
- Cases boundary-clamped in every variant: `3`
- Cases rigid in every variant: `7`
- Cases starved in baseline and latest (`ce_weight2`): `3`
- Cases boundary-clamped in baseline and latest (`ce_weight2`): `3`

## Highest-Leverage Cases

- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: starved=['baseline_500step', 'ce_weight2'], boundary=['baseline_500step', 'ce_weight2'], rigid=['baseline_500step', 'ce_weight2'], variants=baseline_500step, ce_weight2
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: starved=['baseline_500step', 'ce_weight2'], boundary=['baseline_500step', 'ce_weight2'], rigid=['baseline_500step', 'ce_weight2'], variants=baseline_500step, ce_weight2
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: starved=['baseline_500step', 'ce_weight2'], boundary=['baseline_500step', 'ce_weight2'], rigid=['baseline_500step', 'ce_weight2'], variants=baseline_500step, ce_weight2
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: starved=['ce_weight2'], boundary=['ce_weight2'], rigid=['ce_weight2'], variants=baseline_500step, ce_weight2
- `14_oomori_seiko_justadice_tv_size_remu_hard`: starved=['baseline_500step'], boundary=['baseline_500step'], rigid=['baseline_500step'], variants=baseline_500step, ce_weight2
- `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`: starved=['baseline_500step'], boundary=['baseline_500step'], rigid=['baseline_500step'], variants=baseline_500step, ce_weight2
- `19_nekodex_circles_famoss_hard`: starved=['baseline_500step'], boundary=['baseline_500step'], rigid=['baseline_500step'], variants=baseline_500step, ce_weight2
- `23_usao_knight_rider_kuo_kyoka_expert`: starved=['baseline_500step'], boundary=['baseline_500step'], rigid=['baseline_500step'], variants=baseline_500step, ce_weight2
- `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`: starved=['baseline_500step'], boundary=['baseline_500step'], rigid=['baseline_500step'], variants=baseline_500step, ce_weight2
- `27_billiummoto_four_veiled_stars_aries_insane`: starved=['ce_weight2'], boundary=['ce_weight2'], rigid=['ce_weight2'], variants=baseline_500step, ce_weight2
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: starved=[], boundary=[], rigid=['baseline_500step', 'ce_weight2'], variants=baseline_500step, ce_weight2
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: starved=[], boundary=[], rigid=['baseline_500step', 'ce_weight2'], variants=baseline_500step, ce_weight2

## Interpretation

The evidence favors event-ranking calibration. Test a decode-time or target-side rank objective before changing the v3 grammar.

## Next Step

Create a bounded event-ranking calibration card on the fixed 32-case slice.
