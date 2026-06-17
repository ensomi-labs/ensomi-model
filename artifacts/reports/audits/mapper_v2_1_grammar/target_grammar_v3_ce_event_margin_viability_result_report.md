# Target Grammar v3 Event-Margin Decode Viability Result Report

## Scope

This artifact-only diagnostic checks whether existing v3 rollout logit margins support a simple global event-token decode bonus. It does not retrain, change grammar, change tokenizer, or change runtime defaults.

## Result

Decision: `TEST_SELECTIVE_TRACE_ORACLE`.

- Reason: 5/5 starved cases are low-bias, but the class is mixed
- Baseline: `ce_weight2`
- Recommended next step: Add per-step trace/oracle instrumentation for the low-bias starved cases.

## Variant Table

| Variant | cases | starved | rigid | undergen | overgen | median required bias | starved >4 | starved >6 | pass-like <=4 | event top-k minus top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_500step | 32 | 11 | 12 | 10 | 6 | 6.639421 | 8 | 6 | 3 | 0.207920 |
| ce_weight2 | 32 | 5 | 14 | 9 | 12 | 1.206822 | 0 | 0 | 8 | 0.100976 |

## Cross-Variant Sanity Check

- No compared variant has both lower median required bias and worse starvation than baseline.

## Baseline Failure-Class Bias

| Class | cases | median required bias | <=2 | <=4 | <=6 | >6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starved | 5 | 3.260576 | 1 | 5 | 5 | 0 |
| rigid | 14 | 1.359373 | 9 | 14 | 14 | 0 |
| undergenerated | 9 | 1.028271 | 5 | 9 | 9 | 0 |
| overgenerated | 12 | 1.090671 | 12 | 12 | 12 | 0 |
| pass_like | 8 | 1.245080 | 7 | 8 | 8 | 0 |
| failed | 24 | 1.106333 | 18 | 24 | 24 | 0 |

## Highest Required-Bias Failed Cases

- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: required_bias=3.990424, margin=-3.990424, rank=9.000000, classes=starved, rigid, undergenerated, event_ratio=0.522727, second_share=0.000000, rigid=0.977778
- `27_billiummoto_four_veiled_stars_aries_insane`: required_bias=3.306348, margin=-3.306348, rank=4.000000, classes=starved, rigid, undergenerated, event_ratio=0.516129, second_share=0.041667, rigid=0.936170
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: required_bias=3.260576, margin=-3.260576, rank=4.000000, classes=starved, rigid, undergenerated, event_ratio=0.516129, second_share=0.041667, rigid=0.936170
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: required_bias=3.233065, margin=-3.233065, rank=4.000000, classes=starved, rigid, event_ratio=0.842105, second_share=0.041667, rigid=0.936170
- `30_goreshit_one_way_to_hannover_cokiiplay_autophobia`: required_bias=2.655377, margin=-2.655377, rank=2.000000, classes=undergenerated, event_ratio=0.000000, second_share=0.023256, rigid=0.880952
- `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane`: required_bias=2.001572, margin=-2.001572, rank=2.500000, classes=rigid, event_ratio=1.186047, second_share=0.529412, rigid=0.980198
- `20_namirin_kanzen_shouriesper_girl_tailsdk_insane`: required_bias=1.986213, margin=-1.986213, rank=2.500000, classes=rigid, overgenerated, event_ratio=1.291139, second_share=0.529412, rigid=0.980198
- `16_hatsuki_yura_guren_yasha_a_m_d_hard`: required_bias=1.681006, margin=-1.681006, rank=4.000000, classes=overgenerated, event_ratio=1.369863, second_share=0.540000, rigid=0.525253
- `24_hatsuki_yura_guren_yasha_a_m_d_poke_s_insane`: required_bias=1.494416, margin=-1.494416, rank=2.500000, classes=rigid, overgenerated, event_ratio=1.342105, second_share=0.529412, rigid=0.980198
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: required_bias=1.224331, margin=-1.224331, rank=3.000000, classes=rigid, overgenerated, event_ratio=1.324675, second_share=0.529412, rigid=0.980198
- `18_billiummoto_four_veiled_stars_aries_famoss_hard`: required_bias=1.223323, margin=-1.223323, rank=4.000000, classes=overgenerated, event_ratio=1.298701, second_share=0.540000, rigid=0.525253
- `06_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_hd`: required_bias=1.184395, margin=-1.184395, rank=2.000000, classes=overgenerated, event_ratio=1.470588, second_share=0.540000, rigid=0.525253

## Low-Bias Starved Cases

- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: required_bias=3.233065, margin=-3.233065, rank=4.000000, event_ratio=0.842105, second_share=0.041667
- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: required_bias=3.990424, margin=-3.990424, rank=9.000000, event_ratio=0.522727, second_share=0.000000
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: required_bias=3.260576, margin=-3.260576, rank=4.000000, event_ratio=0.516129, second_share=0.041667
- `27_billiummoto_four_veiled_stars_aries_insane`: required_bias=3.306348, margin=-3.306348, rank=4.000000, event_ratio=0.516129, second_share=0.041667
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: required_bias=0.195628, margin=-0.195628, rank=2.000000, event_ratio=1.022472, second_share=0.010989

## Interpretation

A global event bonus is not well supported, but some starved cases may be close enough for a selective state-aware selector. The current summaries are too coarse to define that selector safely.

## Next Step

Add per-step trace/oracle instrumentation for the low-bias starved cases.
