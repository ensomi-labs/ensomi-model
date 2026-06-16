# Target Grammar v3 Event-Margin Decode Viability Result Report

## Scope

This artifact-only diagnostic checks whether existing v3 rollout logit margins support a simple global event-token decode bonus. It does not retrain, change grammar, change tokenizer, or change runtime defaults.

## Result

Decision: `KILL_GLOBAL_EVENT_BONUS`.

- Reason: 8/11 starved cases need median event bias > 4.0
- Baseline: `baseline_500step`
- Recommended next step: Do not run another global decode-bonus sweep. Use selective per-step trace/oracle instrumentation on the few low-bias starved cases, or pivot to v2.1 grammar improvement if the owner wants to stop v3 local probes.

## Variant Table

| Variant | cases | starved | rigid | undergen | overgen | median required bias | starved >4 | starved >6 | pass-like <=4 | event top-k minus top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_500step | 32 | 11 | 12 | 10 | 6 | 6.639421 | 8 | 6 | 3 | 0.207920 |
| same_ms_guard | 32 | 22 | 26 | 19 | 4 | 3.110742 | 5 | 0 | 3 | 0.180471 |
| event_budget_lambda005 | 32 | 18 | 25 | 15 | 8 | 6.793718 | 15 | 13 | 0 | 0.170810 |
| continuation_jump_lambda005 | 32 | 20 | 24 | 17 | 7 | 1.877381 | 2 | 0 | 5 | 0.156951 |

## Cross-Variant Sanity Check

- `same_ms_guard` has lower median required bias (3.110742) than baseline but worse starvation (22 vs 11).
- `continuation_jump_lambda005` has lower median required bias (1.877381) than baseline but worse starvation (20 vs 11).

## Baseline Failure-Class Bias

| Class | cases | median required bias | <=2 | <=4 | <=6 | >6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starved | 11 | 6.822336 | 3 | 3 | 5 | 6 |
| rigid | 12 | 7.301215 | 2 | 2 | 4 | 8 |
| undergenerated | 10 | 6.375671 | 3 | 3 | 5 | 5 |
| overgenerated | 6 | 7.892950 | 0 | 0 | 0 | 6 |
| pass_like | 14 | 5.887003 | 1 | 3 | 8 | 6 |
| failed | 18 | 7.070160 | 3 | 3 | 5 | 13 |

## Highest Required-Bias Failed Cases

- `20_namirin_kanzen_shouriesper_girl_tailsdk_insane`: required_bias=8.860270, margin=-8.860270, rank=9.000000, classes=rigid, overgenerated, event_ratio=1.291139, second_share=0.529412, rigid=0.980198
- `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane`: required_bias=8.846232, margin=-8.846232, rank=9.000000, classes=rigid, event_ratio=1.186047, second_share=0.529412, rigid=0.980198
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: required_bias=8.702404, margin=-8.702404, rank=10.000000, classes=rigid, overgenerated, event_ratio=1.324675, second_share=0.529412, rigid=0.980198
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: required_bias=8.335334, margin=-8.335334, rank=10.000000, classes=rigid, overgenerated, event_ratio=1.324675, second_share=0.529412, rigid=0.980198
- `16_hatsuki_yura_guren_yasha_a_m_d_hard`: required_bias=7.450565, margin=-7.450565, rank=5.000000, classes=overgenerated, event_ratio=1.369863, second_share=0.540000, rigid=0.525253
- `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`: required_bias=7.450126, margin=-7.450126, rank=5.000000, classes=starved, rigid, undergenerated, event_ratio=0.527473, second_share=0.041667, rigid=0.936170
- `19_nekodex_circles_famoss_hard`: required_bias=7.357864, margin=-7.357864, rank=5.000000, classes=starved, rigid, undergenerated, event_ratio=0.533333, second_share=0.041667, rigid=0.936170
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: required_bias=7.244565, margin=-7.244565, rank=8.000000, classes=starved, rigid, undergenerated, event_ratio=0.516129, second_share=0.041667, rigid=0.936170
- `14_oomori_seiko_justadice_tv_size_remu_hard`: required_bias=7.129774, margin=-7.129774, rank=8.000000, classes=starved, rigid, undergenerated, event_ratio=0.551724, second_share=0.041667, rigid=0.936170
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: required_bias=7.010545, margin=-7.010545, rank=10.000000, classes=overgenerated, event_ratio=2.157895, second_share=0.406504, rigid=0.581967
- `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`: required_bias=6.923430, margin=-6.923430, rank=8.000000, classes=starved, event_ratio=1.040816, second_share=0.039216, rigid=0.760000
- `18_billiummoto_four_veiled_stars_aries_famoss_hard`: required_bias=6.822336, margin=-6.822336, rank=7.000000, classes=starved, undergenerated, event_ratio=0.662338, second_share=0.098039, rigid=0.880000

## Low-Bias Starved Cases

- `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv`: required_bias=1.048355, margin=-1.048355, rank=2.000000, event_ratio=0.417582, second_share=0.052632
- `23_usao_knight_rider_kuo_kyoka_expert`: required_bias=1.537447, margin=-1.537447, rank=4.000000, event_ratio=0.471154, second_share=0.061224
- `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`: required_bias=1.836269, margin=-1.836269, rank=4.000000, event_ratio=0.345588, second_share=0.021277

## Interpretation

Event tokens are often valid, but the median logit margin is too negative for a small global event bonus to be a safe next step. A global bonus large enough to move the hard starved cases would likely disturb already-passable or overgenerated cases. Cross-variant evidence strengthens this: lower median event-margin pressure did not reliably reduce starvation in the compared v3 variants.

## Next Step

Do not run another global decode-bonus sweep. Use selective per-step trace/oracle instrumentation on the few low-bias starved cases, or pivot to v2.1 grammar improvement if the owner wants to stop v3 local probes.
