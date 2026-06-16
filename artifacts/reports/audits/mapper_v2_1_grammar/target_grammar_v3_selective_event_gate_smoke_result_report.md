# Target Grammar v3 Selective Event-Gate Smoke Result Report

## Scope

This runtime-backed smoke tests an eval-only selective event gate on the five cases selected by the low-bias trace oracle. It does not change tokenizer, grammar, model weights, training, or runtime defaults.

## Result

Decision: `KILL_SELECTIVE_EVENT_GATE`.

- Reason: 1 pass-like controls regressed
- Recommended next step: Pivot to v2.1 grammar improvement or training-side v3 instrumentation.
- Positive low-bias cases: `2` / `3`
- Illegal cases: `0`
- Overproduced cases: `2`
- Pass-like regressed cases: `1`

## Case Table

| Role | Case | legal | forced | 2nd share base->gate | event ratio base->gate | F1 base->gate | duplicate | boundary | positive |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| low_bias_starved | `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | True | 32 | 0.052632->0.544304 | 0.417582->0.868132 | 0.511628->0.788235 | 0.000000 | 0.037975 | True |
| low_bias_starved | `23_usao_knight_rider_kuo_kyoka_expert` | True | 1 | 0.061224->0.684932 | 0.471154->1.403846 | 0.470588->0.800000 | 0.000000 | 0.013699 | False |
| low_bias_starved | `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | True | 1 | 0.021277->0.684932 | 0.345588->1.073529 | 0.426230->0.900709 | 0.000000 | 0.013699 | True |
| high_bias_starved_control | `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | True | 1 | 0.041667->0.540000 | 0.527473->1.098901 | 0.604317->0.900524 | 0.000000 | 0.020000 | False |
| pass_like_control | `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | True | 5 | 0.446281->0.547297 | 1.186275->1.450980 | 0.825112->0.768000 | 0.000000 | 0.013514 | False |

## Gate Examples

### `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv`
- step=0, ms=8000, argmax=TS_200, forced=EV_1001, rank=5.000000, margin=-1.198876
- step=2, ms=8200, argmax=TS_10, forced=EV_0010, rank=3.000000, margin=-1.308145
- step=18, ms=9800, argmax=TS_10, forced=EV_1001, rank=2.000000, margin=-0.010139
- step=20, ms=10000, argmax=TS_10, forced=EV_1001, rank=2.000000, margin=-0.056963
- step=22, ms=10200, argmax=TS_10, forced=EV_1001, rank=2.000000, margin=-0.067439
- step=24, ms=10400, argmax=TS_10, forced=EV_0100, rank=2.000000, margin=-0.118169
- step=26, ms=10600, argmax=TS_10, forced=EV_1001, rank=2.000000, margin=-0.085105
- step=28, ms=10800, argmax=TS_10, forced=EV_1001, rank=2.000000, margin=-0.088852
### `23_usao_knight_rider_kuo_kyoka_expert`
- step=1, ms=8070, argmax=TS_80, forced=EV_0200, rank=3.000000, margin=-1.375271
### `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`
- step=1, ms=8070, argmax=TS_80, forced=EV_0200, rank=4.000000, margin=-1.651693
### `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- step=2, ms=8150, argmax=TS_100, forced=EV_0010, rank=2.000000, margin=-0.053910
### `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz`
- step=0, ms=8000, argmax=TS_100, forced=EV_1001, rank=5.000000, margin=-1.145198
- step=5, ms=8210, argmax=TS_10, forced=EV_0100, rank=3.000000, margin=-1.804019
- step=7, ms=8310, argmax=TS_10, forced=EV_0010, rank=3.000000, margin=-0.882368
- step=9, ms=8410, argmax=TS_10, forced=EV_0010, rank=2.000000, margin=-0.538710
- step=11, ms=8510, argmax=TS_10, forced=EV_0100, rank=2.000000, margin=-0.087780

## Interpretation

The selective gate affects non-target cases too much.

## Next Step

Pivot to v2.1 grammar improvement or training-side v3 instrumentation.
