# Target Grammar v3 Low-Bias Trace Oracle Result Report

## Scope

This runtime-backed diagnostic reruns selected v3 500-step fixed-slice cases with expanded per-step logit examples. It does not change tokenizer, grammar, model weights, training, or runtime decode defaults.

## Result

Decision: `TEST_SELECTIVE_EVENT_GATE`.

- Reason: 3/3 low-bias starved cases have second-window opportunities
- Recommended next step: Create and run a selective event-gate smoke on these traced cases.
- Positive low-bias cases: `3` / `3`
- Incomplete traces: `0`
- Illegal cases: `0`

## Case Table

| Role | Case | full trace | second-window opportunities | first-window opportunities | second steps | timepoints | dead end | max token |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| low_bias_starved | `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv` | True | 34 | 35 | 74 | 38 | False | False |
| low_bias_starved | `23_usao_knight_rider_kuo_kyoka_expert` | True | 98 | 2 | 106 | 49 | False | False |
| low_bias_starved | `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer` | True | 79 | 1 | 101 | 47 | False | False |
| high_bias_starved_control | `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | True | 10 | 1 | 111 | 48 | False | False |
| pass_like_control | `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz` | True | 1 | 65 | 163 | 121 | False | False |

## Second-Window Opportunity Examples

### `10_graham_kartna_chiltonwalk_temp_nivrad00_nsv`
- step=0, ms=8000, argmax=TS_200, best_event=EV_1001, rank=5.000000, margin=-1.198876
- step=2, ms=8230, argmax=TS_200, best_event=EV_0002, rank=2.000000, margin=-0.347545
- step=4, ms=8460, argmax=TS_200, best_event=EV_0001, rank=2.000000, margin=-0.126377
- step=6, ms=8690, argmax=TS_200, best_event=EV_0001, rank=3.000000, margin=-0.604193
- step=8, ms=8920, argmax=TS_200, best_event=EV_0001, rank=2.000000, margin=-0.419578
- step=10, ms=9150, argmax=TS_200, best_event=EV_0001, rank=2.000000, margin=-0.448841
- step=12, ms=9380, argmax=TS_200, best_event=EV_0001, rank=3.000000, margin=-0.609254
- step=14, ms=9610, argmax=TS_200, best_event=EV_0001, rank=2.000000, margin=-0.480555
### `23_usao_knight_rider_kuo_kyoka_expert`
- step=1, ms=8070, argmax=TS_80, best_event=EV_0200, rank=3.000000, margin=-1.375271
- step=2, ms=8150, argmax=TS_70, best_event=EV_0011, rank=3.000000, margin=-1.162113
- step=3, ms=8220, argmax=TS_80, best_event=EV_0120, rank=3.000000, margin=-1.993159
- step=4, ms=8300, argmax=TS_70, best_event=EV_0011, rank=3.000000, margin=-1.046214
- step=6, ms=8450, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.443474
- step=7, ms=8530, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.237395
- step=8, ms=8610, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.305162
- step=9, ms=8690, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.501888
### `26_billx_punishment_dustvoxx_remix_underjoy_4k_equalizer`
- step=1, ms=8070, argmax=TS_80, best_event=EV_0200, rank=4.000000, margin=-1.651693
- step=2, ms=8150, argmax=TS_70, best_event=EV_0011, rank=3.000000, margin=-1.214706
- step=4, ms=8300, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.167157
- step=5, ms=8380, argmax=TS_80, best_event=EV_0011, rank=3.000000, margin=-1.433125
- step=6, ms=8460, argmax=TS_80, best_event=EV_0011, rank=4.000000, margin=-1.774508
- step=7, ms=8540, argmax=TS_80, best_event=EV_0011, rank=4.000000, margin=-1.579048
- step=8, ms=8620, argmax=TS_80, best_event=EV_0011, rank=4.000000, margin=-1.650302
- step=9, ms=8700, argmax=TS_80, best_event=EV_0011, rank=4.000000, margin=-1.844494
### `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- step=2, ms=8150, argmax=TS_100, best_event=EV_0010, rank=2.000000, margin=-0.053910
- step=4, ms=8300, argmax=TS_100, best_event=EV_0010, rank=2.000000, margin=-0.234151
- step=6, ms=8450, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.223799
- step=8, ms=8600, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.160937
- step=10, ms=8750, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.413346
- step=12, ms=8900, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.749645
- step=14, ms=9050, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.692441
- step=16, ms=9200, argmax=TS_100, best_event=EV_1100, rank=2.000000, margin=-1.944618
### `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz`
- step=0, ms=8000, argmax=TS_100, best_event=EV_1001, rank=5.000000, margin=-1.145198

## Interpretation

The low-bias exception is real enough to justify one bounded selective event-gate smoke.

## Next Step

Create and run a selective event-gate smoke on these traced cases.
