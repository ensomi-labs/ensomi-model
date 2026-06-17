# Target Grammar v3 Low-Bias Trace Oracle Result Report

## Scope

This runtime-backed diagnostic reruns selected v3 500-step fixed-slice cases with expanded per-step logit examples. It does not change tokenizer, grammar, model weights, training, or runtime decode defaults.

## Result

Decision: `TEST_DENSITY_AWARE_TRACE`.

- Reason: only 1/5 low-bias starved cases have enough second-window opportunities
- Recommended next step: Either add density/control-aware trace fields or pivot to v2.1 grammar improvement.
- Low-bias threshold: `4.0`
- Positive low-bias cases: `1` / `5`
- Incomplete traces: `0`
- Illegal cases: `0`

## Case Table

| Role | Case | full trace | second-window opportunities | first-window opportunities | second steps | timepoints | dead end | max token |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| low_bias_starved | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | True | 39 | 1 | 97 | 91 | False | False |
| low_bias_starved | `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | True | 6 | 1 | 111 | 48 | False | False |
| low_bias_starved | `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | True | 6 | 1 | 111 | 48 | False | False |
| low_bias_starved | `27_billiummoto_four_veiled_stars_aries_insane` | True | 4 | 1 | 111 | 48 | False | False |
| low_bias_starved | `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | True | 0 | 1 | 81 | 46 | False | False |
| pass_like_control | `25_nekodex_circles_famoss_insane` | True | 1 | 1 | 163 | 100 | False | False |

## Second-Window Opportunity Examples

### `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- step=2, ms=8170, argmax=TS_100, best_event=EV_0120, rank=2.000000, margin=-0.872976
- step=4, ms=8340, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.082394
- step=6, ms=8510, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.585339
- step=8, ms=8680, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.131228
- step=10, ms=8850, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.218047
- step=12, ms=9020, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.449589
- step=14, ms=9190, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.173209
- step=16, ms=9360, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.611101
### `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`
- step=2, ms=8150, argmax=TS_100, best_event=EV_0010, rank=2.000000, margin=-0.652285
- step=4, ms=8300, argmax=TS_100, best_event=EV_0010, rank=2.000000, margin=-0.794367
- step=6, ms=8450, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.777560
- step=8, ms=8600, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.589817
- step=10, ms=8750, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.850156
- step=14, ms=9050, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.974525
### `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- step=2, ms=8150, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-0.650973
- step=4, ms=8300, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-0.729317
- step=6, ms=8450, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.717705
- step=8, ms=8600, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.503130
- step=10, ms=8750, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.817353
- step=14, ms=9050, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.964094
### `27_billiummoto_four_veiled_stars_aries_insane`
- step=2, ms=8150, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-0.843711
- step=4, ms=8300, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-0.973917
- step=6, ms=8450, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.920231
- step=8, ms=8600, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.731273
### `25_nekodex_circles_famoss_insane`
- step=0, ms=8000, argmax=TS_100, best_event=EV_0011, rank=5.000000, margin=-1.607853

## Interpretation

A simple selective gate is weakly supported at best. More context features are needed before changing decode behavior.

## Next Step

Either add density/control-aware trace fields or pivot to v2.1 grammar improvement.
