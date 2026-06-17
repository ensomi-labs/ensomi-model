# Target Grammar v3 CE Density-Aware Trace Diagnostic Result Report

## Scope

This artifact-only diagnostic reuses the completed CE selective trace-oracle summaries. It does not retrain, change grammar, change tokenizer, or change decode behavior.

## Result

Decision: `MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR`.

- Reason: only 1/5 CE-starved cases have sparse-context opportunity coverage
- Recommended next step: Pivot to target-grammar/training repair, or add richer control/audio trace fields only if they define a concrete guard.
- Positive CE-starved cases: `1` / `5`
- Control conflicts: `0`
- Incomplete traces: `0`
- Illegal traces: `0`
- Sparse density threshold: `<= 1` generated events in previous `1000` ms

## Case Table

| Role | Case | sparse opp | total opp | sparse steps | second steps | event argmax share | second generated events | longest sparse run | positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| low_bias_starved | `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 34 | 39 | 86 | 97 | 0.010309 | 1 | 1 | True |
| low_bias_starved | `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | 1 | 6 | 98 | 111 | 0.018018 | 2 | 1 | False |
| low_bias_starved | `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 1 | 6 | 98 | 111 | 0.018018 | 2 | 1 | False |
| low_bias_starved | `27_billiummoto_four_veiled_stars_aries_insane` | 0 | 4 | 98 | 111 | 0.018018 | 2 | 0 | False |
| low_bias_starved | `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return` | 0 | 0 | 72 | 81 | 0.000000 | 0 | 0 | False |
| pass_like_control | `25_nekodex_circles_famoss_insane` | 0 | 1 | 0 | 163 | 0.331288 | 54 | 0 | False |

## Sparse Opportunity Examples

### `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- step=12, ms=9020, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.449589
- step=14, ms=9190, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.173209
- step=16, ms=9360, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.611101
- step=18, ms=9530, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.479430
- step=20, ms=9700, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.554048
- step=22, ms=9870, density=0, argmax=TS_100, best_event=EV_1000, rank=2.000000, margin=-0.369131
### `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`
- step=14, ms=9050, density=0, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.974525
### `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- step=14, ms=9050, density=0, argmax=TS_100, best_event=EV_2000, rank=2.000000, margin=-1.964094

## Interpretation

Density context explains a narrow subset, but not enough of the remaining CE-starved failures to justify another simple selector.

## Next Step

Pivot to target-grammar/training repair, or add richer control/audio trace fields only if they define a concrete guard.
