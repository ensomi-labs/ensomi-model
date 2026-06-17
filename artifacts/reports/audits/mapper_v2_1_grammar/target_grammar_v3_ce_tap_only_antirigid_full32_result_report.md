# Target Grammar v3 CE Anti-Rigid Full-32 Decode Gate Result Report

## Scope

This pass reruns all 32 committed CE-weight v3 fixed-slice cases with an opt-in v3 anti-rigid logits transform. It does not change mapper defaults, model weights, tokenization, or training.

## Decision

- route: `KILL`
- reason: full-32 hard-block anti-rigid decode failed an original fixed-slice gate
- transform: `hard-block`

## Aggregate

| Metric | CE baseline full-32 | Anti-rigid candidate | Delta |
| --- | ---: | ---: | ---: |
| rigid cases | `11` | `1` | `-10` |
| starved cases | `5` | `5` | `0` |
| mean F1@100ms | `0.691899` | `0.670564` | `-0.021335` |
| mean dominant spacing | `0.745180` | `0.573705` | `-0.171475` |
| mean event ratio | `1.059223` | `1.223779` | `n/a` |
| median event ratio | `1.156108` | `1.393527` | `n/a` |
| max boundary ratio | `0.041667` | `0.051724` | `n/a` |

## Gate Results

- case_coverage: `true`
- all_legal: `false`
- no_dead_end_cases: `false`
- no_max_token_cases: `true`
- starved_below_original_baseline: `true`
- rigid_no_worse_than_original_baseline: `true`
- mean_f1_within_original_floor: `true`
- median_event_count_ratio_in_range: `false`
- max_boundary_no_worse_than_original_baseline: `true`
- ce_rigid_reduction_at_least_four: `true`
- ce_starvation_not_materially_worse: `true`
- ce_mean_f1_not_materially_worse: `true`

## Transform

- total candidates: `470`
- total blocks: `470`
- dead-end cases: `1`
- max-token cases: `0`

## Worst Cases

Highest candidate rigid ratios:
- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: rigid `0.977778` -> `0.977778`, f1 `0.611940` -> `0.611940`, blocks `0`
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: rigid `0.936170` -> `0.936170`, f1 `0.581560` -> `0.581560`, blocks `0`
- `27_billiummoto_four_veiled_stars_aries_insane`: rigid `0.936170` -> `0.936170`, f1 `0.439716` -> `0.439716`, blocks `0`
- `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`: rigid `0.989474` -> `0.837500`, f1 `0.859701` -> `0.533333`, blocks `11`
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: rigid `0.980198` -> `0.800000`, f1 `0.860335` -> `0.819149`, blocks `21`

## Next Step

Do not scale this hard-block policy; inspect full-32 failures or mutate decode/training strategy.
