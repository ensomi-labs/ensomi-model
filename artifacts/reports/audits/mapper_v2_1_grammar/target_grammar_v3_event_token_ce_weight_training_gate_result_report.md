# Target Grammar v3 Event-Token CE Weight Training Gate Result Report

## Scope

This executor pass tested the existing Event-Token CE Weight Training Gate card. It changes no source code and does not alter v3 grammar, decode policy, tokenizer, data slice, or runtime settings. The only training-side intervention is `loss.event_token_loss_weight=2.0`.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_enabled.yaml
# then run run_trained_v3_runtime_rollout_smoke over all 32 fixed manifest rows
```

## Training Result

- completed steps: `500` / `500`
- final eval total loss: `2.262730`
- final eval token loss: `2.176583`
- event token CE weight: `2.0`
- checkpoint: `artifacts/tmp/mapper_v3_event_token_ce_weight_training_gate/train/run_weight2/checkpoint.pt`

## Rollout Gate

| Metric | Baseline | CE weight 2.0 | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| legal cases | `32` | `32` | `0` | all 32 legal |
| dead ends | `0` | `0` | `0` | must be 0 |
| max-token cases | `0` | `0` | `0` | must be 0 |
| starved cases | `11` | `5` | `-6` | below 11 |
| rigid cases | `7` | `11` | `4` | no worse than 7 |
| median event-count ratio | `1.000000` | `1.156108` | `0.156108` | 0.80-1.25 |
| mean F1@100ms | `0.639566` | `0.691899` | `0.052333` | no more than 0.03 below baseline |
| mean second-window share | `0.373612` | `0.437670` | `0.064058` | diagnostic |
| max boundary-event ratio | `0.200000` | `0.041667` | `-0.158333` | no worse than 0.200 |

## What Passed

- All 32 real-audio greedy rollouts completed legally.
- No case hit a dead end or max-token cap.
- Starvation improved from `11` cases to `5` cases.
- Mean F1@100ms improved from `0.639566` to `0.691899`.
- Median event-count ratio stayed in range at `1.156108`.
- Boundary duplication improved: max boundary-event ratio dropped from `0.200000` to `0.041667`.

## What Surfaced

- Rigid cases worsened from `7` to `11`, failing the primary gate.
- CE weighting increased event production and continuation, but part of that gain falls into repeated-spacing grids.
- Two cases became newly starved while eight baseline-starved cases were fixed.
- CE weighting is useful signal, but not a standalone v3 replacement/scaling recipe.

## Decision

Decision: `MUTATE`.

event CE weighting reduced starvation but worsened rigid-grid cases beyond the baseline gate.

The result should be interpreted as a partial positive: event-token CE weighting helps continuation and F1, but it shifts enough mass into rigid-grid behavior that the next loop should mutate the objective or grammar around timing diversity/anti-rigid continuation before any broader v3 training run.

## Worst Cases

Lowest second-window share:
- `15_phantom_sage_holystone_unluckycroco_at_the_point_of_no_return`: second=`0.000000`, ratio=`0.522727`, f1=`0.611940`, rigid=`0.977778`
- `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`: second=`0.010989`, ratio=`1.022472`, f1=`0.266667`, rigid=`0.977778`
- `30_goreshit_one_way_to_hannover_cokiiplay_autophobia`: second=`0.023256`, ratio=`0.000000`, f1=`0.000000`, rigid=`0.880952`
- `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd`: second=`0.041667`, ratio=`0.842105`, f1=`0.514286`, rigid=`0.936170`
- `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`: second=`0.041667`, ratio=`0.516129`, f1=`0.581560`, rigid=`0.936170`

Highest rigid ratio:
- `11_hibiki_sakura_cv_fairouz_ai_naruzo_machio_cv_ishikawa_kaito_onegai_muscle_tv_siz`: rigid=`1.000000`, ratio=`0.382353`, f1=`0.524823`, second=`0.512821`
- `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`: rigid=`0.989474`, ratio=`1.326389`, f1=`0.859701`, second=`0.523560`
- `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal`: rigid=`0.980198`, ratio=`1.324675`, f1=`0.860335`, second=`0.529412`
- `09_namirin_kanzen_shouriesper_girl_tailsdk_hard`: rigid=`0.980198`, ratio=`1.324675`, f1=`0.860335`, second=`0.529412`
- `20_namirin_kanzen_shouriesper_girl_tailsdk_insane`: rigid=`0.980198`, ratio=`1.291139`, f1=`0.850829`, second=`0.529412`

## Artifacts

- summary: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_summary.json`
- report: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_result_report.md`
- config: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_enabled.yaml`
- card: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_event_token_ce_weight_training_gate_experiment_card.md`
- rollout summaries: `artifacts/tmp/mapper_v3_event_token_ce_weight_training_gate/rollouts`

## Next Step

Create a bounded mutation card that keeps the CE continuation benefit but directly targets the rigid-spacing attractor, or pivot back to v2.1 grammar improvement if v3 timing diversity remains resistant.
