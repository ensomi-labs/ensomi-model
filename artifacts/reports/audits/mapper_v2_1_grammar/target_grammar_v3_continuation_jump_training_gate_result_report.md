# Target Grammar v3 Continuation-Jump Training Gate Result Report

## Scope

This pass executes the bounded follow-up from `target_grammar_v3_terminal_jump_guard_experiment_card.md` after the local continuation-jump loss gate passed. It trains a fixed-slice mapper v3 run with `lambda_continuation_jump=0.05`, then evaluates the checkpoint in the same 32-case real-audio free-running rollout gate used by the recent v3 audits.

This is not a tokenizer or grammar replacement. It does not change v3 tokenization, replay, grammar, or rollout semantics. The question is narrower: does a teacher-forced skip-over penalty transfer into better free-running second-window continuation without reintroducing loops or duplicate boundary events?

## Setup

- Training config: `artifacts/tmp/mapper_v3_continuation_jump_guard/train_config_lambda005.yaml`
- Training output: `artifacts/tmp/mapper_v3_continuation_jump_guard/train/run_lambda005`
- Checkpoint: `artifacts/tmp/mapper_v3_continuation_jump_guard/train/run_lambda005/checkpoint.pt`
- Rollout summary: `artifacts/tmp/mapper_v3_continuation_jump_guard/continuation_jump_guard_rollout_summary.json`
- Slice: fixed 32-song / 256-window mapper v3 slice.
- Training horizon: 500 steps.
- Loss settings: `lambda_continuation_jump=0.05`, `continuation_jump_tolerance_ms=100`, `lambda_density=0.05`, `lambda_event_budget=0.0`.
- Rollout settings: real audio, 32 cases, `max_tokens_per_window=512`, `temperature=0.0`, `top_p=null`.

## Training Result

Training completed all 500 steps and the continuation-jump loss was active and finite.

| Split | token loss | total loss | continuation-jump | density | LN close | event budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final eval | 1.905414 | 1.991792 | 0.030944 | 1.616546 | 0.110988 | 0.000000 |
| final train | 1.563098 | 1.649234 | 0.044693 | 1.647005 | 0.075708 | 0.000000 |

Local interpretation: the auxiliary objective is trainable and did not introduce NaNs or an obvious optimization failure on this slice. This only proves the loss can be optimized under teacher forcing; it does not prove generated-state continuation.

## Rollout Gate

Decision: `MUTATE`.

| Run | legal | max-token cases | dead ends | starved cases | rigid cases | median event ratio | median second-window share | mean F1@100ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500-step wide baseline | yes | n/a | n/a | 11 | 7 | 1.000000 | 0.490902 | 0.639566 |
| same-ms event guard | yes | 0 | 0 | 22 | 9 | 0.585366 | 0.047619 | 0.556877 |
| continuation-jump lambda005 | no | 0 | 3 | 19 | 24 | 0.631579 | 0.041667 | 0.569581 |

Passed checks:

- No max-token cases.
- No duplicate or non-increasing spacing regression: max ratio `0.0`.
- Boundary event ratio stayed low: max ratio `0.047619`.

Failed checks:

- `all_32_rollouts_legal`: failed, with 3 dead-end cases.
- `starved_cases_below_baseline_11`: failed, with 19 starved cases.
- `rigid_cases_no_worse_than_baseline_7`: failed, with 24 rigid cases.
- `median_event_ratio_in_0_80_1_25`: failed, with median event ratio `0.631579`.
- `mean_f1_no_more_than_0_03_below_baseline`: failed, with mean F1 `0.569581` versus baseline `0.639566`.

## Failure Cases

The three illegal cases all dead-ended at the first-window boundary pattern:

| Case | generated / reference events | second-window share | token count | last generated times |
| --- | ---: | ---: | ---: | --- |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | 46 / 77 | 0.000 | 138 | `[7220, 7380, 7540, 7700, 7860, 7990]` |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 46 / 93 | 0.000 | 138 | `[7220, 7380, 7540, 7700, 7860, 7990]` |
| `23_usao_knight_rider_kuo_kyoka_expert` | 46 / 104 | 0.000 | 138 | `[7220, 7380, 7540, 7700, 7860, 7990]` |

The broader starved cluster repeats two signatures:

- Early terminal clamp near `7990ms`, with no second-window events.
- Minimal second-window spill via `[... 7980, 7990, 15950, 15990]`, which technically reaches window two but does not continue the chart density.

Examples:

| Case | generated / reference events | second-window share | reference second-window share | dominant spacing | F1@100ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_hatsuki_yura_guren_yasha_a_m_d_normal` | 22 / 63 | 0.000 | 0.730 | 400ms / 0.619 | 0.118 |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | 46 / 77 | 0.000 | 0.623 | 160ms / 0.978 | 0.488 |
| `21_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_insane` | 46 / 86 | 0.000 | 0.465 | 160ms / 0.978 | 0.682 |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | 46 / 93 | 0.000 | 0.548 | 160ms / 0.978 | 0.561 |
| `23_usao_knight_rider_kuo_kyoka_expert` | 46 / 104 | 0.000 | 0.635 | 160ms / 0.978 | 0.467 |

## What Passed

- The 500-step training gate completed with finite losses.
- The auxiliary continuation-jump term was active in the train/eval metrics.
- The same-ms / duplicate wall stayed fixed: max-token cases remained `0`, and duplicate or non-increasing spacing ratio stayed `0.0`.
- The result slightly improved starvation and mean F1 relative to the same-ms-guard-only gate: starved `22 -> 19`, mean F1 `0.556877 -> 0.569581`.

## What Surfaced

- The mutation does not transfer strongly enough from teacher-forced supervision to free-running generated-state rollout.
- It worsens rigidity sharply: rigid cases `9 -> 24` versus the same-ms guard and `7 -> 24` versus the 500-step baseline.
- It introduces three illegal dead ends, all with the same first-window clamp near `7990ms`.
- It remains substantially worse than the 500-step wide baseline on starvation, median event ratio, second-window share, and mean F1.

## Interpretation

The continuation-jump loss is a valid local loss component, but `lambda_continuation_jump=0.05` is not a successful rollout mutation. The likely issue is that teacher-forced skip-over pressure does not correct the model's generated-state continuation policy once it has already entered a rigid local timing attractor. The generated sequence can avoid duplicate same-ms loops while still collapsing into a first-window terminal pattern or sparse second-window spill.

This result weakens the case for scaling scalar auxiliary losses alone. The next loop should not scale this variant. The better mutation target is either planner-side event-budget / completion semantics, a grammar-level continuation target, or the active-goal fallback path: improve v2.1 grammar rather than keep expanding v3 with local losses.

## Decision

Route: `MUTATE`.

Do not scale `lambda_continuation_jump=0.05`. Analyze the failure clusters only if it directly informs the next bounded card; otherwise pivot to a smaller target that changes the continuation signal available during free-running generation.
