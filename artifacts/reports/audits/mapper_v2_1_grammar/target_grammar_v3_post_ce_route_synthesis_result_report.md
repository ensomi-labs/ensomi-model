# Target Grammar v3 Post-CE Route Synthesis Result Report

## Scope

This artifact-only synthesis reads committed v3/C3 audit summaries after the CE, trace, density, and anti-rigid branches. It does not retrain, rerun rollout, alter tokenizer behavior, or change defaults.

## Decision

Route: `TEST_CONDITIONED_EVENT_DISTRIBUTION_OBJECTIVE`.

- Reason: representation and pipeline mechanics passed, simple decode branches are killed, and global scalar event-distribution objectives did not solve the rollout gates
- Recommended next step: Create a conditioned event-distribution objective plumbing card with high-difficulty overgeneration guards.
- Next card: `target_grammar_v3_conditioned_event_distribution_objective`

## Route Checks

| Check | Value |
| --- | ---: |
| `required_present` | `True` |
| `representation_ready` | `True` |
| `pipeline_mechanics_ready` | `True` |
| `real_audio_quality_ready` | `False` |
| `c3_as_target_killed` | `True` |
| `global_event_budget_failed` | `True` |
| `ce_weight_partial_not_promoted` | `True` |
| `selective_trace_narrow` | `True` |
| `antirigid_decode_killed` | `True` |
| `simple_decode_family_killed` | `True` |
| `scalar_objective_not_solved` | `True` |
| `killed_decode_family_count` | `4` |
| `killed_decode_keys` | `['ce_density_trace', 'ce_antirigid_full32', 'ce_antirigid_soft_full32', 'ce_tap_only_antirigid_full32']` |
| `event_budget_route` | `MUTATE` |
| `ce_weight_route` | `MUTATE` |
| `tap_only_route` | `KILL` |
| `tap_only_dead_end_count` | `1` |
| `tap_only_median_event_ratio` | `1.3935272045028142` |
| `broad_antirigid_median_event_ratio` | `1.4123306233062332` |
| `event_budget_max_token_count` | `4` |
| `ce_density_positive_cases` | `1` |
| `c3_route` | `MUTATE_TO_V3_GRAMMAR_REPAIR` |

## Evidence Table

| Key | Family | Route | Status | Reason |
| --- | --- | --- | --- | --- |
| `v3_full_dataset` | representation | `TEST_NEXT` | present |  |
| `v3_real_config` | pipeline | `TEST_NEXT` | present | all real-config cache-backed comparison gates passed |
| `v3_real_audio` | pipeline | `TEST_NEXT` | present | 20-step real-config v3 training and real-audio session rollout gates passed |
| `density_loss` | training_objective | `TEST_NEXT` | present | Density loss was active but eval density did not improve while token loss improved; next mutation needs stronger event-distribution calibration with rollout gates. |
| `event_budget_0_05` | training_objective | `MUTATE` | present | failed checks: all_32_rollouts_legal, no_max_token_cases, starved_cases_below_baseline_11, rigid_cases_no_worse_than_baseline_7, median_event_ratio_in_0_80_1_25, max_boundary_ratio_no_worse_than_baseline_0_200 |
| `ce_weight` | training_objective | `MUTATE` | present | event CE weighting reduced starvation but worsened rigid-grid cases beyond the baseline gate |
| `ce_residual_cluster` | trace_oracle | `TEST_EVENT_RANKING_CALIBRATION` | present | event candidates are often rank-near but underselected |
| `ce_margin` | trace_oracle | `TEST_SELECTIVE_TRACE_ORACLE` | present | 5/5 starved cases are low-bias, but the class is mixed |
| `ce_selective_trace` | decode_selector | `TEST_DENSITY_AWARE_TRACE` | present | only 1/5 low-bias starved cases have enough second-window opportunities |
| `ce_density_trace` | decode_selector | `MUTATE_TO_TARGET_GRAMMAR_OR_TRAINING_REPAIR` | present | only 1/5 CE-starved cases have sparse-context opportunity coverage |
| `ce_antirigid_full32` | decode_policy | `KILL` | present | full-32 hard-block anti-rigid decode failed an original fixed-slice gate |
| `ce_antirigid_soft_full32` | decode_policy | `KILL` | present | full-32 soft-penalty-4 anti-rigid decode failed an original fixed-slice gate |
| `ce_tap_only_antirigid_full32` | decode_policy | `KILL` | present | full-32 hard-block anti-rigid decode failed an original fixed-slice gate |
| `c3_v3_complexity` | c3_route | `MUTATE_TO_V3_GRAMMAR_REPAIR` | present | C3 remains reconstructive but side-stream/state costs are not target-grammar competitive with v3 |

## What This Proves

- v3 representation and pipeline mechanics remain the active replacement route, not C3-as-target.
- Simple local decode branches are not the next best step after CE: selector evidence is too narrow, and anti-rigid variants fail full32 gates.
- Global scalar event-distribution pressure is not sufficient as tested.

## What Remains Unproven

- v3 trained chart quality is not replacement-ready.
- Full-dataset/full-song v3 runtime quality remains unproven.
- A conditioned event-distribution objective has not yet been implemented or trained.

## Next Step

Create a conditioned event-distribution objective plumbing card with high-difficulty overgeneration guards.
