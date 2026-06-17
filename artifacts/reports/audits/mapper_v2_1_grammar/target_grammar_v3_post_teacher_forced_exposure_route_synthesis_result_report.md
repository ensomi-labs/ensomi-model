# Target Grammar v3 Post-Teacher-Forced Exposure Route Synthesis Result Report

## Scope

This artifact-only synthesis reconciles the teacher-forced v3 time-shift logit audit with existing generated-prefix rollout, decode-policy, event-margin, conditioned-objective, time-shift, and C3 route artifacts. It does not train, rerun rollout, change tokenizer behavior, or change mapper defaults.

## Decision

Route: `TEST_GENERATED_PREFIX_STATE_TRACE`.

- Reason: teacher-forced timing ranks are healthy, generated-prefix failures persist, and decode/objective shortcuts are already killed or gated
- Next card: `target_grammar_v3_generated_prefix_state_trace_audit`
- Next step: Create a generated-prefix state trace audit on high-leverage fixed-slice cases.

## Route Checks

| Check | Value |
| --- | ---: |
| `required_present` | `True` |
| `teacher_forced_healthy` | `True` |
| `teacher_route` | `TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC` |
| `teacher_rows` | `3886` |
| `teacher_recall_at_5` | `0.9233144621718992` |
| `teacher_median_rank` | `1.0` |
| `teacher_argmax_top_shift_share` | `0.31960885229027275` |
| `generated_prefix_failures_persist` | `True` |
| `generated_event_tokens_broadly_valid` | `True` |
| `continuation_baseline_starved` | `11` |
| `continuation_baseline_rigid` | `12` |
| `continuation_baseline_boundary_clamped` | `10` |
| `continuation_event_valid_ratio` | `0.9958870412089413` |
| `decode_only_killed` | `True` |
| `decode_policy_pass_count` | `0` |
| `event_margin_global_bonus_killed` | `True` |
| `event_margin_starved_gt4_share` | `0.7272727272727273` |
| `conditioned_objective_killed` | `True` |
| `conditioned_starved_case_delta` | `-2` |
| `conditioned_mean_f1_delta` | `0.050044943598297253` |
| `conditioned_event_ratio_guard_passed` | `False` |
| `time_shift_objective_failed` | `True` |
| `c3_mapper_diminishing` | `True` |
| `shortcuts_exhausted` | `True` |
| `exposure_trace_justified` | `True` |

## Evidence Table

| Key | Family | Route | Exists | Reason |
| --- | --- | --- | ---: | --- |
| teacher_forced_time_shift | teacher_forced | `TEST_DECODE_STATE_EXPOSURE_DIAGNOSTIC` | `True` | teacher-forced time-shift ranks are good enough that free-running state exposure is the likely bottleneck |
| generated_continuation | generated_prefix | `TEST_DECODE_TIMING_CALIBRATION` | `True` | boundary/starvation persists while event tokens remain broadly valid |
| decode_policy_sweep | decode_policy | `KILL` | `True` | no tested decode policy reduced rigid-grid collapse while preserving legality and continuation guards |
| event_margin_viability | decode_policy | `KILL_GLOBAL_EVENT_BONUS` | `True` | 8/11 starved cases need median event bias > 4.0 |
| conditioned_short_rollout | training_objective | `KILL` | `True` | conditioned objective tripped a hard safety or event-count calibration guard: median_event_count_ratio_in_range |
| post_time_shift_full32 | training_objective | `MUTATE_TIME_SHIFT_DISTANCE_AND_TEST_C3_TARGET_SIDE_STREAM` | `True` | row-filtered time-shift distance trained stably but failed full32 rollout checks, while C3 remains the strongest reconstructive tokenizer-side representation result |
| c3_diminishing_returns | c3_route | `MUTATE_C3_MAPPER_PATH_TEST_V3_TIMING_LOGIT_AUDIT` | `True` | C3 mapper-side follow-ups are now mostly MUTATE while v3 timing-collapse diagnostics point to a smaller timing-logit calibration question. |

## What Passed

- The teacher-forced audit is faithful, non-smoke, and ranks target time shifts strongly on the fixed eval split.
- The generated-prefix diagnostics still show continuation/starvation/rigid-grid failures under rollout.
- Existing deterministic decode-only, global event-bonus, conditioned event-distribution, and time-shift-distance shortcuts are not promotable as tested.

## What Surfaced

- The remaining failure is exposed under generated prefixes, not under teacher-forced time-shift logits.
- Another scalar teacher-forced timing loss is not the smallest next test unless a new mechanism is tied to generated-prefix guards.
- C3 remains a proven codec-side representation result, but current mapper-side C3 routes have diminishing returns.

## What This Does Not Prove

- It does not prove v3 replacement readiness.
- It does not prove the generated-prefix trace will fix rollout quality.
- It does not prove v2.1 grammar work is unnecessary.
- It does not evaluate the full 4k dataset.

## Interpretation

The next useful v3 experiment is a narrow per-step generated-prefix state trace. It should identify whether collapse starts from time-shift choice, event underselection, carry/state mismatch, or boundary state drift before another training run.
