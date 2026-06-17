# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Rollout Gate Result Log

## Mode

- Mode: executor
- Experiment Card existed before execution: yes
- Route entering executor: `TEST`
- Stopped and returned to planner mode: no
- Source snapshot / evidence grade: medium local evidence from three high-risk real-audio rollout pairs.

## Experiment

- Experiment Card: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_experiment_card.md`
- Date: 2026-06-17
- Commit / run id: `d1de8fd` plus generated rollout artifacts
- Root objective: verify that the repaired time-shift distance auxiliary does not regress early generated-output behavior before full32 escalation.
- Goal decomposition: matched baseline/enabled rollouts, rollout-pair metric comparison, route decision.
- Candidate variants considered: 3-case high-risk rollout gate, 2-case smoke, full32 immediate, synthetic-only rollout.
- Selected variant: 3-case high-risk rollout gate.
- Selection pressure from card: legality, no new starvation, mean rigid ratio not worse by more than `0.02`, bounded runtime.
- Dataset slice: cases `31_camellia...equatorial`, `03_namirin...ash_s_normal`, `12_ohara...asha_s_hd`.
- Baseline / comparator: matched 80-step baseline checkpoint against matched 80-step enabled checkpoint.
- Runtime: six real-audio rollouts plus comparator completed locally.
- Files changed: audit card, rollout pair manifest, comparator summary/report, this result log.
- Read-only context files consulted: prior 32-case fixed-slice summary, rollout smoke evaluator, tiny-gate comparator.

## Result

- Primary metric value: route `TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE`.
- Secondary metric value: all three baseline and enabled rollouts were legal; no dead ends; no max-token exits.
- Baseline / comparator: baseline and enabled rollout metrics were identical on all three cases.
- Verify command / result: comparator completed with `route=TEST_NEXT_FULL32_TIME_SHIFT_DISTANCE`.
- Guard command / result: `uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py -q` -> `19 passed`.
- Qualitative observations: the enabled auxiliary did not visibly change greedy rollout behavior at this 80-step horizon.
- Positive signal observed: yes, stability gate passed.
- Negative signal observed: no regression, but no improvement signal either.
- Kill criteria triggered: no.

## Rollout Aggregate

- rollout pairs: `3`
- baseline all legal: `True`
- enabled all legal: `True`
- new starved cases: `0`
- mean dominant-spacing ratio delta: `0.0`
- mean event-count ratio delta: `0.0`
- mean F1@100ms delta: `0.0`
- mean second-window share delta: `0.0`

## Per-Case Outcome

| Case | Baseline legal | Enabled legal | Rigid delta | F1 delta | Event-count delta | Second-window delta | New starved |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `True` | `True` | `0.0` | `0.0` | `0.0` | `0.0` | `False` |
| `03_namirin_kanzen_shouriesper_girl_tailsdk_ash_s_normal` | `True` | `True` | `0.0` | `0.0` | `0.0` | `0.0` | `False` |
| `12_ohara_yuiko_zero_centimeters_tv_size_mikan_asha_s_hd` | `True` | `True` | `0.0` | `0.0` | `0.0` | `0.0` | `False` |

## Commands

```bash
uv run python - <<'PY' ... run six baseline/enabled real-audio rollout smokes and write rollout_pairs.json ... PY
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate --baseline-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/baseline/report.json --enabled-report artifacts/tmp/mapper_v3_time_shift_distance_tiny_training_gate/train/enabled/report.json --rollout-pairs-json artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_pairs.json --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_rollout_gate_result_report.md
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py -q
```

## Verification / Failure Modes

- Checks performed: six rollout smokes, rollout-pair comparator, rollout guard tests.
- Failed checks: none.
- Suspected confounders: 80-step checkpoints may be too close for the auxiliary to affect greedy decode; all rollout deltas being exactly zero weakens any improvement claim.
- Expected failure modes observed: none.
- Unexpected failure modes: none.
- Reproducibility notes: rollout summaries and pair manifest are written under the paths referenced by the comparator summary.
- Evidence gaps: no full32 run, no full 4k audit, and no demonstrated generated-quality improvement yet.

## Interpretation

- What the result supports: the repaired auxiliary does not regress this tiny high-risk rollout slice and can be escalated to a full32 500-step gate.
- What the result does not support: it does not show decode improvement, final v3 readiness, or full-pipeline replacement readiness.
- Alternative explanations: the tiny 80-step enabled checkpoint may be effectively behavior-identical to baseline under greedy decode.
- Positive / negative / ambiguous classification: positive for safety/escalation, neutral for quality.
- Recommended next step: `TEST` a full32 500-step repaired-loss gate before changing defaults.
- Human owner decision: pending.
