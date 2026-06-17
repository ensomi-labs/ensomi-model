# Target Grammar v3 Time-Shift Distance Row-Filtered Repair Full32 Gate Result Log

## Mode

- Mode: executor
- Experiment Card existed before execution: yes
- Route entering executor: `TEST`
- Stopped and returned to planner mode: no
- Source snapshot / evidence grade: strong local full32 evidence on the fixed 32-case real-audio slice.

## Experiment

- Experiment Card: `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_experiment_card.md`
- Date: 2026-06-17
- Commit / run id: `b616b29` plus generated full32 artifacts
- Root objective: test whether the repaired time-shift distance auxiliary remains useful beyond the tiny gate before changing defaults.
- Goal decomposition: matched 500-step baseline/enabled training, full32 rollout pairs, training/rollout comparator, route decision.
- Candidate variants considered: matched full32 gate, enabled-only comparison to older 500-step checkpoint, lambda/scale mutation first, direct full 4k training.
- Selected variant: matched 500-step baseline/enabled full32 gate.
- Selection pressure from card: finite enabled training, legal full32 rollouts, no new starved cases, mean rigid ratio not materially worse.
- Dataset slice: fixed 32-song / 256-window training slice and 32 fixed-slice real-audio rollout cases.
- Baseline / comparator: `lambda_time_shift_distance=0.0`, `event_token_loss_weight=2.0`.
- Enabled: `lambda_time_shift_distance=0.5`, `event_token_loss_weight=2.0`.
- Files changed: result summary/report, rollout-pair manifest, this result log.

## Result

- Primary metric value: route `MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE`.
- Reason: failed rollout checks `no_new_starved_cases` and `mean_rigid_not_worse`.
- Positive signal observed: training stability and finite auxiliary signal.
- Negative signal observed: full32 generated-output regression.
- Kill criteria triggered: not a hard kill for the repaired loss implementation, but the current objective/lambda should not be scaled or set as default.

## Training Evidence

- baseline completed steps: `500`
- enabled completed steps: `500`
- baseline final eval total loss: `2.2627727390410297`
- enabled final eval total loss: `2.258821244279302`
- baseline final eval token loss: `2.176626054738027`
- enabled final eval token loss: `2.172651960832534`
- enabled `loss/time_shift_distance`: `0.017952436666573316`
- token-loss delta: `-0.003974093905492836`
- total-loss delta: `-0.003951494761727734`
- all training checks passed: yes

## Rollout Evidence

- rollout pairs: `32`
- baseline all legal: `True`
- enabled all legal: `True`
- baseline starved cases: `5`
- enabled starved cases: `8`
- new starved cases: `3`
- baseline rigid cases: `10`
- enabled rigid cases: `13`
- mean dominant-spacing ratio delta: `+0.06219724054108845`
- mean F1@100ms delta: `-0.030968231114384386`
- mean second-window share delta: `-0.0343462851188464`
- mean event-count ratio delta: `-0.03481525942636601`

## Failure Cases

| Case | New starved | Rigid delta | F1 delta | Event-count delta | Second-window delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `14_oomori_seiko_justadice_tv_size_remu_hard` | `True` | `+0.4109` | `-0.3231` | `-0.5977` | `-0.4983` |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `True` | `+0.4109` | `-0.3515` | `-0.6753` | `-0.4983` |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `True` | `+0.5154` | `-0.3488` | `-0.9796` | `-0.4681` |

Additional notable regressions:

- `08_moso_calibration_sakurairo_diary_tv_size_drum_hitnormal_hard`: rigid delta `+0.4453`.
- `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`: rigid delta `+0.3784`, event-count ratio delta `+0.3194`.
- `30_goreshit_one_way_to_hannover_cokiiplay_autophobia`: enabled generated only `2` timepoints but was not counted as newly starved by the current threshold because the second-window share increased.

## Commands

```bash
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_baseline.yaml
uv run python -m pulsefield_model.training.mapper_v3 --config artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_enabled.yaml
uv run python - <<'PY' ... run 32 baseline/enabled real-audio rollout pairs and write full32_gate_pairs.json ... PY
uv run python -m pulsefield_model.evals.mapper_v3_time_shift_distance_tiny_training_gate --baseline-report artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/train/baseline/report.json --enabled-report artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/train/enabled/report.json --rollout-pairs-json artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_pairs.json --summary-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json --report-output artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_result_report.md
uv run --group dev pytest tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/evals/test_mapper_v3_trained_runtime_rollout.py tests/evals/test_mapper_v3_time_shift_distance_tiny_training_gate.py -q
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json >/tmp/full32_time_shift_distance_summary.valid.json
uv run python -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_pairs.json >/tmp/full32_time_shift_distance_pairs.valid.json
```

## Command Results

- baseline training: `mapper_v3_training_done steps=500 final_loss=2.262773`
- enabled training: `mapper_v3_training_done steps=500 final_loss=2.258821`
- comparator: `mapper_v3_time_shift_distance_tiny_training_gate_done route=MUTATE_TIME_SHIFT_DISTANCE_OBJECTIVE token_loss_delta=-0.003974`
- training/model guard: `24 passed`
- rollout/evaluator guard: `19 passed`
- JSON validation: passed

## Verification / Failure Modes

- Checks performed: config validation, two matched 500-step trainings, 64 real-audio rollouts, rollout-pair comparator, guard tests, JSON validation.
- Failed checks: rollout checks `no_new_starved_cases` and `mean_rigid_not_worse`.
- Suspected confounders: the expected-value time-shift distance term can lower teacher-forced loss while pushing greedy decode toward more rigid or starved attractors.
- Expected failure modes observed: teacher-forced auxiliary viability did not translate to full32 generated-output quality.
- Unexpected failure modes: enabled remained legal in every case, so the failure is quality/regression rather than runtime legality.
- Reproducibility notes: full32 rollout summaries are under `artifacts/tmp/mapper_v3_time_shift_distance_row_filtered_repair_full32_gate/rollouts/`; audit-facing pair manifest and comparator summary are in the report directory.

## Interpretation

- What the result supports: the row-filtered loss implementation is numerically stable, and the auxiliary can train for 500 steps.
- What the result does not support: using the current `lambda_time_shift_distance=0.5` expected-shift objective as a default or scaling it toward replacement.
- Alternative explanations: lambda/scale may be too strong, the expected-value target may be too smooth, or greedy decode may need a different timing-aware objective.
- Positive / negative / ambiguous classification: negative for the current objective formulation at full32 scale.
- Recommended next step: `MUTATE` the timing objective or deprioritize it; do not proceed to full 4k or replacement with this formulation.
- Human owner decision: pending.
