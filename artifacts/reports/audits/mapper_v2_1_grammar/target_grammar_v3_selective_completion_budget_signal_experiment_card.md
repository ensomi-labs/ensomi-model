# Target Grammar v3 Selective Completion-Budget Signal Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: `target_grammar_v3_generated_prefix_event_budget_calibration_audit` selected `TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL` after showing that primary failures contain rank-near event opportunities and large second-window deficits, while broad spacing release overproduces sentinels.
- Acceptance source, if any: active thread goal plus committed v3/C3 audit artifacts.
- Source snapshot / evidence grade: strong evidence that broad scalar/decode repairs are not promotable; medium evidence that a selective generated-prefix budget signal exists; weak evidence that an online budget-conditioned gate will pass sentinel guards until this smoke runs.

## Hypothesis

A v3 generated-prefix event opportunity should only be promoted when the generated chart is under an online completion budget. Using a control-density-derived budget envelope plus generated-prefix event counts can preserve the primary continuation gain while avoiding the sentinel flooding that killed broad spacing escape and the earlier selective event gate.

## Root Objective

Move v3 toward full-pipeline replacement readiness by testing the smallest legal mapper-facing mechanism that addresses free-running second-window starvation without changing v3 tokenization, grammar, replay, model weights, or defaults.

## Goal Decomposition

- Subgoal 1: Preserve the v3 representation contract: reversible event reconstruction, lower target bits/tokens than v2.1, local online inference, and no C3-style cross-window target replay.
- Subgoal 2: Add a default-off runtime smoke that uses only online-available inputs: generated prefix state, valid-token mask/logits, current time, and control/density features produced from audio.
- Subgoal 3: Gate rank-near event promotion by an estimated remaining event budget so primary deficits can move but sentinels cannot flood.
- Subgoal 4: Evaluate on the six primary generated-prefix failures plus four sentinel controls, with hard event-ratio and legality guards.

## Candidate Variants

- Variant A: Re-run broad trace-conditioned spacing escape with stricter caps. Rejected because the existing smoke already recovered continuation by flooding events; stricter caps would be a threshold sweep of a killed local decode repair.
- Variant B: Re-run scalar event-budget or continuation-jump training at a lower weight. Rejected because previous gates show undertransfer, dead ends, rigidity, and blunt event-count calibration failure.
- Variant C: Raw selective event gate on rank-near opportunities. Rejected as implemented because `target_grammar_v3_selective_event_gate_smoke` regressed pass-like controls and overproduced selected cases.
- Variant D: Control-budget-conditioned selective completion gate. Selected because it adds the missing selectivity source: act only when generated events are below an online budget estimate.
- Variant E: Immediate v2.1 grammar-improvement card. Deferred only until Variant D fails; the current audit gives v3 one narrower, fast-failing mechanism to test.

## Local Verification Matrix

| Variant | Smallest check | Pass condition | Fail condition |
| --- | --- | --- | --- |
| A | Existing spacing-escape smoke | Primary and sentinel event-ratio guards pass | Already failed sentinel and primary event-ratio guards |
| B | Existing scalar-loss rollout gates | Legal full32 improvement without rigidity/starvation regression | Already failed continuation, event-budget, and conditioned-distribution gates |
| C | Existing selective event-gate smoke | Controls preserved and no overproduction | Already failed pass-like control/overproduction guards |
| D | Ten-case control-budget selective smoke | >=4/6 primary cases improve continuation with all primary/sentinel event ratios within cap | Flooding, under-moving, dead end, max-token, or control regression |
| E | Card-only v2.1 pivot | v3 selective signal exhausted | Deferred until D result |

## Selected Variant

- Selected: Variant D, a default-off control-budget-conditioned selective completion gate smoke.
- Rejected: Variants A, B, and C because they are already killed or regressed by committed evidence.
- Deferred: Variant E until this one bounded v3 test fails.
- Why this is the smallest useful test: it uses the existing v3 rollout/logits-transform path and committed 10-case evidence slice. It does not require training, a new target grammar, C3 side-stream conditioning, or default inference changes.

## Selection Pressure

- Primary pressure: recover second-window continuation only in under-budget generated-prefix states.
- Guard pressure: all primary and sentinel event-count ratios must stay within the hard cap; no legality, dead-end, max-token, duplicate, or boundary regressions.
- Runtime pressure: one runtime-backed smoke over 10 real-audio 16s rollouts before any training.
- Kill pressure: if online budget conditioning cannot prevent sentinel overproduction, stop v3 decode/signal patching and pivot to v2.1 grammar improvement or a structural v3 grammar mutation.

## Research Question

Can an online control-budget-conditioned event opportunity gate convert rank-near v3 generated-prefix opportunities into continuation without reproducing the flooding behavior that killed broad spacing escape and the earlier selective event gate?

## Closest Analogies / Novelty Layer

- Closest analogies: constrained decoding with coverage budgets, budget-aware sequence completion, density-conditioned decoding, and exposure-bias diagnostics.
- Relevant taxonomy bucket: local verification and training/inference calibration, not a new architecture.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: v3 event-token grammar remains the representation candidate; this is an engineering calibration probe around generated-prefix completion.

## Minimal Change

Add a runtime-backed eval smoke that:

- builds a default-off logits transform or context-aware transform wrapper;
- estimates a per-window or per-half-window event budget from online `density_teacher_8s` / control-density features, not from target beatmaps;
- tracks generated event count by current window/half;
- promotes only rank-near event tokens when generated count is below the budget envelope;
- caps promoted events, enforces minimum event spacing, and records skipped reasons;
- compares candidate metrics against committed baseline rows for six primary cases and four sentinel controls.

If the current logits-transform hook cannot access density features cleanly, add the smallest default-off eval-only plumbing needed to provide the transform with the current window's control-density tensor. Do not change default runtime behavior.

## Files Likely to Change

- `src/pulsefield_model/evals/mapper_v3_selective_completion_budget_signal_smoke.py`
- `tests/evals/test_mapper_v3_selective_completion_budget_signal_smoke.py`
- Possibly `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py` if the eval needs to expose the current control-density feature to the transform.
- Possibly `src/pulsefield_model/inference/mapper_v3_rollout.py` only if a default-off context-aware transform hook is required.
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_completion_budget_signal_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_completion_budget_signal_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_completion_budget_signal_summary.json`

## Read-Only Context Files

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_event_budget_calibration_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_generated_prefix_state_trace_audit_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trace_conditioned_spacing_escape_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_event_gate_smoke_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_500step_fixed_slice_wide_audit_summary.json`
- Per-case baseline summaries under `artifacts/tmp/mapper_v3_500step_fixed_slice_wide_audit/rollouts/`

## Dataset Slice

Use the latest 10-case calibration slice:

- Primary generated-prefix failures: `14_oomori_seiko_justadice_tv_size_remu_hard`
- Primary generated-prefix failures: `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx`
- Primary generated-prefix failures: `19_nekodex_circles_famoss_hard`
- Primary generated-prefix failures: `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair`
- Primary generated-prefix failures: `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial`
- Primary generated-prefix failures: `18_billiummoto_four_veiled_stars_aries_famoss_hard`
- Sentinel controls: `23_usao_knight_rider_kuo_kyoka_expert`
- Sentinel controls: `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent`
- Sentinel controls: `28_usao_knight_rider_kuo_kyoka_cs_lone_sonorous`
- Sentinel controls: `01_hatsuki_yura_guren_yasha_a_m_d_normal`

## Baseline / Comparator

- Generated-prefix trace baseline: 6/6 primary failures have rank-near event opportunities, 336 total candidate opportunities, mean event ratio `0.540072`, mean generated second-window share `0.034722`, mean reference second-window share `0.600482`.
- Broad spacing escape comparator: primary second-window improves in 6/6, but primary event-ratio-over-cap cases are `3`, sentinel event-ratio-over-cap cases are `4/4`, and sentinel median event ratio is `1.634997`.
- Prior selective event gate comparator: positive on 2/3 low-bias cases, but killed by overproduction and pass-like control regression.
- 500-step fixed-slice baseline rows are the per-case comparator for event ratio, second-window share, F1@100ms, dominant spacing, boundary ratio, and legality.

## Primary Metric

Positive primary count:

- at least `4/6` primary cases improve second-window event share by `>=0.15`;
- each positive primary case remains legal and has candidate event-count ratio `<=1.25`;
- no primary case exceeds event-count ratio `1.25`.

## Secondary Metric

- Sentinel event-count ratio: all `4/4` sentinel cases must stay `<=1.25`, and sentinel median event ratio must stay `<=1.25`.
- Mean primary F1@100ms must not regress by more than `0.05`.
- No dead-end cases, no max-token cases, no duplicate/non-increasing spacing regression.
- Candidate boundary-event ratio must not exceed baseline max by more than `0.05`.
- Report forced/promoted event counts, skipped reasons, and generated-vs-budget traces by case.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_selective_completion_budget_signal_smoke.py tests/evals/test_mapper_v3_generated_prefix_event_budget_calibration_audit.py -q
uv run python -m pulsefield_model.evals.mapper_v3_selective_completion_budget_signal_smoke
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_selective_completion_budget_signal_summary.json >/dev/null
git diff --check
```

## Guard Check

- No training.
- No tokenizer, grammar, replay, vocab, or model-weight change.
- No default runtime behavior change.
- No target-derived C3 conditioning.
- No target beatmap or reference event count may be used by the transform during generation; reference data is evaluation-only.
- If a control-density budget cannot be computed from online features, the experiment must return `MUTATE_BUDGET_SOURCE` rather than silently using target labels.

## Qualitative Check

The report must show whether budget conditioning actually suppresses sentinel flooding. A result that improves primary continuation by flooding primary or sentinel charts is negative even if F1 rises.

## Positive Signal

`TEST_NEXT_FULL32_BUDGET_SIGNAL` if all primary/sentinel event-ratio guards pass, at least four primary cases improve continuation, and no legality or duplicate/boundary guard regresses.

## Negative Signal

`KILL_SELECTIVE_COMPLETION_BUDGET_SIGNAL` if the gate overproduces any sentinel, overproduces any primary above `1.25`, fails to improve at least four primary cases, or needs target/reference information to decide budget.

## Kill Criteria

- Any sentinel event-count ratio exceeds `1.25`.
- Any primary event-count ratio exceeds `1.25`.
- Any dead-end, max-token, duplicate/non-increasing spacing, or boundary spill regression appears.
- The transform uses target beatmap/reference information during generation.
- The only positive cases require thresholds that are effectively the killed broad event gate.

## Expected Failure Modes

- Control-density budget may be too weak or poorly calibrated for event counts.
- Budget gating may under-move hard primary cases.
- Budget gating may still overpromote sentinels if density estimates are high.
- Forced events may improve recall/F1 while damaging lane quality not measured by timestamp F1.
- Runtime access to density features may require a small default-off hook.

## Confounders

- The slice has only 10 cases and is intentionally adversarial.
- `density_teacher_8s` is named "teacher" historically but is an inference density feature in runtime; the implementation must verify it is not target-derived in this path.
- F1@100ms can reward regular dense grids.
- Device/runtime variance may slightly change logits or timing diagnostics.

## Expected Runtime / Runtime Budget

Expected runtime: under 25 minutes for 10 real-audio 16s rollouts plus focused tests. Stop if the density budget source is not online-legal or if the first three cases all violate event-ratio guards.

## Result Interpretation Plan

- Positive result would suggest: create a full32 budget-conditioned signal audit before any training or default change.
- Negative result would suggest: kill v3 local decode/signal repair and pivot to v2.1 grammar improvement or a structural v3 grammar mutation.
- Ambiguous result would require: one budget-source calibration audit, not a training run.
- Human owner decides: whether this is enough to keep v3 as the active mapper-facing branch.
- Next-loop action if positive: create `target_grammar_v3_selective_completion_budget_full32_audit`.
- Next-loop action if negative: create a v2.1 grammar-improvement Experiment Card.
- Next-loop action if ambiguous: create a control-density budget-source calibration card.

## Result Log Template

- Experiment: target_grammar_v3_selective_completion_budget_signal
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: budget calibration from control density may be insufficient; if so, stop with `MUTATE_BUDGET_SOURCE` rather than training.

## Next-Loop Action

- If positive: plan a full32 budget-conditioned audit.
- If negative: pivot to v2.1 grammar improvement or structural v3 grammar mutation.
- If ambiguous: audit the online budget source before any training.

## Novelty Notes

- Closest analogies: budget-aware constrained decoding, coverage control, and density-conditioned sequence completion.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering calibration around v3's existing event-token representation, not a new representation claim.
