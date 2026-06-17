# Target Grammar v3 Generated-Prefix Event-Budget Calibration Audit Result Report

## Scope

This artifact-only audit executes `target_grammar_v3_generated_prefix_event_budget_calibration_audit_experiment_card.md`. It reads committed v3/C3 summaries, does not retrain, does not rerun rollout, and does not change tokenizer, grammar, replay, inference, or defaults.

## Decision

Route: `TEST_SELECTIVE_COMPLETION_BUDGET_SIGNAL`.

- Reason: 6 primary cases have rank-near event opportunities and 6 have large second-window deficits, while broad spacing release overproduces sentinels
- Next step: Create a bounded selective completion/budget signal card with hard primary and sentinel event-ratio guards; do not scale scalar losses or local decode repair.

## Guard Results

| Guard | Value |
| --- | ---: |
| `artifact_only` | `True` |
| `no_training` | `True` |
| `no_rollout_rerun` | `True` |
| `no_tokenizer_or_default_change` | `True` |
| `required_artifacts_present` | `True` |
| `rank_near_primary_signal` | `True` |
| `second_window_deficit_signal` | `True` |
| `sentinel_overproduction_guard_needed` | `True` |
| `prior_scalar_decode_repairs_failed` | `True` |

## Primary Generated-Prefix Evidence

- Primary cases: `6`
- Rank-near event opportunity cases: `6` / `6`
- Rank-near opportunity candidates: `336` total, mean `56.000000` per rank-near case
- Large second-window deficit cases: `6` / `6`
- Mean generated second-window share: `0.034722`
- Mean reference second-window share: `0.600482`
- Mean second-window share deficit: `0.565760`
- Mean event-count ratio: `0.540072`

| Case | Rank-near | Candidates | Best rank | Event ratio | Second share | Reference share | Deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14_oomori_seiko_justadice_tv_size_remu_hard | `True` | `50` | `2` | `0.551724` | `0.041667` | `0.540230` | `0.498563` |
| 17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx | `True` | `74` | `2` | `0.527473` | `0.041667` | `0.549451` | `0.507784` |
| 19_nekodex_circles_famoss_hard | `True` | `73` | `2` | `0.533333` | `0.041667` | `0.611111` | `0.569444` |
| 22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair | `True` | `50` | `2` | `0.516129` | `0.041667` | `0.548387` | `0.506720` |
| 31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial | `True` | `41` | `4` | `0.449438` | `0.000000` | `0.730337` | `0.730337` |
| 18_billiummoto_four_veiled_stars_aries_famoss_hard | `True` | `48` | `2` | `0.662338` | `0.041667` | `0.623377` | `0.581710` |

## Spacing-Escape Evidence

- Primary broad-release recovery: `True`
- Primary second-window improved: `6` / `6`
- Primary event-ratio-over-cap cases: `3`
- Primary mean event ratio under broad release: `1.222994`
- Primary mean second-window share delta: `0.447047`
- Sentinel event-ratio-over-cap cases: `4` / `4`
- Sentinel median event ratio under broad release: `1.634997`
- Sentinel mean event-ratio delta: `0.799908`

## Failed Branch Checks

- Continuation-jump failed: `True`
- Event-budget objective failed: `True`
- Conditioned event-distribution objective failed: `True`
- C3 target-side route costly: `True`
- Scalar/decode repairs failed as a family: `True`

## What Passed

- The required committed summaries were present and parseable.
- The audit stayed artifact-only: no training, rollout rerun, tokenizer change, grammar change, C3 conditioning, or default behavior change.
- The primary trace set contains rank-near event opportunities during second-window deficits.

## What Surfaced

- Broad spacing release can recover primary continuation but overproduces sentinels, so the useful next signal must be selective.
- The recent scalar or local-decode families are not promotable as tested: continuation-jump undertransfers, event-budget is too blunt, conditioned event distribution misses the event-ratio guard, and spacing escape floods.
- C3 remains strong codec-side evidence but is not the immediate emitted target route because target-side costs and production-input legality still fail.

## Interpretation

The v3 branch still has one narrow mapper-facing test left: a selective completion/budget signal tied to generated-prefix state and protected by sentinel overproduction guards. This audit does not approve v3 replacement or another broad loss sweep.

## Evidence Boundary

- Proved: the committed summaries contain selective primary deficits plus sentinel overproduction risk.
- Not proved: that a completion/budget signal will train, improve full32 rollout, or beat v2.1 in production.
- Not evaluated: full 4k replacement readiness, planner integration, or v3 default enablement.

## Next Step

Create a bounded selective completion/budget signal card with hard primary and sentinel event-ratio guards; do not scale scalar losses or local decode repair.
