# Mapper v2.1 Terminal Guard / v3 Route Synthesis Result Report

## Scope

This artifact-only synthesis reconciles the v2.1 terminal LN-start guard full32 pass with the current v3 and C3 route artifacts. It does not train, rerun rollout, retokenize data, or change defaults.

## Decision

Route: `TEST_V3_BOUNDARY_OR_SPACING_STATE_REPAIR_KEEP_V21_GUARD_OPT_IN`.

- Reason: v2.1 terminal guard is now a legal full32 comparator, C3 mapper integration remains diminishing, and v3 remains the lower-bit local target but still has generated-prefix state/timing failures
- Recommended next step: Keep `min_ln_duration_ms` default-off and use the repaired v2.1 guard as the comparator while creating the next v3 boundary/spacing-state repair card.
- Next card: `target_grammar_v3_boundary_or_spacing_state_repair`

## Guard Results

| Guard | Value |
| --- | ---: |
| required artifacts present | `True` |
| artifact only | `True` |
| no training | `True` |
| no rollout rerun | `True` |
| no tokenizer/default change | `True` |
| v2.1 terminal guard full32 passed | `True` |
| v3 representation ready | `True` |
| C3 mapper path diminishing | `True` |
| same-slice comparison available | `True` |
| v3 repair route open | `True` |

## Evidence Snapshot

| Family | Metric | Value |
| --- | --- | ---: |
| v2.1 terminal guard | runs/cases | `64 / 32` |
| v2.1 terminal guard | all legal | `True` |
| v2.1 terminal guard | new starved | `0` |
| v2.1 terminal guard | mean F1 delta vs comparator | `0.008080` |
| v2.1 terminal guard | anti-rigid blocked | `1906` |
| v3 representation | reconstruction mismatches | `0` |
| v3 representation | token reduction | `19.955%` |
| v3 representation | total-bit reduction | `5.431%` |
| v3 quality | generated-prefix route | `TEST_BOUNDARY_OR_SPACING_STATE_REPAIR` |
| v3 quality | trace mean event ratio | `0.533579` |
| v3 quality | trace second-window share | `0.034722` |
| C3 | route | `MUTATE_TO_V3_GRAMMAR_REPAIR` |
| C3 | production input legal | `False` |
| C3 | codec delta bits/event | `-0.384910` |

## Same-Slice v2.1 Repair vs v3

| Candidate | Cases | Legal | Candidate F1 | v3 F1 | F1 delta | Starved delta | Event-ratio delta | Rigid delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline + terminal guard` | 32 | 32 | 0.755842 | 0.639566 | 0.116277 | -10 | 0.210121 | 0.205960 |
| `anti-rigid + terminal guard` | 32 | 32 | 0.712436 | 0.639566 | 0.072871 | -9 | 0.167097 | -0.029038 |

## What Passed

- The v2.1 terminal guard converts the previous illegal fixed-slice v2.1 paths into legal full32 rollouts with no new starvation.
- v3 remains exactly reconstructive, lower-token, lower-bit, local, and teacher-forcing friendly in the representation artifacts.
- C3 remains a strong codec-side result, but the loaded target-complexity artifact still rejects it as production input or emitted target replacement.

## What Surfaced

- The v2.1 terminal guard is a stronger comparator and short-term hardening result, not a final target-grammar replacement.
- v3 rollout quality remains blocked by generated-prefix state/timing failures, so the next useful v3 card should target boundary or spacing state repair.
- The route should not go back to C3 mapper-side integration unless a new formulation solves production legality and target cost.

## Next Step

Keep `min_ln_duration_ms` default-off and use the repaired v2.1 guard as the comparator while creating the next v3 boundary/spacing-state repair card.
