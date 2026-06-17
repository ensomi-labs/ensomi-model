# Mapper v2.1 Soft Penalty Sweep Stress Result Report

## Scope

This sweep tests weak finite anti-rigid penalties on the four committed stress cases. It does not change model weights, tokenizer, training, or defaults.

## Decision

Decision: `KILL`.

- Reason: no tested weak penalty cleared legality, starvation, rigidity, and hard-block-separation gates
- Selected penalty: `None`
- Passing penalties: `0`

## Penalty Table

| Penalty | Pass | Legal | New starved | Rigid improved | Mean F1 delta vs baseline | Mean rigid delta vs baseline | Differs from hard | Differs from baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.500000 | False | 3/4 | 1 | 2 | 0.076127 | -0.048968 | 3 | 2 |
| 1.000000 | False | 3/4 | 1 | 3 | 0.006126 | -0.102998 | 3 | 3 |
| 2.000000 | False | 3/4 | 1 | 4 | 0.005383 | -0.188083 | 2 | 4 |

## What Passed

- The sweep completed all `12` planned real-audio stress rollouts.
- At least one weak penalty separated from hard-block behavior; best hard-block-separation count was `3` / `4` cases.
- The best rigidity signal improved `4` / `4` cases versus baseline.
- The best mean F1 delta versus baseline was `0.076127`.

## What Surfaced

- No penalty cleared the primary gate.
- Every tested penalty had only `3` / `4` legal stress rollouts.
- New-starvation count never reached zero; best observed count was `1`.
- Case `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` remained the invariant failure: dead-end plus starvation across all tested penalties.
- This suggests the simple repeated-spacing token suppression schedule is not enough; penalty strength alone is not the right next lever.

## Interpretation

The weak penalty sweep did not find a safe finite penalty regime. Simple repeated-spacing token suppression should be deprioritized.

## Next Step

Kill the simple finite-penalty schedule; mutate the detector/schedule or return to v3/C3 diagnostics.
