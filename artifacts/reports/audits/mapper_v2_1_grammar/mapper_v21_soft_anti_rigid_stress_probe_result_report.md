# Mapper v2.1 Soft Anti-Rigid Stress Probe Result Report

## Scope

This probe reruns only the finite-penalty anti-rigid variant on four stress cases from the committed hard-block fixed-slice comparison. It does not change model weights, tokenizer, or defaults.

## Decision

Decision: `KILL`.

- Reason: soft penalty failed legality on the stress set
- Cases: `4`
- Soft penalty: `4.0`
- Soft all legal: `False`
- Soft new starved vs baseline: `2`
- Soft rigid-improved cases vs baseline: `4`
- Soft mean F1 delta vs hard block: `0.000000`
- Soft mean rigid delta vs baseline: `-0.261758`

## Aggregate Table

| Variant | Legal | Mean F1 | Mean rigid | Starved | Mean event ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2.1 baseline` | `False` | 0.514434 | 0.895081 | 1 | 1.205879 |
| `hard block` | `False` | 0.411986 | 0.633323 | 2 | 0.860919 |
| `soft penalty` | `False` | 0.411986 | 0.633323 | 2 | 0.860919 |
| `v3 500 wide` | `True` | 0.464883 | 0.744172 | 2 | 1.153683 |

## What Passed

- The stress probe completed `4` real-audio soft-penalty rollouts.
- The finite penalty activated `66` times on the selected stress cases.
- Soft penalty reduced dominant-spacing ratio versus baseline in `4` cases.
- Case `04` stayed legal under soft penalty and avoided the baseline dead-end/starvation pattern.

## What Surfaced

- Soft penalty legality failed: `3` / `4` cases were legal.
- Soft penalty introduced `2` new starved case(s) versus baseline.
- Penalty `4.0` matched the hard-block aggregate exactly on mean F1 and mean rigidity, so it behaves as effectively hard under greedy decoding on this stress set.
- Case `05` still dead-ended at the first-window boundary; case `29` remained starved.

## Case Table

| Case | Baseline F1/Rigid/Starved | Hard F1/Rigid/Starved | Soft F1/Rigid/Starved | v3 F1/Rigid/Starved | Soft legal | Blocks |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | 0.202532/0.909091/True | 0.677686/0.584906/False | 0.677686/0.584906/False | 0.709220/0.712329/False | True | 19 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | 0.645161/0.671233/False | 0.439024/0.548387/True | 0.439024/0.548387/True | 0.662252/0.530000/False | False | 1 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | 0.617450/1.000000/False | 0.236364/0.733333/True | 0.236364/0.733333/True | 0.240000/0.760000/True | True | 23 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 0.592593/1.000000/False | 0.294872/0.666667/False | 0.294872/0.666667/False | 0.248062/0.974359/True | True | 23 |

## Interpretation

Penalty 4.0 should not be scaled. The stress set reproduced a safety failure or failed to remove starvation.

## Next Step

Do not scale penalty 4.0; mutate the detector or penalty schedule.
