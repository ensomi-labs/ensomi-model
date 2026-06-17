# Mapper v2.1 Terminal LN-Start Guard Repair Result Report

## Scope

This eval reruns v2.1 fixed-slice real-audio cases with the existing grammar `min_ln_duration_ms` primitive exposed through runtime rollout. The guard is opt-in and mapper defaults remain unchanged.

## Decision

Decision: `TEST_NEXT`.

- Reason: terminal LN-start guard passed the four-case stress gate
- Full32: `False`
- Runs: `8`
- Cases: `4`
- Primary case 04 baseline legal: `True`
- Primary case 05 anti-rigid legal: `True`
- All candidate legal: `True`
- Max-token count: `0`
- New starved count: `0`
- Mean F1 delta vs matching comparator: `0.064638`
- Anti-rigid blocked count: `98`

## Aggregate Table

| Mode | Legal | Mean F1 | Starved | Mean rigid | Mean event ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline + terminal guard` | `4` / `4` | 0.608538 | 0 | 0.882094 | 1.478267 |
| `anti-rigid + terminal guard` | `4` / `4` | 0.447158 | 1 | 0.622663 | 1.140919 |

## What Passed

- The eval completed `8` real-audio guarded rollouts.
- The terminal guard used `min_ln_duration_ms=20` and remained opt-in.
- Primary case `04` baseline legality: `True`.
- Primary case `05` anti-rigid legality: `True`.

## What Surfaced

The terminal guard passed the stress gate. The next required check is the full32 widening run before changing defaults.

## Case Table

| Case | Mode | Previous legal | Candidate legal | Previous F1/Rigid/Starved | Candidate F1/Rigid/Starved | F1 delta | Terminal ms | Tokens | Blocks |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `baseline_guard` | False | True | 0.202532/0.909091/True | 0.578947/0.857143/False | 0.376416 | 16000 | 339 | 0 |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | `anti_rigid_guard` | True | True | 0.677686/0.584906/False | 0.677686/0.584906/False | 0.000000 | 16000 | 199 | 19 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `baseline_guard` | True | True | 0.645161/0.671233/False | 0.645161/0.671233/False | 0.000000 | 16000 | 275 | 0 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | `anti_rigid_guard` | False | True | 0.439024/0.548387/True | 0.579710/0.505747/False | 0.140686 | 16000 | 338 | 33 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `baseline_guard` | True | True | 0.617450/1.000000/False | 0.617450/1.000000/False | 0.000000 | 16000 | 351 | 0 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | `anti_rigid_guard` | True | True | 0.236364/0.733333/True | 0.236364/0.733333/True | 0.000000 | 16000 | 274 | 23 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `baseline_guard` | True | True | 0.592593/1.000000/False | 0.592593/1.000000/False | 0.000000 | 16000 | 351 | 0 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `anti_rigid_guard` | True | True | 0.294872/0.666667/False | 0.294872/0.666667/False | 0.000000 | 16000 | 282 | 23 |

## Next Step

Run the optional full32 widening gate before changing defaults.
