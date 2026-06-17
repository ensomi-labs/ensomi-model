# Mapper v2.1 Tap-Only Anti-Rigid Stress Probe Result Report

## Scope

This probe reruns hard-block anti-rigid suppression only when the recent repeated-spacing run is tap-only. It uses the four committed stress cases and does not change defaults.

## Decision

Decision: `KILL`.

- Reason: tap-only detector failed legality on the stress set
- Cases: `4`
- Tap-only all legal: `False`
- New starved vs baseline: `0`
- Rigid-improved cases vs baseline: `3`
- Blocked candidates: `117`
- Hard-identical cases: `0`
- Baseline-identical cases: `1`

## Aggregate Table

| Variant | Legal | Mean F1 | Mean rigid | Starved | Mean event ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2.1 baseline` | `False` | 0.514434 | 0.895081 | 1 | 1.205879 |
| `hard block` | `False` | 0.411986 | 0.633323 | 2 | 0.860919 |
| `tap-only` | `False` | 0.431285 | 0.733392 | 1 | 1.126496 |
| `v3 500 wide` | `True` | 0.464883 | 0.744172 | 2 | 1.153683 |

## What Passed

- Tap-only mode fixed the invariant case `05`: it completed legally and was not starved.
- Tap-only introduced `0` new starved cases versus baseline.
- Tap-only improved rigidity in `3` / `4` stress cases.
- Tap-only no longer matched hard-block exactly on any stress case; hard-identical cases: `0`.

## What Surfaced

- Tap-only failed the primary legality gate: only `3` / `4` stress cases were legal.
- Case `04_oomori_seiko_justadice_tv_size_remu_normal` reverted to the baseline dead-end/starvation pattern because no tap-only candidates were blocked.
- The two dead-end cases require opposite behavior: case `05` needs LN suppression skipped, while case `04` needs some LN-pattern intervention.
- A simple tap-only predicate is therefore too blunt for default or full-slice scaling.

## Case Table

| Case | Baseline F1/Rigid/Starved | Hard F1/Rigid/Starved | Tap-only F1/Rigid/Starved | v3 F1/Rigid/Starved | Tap legal | Blocks |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `04_oomori_seiko_justadice_tv_size_remu_normal` | 0.202532/0.909091/True | 0.677686/0.584906/False | 0.202532/0.909091/True | 0.709220/0.712329/False | False | 0 |
| `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` | 0.645161/0.671233/False | 0.439024/0.548387/True | 0.637931/0.523077/False | 0.662252/0.530000/False | True | 32 |
| `29_camellia_beyond_the_geostationary_orbit_level_leniane_synergic_ascent` | 0.617450/1.000000/False | 0.236364/0.733333/True | 0.575000/0.809091/False | 0.240000/0.760000/True | True | 64 |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | 0.592593/1.000000/False | 0.294872/0.666667/False | 0.309677/0.692308/False | 0.248062/0.974359/True | True | 21 |

## Interpretation

The tap-only detector should not be scaled from this probe. It failed a stress-set safety or usefulness gate.

## Next Step

Kill this detector mutation; return to v3/C3 diagnostics or design a different v2.1 grammar change.
