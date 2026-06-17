# Target Grammar v3 Generated-Prefix State Trace Audit Result Report

## Scope

This pass reruns the selected six high-overlap v3 failure cases under generated prefixes and records per-step state/logit traces. It does not train, change tokenizer behavior, change grammar defaults, or add C3 sidecar conditioning.

## Result

Decision: `TEST_BOUNDARY_OR_SPACING_STATE_REPAIR`.

- Reason: 6/6 cases share `time_shift_repetition` as the primary generated-prefix class
- Selected cases: `6`
- Traced cases: `6`
- Concrete classes: `6`
- Failure classes: `{'time_shift_repetition': 6}`
- Secondary flags: `{'boundary_drift': 6, 'event_underselection': 6, 'time_shift_repetition': 6}`
- Mean event-count ratio: `0.533579`
- Mean second-window event share: `0.034722`
- Mean dominant-spacing ratio: `0.942535`

## What Passed

- The existing v3 runtime rollout path produced generated-prefix traces for the selected cases.
- The trace rows align logits, valid-token masks, replay state, emitted token, and post-token state.
- The result preserves the guard: no training, no tokenizer change, no default decode change, and no target-derived C3 input path.

## What Surfaced

- The failure is generated-prefix local: the traces classify rollout behavior, not teacher-forced token learnability.
- The primary surfaced mechanism is the class distribution shown below; secondary flags show co-occurring symptoms.
- This is diagnostic evidence only. It does not prove that a spacing, boundary, or event-ranking repair will pass a wider rollout gate.

## Cases

| case | class | first ms | generated/ref | second share | dominant spacing | best-event rank | emitted | secondary flags |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `14_oomori_seiko_justadice_tv_size_remu_hard` | `time_shift_repetition` | `1460` | `48/87` | `0.041667` | `0.936170` | `1` | `EV_1000` | `time_shift_repetition,event_underselection,boundary_drift` |
| `17_tamura_yukari_fantastic_future_tv_size_victorica_db_4k_mx` | `time_shift_repetition` | `1460` | `48/91` | `0.041667` | `0.936170` | `1` | `EV_0100` | `time_shift_repetition,event_underselection,boundary_drift` |
| `19_nekodex_circles_famoss_hard` | `time_shift_repetition` | `1460` | `48/90` | `0.041667` | `0.936170` | `1` | `EV_0010` | `time_shift_repetition,event_underselection,boundary_drift` |
| `22_oomori_seiko_justadice_tv_size_remu_grimoire_of_despair` | `time_shift_repetition` | `1460` | `48/93` | `0.041667` | `0.936170` | `1` | `EV_0001` | `time_shift_repetition,event_underselection,boundary_drift` |
| `31_camellia_beyond_the_geostationary_orbit_level_leniane_equatorial` | `time_shift_repetition` | `2420` | `40/89` | `0.000000` | `0.974359` | `1` | `EV_0021` | `time_shift_repetition,event_underselection,boundary_drift` |
| `18_billiummoto_four_veiled_stars_aries_famoss_hard` | `time_shift_repetition` | `1460` | `48/77` | `0.041667` | `0.936170` | `1` | `EV_1000` | `time_shift_repetition,event_underselection,boundary_drift` |

## Interpretation

The generated-prefix trace localizes the remaining v3 failure to rollout-state behavior, especially rigid time-shift repetition and boundary continuation. This does not prove a repair, but it defines a smaller target than another global tokenizer or broad mapper objective.

## Evidence Boundary

- Proved: the selected generated-prefix failures can be traced and classified without another training run.
- Proved: the remaining issue is not explained away by healthy teacher-forced time-shift ranks; it appears after generated prefixes enter the replay loop.
- Not proved: active-hold or C3 causality, end-to-end replacement readiness, or that any proposed repair will improve full32 rollout quality.

## Next Step

Create one bounded spacing/boundary state repair card; do not start broad v3 retraining.
