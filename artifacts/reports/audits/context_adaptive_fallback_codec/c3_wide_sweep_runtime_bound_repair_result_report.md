# C3 Wide-Sweep Runtime-Bound Repair Result Report

## Scope

This pass executes `c3_wide_sweep_runtime_bound_repair_experiment_card.md`. It changes only the C3 audit harness candidate-option construction and tests. It does not change mapper defaults, tokenizer outputs, training data, or inference behavior.

## Decision

Decision: `TEST_NEXT`.

The repair passed. The full-cache `--c3-wide-sweep` audit completed on committed code `c71c22141768d97bac03b695a8e0021f4e4ea12f` in `819.0450222920044` seconds and preserved C3 hardening gates.

## What Changed

The audit now builds one largest-window exact/mirror/skeleton candidate superset and derives per-policy options by filtering:

- allowed match type;
- policy window distance;
- phase filter.

This avoids rebuilding mirror/skeleton match candidates independently for each A6/A7/A8 policy.

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening --c3-wide-sweep --limit-chunks 5000 --report-path artifacts/tmp/c3_wide_sweep_runtime_bound_smoke_report.json --result-log-path artifacts/tmp/c3_wide_sweep_runtime_bound_smoke_result_log.md --comparison-csv-path artifacts/tmp/c3_wide_sweep_runtime_bound_smoke_comparison.csv --diagnostics-csv-path artifacts/tmp/c3_wide_sweep_runtime_bound_smoke_diagnostics.csv
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening --c3-wide-sweep
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json >/dev/null
```

Results:

- Focused tests: `10 passed in 0.86s`
- Broader guard tests: `21 passed in 0.91s`
- Limited wide smoke: completed in `3.366161124984501s` with all A6/A7/A8 variant families present
- Full wide audit: completed in `819.0450222920044s`
- JSON validation: passed

## Full Wide Result

- `limited`: `false`
- `c3_wide_sweep`: `true`
- Selected variant: `a2_skeleton_residual_all_fallback_w256`
- Research promotion pass: `true`
- Engineering promotion pass: `true`
- Hard reconstruction mismatches: `0`
- Selected test delta: `-0.3849096630040947`
- Same-song-filtered delta: `-0.38592890137466185`
- Clean-trace test delta: `-0.3809793375551669`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`

## A6 Window Sweep

| Variant | Test Delta |
| --- | ---: |
| `a6_active_all_fixed_w16` | `-0.17160129504005273` |
| `a6_active_all_fixed_w32` | `-0.20256685167133437` |
| `a6_active_all_fixed_w64` | `-0.22649303830854528` |
| `a6_active_all_fixed_w128` | `-0.23783549837040585` |
| `a6_active_all_fixed_w512` | `-0.2440259369618829` |
| `a6_active_all_fixed_w1024` | `-0.23841577518529888` |
| `a6_active_all_fixed_full_history` | `-0.2682654572065113` |

Interpretation: bounded windows retain most of the active-C3 gain. Full history improves over bounded windows, but the 64-512 range already gives a strong negative delta, so C3 does not depend solely on unbounded history.

## A7 Pointer-Code Sweep

| Variant | Test Delta |
| --- | ---: |
| `a7_active_all_elias_gamma_w256` | `-0.24825851104511987` |
| `a7_active_all_elias_delta_w256` | `-0.24476655201139685` |
| `a7_active_all_power_bucket_w256` | `-0.2351306921141303` |
| `a7_active_all_train_fitted_w256` | `-0.24349222205415888` |
| `a7_active_all_online_adaptive_w256` | `-0.09853275142624174` |

Interpretation: the active-C3 signal survives several charged pointer-code families. Online adaptive is much weaker but still negative, so the result is not only a fixed-width accounting artifact.

## A8 Source-Filter Sweep

| Variant | Test Delta |
| --- | ---: |
| `a8_active_all_fixed_same_offset_w256` | `-0.20670045531494186` |
| `a8_active_all_fixed_same_bar_phase_w256` | `-0.21168230393201704` |

Interpretation: phase-restricted references still win. The result is compatible with rhythm-phase-aligned local reuse, but not limited to a single narrow phase filter.

## What Surfaced

The previous runtime blocker is cleared for the current full-cache wide gate. The repair does not remove the broader C3 caveats:

- active-hold causality is still not proved;
- C3 is still a fallback side-stream representation, not a simple global tokenizer replacement;
- mapper-side learnability and generated chart quality are still unproven.

The generated hardening report records `code_dirty=true`. For this run, source-code provenance is still clear because `code_commit` is `c71c22141768d97bac03b695a8e0021f4e4ea12f`; the dirty flag is expected from generated artifact outputs.

## Next Loop

The C3 hardening gate now includes full A6/A7/A8 decomposition. The next bounded step should move from codec-side verification to one legal mapper-facing exposure path: side-stream tokenization artifact, auxiliary target, emitted target grammar, or two-stage predicted C3 plan.
