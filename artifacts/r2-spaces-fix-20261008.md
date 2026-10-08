# The three approved spaces fixes, applied and re-checked (2026-10-08)

Astra job `20261008-072613-r2-spaces-fix` (bings-mac, effort xhigh, default tier, workspace-write), brief `~/ensomi/.sync/cp/scratch/r2-spaces-fix/brief-astra.md`, approved by the human ([d-unit-composition](r2-representation-20261008.md#d-unit-composition), item 1). Finished 07:57 UTC, exit 0; 399 s of job wall time on CPU (MPS is not available in the Codex sandbox).

- Report: `artifacts/r2-spaces-fix-20261008/report.md` (mirrored); final message `~/ensomi/.sync/mac/jobs/20261008-072613-r2-spaces-fix/last.md`.
- Outputs (mirrored): `constants.json`, `invariance.json`, `results.json`, `lens-before-after.csv`, `labelfree-before-after.csv`, `targeted-coupling.json`, `residual-fit-r2.csv`, `seed-spread-{before,after}.json`, `run-records.json`. The fixed Lens table, all spaces, 3,763 sections × 188 columns: `lens.parquet` (mac). Scripts: `artifacts/r2-spaces-fix-20261008/scripts/`.

<a id="o-fixes"></a>
## The fixes and whether each met its own label-free purpose [M]

| Fix | Change | Check, before → after | Verdict |
|---|---|---|---|
| S5 and S8 residualised on S3 | Coordinate-wise OLS on 17,454 corpus windows (4, 8, 16 beats) from 500 fit_train charts; mean R² removed S5 .211, S8 .124 | Nonlinear R² from S3: S5 .245 → −.035; S8 .200 → −.010 | pass |
| S6 shrinkage stable under row count | Prior strength scaled with rows, s(n) = s64 · n/64, s64 the corpus beta-binomial MLE (chord, LN, release 6.81; jack 4.64; switch 68.1) | Worst truncation discrepancy 2.850 → .807 null SD | pass |
| S4 residualised on mass | OLS on held and LN mass (adding LN lowered corpus CV MSE .939 → .860); mean R² removed .151 | Nonlinear R² from mass: held + LN .431 → .157; full mass .500 → .324 | fail (target near 0) |

- The residual maps were fitted on the same corpus windows the coupling check uses. In-sample linear R² is zero by construction; the nonlinear check is the meaningful one, and with 26 predictors on 17,454 windows the in-sample advantage is small [I].
- Invariance after the fixes: mirror and tempo pass for S4, S5, S6 and S8 (S8 changes with tempo by design). Truncation: S6 now passes; S5 passes (.97); S4 (2.11) and S8 (2.38) still fail.
- Coupling after the fixes (R² from the other blocks / from S1 mass): S4 .58 / .32, S5 .22 / −.07, S6 .33 / .06, S8 .32 / .04. Distance correlations stay above their permutation nulls.

<a id="o-fixes-lens"></a>
## Lens results before → after, post hoc and descriptive [M]

Every split was looked at before; claims and strata are unchanged, and nothing is re-decided. AUC stratified; inc = increment over the mass baseline L.

| Space / concept (claim) | V1 AUC | V1 inc | V2 AUC | V2 inc | Swap AUC |
|---|---|---|---|---|---|
| S5 / stream (order) | .698 → .537 | .015 → .004 | .725 → .496 | .026 → .013 | .656 → .558 |
| S5 / jack (blind) | .681 → .464 | .089 → −.005 | .778 → .540 | .093 → .004 | .721 → .468 |
| S8 / stream (blind) | .763 → .609 | .032 → .004 | .718 → .618 | .023 → .008 | .768 → .679 |
| S8 / trill (blind) | .625 → .552 | −.024 → −.050 | .689 → .673 | .189 → .150 [.037, .256] | .698 → .594 |
| S4 / LN (order) | .939 → .888 | .067 → .062 [.043, .082] | .773 → .751 | −.001 → .004 | .944 → .897 |
| S6 / jack (weak) | .672 → .683 | .019 → .027 | .650 → .694 | .047 → .074 [.009, .149] | .681 → .680 |

Full table with intervals: `lens-before-after.csv` (63 rows).

<a id="v-fixes-checks"></a>
## Main-thread checks (2026-10-08)

- `lens-before-after.csv`: the rows above match the final message.
- `table-checks.json`: unchanged coordinates (S1-S3) are identical to the original tables, same section ids and order.
- `report.md` line 20: the residual fit sample is 500 charts / 469 song groups, fit_train only, no validation-group overlap. The brief asked for the full corpus fit set (7,836 charts); the job used the original coupling sample instead. Its effect on the fitted maps is not checked.

<a id="s-fixes-reading"></a>
## Reading (main thread)

1. **S5 (contour) holds nothing for stream or jack beyond S3.** Its earlier stream ordering and its jack leak were both the part S5 shares with S3: after residualisation they go to chance on all three splits. The perception worker's warning (residualising would remove part of what reads as stream) is borne out in the sense that the stream signal goes; what goes is shared with S3, not contour of its own.
2. **S4's LN increment over mass survives** (V1 .062 [.043, .082], swap similar), so S4's LN signal beyond mass does not come from mass. It still does not show on the 49 human cells, and S4 keeps a third of its variance predictable from mass nonlinearly.
3. **S8 keeps trill information on the human cells** after residualisation (V2 increment .150 [.037, .256]), unclaimed and post hoc. Physical speed seems to carry something about how trills are played that S3 does not; worth a look once a preregistered check is possible.
4. **S6 is now stable under truncation**, which the segment unit needs (variable-length units).
5. None of this is a decision. The designed spaces now separate a little better (S5, S8 from S3; S6 stability), and S3 remains the only space shown to recognise human labels beyond mass.
