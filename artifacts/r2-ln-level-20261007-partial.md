# R2 phase N: why the natural model does not hold a normal LN level (PARTIAL report)

Partial report of the fresh Opus subagent for [q-lnlen-hintfree](r2-phasen-and-lens-20261006.md#q-lnlen-hintfree), copied verbatim at the 2026-10-07 closeout (source `~/ensomi/.sync/cp/scratch/r2-ln-level/report-partial.md`). Not reviewed by the main thread beyond its summary; model-side runs were still running on bings-mac (see its section 3). Mac outputs: `artifacts/r2-ln-level-20261007/`.

Status: **partial**. The main session closed before the model-side runs finished. Everything below
marked **[measured]** comes from finished jobs on bings-mac. Anything about the model's own behaviour
in the new runs is **not yet measured**: the jobs are still running and the analysis script has not
been run on their output. Inferences are marked **[inferred]**. No code was changed, nothing was trained,
and nothing was written under `artifacts/r2-runs/`.

## 0. Summary so far

- **Mechanism 1: skeleton informativeness [measured, strong].** The head skeleton predicts little of
  a chart's LN level.
  - Whole-song features: R² 0.17 (MLP; ridge 0.13), correlation 0.42, AUC 0.73 for LN share ≥ 0.3.
    Fit on 11,368 fit_train charts, scored on 1,163 fit_dev charts.
  - Star band alone: R² 0.00.
  - Among skeleton neighbours, the LN share still has SD 0.158, against 0.186 over all charts.
  - So about 83 % of the variance in LN level is the mapper's choice. A hint-free model has to draw
    it, not predict it.
  - What an ideal hint-free sampler would score (**[inferred]**, from the R² above):
    - correlation of a run's level with the source chart's level: about R² ≈ 0.17 (not √R²);
    - correlation with the skeleton-predicted level: about 0.42;
    - SD ratio against the sources: 1.
  - Phase N's measured r with the source (0.00 at 48M, −0.16 at 64M) is therefore below an ideal
    sampler's 0.17, but the gap is small at n = 16. The real failure is the spread: SD ratio 0.43
    at 48M.
- **The model's local inputs already carry the skeleton information [measured].**
  - The features one decision sees (next 16 gaps, densities 1-32 beats ahead, the previous ≤ 511 row
    gaps) predict the whole-song share with R² 0.14 at one decision. Averaged over 8 decisions per
    chart, R² is 0.17, the same as whole-song features.
  - So a whole-skeleton summary input would add almost no information (**[inferred]**, strong).
- **The real level is a per-chart constant plus short local noise [measured, strong].**
  - Within a chart, the autocorrelation of LN share in 64-row blocks is 0.37 at a lag of 64 rows,
    0.06 at 128, and about 0 beyond (chart-demeaned).
  - Across charts it stays at 0.60-0.64 from 256 to 1,024 rows.
  - Real charts keep their level:
    - correlation between the first and last thirds 0.86, slope 0.92;
    - mean |last − first| 0.069, mean drift +0.008.
- **Memory beyond 511 rows helps, but modestly [measured].**
  - Next-256-row LN share predicted from the last 512 rows: R² 0.655. Adding rows [k−2048, k−512)
    raises it to 0.703 (49,742 positions with k ≥ 1,024).
  - In the joint fit, the two windows get about equal weight (0.46 and 0.52).
  - On positions with k ≥ 2,048, the oracle whole-song share reaches 0.65, against 0.53 for the
    last 512 rows alone.
  - The 511-row receptive field can carry the level; long memory would sharpen it by about 5 R²
    points.
- **History is the dominant source of level information [measured].**
  - Local skeleton alone predicts the next 256 rows' share with R² 0.13.
  - Adding the LN share of the last 511 rows raises that to 0.66.
  - So the decisive question is how the model reads its own history: mechanisms 2 and 3. Their
    model-side measurements are running.
- **The distribution is not cleanly bimodal [measured].**
  - Of 12,531 charts, 20 % have LN share < 0.025 (8.5 % exactly 0) and 28 % < 0.05. The median is
    0.129, 21 % are ≥ 0.3 and 6.5 % are ≥ 0.5.
  - The shape is a near-zero spike plus a broad decaying component, not two modes.
  - In band 5 about half the charts are in the < 0.05 bin and the rest spread flat to 1. That is the
    "LN-light or LN-heavy" picture, and the three failing panel charts belong to its near-zero part.
  - Band means are all 0.17-0.19.
- **Mechanisms 2-4 (model side): not yet measured.** The runs finished or in progress are listed in
  §3, and `analyze_runs.py` computes every table the brief asks for (§4).

## 1. Mechanism 1: skeleton informativeness [measured]

Fit on fit_train, scored on fit_dev (1,163 charts). Target: whole-song LN share = LN heads / heads. The
48 whole-song features are head times and grid only: gap quantiles in ms and beats, densities, snap
classes, BPM, breaks, and the dense-row fraction.

| Predictor | fit_dev R² (MLP) | R² (ridge) | corr | AUC, share ≥ 0.3 |
| --- | ---: | ---: | ---: | ---: |
| whole-song skeleton | 0.17 | 0.13 | 0.42 | 0.73 |
| skeleton + band one-hot (reference only: band depends on the arrangement) | 0.22 | 0.17 | 0.47 | 0.74 |
| band only | 0.00 | 0.00 | 0.00 | 0.50 |
| density only (rows/s, rows/beat, dense-row fraction, median gap) | 0.11 | 0.03 | 0.33 | 0.67 |
| gap quantiles only | 0.13 | 0.06 | 0.36 | 0.69 |
| local view at one decision → whole-song share | 0.14 | 0.06 | 0.38 | 0.70 |
| same, averaged over 8 decisions per chart | 0.17 | – | – | – |
| local view → next-256-row share | 0.13 | 0.06 | 0.36 | 0.69 |
| whole-song skeleton → next-256-row share | 0.12 | 0.09 | 0.35 | 0.68 |
| local view + LN share of the last 511 rows → next-256-row share | **0.66** | 0.63 | 0.81 | 0.92 |

- **Within band.** MLP R² by band 2-5: 0.10 / 0.20 / 0.21 / 0.16.
- **Strongest univariate correlates.** Rows per second −0.20, gap quantiles +0.15 to +0.20, density
  −0.17 to −0.18. Denser skeletons have fewer LNs, but weakly.
- **Conditional spread.** Among the 200 fit_train charts with the nearest MLP prediction, the LN share
  has a mean SD of 0.158, against a marginal SD of 0.186.
  - The PIT of fit_dev sources inside these neighbourhoods is uniform: SD 0.291 against 0.289 ideal,
    deciles 95-141 per 116.
  - So the neighbour distributions are a calibrated "comparable skeleton cell" reference. They are
    stored per fit_dev chart in `data/skel_pred.parquet`.
- **Strength.** Strong for the bound: a held-out split of 1,163 charts and two model families. The MLP
  is small and early-stopped, so a stronger learner might reach R² of about 0.2-0.25 (**[inferred]**).
  That would not change the conclusion that most of the level is the mapper's choice.

Does the model's BOS level track the skeleton-predicted level? **Not yet measured.** `analyze_runs.py`
computes it (`m1`: corr(run, pred) against corr(source, pred), PIT of runs, the zero-mode and heavy
fractions against neighbour expectations, and per-chart detail for the 16 panel charts).

## 2. Mechanism 2: commitment and memory, data side [measured]

**Persistence by history length.** Pooled over charts, the target is the LN share of the next 256 rows,
regressed on the share of the last N rows (`data/fitskel.json`, `persistence_*`).

| History | all eligible positions: n / R² / slope | positions with k ≥ 2,048 (15,744 positions, 980 charts): R² |
| --- | --- | ---: |
| last 64 rows | 174,519 / 0.60 / 0.70 | 0.54 |
| last 256 | 138,466 / 0.68 / 0.85 | 0.53 |
| last 512 | 99,096 / 0.71 / 0.91 | 0.53 |
| last 1,024 | 49,742 / 0.69 / 0.95 | 0.55 |
| last 2,048 | 15,744 / 0.58 / 0.93 | 0.58 |
| last 512 + rows [k−2048, k−512) | – | 0.59 (coefficients 0.42 / 0.51) |
| last 512 + whole history so far | – | 0.60 |
| oracle whole-song share | – | 0.65 |

On positions with k ≥ 1,024 (49,742), the last 512 rows give R² 0.655. Adding rows [k−2048, k−512)
gives 0.703.

**Block autocorrelation of LN share** (64-row blocks with ≥ 20 heads):

| Lag (rows) | 64 | 128 | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled | 0.80 | 0.70 | 0.64 | 0.63 | 0.60 | 0.47 |
| within chart (demeaned) | 0.37 | 0.06 | −0.08 | −0.06 | −0.08 | −0.08 |

The small negative within-chart values are the demeaning bias.

**Thirds, all 12,531 charts.**
- Correlation between the first and last thirds 0.86, slope 0.92; first third against the rest 0.89,
  slope 0.90.
- Mean |last − first| 0.069, mean last − first +0.008.
- Persistence rises with band: first third against the rest 0.85 (band 2), 0.89, 0.90, 0.94 (band 5).

**Reading [inferred].**
- In real charts the LN level is a chart constant set at the start. Local fluctuation around it
  decorrelates within about 128 rows.
- A model with a calibrated response to its last 511 rows could therefore keep a level almost as well
  as the data. The receptive field limits precision (about 5 R² points), not the ability to commit.
- If free runs compress or drift, the cause has to lie in how the model responds to history (gain), or
  in its own history differing from real history. It does not lie mainly in the 511-row limit.
- Strength: moderate. The data side is strong; the model side is still to be measured.

## 3. Jobs and outputs

| Job id (`ens`) | What | State at writing | Output (mac, `artifacts/r2-ln-level-20261007/`) |
| --- | --- | --- | --- |
| `20261007-124639-lnlev-envcheck` | environment check (no sklearn; scipy, torch present) | done | log only |
| `20261007-125041-lnlev-selftest` | `lnlib.decode` and `chart_stats` against `features.derive` and the earlier census, 30 charts | done; all agree (the 5 "mismatches" are charts without holds, where the census stores NaN) | log only |
| `20261007-125139-lnlev-skeleton` | per-chart skeleton features, LN stats, 64-row blocks, local views; 12,531 charts in 89 s | done | `data/charts.parquet`, `blocks.parquet`, `local.parquet` |
| `20261007-125526-lnlev-fitskel` | §1, §2 | done | `data/fitskel.json`, `level_distribution.png` (both mirrored), `skel_pred.parquet`, `skel_pred_train.parquet`, `persistence_rows.parquet`, `thirds.parquet` |
| `20261007-125322-lnlev-run-smoke`, `...-125520-lnlev-run-smoke2` | pipeline check (2 charts) | done | `runs-smoke/smoke/` |
| `20261007-130714/…-analyze-smoke`, `-smoke2`, `-smoke3` | analysis pipeline check | done (the first two failed on bugs that are now fixed) | `analysis-smoke.json` (meaningless, 2 charts) |
| **`20261007-125618-lnlev-run-n48`** | 48M: TF on T (216 charts), BOS free runs F (116 charts × 3 seeds), prefix runs, counterfactual prefixes + free continuations | **running** (in the `cf`/`cffree` stage; `tf`, `fr` and `pre` done) | `runs/n48/` |
| **`20261007-130636-lnlev-run-n64`** | same at 64M | **running** (in the `pre` stage; `tf` and `fr` done) | `runs/n64/` |

- **Check.** Teacher-forced marginals on a stored free run equal the sampling-time marginals of the
  earlier probe to within 1.2e-6 (3 runs). So the TF pass on generated charts measures exactly what
  the sampler drew from.
- **Sampler throughput.** 150-350 decisions/s per process, falling for long charts. `continue_chart`
  re-derives the whole history at every step.

Do not kill the two running jobs; their outputs are needed for §4.

**Chart sets** (`runs/<tag>/charts.json`):
- F = the 16-chart selection panel + 25 fit_dev charts per band 2-5 with 300 ≤ K ≤ 2,500 (seed 17).
- T = F + 25 more per band with 200 ≤ K ≤ 3,000.

**Scripts** (control plane `~/ensomi/.sync/cp/scratch/r2-ln-level/`; mac `../.sync/cp/scratch/r2-ln-level/`):
- `lnlib.py`: vectorised decode, holds, near-head, per-chart stats;
- `skeleton.py`: data tables;
- `fitskel.py`: §1-2;
- `lnrun.py`: model runs;
- `analyze_runs.py`: all model-side tables;
- `dataq.py`: zero-mode by band and density, skeleton AUC for the zero mode, how early the level is
  revealed. Written, not run;
- `selftest.py`, `envcheck.py`.

## 4. Unfinished: what to run next, and what each output answers

Run once both jobs have exited (`ens ps`), at most two at a time. Use the frozen `PYTHONPATH`:

```
PYTHONPATH=artifacts/r2-runs/r2-phaseN-20261006/code .venv/bin/python ../.sync/cp/scratch/r2-ln-level/analyze_runs.py artifacts/r2-ln-level-20261007 n48
... same with n64
.venv/bin/python ../.sync/cp/scratch/r2-ln-level/dataq.py artifacts/r2-ln-level-20261007/data
```

These write `analysis-<tag>.json` and `.png` (mirrored), plus `runs/<tag>/runs.parquet`, `states.parquet`,
`cffree.parquet` and `data/dataq.json`. Keys and the question each answers:

- **`m1`: Mechanism 1 and criterion (a).**
  - Run against the source and against the skeleton prediction, compared with the source against the
    prediction (the ideal).
  - SD of runs against sources, and between-chart against within-chart (across seeds) SD.
  - PIT of each run in its chart's skeleton-neighbour distribution: compressed if the SD is below
    0.289 and mass sits mid-range.
  - Fractions in the zero mode (< 0.025) and heavy (≥ 0.3): runs against sources against what the
    neighbours predict.
  - The 16 panel charts' predictions against their sources. Is a dense, LN-light 5★ source in the low
    tail of its skeleton cell?
- **`m2_runs`: persistence within runs (b).** Correlation and slope of the last third on the first in
  runs against sources; decile shares; prefix runs (continuation against the source prefix, compared
  with the source's own continuation).
- **`m2_response`: gain and slope, on real history.**
  - Model P(LN | share of the last 64, 256 or 511 rows) against the data's empirical rate, binned and
    as fixed-effect slopes (band × next-gap bin).
  - The `beyond` term: the data's slope on rows [k−1023, k−511) against the model's, which should be 0.
  - A dense/non-dense split of history LNs.
  - `early`: the commitment phase, k < 128 or 256, and the spread of the first 128 and 256 rows'
    share in runs against sources.
- **`gain_map`.** Expected next-row LN share against the history share, model on real history against
  data against model on own history, with a linear fixed point. This is the drift mechanism.
- **`m3_matched`: Mechanism 3.**
  - Own history against real history P(LN) at matched history-share bin × band × next-gap bin, also
    matched on the dense fraction of history LNs.
  - By position (first against last third).
- **`m3_counterfactual`: Mechanism 3 probes.** Teacher-forced response over the next 64 and 256 rows
  after the boundary b, under these prefixes:
  - own against source prefix;
  - own prefix with dense-born LNs stripped, against the same number of non-dense LNs stripped;
  - source prefix plus m dense-row short LNs, against plus m non-dense LNs, at equal Δshare;
  - a chart-fixed-effect regression on rf_share and own_made.
- **`m3_cffree`.** Realised LN share of the free continuation from b under: the source prefix, the own
  prefix with dense LNs stripped, the source prefix with dense LNs added, and the original own run.
- **`m4`: Mechanism 4.** Placement at matched LN-share bin × band group, BOS runs against fit_dev
  sources:
  - LN rate at dense and non-dense free lanes;
  - short holds per LN; near-head releases per LN;
  - a regression for the run effect on log(dense / non-dense rate) controlling for log share,
    dense-row fraction and band;
  - per panel band-5 chart, the run against band-5 fit_dev charts at the run's level.

## 5. Remedies: provisional ranking from the data-side evidence only

The model-side measurements (§4) can still reorder this.

1. **A long-horizon own-history objective on the LN level.** A prefix-rollout calibration term:
   - start from a real prefix and roll out 256-512 rows;
   - penalise the gap between the expected LN share on the rollout and the source continuation's;
   - the machinery exists as `train_ce.relaxed_proxy` for difficulty.

   The target is per chart and hint-free, because the real prefix supplies the level and data
   persistence (r 0.89) says the continuation should keep it. It acts on drift and on
   self-misidentification.
   - Cost [inferred]: about +85 % wall from scratch (about 18 h for 64M), or a 8-16M fine-tune from
     48M in about 3-5 h.
   - Confirms: own-history LN-birth rate within ±15 % of the same chart's teacher-forced rate, and
     |drift| ≤ 0.02.
   - Refutes: drift fixed but the BOS level still compressed (PIT SD well below 0.29).
2. **A distribution-level calibration of the BOS draw.** Without a per-chart target, compare the batch
   of rollout levels from BOS with the skeleton-conditional neighbour distribution (moment or quantile
   matching on PIT). It targets under-commitment (a). Cost as in remedy 1. **Still unmeasured: whether
   the BOS draw is in fact compressed or shifted relative to the skeleton cell** (`m1` PIT).
3. **A per-chart latent level, learned without labels.**
   - A VAE-style latent: the posterior reads the whole chart in training; a prior p(z | skeleton) is
     sampled at generation.
   - Generation is hint-free: the model draws its own level. In training, though, the latent is a
     learned summary of the target chart, a label-free cousin of the LN-level input. Whether it
     "counts" as a hint is the human's call.
   - It gives exact commitment by construction.
   - Highest cost and risk (KL tuning, posterior collapse): a new module and a retrain of about 10 h
     or more.
4. **Long-range memory on (landmarks).** The data shows +5 R² points from history beyond 511 rows,
   with equal weights.
   - Earlier evidence: R2 v1's landmark readout was unused at 39-61M (lesion ΔNLL ≈ 0), and v1 drifted
     more (2.4×). That is confounded with conditions and lr.
   - Cost: a fine-tune from 48M with the zero-initialised readout, about 1-2 h for 8-16M exposures.
     A retrain is about 10 h.
   - Expected to help criterion (b) a little. It does not address (a). If own history is misread
     (mechanism 3), it could lock in the wrong level more firmly [inferred].
5. **Whole-skeleton summary input.** Not supported: the local view already carries the same
   skeleton information (R² 0.14-0.17 against 0.17).
6. **Longer windows or more BOS windows.** Training already encodes the full prefix (`window_hands`
   encodes from row 0), so window length changes only which decisions are scored. More BOS weight
   helps only if early-song calibration is off, which `m2_response.early` will show. Not supported yet.
7. **Loss reweighting.** No evidence yet.

**The oracle-level probe** (the Astra-built `ln_level` fine-tune, oracle mode, used once). What it
would add beyond these measurements [inferred]:
- An intervention test of criterion (c): whether placement within a level is right once the level is
  right. `m4` answers this only observationally, at matched level bins.
- Whether a level input, once given, is held over the song. That separates "cannot commit" from
  "commits to the wrong level".

If `m4` shows runs placing LNs on dense rows like real charts at the same level, and `m3` shows no
own-history misreading, the probe adds little. If either is ambiguous, it is the cleanest single test.

## 6. Known unknowns and caveats

- All model-side conclusions are pending (§4).
- The ideal-sampler numbers in §0 (r ≈ R², SD ratio 1) assume the level is well described by
  skeleton prediction plus independent mapper choice [inferred].
- The skeleton predictor is a small MLP and ridge, with no gradient boosting (sklearn is not
  installed). Its R² is a lower bound on the skeleton's information.
- Persistence regressions are pooled OLS over overlapping windows, so their standard errors are
  understated. The R² differences are large relative to sampling noise at these n.
- One violation of the resource rule: three of my jobs ran at the same time for about one minute
  (a 2-chart smoke analysis started while both model runs were active).
