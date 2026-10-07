# Opus report: empirical diagnosis of R2 phase-N pattern collapse

Report of the Opus subagent (fresh context) for [r-collapse-agents](r2-collapse-20261007.md#r-collapse-agents), copied verbatim between the markers. It was returned as text. Main-thread check: `artifacts/r2-collapse-20261007/opus/analyze.log` (mirrored) matches the stay and entry rates of D1, 0.534 / 0.57 against 0.237 for stay and 0.059 / 0.069 against 0.059 for entry. The other tables were not rechecked.

----- BEGIN report -----

## Opus report: empirical diagnosis of R2 phase-N pattern collapse

**Bottom line.** On average, natural-model collapse is not a late-song slide into repeating loops. It is a chart-level failure:
- the chart's regime is set wrong at the start and then confirmed by the model's own rows;
- the regime then wanders, because nothing holds it in place;
- degenerate stretches start as often as in real charts but last about twice as long.

LN-level wandering is one coordinate of this. In band 2 it shows as "all four lanes busy" in 25–40 % of windows. That is the natural-generation version of the human's "full 4 lane" call.

Scripts are in `~/ensomi/.sync/cp/scratch/r2-collapse/opus/`. Mac outputs are in `artifacts/r2-collapse-20261007/opus/`. Logs: `analyze.log`, `memory.log`, `bandpos.log`, `calib.log`, `ck_agg.log`, `startpre.log`, `skelr2.log`. Tables: `windows-flagged.parquet`, `whole.csv`, `thresholds.csv`. Process slip: twice, a roughly 20 s pandas job overlapped two sampling jobs, so three processes ran for a moment.

### 1. Formal problem statement

**Window statistics.** Charts are cut into non-overlapping windows of 64 head rows. The statistics per window:
- heads per row `nh`;
- rate of rows with 3 or more heads `c3`, and of 4-head rows `c4`;
- jack rate, and fast-jack rate (gap under 100 ms);
- all-lanes-busy rate `bus4` (every lane has a head or is held);
- held share (the LN level);
- head-pattern entropy `pent`, and the distinct 4-gram ratio `ng4`;
- repeat rate, and loop rate (best period from 2 to 16 rows);
- top lane share and top hand share;
- lock runs: one lane with a head in every row (`hlock`), or one lane busy in every row (`lock`).

Each run is compared with its own source on the same skeleton, and with a per-band envelope built from 116 fit_dev sources.

**Pattern collapse, as four computable quantities:**
- **C1, envelope exits.** The share of windows with any statistic beyond its band's 1 % / 99 % source quantile ("degenerate windows"). Sources come out at about 6–9 % by construction.
- **C2, absorption.** P(degenerate next window | degenerate now), the "stay" rate, and P(degenerate next | not now), the "entry" rate.
- **C3, loss of chart identity.**
  - Per-chart correlation between the first and last thirds.
  - Between-chart and within-chart SD ratios.
  - Autocorrelation of window deviations from the run's own mean, at lags of 1–8 windows.
  - How much of a real prefix's identity survives in the continuation.
- **C4, regime bias.** The paired run-minus-source offset per band.

**LN-level wandering is one instance of this, not a separate problem [measured].** Held share shows the same signature as the pattern statistics. First-to-last-third correlation:

| Statistic | Runs | Sources |
|---|---|---|
| held | 0.59–0.64 | 0.91 |
| nh | 0.57–0.58 | 0.85 |
| c3 | 0.55–0.56 | 0.83 |
| jack | 0.49–0.51 | 0.80 |
| pent | 0.53–0.59 | 0.82 |

Held is the strongest instance at 64M: within-chart SD is 1.48× the sources', and lag-2 autocorrelation is 0.37 against −0.04.

**Proposed target, "playable and stable over a whole song"** (my thresholds, not ratified). On at least 100 fit_dev skeletons × 3 seeds, in every band and every fifth of the song:
- (T1) degenerate-window rate at most 1.25× the source rate;
- (T2) stay rate at most 0.35 (sources 0.24);
- (T3) first-to-last-third correlation at least 0.75 for nh, c3, held and pent;
- (T4) band offset within ±0.05 heads per row and ±0.03 held share;
- (T5) no head-lock run of 30 rows or more, and band-2 `bus4` exits at most 2× the source rate;
- plus the human's blind playability screen on a fixed panel.

### 2. Causal map

**Mechanism [inferred from measured parts].** Teacher-forced CE learns the next row given the last 511 rows and local skeleton features. That prediction already averages over the chart's style z (density, chord habit, LN level, pattern vocabulary). The step is calibrated on real histories [measured: expected heads per row within ±0.05 of realised, every band × position]. In free generation, z is never drawn once. The model keeps re-reading it from its own last 511 rows. This gives two failure modes:
- **(M1) Start.** At the beginning of the song the guess about z is the average chart. The model's early rows look average, and it then treats them as the chart's identity.
- **(M2) Wander.** Chance excursions get read as style and kept. There is no pull back toward the chart's original identity.

| Link | Evidence for | Evidence against | Strength |
|---|---|---|---|
| **Objective** (uniform per-decision CE on real histories) | Per-step calibration is exact while whole-run statistics are off. Entry rate matches the sources (0.059–0.069 against 0.059); stay rate doubles (0.53–0.57 against 0.24). | none | strong that CE is satisfied while rollout fails |
| **No chart-level variable on the information path** (no z input, `memory: none`, no band input) | Regression to the corpus mean by band (§3 D3). Runs follow local timing more than real charts do (D6). A run's chart mean barely follows its own source: nh correlation 0.14–0.22, c3 about 0, held about 0. | none | strong |
| **Skeleton alone cannot supply z** | A crude 11-feature whole-skeleton summary (leave-one-out ridge) predicts little of the sources' chart means: R² nh 0.14, c3 0.05, held 0.03, pent 0.20. Runs follow the skeleton more than sources do (loop R² 0.40–0.61 against 0.08). | Features are crude; the LN work's richer skeleton model did better for LN length. | moderate |
| **511-row receptive field** | After a real prefix, identity decays mostly past 512 rows. At 48M the nh correlation goes 0.56 → 0.45 → 0.35, and c3 0.46 → 0.11. The mean absolute distance of nh from the source rises to the from-start level by about 1k rows. | 64M keeps held identity past 1k rows (0.72 → 0.74). The band-2 offset is set by row 64 and stays flat, so the receptive field cannot cause it. | moderate; limits how long identity lasts, does not set the start regime |
| **Self-conditioning on own rows**, as opposed to misreading out-of-distribution history | A 64-row real prefix removes about two thirds of the band-2 excess, for the whole song (§3 D7). Degenerate stretches are absorbing. The LN work found own history is read like real history, not misread. | Part of persistence follows the skeleton: same chart, same window, seed B is degenerate given seed A 0.31–0.36 of the time, against a 0.11 base rate. | strong for self-conditioning; against the narrow "out-of-distribution history" reading |
| **Training windows** (256 rows, 1/8 from the song start) | none | Calibration at rows 0–64 on real histories is fine (band 2: expected 1.29 against realised 1.26 heads per row). | weak / not supported |
| **Exposure** (more training) | none that it helps | Free-run statistics swing from 16M to 64M with no trend. 48M → 64M gets worse: degenerate windows 0.11 → 0.14; busy-lock runs of 30+ rows 14 → 70 of 348. | strong that exposure does not fix it (32 charts × 1 seed per checkpoint, plus 116 × 3 at 48M and 64M) |
| **Row-head factorisation** | Chord statistics are the most compressed between charts: c4 SD ratio 0.41–0.52, c3 0.69–0.70. | Not separable from the missing chart variable with these data. | weak |
| **Ancestral sampling at T = 1** | Not tested here (Astra's lens). | | — |

### 3. Diagnostics run

All [measured]. Free runs are from the start of the song (BOS) unless marked as prefix runs. Data: n48 and n64 (116 charts × 3 seeds), plus new runs.

**D1. Envelope exits and absorption.**
- Degenerate-window rate: 0.11 (48M) and 0.14 (64M), against 0.08 in the sources. It rises along the song at 64M: 0.12 → 0.15 by fifths.
- Stay rate 0.53 / 0.57 against 0.24. Entry rate 0.059 / 0.069 against 0.059.
- Sustained degeneracy (two windows in a row): 26 % / 32 % of runs, against 17 % of sources.
- Onset row quartiles 64/256/656 (48M) and 0/192/640 (64M). A quarter of runs are degenerate from the start.

**D2. By band.** Degenerate-window rate in band 2: 0.25 (48M) and 0.40 (64M), against 0.085 in the sources. Inside band 2:
- `bus4` exits in 16 % / 31 % of windows, against 1.3 %; 4-chord exits 7–11 %, against 0 %;
- 23–28 of the 31 band-2 charts are affected;
- in the flagged windows, held share is 0.27–0.35 against 0.077 in the sources, so LN pile-up plus extra chords drives it.

Bands 4–5 go the other way: sparse and low-diversity. At 48M in band 5, low-4-gram exits are 8 % and loop exits 6.5 %; late in the song the loop rate is +0.087 above source.

**D3. Regression to the mean, set early [measured].**

| Band | Paired d_nh, rows 0–64 | Rows 64–192 onward |
|---|---|---|
| 2 | +0.14 | +0.15, then flat |
| 3 | about 0 | about 0 |
| 4 | −0.12 | −0.20 to −0.22, then flat |
| 5 | −0.07 | −0.20 to −0.22, then flat |

The offset is in place by about 192 rows and does not grow afterwards.

**D4. Identity loss.**
- Between-chart SD ratio, runs to sources: nh 0.74–0.80, c3 0.69–0.70, c4 0.41–0.52.
- Within-chart SD ratio: nh 1.06–1.09; held 1.09–1.48.
- Window autocorrelation at lags 2 / 4, held share: 0.26–0.37 / 0.06–0.17 in runs, against −0.04 / −0.10 in sources. Real charts swing back toward their own mean; runs do not.

**D5. A real prefix fades.** Prefix runs continue from a third of the way in. Correlation with the source prefix's identity, by rows since the prefix ends (0–128 … 1k+):
- nh: 0.63 → 0.35 (48M) and 0.70 → 0.50 (64M);
- the source itself holds 0.78–0.92;
- runs from the song start reach only 0.15–0.20 in the early windows.

**D6. Runs follow local timing more than real charts do.** Correlation with log median row gap, runs against sources: loop −0.51 against −0.23, pent 0.46 against 0.30, nh 0.42 against 0.31. Window values correlate more between two seeds than between a seed and the source: nh 0.40 against 0.17, loop 0.67 against 0.36.

**D7. Start-regime test (new runs; 56M, seed 954; band 2 has 31 charts, band 5 has 28).** Paired d_nh by rows since the song start:

| Band | Start | 0–256 | 256–512 | 512–1k |
|---|---|---|---|---|
| 2 | from row 0 | +0.31 | +0.25 | +0.31 |
| 2 | 64 real rows first | +0.10 | +0.11 | +0.10 |
| 5 | from row 0 | −0.12 | −0.17 | −0.15 |
| 5 | 64 real rows first | −0.05 | −0.07 | −0.14 |
| 5 | 256 real rows first | 0 (prefix) | −0.03 | −0.12 |

In band 2, `bus4` exits drop from 0.15 to 0.03–0.065 with the 64-row prefix. So band 2 is mostly a start problem, and the fix lasts the whole song. Band 5 drifts back to the sparse regime past about 512 rows.

**D8. Per checkpoint (new runs; 32 charts × seed 954; 16M–64M).**
- Degenerate rate 0.08–0.18, with no trend; sources 0.06.
- 24M falls into a sparse, looping regime: nh 1.19 against 1.46, loop 0.32 against 0.26, band-5 pent 2.20 against 2.85.
- 40M band 2: degenerate rate 0.67, `bus4` exits 0.50.
- First-to-last-third correlation for nh: 0.62–0.73 against 0.80.
- So the global regime flips between checkpoints while NLL stays flat.

**D9. Own-run model entropy.**
- 48M: falls from 2.16 to 1.67 nats between rows 0–128 and 1k–2k. Teacher-forced on sources it stays at 2.0–2.15. The model grows more confident on its own output, and loops rise.
- 64M: stays at 2.18–2.31 and becomes more diffuse (4-gram ratio +0.04).

**D10. R1-style fixed-lane runs are not R2's failure.**
- Head-lock of 30+ rows: 1/348 runs (48M) and 4/348 (64M), against 0/116 sources.
- Busy-lock of 30+ rows: 4 % (48M) and 20 % (64M) of runs, against 15 % of sources. The longest is 395 rows at 64M, against 84: long holds.

**D11. What comes before onset (weak).** At 64M, held share and `bus4` rise in the window before onset (held 0.175 against a 0.138 mean). At 48M the window before is thinner and less varied (nh 1.30 against 1.36).

### 4. Core questions (ranked)

1. Is the band regime set in the first rows and then self-confirmed, or produced by drift? D7 suggests the answer depends on band. E1 settles it.
2. Can the skeleton identify z, or must z be drawn per chart? E4.
3. Given a fixed per-chart z, does a CE-trained model hold it for the whole song? E3. This is the main question.
4. Is absorption self-conditioning on the model's own recent rows, or driven by the skeleton? E2.
5. Does sampling temperature change absorption? E5, low priority; Astra may cover it.

### 5. Experiments

**E1. Start against drift (no training).**
- Question: question 1.
- Manipulation: start from a real prefix of 16 or 64 rows instead of the song start. All bands, 116 F charts × 3 seeds, 56M.
- Control: runs from the song start on the same skeletons and seeds.
- Metric: per band at rows 256–512 and 512–1k, paired d_nh, d_held and `bus4`-exit rate.
- Pass ("start-dominant"): the 64-row prefix removes at least 50 % of |d_nh| at 256–512, and at least 50 % still remains removed at 512–1k.
- Fail ("drift-dominant"): less than 25 % removed at 512–1k.
- Cost: about 25 min of Mac time on 2 processes.
- Next if start-dominant: draw z once at the start (a prior draw) and hold it. If drift-dominant: anchoring along the whole song is needed (a z input, or memory). D7 already gives band 2 pass, band 5 fail.

**E2. Self-conditioning against skeleton (no training).**
- Question: question 4.
- Manipulation: generate in 128-row segments. Before each segment, replace history older than 64 rows with the source's rows, splicing only where no lane is held. A scratch wrapper around `continue_chart(..., stop=)` does this; no tracked edits.
- Control: plain runs from the song start, same seeds.
- Metric: stay rate, first-to-last-third correlation, band offset.
- Pass: stay rate at most 0.35 and |d_nh| at most 0.05 per band. Then persistence comes from reading its own rows, and a held z or a sequence-level objective is the lever.
- Fail: stay rate at least 0.45. Then the map from local timing to pattern is at fault, and the feature and row-head path is next.
- Cost: about 1 h of code plus 20 min of Mac time.

**E3. Oracle chart-latent fine-tune (training; above today's 15-minute budget).**
- Question: question 3.
- Manipulation: fine-tune 56M with a standardised per-chart z through the existing FiLM gate. z holds whole-chart nh, c3, jack, held share, median LN length, pent, loop and hand balance. 8M exposures.
- Control: the identical fine-tune with z zeroed, so extra training is controlled for.
- Evaluation: 116 × 3 runs from the song start with the source's z.
- Metric: first-to-last-third correlation (nh, c3, held, pent), stay rate, band-2 `bus4` exits, correlation between run and source chart means.
- Pass: first-to-last-third correlation at least 0.75 on all four, stay rate at most 0.35, and run–source correlation at least 0.7 for nh and held.
- Fail: first-to-last-third correlation below 0.65, or run–source correlation below 0.4.
- Cost: about 1.2 h per arm × 2, plus about 50 min of evaluation.
- Next on pass: collapse is a missing-latent problem; build a prior over z (E4), and conditions become coordinates of z. On fail: CE does not make the model use a global input over its history, so move to a sequence-level or rollout objective.
- This generalises today's `ln_level` fine-tune. The difference is that it is judged on free-run stability, not NLL.

**E4. Can the skeleton identify z? (data only).**
- Question: question 2.
- Manipulation: richer skeleton features (the LN work's `fitskel.py`) on fit_train, per band.
- Control: the crude 11-feature R² from D-skel: nh 0.14, c3 0.05, held 0.03, pent 0.20.
- Pass/fail: R² below 0.3 means z must be sampled per chart and committed to at the start. R² above 0.6 means a whole-song skeleton summary input suffices.
- Cost: about 10 min of Mac time.

**E5. Temperature (no training).**
- Manipulation: T = 0.8 against T = 1, 116 × 1 runs.
- Pass: stay rate at most 0.35 without loop or repeat exits rising above the source rate.
- Expected: fail [inferred]. A sharper sampler cannot add an anchor, and 48M already sharpens into loops.
- Cost: 20 min.

### 6. The control problem

LN share and length, difficulty and the five style concepts are all coordinates of z. Today the model reads any such quantity only through its last 511 rows, so a request would hold for roughly 128–256 rows and then wander. A positive tilt pushes runs into the absorbing band-2 regime, which matches the C0 screen's "full 4 lane".
- If E3 passes, control means training the z input with prior draws. Obedience must be scored on whole-song free runs (first-to-last-third correlation, stay rate), not on teacher-forced response.
- If E3 fails, no condition head will be stable. Hold all condition work until the primary target (T1–T5) passes.

### 7. What to stop or not do

- **Do not pick checkpoints by teacher-forced NLL or guard pass/fail as a stability proxy.** Free-run regimes flip between checkpoints (24M sparse and looping, 40M band-2 collapse). Use C1–C4 on at least 100 charts × 3 seeds.
- **Do not train longer expecting rollout to improve.** 48M → 64M is worse on C1, C2 and lock runs.
- **Do not fix LN level, chord density and pattern entropy one at a time.** They share one signature. Keep the `ln_level`-only fine-tune stopped.
- **Do not decide on 16-chart, single-seed panels.** Band effects need about 25 charts per band.
- **Do not expect landmark memory alone to fix it.** The band-2 regime is set by row 64, and v1's landmark readout went unused under CE.
- **Do not run further difficulty-measure testing** until the primary target passes.

----- END report -----
