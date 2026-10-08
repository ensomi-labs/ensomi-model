# What the supplied head rows carry, and whether the separation of concerns holds (opus-skeleton, 2026-10-08)

Report of the fresh Opus subagent "opus-skeleton" in the representation round ([r-represent-relaunch](r2-representation-20261008.md#r-represent-relaunch)), returned 2026-10-08 about 03:40 UTC. Brief: `~/ensomi/.sync/cp/scratch/r2-represent/brief-opus-skeleton.md`. About 18 minutes of mac time at ≤ 2 threads. Scripts in `~/ensomi/.sync/cp/scratch/r2-represent/opus-skeleton/`; outputs on bings-mac in `artifacts/r2-represent-20261008/opus-skeleton/` (json, csv, log mirrored). Saved as returned, condensed in layout only, below the main thread's checks.

<a id="v-skeleton-checks"></a>**Main-thread checks (2026-10-08).**
- **Checked against the mirrored outputs:**
  - `rowlevel2.json`: human pairs at F1 ≥ 0.95 give chord r 0.767 and LN-head r 0.637; rows-only on the same 300 charts gives 0.589 and 0.324; rows-only against source on the 99 panel charts gives 0.554 and 0.289.
  - `pairs2.json`: same-band ICC, non-duplicate pairs: nh 0.578, held 0.628, LN share 0.684, hand 0.174.
  - `fit.json`: band accuracy from RICH rows with HGB 0.699; star R² 0.813 from S48.
  - `copies.log`: rate edits and copies are concentrated below F1 0.8, which is why those buckets were set aside.
- **Caveats the main thread adds:**
  - The human-pair "ceiling" may include same-mapper set conventions. Mapper identity per pair was not checked; the `.osu` Creator field could separate it.
  - "R2 equals an ideal rows-only sampler" is an inference from the product of correlations, not a direct test.
  - Section 4 rests on 4 songs.

---

## The short answer

- **The rows fix rhythm, density and the between-part notes-per-second profile completely, and encode most of the difficulty.** Band from rows alone: 70 % accuracy (majority 31 %); star R² 0.83 [M].
- **The rows fix little of the arrangement's identity:** at most 41 % of chart-level chord rate, 21 % of LN share, 0 % of hand balance [M].
- **Where chords and LN heads fall row by row is mostly carried by something outside the rows.** Two human charts on near-identical rows agree at r 0.77 (chord size) and 0.64 (LN head) [M]. A rows-only predictor reaches 0.59 / 0.32 [M]. R2 reaches 0.30 / 0.08 against the source, which is what an ideal rows-only sampler would give [M numbers, I equivalence].
- <a id="o-skeleton-identity"></a>**R2 reproduces the rows-determined part of identity and replaces the rest with a draw.** Its chart θ correlates with a rows-only prediction (0.48-0.64) much more than with the source (0.12-0.41) [M].
- <a id="o-skeleton-shift"></a>**On rows from the legacy timing model, R2 loses the difficulty and LN level that human rows had been leaking to it.** Band-5 song: 3-row-chord rate 0.14 → 0.03-0.05 [M, 4 songs].

Data: 12,272 fit_train/fit_dev charts with K ≥ 256 (11,118 train, 1,154 dev). Tags: [M] measured, [I] inferred, [P] proposed.

## Pilot tables

**A. Chart level: R² on fit_dev of rows-only predictions.** S48: the 48 skeleton summaries from `fitskel.py`. RICH: S48 plus 284 row features (gap-class uni/bigrams, beat onset patterns, bar rhythm repetition, density profile and its autocorrelation, equal-gap runs, NPS, position profile). HGB: histogram gradient boosting. "Within" is the share of within-band variance explained. kNN-32: B3's donor-prior construction. "+band" adds the true band, for reference only.

| target | S48 ridge | S48 HGB | RICH HGB | RICH HGB within-band | kNN-32 (S48) | RICH+band HGB |
|---|---:|---:|---:|---:|---:|---:|
| nh (heads per row) | 0.23 | 0.37 | **0.41** | 0.33 | 0.26 | 0.66 |
| c3 (rows with ≥3 heads) | 0.16 | 0.29 | 0.31 | 0.14 | 0.20 | 0.58 |
| c4 | 0.03 | 0.06 | −0.02 | −0.09 | 0.08 | 0.29 |
| jack | 0.16 | 0.27 | 0.27 | 0.18 | 0.18 | 0.53 |
| held (LN level) | 0.14 | 0.19 | 0.22 | 0.20 | 0.11 | 0.31 |
| LN share | 0.14 | 0.17 | 0.21 | 0.21 | 0.11 | 0.25 |
| LN length (beats) | 0.25 | 0.21 | 0.23 | 0.15 | 0.23 | 0.24 |
| pattern entropy | 0.24 | 0.36 | 0.41 | 0.35 | 0.23 | 0.56 |
| loop | 0.14 | 0.20 | 0.27 | 0.24 | 0.11 | 0.31 |
| hand share | 0.00 | 0.00 | 0.00 | 0.00 | −0.01 | 0.00 |
| trill | 0.05 | 0.07 | 0.08 | 0.07 | 0.05 | 0.08 |
| roll | 0.21 | 0.27 | 0.31 | 0.30 | 0.21 | 0.39 |
| SD over 128-row windows of nh | 0.21 | 0.22 | 0.27 | 0.04 | 0.18 | 0.36 |
| corr(window nh, row density) | 0.07 | 0.09 | 0.17 | 0.16 | 0.07 | 0.17 |
| SD over windows of log notes/s | 0.69 | 0.71 | **0.77** | 0.72 | 0.59 | 0.78 |
| **band** (accuracy / macro F1 / within ±1) | 0.66 / 0.64 (logistic) | 0.68 / 0.66 / 0.99 | **0.70 / 0.67 / 0.99** | | | |
| **star** | | 0.81 | **0.83** | | | |

- Richer row features add 0.03-0.07 R² over S48; the ≤ 0.3 found by `fitskel.py` was not an artefact of weak features [M].
- kNN-32, the donor prior, recovers 0.11 of LN level and 0.26 of nh, about half of what HGB gets. It was knowable before the night that the prior is close to a draw from the band [M].

**B. Window level (128 rows), fit_dev windows; HGB on 30k fit_train windows.** "Oracle identity" = the chart's true whole-chart coordinates (slight leak for short charts).

| window target | within-chart variance share | chart rows | chart + window rows | oracle identity | identity + window rows | within-chart variance explained by the window's rows |
|---|---:|---:|---:|---:|---:|---:|
| nh | 0.31 | 0.27 | 0.35 | 0.67 | 0.80 | **0.41** |
| jack | 0.39 | 0.19 | 0.30 | 0.59 | 0.73 | **0.37** |
| pattern entropy | 0.35 | 0.26 | 0.34 | 0.63 | 0.74 | 0.29 |
| roll | 0.49 | 0.15 | 0.24 | 0.48 | 0.60 | 0.25 |
| loop | 0.51 | 0.12 | 0.22 | 0.47 | 0.55 | 0.16 |
| held | 0.22 | 0.17 | 0.20 | 0.77 | 0.80 | 0.14 |
| LN share | 0.25 | 0.15 | 0.16 | 0.74 | 0.77 | 0.11 |
| trill | 0.56 | 0.02 | 0.04 | 0.41 | 0.42 | 0.03 |
| hand | 0.81 | 0.00 | 0.00 | 0.13 | 0.14 | 0.01 |

The chart sets the level. Given the level, the local rows steer chord and jack modulation (about 40 % of within-chart variance), LN modulation weakly (11-14 %), and hand and trill not at all [M].

<a id="o-skeleton-placement"></a>**C. Row level: do charts on the same rows put chords and LN heads on the same rows?** Within-chart correlations over matched rows, median per chart. Human pairs: same song group, offset-aligned head F1 ≥ 0.95 at 20 ms; copies (lane-pattern agreement ≥ 0.9) excluded.

| comparison | chord size r | LN head r | exact lane pattern (chance) | up to mirror (chance) |
|---|---:|---:|---:|---:|
| Human vs human, same band (n = 615) | **0.79** | **0.65** | 0.21 (0.10) | 0.39 (0.19) |
| Human vs human, different band (n = 1,556) | 0.77 | 0.66 | 0.18 (0.09) | 0.34 (0.17) |
| Rows-only HGB vs truth, 300 of those human charts | 0.59 | 0.32 | | |
| Rows-only HGB vs source, 99 panel charts (±16 gaps: 0.55 / 0.29) | 0.56 | 0.30 | | |
| R2 B0 (56M) vs source, same rows | **0.30** | **0.08** | 0.15 (0.12) | 0.29 (0.24) |
| R2 B0 seed 954 vs seed 955 | 0.30 | 0.08 | 0.15 (0.12) | 0.29 (0.24) |
| B1 / B2 / B3 / B3-oracle / d0-env / d0-phi vs source (chord) | 0.30 / 0.27 / 0.29 / 0.26 / 0.31 / 0.31 | 0.08-0.10 | | |

- Two independent samples from a rows-only conditional would agree at about r² of its mean predictor: 0.56² ≈ 0.31 for chords, 0.29² ≈ 0.08 for LN heads. R2 sits exactly there [M numbers, I reading].
- Humans share much more than the local rows carry: information outside the local rows that two arrangers of the same song share (audio accents and sustains, song structure, or set conventions). Not separated [I].

**D. Chart-level agreement on the same rows, 99 panel charts (BOS runs, mean of 3 seeds, first 2,560 rows).** "Rows" = the RICH HGB prediction.

| coordinate | B0 gen vs source | B0 gen vs rows | source vs rows | B3-oracle gen vs source |
|---|---:|---:|---:|---:|
| nh | 0.23 | 0.48 | 0.56 | 0.55 |
| jack | 0.19 | 0.56 | 0.47 | 0.43 |
| held | 0.20 | 0.49 | 0.44 | 0.81 |
| LN share | 0.12 | 0.53 | 0.37 | 0.82 |
| loop | 0.41 | 0.64 | 0.52 | 0.53 |
| hand | −0.04 | −0.02 | −0.20 | 0.20 |

gen-vs-source ≈ the product of the other two (nh 0.23 vs 0.27; held 0.20 vs 0.22; jack 0.19 vs 0.26): generated and source agree only through the rows [M numbers, I reading]. B1, B2, B3 and d0-phi show the same [M].

**E. Same rows, between-part variation (median over panel charts).**

| | source | B0 | B1 | B3 | d0-phi |
|---|---:|---:|---:|---:|---:|
| SD over windows of log notes/s | 0.204 | 0.207 | 0.206 | 0.211 | 0.191 |
| share of that variance fixed by row density | 1.00 | 1.04 | 0.87 | 0.93 | 1.06 |
| SD over windows of nh | 0.127 | 0.152 (+20 %) | 0.184 (+45 %) | 0.168 | 0.072 (−43 %) |
| SD over windows of held | 0.055 | 0.066 | 0.067 | 0.066 | 0.027 |
| corr(window nh, row density) | −0.20 | −0.27 | −0.11 | −0.20 | −0.30 |
| window nh vs source, within chart | | 0.20 | 0.21 | 0.34 | 0.25 |

## 1. What do the supplied rows carry?

| property | carried by rows? | evidence |
|---|---|---|
| Rhythm, onset choice | fully; the mapper's choice, not the audio's | same-audio human head F1 at 20 ms median 0.785 (2026-09-23 audio-skeleton audit) [M, earlier] |
| Density, NPS profile, section dynamics | fully | table E: rows fix about 100 % of between-part log-notes/s variance, in sources and every arm [M] |
| Difficulty | mostly | band accuracy 0.70, star R² 0.83; yet 69 % of near-identical-row pairs (1,558 of 2,243) sit in different bands: the same rows carry several difficulties through chord density [M] |
| LN-ness through gaps | weakly | LN share R² 0.21, LN length 0.23, row-level LN-head r 0.30 [M] |
| Mapper's intent (where chords and LNs go) | partly for chords, little for LN | table C: rows 0.56 / 0.30; humans share 0.77 / 0.64 [M] |
| Style leakage | partly | nh 0.41, entropy 0.41, roll 0.31, jack 0.27, loop 0.27, trill 0.08, hand 0 [M] |

<a id="o-skeleton-misattributed"></a>**Treated as R2's failure while the rows fix it, or fix it partly:**
- **"Difficulty diverging between parts" (X0 notes).** As density or notes per second it is fixed by the rows: generated and source identical (0.207 vs 0.204) [M]. Whatever diverges is composition: window chord spread +20 % in B0, +45 % in B1, LN spread +20 % [M]; the level random walk seen through allocation [I]. A density-dominated window measure could not separate generated from real in X0, because both had the same rows [I].
- **Band as the reference.** m5, m6, the d0-env band envelopes, the band-calibrated guards and F1 ("start pulled toward the band's corpus mean") mix a rows property with an arrangement property [I from M]. The right reference is the row-conditional expectation. Against it, R2 does not drift to the band mean: it reproduces E[θ | rows] and adds noise (table D) [M]. m5 gets about 40 % of the nh variance from the rows for free, so it understates missing identity variance [I].
- **Paired comparisons with the source.** The source is one draw. A second human on near-identical rows, same band, agrees at chart-level ICC nh 0.58, jack 0.63, held 0.63, LN share 0.68, loop 0.68, hand 0.17; median |Δ| 0.74 SD for nh, 0.17 SD for LN share [M]. B3-oracle (held r 0.81) and the decode round's "identity r 0.94" exceed that ceiling, so they copy the source rather than hit a deployable target [I].
- **Mechanical jacks.** About a quarter of chart-level jack rate and 37 % of within-chart jack modulation follow the rows [M]; dense regular rows in real charts carry jacks too [I].
- **Correctly left to R2:** hand balance and trills (rows carry nothing), LN level (rows carry about 20 %) [M].

## 2. "R2 achieves X without Y"

<a id="s-skeleton-achieves"></a>**Supported:**
- The source's rhythm, density and section dynamics, without audio or a timing model. Exact, by construction (table E) [M].
- The rows-determined part of chart identity, without a chart-level input: gen vs rows-only prediction r 0.48-0.64, as close as the source itself (0.37-0.56) [M].
- Row-level chord and LN placement as far as the local rows allow, without audio: 0.30 / 0.08, equal to an ideal rows-only sampler [M numbers, I equivalence].
- An approximate difficulty level without a difficulty input, but only on human rows (star R² 0.83; section 4 shows it breaks) [M].
- A clean teacher-forced target with no timing error [I].

**What the separation cannot give:**
- A chart-level identity beyond the rows: 59-80 % of the between-chart variance of nh, jack, held, LN share, entropy and loop; 100 % of hand balance [M].
- The row-level choreography mappers share: chord placement 0.77 vs 0.30, LN-head placement 0.64 vs 0.08 [M]. LN placement is where the gap is largest, which matches the human's "weird LN distribution" [I].
- Lane choice, or melody-to-lane mapping: humans on the same rows agree on exact lanes only at twice chance (0.21 vs 0.10), so lanes are largely free and belong to identity. Melody-to-lane not tested.
- Section intent beyond density: chord allocation against density is R² 0.17 from rows [M].

## 3. How much do the rows determine?

- **Chart identity:** at most 0.41 R², kNN-style priors about half that.
- **Local modulation, given identity:** about 40 % of within-chart chord and jack variation, 11-14 % of LN variation.
- **Row-level placement:** r about 0.56 for chords, 0.30 for LN heads; humans share 0.77 / 0.64.
- Near-identical skeletons in the same song share chart texture at ICC 0.5-0.7 within band; across bands, nh ICC 0.17 against 0.58 within. Rate edits and retimed copies make up 34-42 % of same-band pairs below F1 0.8, so those buckets were not used [M].

## 4. Deployment shift (rows from the legacy timing model)

**Data:** the 2026-09-23 audio-skeleton integration runs (`artifacts/audio-skeleton/20260923-v1/integration*/…/full_timing-*`). 4 of 6 songs are in the R2 cache (all fit_train), 6 timing configurations each. The beat grid is the source's.

**Rows [M]:** head F1 against the source 0.70 at 20 ms, 0.82 at 40 ms (human vs human on the same audio 0.785 at 20 ms). Row count ratio median 1.12 (0.50-2.18). Band predicted from timing rows agrees with band from source rows in 25 % of runs; predicted star shifts by a median 0.86 SD. The spread across songs of rows-only predictions shrinks to 0.17× (star), 0.43× (nh), 0.48× (held), 0.65× (LN share): timing rows all look like band-3 rows. Band-5 song `a9ddd10c87b6`: predicted star 5.3 from its own rows, 3.2-4.4 from timing rows.

**R2 56M on those rows (2 seeds each) [M; 4 songs, direction only]:**

| song (band) | c3: source / R2 on source rows / R2 on timing rows | jack: same | LN share: same |
|---|---|---|---|
| a9dd (5) | 0.15 / 0.14 / 0.03-0.05 | 0.23 / 0.24 / 0.06-0.14 | 0.13 / 0.17 / 0.13-0.36 |
| b23b (2) | 0.04 / 0.08 / 0.06-0.15 | 0.11 / 0.14 / 0.17-0.21 | 0.06 / 0.11 / 0.14-0.33 |
| c56c (4, LN-heavy) | 0.07 / 0.01 / 0.00-0.03 | 0.14 / 0.11 / 0.05-0.09 | 0.40 / 0.28 / 0.005-0.25 |
| a01b (2) | 0.00 / 0.04 / 0.01-0.04 | 0.15 / 0.15 / 0.04-0.08 | 0.09 / 0.54 / 0.01-0.21 |

**Habits that would break [I]:**
1. Difficulty read from the rows: on timing rows the hard song loses its chords and jacks and the easy song gains them; compression toward the middle.
2. Style leakage: between-chart variety and LN level lose their row anchor, so an identity latent must carry what the rows used to leak.
3. Local cues: about 30 % of heads missing or extra at 20 ms change the gap patterns R2 reads for LN starts, jack runs, chord placement and release candidates.
4. The beat grid: not measured (the source's red lines were used); a timing model must also supply it.

## 5. Counterfactual: had the agents taken the supplied rows seriously

- **R2 v2 design:** split identity into a computed rows part and a free latent; a whole-song row summary is a legitimate input (R2 sees only 511 rows back and 32 beats ahead). With no chart-level variable the model can only produce E[θ | rows] plus drift. Put the rows-only floor (0.31 chord placement, 0.08 LN heads) beside the human ceiling (about 0.6 / 0.4) before scoping audio out; LN and chord placement point at audio [P/I].
- **Collapse diagnosis:** test the pull against the row-conditional mean, not the band mean (F1); split "difficulty diverging" into its rows part and its composition part; drop density-driven measures from the X0 search; residualise measures on the rows [P].
- **Bake-off:** the donor prior's weakness was visible in minutes (kNN-32 R² 0.11 held, 0.26 nh); m5 and m6 are partly rows properties; agreement with the source needs the same-rows human ceiling; B3-oracle and the d0 targets copy the source; R2 should have been run on the timing-model rows (on disk since 2026-09-23) before deciding that rows supply difficulty [P].

## Three most consequential implications

1. **Represent the arrangement given the rows, not the section.** A section representation or measure has to be row-conditional (a residual against E[· | rows], or defined relative to the rows). Otherwise it rediscovers the rows (70 % band accuracy, about 100 % of the notes/s profile), which are identical between generated and source charts [I from M].
2. **Chart identity has to be an explicit, sampled-and-held latent, with the rows part computed from the whole song outside the sequence model.** Rows fix at most 41 % of identity and none of hand balance; the same rows support different bands; under timing-model rows the rows' share shrinks further (0.17-0.65×), so the latent must also carry difficulty and LN level [I from M].
3. <a id="p-skeleton-benchmark"></a>**Row-level placement is a measurable texture gap, and a benchmark exists.** About 2,240 same-song pairs on near-identical rows (aligned F1 ≥ 0.95) give a human ceiling (chords 0.77, LN heads 0.64), a noise floor for any comparison with a source, and a label-free test for a representation: it has to tell R2-on-rows from a second human on the same rows. R2 already sits at the rows-only ceiling, so closing the gap needs information outside the rows: audio accents and sustains, or song structure [I from M; P].

## Not done

- Audio vs song structure vs set convention in the human-pair agreement not separated; mapper identity per pair unknown.
- The timing model's own beat grid not tested; section 4 rests on 4 songs.
- No probe on the Lens labels or X0 (fable-represent covered those). No melody-to-lane test.

## Jobs, scripts, outputs

- Scripts: `build.py`, `fit.py`, `alloc.py`, `pairs2.py`, `q4.py`, `rowlevel.py`, `rowlevel2.py`, `q4gen.py`.
- Jobs (all exit 0): `20261008-031710-osk-build-smoke`, `-031753-osk-build` (61 s), `-031917-osk-fit` (about 8 min), `-032748-osk-alloc`, `-032853-osk-pairs2`, `-033018-osk-q4`, `-033116-osk-rowlevel`, `-033323-osk-rowlevel2`, `-033413-osk-copies`, `-033523-osk-q4gen`, plus `-osk-peek*` log reads.
- Outputs: `fit.json`, `alloc.json`, `pairs2.json`, `rowlevel.json`, `rowlevel2.json`, `q4.json`, `q4_runs.csv`, `q4gen.json`, `q4gen.csv`, `*.log`. Mac only: `data/{charts,windows,pairs}.parquet`, `pairs2.parquet`, `pred_dev.parquet`, `smoke/`.
- Superseded, not wrong: the `pairs` section of `fit.json` (unaligned F1; replaced by `pairs2.json`); `rowlevel.json` (±4 gaps; `rowlevel2.json` at ±16 confirms it).
