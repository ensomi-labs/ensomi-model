# Step 1: representation R with a same-rows human null (2026-10-08)

Report of the fresh Fable subagent that ran step 1 of [p-represent-next](r2-representation-20261008.md#p-represent-next), authorised by the human ([r-step1](r2-representation-20261008.md#r-step1)). It ran 04:04-04:30 UTC with about 12 minutes of mac time at ≤ 4 threads. Brief: `~/ensomi/.sync/cp/scratch/r2-represent-step1/brief.md`. Scripts: `~/ensomi/.sync/cp/scratch/r2-represent-step1/worker/`. Outputs on bings-mac: `artifacts/r2-represent-20261008/step1/` (md, json and csv are mirrored). Saved as returned, condensed in layout, below the main thread's checks.

<a id="v-step1-checks"></a>**Main-thread checks (2026-10-08).**
- **Checked against the mirrored outputs:**
  - `mapper-split.md`: 568 / 47 same-band pairs; LN-head r 0.671 vs 0.451; ICC held 0.675 vs 0.195; difference of medians 0.221 [0.146, 0.303].
  - `labelfree.md`: chart AUC block 1 0.803 [0.76, 0.84], block 2 0.627, block 3 0.688; seed spread n1 0.425, n2 0.243, n3 0.138.
  - `x0-rescore.md`: one-sided Δ ln2 flags 0.732 of marked windows and 0.000 of no-item windows. Two-sided, it flags 0.734 of no-item windows.
- **Deviations, for the human:**
  - **The worker refit the block-3 contrastive encoder for 240 s of CPU.** The pilot had saved no weights. The worker read "no training" as "no R2 training". The main thread accepts that reading: the encoder is part of the measuring instrument, and the R2 generator was not trained.
  - **The brief gave a wrong path for the X0 judge sheet** (the main thread's error). The worker found the right one: `artifacts/r2-collapse-20261007/phase0-astra/x0-human/judge-sheet-human.csv`.
  - **The different-band null used 700 of the 1,556 pairs**, a seeded sample.
- **Post hoc:** the one-sided direction of the X0 flag was chosen after seeing the no-items. No measure is validated by this report.

---

## 0. Short answer

- <a id="o-step1-null-same-mapper"></a>**The null exists and is tight because it is mostly a same-mapper null [M].**
  - Of the 615 same-band pairs at F1 ≥ 0.95, 568 (92 %) are two difficulties by the same creator.
  - A second human on the same rows differs from the first by |Δ ln2| ≤ 0.245 per window in 90 % of windows. Split by creator: 0.229 same creator, 0.406 different creator (47 pairs).
- <a id="o-step1-x0"></a>**On that scale, the X0 marks are LN excess in one direction, and mostly one chart [M, post hoc].**
  - The one-sided flag "Δ ln2 > +1 null unit" catches 30 of 41 marked windows (73 %), 0 of 94 windows on no-items, and 16 % of unmarked windows on yes-items. The human null fires one-sided on 4.4 % of windows.
  - Without x0-02: 3 of 14 marks.
  - The two no-items (x0-05, x0-10) are where R2 holds far *less* than the source (−1.6 and −2.4 chart units). A two-sided flag would mark 73 % of their windows. Direction is part of the measure.
- <a id="o-step1-jack"></a>**Jack and repetition measures, paired to the source as item statistics, do not separate yes from no items [M].**
  - AUC 0.25-0.75, 6 against 2.
  - Fast jacks reach |Δ| AUC 0.875, and the block-2 norm 1.0. Both are post hoc on 8 items; by chance alone, 1 in 28 orderings reaches 1.0.
  - Unpaired, fast-jack and full-jack rates separate generated from real at 0.84-0.94.
- <a id="o-step1-mapper"></a>**Mapper split: LN agreement beyond the rows is largely one mapper's convention; chord agreement is in the song [M, 47 different-creator pairs].**

  | | Same creator | Different creator | Rows-only | R2 |
  |---|---:|---:|---:|---:|
  | LN-head r | 0.671 | 0.451 | 0.32 | 0.08 |
  | LN-level ICC | 0.68 | 0.20 | | |
  | Chord r | 0.792 | 0.735 | 0.59 | 0.30 |

- <a id="o-step1-labelfree"></a>**Label-free: R tells R2 from the second human, block 1 best [M].**

  | Level | Block 1 | Block 2 | Block 3 | All three |
  |---|---:|---:|---:|---:|
  | Chart AUC | 0.80 [0.76, 0.84] | 0.63 | 0.69 | 0.79 |
  | Window AUC | 0.73 | 0.64 | 0.77 | |
  | Chart AUC, 9 different-creator pairs | 0.72 | 0.44 | 0.73 | |

  Block 2's separation is inside R2's own seed spread.

## 1. Representation R [M]

`worker/represent.py` imports fable-represent's `pilot.py` and regroups its features.
- **Block 1:** pilot B's 15 LN columns plus A's held, bus4 and lock (18 dims).
- **Block 2:** the other 33 B columns plus A's pent, ng4, rep1, loop, nh, c3, c4, jack, fjack, lmax, hmax and hlock (45 dims). lrows and ldtmed were dropped as properties of the rows.
- **Block 3:** the pilot's 64-d contrastive encoder, refit for 240 s on 1,500 fit_train charts outside the pairs and X0. Loss went 5.30 → 2.75 (chance 5.55); the pilot reached 2.55.

R is computed per 64-row window (stride 32) and per chart. Paired use takes chart B's features on the time span of A's window after the pair's alignment shift. Block norms are the RMS of Δ, standardised by the same-band human SD per feature. Block 3 uses the Euclidean distance between embeddings.

## 2. Null [M]

Pairs: `pairs2.parquet`, aligned F1 ≥ 0.95, copies excluded; all 615 same-band pairs and 700 of 1,556 different-band pairs. Matched windows: 17,629 same band, 19,032 different band. Values are |Δ| = |human B − human A| as q50 / q90 / q99.

| measure | blk | window same band | window diff band | chart same band | chart diff band |
|---|---|---|---|---|---|
| ln2 (rows ≥ 2 held) | 1 | 0.000 / 0.245 / 0.672 | 0.000 / 0.241 / 0.667 | 0.020 / 0.207 / 0.546 | 0.016 / 0.217 / 0.523 |
| held share | 1 | 0.012 / 0.156 / 0.396 | 0.012 / 0.156 / 0.380 | 0.025 / 0.132 / 0.308 | 0.023 / 0.136 / 0.307 |
| tap while held | 1 | 0.025 / 0.387 / 0.883 | 0.024 / 0.359 / 0.841 | 0.052 / 0.286 / 0.583 | 0.046 / 0.253 / 0.574 |
| ln_share | 1 | 0.020 / 0.265 / 0.747 | 0.021 / 0.231 / 0.701 | 0.031 / 0.215 / 0.568 | 0.025 / 0.189 / 0.422 |
| fjack (< 100 ms) | 2 | 0.000 / 0.010 / 0.123 | 0.000 / 0.000 / 0.123 | 0.000 / 0.008 / 0.085 | 0.000 / 0.008 / 0.081 |
| jack | 2 | 0.055 / 0.168 / 0.306 | 0.081 / 0.232 / 0.421 | 0.066 / 0.134 / 0.236 | 0.103 / 0.225 / 0.387 |
| pent | 2 | 0.200 / 0.520 / 0.913 | 0.262 / 0.651 / 1.088 | 0.216 / 0.455 / 0.722 | 0.301 / 0.579 / 0.893 |
| nh | 2 | 0.164 / 0.389 / 0.656 | 0.250 / 0.578 / 1.010 | 0.172 / 0.324 / 0.484 | 0.276 / 0.558 / 0.944 |
| hand_bal | 2 | 0.018 / 0.054 / 0.113 | 0.018 / 0.053 / 0.108 | 0.005 / 0.017 / 0.039 | 0.005 / 0.016 / 0.029 |
| block 1 norm | 1 | 0.488 / 1.645 / 3.141 | 0.531 / 1.649 / 2.950 | 0.527 / 1.423 / 3.132 | 0.566 / 1.484 / 2.793 |
| block 2 norm | 2 | 0.853 / 1.367 / 2.258 | 1.007 / 1.676 / 2.707 | 0.768 / 1.460 / 5.092 | 0.998 / 1.914 / 4.332 |
| block 3 dist | 3 | 0.772 / 1.116 / 1.338 | 0.900 / 1.232 / 1.411 | 0.521 / 0.837 / 1.040 | 0.656 / 0.974 / 1.210 |

All 66 measures: `null-table.md`. The same-band q90 by creator (`null-by-creator.md`), same creator vs different creator:

| Measure | Same creator | Different creator |
|---|---:|---:|
| ln2 | 0.229 | 0.406 |
| held | 0.147 | 0.262 |
| tap_held | 0.359 | 0.612 |
| block 1 | 1.58 | 2.27 |
| block 2 | 1.35 | 1.66 |
| block 3 | 1.11 | 1.21 |

The pooled q90 fires on 19-26 % of different-creator windows for the LN measures. **The null is for the most part the spread of one mapper across two difficulties.**

How the null behaves [I from M]:
- **The LN null does not depend on band.** The chord, jack and entropy null widens across bands.
- **Flags per pair (one-sided, ln2):**
  - same creator: median 0, q90 0.11; 2.8 % of pairs have over half their windows flagged;
  - different creator: q90 0.31; 8.5 % of pairs have over half flagged.

## 3. X0 re-scored [M, post hoc]

The 8 generated items against their sources: 551 windows, 41 marks (x0-02 27, x0-03 5, x0-11 3, x0-07 2, x0-09 2, x0-12 2). The window AUCs of signed Δ equal fable-represent's. The new content is the null unit, the flags and the removal of x0-02.

| score | L1 | L2 | within-yes | L1 no x0-02 | L2 no x0-02 | within-yes no x0-02 |
|---|---|---|---|---|---|---|
| Δ ln2 | 0.989 | 0.855 | 0.825 | 0.969 | 0.702 | 0.629 |
| Δ held | 0.972 | 0.838 | 0.808 | 0.917 | 0.645 | 0.571 |
| Δ bus4 | 0.993 | 0.872 | 0.844 | 0.980 | 0.725 | 0.655 |
| block 1 norm | 0.672 | 0.793 | 0.820 | 0.299 | 0.544 | 0.610 |
| block 2 norm | 0.618 | 0.478 | 0.446 | 0.826 | 0.728 | 0.701 |
| block 3 dist | 0.612 | 0.712 | 0.735 | 0.555 | 0.660 | 0.689 |

**Flags, one-sided Δ > +1 unit** (the unit is the pooled same-band q90; the different-creator unit in brackets):

| Measure | Marked windows | Unmarked windows, yes-items | No-item windows | Marked windows, without x0-02 |
|---|---:|---:|---:|---:|
| Δ ln2 | 0.732 (0.659) | 0.163 (0.099) | 0 | 3/14 (1/14) |
| Δ held | 0.780 (0.732) | 0.279 (0.127) | 0 | 0.357 (0.214) |

- On x0-02, every mark is flagged and 0.74 of its windows. That share exceeds the per-pair q90 of both human groups.

**Chart level, paired, AUC yes vs no among the 8 (6/2):**

| Measure | AUC (signed / abs) | Notes |
|---|---|---|
| fjack | 0.71 / 0.88 | |
| jack | 0.42 / 0.58 | Large only on x0-03: +1.78 units, 99th percentile |
| jack_full | 0.25 / 0.25 | |
| pent | 0.33 / 0.50 | x0-02 is less diverse than its source; x0-03 is more diverse |
| hand_bal | 0.25 / 0.50 | |
| block 2 norm | 1.00 / 1.00 | Post hoc |
| block 1 norm | 0.33 / 0.33 | The no-items are high, from their LN deficit |

**Unpaired, all 12 items, AUC yes (7/5) / generated vs real (8/4):**

| Measure | Yes | Generated vs real |
|---|---:|---:|
| fjack | 0.86 | 0.84 |
| jack_full | 0.77 | 0.94 |
| ng4 | 0.54 | 0.94 |

Readings [M]:
- Paired "lack of variance" appears only in x0-02. Paired "mechanical jacks" is large only on x0-03, beyond the 97th-99th percentile of the human null.
- Elsewhere R2's jack level sits inside the human spread. The unpaired generated-vs-real fingerprint (fast jacks, full jacks) remains.

## 4. Mapper split [M]

**Data:** creator from `data/r2-corpus.parquet` (the `.osu` Creator field, none missing). Of 2,171 pairs, 2,074 have the same creator and 97 different creators. Every pair is listed in `mapper-pairs.csv`.

| group | pairs | chord r median | LN-head r median | ICC nh | ICC held | ICC LN share | ICC jack | ICC hand |
|---|---|---|---|---|---|---|---|---|
| same creator, same band | 568 | 0.792 | 0.671 | 0.56 | 0.68 | 0.74 | 0.63 | 0.19 |
| different creator, same band | 47 | 0.735 | 0.451 | 0.67 | 0.20 | 0.20 | 0.58 | 0.02 |
| same creator, different band | 1,506 | 0.768 | 0.667 | 0.16 | 0.67 | 0.77 | 0.29 | 0.28 |
| different creator, different band | 50 | 0.710 | 0.403 | 0.17 | 0.20 | 0.33 | 0.03 | −0.21 |

- **Same minus different creator, same band** (bootstrap 95 %): chord r +0.057 [0.027, 0.086]; LN-head r +0.221 [0.146, 0.303].
- **Excluding guest-difficulty-looking versions:**

  | | Chord r | LN-head r | ICC held | ICC LN share |
  |---|---:|---:|---:|---:|
  | Same creator (388) | 0.804 | 0.714 | 0.648 | 0.71 |
  | Different creator (27) | 0.764 | 0.514 | 0.112 | 0.14 |

Readings [I from M]:
- **Chord placement:** the agreement beyond the rows is mostly in the song. It is 0.735 across mappers, against 0.59 from the rows alone.
- **LN-head placement and LN level:** these are mostly one mapper's convention.
- **The previous round's "human ceiling" is a same-mapper ceiling.**
- **For R2's scope:**
  - the LN information missing from the local rows looks more like a chart-level identity than an audio cue;
  - chord placement is where audio or song structure would add most.
- **Caveats:**
  - only 47 different-creator pairs;
  - creator equality is by string;
  - the guest-difficulty heuristic is a regex on version names;
  - 7 % of same-creator same-band pairs agree on over half their lane patterns, so they are edits more than independent arrangements.

## 5. Label-free test [M]

**Setup:** 120 same-band pairs, one per song group (bands 2/3/4/5: 27/47/37/9); 111 have the same creator.
- R2 56M generated on A's rows: BOS, T = 1, seeds 954 and 955, first min(K + 1, 2,048) decisions; 240 generations in 367 s.
- d is the block distance of an arrangement from A. The score is the AUC of d(gen, A) against d(B, A), with a cluster bootstrap by pair.

| block | window AUC [CI] | median d human / R2 | chart AUC [CI] | median d human / R2 |
|---|---|---|---|---|
| 1 LN relations | 0.732 [0.69, 0.77] | 0.39 / 1.24 | **0.803 [0.76, 0.84]** | 0.51 / 1.17 |
| 2 grammar with hands | 0.639 [0.61, 0.67] | 0.88 / 1.04 | 0.627 [0.58, 0.68] | 0.78 / 0.97 |
| 3 embedding | 0.766 [0.73, 0.80] | 0.77 / 1.04 | 0.688 [0.64, 0.74] | 0.51 / 0.68 |
| 1+2+3 (rank sum) | 0.778 [0.74, 0.81] | | 0.793 [0.75, 0.84] | |

- **By band:** block 1 stays 0.77-0.86 in every band.
- **By the pair's creator:** same creator 0.81 / 0.64 / 0.69; different creator (9) 0.72 / 0.44 / 0.73.
- **Signed single measures are near chance (0.40-0.59).** R2 places fewer chords than A: Δ nh −0.09, against −0.02 for B.
- **Seed spread of d at chart level** (median |d(954) − d(955)|): 0.42 / 0.24 / 0.14, against human-to-R2 gaps of 0.66 / 0.19 / 0.17.

Readings [I from M]:
- **Block 1 separates R2 from the second human.** Since that human is almost always the same mapper, the fair statement is: "R2 is further from A in LN relations than A's own mapper is across difficulties."
- **This is a texture gap, not a validated defect.** A measure of distance from the source would also flag the LN-deficit items the human passed.

## 6. What this means for "R measures what the human marks"

1. **[M] The window claim belongs to one chart.**
   - Block 1, one-sided, puts 73 % of the marks beyond the one-sided 95 % point of the same-mapper spread, and none of the no-item windows.
   - Without x0-02 it catches 3 of 14 marks. On x0-02 itself it is clean.
2. **[M] The other marks are not located by block 1.**
   - The 12 marks on x0-03, x0-09, x0-11 and x0-12 fall outside it.
   - Without x0-02, the block-2 norm locates best (L1 0.83, within-yes 0.70). That rests on 14 marks on 4 charts, unsigned.
3. **[M] Jack and repetition, per chart, are not an excess over the source**, except on x0-03 (jacks) and x0-02 (entropy). The generated-vs-real fingerprint is unpaired.
4. **[M] The null unit depends on who the second human is.** Across mappers it is 1.8× wider for the LN measures. A flag threshold has to name its null.
5. **[I] Nothing here validates a measure in the sense of guardrail 1.** Ready for a preregistered screen:
   - the one-sided block-1 flag with the different-mapper unit (0.406 for ln2, 0.262 for held);
   - the block-2 chart norm, as a candidate for the non-LN complaints.
6. **[I] Alternative explanations:**
   - a. The same-rows null is a within-mapper spread, so "beyond the human null" overstates for x0-02 unless the different-mapper unit is used. That unit still flags 66 % of its marks.
   - b. The label-free AUC may reflect that R2 lacks the chart-level LN identity a mapper carries across difficulties, rather than a local defect.
   - c. With 2 "no" items among 8, any item-level AUC has a 1-in-28 chance of reaching 1.0.

## 7. Runs, paths, not done

**Jobs** (all exit 0, ≤ 4 threads, about 12 min of mac time):
- `20261008-040834-s1-smoke`
- `-041300-s1-gen` (368 s)
- `-041303-s1-null` (326 s, including the 240 s encoder fit)
- `-041848-s1-x0`
- `-041931-s1-labelfree`
- `-042011-s1-mapper`
- `-042221-s1-bycreator`

**Scripts:** `represent.py`, `null_pairs.py`, `x0_rescore.py`, `mapper_split.py`, `gen_pairs.py`, `labelfree.py`, `null_by_creator.py`, `smoke.py`.

**Outputs:**
- Mirrored: `null.json`, `null-table.md`, `null-table.csv`, `null-by-creator.*`, `x0-rescore.*`, `mapper-split.*`, `mapper-pairs.csv`, `labelfree.*`, `lf-pairs.csv`, `gen.json`, `encoder.json`.
- Mac only: `encoder.pt`, `*-windows.parquet`, `null-charts.parquet`, `gen/*.npz` (240 generations).

**Not done:**
- bootstrap intervals on the X0 AUCs;
- different-band pairs in the label-free test;
- more different-creator pairs in the label-free test (9 used of 47);
- resolving creator aliases;
- hold lengths of LNs that cross the 2,048-row cut, for 5 generated charts.

**Failed paths:** none.
