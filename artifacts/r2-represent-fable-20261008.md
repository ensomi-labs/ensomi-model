# Representations of chart sections and the measures that follow, with a pilot (fable-represent, 2026-10-08)

Report of the fresh Fable subagent "fable-represent" in the representation round ([r-represent-relaunch](r2-representation-20261008.md#r-represent-relaunch)), returned 2026-10-08 about 03:45 UTC. Brief: `~/ensomi/.sync/cp/scratch/r2-represent/brief-fable-represent.md`. About 8.5 minutes of mac time at 2 threads. Scripts in `~/ensomi/.sync/cp/scratch/r2-represent/fable-represent/` (`pilot.py`, `x0_items.py`, `x0_paired.py`); outputs on bings-mac in `artifacts/r2-represent-20261008/fable-represent/` (mirrored). Saved as returned, condensed in layout only, below the main thread's checks.

<a id="v-fable-checks"></a>**Main-thread checks (2026-10-08).**
- **Checked against the mirrored outputs:**
  - `pilot.json`: the per-concept logistic F1 values of A, B, C, C0 and A+B+C match the table.
  - `x0-paired.json`: Δ ln2 has L1 0.991, L2 0.862, within-yes 0.832. Δ held share has 0.976 / 0.838 / 0.807. Δ tap-while-held has 0.943 / 0.801 / 0.769. Δ pent has L1 0.293, reported as 0.707 with the sign reversed.
- **Caveats the main thread adds:**
  - **The X0 window marks are concentrated.**
    - Unpaired: 38 of the 50 marks are in band 5, from two items. These are x0-02 (generated) and x0-08 (real).
    - Paired: x0-02 holds 27 of the 41 marks.
    - So the paired LN result rests mostly on one generated chart.
  - **The scores were chosen after seeing X0** (the worker says so). Every X0 AUC here is optimistic.
  - **The Lens probe measures agreement with the Astra labeller.** About 96 % of the cells are machine labels.
  - **The paired measures are scaled by band SDs of real windows,** not by the spread between two humans on the same rows. A second human on the same rows also differs from the source; opus-skeleton measured LN-share ICC 0.68 ([p-skeleton-benchmark](r2-represent-skeleton-20261008.md#p-skeleton-benchmark)). The paired measures have no null yet.

---

## 0. Summary

- <a id="o-fable-level-vs-relation"></a>The four representations used so far (row-symbol HMMs, 64-row window statistics, TCN state, θ) are all *level* representations: how much of something. Every complaint of the human and every Lens definition is *relational*: how things are arranged on lanes, hands, holds and gaps [I].
- <a id="o-fable-lens"></a>Lens concepts, 2,888 labelled sections, chart-grouped 5-fold CV, macro F1: window statistics 0.677; hand-built structure 0.719; 7-minute contrastive embedding 0.687 (random-weights control 0.613); all three together 0.759; majority 0.258. Structure wins where the concept is a relation (trill 0.598 → 0.815); the embedding wins where the concept is a residue (tech 0.517 → 0.614) [M].
- <a id="o-fable-x0-ln"></a>X0: the human's marks, as placed, are almost entirely "LN relations in excess of the band and of the source chart on the same rows". Paired against the source on identical head rows, the signed excess of rows with two or more lanes held gives window AUC 0.99 (marked vs no-item windows) and 0.83 within the yes-items; the same score without LN columns is 0.40-0.53 [M]. Band-matched typicality in window-statistic space reaches 0.956/0.823 but falls to 0.61 once held share is regressed out [M].
- <a id="o-fable-two-targets"></a>Two targets are mixed in X0: what the human calls collapse (LN-driven, and it includes a real band-5 LN chart) and what distinguishes generated from real (fixed-group jacks, fast jacks, hand imbalance, within-chart embedding dispersion: item AUC 0.81-0.94 generated vs real, but near 0.5 against the marks) [M, 12 items]. Measures must say which target they serve [I].
- Recommended: a relational row-token representation in three blocks (LN relations; timing-conditioned pattern grammar with hands; a small contrastive embedding for the residue), computed *given the supplied rows* and paired with the source whenever the skeleton comes from a real chart [P].

Tags: [M] measured, [I] inferred, [P] proposed.

## 1. Counterfactual analysis

### 1.1 What our representations were, and what each threw away [I, from code and notes]

| Representation | Sees | Threw away |
|---|---|---|
| Row-symbol n-grams / HMMs (48-81 symbols) | symbol frequencies, 1-2 step transitions, sticky regimes | the gaps (an 80 ms and a 320 ms passage are the same string); lane continuity beyond n; hands as units; hold length and release relations; regime durations (self-transitions 0.85-0.97 make a long stay *likely*). Likelihood ranks by commonness: a density meter, which is why it ranked the marks backwards (decode round). |
| 64-row window statistics (`collapse.py`, 15 scalars) | levels (nh, held, jack, bus4), repetitiveness as rep1/loop/ng4/pent, hmax/lmax/lock | order inside the window beyond lag 1 and fixed periods; which lanes jack and for how long; how holds relate to taps and to each other (held share says how much, not how); anything conditional on the gap (dtmed only); variation inside the window. These are the band-marginal levels the model copies well, so by construction they are blind to what it gets wrong. |
| TCN generator state | what the next-row distribution needs | whole-episode properties (causal, 511 rows); under teacher forcing it re-encodes recent levels, so state ≈ features (0.580 vs 0.587 in the earlier probe) and the sum adds nothing. |
| θ (10 chart means) | chart identity as levels | everything local; cannot describe a section; B3 barely followed it. |

The human's words map to relations: "mechanical same-lane jacks" = lane identity × run length × gap; "repetition, lack of variance" = few pattern types over time, not the entropy of a 16-bin histogram; "poor hand balance" = hand as a unit over time; "weird LN distribution" = hold lengths and their relation to taps and other holds (the Lens definition of LN coordination is "independent control relationships" between complete holds and taps); "difficulty diverging between parts" = a chart-level trajectory. The Lens definitions are relational too: Jack = recurrence *sequence*, fixed-group vs changing-chord vs localized; Stream = continuity, direction, chord placement, resets; Trill = two disjoint groups alternating; Tech = hard to anticipate through familiar patterns (the residue of a grammar); LN coordination = control relations.

### 1.2 Candidate representation spaces [I; cost for a 64-row window]

| Space | Sees | Cannot see | Cost | Human / Lens concepts |
|---|---|---|---|---|
| Per-hand trajectories (lane within hand, chord within hand, hold state; alternation, one-hand runs, load, anchors) | hand balance over time, anchors, same-lane mechanics, cross-hand trills, one-hand bursts | musical intent; rhythm beyond gaps | trivial | hand balance, same-lane jacks / Jack, Trill, part of Stream |
| Pattern-grammar parse (per-row token: full/partial jack, trill, stream step, stair, roll turn, jump, chordjack, LN start while held, release kinds; then run lengths and token n-grams) | episode organisation, run lengths ("mechanical"), pattern mixture ("lack of variance" = few token types, long runs) | tech by definition, but the unparsed residue is a tech proxy | trivial once tokens are defined (community taxonomies) | repetition, jacks / Jack, Trill, Stream |
| Timing-conditional choices (gap class relative to local tempo, beat position × what the arrangement does there) | chords on strong beats, jacks only at slow gaps in real charts, LN heads at long gaps | what the music does | trivial with the grid | weird LN, difficulty divergence / Tech, Stream continuity |
| Residual given the skeleton (arrangement statistics minus what the rows predict, or paired against the source on the same rows) | what R2 actually decided | whether the skeleton itself suits | low | the fair measure of R2; §1.3 |
| Section embedding (contrastive nearby-windows-of-one-chart vs other charts, mirror-augmented; or masked-section modelling) | whatever separates charts locally; the residue | named concepts without a probe; may latch onto the skeleton unless positives share it | 7 min CPU here | lack of variance, diverging parts / all five via a probe, tech best |
| Physiology / strain (`strain.py` ras-v1) | difficulty trajectory, jack strain spikes, skeleton strain vs arrangement increment | organisation quality at equal strain | low | difficulty diverging, same-lane jacks |
| Transition graphs (self-loops = jacks, 2-cycles = trills, 4-cycles = rolls; edge entropy) | compact pattern vocabulary; repetition as low edge entropy | timing, LN | trivial | mechanical repetition / Jack, Trill, Stream |

### 1.3 The supplied rows [I]

R2 does not choose the rows, so a measure that is mostly a function of the skeleton (rows per second, dtmed, much of nh and c3) credits or blames R2 for the source mapper's rhythm. (i) Measures must be conditional on gap and beat, or residualised; the bake-off's m vs m_src did this only at the level of marginals. (ii) "Difficulty diverging between parts" is partly the skeleton's density and partly R2's chords, jacks and holds; a strain model split into skeleton strain and arrangement increment separates them. (iii) When the skeleton comes from a real chart, the source arrangement sits on the *same rows*, so each generated window has a paired real counterpart; this pairing is the sharpest localiser of the human's marks (§2.3). The limit: choreography normally decides rows and lanes jointly, so R2 can only arrange, never re-time; some "weird LN" is R2 holding across rows where the source tapped. A representation of R2's choices should include the source's per-row decision classes as the natural reference.

## 2. Pilot [M]

**Setup.** Lens-labelled sections joined to the R2 cache (`artifacts/r2-style-module-20261006/sections-0{0,1}.jsonl`, fit_train and fit_dev, rate 1), sections with ≥ 8 head rows: 2,888 of 3,268, on 1,021 charts / 992 song groups; rows per section p10/50/90 = 11/36/106; 127 sections carry a human-authority cell, the rest are machine (Astra) labels. Supports absent/supporting/prominent: jack 653/847/493, stream 342/693/1139, trill 1353/261/269, tech 1378/211/141, LN 1506/260/232. Probes: multinomial logistic regression (class-balanced, C ∈ {0.01, 0.1, 1} on an inner grouped split) and 15-NN cosine, 5-fold GroupKFold by song group, macro F1 pooled over folds.
- **A** = the `collapse.py` window statistics on the section rows (17 dims).
- **B** = hand-built structure (48 dims: hand trajectories, jack/trill/stream/stair/jumpstream/chordjack grammar, LN relations, choices conditioned on gap class relative to the section's median gap).
- **C** = 64-d output of a 3-layer dilated Conv1d over per-row tokens (head, LN head, held per lane, log gap, gap relative to local median), trained 420 s / 4,024 steps on 1,500 unlabelled fit_train charts with InfoNCE (positives: two 32-row windows of one chart within 96 rows, each randomly mirrored; loss 5.32 → 2.55, chance 5.55); section = mean over sliding windows. **C0** = the same encoder untrained.

### 2.1 Lens concepts (macro F1, logistic; kNN mean in the last column)

| rep | dims | jack | stream | trill | tech | LN | mean | kNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A window statistics | 17 | 0.832 | 0.747 | 0.598 | 0.517 | 0.691 | **0.677** | 0.624 |
| B structure | 48 | 0.809 | 0.725 | 0.815 | 0.523 | 0.721 | **0.719** | 0.626 |
| C0 untrained conv | 64 | 0.662 | 0.679 | 0.594 | 0.461 | 0.666 | 0.613 | 0.573 |
| C contrastive | 64 | 0.756 | 0.714 | 0.614 | 0.614 | 0.738 | **0.687** | 0.645 |
| A+B | 65 | 0.839 | 0.767 | 0.790 | 0.578 | 0.731 | 0.741 | 0.642 |
| B+C | 112 | 0.823 | 0.759 | 0.791 | 0.624 | 0.744 | 0.748 | 0.662 |
| A+B+C | 129 | 0.833 | 0.775 | 0.793 | 0.640 | 0.754 | **0.759** | 0.663 |
| majority | | 0.199 | 0.229 | 0.279 | 0.296 | 0.287 | 0.258 | |

Sections with ≥ 24 rows only (n = 1,909): A 0.640, B 0.690, C 0.647, A+B+C 0.720. Levels already carry jack and stream; trill needs the relation (B +0.22); tech and LN gain most from the learned residue (C); the three are complementary (+0.08 over A). Tech stays the weakest (≤ 0.64). The earlier 0.587 is **not** comparable (different pool: 974/94 current-evidence machine sections, fixed split, memory-on checkpoint features, different probe); the like-for-like comparison is within this table.

### 2.2 X0, unpaired (12 items, 64-row windows, stride 32: 825 windows, 50 marked; marks by band: 5 → 38, 2 → 7, 4 → 3, 3 → 2)

Labellings: L1 marked vs windows of no-items; L2 marked vs all unmarked; within-yes: marked vs unmarked inside yes-items; item: mean window score vs collapse_any (7 yes / 5 no); gen-vs-real: item mean vs kind (8 / 4). AUC:

| score | L1 | L2 | within-yes | item | gen-vs-real |
|---|---:|---:|---:|---:|---:|
| A rarity (10-NN distance to band-matched real windows, standardised) | 0.956 | 0.823 | 0.755 | 0.914 | 0.719 |
| A rarity without held/bus4/lock | 0.730 | 0.639 | 0.592 | 0.914 | 0.656 |
| A rarity with held share regressed out | 0.609 | 0.573 | 0.554 | 0.714 | 0.656 |
| B rarity | 0.869 | 0.738 | 0.670 | 0.771 | 0.500 |
| B rarity without LN columns | 0.520 | 0.523 | 0.525 | 0.657 | 0.344 |
| A chart-relative deviation (own leave-one-out mean) | 0.816 | 0.689 | — | 0.914 | 0.781 |
| C chart-relative deviation | 0.499 | 0.399 | — | 0.771 | **0.906** |
| C rarity | 0.417 | 0.396 | — | 0.543 | 0.438 |
| Lens probe (A): expected LN-coordination level | 0.887 | 0.798 | 0.752 | 0.829 | 0.656 |
| Lens probe (A): P(tech prominent) | 0.794 | 0.716 | 0.690 | 0.686 | 0.469 |
| Lens probe (A): expected jack level | 0.347 | 0.376 | 0.391 | 0.486 | 0.844 |
| Lens probe (A/B/C): stream level | 0.12-0.16 | 0.23-0.28 | — | 0.14-0.23 | 0.09-0.22 |
| single: ln2 (rows with ≥ 2 held) | 0.926 | 0.820 | 0.766 | 0.886 | 0.625 |
| single: tap while held | 0.906 | — | — | 0.857 | — |
| single: fast jacks (< 100 ms) | 0.758 | 0.667 | 0.621 | 0.857 | 0.844 |
| single: full-jack rate | 0.523 | 0.473 | 0.447 | 0.771 | **0.938** |
| single: hand imbalance | 0.621 | 0.563 | 0.534 | 0.857 | 0.812 |
| single: rep1 / pent / loop / hlock / ng4 | 0.52 / 0.40 / 0.61 / 0.48 / 0.42 | | | 0.77 / 0.37 / 0.60 / 0.51 / 0.40 | |

Per item (mean over windows): held share / ln2 are 0.32/0.45 (x0-02, generated, band 5, 27 marks), 0.37/0.55 (x0-08, **real**, band 5, judged collapsed, 9 marks), 0.15/0.13 (x0-12), against 0.00-0.09 for every no-item. <a id="o-fable-x0-09"></a>x0-09 (generated, band 3, judged yes with four short marks) has held 0.005 and ln2 0.003 and is ranked low by every LN-based and typicality score: the one X0 item the LN story does not cover. Fast jacks are 0.000-0.001 of heads on the four real items and 0.003-0.046 on generated ones (x0-02 4.6 %).

### 2.3 X0, paired on the same rows (8 generated items vs their source charts; 551 windows, 41 marked)

Score = (generated − source) per window, in band SDs of real reference windows.

| paired score | L1 | L2 | within-yes | item (6 vs 2) |
|---|---:|---:|---:|---:|
| Δ ln2 (rows with ≥ 2 held) | **0.991** | 0.862 | **0.832** | 1.0 |
| Δ held share | 0.976 | 0.838 | 0.807 | 1.0 |
| Δ tap while held | 0.943 | 0.801 | 0.769 | 1.0 |
| norm of Δ over B's LN block | 0.643 | 0.790 | 0.823 | — |
| norm of Δ over A without LN / B without LN | 0.530 / 0.530 | 0.460 / 0.422 | 0.444 / 0.397 | 0.75 / 0.83 |
| Δ pattern entropy (reversed sign: generated less diverse than source) | 0.707 | 0.603 | 0.579 | 0.75 |
| Δ jack, Δ full jack, Δ rep1, Δ fast jack, Δ hand imbalance | 0.41, 0.41, 0.41, 0.45, 0.52 | | | |

x0-03 (band 2) sits 14.6 reference SDs above its source in fast jacks and 5.5 SDs in non-LN window statistics (band-2 real charts have no fast jacks), yet those differences do not localise its five marks. Reading: the marks are placed where R2 holds more lanes at once than the source did on the same rows; the jack and repetition complaints are chart-level impressions (the human's note on x0-02: "I didn't examine full song") that show up as item-level fingerprints, not as window marks. "Lack of variance" is visible only in paired form (entropy deficit vs the source, 0.71) and is hidden in the unpaired statistics (pent 0.40).

### 2.4 Limits

Machine labels for about 96 % of Lens cells (the probe measures agreement with the Astra labeller); 380 sections under 8 rows excluded; 12 X0 items with 50 marks, 38 of them in band 5 from two items, and the scores were chosen after seeing X0, so every X0 AUC is optimistic; the encoder saw 7 minutes of CPU; no bootstrap intervals. The X0 "collapse" target includes a real LN chart (x0-08): a measure anchored on "unlike real charts" would pass x0-08; the human did not.

## 3. Recommended representation and measures [P]

<a id="p-fable-representation"></a>Representation **R**: per-row relational tokens over the decoded lane automaton, summarised per 64-row window (and per Lens-style section) in three blocks, each computed on the arrangement *given the supplied rows*, and paired with the source arrangement whenever the skeleton comes from a real chart:
1. **LN relations:** rows with ≥ 2 lanes held, taps while held, LN starts while held, release kind (on head / in gap / release-and-repress), simultaneous releases, hold-length median and spread, held-set change rate. This is what the human marks.
2. **Timing-conditioned pattern grammar with hands:** full/partial jack, jack run length, fast-jack rate (gap < 100 ms), trill and trill run, stream run, stair/roll direction changes, jumpstream, chordjack, hand switch rate, one-hand runs, hand load imbalance, each also split by gap class (fast / on / slow relative to local tempo). The generated fingerprint and the Lens jack/trill/stream material.
3. **Learned residue:** the 64-d contrastive embedding over the same row tokens (mirror-augmented), for tech and within-chart dispersion.

Measures:
- **m-LNx (paired LN excess):** Δ ln2 and Δ tap-while-held vs the source per window in band SDs; chart statistics = share of windows above +1 SD and the longest run of such windows. For BOS runs without a source: 10-NN typicality in blocks 1+2 against band-matched real windows (0.96/0.82 on X0 post hoc).
- **m-FJ (fast-jack rate per head, by band):** real ≈ 0, generated 0.3-4.6 %. A guard, not a window locator.
- **m-ENT (paired entropy deficit):** pent_gen − pent_src per window; the measurable form of "lack of variance".
- **m-Lens (concept trajectories):** predicted level of each concept along the chart from the A+B+C probe; LN-coordination level above the band's real distribution; concept drift between thirds as the measurable form of "difficulty diverging between parts". Tech predictions are weak (F1 0.64).
- **m-DISP (within-chart dispersion of block-3 embeddings):** generated vs real item AUC 0.906 on 12 items; exploratory.
- **G** should be reported per target: "human-marked collapse" (m-LNx, m-ENT) and "generated fingerprint" (m-FJ, full-jack rate, hand imbalance, m-DISP).

## 4. Cheap validation against the human [P]

<a id="p-fable-validation"></a>
1. Preregister on **new** items, not X0: 8 generated + 4 real, plus 2 generated runs with LN suppressed by mask, so the non-LN part of R is tested. Produce window flags from m-LNx and typicality before the human judges. Pass: window AUC ≥ 0.75 within yes-items for m-LNx; typicality ≥ 0.70 on L2; false-flag rate ≤ 10 % of windows on charts the human passes.
2. Cheaper, no new whole-song judging: 10 blind pairs of 20-second clips from the same generated chart, one top-decile by m-LNx and one median; the human says which is worse. ≥ 8 of 10 supports the measure; about 15 human-minutes.
3. Lens side: run the Lens labeller on 50 generated sections (band-stratified) and compare with the A+B+C probe at the real-chart agreement rate. A drop shows the probe does not transfer to generated charts.

## 5. What would show it is wrong [P]

- New marks in LN-free passages or in band 2-3 charts that m-LNx and typicality miss. x0-09 is already one: if the new set has several, block 1 is the human's *first* complaint, not the whole of it, and blocks 2-3 need their own window-level validation.
- m-LNx flagging windows the human passes on a real LN-heavy band-5 chart other than x0-08: then it tracks LN density, not the human's judgment.
- The Lens probe not transferring (item 3 above), or the paired entropy deficit failing to replicate (0.71 on 41 marks is weak).
- Typicality without LN columns at 0.59 within yes-items: if that stays after block 2 is extended with per-window gap-class jack rates and hand runs, the organisational complaints are chart-level only and should be measured as item statistics, never as window flags.

## 6. Runs, paths, not done

- Jobs (bings-mac, 2 threads each, exit 0): `20261008-032153-fr-selftest` (20 s), `20261008-032327-fr-pilot` (455 s), `20261008-033346-fr-x0items` (13 s), `20261008-033532-fr-x0paired` (7 s).
- Outputs: `pilot.md`, `pilot.json`, `x0-windows.csv`, `x0-windows-scored.csv`, `x0-items.csv`, `x0-items.json`, `x0-paired-windows.csv`, `x0-paired-items.csv`, `x0-paired.json`, `selftest/`.
- sklearn 1.7.2 for the probes (n_jobs 1), torch for the encoder; nothing imported from `src/`.
- Failed paths: none; the first pilot run did not save per-window X0 scores, hence the two follow-up scripts.
- Not done: bootstrap intervals on Lens F1 differences and X0 AUCs; a like-for-like rerun of the earlier 0.587 probe; the Lens labeller on generated sections; beat-position conditioning from the grid (gap classes relative to the local median instead); a strain block; the transition-graph block; encoder training beyond 7 minutes.
