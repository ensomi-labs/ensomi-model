<a id="represent-design"></a>Design study by a fresh Opus subagent, returned 2026-10-08 about 05:35 UTC ([r-represent-design](r2-representation-20261008.md#r-represent-design)). It responds to the human's direction [d-projection-spaces](r2-representation-20261008.md#d-projection-spaces) and to the Lens-pool direction [d-lens-pool-primary](r2-representation-20261008.md#d-lens-pool-primary). The source copy is `~/ensomi/.sync/cp/scratch/r2-represent-design/worker/design.md`. Outputs on bings-mac are in `artifacts/r2-represent-20261008/design/`, mirrored.

<a id="v-design-checks"></a>**Main-thread checks (2026-10-08).**
- **Confirmed against the mirrored `design/stats/*.json`:**
  - participation ratio 9.81;
  - chord recurrence over chance: row lag 1 0.20, row lag 2 1.00; beat lag 1/4 0.56, 1/2 0.97.
- **Review notes for the human:**
  - **The proposed pass bar is lenient.** It is a stratified AUC whose interval excludes 0.5, and with thousands of sections almost any signal passes. A bar relative to a baseline would bite: for example, beat a mass-only model within the same strata, or set a minimum effect.
  - **S7 is a fitted count model.** It waits for the human's permission (Q-design in section 8).
  - **Q4 asks to open the calibration and heldout charts** for validation. That breaks a standing rule, so only the human can decide it.
- **Unchanged below:** the body is saved as returned.

---

# Representation spaces for 4K chart sections: a design (represent-design, 2026-10-08)

Fresh Opus worker, brief `~/ensomi/.sync/cp/scratch/r2-represent-design/brief.md`, plus the coordinator's 05:10 direction (Lens pool as primary data). Guardrails 1, 2, 4 and 8 applied as stated in each section. About 3 minutes of mac time at ≤ 2 threads, statistics only, no training, no generation. Tags: **[M]** measured here, **[M-prev]** measured by an earlier worker, **[S]** read in a source, **[I]** inferred, **[P]** proposed.

## 0. Short answer

- **Finished:** principles; a catalogue of eight spaces with mappings, invariances, constants, metrics, Lens expectations and rows relations; a coupling plan; an audit table of the current R; measures and nulls; a Lens-backed build-and-check plan with splits; analogues; open questions. Grounded by corpus and Lens-pool measurements (section 9).
- **Not finished:** no space is built; no Lens separation is measured for any proposed space (by design, to keep the labels clean); the elastic shape variant, the anticipation model (S7) and the chart-level segmentation are specified less tightly than S1-S6; citations other than four were checked from the standard record, not opened.
- **What the measurements changed in the design [M]:**
  - Repetition in human charts is metrical, not sequential: identical chords recur at 1.35-1.57× chance at 1, 2, 4, 8 and 16 beats, against at most 1.21× at any row lag 2-32. The recurrence space is indexed in beats.
  - Lane choices are strongly under-dispersed: the Hellinger distance between two halves of a window is 0.22-0.38 of the multinomial expectation. At ≤ 64 rows, the lane distributions of two different charts differ only 1.0-1.13× as much as two halves of one window. The 4-lane marginal carries almost no section identity; chord-subset distributions (1.4-2.1×) and held-set distributions (2.2-3.0×) carry it. Any null for these distributions has to be fitted, not taken from the multinomial law.
  - A within-section "shape" exists only at low resolution and grows with section size: for chord size, the first DCT coefficient carries 1.4× the energy of a row-shuffled control at 16-32 rows, 6.2× at 128-256 rows; the number of usable coefficients goes from about 1 to about 8. The shape space's resolution must be a fitted function of size.
  - Hand use is anti-persistent: the low-frequency hand profile has 0.2-0.36× the energy of a shuffled control. Mappers restore balance locally.
  - 97 % of beat-relative gaps and 95 % of hold lengths sit at simple ratios; holds fill 1/3, 1/2 or 2/3 of the gap to the next note on their lane. These give fitted alphabets instead of hand-set thresholds.
  - The 63 features of R blocks 1-2 have an effective dimension of about 10 (participation ratio 9.8), 44 pairs at |ρ| ≥ 0.8 and two exact duplicates; held share alone explains rank R² 0.54-0.74 of seven block-1 features.
  - The Lens labels themselves load on level quantities: LN coordination on held share (ρ 0.70 machine, 0.71 human), stream on rows per second (0.38 / 0.45) and against chord size (−0.33 / −0.50), jack on chord size (0.45 / 0.34).
- **The Lens pool now [M]:** 6,039 sections on 3,132 charts, 22,361 settled cells; 230 sections / 598 cells are human. 4,143 sections join the R2 cache. The 1,896 that do not are all in the R2 corpus; the cache's scope excludes them (calibration split 987, heldout 402, star bands outside 2-5: 502).
- **Against the human's pasted inventory (05:15) [M]:** the 2,860 complete five-concept sections reproduce exactly (current-evidence machine or human labels on all five), as do Oct 6 = 1,023, Oct 7 = 1,042, and the remaining 795. Reported, not resolved: on disk the October campaigns hold 2,400 sections with at least one supported label (Oct 6 1,175 including three supplements, Oct 7 1,225). Of these, 2,065 are complete and 335 are not; 335 matches the human's "disputed". So 2,065 is the complete count, while the human's text calls it "2,000 exported plus 65 extras". Only 1,726 of the 2,860 complete sections join the R2 cache; Oct 6 has 596 of its 1,023 outside it.

## 1. Principles

### 1.1 What a space must satisfy

A space is a map π from a section (its supplied rows and its arrangement) to a set X with a metric d. It must state:

1. **Its equivalence.** π(c) = π(c′) exactly when c′ is obtained from c by the space's declared nuisance group G (and its declared coarse-graining). G is written down, and π is checked on synthetic transforms: mirror, time rescale, lane relabelling, truncation to fewer rows [P].
2. **Identifiability of its target.** For the factor the space is meant to see, there exist pairs that differ only in that factor and get different images. Checked by edits that change only that factor, and by the Lens concept it is meant to order (section 5) [P].
3. **One owner per factor.** Each underlying factor (chord mass, LN mass, lane balance, recurrence, contour, hold relations, rhythm-conditioned placement, physical speed, the rows) has exactly one owning space. Every other space is made invariant to it by normalisation, chance-correction or residualisation (section 3) [P].
4. **A rows channel kept apart.** The rows are supplied. Each space states what it shows about the rows alone; that channel is reported but never scored against R2 while the rows are supplied. The arrangement channel is either chance-corrected against a random arrangement of the same rows, residualised on E[· | rows] fitted on the corpus, or paired with a human chart on the same rows (guardrail 4) [P].
5. **Corpus calibration.** Every constant is (a) structural, derived from a declared symmetry, or (b) fitted to the corpus by a named estimator on named charts, or (c) fitted to a null. No threshold is set by hand [P].
6. **A declared null at matched size.** Every distance is read as a quantile of a null at the same section size: within-chart split halves for sampling noise, a random arrangement of the same rows for "no skill", same-song human pairs split by creator, and generator seed against seed for arm comparisons (guardrail 2) [P].
7. **Two scales.** Spaces are defined on sections. A chart is lifted into each space as the empirical distribution of its section points (order-free) plus their trajectory (ordered). Whole-song impressions live in the lift [P].

### 1.2 How the invariances are chosen

The invariances come from what the human and the Lens Foundation compare and what they say does not decide.

| Source | What it says | Invariance or owner it implies |
|---|---|---|
| Formulation, canonical profile | Left-right mirror μ is a symmetry; outer/inner roles are distinct | All spaces μ-invariant; role kept [S]. Corpus: hand contrast mean −0.0001 ± 0.0003 (SE), role contrast −0.017 ± 0.0007 over 3,000 charts [M], so mirror holds and role is a real asymmetry |
| Jack | "Judge the recurrence sequence without using speed alone"; "Density, chord size ... insufficient"; fixed-group vs changing-chord vs localized | Recurrence space in metrical units, blind to ms and to chord mass; speed owned by a separate space [S→P] |
| Stream | "continuity, direction, chord placement, resets"; density or A/B alternation alone insufficient | Contour space: direction-sensitive up to mirror; trill-like zigzag separated by recurrence [S→P] |
| Trill | "Two fixed, disjoint column groups alternate"; same-hand and cross-hand both included | Recurrence space quotiented by lane relabelling: which lanes does not matter [S→P] |
| Tech | "hard to anticipate through familiar patterns"; variation, irregularity, density alone insufficient | Surprise under a model of familiar patterns, residualised on mass and entropy [S→P] |
| LN coordination | "at least two columns occupied by LNs"; "independent control relationships ... order of interacting events"; overlap, LN presence, synchronized holds, one hold with taps insufficient; "No required LN percentage" | Interval-relation space, order-only, normalised per hold; LN mass owned elsewhere [S→P] |
| All five | "Typicality can inform strength; coverage or duration alone does not decide it"; "no universal numerical tolerance" | Blind to duration and row count; tolerances fitted from the corpus; typicality = density in a space [S→P] |
| X0 marks | LN-relation excess on one chart, in place | LN mass (S1) and LN relations (S4), windowed [M-prev] |
| X0 whole-song impressions | Mechanical same-lane jacks, repetition | Chart-level lift of recurrence (S3) and speed (S8) [M-prev, human answer 03:52] |
| Earlier complaints | Poor hand balance; weird LN distribution; difficulty diverging between parts | S1 hand coordinate and its anti-persistence; S4 and S6; chart-level lift of S1 and S2 [I] |

What the labels themselves do (descriptive, all joined labels with ≥ 8 rows, so fit_dev and human cells were touched for description only) [M]:

| Concept | n machine / human | ρ with rows/s | ρ with heads per row | ρ with held share | ρ with row count |
|---|---|---|---|---|---|
| Jack | 2,755 / 62 | −0.07 / −0.05 | **0.45 / 0.34** | −0.16 / −0.10 | 0.05 / −0.06 |
| Stream | 2,982 / 62 | **0.38 / 0.45** | **−0.33 / −0.50** | −0.24 / −0.08 | 0.42 / 0.46 |
| Trill | 2,664 / 65 | 0.33 / −0.07 | 0.28 / 0.16 | −0.21 / −0.08 | −0.22 / −0.20 |
| Tech | 2,453 / 98 | 0.34 / 0.33 | −0.19 / −0.27 | 0.17 / 0.03 | 0.37 / 0.23 |
| LN coordination | 2,788 / 72 | −0.04 / −0.06 | −0.01 / −0.02 | **0.70 / 0.71** | −0.03 / −0.07 |

Reading [I]: the labels are functions of level and organisation together. A space blind to level cannot reproduce a label's marginal ordering, and should not. Each organisation space is checked by its ordering of a concept *within strata of the level that owns it* (S1 for chord and LN mass; the rows channel for rows per second). Stream's dependence on rows per second means part of a Stream label is a property of the supplied rows. Label co-occurrence: tech ~ LN 0.35, stream ~ LN −0.21, others |ρ| ≤ 0.17 [M], so tech labels partly track LN.

### 1.3 Fixed inputs and what they imply (guardrail 4)

| Fixed input | What the spaces can and cannot see | What the evaluation can and cannot see | Complaints it could cause or hide |
|---|---|---|---|
| Supplied head rows | Density, rhythm and the between-part notes-per-second profile are fixed; each space shows them only in its rows channel | A paired comparison cancels the rows; an unpaired one credits or blames R2 for them; stream labels follow rows per second (ρ 0.38-0.45) | Hides "difficulty diverging" when it is the rows' density; can make a stream measure look like R2's doing |
| R2 cache scope (fit split, star bands 2-5) | Spaces can be computed on 4,143 of 6,039 Lens sections; bands 1 and 6-8 unseen | V1 is small unless calibration and heldout sections are opened (Q4); human anchor 164 of 230 sections | Concept behaviour at the difficulty extremes is unchecked |
| Lens scope policy per batch | Pre-October scopes vary up to 555 s; October scopes are about 10 s | Row count and rows per second are confounded inside fixed-duration batches | A space that is not row-count invariant shows a spurious batch effect |
| Machine labels (97 % of cells) | — | Agreement with the labeller, not with the human, except on V2 | A space tuned to the labeller's habits passes V1 and fails the human |
| Same-rows human pairs (92 % same creator) | — | The tight null is one mapper across two difficulties | Overstates "beyond the human spread" unless the different-creator null is used |
| Grid beat with a factor-2 fold | Beat-unit coordinates move by a factor 2 between charts | Lags and gap categories must be tatum-relative or powers of two | A fold artefact read as a style difference |

Guardrails 1, 2 and 8 are applied in sections 5 and 6: no space decides anything before it passes a preregistered check on labels kept apart from its design; every distance is read in a size-matched null; claims are tagged, and every ratio has its denominator in section 9.

## 2. Catalogue of spaces

**Notation.** A section has rows k = 1..n with supplied times t_k and grid beat positions b_k. The grid's beat carries a factor-2 fold ambiguity: the corpus summary records a dominant fold of −1 for 14,581 and 0 for 6,822 charts, and the median chart's grid beat is 600 ms [M]. Head sets are A_k ⊆ {1,2,3,4}, LN-head sets L_k ⊆ A_k, and H_k is the set held entering row k. Holds are intervals [s_j, e_j) on lane l_j, complete through exact replay (entering holds included). Lanes have hand h and role r: lane 1 (+1, outer), 2 (+1, inner), 3 (−1, inner), 4 (−1, outer). For a lane function f, the hand × role Walsh coefficients are F_0 = Σ f, F_h = Σ h f, F_r = Σ r f, F_hr = Σ h r f. Mirror μ maps (F_0, F_h, F_r, F_hr) → (F_0, −F_h, F_r, −F_hr). Orbit distance under a finite group G: d_G(x, y) = min_{g∈G} d(x, g·y).

### S1. Mass and lane distribution: no time, no order (the human's example 2)

- **Encodes and compares:** how much of each kind of action there is per row, and how it is spread over lanes, chord types and held sets. The bag of row events {(A_k, L_k, H_k)} with order and gaps forgotten.
- **Indistinguishable by design:** time, order, tempo, position, duration, row count (everything is per row), mirror. Rows per second are invisible: they belong to the rows.
- **Mapping [P]:**
  - Mass, polar part 1: m_tap = (1/n) Σ |A_k \ L_k|, m_ln = (1/n) Σ |L_k|, m_held = (1/n) Σ |H_k|. Held occupancy is counted in row slots, the time-free reading.
  - Distribution, polar part 2: lane distributions p_κ = f_κ / F_0(f_κ) for κ ∈ {tap, LN head, held}; chord-subset distribution q_A over the 15 non-empty subsets; held-set distribution q_H over the 16 subsets.
  - Interpretable coordinates: the Walsh invariants of each p_κ: F_r/F_0 (outer vs inner), (F_h/F_0)² (hand imbalance), (F_hr/F_0)² (diagonal), F_h F_hr / F_0² (their relative sign). Under μ the 15 chord subsets fall into 9 classes: outer single, inner single, same-hand pair, outer pair {1,4}, inner pair {2,3}, diagonal pair {1,3}/{2,4}, triple missing an outer lane, triple missing an inner lane, quad.
- **Constants and curves, fitted [P]:**
  - Mass transform g: chosen from the Box-Cox family by requiring the within-chart split-half variance of g(m) to be independent of the level of m, on fit_train charts. No pseudo-count by hand.
  - Null scale for distributions: E[d_H²] = α n^(−β), fitted on within-chart split halves by size. Measured for the head lane distribution: 0.0248 at 16 rows, 0.0090 at 32, 0.0036 at 64, 0.0019 at 128, 0.0010 at 256, so β ≈ 1.15 [M]. The multinomial law 3/8 (1/N_a + 1/N_b) over-predicts by 2.6-4.5× [M], so it is not used.
- **Metric [P]:** Hellinger (Fisher-Rao) on p_κ, q_A and q_H, taken as the orbit distance under {id, μ}, divided by the fitted null scale at the pair's harmonic size; |g(m) − g(m′)| over its null SD for each mass. Hellinger because the square-root map is the variance-stabilising chart of the simplex. When mass and distribution must be compared jointly, the Hellinger-Kantorovich (unbalanced transport) distance has one constant, the transport length scale, fitted so that split-half distances are size-stable.
- **Should separate / must be blind to:**
  - Lens: orders LN coordination through m_held and the held-pair classes of q_H (ρ 0.70 with held share [M]). Orders jack positively and stream negatively through chord mass (ρ 0.45, −0.33 [M]). For trill and tech, S1 should show only the level effects (ρ 0.28 and −0.19 with chord size [M]); its distribution coordinates should add little once mass is fixed.
  - Where S1's identity sits [M]: the mean Hellinger² between blocks of different charts, over that between adjacent halves of one window, is 1.0-1.13 for the head-lane distribution at 16-64 rows (1.5 at 128). It is 1.44 / 1.64 / 1.87 / 2.13 for the chord-subset distribution at 16 / 32 / 64 / 128 rows, 2.2-3.0 for the held-set distribution and 1.4-1.7 for the LN-head lane distribution (μ-orbit distances, 2,000 charts). The head-lane marginal carries little identity at section level; chord classes and held sets carry it.
  - Generated vs real: hand balance. Corpus charts have chart-level SD of F_h/F_0 of 0.014 [M], so an imbalanced generated chart stands out at chart level.
  - Same vs different mapper: LN mass ICC 0.68 vs 0.20 [M-prev], so m_ln and m_held should separate the two pair types.
  - Complaints: LN excess (X0 marks, through m_held and ≥ 2-held classes), hand balance.
- **Relation to the supplied rows:** rows contribute only n, which is normalised away. The rows predict part of the mass: chart-level R² 0.41 for chord rate, 0.22 for held [M-prev]. The arrangement's own choice is the residual g(m) − E[g(m) | rows features], fitted on fit_train with the RICH row features of opus-skeleton, or the paired difference with the source.

### S2. Within-section shape across durations and row counts (the human's example 1)

- **Encodes and compares:** the form of intensity through the section: rising, falling, a peak, a plateau of holds. Comparable between a 3-second 20-row section and a 20-second 200-row one.
- **Indistinguishable by design:** duration, row count, tempo, absolute level of each channel (owned by S1), mirror. Not indistinguishable: time reversal (a build-up is not a fade) and the position of features within the section, unless the human answers Q1 the other way.
- **Mapping [P]:**
  - Normalised position u_k = (t_k − t_1)/(t_n − t_1). Time, not row index: the measured order structure of held and LN-head profiles is 1.3-1.9× stronger in time than in row index; for chord size the two agree [M].
  - Arrangement channels: chord size c_k = |A_k|, LN heads ℓ_k = |L_k|, held count η_k = |H_k|. Rows channel: ρ_k = 2/(t_{k+1} − t_{k−1}).
  - Each channel binned on u into M = 16 bins (bin means, empty bins interpolated), divided by its section mean, transformed by the orthonormal DCT-II. Keep coefficients a_1..a_K(n).
  - An LN channel is defined only when its section count reaches m_min (fitted: the count where split-half variance of a_1 falls below between-section variance). Below it the channel is missing, never zero.
- **Constants and curves, fitted [P]:**
  - K(n), the resolution, from the energy ratio of real to row-shuffled profiles on corpus sections: keep a_k while the ratio exceeds 1 by more than its bootstrap SE. Measured for chord size, time parametrisation: coefficient 1 ratio 1.38 at 16-32 rows; 2.09 / 1.51 / 1.27 for coefficients 1-3 at 32-64; 3.8 / 2.4 / 1.8 / 1.5 / 1.2 at 64-128; above 1.4 for coefficients 1-8 at 128-256 [M]. So K ≈ 1, 3, 5 and 8 in these classes.
  - Per-coefficient null variances σ_k²(n) from adjacent windows of the same chart.
- **Metric [P]:** for sections of sizes n and n′, d² = Σ_{k ≤ min(K(n), K(n′))} (a_k − a′_k)² / σ_k². Comparison happens at the coarser resolution of the two, which is what makes different row counts comparable. Variant arm: an elastic distance (square-root velocity transform, warp penalty λ fitted so that within-chart adjacent windows sit at the same null quantile as the default). Full warping invariance is not the default, because it forgets where in the section a rise happens.
- **Should separate / must be blind to:**
  - Lens: expected mostly blind. The five concepts are local organisation, not intensity form. A pass is stratified AUC near 0.5 for jack, stream, trill and tech. Exception: LN coordination may associate with held-count plateaus.
  - Generated vs real, and the complaint "difficulty diverging between parts": level divergence between sections is S1's chart-level lift. S2 owns the form within sections and the dispersion of forms across a chart. Generated window chord SD is +20 % (B0) to +45 % (B1) over the source on the same rows [M-prev].
  - The hand channel is excluded from the shape vector: it is anti-persistent (shape-energy ratio 0.20-0.36 [M]). Its departure toward 1 ("hands drift like random order") is a single scalar coordinate, owned by S1's chart lift as hand-balance restoration.
- **Relation to the supplied rows:** the rows channel ρ(u) is fully fixed by the rows; its shape ratio is 2.4-13.6 at coefficient 1 [M], the strongest shape in a chart, and it is not R2's. The arrangement's shape given the rows is a − B ã_ρ, where ã_ρ are the rows channel's coefficients and B is a ridge map fitted on fit_train. Window chord level against row density correlates −0.20 in sources and −0.27 in B0 [M-prev]; that coupling is a coordinate of this space.

### S3. Metrical recurrence: repetition, jacks and trills without lane identity or speed

- **Encodes and compares:** how the arrangement repeats itself at metrical distances; jack and trill organisation as recurrence relations.
- **Indistinguishable by design:** which lanes (any relabelling of the four lanes, which includes mirror), the section's chord distribution (chance-corrected), tempo and ms (lags in beats), row count, position.
- **Mapping [P]:**
  - Lag pairs P_τ = {(i, j): |b_j − b_i − τ| < ε_b} for τ in a fitted lag set T.
  - Exact repetition: r_same(τ) = mean over P_τ of 1[A_i = A_j], divided by Σ_m q_A(m)², the chance level under the section's own chord distribution.
  - Partial repetition: r_over(τ) = mean Jaccard(A_i, A_j) divided by its chance expectation under independent draws from q_A.
  - Sequential relations, chance-corrected the same way: lane sharing between consecutive rows (jack relation); alternation, A_{k+2} = A_k with A_{k+1} ∩ A_k = ∅ (trill relation); the distribution of maximal same-lane run lengths (jack runs).
  - Long-range: determinism of the recurrence matrix R_ij = 1[A_i = A_j], the share of recurrent pairs on diagonal segments of at least ℓ_min rows; Lempel-Ziv phrase rate of the chord string after canonical relabelling (lanes renamed in order of first appearance).
  - All recurrence equalities are invariant under any lane permutation, so the quotient by all 24 lane relabellings holds by construction.
- **Constants and curves, fitted [P]:**
  - T: the peaks of corpus recurrence. Measured, identical chord over chance: 1/4 beat 0.56, 1/2 0.97, 1 beat 1.35, 1.5 1.13, 2 1.48, 3 1.27, 4 1.53, 6 1.32, 8 1.57, 16 1.49 [M]. At row lags: 0.20 at lag 1, then 0.89-1.21 for lags 2-32 [M]. So T = {1, 2, 4, 8, 16} beats plus the sequential relations; powers of two are robust to the grid's factor-2 fold.
  - ε_b: from the corpus distribution of beat positions around simple ratios (97.3 % of beat gaps fall within ±0.04 octave of a simple ratio [M]).
  - ℓ_min: the length where the corpus diagonal-length distribution departs from the row-shuffled one.
- **Metric [P]:** log excess ratios, Mahalanobis with within-chart split-half covariance at matched size. For whole songs, the chart lift.
- **Should separate / must be blind to:**
  - Lens: orders jack (consecutive lane sharing, jack runs) and trill (alternation) within S1 chord-mass strata. Must not separate one-hand from cross-hand trills (Foundation: both included). Weak for stream.
  - Generated vs real: full jacks and repetition. Unpaired full-jack rate separated generated from real at item AUC 0.94 on X0 [M-prev].
  - Complaints: "mechanical same-lane jacks" and "repetition, lack of variance", both at chart level.
- **Relation to the supplied rows:** the rows' own rhythm recurs (the same bar rhythm), and that bounds where the arrangement can repeat. Isolate the arrangement by restricting P_τ to pairs whose local rhythm context matches (same gap categories before and after, S6's alphabet): "given the same rhythm, does the arrangement repeat?".

### S4. Hold relations: order only, no time scale

- **Encodes and compares:** how complete holds relate to each other and to taps: staggered, nested, synchronised, released and re-pressed; the order of interacting events; how much of the available gap a hold fills.
- **Indistinguishable by design:** absolute time and tempo (relations are invariant to any increasing time map), LN mass (normalised per hold; S1 owns it), mirror. Kept: whether two related holds are on the same hand or across hands, and outer/inner.
- **Mapping [P]:**
  - Objects: holds [s_j, e_j) on lane l_j, from exact replay including entering holds, and taps as points.
  - Interacting pairs: objects on different lanes that overlap, or where one starts within one section tatum τ_0 of the other's end. τ_0 is the modal gap in the section, fitted per section.
  - Relation of each hold-hold pair: Allen's relations up to inversion: before-within-τ_0, meets, overlaps, starts, during, finishes, equals. Tap-hold: coincident with press, during, coincident with release. Same-lane successor: release-and-repress gap in tatums.
  - Histogram π over (relation × hand relation of the pair), normalised by the number of holds.
  - Fill ratio φ_j = (e_j − s_j) / (t_next(l_j) − s_j), with t_next the next head on that lane. Hold length in tatums, log2.
  - Event grammar: the time-ordered sequence of press, release and tap events, simultaneous events grouped, coded by hand relation to the previous active hold; bigram distribution.
- **Constants and curves, fitted [P]:**
  - Categories of φ from the corpus density. Measured: median 0.50, quartiles 0.33 and 0.53, histogram modes in [0.30, 0.35), [0.50, 0.55) and [0.65, 0.70), with a secondary one at [0.75, 0.80) [M]. Hold lengths: median 0.25 beat (167 ms), 94.8 % at simple beat ratios [M].
  - τ_0 per section; no ms constants.
- **Metric [P]:**
  - π: transport (W1) with ground metric equal to path length in Freksa's conceptual-neighbourhood graph of Allen relations, where two relations are neighbours if a continuous deformation turns one into the other. This is a structural metric, not a hand-set one.
  - The alphabet is not degenerate [M, 426,309 holds, 623,943 interacting pairs, 2,000 charts]: meets 37 % (27 % cross-hand), before within one gap 33 %, equals 7.0 %, finished-by 7.1 %, starts 6.4 %, overlaps 6.4 %, contains 3.3 %. Independent relations (overlaps, contains) are about 10 % of pairs, synchronised ones (starts, finishes, equals) about 20 %. Taps on other lanes fall at a release 44 %, at a press 32 %, during a hold 25 %. The shares are identical at 2 ms and 10 ms tolerance because releases are snapped, so no tolerance constant is needed.
  - φ: W1 on [0, 1], the L1 distance between quantile functions.
  - Event bigrams: Hellinger with the fitted null.
  - All in null units at matched number of holds.
- **Should separate / must be blind to:**
  - Lens: orders LN coordination within held-share strata. This is the key test: the Foundation says the LN percentage does not decide, so S4 must order the label beyond S1. Expected direction: share of independent relations (overlaps, during with independent press and release) up; pure synchronised relations and one-hold-with-taps down (both named exclusion cues). Blind to jack, stream and trill on LN-free sections, where S4 is undefined, not zero.
  - X0: the marks are LN-relation excess; S1 owns the excess in mass, S4 says whether the relations themselves are unlike human ones.
  - Generated vs real, mapper: R2's LN-head placement agrees with the source at r 0.08, humans at 0.45 (different creator) to 0.67 (same creator) [M-prev]. S4's fill-ratio and relation distributions are where a placement failure should show as a distribution difference.
  - Complaint: "weird LN distribution".
- **Relation to the supplied rows:** hold starts sit on supplied rows; releases are R2's choice, on a head row or in a gap. The rows give the release slots; the share of releases on head rows is partly a rows property. Isolate by the paired comparison with the source and by a random-arranger null that keeps the rows and each row's LN count.

### S5. Contour: direction and continuity without time

- **Encodes and compares:** how the arrangement moves across the lane line: stairs, rolls, zigzags, resets, chord placement within a flow.
- **Indistinguishable by design:** time and tempo (order kept), mirror (negation of all steps), mass (per row), the section's lane marginals (chance-corrected).
- **Mapping [P]:**
  - Lane position x(l) = l − 2.5. For consecutive single-note rows, steps Δ_k = x_{k+1} − x_k ∈ {−3..3}. Chord rows are coded by size class and the position of the chord's centre, so a flow's chord placement is kept.
  - Step-bigram distribution over (Δ_k, Δ_{k+1}), compared as an orbit under μ (negation).
  - Runs: lengths of maximal monotone unit-step runs (stairs, rolls) and reset rate.
  - Chance correction: expected bigram distribution when positions are drawn independently from the section's own lane distribution. The excess decouples S5 from S1.
- **Constants:** none beyond structural ones; the roll condition "evenly spaced rhythm" is read from S6's rhythm alphabet, not set in ms.
- **Metric [P]:** W1 on the bigram plane with ground metric |Δa| + |Δb|, orbit under μ, null units.
- **Should separate / must be blind to:**
  - Lens: orders stream (continuity, direction runs). Trill zigzags also look like alternating steps here; the Foundation calls pure trill episodes stream-absent, so stream vs trill separation needs S3 and S5 jointly. That is declared, not hidden.
  - Blind to LN coordination and to chord mass.
- **Relation to the supplied rows:** none except the rhythm regularity that defines a roll group, taken from S6's rows-only alphabet.

### S6. Placement given the rhythm: the arrangement conditioned on the rows

- **Encodes and compares:** where, in the rows' rhythm, the arrangement puts chords, LN heads, jacks, hand switches and releases. This is the space defined by the rows-arrangement relation, so it isolates R2's own decisions by construction.
- **Indistinguishable by design:** tempo (contexts in tatum units), mirror, position, size, and the marginal rate of each decision (owned by S1; S6 holds log-odds ratios against the section's own marginal).
- **Mapping [P]:**
  - Context of row k: categories of the gap before and after, in units of the section tatum τ_0 (fold-free), plus metric phase from the grid (flagged as fold-dependent). The fold is visible in the corpus: a chart's modal gap is 1/4 grid beat in 1,134 of 1,927 charts, 1/8 in 499 and 1/2 in 207 [M]. Tatum units remove it; beat units do not.
  - Decisions: chord-size class, LN head present, jack (shares a lane with the previous row), hand switch, release on this row or in the preceding gap.
  - Coordinates: λ_{d,c} = logit P(d | c) − logit P(d), shrunk by a beta-binomial prior whose centre and strength are fitted on the corpus.
- **Constants and curves, fitted [P]:**
  - Gap categories: valleys of the corpus density of log2(gap / τ_0). Measured on beat gaps: bin masses peak at 1/4 beat (0.24), 1/8 (0.17), 1/2 (0.07), 1/6 (0.03), 1/12 (0.02), 1/16 (0.01), 1/3 (0.01); successive gaps are equal 72.6 % of the time, halve or double 17.9 %, ×1.5 2.0 %, ×3 2.1 % [M]. This replaces the hand-set 0.75 × and 1.5 × local-median classes.
  - Prior strength by maximum marginal likelihood on fit_train.
- **Metric [P]:** Mahalanobis on λ with within-chart split-half covariance; for pairs on the same rows, the row-level agreement of decisions as a second reading.
- **Should separate / must be blind to:**
  - Generated vs real: this is where R2 is furthest from humans. Chord placement r 0.30 vs 0.735-0.79 for humans, LN heads 0.08 vs 0.45-0.67 [M-prev].
  - Same vs different mapper: LN-head placement 0.67 vs 0.45 [M-prev].
  - Lens: tech ("rhythm or articulation relationships"); jack through jacks at fast tatums vs slow. Blind to mass.
- **Relation to the supplied rows:** contexts are rows-only; the rows channel is the context histogram, which R2 and the source share exactly.

### S7. Anticipation: surprise under a model of familiar patterns (built last)

- **Encodes and compares:** per row, information content IC_k = −log p(A_k, L_k | history, rows) under a corpus model of familiar patterns; the section's mean IC and IC profile, residualised on S1 mass and S3 recurrence.
- **Indistinguishable by design:** whatever the model's viewpoints forget: lane relabelling for the recurrence viewpoint, mirror, tempo. Mass and plain repetition through residualisation.
- **Mapping [P]:** a variable-order Markov model with multiple viewpoints (canonical chord string, contour step, rhythm context), fitted by counting on fit_train song groups outside every validation group, with order and blending chosen by held-out log-loss on the corpus. This is a fitted model and so comes after the training ban is lifted or the human allows count-based fitting.
- **Metric [P]:** W1 between IC distributions (1-D transport); mean IC difference in null units.
- **Should separate / must be blind to:** Lens tech beyond S1 and S3 entropy-like coordinates (the Foundation excludes plain variation). Generated vs real: "lack of variance" as an IC deficit against the source on the same rows; collapse as falling IC across the song. Must be blind to LN mass; the label correlation tech ~ LN 0.35 [M] is a coupling to check, not a target.
- **Relation to the supplied rows:** the rows enter only as conditioning context.

### S8. Physical timing: speed kept in milliseconds on purpose

- **Encodes and compares:** the time the arrangement leaves a finger and a hand between actions: same-lane and same-hand intervals in ms, including jack speed.
- **Indistinguishable by design:** mirror; lane identity within a hand role. Not tempo: physical feasibility is absolute.
- **Mapping [P]:** distributions of same-lane and same-hand inter-action intervals (press to press, release to press), reported as the ratio of the chart's cumulative distribution to that of a random arrangement of the same rows (lanes drawn uniformly among free lanes, chord sizes and LN counts kept), at the corpus deciles of each band.
- **Constants:** the evaluation grid is the corpus deciles; no 100 ms threshold.
- **Metric [P]:** W1 on log intervals relative to the random-arranger null.
- **Should separate / must be blind to:** generated vs real (fast jacks are 0-0.1 % of heads on real X0 items and 0.3-4.6 % on generated ones [M-prev]); the complaint "mechanical same-lane jacks" with S3. Blind to Lens organisation: the Foundation says speed alone does not decide.
- **Relation to the supplied rows:** the rows' gaps in ms set the shortest possible interval; the random-arranger null removes that.

### Lens expectations per space, in terms of the pool [P]

"Orders" means the stated coordinate should rank absent < supporting < prominent *within strata of the owning level* (S1 mass terciles; rows-per-second terciles for stream and tech). "Blind" means stratified AUC near 0.5. "—" means no claim. Counts are per-concept cells in the whole pool (absent / supporting / prominent); the joined design and validation subsets are smaller (section 6).

| Space | Jack 1,559 / 1,995 / 1,048 | Stream 783 / 1,800 / 2,144 | Trill 3,424 / 556 / 394 | Tech 2,974 / 561 / 538 | LN coordination 3,132 / 661 / 792 |
|---|---|---|---|---|---|
| S1 mass and lanes | Orders through chord mass (unstratified only; it owns the level) | Orders inversely through chord mass (same) | Level effect only (ρ 0.28 with chord size); its distribution coordinates should add little at fixed mass | Level effect only (ρ −0.19); same | Orders through m_held and held pairs (same) |
| S2 shape | Blind | Blind | Blind | Blind | Weak: held plateaus |
| S3 recurrence | Orders: lane sharing, jack runs | Weak, jointly with S5 | Orders: alternation; blind to one-hand vs cross-hand | — | Blind |
| S4 hold relations | Undefined on LN-free sections | Undefined | Undefined | — | Orders within held-share strata: the decisive check |
| S5 contour | Blind | Orders: continuity, direction runs; with S3 against trill | Separated from stream only jointly with S3 | — | Blind |
| S6 placement | Weak: jacks by gap category | — | — | Orders weakly | Weak: LN heads by gap category |
| S7 anticipation | — | — | — | Orders beyond S1 and S3 | Must not follow LN mass |
| S8 physical | Blind (speed alone does not decide) | Blind | Blind | Blind | Blind |

For trill, tech and LN the absent class dominates, so the main contrast is absent against present. Supporting against prominent is checked but not required.

**Other separation targets per space [P; evidence from earlier rounds marked M-prev]:**

| Space | Generated vs real | Same vs different mapper | Human complaint |
|---|---|---|---|
| S1 | Hand imbalance (X0 item AUC 0.81, unpaired [M-prev]) | LN mass: ICC 0.68 vs 0.20 [M-prev] | In-place LN excess (held share 0.32 on x0-02, generated, and 0.37 on x0-08, real, vs ≤ 0.09 on items the human passed [M-prev]); hand balance |
| S2 | Within-section form; window chord spread +20-45 % over the source [M-prev] | Unknown | Part of "difficulty diverging between parts" |
| S3 | Full-jack rate (item AUC 0.94 unpaired [M-prev]); long-lag repetition | Unknown | Mechanical jacks; repetition |
| S4 | LN placement (LN-head r 0.08 vs ≥ 0.45 [M-prev]) seen as relation and fill distributions | Expected, given LN-head r 0.67 vs 0.45 | Weird LN distribution |
| S5 | Unknown | Unknown | Stream quality; lack of variance in motion |
| S6 | Chord and LN placement given the rows (0.30 / 0.08 vs human 0.735-0.79 / 0.45-0.67 [M-prev]) | LN-head placement 0.67 vs 0.45 [M-prev] | Weird LN distribution; jacks at the wrong rhythm |
| S7 | IC deficit against the source | Unknown | Lack of variance; collapse over the song |
| S8 | Fast jacks (0-0.1 % of heads real vs 0.3-4.6 % generated on X0 [M-prev]) | Not expected | Mechanical same-lane jacks |

### The chart-level lift (not a separate space)

- For each space, a chart is the empirical measure of its section points μ_c = (1/S) Σ_s δ_{x_s}, plus the trajectory (x_1, ..., x_S).
- Whole-song measures: dispersion (mean pairwise distance in null units), drift along the trajectory, section-level recurrence, and the distance between two charts' section clouds (W2, or MMD with kernel scale at the null median).
- Section unit for whole songs: metrical windows of W beats, W fitted from the corpus recurrence (8-16 beats carry the highest recurrence [M]), with sensitivity at W/2 and 2W. Novelty segmentation is the alternative (Q5).

## 3. Coupling and independence

**Ownership table [P].**

| Factor | Owner | How the others are kept from it |
|---|---|---|
| Rows: rhythm, density, tempo | Rows channel of S2, S6, S8 | Never scored while the rows are supplied; arrangement channels conditioned or residualised |
| Chord mass | S1 | S2 divides by the section mean; S3 corrects by chance under q_A; S5 per row with lane-marginal chance; S6 holds log-odds against the marginal; S7 residualised |
| LN mass | S1 | S4 normalises per hold; the S2 LN channel divides by its mean; S6 holds log-odds against the marginal |
| Lane balance (hand, role) | S1 | S5 chance-corrects against the section's lane marginals; S3 is lane-relabelling invariant |
| Recurrence | S3 | S7 residualised on S3; S5 uses only adjacent steps |
| Direction | S5 | S3 cannot see it (relabelling invariance) |
| Hold relations | S4 | S1 sees only held-set mass, not relations |
| Rhythm-conditioned placement | S6 | Every other space is rhythm-free or uses the rhythm only as a declared context |
| Speed (ms) | S8 | S3, S4, S5 and S6 are in beats, tatums or order only |

**Declared couplings that remain [I]:** stream needs S3 and S5 jointly; LN coordination needs S1 (necessary condition) and S4; the Lens labels themselves couple tech and LN (ρ 0.35).

**How to check on the corpus [P]:**
1. On fit_train sections, the distance correlation between every pair of spaces' coordinate blocks, against a permutation null; and the R² of each block from all other blocks (gradient boosting, grouped CV). Pass: below a level fixed from the split-half reliability of the block (a block cannot be explained by others better than it explains itself).
2. One-factor test: R² of each space's coordinates from S1 mass alone should be near zero by construction for S2-S6. The current R fails this badly (section 4).
3. Aggregation rule: spaces are never summed into one distance. Any combined score is a model fitted on the design labels that reports each block's contribution, and must not fit better than the sum of single-block fits suggests (a check for double counting).

## 4. Audit of the current R

Measured coupling [M]: 18,000 corpus windows of 64 rows from 3,000 fit_train charts, 63 features of blocks 1 and 2 (lrows, ldtmed removed):
- 44 feature pairs at |Spearman ρ| ≥ 0.8, 148 at ≥ 0.6;
- PC1 explains 25 %, 5 PCs 57 %, 25 PCs reach 90 %; participation ratio 9.8;
- exact duplicates: `rep1` ≡ `jack_full` (ρ 1.00), `hmax` ≡ `hand_bal` (ρ 1.00);
- held share alone gives rank R² 0.54-0.74 for `held`, `ln_share`, `tap_held`, `heldset_chg`, `o_ln`, `ln_stag`, `lock`; held share, chord density and log row rate together give R² ≥ 0.5 for 17 of the 63.

| Prior or coupling | Where | What it does to the comparison | Effect on the evaluation |
|---|---|---|---|
| 64-row window, stride 32 | X0, null, label-free test | A row-count window spans 2-8 s depending on the rows' density; content mixes the rows' density into every feature | Dense passages are flagged per second faster; window distances partly compare rows [I] |
| Block norm = RMS of features standardised by human-pair SD, equal weights | Block 1 and 2 norms | Seven block-1 features are one factor (held share, R² 0.54-0.74) | The block-1 label-free AUC 0.80 cannot be told apart from a held-share AUC; LN relations vs LN mass unresolved [I from M] |
| Gap classes at 0.75 × and 1.5 × the window's median gap | `f_*`, `o_*`, `s_*` | The corpus gaps are categorical (97 % at simple ratios; successive gaps equal 72.6 %); a median over a window mixes categories | Class membership shifts with window content, not rhythm [I from M] |
| Fast jack at < 100 ms | `fjack` | Tempo-dependent: corpus gaps have median 112 ms, so 100 ms sits inside the ordinary range of the faster half of charts, where common 1/4- and 1/8-beat jacks count as fast | Generated-vs-real fjack AUC 0.84 mixes jack speed with tempo [I] |
| Loop over row periods 2-16, 4-grams, 16-symbol entropy | `loop`, `ng4`, `pent` | Repetition in human charts is metrical (1.35-1.57× chance at 1-16 beats vs ≤ 1.21× at row lags) | Row-lag repetition misses most of the repetition the corpus has [I from M] |
| Lane identity kept in pattern counts | `pent`, `ng4`, trill groups | A 1-2 trill and a 3-4 trill count as different symbols | Repetition and entropy partly measure lane choice, not organisation [I] |
| Hold length as log1p(median ms) and CV | `hold_lmed`, `hold_lcv` | Tempo-dependent; corpus holds sit at simple beat ratios and fill 1/3, 1/2, 2/3 of the gap | LN length differences mix tempo with LN choice [I from M] |
| `inhand_alt_L`, `inhand_alt_R` as separate coordinates | Block 2 | Not mirror-invariant as a vector | Distances change under mirror unless scales match [I] |
| Rows properties in A | `lrows`, `ldtmed` in the pilot's A | Lens probe A (0.677) included rows | Part of A's Lens F1 is the rows' density, which also drives stream labels (ρ 0.38-0.45) [I from M] |
| Token scaling log1p(dt)/7, clip(log(dt/median), −2, 2)/2, missing first gap = 200 ms | Block 3 tokens | Hand-set normalisations in ms | The embedding sees tempo and density [I] |
| InfoNCE τ = 0.1, batch 256, lr 1e-3, positives within ±96 rows, 32-row windows, wall-clock training (420 s, then 240 s) | Block 3 | "Same chart, nearby" is the equivalence; positives share the rows; no seed spread of the embedding | It learns chart identity including rhythm; refits differ (loss 2.55 vs 2.75); untrained control already 0.613 of 0.687 Lens F1 [M-prev] |
| Null q90 pooled, 92 % same creator | Null units | The threshold is one mapper's spread across two difficulties; different creator 1.8× wider | Flags overstate "beyond the human spread" [M-prev] |
| One-sided flag direction chosen after seeing X0 | m-LNx | A post-hoc prior on direction | X0 AUCs optimistic [M-prev] |
| MIN_ROWS 8, span ±10 ms, ≥ 32 rows for span features | Data selection | Hand-set inclusion | 380 of 4,143 joined Lens sections have < 8 rows and are dropped [M] |
| Lane distribution read against no fitted noise | `lane_ent`, `lmax`, `hand_bal` | Lane choices are under-dispersed (0.22-0.38 of multinomial noise) and at ≤ 64 rows differ between charts only 1.0-1.13× as much as within a window | Section-level lane features carry little identity; null scales from a multinomial would be 2.6-4.5× too wide [M] |

## 5. Measures and evaluation on top

**Natural measures per space [P].**

| Space | Measure | Unit | Target |
|---|---|---|---|
| S1 | LN mass excess over E[· given rows] and over the source; held-pair class excess; chart-level hand imbalance; typicality as kNN density among band-matched corpus sections | Null quantile | In-place LN marks; hand balance |
| S2 | Arrangement-shape distance to the source on the same rows; chart dispersion of shapes | Null quantile | Within-section form; part of "diverging parts" |
| S3 | Chart-level recurrence excess at 1-16 beats; jack-run and full-jack excess | Null quantile | Whole-song jacks and repetition |
| S4 | Relation-distribution distance; fill-ratio W1; event-grammar distance | Null quantile | Weird LN distribution; LN relations in place |
| S5 | Step-bigram excess distance; roll-run rates | Null quantile | Stream organisation |
| S6 | Decision log-odds distance to the source; row-level decision agreement | Null quantile; r | R2's placement given the rows |
| S7 | IC deficit vs the source; IC drift over the song | Null quantile | Lack of variance; tech |
| S8 | Same-lane interval ratio to the random-arranger null at the lowest corpus decile | Ratio, null quantile | Mechanical fast jacks |

**Nulls, every one at matched size (guardrail 2) [P]:**
1. Within-chart split halves and adjacent windows: sampling noise.
2. Random arrangement of the same rows (legal lanes drawn uniformly, chord sizes and LN counts kept): the "no-skill arranger" floor.
3. Same-song human pairs on near-identical rows (`pairs2.parquet`), always split: same creator (568 same-band pairs) and different creator (47 same-band, 97 in all). A threshold names its null; the different-creator null is the honest "another human".
4. Generator seed against seed and neighbouring checkpoints, for any comparison between arms; never a selected checkpoint as the only baseline.

**Validation labels, kept apart from design labels (guardrail 1) [P]:**
- Lens concept levels: each space is scored on what it should order (ordinal AUC between adjacent levels, Kendall τ-b) *within strata of the owning level* (S1 mass terciles, rows-per-second terciles), and on what it must be blind to (stratified AUC near 0.5 with its interval).
- Label-free: same vs different creator pairs; R2 on A's rows vs human B on the same rows (with the different-creator subset reported separately).
- X0 (12 items) and C0 (48 pairs): final sanity checks only, never design. Too small to carry a decision.
- No measure selects, ranks, stops or trains anything until it passes a preregistered criterion on validation labels. Until then the spaces characterise.

## 6. Build-and-check plan, with the Lens pool as backbone

**Two units of evidence [P, following the human's 05:15 note].**
- **Complete sections:** 2,860 with current labels on all five concepts. Used wherever a check needs several concepts at once: blindness checks, stratified orderings that condition on another concept, and the coupling of labels. 1,726 join the R2 cache now [M].
- **Per-concept labels:** every supported cell, including the supported cells of the 335 disputed October sections. Used for single-concept orderings. 4,143 sections join [M].

**Splits, fixed before any space is computed [P].** Two axes, crossed: source batch and song group. Song groups never cross splits; 912 pairs of overlapping sections on 251 charts stay inside one group [M-prev].

| Batch (complete sections joined now: fit_train / fit_dev) [M] | Role |
|---|---|
| Pre-October machine (v3 and other eligible): 502 / 43 | Design |
| Oct 6 campaign: 389 / 38 (596 more complete sections outside the cache) | Design, then swapped with Oct 7 as a batch check |
| Oct 7 campaign: 640 / 56 (346 more outside the cache) | Validation, then swapped with Oct 6 |
| Human, the 600 observations on 230 sections (92 complete; 164 sections joined) | Validation anchor, reported separately, never used for design |

| Split | Content | Use |
|---|---|---|
| Corpus fit set | fit_train charts outside every Lens validation song group | All label-free constants: null scales, K(n), lag set, gap and fill categories, Box-Cox, priors, E[· given rows] maps. Most of each space's constants come from here and need no labels |
| D, design labels | Machine labels from pre-October and Oct 6, on fit_train song groups that hold no human label and no Oct 7 section | The few label-dependent choices: stratum boundaries, probe forms |
| V1, machine validation | Oct 7 labels on song groups disjoint from D, plus fit_dev of every batch; if the human allows (Q4), the Lens sections on calibration and heldout charts | Preregistered pass or fail per space |
| Batch check | Swap the October batches (design on pre-October + Oct 7, validate on Oct 6); also leave out the pre-October batch | The October execution setup changed (25-section batches, stricter completeness, retry); a space whose result moves with the batch is reported as batch-dependent. Scope policy also differs (variable scopes up to 555 s before October, about 10 s in October), so this doubles as a row-count-invariance check |
| V2, human validation | The 600 human observations: 164 sections joined now (147 fit_train, 17 fit_dev, 141 song groups); 66 more with a cache extension | Reported separately with song-group bootstrap intervals; wide (62-98 cells per concept in the joined subset) |
| Sensitivity | All-evidence machine labels, adding "changed" and "untracked" evidence: 3,657 complete sections instead of 2,860 | Whether older evidence moves a result |

**Order, smallest first. Mac compute is estimated from this round's timings: 3,000 charts with window features in 13 s; 4,143 Lens sections joined and summarised in 3 s; 2,000 charts of shape and recurrence statistics in about 60 s [M].**

1. **Section table** (≈ 1 min): exact-replay objects for the 4,143 joined Lens sections and 16-beat windows of 3,000 corpus-fit charts; synthetic transforms (mirror, ×1.5 tempo, truncation, lane relabelling) for the invariance unit tests. Kill rule: any space whose invariance test fails exactly is stopped.
2. **S1** (≈ 1 min): fit Box-Cox and α, β on corpus split halves; unit tests; on D: LN-coordination ordering by m_held and q_H, jack and stream by chord mass, blindness of trill and tech within mass strata; label-free same vs different creator on LN mass.
3. **S3 and S5** (≈ 2-3 min; lag pairs are O(n²) per section, sections are short): fit T, ε_b, ℓ_min; on D: jack and trill within chord-mass strata; stream for S5, and stream vs trill jointly.
4. **S4** (≈ 2 min): fit φ categories and τ_0; on D: LN coordination within held-share terciles. This is the decisive check that LN relations are seen beyond LN mass.
5. **S6** (≈ 3 min): fit the gap alphabet and priors; same vs different creator pairs; R2 on A's rows vs human B, using the 240 step-1 generations already on the mac (`artifacts/r2-represent-20261008/step1/gen/`), no new generation.
6. **S2** (≈ 2 min): K(n) with bootstrap SE, null σ_k²; blindness on D; arrangement shape given the rows on the human pairs.
7. **Coupling** (≈ 3 min): distance correlations and cross-block R² on corpus sections.
8. **S8** (≈ 1 min) with the random-arranger null. **S7** (≈ 10-20 min of counting) only after the human allows count-based model fitting.
9. **Validation, once, preregistered:** V1 machine, then V2 human, then X0 and C0 as sanity. Proposed pass for an organisation space: stratified ordinal AUC on V1 above 0.5 with its 95 % song-group bootstrap interval excluding 0.5 for each concept it claims, and stratified AUC within [0.4, 0.6] for each concept it must be blind to. V2 is reported with intervals and does not by itself pass or fail a space.

Total for steps 1-7: about 15 minutes of mac compute at 2 threads.

## 7. Analogues

Citations: a helper checked bibliographic details of all of them; items marked [S] were checked against a fetched source (Müller et al. 2005 ×2, Quinn 2006-07 bibliographic record, Gouyon et al. 2006). The rest rest on the standard record, not a fetched source; claims below are kept to what the titles and well-known abstracts state. Freksa 1992 and Székely et al. 2007 were not checked at all.

| Analogue | Invariance it solves | Transfers | Does not transfer |
|---|---|---|---|
| Chroma / pitch-class profile (Fujishima 1999, ICMC; Bartsch & Wakefield 2001, WASPAA) | Octave and time forgotten; pitch class kept | S1: a lane histogram forgets time as chroma forgets octave; compare after quotienting | Chroma's group is cyclic transposition on 12 classes; lanes carry hand × role structure with only a mirror symmetry |
| DFT of pitch-class sets (Lewin 1959, JMT 3(2); Quinn 2006-07, PNM 44(2), 45(1); Amiot 2016, Springer) | Fourier magnitudes invariant to transposition and inversion | S1's Walsh coefficients on the hand × role group, with μ-odd coefficients squared, are the same construction for our group | The group is tiny, so there are only five invariants |
| CENS (Müller, Kurth & Clausen 2005, ISMIR) [S] | Robustness to local tempo by short-time statistics and downsampling | S2: resolution chosen by statistics over windows, K(n) | They handle global tempo at matching time; our sections are normalised in time instead |
| Scale transform (Holzapfel & Stylianou 2011, IEEE TASLP 19(1)); tempo octave errors, "Accuracy 2" (Gouyon et al. 2006, IEEE TASLP 14(5)) [S, factors read from the abstract only loosely] | Tempo invariance of rhythm; metrical-level ambiguity | S3, S6: lags and gaps in beat or tatum units; the grid's factor-2 fold is the same problem as a tempo octave error, handled by power-of-two lags and tatum-relative gaps | We have a symbolic grid, so no audio periodicity estimation is needed |
| Rhythmic categories (Desain & Honing 2003, Perception 32(3)) | Performed intervals fall into ratio categories | S6: gap categories fitted from the corpus density (97 % at simple ratios [M]) | Their categories are perceptual; ours are notational, closer to snapping |
| Rhythm similarity measures (Toussaint 2004, ISMIR) | Distances between onset patterns | S6: swap-type distances are transport on onset positions | They compare single rhythms; we compare conditional choices |
| Earth Mover's Distance (Rubner, Tomasi & Guibas 2000, IJCV 40(2)); melodic transport distance (Typke et al. 2003, ISMIR) | Distribution comparison with a ground metric | S4 (Allen graph metric), S5 (step plane), S1 alternative | The ground metric must be structural or fitted; it is a constant |
| Unbalanced transport (Chizat et al. 2018, J. Funct. Anal. 274(11); Liero, Mielke & Savaré 2018, Invent. Math. 211) | Mass and distribution compared together | S1 joint variant with one fitted length scale | The default keeps mass and distribution apart, one owner each |
| Elastic curves (Srivastava et al. 2011, IEEE TPAMI 33(7)); Kendall shape space (1984, Bull. LMS 16(2)); DTW (Sakoe & Chiba 1978, IEEE TASSP 26(1)); PAA (Keogh et al. 2001, KAIS 3(3)); registration (Ramsay & Silverman 2005) | Shape modulo reparametrisation, scale, translation | S2: under the square-root velocity representation the elastic metric becomes L2 and reparametrisation acts by isometries, which gives the warp-invariant variant; PAA is S2's binning | Full warp invariance would forget where in a section a rise happens; kept as a variant pending Q1 |
| Relational motion features (Müller, Röder & Clausen 2005, ACM TOG 24(3)) [S, invariance details from the paper body] | Qualitative geometric relations between body points, robust to time warping and global placement | S4 and S5: relations between holds and between positions instead of coordinates | Bodies have continuous geometry; lanes are four discrete points |
| Interval algebra (Allen 1983, CACM 26(11)); conceptual neighbourhoods (Freksa 1992, Artif. Intell. 54; not checked) | Order-only relations between intervals | S4's relation alphabet and its ground metric | Allen ignores durations entirely; the fill ratio φ adds them back scale-free |
| Recurrence plots and RQA (Eckmann, Kamphorst & Ruelle 1987, EPL 4(9); Marwan et al. 2007, Phys. Rep. 438); novelty (Foote 2000, ICME) | Repetition structure independent of content | S3 determinism and laminarity; chart-level segmentation | RQA usually needs a hand-set recurrence threshold; chord identity is discrete, and ℓ_min is fitted |
| Lempel-Ziv complexity (Lempel & Ziv 1976, IEEE TIT 22(1)) | Repetition as compressibility | S3 phrase rate after canonical relabelling | Sensitive to length; read against the size-matched null |
| Multiple viewpoints (Conklin & Witten 1995, JNMR 24(1)); IDyOM (Pearce 2005, PhD, City University); information dynamics (Abdallah & Plumbley 2009, Connect. Sci. 21(2-3)) | Several derived symbolic views, each with its own invariance; expectation as information content | The whole catalogue is a viewpoint system; S7 is IC under familiar patterns | IC models a listener's expectation of melody; tech is a player's anticipation of patterns, so S7 must show it orders tech beyond entropy |
| Dance Dance Convolution (Donahue, Lipton & McAuley 2017, ICML) | Step placement separated from step selection | Rows given, arrangement chosen: S6 evaluates selection given placement | Their selection was scored by perplexity, not against style concepts |
| Spectrum kernel (Leslie, Eskin & Noble 2002, PSB) | n-gram counts as features | S5 step bigrams, S4 event bigrams | — |
| Kernel two-sample test (Gretton et al. 2012, JMLR 13); distance correlation (Székely, Rizzo & Bakirov 2007, Ann. Stat. 35(6); not checked) | Calibrated distribution comparison; dependence between blocks | Chart lift (MMD with null-median kernel scale); coupling check | — |

## 8. Open questions for the human (each changes the design)

1. **Shape (S2):** is *where* in a section a rise or peak happens part of its overall shape (a build-up at the start differs from one at the end), or should shape forget placement and keep only the sequence of rises and falls? The first gives the low-DCT L2 metric; the second the elastic, warp-invariant one.
2. **Mass (S1):** mass per row (time-free; what R2 controls given the rows) or per beat or second (what a player feels, but then partly the rows')? The brief's example reads as per row; confirming fixes S1's units.
3. **Lane identity in jacks and trills (S3):** compare up to any relabelling of lanes (a 1-2 trill equals a 1-4 trill, as the Foundation's trill definition allows), or keep the hand structure (one-hand vs split)? This picks S3's group: all lane permutations, or only mirror.
4. **Validation reserve:** may the 1,389 Lens sections on calibration and heldout charts (49 of them human) be opened as the validation set for the spaces? That needs a cache extension (or reading the corpus `.osu` directly) and breaks the rule that those charts stay closed. It would roughly quadruple V1.
5. **Section unit for whole songs:** fixed metrical windows (16 beats, fitted), or a segmentation by novelty, or the Lens labelling scopes?
6. **Speed (S8):** is speed in ms a legitimate part of the "mechanical jacks" judgment? The Foundation says speed alone does not decide; X0's strongest generated fingerprint is fast jacks. S8 exists only if yes.

## 9. Data and grounding statistics

### 9.1 The Lens pool (inventory rerun 2026-10-08 05:13 UTC, same reducer as the 2026-10-06 inventory) [M]

- **Current resolved pool:** 6,039 sections on 3,132 charts; 22,361 settled cells (11,872 absent, 5,573 supporting, 4,916 prominent); 3,657 sections with all five concepts; no conflicting cells; all under Foundation `15fa6891...` (version 2, human-approved 2026-09-05); playback rate 1 only.
- **Provenance:** 598 human cells (600 records) on 230 sections / 198 charts, confidence High 171, Low 23, unspecified 406. 21,763 machine cells, all `agent-reviewed` (current machine admission requires an agent review). Auxiliary evidence of machine cells: current 16,508, changed 2,399, untracked 2,856.
- **Since the 10-06 inventory:** +1,325 sections, from astra-new-1000 supplements 02 (75) and 03 (25) and astra-next-1000-20261007 (1,225 sections, 5,922 cells).
- **Main batches (supported cells; joined fit_train / fit_dev sections):** astra-next-1000 5,922 (761 / 70); astra-new-1000 initial 4,867 (389 / 41) + supplements 849 (67 / 9); subagent-annotation-1000 4,616 (710 / 66); corpus-500-original 2,363 (1,466 / 128; variable scopes up to 555 s); scale-500 2,420 (338 / 31; evidence "changed"); smaller repairs and pilots ≤ 234 each.
- **Per concept (absent / supporting / prominent; missing):** jack 1,559 / 1,995 / 1,048 (1,437); stream 783 / 1,800 / 2,144 (1,312); trill 3,424 / 556 / 394 (1,665); tech 2,974 / 561 / 538 (1,966); LN coordination 3,132 / 661 / 792 (1,454). Human: jack 32 / 45 / 23; stream 15 / 41 / 43; trill 79 / 22 / 22; tech 86 / 39 / 27; LN 82 / 28 / 14.
- **Join limits:** 4,143 sections join the R2 cache (3,793 fit_train, 350 fit_dev; 1,896 charts). The other 1,896 sections (1,236 charts) are all in the R2 corpus, so the limit is the cache's scope, not missing data: calibration split 987, heldout 402, fit split outside star bands 2-5 502 (band 1: 161, 6: 258, 7: 80, 8: 3), fit in-band 5. Human: 164 joined; 66 not (calibration 44, heldout 5, fit out-of-band 13, fit in-band 4).
- **Section sizes (joined):** rows p10 / p50 / p90 = 8 / 42 / 109; 380 sections under 8 rows; durations p50 7.1 s, p75-p90 10 s (fixed-scope campaigns), p99 165 s.
- **Reconciliation with the human's pasted inventory (05:15):**
  - Agrees: 2,860 complete sections (= the "strict current evidence" scenario, all five concepts); Oct 6 1,023 (initial 872 + supplements 59, 68, 24); Oct 7 1,042; remainder 795 (my split: 92 human-only complete sections + 703 pre-October machine; the human's split is v3 736 + other 59, which I did not check because v3 membership is not in the section manifest); 600 human observations; v3's machine sections not all complete (the 10-06 inventory gives 620 complete of 954).
  - Disagrees, not resolved: "2,000 exported sections plus 65 extras". On disk, the October campaigns have 2,400 sections with at least one supported label, 2,065 complete, 335 not complete. The 335 match; the 2,000 + 65 equals the complete count, not the section count.
  - Join of the complete set: 1,726 of 2,860 join the cache (pre-October 545, Oct 6 427, Oct 7 696, human-only 58); 1,134 do not (Oct 6 596, Oct 7 346, pre-October 158, human-only 34).

### 9.2 Corpus statistics (3,000 fit_train charts unless stated) [M]

| Statistic | Result | Design use |
|---|---|---|
| Beat-relative gaps | 97.3 % within ±0.04 octave of a simple ratio; bin peaks 1/4 (0.24), 1/8 (0.17), 1/2 (0.07) beat | S6 alphabet; ε_b |
| Successive gap ratio | equal 72.6 %, ×2 or ÷2 17.9 %, ×1.5 2.0 %, ×3 2.1 %, ×4 1.3 % | S6 context |
| Gaps in ms | p10 / p50 / p90 = 60 / 112 / 227 ms; chart median grid beat 414 / 600 / 706 ms | Grid fold; S8 grid |
| Hold lengths | p50 0.25 beat (167 ms); 94.8 % at simple beat ratios | S4 lengths in tatums |
| Hold fill of the next same-lane gap | p25 / p50 / p75 / p90 = 0.33 / 0.50 / 0.53 / 0.71; modes near 1/3, 1/2, 2/3 | S4 φ categories |
| Lane Walsh contrasts per chart | hand −0.0001 (SE 0.0003, SD 0.014); role −0.017 (SE 0.0007, SD 0.038); diagonal 0.0012 (SE 0.0003, SD 0.015) | μ quotient; role kept |
| Lane Hellinger², observed / multinomial | 16 rows 0.38 (adjacent halves), 0.38 (far, same chart), 0.38 (other charts); 32: 0.27 / 0.27 / 0.28; 64: 0.22 / 0.27 / 0.25; 128: 0.23 / 0.24 / 0.35; 256: 0.25 / 0.28 / 0.33 | S1 null law; low section-level lane identity |
| Shape energy, real / row-shuffled, coefficient 1 (time) | chord 1.38 / 2.09 / 3.80 / 6.21 at 16-32 / 32-64 / 64-128 / 128-256 rows; held 3.0 / 5.6 / 5.9 / 7.2; hand 0.33 / 0.36 / 0.27 / 0.25; row rate 2.4 / 4.0 / 7.7 / 13.6 (2,000 charts) | S2 K(n); hand anti-persistence; rows dominate shape |
| Recurrence of identical chords over chance | row lag 1: 0.20; row lags 2-32: 0.89-1.21; beat lags 1/4: 0.56, 1/2: 0.97, 1: 1.35, 2: 1.48, 4: 1.53, 8: 1.57, 16: 1.49 (2,000 charts) | S3 lag set in beats |
| Identity of S1 distributions: Hellinger² other charts / adjacent halves (μ-orbit) | head lanes 1.0-1.13 at 16-64 rows; chord subsets 1.44 / 1.64 / 1.87 / 2.13 at 16 / 32 / 64 / 128 rows; held sets 2.23-2.99; LN-head lanes 1.36-1.74 (2,000 charts) | Which S1 coordinates carry section identity |
| Allen relations of interacting holds | meets 37 %, before 33 %, equals 7.0 %, finished-by 7.1 %, starts 6.4 %, overlaps 6.4 %, contains 3.3 %; same at 2 and 10 ms tolerance (623,943 pairs) | S4 alphabet; no tolerance constant |
| Taps on other lanes relative to holds | at release 44 %, at press 32 %, during 25 % | S4 tap relations |
| Chart modal gap, in grid beats | 1/4: 1,134; 1/8: 499; 1/2: 207 of 1,927 charts | Fold; tatum units in S6 |
| Coupling of R blocks 1-2 | section 4 | Audit |

### 9.3 Runs and paths

- Scripts (control plane, synced): `~/ensomi/.sync/cp/scratch/r2-represent-design/worker/` — `inventory_rerun.py` (copy of `r2-style-module-20261006/inventory.py` writing to a new directory; reads only), `stats.py`, `join_limits.py`, `shape_rec.py`, `s1s4.py`; this document as `design.md`.
- Jobs (bings-mac, exit 0, ≤ 2 threads, about 3 minutes in total): `20261008-051304-rd-inventory`, `-051501-rd-stats-smoke`, `-051535-rd-stats`, `-051638-rd-joinlim`, `-051720-rd-joinlim2`, `-051840-rd-shaperec`, `-053053-rd-s1s4`.
- Outputs (mirrored): `ensomi-model/artifacts/r2-represent-20261008/design/inventory/` (`inventory.json`, `sections-0{0,1}.jsonl`, `human-gold.json`, …), `design/stats/stats.json`, `design/stats/join-limits.json`, `design/stats/shape-rec.json`, `design/stats/s1s4.json`; `design/stats-smoke/` is the 40-chart smoke run.
- Limits of the statistics: the coupling R² uses ranks against raw regressors (approximate); the Lens dependence table touched fit_dev and human labels for description only; the shape and recurrence statistics use one random window per size class per chart; the grid beat carries a fold ambiguity, so beat-unit numbers are in grid beats.
