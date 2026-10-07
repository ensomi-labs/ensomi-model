# Decode-time selection design, Fable (2026-10-07)

Report of the fresh Fable subagent for [r-decode-design](r2-bakeoff-night-20261007.md#r-decode-design), copied unchanged from `~/ensomi/.sync/cp/scratch/r2-decode/fable/report.md`. Probe outputs: `artifacts/r2-decode-20261007/fable/` on bings-mac (mirrored); scripts in `~/ensomi/.sync/cp/scratch/r2-decode/fable/`. The main thread spot-checked the probe table against the report; the design is a proposal.

# Fable report: a stateful whole-sequence score for decode-time selection in R2 (2026-10-07)

Design round for [r-decode-design](r2-bakeoff-night-20261007.md#r-decode-design). Started 18:55 UTC, delivered about 19:20 UTC. Two probes on existing runs (section 7); no generation, no tracked file touched. Claims are marked **measured**, **inferred** or **proposed**.

**Bottom line.** The assumption "the charts we want are in p_θ, plain sampling does not find them" has a weak form that X0 already supports (2 of 8 plain samples passed the human's whole-song screen, so the acceptable mass per song is of order 0.1-0.4, not 10⁻³) and a strong form that is untested: that a computable score can *find* those samples. Sequence likelihood under a real-fitted model, over row symbols or over block statistics, is the wrong finder: it ranks by density and LN rarity, not organisation (probes A and B, measured). The stateful part that can see collapse is regime *durations* under a per-chart-type law; used rank-only and clipped at the real-chart range ("satisficing", not maximising), inside SMC at block boundaries. The first experiment is a whole-song rerank of N = 32 plain samples per song, judged by paired whole-song calls, which tests the assumption and the ranker at once for about 4 Mac-hours and 1 human-hour.

## 1. The assumption

**Formal statement (proposed).** Fix a skeleton S. Let A(S) be the set of whole charts the human passes on a whole-song screen (unobservable except through the screen). The assumption is about the acceptable mass m(S) = p_θ(A(S) | S) and about findability:

- weak form: m(S) is not tiny, so that best-of-N at a feasible N contains an acceptable chart with high probability, 1 − (1 − m)^N;
- strong form: there is a computable score s(x) whose ranking puts acceptable charts first, so that top-1-of-N by s passes at a rate well above m.

Any selection from N samples moves the output law q away from p_θ by at most KL(q‖p_θ) ≤ log N nats (the best-of-N bound), so selection cannot reach charts with m ≪ 1/N. For N = 32 the reachable mass is about 3 % per song; for N = 128 about 1 %. Anything rarer is a training problem by construction.

**What is already known (measured).** X0 is a best-of-1 screen: 2 of 8 generated songs passed (x0-05 band 3, x0-10 band 4), 6 failed; 3 of 4 real foils passed. The Clopper-Pearson 90 % interval for m from 2/8 is 0.05-0.60. So the weak form holds at any N ≥ 8 unless the two passes were luck of the band (both are mid bands; the band-2 and band-5 items all failed). X1 adds that a 64-row real prefix removes half of the band-2 offset for the whole 1k rows, so a good start is sampled sometimes and holds for a while.

**Cheap test of the strong form (proposed): best-of-N with a paired blind screen.** For each of 8 long songs, draw N = 32 plain samples (about 4 s each with the incremental sampler, 2 min per song), rank them by the candidate score, and show the human the top-ranked and one random sample of the same song, blind and in random order, with 2 real foils. Section 5 gives the protocol and section 6 the pass rule. Cost: about 20 Mac-minutes of generation plus the human's hour.

**Mass estimate without the human (proposed, a proxy).** On 100 long charts × N = 32, compute per song the fraction of samples whose whole-song profile lies inside the sources' range on measures *not used for ranking*: stay rate ≤ 0.35, first-to-last-third correlation of nh and held ≥ 0.7 against the sample's own first third, envelope-exit rate ≤ 1.25× the band's source rate, no head-lock run of 30+. The per-band mean of that fraction is m̂ on the statistical proxy; it is reported next to the human's.

**What shows it false (proposed).** Either of:
- the human passes the top-ranked-of-32 in no more songs than the random sample (the ranker finds nothing), *and* m̂ < 1/32 in the proxy, in 3 of 4 bands: the acceptable charts are not in the reachable mass; training must change (on-policy objective or the anchor arms), and selection stays as a guard only;
- the top-ranked passes but the proxy measures of the selected charts sit outside the real range in the *other* direction (sparser, less LN, lower variety than any source): the ranker found bland charts, not good ones; the score is being gamed (section 4).

If the top-ranked passes in ≥ 5 of 8 songs with the random arm at ≤ 3, the strong form holds for that score.

## 2. Candidate stateful scores

All candidates share the same row observation o_t: the 4-bit head pattern, the lanes held before the row, the skeleton gap bucket (the gap is given, so it conditions emissions rather than being scored), and optionally the generator's own per-row entropy and log-probability of the sampled action. All decompose as a sum of per-row or per-block terms with a small carried state, so the score of a block or a whole song is a sum, computed forward, causally, once.

| | (a) Explicit-duration HMM over row symbols | (b) Segmental HSMM over block statistics, chart-level mixture | (c) Linear-chain CRF / HCRF discriminator | (d) Semi-Markov CRF over sections | (e) Twisted SMC with a learned value |
|---|---|---|---|---|---|
| State | regime z_t ∈ {1..S}, S = 6-8 (stream, jack, LN-held, chord-dense, sparse, mixed), plus elapsed duration d_t ≤ D | regime z_b per 16-row block, duration d_b, and a chart type c ∈ {1..C} drawn once (C = 8-16 clusters of whole-chart θ) | label y_t ∈ {ok, degenerate} or hidden h_t; sequence label real/own | segments [s_j, e_j) with label and length 16-256 rows | the SMC particle's prefix; value V(x_{1:t}) ≈ E[final score] |
| Potentials | duration law p(d \| z) (histogram to D = 128 rows), transition A, emission p(o_t \| z, gap bucket), optionally autoregressive p(o_t \| o_{t−1}, z) | per-state Gaussian over standardised block statistics (nh, c3, held, jack, rep1, pattern entropy, lane max, bus4), duration law per state and chart type in blocks, transition per chart type, initial state per band | node φ(y_t, f_t) with f_t = symbol, 2- and 4-gram context, gap bucket, generator entropy; edge ψ(y_{t−1}, y_t) | segment feature g (statistics, their change from the previous segment, duration); label transition | the score's forward message as the twisting potential; optional critic |
| Sees: absorption durations | yes, directly: p(d \| z) is the object; the 2× stay rate is a duration anomaly | yes, in blocks, per chart type (probe B says the law must be per type) | only through the edge potential (geometric) | yes | inherits the twisted score |
| Sees: start regime | weakly (initial state per band) | yes: p(c \| band, skeleton summary) and p(z_1 \| c) | no | weakly | inherits |
| Sees: held identity | no: states are local | yes: a mid-song switch of chart type has low likelihood under every c | no | partly (segment-to-segment change) | inherits |
| Sees: local organisation (F3) | partly; needs the autoregressive emission or the 2-gram context | no (block statistics average it out); add rep1 and the 4-gram ratio as block features | yes, through the context features | partly | inherits |
| Sees: LN and jack patterns | yes (symbols) | yes (held, jack, bus4) | yes | yes | inherits |
| Fitted on | real charts only (EM); contrastive version also on own runs → LLR | real charts only; contrastive duration laws on own runs | real vs own runs (discriminative); or the human's marks (12 items: too few) | needs segment labels: from real-chart section boundaries (block-statistic change points) | the base score plus rollouts |
| Per-row cost | S·D multiply-adds ≈ 1k flops (HSMM), 36-64 (HMM) | C·S² per block ÷ 16 rows ≈ 50 flops per row | F + 4 flops, F ≈ 50 | S·L per row with L segment lengths ≈ 500 | the base score; the critic is a small MLP |
| Decomposition | log p(x_{1:t}) = Σ log c_t (forward normalisers); state α_t carried across blocks | same, per block, for each c; log-sum over c at the end, or fix c once | per-row log-odds via forward messages | per-segment terms | per block |
| Combination with log p_θ | rerank: log p_θ(x) + λ s(x); or twisting in SMC | same; the chart type c plays the role of B3's θ at decoding | product of experts p_θ·exp(λ·logit) | rerank | SMC |

Every per-row cost is below 1 % of the 1.5 ms generation step (measured in probe A: 822 whole songs scored in 10 s with numpy).

**Reading (inferred, with the probes as evidence).** (a) in its plain HMM form is a density model of row symbols; its log-likelihood ranks charts by how common their symbols are (sparse, LN-free, jack-free), so it is both a weak finder and the most gameable (section 4, probe A). Its contrastive LLR cancels the rarity term but then mostly learns the band offset of the checkpoint it was fitted against. What makes (a) worth keeping is the explicit duration law, which neither the plain HMM nor the hand-built envelope has: a sticky HMM assigns long stays *high* likelihood, while p(d | z) prices a too-long stay per run. (b) is the score that matches the collapse formalisation (F1 by the initial law, F2 by the chart type held for the whole song, absorption by durations); probe B shows that durations without a chart type are confounded by chart type (uniform real charts exist, x0-01), which is why the duration law must be conditional on c. (c) is a "generated detector": it will find the model's fingerprints (the mirror symmetry of the head, the F3 adjacency deficit) that search cannot fix, and it inherits the GAN-discriminator failure of being satisfied by statistically real-looking charts. It is the right tool for the *held-out* evaluator, not for selection. (d) is the human's description ("difficulty diverging between parts", "repetitive stretches") in model form, but needs segment labels, so it comes after (b) has given regime paths to label with. (e) is the search, not a score; it makes (a)/(b) usable in real time.

**Recommended score (proposed): (b) with (a)'s duration law**, used rank-only, as s(x) = Σ_b [log p(φ_b | z_b, c) + log p(d | z, c) at regime ends] with c fixed once per song (drawn from p(c | band, skeleton) or taken as the type of the first 128 rows), plus the generator's log p_θ(x)/K as a *floor*, not a term: candidates below the 10th percentile of the plain samples' log p_θ/K are dropped before ranking.

## 3. Search over whole sequences

| Option | What it does | Mac time for 100 long charts (K ≈ 2,500, 1.5 ms per row) | Causal / real-time |
|---|---|---|---|
| Whole-song rerank of N samples | N plain samples per song, rank by s (and the log p_θ floor), keep one | N = 16: 1.7 h; N = 32: 3.3 h; N = 64: 6.7 h. Scoring is negligible | not causal: picks after the song ends. Offline only |
| SMC with P particles, resampling at 64-row boundaries, weights ∝ exp(λ Δs_block) (the score's forward message is the twisting function) | keeps good prefixes alive; for an additive score it matches best-of-N at N ≫ P | P × 3.75 s per song: P = 8: 50 min; P = 16: 1.7 h | causal by block. After resampling, particle ancestries coalesce; the prefix common to all particles can be published. Coalescence is typically 2-4 blocks (128-256 rows, 10-30 s of play) at P = 8, so publish-ahead needs that lag. Per-row compute P × 1.5 ms = 12 ms at P = 8, inside the 69 ms p95 budget |
| Beam over blocks, width B, N candidates per entry | tonight's d0 is B = 1, N = 4 | B·N × 3.75 s: B = 4, N = 4: 1.7 h | causal with the same coalescence lag as SMC; deterministic, loses diversity faster than SMC |
| Rerank at the end of SMC | keep all P final particles, choose by s or by the held-out floor | free | offline |

**Mac night at hand (inferred):** the rerank at N = 32 on 100 charts (3.3 h) plus SMC P = 8 on the same 100 (50 min) both fit one night with the human-screen generation (20 min).

## 4. Reward hacking

| Score | How generator + search games it | Signature to watch |
|---|---|---|
| Raw real-fitted likelihood (a, b without contrast) | bland safe charts: the modal symbol (single head, no hold) wins; LN avoidance; jack avoidance; in bands 4-5 the selected set drifts sparser than any source | **measured in probes A and B**: Spearman of log p with held −0.85 to −0.89, jack −0.72 to −0.80, nh −0.47 to −0.50 on L0 runs; AUC source-vs-run 0.86-0.90 in band 2 but 0.36-0.49 in bands 4-5 |
| Contrastive LLR against own runs | moves away from the fitted checkpoint's habits whatever they are: for 48M/64M that is "be busier" (LLR vs nh +0.69 to +0.90, probe A; +0.76 probe B); against 56M it would be the band offset again | selected set's band offset flips sign |
| Duration law (a, b) | regimes of exactly the modal duration; sections that end on schedule rather than on music ("mechanical") | run-length histogram of the selected set narrower than the sources' |
| Chart-type mixture (b) | copying the cluster centre: F1 made permanent; loss of between-chart variety | between-chart SD ratio of the selected set < 0.8 of the sources' |
| CRF discriminator (c) | fingerprint removal, not quality; statistically real-looking collapse | discriminator score of selected charts above the real charts' median |
| Any score with λ large | the KL budget spent on the score's blind spots; variety across seeds collapses | KL proxy: mean log p_θ/K of the selected set outside the plain samples' interquartile range |

**Safeguards (proposed, all cheap).**
1. **Satisficing clip, not maximising.** A candidate's score is compared with the real charts' per-band score distribution; any candidate inside [q05, q95] of the real range is "acceptable" and the choice among acceptable candidates is random (or by the log p_θ floor), never by the highest s. Scores above the real q95 are treated as suspicious, not as better. This removes most of the bland-chart pull in one move.
2. **The log p_θ floor** (section 2), which keeps selection inside the model's typical set, and the **KL budget**: N ≤ 32 or P ≤ 16, λ chosen so that the selected set's mean log p_θ/K stays within the plain samples' interquartile range.
3. **Held-out evaluators never used in selection**: stay rate, first-to-last-third correlation, envelope-exit rate, adjacency MI, between-chart SD ratio, head-lock runs, and the LN-share distribution per band against the sources. The CRF discriminator (c) joins this set.
4. **Variety guards**: between-seed and between-chart SD ratio ≥ 0.8 of the sources on nh, held, jack, pattern entropy; LN share per band within the source interquartile range (so LN avoidance shows up).
5. **Rank-only use**: no score ever becomes a training signal or a logit bias; the generator is frozen.
6. **The human screen is the final judge**, with real foils so that the screen's own false-positive rate (x0-08) is visible, and with the random-sample arm so that "better than plain" is measured, not assumed.

## 5. Human blind screen protocol

Built on what the human does reliably: whole-song calls and pairwise preference on the same skeleton; not time ranges.

- **Items.** 8 long songs (K ≥ 1,500; 2 per band), each shown as a **pair**: the top-ranked sample and one random sample of the same N = 32, same skeleton, so song difficulty and skeleton cancel. Plus 2 real foils (one LN-heavy, since x0-08 showed the screen's LN sensitivity). 18 items.
- **Presentation.** Random order, blind labels, the pair members not adjacent. The human may skim; X0 took about 3 min per item, so about 1 hour.
- **Calls per item.** `collapse_any` (yes/no), `playable` (yes/partly/no), confidence (high/low), free type note. Per song, after both members are seen (revealed as a pair only at the end, still blind to arm): "which of the two is the better chart" (forced choice) and "are they the same chart type" (yes/no), the latter a variety check.
- **Pass (proposed, fixed in advance).** Over the 8 pairs: the selected arm is called collapsed in ≤ 2 songs, the random arm in ≥ 4; at least 4 pairs are discordant in favour of the selected arm and none against (one-sided sign test, p = 0.06 at 4/4; p = 0.03 at 5/5); the forced choice favours the selected arm in ≥ 6 of 8. Both foils called not collapsed, or the screen is recalibrated before the result counts. The selected arm must also pass the variety guard of section 4 on the 100-chart panel.
- **Fail.** Selected collapsed in ≥ 4 songs, or discordant pairs ≤ 2, or any pair against.

## 6. Recommended first experiment

**Whole-song rerank of N = 32 plain samples on the frozen 56M, judged against plain sampling and against tonight's d0.** One Mac night (about 4 h) plus about 1 hour of the human's time.

1. Panel: the long-chart panel of the bake-off (25 groups per band, X0 charts excluded), 100 charts, BOS, 32 seeds each with the incremental sampler: 3.3 h. Record per sample the generator's log p_θ(x)/K.
2. Scores computed on every sample (seconds in all): (b) the block HSMM with chart type fixed from the first 128 rows and the duration law per type; the plain HMM log-likelihood as a control that is expected to game (probe A); tonight's `phi` and `env` block scores summed over the song.
3. Selection per song and score: satisficing clip at the real [q05, q95] with a random pick among acceptable candidates, and, as a second arm, the plain argmax, to measure how much the clip costs or saves.
4. Automatic measures, held out from selection, on the selected set per score and on d0-phi, d0-env and seed-954 plain: stay rate, first-to-last-third correlation (nh, held), envelope-exit rate, head-lock runs, between-chart SD ratio, LN share per band, adjacency MI, mean log p_θ/K.
5. Human screen (section 5) on 8 of the 100 songs for the recommended score only.

**Pass/fail (fixed).**
- Pass: the human rule of section 5 passes, and on the 100-chart panel the selected set has stay rate ≤ 0.35 (sources 0.24; plain 0.53-0.57) and first-to-last-third correlation ≥ 0.70 for nh and held, with the variety guard (SD ratio ≥ 0.8) and the KL proxy (mean log p_θ/K inside the plain interquartile range) both holding.
- Fail of the ranker: the human rule fails while the proxy m̂ of section 1 is ≥ 0.1; then the score is wrong, and the next candidate is (b) with the segment CRF (d) on top.
- Fail of the assumption: the human rule fails, m̂ < 1/32 in 3 of 4 bands, and the best-of-32 by *any* of the held-out measures is still outside the source range in most songs; then training must change, and selection is kept only as a guard.
- Against d0: the rerank arm must beat d0-env on stay rate and first-to-last-third correlation by the chart-bootstrap 90 % CI, or d0 (cheaper and causal) stays and the next step is SMC with the same score rather than rerank.

## 7. Probe results (measured)

Scripts `~/ensomi/.sync/cp/scratch/r2-decode/fable/probe_hmm.py` (probe A) and `probe_blockhmm.py` (probe B); outputs on the mac in `artifacts/r2-decode-20261007/fable/` (A: `summary.json`, `scores.csv`, `x0_items.csv`, `x0_windows.csv`, `x0_profile.csv`, `models.npz`; B: `summary_block.json`, `scores_block.csv`, `x0_items_block.csv`), mirrored. Jobs: `20261007-190014-fable-decode-probe` (failed: X1 runs stop at row 1,024 and the source was not cut to the same length; fixed), `20261007-190149-fable-decode-probe2` (exit 0, 10 s), `20261007-190552-fable-decode-probeB` (exit 0, 5 s).

**Setup.** Probe A: row symbol = head pattern × lanes held {0, 1, 2+}, 48 symbols; first- and second-order Markov chains and a 6-state HMM with categorical emissions. Probe B: 6-state Gaussian HMM over standardised 16-row block statistics (nh, c3, held, jack, rep1, pattern entropy, lane max, bus4), plus the Viterbi regime path under the real model, whose run lengths are scored under real-fitted and own-fitted per-state duration histograms (a two-stage stand-in for an HSMM). Each model is fitted twice: on 640 fit_train charts (160 per band, first 1,536 rows; none in set F or X0) and on the 696 n48 + n64 free runs (48M and 64M, not 56M). Scores per sequence: mean per-row (per-block) log-likelihood under the real fit (`lp_*`) and the log-likelihood ratio real minus own (`llr_*`); probe B adds the duration log-likelihood (`dur_real`, `llr_dur`) and the mean and longest regime run. Scored: 116 set-F sources (full, and cut to 1,024 rows for the paired comparison), the X1 56M runs L0 (173), L16 (232), L64 (173), and the 12 X0 items; the C1 envelope rate and stay rate of `collapse.py` were computed alongside.

**Probe A: sequence likelihood over row symbols is a density proxy, not a collapse finder.**

| Score | AUC source vs L0 run, all / band 2 / 3 / 4 / 5 | Spearman on L0 runs with held / jack / nh / stay15 | X0 AUC vs human collapse (7 of 12) / vs generated / collapse within generated (6 of 8) |
|---|---|---|---|
| lp_mk1 (real bigram) | 0.61 / 0.86 / 0.67 / 0.47 / 0.46 | −0.89 / −0.72 / −0.47 / −0.35 | 0.83 / 0.75 / 0.50 |
| lp_hmm (real HMM) | 0.61 / 0.85 / 0.64 / 0.46 / 0.49 | −0.89 / −0.63 / −0.56 / −0.20 | 0.74 / 0.69 / 0.42 |
| llr_mk2 (contrastive trigram) | 0.68 / 0.63 / 0.63 / 0.78 / 0.78 | −0.34 / +0.27 / +0.44 / −0.17 | 0.49 / 0.66 / 0.50 |
| llr_hmm (contrastive HMM) | 0.52 / 0.28 / 0.61 / 0.62 / 0.62 | +0.08 / +0.40 / +0.69 / +0.13 | 0.49 / 0.22 / 0.67 |

- The real-fitted likelihood separates sources from runs only in band 2 (where runs are too busy and LN-heavy) and is at chance in bands 4-5 (where runs are too sparse, hence *more* likely). Its correlation with held share is −0.89 on runs and −0.84 on sources: it is an LN-rarity meter. Selected by it, charts would lose LN and jacks (section 4).
- The contrastive LLR cancels the rarity term, and what remains is the band offset of the fitted checkpoints: llr correlates +0.69 to +0.90 with nh. It is at chance against the human's calls (0.49) and ranks x0-03 (band 2, called collapsed, envelope-exit rate 0.89, the clearest case on the hand-built measures) as the most real of the 12.
- None of the six scores sees absorption: |Spearman| with the stay rate ≤ 0.35, and the LLR change from L0 to L64 (where a real prefix halves the band offset) is 0.00-0.02 nats per row; the real-fitted lp does move with the prefix in band 2 (+0.22 to +0.37 nats per row), as the envelope does.
- The X0 AUC of 0.83 for lp_mk1 is the same between-chart LN-level effect the X0 scoring found for the LN measures: it falls to 0.50 within generated items. The worst 64-row window by LLR does not coincide with the human's first mark in any marked item (`x0_profile.csv`).
- Fitted self-transition probabilities are 0.85-0.97 (real) and 0.91-0.98 (own): a sticky HMM assigns long stays high likelihood, so it cannot price the 2× stay rate.

**Probe B: block statistics do not change the picture; regime durations carry a within-generated signal but are confounded by chart type.**

| Score | AUC source vs L0 run, all / band 2 / 3 / 4 / 5 | Spearman on L0 runs with held / jack / nh / stay15 | X0 AUC vs human collapse / vs generated / collapse within generated |
|---|---|---|---|
| lp_bhmm (real block HMM) | 0.55 / 0.90 / 0.58 / 0.36 / 0.39 | −0.85 / −0.80 / −0.50 / −0.36 | 0.77 / 0.75 / 0.50 |
| llr_bhmm | 0.45 / 0.20 / 0.45 / 0.56 / 0.65 | +0.38 / +0.77 / +0.76 / +0.26 | 0.29 / 0.38 / 0.58 |
| llr_dur (duration LLR on the Viterbi path) | 0.58 / 0.55 / 0.47 / 0.61 / 0.72 | +0.18 / +0.13 / +0.08 / +0.21 | 0.06 / 0.25 / 0.17 |
| mean regime run (blocks; AUC of −run) | 0.61 / 0.73 / 0.53 / 0.56 / 0.58 | −0.27 / −0.36 / −0.17 / +0.12 | 0.31 / 0.63 / 0.17 |

- The block-statistic likelihood behaves like the row-symbol one (band 2 only; held −0.85).
- Regime runs under the real model are *shorter* in 56M runs than in sources: mean 2.7 blocks (L0) against 3.3 (sources cut to 1k rows) and 3.5 (full sources); longest run 10.1 against 11.0-12.1. In regime space the runs wander more, they do not stick more (inferred: F2 as drift, consistent with the LN variogram). The envelope's 2× stay rate is "too long outside the band's range", which is a different object from "too long in one of six real regimes".
- Within the 8 generated X0 items, longer regime runs go with the human's collapse call: AUC 0.83 for the mean run and 0.92 for the longest run (6 against 2 items: a hint, not a result). Across all 12 the direction inverts (0.31), because uniform real charts exist: x0-01, a real band-3 chart the human passed, sits in one regime for 35 blocks (held share 0.000), and it is the same chart that gives the C1 envelope its false positive. So a duration law must be conditional on the chart type (section 2, candidate (b)), or a uniform real chart is scored as collapsed.
- x0-08, the LN-heavy real chart the human called collapsed, is ranked most real by every contrastive score, as it was by the LN measures.

**Inferred from both probes.** A generative sequence model fitted on real charts and used as a likelihood is not the stateful score: its likelihood is dominated by symbol rarity (density, LN, jacks), it rewards sparse bland charts, and it is blind to durations. The contrastive version removes rarity and keeps mostly the fitted checkpoint's band offset. What survives as a design is: per-chart-type regime durations (b), rank-only, satisficing-clipped, with held-out evaluators and the paired human screen. The probes also confirm the cost claim: whole-song scoring of 822 sequences took 10 s, so any of the section-2 scores is free next to generation.

## Failed and unfinished

- Probe A's first run failed on the X1 length (fixed in the second run).
- Not done, for time: an explicit-duration HSMM fitted by EM (probe B's duration law is a Viterbi two-stage approximation); the chart-type mixture (b) itself; any use of the generator's own log p_θ (not stored in the X1 or X0 files); the 2-gram-context CRF (c); the segment CRF (d). No generation, per the budget.
