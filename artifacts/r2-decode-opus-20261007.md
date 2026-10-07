# Decode-time selection design, Opus (2026-10-07)

Report of the fresh Opus subagent for [r-decode-design](r2-bakeoff-night-20261007.md#r-decode-design), 18:56-19:11 UTC. The harness blocked its write to the scratch report, so the main thread saved its returned text here, unchanged in content. Probe outputs are on bings-mac in `artifacts/r2-decode-20261007/opus/` (mirrored). Scripts are in `~/ensomi/.sync/cp/scratch/r2-decode/opus/` (`hmmprobe.py`, `hazard.py`, `idscore.py`). The main thread checked the hazard table against `hazard.json`. The design is a proposal.

Tags: [M] measured, [I] inferred, [P] proposed.

## Headline

- **Real-vs-own density ratio is inverted.** A row-level autoregressive HMM ratio, real charts against own runs, ranks the human's marks the wrong way round [M]:
  - window AUC 0.24-0.31 and item AUC 0.31-0.43, at 8, 16 and 6×4 states;
  - in 5 of 7 called songs, the marked windows look *more* real than the unmarked ones.
  - Reading [I]: the corpus has jack runs and LN passages in charts built around them, so a score without chart context rewards out-of-context jacks and LN.
- **A chart-relative identity score points the right way:** block statistics against the chart's own running level give item AUC 0.83 and window AUC 0.60-0.66, robust to dropping x0-02 [M].
  - Exploratory: its coordinates were chosen after reading the human's type notes, and it misses the 0.75 bar.
- **The model proposes exits from degenerate stretches, but the chance falls with duration** [M]: 0.57-0.62 after one flagged window, 0.15-0.31 after 3-6, against 0.74-0.80 in sources.
  - So selection must act within the first one or two blocks of a stretch [I].
- **Recommended: `d1`** [P]. An SMC with 4 particles over 64-row blocks, weighted by that identity score plus a duration term.
  - Compared with `d0` and plain sampling.
  - A privileged source-oracle arm tests the assumption directly.

## 1. The assumption

**Formal statement [P].**
- S is the skeleton and x = (a_1..a_K) a chart. The sampler gives p_θ(x|S) = Π_k p_θ(a_k | a_<k, S).
- 𝒜(S) is the set of charts the human's whole-song blind screen passes.
- The assumption has two parts:
  - (A1) mass: m(S) = p_θ(𝒜(S)|S) is large enough for the search budget;
  - (A2) findability: a computable score ranks members of 𝒜 above non-members.

**Two masses matter.**
- Whole-song reranking of N samples succeeds with probability at most 1 − (1−m)^N.
- Block search needs the conditional block mass m_b to stay away from zero at every block.
- If acceptability is mostly a conjunction of local conditions, m(S) ≈ Π_b m_b falls exponentially with K while m_b stays flat [I].
- Difficulty divergence and identity drift are not local, so the score must also carry chart-level state.

**What the evidence says.**
- X0 [M, n = 8]: 2 of 8 plain whole songs passed (x0-05 and x0-10). They are the two shortest generated items (K 1,545 and 1,553). All six with K ≥ 1,626 were called collapsed. Real foils passed 3 of 4.
- Fitting m = m_b^(K/64) to those items gives m_b ≈ 0.96 per 64-row block [I, back-of-envelope]. That means m ≈ 0.35 at K = 1,550 and m ≈ 0.13 at K = 3,000.
- With a perfect ranker [I]: the best of 8 whole songs passes a 3,000-row song about 67 % of the time; best-of-4 per block almost always does if candidates are independent given the prefix.
- The exit hazard declines with duration (P2) [M]. Best-of-4 leaves a fresh stretch with probability ≈ 0.97, and a stretch three or more windows old with ≈ 0.6-0.75 [I].

**Cheap tests [P].**

| Test | What it isolates | Cost | Reading |
|---|---|---|---|
| T1: plain-sample pass rate by song length | m(S) directly | none beyond §5 | Gives m and the N a whole-song rerank needs |
| T2: source-oracle selection. The same block search, but each block is weighted by closeness to the real chart's same rows | Upper bound on selection among the model's own proposals (A1 at block level) | same as `d0` | If even this cannot reach the source on stay rate and identity, the mass is not in the proposals |
| T3: top-ranked vs random sample of the same song | A2 given A1 | in §6 | If ranked ≈ random while T2 succeeds, the score is at fault |

**What would show it false [P].**
- T2 misses stay rate ≤ 0.40 at rows 256 and later, or nh/held identity correlation ≥ 0.70 at rows 1k+.
- Or T2 meets both but the human still calls its charts collapsed as often as plain ones.
- Then the lever is training (an on-policy objective or a chart latent), not decoding.

## 2. Candidate stateful scores

Each decomposes into per-block increments from forward messages.

**S1. Autoregressive HMM density ratio.** K regimes. The emission is log p(x_t | x_{t-1}, z_t) over 81 row symbols. Score = Σ_t [log p_R − log p_O].
- Sees: local organisation and LN at row scale. Its regimes last 17-65 rows. It has no chart identity.
- Cost: O(K²) per row.
- Probed: anti-aligned with the human [M].

**S2. Hierarchical HMM.** A chart cluster c (fixed per chart) times a regime z. The causal posterior over c carries identity.
- Cost: O(C·K²) per row.
- Probed at 6 clusters × 4 regimes: no better than S1 (window AUC 0.31) [M].

**S3. Kalman identity anchor, the stateful form of `phi`.** The chart level θ_b over (nh, held, c3, jack, rep1, hmax, lmax, pent) is a Gaussian random walk. Block statistics are y_b ~ N(θ_b, R). The prior is the band's start distribution. Score: log N(y_b | θ̂_{b|b−1}, P+R).
- Sees: the start (through the prior), held identity, difficulty divergence, and LN, jacks and repetition relative to the chart's own level (the x0-08 lesson).
- Fitted on real charts only.
- Cost: O(1) per row, O(d³) per block.
- A simplified version (sID) was probed: item AUC 0.83, window AUC 0.60-0.66 [M].

**S4. Explicit-duration (semi-Markov) regime score.** A regime r (clean, or flagged by type) and its duration d.
- Entry costs 0; the entry rate is already right. Staying costs −log[S_real(d+1|r)/S_real(d|r)], from the real duration law.
- Sees: absorption.
- Cost: O(R·D_max) per window.
- Probed with a fixed penalty [M]: it separates runs from sources (span means 0.07-0.14 vs 0.00-0.02) but not the human's marks (AUC 0.51-0.58).

**S5. Linear-chain CRF over windows.** Labels {ok, collapsed}, with chart-relative features. Fitted on the human's marks and pairwise rankings.
- [P] Use only as a held-out evaluator until about 40 or more human-judged songs exist.

**S6. Twisted SMC with a value look-ahead.** Twist ψ_b ≈ log E[exp(λ S_future) | x_≤b].
- For S4 there is a closed form from the measured hazard. Otherwise a critic regressed on rollouts.
- [P] Second step.

**Combination [P].**
- Target: the tilted distribution π(x) ∝ p_θ(x|S)·exp(λ Σ_b s_b(x)). SMC samples it, so log p_θ stays in the target.
- Best-of-N per block moves at most log N − (N−1)/N from p_θ per selection: 0.64 nats per block at N = 4.
- A row-level product of experts is possible only for S1.

**Why S1 fails [I].**
- A row-level ratio cannot tell whether a pattern fits this chart and this music. In x0-11, marked windows score 0.147 nats/row against 0.032 for unmarked ones.
- Six chart clusters did not fix it.
- Any real-against-own ratio risks the same inversion unless it is conditioned on the chart's established identity.

## 3. Search

For 100 long charts, mean K ≈ 2,400, at 1.5 ms per row, serial. Plain sampling takes about 6 min.

| Method | Ranks | Mac time | Real-time lag | Note |
|---|---|---|---|---|
| Whole-song rerank of N | exact whole sequences | 6N min (96 at N = 16) | whole song | needs m ≳ 0.1, about the estimated long-song mass [I] |
| `d0` greedy best-of-N per block | blocks, myopically | 24 min at N = 4 | one block | cannot recover from a trap |
| **SMC, P particles** | the tilted whole-sequence target | 6P min (24 at P = 4) | ~128 rows (15-30 s of music) | recommended; a slightly worse prefix can survive |
| Beam over blocks | deterministic | — | — | degeneracy cuts variety |

SMC commit rule: publish block b once block b+2 exists, or earlier once all particles share b.

## 4. Reward hacking

| Score | How it can be gamed | Guard |
|---|---|---|
| S1, S2 | rewards patterns rare in own output (out-of-context jacks and LN, measured on X0); copies the commonest real rows | do not select on them; at most a low-side tripwire |
| S3 | bland, constant charts; locks in a wrong start; copies the band mean | log-density with real within-chart variance, so too-constant costs; prior = the band's spread of start levels, not its mean; two-sided rate checks |
| S4 | avoids flagged regimes entirely (no LNs or jacks, lower density) | zero potential on entry; keep the flagged-entry rate near the sources' 0.059 |
| Any search | loss of variety across seeds; exploiting statistic definitions (one row breaking a lock run) | the safeguards below |

**Safeguards, fixed before the run [P].**
1. Keep log p_θ: SMC from the tilted target with λ = 1 (nats). Report the effective sample size and the KL per block against the 0.64-nat best-of-4 reference.
2. Held-out evaluators never used in selection: adjacency-MI gap; first-to-last-third identity correlation; two-sided LN share and LN length against the band's source IQR; two-sided chord and jack rates; p_θ NLL per row of the chosen chart against plain.
3. Diversity: two seeds per chart. Cross-seed distinct-4-gram and identity distance at least 0.8× plain's.
4. Rank only: the human never sees scores, and λ is not tuned on the human's screen.
5. Final judge: the human screen, with real foils.

## 5. Human blind screen [P]

- **Unit:** one whole song with one chart. Answers as in X0: collapsed yes/no, a type note, confidence; time ranges optional.
- **Pairing:** per song, plain, `d0` and `d1` are shown blind in random order and ranked (ties allowed). Same song and skeleton removes song, band and length effects.
- **Songs:** 6 fit_dev songs, X0 charts excluded, K 1,600-3,400, bands 2, 3, 4, 4, 5, 5, chosen by a fixed hash rule before generation; seed 954.
- **Foils:** the source charts of two of the six songs.
- **Size:** 20 charts, about 60 min.
- **Arm passes:** not collapsed in at least 4 of 6. If plain passes each song with probability 0.25, chance gives this with p ≈ 0.04; at 0.35, p ≈ 0.12.
- **A beats B:** A is ranked above B in at least 5 of 6 songs, or A is not collapsed in at least 2 more songs.
- **Session valid:** at least 1 of the 2 foils passes; otherwise read the session as rankings only.

## 6. Recommended first experiment: `d1` [P]

**Setup:** frozen 56M, BOS, T = 1. Panel: 48 charts of the long-chart panel (12 per band 2-5), seeds 954 and 955.

**Arms**, with Mac time for 48 charts × 2 seeds:
- **plain:** reuse B0's runs if they match. About 6 min.
- **`d0`:** as built, `env` score; reuse its outputs if they cover the panel. About 23 min.
- **`d1`:** about 23 min.
  - Search: SMC, P = 4 over 64-row blocks; multinomial resampling at every boundary on exp(s_b); a fixed-lag commit of 2 blocks; the final chart drawn by weight.
  - s_b = S3 increment + S4 increment.
- **oracle (T2):** the same SMC with s_b = −½Σz² of the block's statistics against the real chart's same rows. About 23 min.

**Score increments.**
- **S3:** −½Σ z_m², with z clipped to ±5.
  - z is taken against the chart's running block mean, with one pseudo-block at the band's source median.
  - The scale is the per-band root-mean within-chart block variance of fit_train sources.
  - Coordinates: nh, held, c3, jack, rep1, hmax, lmax, pent.
- **S4:** −1.4·(d−1)₊, where d is the current run of `env`-flagged blocks; 1.4 ≈ −log(1 − 0.75).

**Build:** extend the `block_select.py` loop to keep 4 particles, each with its own incremental sampler state, resampled on the weights. Total Mac time about 75 min.

**Automatic stage**, before the human stage. `d1` must:
- bring the stay rate to ≤ 0.40 (plain 0.53-0.57, sources 0.24);
- lower mean sID against plain;
- leave no held-out evaluator worse than plain beyond its chart-bootstrap 90 % CI;
- keep diversity ≥ 0.8× plain.

The oracle's stay rate and its identity correlation at rows 1k+ are recorded as T2.

**Human stage:** as in §5, 6 songs × (plain, `d0`, `d1`) + 2 foils, about 1 h.

**Fixed rule.**
- `d1` passes: not collapsed in at least 4 of 6, at least 2 more songs than plain, and ranked above `d0` in at least 4 of 6.
- Assumption supported: `d1` or `d0` is not collapsed in at least 4 of 6 and in at least 2 more than plain.
- Assumption false at this budget, so train: `d0` and `d1` are both at or below 2 of 6, AND the oracle misses stay ≤ 0.40 or identity correlation ≥ 0.70 at rows 1k+.
- Score problem, not mass: the oracle meets those targets but `d1` fails the human. Then show the oracle arm to the human and learn the score from pairwise rankings (S5) before any retraining.

## 7. Probe results

No generation was run. Each job scored existing runs only, on 2 threads, in under 1 minute.

**Jobs.**
- `20261007-190009-r2dec-opus-hmm` (8 and 16 states);
- `20261007-190319-r2dec-opus-hmmmix` (6 chart clusters × 4 regimes);
- `20261007-190217-r2dec-opus-hazard`;
- `20261007-190736-r2dec-opus-idscore3` (supersedes 190632 and 190658).

**P1. HMM fitted on real charts against one fitted on own runs.**
- Setup: R on 360 fit_train charts (90 per band, ≤ 3,000 rows each, 440k rows). O on 56M X1 L0 runs (rows < 1,024) of set F, 2 folds by chart, X0 excluded. EM was stopped at 25 iterations, unconverged. The score is the mean LLR per row.
- AUCs at 8 states / 16 states / 6×4 clusters:
  - X1 L0 below its source: 0.65 / 0.70 / 0.66;
  - X0 items against the human's calls: 0.37 / 0.43 / 0.31;
  - X0 items, generated vs source: 0.53 / 0.72 / 0.44;
  - X0 windows, half label: **0.26 / 0.24 / 0.31**;
  - X0 windows, within called items: 0.29 / 0.27 / 0.34;
  - X0 windows, any-overlap label: 0.36 / 0.36 / 0.37.
- Findings [M]:
  - In 5 of 7 called items, marked windows have the higher LLR (exceptions: x0-08, x0-12). Without x0-02 the AUC is 0.28-0.29.
  - Band dominates the item ranking: the band-5 LN items score most "real".
  - L0 minus source per row, by band (8 states), 90 % CI: band 2 +0.001 [−0.016, 0.020]; band 3 −0.035; band 4 −0.057; band 5 −0.065.
  - A 64-row prefix helps only in band 4: +0.017 [0.004, 0.029].
  - The gap is largest in rows 0-128, so the score sees the start draw.
  - R's states last 17-39 rows. Under R, runs dwell 0.76-0.96× as long as their sources, so row-scale regimes miss the window-scale absorption.
  - Seed SD within a chart (0.037-0.047) is about the source-run gap (0.038-0.055). The best of up to 6 runs reaches its source in 57-72 % of charts. Since the score is invalid, this says nothing about acceptability.

**P2. Exit hazard of degenerate stretches** (C1_any11 flags on 64-row windows, band envelope of set-F sources, X0 excluded; h(d) = P(next window clean | d flagged windows), with counts at risk in brackets).

| Set | Sequences | Flagged | h(1) | h(2) | h(3) | h(4) | h(5) | h(6) |
|---|---|---|---|---|---|---|---|---|
| Sources | 116 | 0.077 | 0.76 (106) | 0.74 (23) | 0.80 (5) | 1.0 (1) | – | – |
| n48 runs, full songs | 348 | 0.112 | 0.57 (312) | 0.43 (125) | 0.31 (65) | 0.46 (41) | 0.25 (20) | 0.15 (13) |
| n64 runs, full songs | 348 | 0.137 | 0.59 (352) | 0.42 (134) | 0.21 (70) | 0.26 (50) | 0.26 (34) | 0.18 (22) |
| 56M X1 L0, rows < 1,024 | 173 | 0.121 | 0.62 (144) | 0.57 (51) | 0.29 (21) | 0.50 (12) | 0.0 (6) | 0.33 (6) |

- The hazard declines with duration in runs and is flat in sources [M].
- Whole songs with no flagged window: 0.38-0.52 of sources and 0.13-0.55 of runs, so "no flags" is not an acceptability proxy [M].

**P3. The `d1` score parts** (higher is worse; sID = the simplified S3; flag = an `env` exit; sDUR = (d−1)₊; comb = sID + 1.4·sDUR).

| AUC | sID | flag | sDUR | comb |
|---|---|---|---|---|
| X0 windows, half label | 0.65 | 0.58 | 0.53 | 0.65 |
| X0 windows, half label, within called items | 0.60 | 0.57 | 0.52 | 0.60 |
| X0 windows, half label, without x0-02 | 0.65 | 0.71 | 0.58 | 0.64 |
| X0 windows, any label | 0.66 | 0.57 | 0.51 | 0.66 |
| X0 items, human calls | **0.83** | 0.79 | 0.57 | **0.83** |
| X0 items, generated vs source | 0.94 | 0.75 | 0.56 | 0.94 |
| X1, L0 worse than source (rows 64-1,023) | 0.56 | 0.55 | 0.56 | 0.56 |
| X1, L0 worse than L64 | 0.47 | 0.50 | 0.52 | 0.47 |

- sID's only item miss is x0-08, the real LN chart the human called collapsed [M].
- sDUR by span (rows 64-256 / 256-512 / 512-1k): L0 0.07 / 0.14 / 0.11, L64 0.02 / 0.05 / 0.03, sources 0.01 / 0.02 / 0.00 [M].
- Headroom under comb [M]: seed SD 0.51 against a mean gap of 0.07. The better of two L0 seeds scores at or below its source in 72 % of 57 charts, against 54 % for a single seed.
- sID is chart-relative, so it cannot see a wrong but self-consistent start (F1). That explains X1 AUCs near 0.5 [I].
- Limits: 12 human-judged items and 26 positive windows (15 from x0-02); sID's coordinates were chosen after reading the type notes; no measure reaches the 0.75 window bar.

**Failed path.** The first `idscore.py` run scaled by the band median within-chart SD, which is near zero for band-2 held and c3, so z² exploded. It was fixed with root-mean variance and z clipped to ±5. Item AUC stayed at 0.83; window AUC went from 0.70 to 0.65.

**Not done:**
- an EM-fitted explicit-duration HMM;
- a full Kalman fit of S3;
- per-band HMMs;
- P1 and P3 on the full-song n48 and n64 runs;
- a look-ahead (twisted) SMC;
- a converged EM fit.
