# Design round, Opus: constraints, alternative systems, bake-off (2026-10-07)

Report of a fresh Opus subagent for [r-phase0-design](r2-collapse-20261007.md#r-phase0-design), from the same brief as [r2-design-fable-20261007](r2-design-fable-20261007.md). It is copied verbatim between the markers. Main-thread check: `artifacts/r2-collapse-20261007/design-opus/speed.json` and `panel_census.json` (mirrored) match the sampling speeds (582 → 193 decisions/s, 7.9 s for 2,048 rows) and the fit_train K ≥ 1,500 band-2 count (22 groups). Proposals here are not decided.

----- BEGIN report -----

## Design round: Opus (design-opus)

**In short.** Two choices produce F1 and F2 together: phase N has no chart-level variable, and the natural query has no statistic that converges over the song. The 511-row receptive field is a secondary limit. For the first night I recommend three trained arms plus the baseline:
- **B1:** a plain fine-tune control.
- **B2 (S1):** a self-anchor with history dropout.
- **B3 (S2):** a held chart vector θ drawn from a skeleton-matched donor prior, plus the same anchor channels.

All three share one long-chart panel and one gap-closure rule. That is about 5.5-6 Mac-hours with serial training.

Two new measurements change the protocol:
- **Band 2 has almost no long charts.** fit_dev has only 4 band-2 charts with K ≥ 1,500, so "25 per band at K ≥ 1,500" cannot be built.
- **Sampling is cheaper than assumed but grows with the prefix.** A full-panel evaluation costs about 1.1 process-hours per system.

### 0. Probes run (measured; nothing trained, no tracked edits)

Scripts are in `~/ensomi/.sync/cp/scratch/r2-collapse/design-opus/` (`panel_census.py`, `speed.py`). Outputs on the mac are in `artifacts/r2-collapse-20261007/design-opus/` (`panel_census.json`, `speed.json`; both mirrored).

**Census** (job `20261007-165705-design-opus-census2`). Distinct song groups available at each length threshold:

| | Band 2 | Band 3 | Band 4 | Band 5 |
|---|---|---|---|---|
| fit_dev, K ≥ 1,500 | 4 | 30 | 93 | 67 |
| fit_dev, K ≥ 1,000 | 42 | 116 | 167 | 94 |
| fit_train, K ≥ 1,500 | 22 | 310 | 865 | 553 |

- In fit_train, charts with K ≤ 600 are 27.5 % of charts but hold only 11 % of decisions. Charts with K > 1,500 hold 42.5 % of decisions.
- So the trainer's free-run panel (`max_rows=600`) samples only the part of the corpus that holds the fewest decisions.

**Sampling speed** (job `20261007-165744-design-opus-speed`; 56M, 1 thread, band-4 chart with K = 2,595):
- 582 decisions/s over rows 0-256, falling to 193 decisions/s over rows 1,024-2,048.
- The cost per step grows with the prefix: about 1.4 ms plus 2.5 µs × k.
- A 2,048-row run takes 7.9 s.

### 1. Constraint table (corrected and extended)

Evidence codes: M = measured, C = read from the code, I = inferred.

| # | Choice | What it rules out or makes hard | Defect | Evidence |
|---|---|---|---|---|
| 1 | Causal TCN over 511 rows, `memory: none` | **Hard limit:** nothing older than 511 rows reaches the logits, except exact lane clocks. **Correction to the draft:** this is not the window that drives drift. The window that actually sets the chart's level is about 64-128 rows: on its own output the model puts β_recent 0.64-0.70 on the last 64 rows, and the LN-share ACF reaches zero at 320-384 rows in runs against 128 in sources. The 511 boundary limits how long a *correct* prefix survives (nh correlation 0.63 → 0.35 past 512). | F2 (how long a correct start survives); not F1 | M (Astra probe; Fable add. 2; Opus D5) |
| 2 | **No cumulative statistic in the natural query.** `cum_*` reaches only the FiLM frames. *(Missing from the draft.)* | No convergent anchor. The chart's level is re-estimated from window content. That is a valid proxy while the history holds a constant level, as real charts do; on the model's own output it is not. | F2 (V(8)/V(1) 1.4-1.9 against 0.84-0.99); persistence of regime-C excursions | M + C (β_cum 0.51 → 0.23 on own prefixes) |
| 3 | No chart-level variable in phase N: no θ or z, no band or star | The start regresses to the corpus mean given the skeleton, and between-chart variance is compressed. Two seeds agree with each other more than with their source (nh 0.40 against 0.17). | F1 (band 2 +0.14 to +0.31 nh, `bus4` 16-31 % against 1.3 %; bands 4-5 about −0.20); compression (SD ratios: nh 0.74-0.80, c3 0.69, c4 0.41-0.52) | M |
| 4 | Teacher-forced per-decision CE on real histories only | No gradient on own-history states. The quantity that decides rollout (the own-prefix gain) is nearly flat in NLL, worth R² +0.05, a few millinats. Checkpoints change regime at equal NLL. | F2, absorption, regime flips | M (48M and 64M within ±0.014 nats, drifting in opposite directions; D8) |
| 5 | Ancestral Gumbel-max at T = 1. Every draw is committed; no lookahead, selection or revision. *(Missing.)* | A tail draw becomes evidence for the next 64-128 rows, and nothing can reject an excursion. Temperature trades drift against composition. | Absorption: stay rate 0.53-0.57 against 0.24 at an equal entry rate; band-2 4-chord exits 7-11 % against 0 | M (D1, D2; Astra T = 0.8 gave 91 % single-head rows) |
| 6 | Fixed head skeleton; a decision only at head rows | Intensity can move only through chords, jacks and holds. A too-intense regime on a sparse (band-2) skeleton therefore appears as chord and LN pile-up ("full 4 lane"). Difficulty tilts do the same. | The *form* F1 takes in band 2; control defects | I from C, with M support (flagged band-2 windows have held share 0.27-0.35 against 0.077; C0 screen) |
| 7 | Skeleton seen locally only: 16 gaps, densities up to 32 beats, phase at 1/4/16 beats | No section or repeat structure. The model leans on the only structure it has: runs follow local gaps more than sources do (loop-gap r −0.51 against −0.23). **Corrections:** grid phrase features would not help, because changes cluster at 4-bar starts and not further at 8- or 16-bar ones. A whole-skeleton summary identifies little of θ (R² 0.03-0.20; X3 pending). | Mostly feeds F1 and F2; little evidence for F3 | M (D6; Fable E1) |
| 8 | No audio | Section ends and intent are invisible; nothing ends an excursion. | F2 absorption (excursions last 2-4× longer) | I. Real section ends fall just before 4-bar starts (0.65×). Out of R2 v2 scope. |
| 9 | **LN length is a chain of keep/release decisions plus a release pointer, with no chart-level length input.** *(Missing.)* | A per-chart length style must be held across many hazard decisions read from the window. | Regime C for length: per-chart median LN length SD about 0.6× real | M (`o-lnlen-hintfree`) + C |
| 10 | Joint row head: mirror-equivariant unaries plus rank-16 coupling | Exactly hand-symmetric at BOS, so handedness must emerge from the model's own draws. Chord-size interactions depend only on per-hand head counts (rank ≤ 3), so the head can represent quads and chord habits. | F3? Weak. Chord compression is explained by #3. | C + I. Untested; a teacher-forced NLL screen settles it. |
| 11 | Training windows, start rule, chart-uniform sampling | **Not a constraint.** The full prefix is encoded, 46 % of decisions fall at row ≥ 511, and calibration at rows 0-64 is fine. | None | M |
| 12 | Exposure and capacity (2.4M parameters, 64M exposures) | NLL is flat from 40M. | None; regimes flip | M |
| 13 | Selection on teacher-forced NLL plus guards from K ≤ 600 panels (16 charts; 64-row own-history continuations) | Cannot see F1 or F2, so 48M → 64M got worse unnoticed. The panel covers 11 % of decisions (census above). | Selection blind to collapse | M |
| 14 | **The condition path is interval-local FiLM (identity gate, rule L) on a frozen phase-N base.** *(Missing.)* | A chart-scope request has to fight a frozen natural policy that re-estimates θ from its own window at every step. Obedience should decay like F2, and strong tilts push into absorbing regimes. | Control defects ("quad walls", "full 4 lane") | C + M (screen) + I |
| 15 | Corpus length by band *(protocol constraint)* | Band 2 has 4 fit_dev and 22 fit_train long groups (K ≥ 1,500). Long-song holding in band 2 can barely be trained or tested at that threshold. | Protocol | M (census) |

### 2. Alternative systems

All costs assume 1,800 decisions/s for training at 4 threads (the brief's figure; 8M exposures ≈ 1.25 h) and the measured sampling speed above.

#### S1, self-anchor with history dropout (SA)

**Delta from R2.**
- About 16 new query channels: cumulative rates over the committed prefix [0, k).
  - Heads per row, c3 and c4 rates, jack rate, held share, LN-head share, mean log2 LN length.
  - Prefix pattern entropy (from 15 prefix-summed mask counts), rep1/rep2/rep4, and log1p(k).
  - Hand share goes in as an own-hand channel per hand view, which keeps mirror equivariance.
- The channels feed a zero-initialised, bias-free reader on `exact[0]`, the same pattern as `ln_level_reader`.
- **History dropout:** on 25 % of windows the TCN input is truncated at j − L, with L ~ U{64..128}, behind the existing `TRUNCATED` boundary. On those windows the channels are the only source of the chart's level.
- Warm start from 56M with the `ce_v2_n_lnlevel2_ft` schedule (lr 1e-4 → 3e-5).

**What it lifts.** Rows #2 and #1 (it supplies a convergent statistic). It should turn the windowed urn into a converging one and fix F2: drift, persistence of excursions, the regime-C ACF. It cannot fix F1, because it locks in whatever the start drew, and it does not restore between-chart variety.

**Cost.** 1.25 Mac-hours (inferred: truncated windows are cheaper to encode, so the arm gets more exposures per hour) plus about 1.1 process-hours of evaluation.

**Code it touches.**
- `features.derive`: new prefix sums.
- `features.query_features`: an `anchor` flag.
- `model.py`: `R2Config.anchor` and `anchor_reader`.
- `train_ce.py` / `window_hands`: a `history_dropout` key and a `history_from` truncation.
- `sampling.py`: none, since the channels come from the chart.
- Tests: identity at initialisation.

**Main risk and early sign.**
- Risk: the identifiability failure, where CE leaves the reader near zero.
- Early sign: at 2M exposures (about 20 min), the reader's weight norm is still near zero, *or* the own-prefix β_cum stays below 0.35 on a 32-chart mini-panel (Fable's `model_cum2.py`).

**Controls and scenarios.** It does not take requests. A possible later route (I): pseudo-counts that seed the cumulative channels at a requested level, a "virtual prefix". It suits real-time play (no extra cost).

**Variant S1-M, learned memory.** `memory: landmarks` with the same history dropout. The landmark parameters already exist in the 56M checkpoint, with a zero-initialised `lm_out`. It is the learned version of the same anchor, but v1's landmark readout went unused, and softmax attention may rebuild a recent-biased proxy. Run it only if S1 passes and regime C stays open.

#### S2, held chart vector θ with a donor prior plus anchor channels (Θ, hierarchical)

**Delta from R2.**
- **θ:** 10 standardised whole-chart coordinates: nh, c3, c4, jack, held share, LN-head share, median log2 LN length, pattern entropy, loop rate, own-hand share. They come with a known bit, through a zero-initialised reader.
- θ is noised in training with N(0, half-chart disagreement SD) and dropped to unknown on 30 % of windows.
- S1's anchor channels and history dropout are included, so the model sees θ − realised and can learn a restoring force.
- **Prior at generation:** a nonparametric donor prior. Take the θ of a random one of the 32 nearest fit_train charts in skeleton-feature space (`fitskel.py` features). This generalises `ln-level-prior-v1.json`. Modes: prior, unknown, oracle (oracle is a diagnostic only).

**What it lifts.** Rows #3, #2, #9 and #14 (θ becomes the slot conditions use).
- **F1:** the start becomes a draw from p(θ | S) instead of the corpus mean.
- **Variety:** between-chart variance is restored by construction if θ is obeyed.
- **F2:** a held input plus feedback on the gap from θ. CE will use θ, because the whole-song level predicts better than the window (R² 0.65 against 0.53).

**Cost.** Building the θ table and the donor index takes about 10 min on 2 processes. Training is 1.25 h; evaluation is about 1.1 process-hours, plus 0.3 for the unknown and oracle modes on 1 seed.

**Code it touches.**
- New `theta.py`: θ from the cache arrays, `theta-v1.parquet`, the donor kNN.
- `features.query_features`: θ channels; `model.py`: `theta_reader`.
- `data.Corpus`: θ lookup, noise and dropout; `train_ce.py`: config keys.
- `sampling.continue_chart`: `theta=` taking prior, oracle, unknown or a vector; `generate.py` records the θ used.
- This is the `ln_level`/`ln_length` pattern (DEVIATIONS 11) widened from 2 to 10 coordinates.

**Main risk and early sign.**
1. **Obedience on own history fails:** θ is read at the start, but the window still wins later. Early sign: the slope of realised run means on requested θ at rows 1k+ is below 0.6 on the mini-panel at 4M.
2. **Budget leak:** θ plus the cumulative channels make the end of the song too predictable, which shows as compensation dumps late in the song. Guard: envelope exits in the last 10 % of rows at most 1.5× the source's.
3. **Donor θ inconsistent with the skeleton.** Watch the exits on prior draws against oracle draws.

**Controls and scenarios.** Every planned control is a θ coordinate:
- LN share and length (the stopped fine-tune was S2 with 2 coordinates and no anchor);
- difficulty (add Difficulty_ν or `ras-v1` as a coordinate);
- the five style concepts, once their labels exist.

A request fixes some coordinates and the rest are drawn from donors near the request. For the scenarios:
- **Ambient play:** draw θ once at song start.
- **Practice:** set the target coordinate.
- **Section regeneration:** estimate θ from the surrounding chart (the right-context fit is still a separate problem).

#### S3, plan-then-realise (P)

**Delta from R2.**
- S2, plus per-section plan channels: the current and the next 4-bar section's targets (nh, held, c3, jack, pattern entropy) and progress within the section.
- **Training:** the source's per-section statistics, noised and dropped.
- **Generation:** a tiny section-level planner. It is an AR(1) toward θ with skeleton-section regressors, fitted by ridge on fit_train in minutes. A donor-trajectory option is possible as a variant.

**What it lifts.** Rows #7 and #8: the plan ends excursions at planned times, and within-chart variation becomes planned rather than diffusion.

**Cost.** 1.25-1.85 h of training, the planner fit about 10 min, evaluation as S2.

**Code it touches.** As S2, plus a section index from `grid_bars`, plan channels, and a new `plan.py`.

**Main risk and early sign.**
- **Leak:** section statistics nearly determine the rows inside the section, so CE leans on the plan and generation becomes plan-forced. Early sign: teacher-forced NLL with an oracle plan drops by more than 0.1 nats while free-run pattern entropy and 4-gram measures leave the envelope.
- **Mistimed plans:** planner section changes land where the music does not change.

**Controls and scenarios.** Plan coordinates per section serve practice-targeted sections, mapper edits of a section's plan, and real-time play (the planner runs one section ahead, causally).

#### S4, on-policy moment term on top of S2 or S1 (R)

**Delta from R2.**
- On 1 of 4 windows: take a real prefix up to j, sample 256 rows with the current model (no gradient), and re-score them teacher-forced.
- Loss: Σ_m (E_θ[Φ_m] − Φ_m^src[j, j+256))² / σ_m², with expected counts as in `proxy.expected_proxies`.
- It reuses the stage-2 `mu_star` loop in `train_ce` (sample from the real prefix, re-score, penalise).

**What it lifts.** Row #4. It adds a gradient on own-history states, which teaches holding and reversion (DAgger-like, with moment targets). Alone it is limited by the information path: once the real prefix leaves the window it pushes toward the conditional mean, so put it on S2.

**Cost.** Sampling adds about 0.5-0.8 s per on-policy window, so about 1.5-2× wall time per exposure. At equal hours it reaches about 4-5M exposures.

**Code it touches.** `train_ce.py` (generalise `mu_star` to natural Φ targets), `proxy.py`.

**Main risk and early sign.**
- Risk: moment matching flattens real section variation, or meets targets cheaply (nh matched by stacking walls).
- Early sign: the within-chart SD ratio falls below 0.8, or c4 and 4-gram exits rise while the matched moments improve.

**Controls.** With requested targets, the same term is the obedience objective for conditions.

#### S5, decode-time block selection (D; training-free)

**Delta from R2.**
- Sample N = 4 candidate 64-row blocks.
- Keep the one with the highest log-density under a corpus model p(Φ_block | running chart Φ, skeleton block), the ridge of Fable's `cum_anchor` regression fitted on fit_train.

**What it lifts.** Row #5. It imposes the data's restoring force (β_cum ≈ 0.5) at decode time.

**Cost.** No training. The full panel costs about 4× sampling (≈ 4.4 process-hours), so use a 1-seed panel (≈ 1.5).

**Code it touches.** A wrapper around `continue_chart(stop=)`; `head_mask_bias` is not needed.

**Main risk and early sign.**
- Risk: it is a rule residual in disguise (R1's path) and cannot fix what Φ does not measure.
- Early sign: the Φ gap closes while non-Φ measures (4-grams, c4, jack runs) get worse.

**Scenarios.** It fits real-time play if publication runs 64 rows ahead.

**It is an upper bound on what holding buys without training**, and a fallback if S1 and S2 both fail.

**Row-head screen (H; not a bake-off arm).** Add a zero-initialised, mirror-symmetrised full 625-way residual on the concatenated hand vectors (about 230k parameters) and train 2M exposures (about 20 min). Keep it only if fit_dev NLL beats the plain fine-tune by at least 3 millinats *and* X2 finds F3 local.

### 3. Bake-off protocol

**Panel.** fit_dev, one chart per song group, 25 groups per band. Band 2 uses K ≥ 1,000 (42 groups available); bands 3-5 use K ≥ 1,500 (30, 93 and 67 groups). That is about 228k rows per seed.
- BOS runs at seeds 954-956, capped at 2,560 rows.
- One prefix-continuation seed from the first third of each song, to isolate F2.
- Cost: about 1.1 process-hours per system, about 17 min wall at 4 processes. This is inferred from one band-4 chart; LN-heavy charts may be slower.
- The panel is fixed before training.

**Measures.** These hold until X0 validates a subset (AUC ≥ 0.75); then only that subset counts, fixed before the night. Per band:
- m1, degenerate-window rate;
- m2, stay rate;
- m3, 1 − first-to-last-third correlation, mean over nh, c3, held and pent;
- m4, V(8)/V(1) for held and nh;
- m5, |1 − between-chart SD ratio|, mean over nh, c3, c4, held, pent and LN length;
- m6, |paired band offset| for nh and held.

**Single decision rule.**
- **Gap closure:** for each measure and band, g = (m_sys − m_src) / (m_base − m_src). G is the mean over measures and the 4 bands, with equal weights. G = 0 means source-like, G = 1 means baseline.
- **Guards, which must all pass:**
  - legal export;
  - fit_dev NLL with θ unknown no worse than baseline + 0.02 nats per decision;
  - single-head-row share and 4-gram ratio inside the source envelope;
  - end-of-song (last 10 %) exits at most 1.5× the source's;
  - no head-lock run of 30 rows or more.
- **Winner:** the guard-passing system with the lowest G, if its G is at least 0.15 below the fine-tune control's, with a chart-bootstrap 90 % CI on the difference that excludes zero.
- **Advance** if G ≤ 0.5. The human then runs a blind screen on 8 long charts (winner, baseline and source). Confirmed if the winner is preferred, or not called collapsed, in at least 6 of 8.

**Fairness.**
- Equal training wall-clock (1.25 h at 4 threads). Exposures are recorded, not equalised.
- Same warm start (56M), schedule, draw seed and panel.
- The final checkpoint for every arm, so there is no per-arm selection on the panel.
- S2 is judged in prior mode. Oracle mode is a diagnostic only.

**Controls.**
- B0, the 56M baseline.
- B1, the fine-tune-only control: 56M plus the same schedule, no new inputs.
- Free ablations: S2 trains with θ dropout, so its θ-unknown mode is a jointly trained self-anchor.

### 4. First bake-off: B0, B1, B2 = S1, B3 = S2

**Why these.** They split information-path fixes into the anchor alone against a drawn θ plus the anchor, with a clean control for fine-tuning itself. Both are the `ln_level` code pattern widened. Plan-then-realise and the on-policy term are better placed on top of the winner. Decode selection is a fallback.

**Night schedule.**

| Step | Time |
|---|---|
| θ table and donor index | ~10 min |
| Training B1, B2, B3 | 3 × 1.25 h = 3.75 h serial (about 2.2 h if two run in parallel; parallel speed unmeasured) |
| Mid-run mini-panels at 4M (32 charts × 1 seed: β_cum, obedience slope, reader norms) | ~0.3 h |
| Full panels for 4 systems plus B3's unknown and oracle modes (each starts as its arm finishes) | ~5.5 process-hours, ~1.5 h wall at 4 processes |
| **Total** | **~5.5-6 h serial**, inside the 8-hour budget |

**Expected outcomes and what each means (I):**
- **B1 ≈ B0** (G 0.8-1.1). This is my strong prior. If B1 differs from B0 by more than 0.2 in G, fine-tuning alone moves the regime (as 24M and 40M did). Then any margin needs a second training seed, and the next night replicates.
- **B3 < B2 < B1** (B3 G about 0.4-0.6, mostly from m5, m6 and m1). The chart variable is the lever. Build controls as θ coordinates. Night 2: S3 against S2, and S4 on S2.
- **B2 ≈ B3, both well below B1.** Holding is the main fix and the start bias is smaller than X1's band-2 pilot suggested. Keep the anchor and add θ only for controls.
- **B3 fixes m5 and m6 but not m2-m4; B2 is null.** CE reads chart inputs at the start but the window wins on own output. Separate "reader never learned" (norms near zero) from "learned but overridden" (own-prefix β_cum still ≈ 0.25). Night 2 is then the objective: S4 on S2, with S5 as the training-free bound.
- **Nothing beats B1.** CE will not make R2 hold any chart-level input. Move to rollout objectives and decode-time selection as the primary route.

Phase-0 results due at 17:45 adjust this:
- X3's R² per coordinate decides how informative the donor prior is. Above 0.6, θ̂(S) plus a residual draw would suffice.
- X1 by band sets how much of the gap B2 can close: drift-dominant bands should improve with B2, start-dominant bands only with B3.
- X2 decides whether the row-head screen enters night 2.

----- END report -----

## Addendum: network families per component (Opus, about 17:00-17:12 UTC, at the human's request)

Main-thread check: `famcost.json` and `rowrate.json` (mirrored) match the per-step costs, training costs and rows-per-second figures quoted.

----- BEGIN addendum -----

**In short.** No history-encoder family removes F2 on its own. Under cross-entropy on real histories the window proxy is as good as a convergent statistic, whatever the encoder. A persistent-state encoder in R1 and landmark memory in R2 v1 both went unused or failed in free running.

What decides holding is the recipe (history dropout, own-history training) and the inputs (θ), not the family. Per-step latency does not separate the families at R2's size: the real-time budget is 10-45× larger than the per-step cost.

So no family swap replaces B2 or B3 tonight:
- **Optional now:** one warm-startable family arm, a two-rate encoder (B4).
- **Round two:** a residual latent z, a block reranker and a from-scratch encoder swap, each with the fair protocol in §5.

Evidence codes: M = measured, C = read from the code, I = inferred.

### 0. New measurements (M; 1 CPU thread on the mac, read-only)

Scripts are in `~/ensomi/.sync/cp/scratch/r2-collapse/design-opus/` (`famcost.py`, `rowrate.py`). Outputs are in `artifacts/r2-collapse-20261007/design-opus/` (`famcost.json`, `rowrate.json`; mirrored). Jobs: `20261007-170624-design-opus-famcost2`, `20261007-170821-design-opus-rowrate`.

**The k-linear part of the per-step cost is an implementation artifact.**
- `sampling.continue_chart` calls `base.with_decisions(actions[:k])` and `chart.derived()` at every step. That costs 0.23 ms at k = 128 and 5.0 ms at k = 2,000, which is the whole 2.5 µs × k term.
- An incremental state makes each step about 1.5 ms flat.
- The fixed parts per step:

| Part | ms per step |
|---|---|
| TCN cache append | 0.93 |
| Query features | 0.42-0.47 |
| Hands plus joint head | 0.11 |

**Per-step cost of the alternative encoders** (H = 128, two hand sequences):

| Encoder | ms per step |
|---|---|
| GRU, 2 layers | 0.05 |
| Linear RNN, 4 layers | 0.20 |
| Transformer, 4 layers, KV cache at L = 512 | 0.38 |
| Transformer, 4 layers, KV cache at L = 2,048 | 0.86 |

Every alternative is at or below the TCN's 0.93 ms.

**Training cost, forward plus backward over 2,304 tokens** (2,048 prefix rows plus a 256 window):

| Encoder | ms |
|---|---|
| TCN | 294 |
| GRU, 1 layer | 201 |
| Full causal transformer, 4 layers | 1,611 (5.5× the TCN) |
| Transformer, 512 context over 768 tokens | 206 |

**Real-time budget.** Rows per second of song in fit_train, median and p95 by band:

| Band | Median | p95 |
|---|---|---|
| 2 | 4.7 | 6.4 |
| 3 | 6.2 | 8.6 |
| 4 | 7.8 | 11.2 |
| 5 | 9.8 | 14.5 |

- So each row has about 69 ms at p95 band 5, against about 1.5-8.7 ms of compute. Latency binds only for paradigms that multiply passes.
- A 64-row block is 6.6 s (band 5) to 13.5 s (band 2) of music.

**Chart-level sample size.** 11,368 fit_train charts in 4,167 song groups. Anything learned per chart (θ prior, z, codebook, planner) is fitted on about 4k independent songs, not 12.7M decisions.

**Recipe lever that holds for every family (I, from the 294 ms figure).** The current window draw re-encodes the full prefix to score 256 rows: about 5 encoded positions per scored one at K = 2,000.
- Scoring whole charts, or long segments, in one pass spreads the encoder cost over every position. Only the features, head and pointer still scale with the number scored.
- That probably gives 2-3× more exposures per Mac-hour.
- It is what makes from-scratch family comparisons affordable. It changes batch composition, not the per-decision weighting.

### 1. History encoder

The table uses the constraint-table row numbers from my main report.

| Family | Advantages here | Disadvantages here | Rows lifted / added |
|---|---|---|---|
| **Dilated causal TCN (current)** | Parallel over positions; exact, testable crop invariance; strong local bias suited to about 4k songs; cheap cached step; warm start at 56M | Hard 511-row horizon with no state beyond it; activations grow with the prefix (needs checkpointing); the costliest step part (0.93 ms) | Keeps #1 |
| **Causal transformer, full attention plus KV cache** | Reaches any committed row, so it lifts #1. It can *retrieve* the chart's own earlier patterns (motif and vocabulary reuse), which a cumulative statistic cannot: a possible lever on regime C. | Training 5.5× the TCN at 2,304 tokens (M) and O(T²) up to 3k rows. Weak inductive bias for about 4k songs. Attention under CE learns the same recency the data rewards. Training from scratch is about 10 Mac-hours at 64M-equivalent. | Lifts #1. Adds training cost and data hunger. |
| **Windowed transformer (512)** | Cheap (206 ms), another local model, maybe better F3 | Same horizon as the TCN; lifts nothing structural | None |
| **RNN / GRU with persistent state** | O(1) step (0.05 ms), unbounded memory in principle, ideal for real time, natural with whole-chart streaming (each token encoded once) | Sequential in time, so cost grows with depth and deep stacks are slow on CPU. TBPTT truncates credit. Gates learn the short timescale CE rewards. **R1 had a GRU over all committed rows plus landmarks and still collapsed** (`s-likelihood-vs-rollout`). | Lifts #1 formally; #2 only if the slow state is held |
| **Linear RNN / SSM (S4, LRU; Mamba-type)** | O(1) step (0.20 ms). Parallel training by scan or convolution for time-invariant S4/LRU. Eigenvalues initialised near 1 (time constant about 1,000 rows) build in slow channels, a learned near-anchor. Mamba's input-dependent gating could learn section resets. | No fused selective-scan kernel on CPU or MPS (I), so Mamba-type training is slow or risky here. Fixed-decay channels are exponential averages, not a convergent cumulative mean. Their readout faces the same CE indifference. From scratch. | Lifts #1, partly #2. Adds implementation risk. |
| **Hierarchical / two-rate** (fast TCN plus a slow GRU or attention over 64-row block summaries, e.g. the existing stride-64 landmark slots) | Matches the problem's chart → section → row structure. The slow path is cheap (30-50 blocks per song) and updates once per block, so the per-step cost is unchanged. **Warm-startable from 56M** (the TCN is kept; the slow path is zero-initialised). It is the natural home for θ, z or a plan. | Same CE indifference: needs history dropout on the fast path. Block boundaries are arbitrary unless 4-bar aligned (regime changes sit at 4-bar starts, Fable E1). | Lifts #1, #2 (pooled slow state), partly #7 |

**Does any family remove F2 by itself? No.**

Three reasons:
1. **It is a property of the data and the objective.** On real histories the cumulative and recent statistics carry equal weight (β 0.57 / 0.43) and the cumulative one adds only R² +0.05. So *any* encoder that minimises teacher-forced CE can be optimal using the window proxy (M, Fable addendum 2).
2. **Persistent memories already went unused.** R1's all-rows GRU did not prevent free-running collapse. R2 v1's landmark readout was unused, and its lesion improved NLL (M, from the notes).
3. **A family only changes what is possible and what it costs.** It changes whether a convergent anchor is *representable* (the TCN cannot reach past 511 rows) and the cost. Pressure to use it comes from history dropout, own-history training or a held θ.

One caveat (I): attention and hierarchical encoders could help regime C (copying the chart's own vocabulary) in a way no statistic can. That is the only family-specific gain I would test, and only after B3.

### 2. Chart-level and plan representation

| Family | Advantages | Disadvantages | Posterior collapse | Control surface | Labels |
|---|---|---|---|---|---|
| **Explicit θ (B3)** | Supplied, not inferred, so it is used whenever it predicts (whole-song level R² 0.65 against 0.53 for the window). Nonparametric donor prior; trivial to fit on 4k songs. | Covers only the chosen coordinates (not vocabulary, motif or "feel"); can be inconsistent with the skeleton; budget leak alongside the anchor channels | None | Direct: LN share and length, difficulty and style are coordinates | Computable statistics; style concepts need the Lens labels |
| **Continuous z (VAE / CVAE with a skeleton prior p(z\|S))** | Captures unnamed dimensions (regime C); the skeleton prior is built in; sampling gives variety | **High collapse risk here (I).** The decoder re-infers θ from about 64-128 rows, so under teacher forcing z is worth about the first 100 rows plus drift. KL then removes exactly the "hold the level" dimensions we need. Needs free bits (about 4-8 nats per chart), decoder history dropout, and auxiliary θ prediction. 4k songs limits z to about 16 dimensions or fewer. | High | Indirect: needs a post-hoc map, or semi-supervised dimensions tied to θ | None, but loses meaning |
| **Discrete codebook (VQ, style tokens)** | Mixture of styles; categorical prior given the skeleton; requestable as "style k"; avoids KL collapse | Codebook usage collapse at about 4k songs (≤ 32 codes is realistic); loses continuous controls (LN share) unless combined with θ | Usage collapse instead of KL collapse | Only if codes are supervised against the Lens concepts | Optional; supervised codes are better |
| **Learned plan sequence (per-section z_s, conductor-style)** | Models within-chart structure; section control and regeneration; the planner has about 440k section transitions (I), enough for a small model | Two levels of collapse. Leak if the plan is computed from the section being generated. Obedience is needed at two scales. Most code. | Two levels | Section coordinates (practice targets, mapper edits) | None for learned plans; an explicit section-statistics plan (S3) needs none either |

**Order.** Explicit θ first (B3). Then θ plus a small residual z with free bits, only if regime C remains after B3, measured by between-chart SD ratios of pattern entropy, 4-gram vocabulary and c4. Explicit section plans (S3) come before learned plans.

### 3. Row head

| Family | Advantages | Disadvantages | Rows |
|---|---|---|---|
| **Joint factorised (current): hand unaries plus rank-16 coupling** | Exact normalisation and masking over 625. Mirror equivariance doubles the effective data. 0.11 ms per step together with the hands. Chord-size interactions (rank ≤ 3 in per-hand head counts) are representable (I). | Pairwise between hands with rank-16 coupling; hand-symmetric at BOS | #10 (weak) |
| **Autoregressive over lanes** | Any joint distribution with small 5-way heads; natural per-lane masks | Order dependence breaks equivariance unless marginalised over two orders (2× cost, as the pointer already does). Four sequential steps per row. No evidence it fixes a defect. | None |
| **Full 625-way softmax** | Full joint expressivity, about 160k parameters | Loses sharing and equivariance unless symmetrised. Rare joint actions (4-chords mixed with LN codes) are poorly estimated. Expected gain small (I). | Lifts #10 |
| **Energy / contrastive reranking (rows or blocks)** | Can encode non-local qualities (playability, consistency with chart identity) that factorised AR cannot. Trained on real against own blocks, it learns the collapse signal directly. It fills the empty `selection` node. | Needs on-policy negatives (sampling cost). Energy hacking. N× generation cost. Close to an R1-style rule residual if built on hand features. | Lifts #5, partly #4 (trained on own samples) |

The row head is not where F1 or F2 live. The only head question worth Mac time is the zero-initialised 625-way residual screen (about 20 min, teacher-forced NLL), and only if X2 finds F3 local.

### 4. Generation paradigm

| Paradigm | Global consistency | Causal / real-time | Training cost | Rows |
|---|---|---|---|---|
| **Row-by-row autoregressive (current)** | Only through the inputs (θ, anchor) | Ideal: no lookahead, 1.5-8.7 ms per row against a ≥ 69 ms budget (M) | Warm start | Keeps #4, #5 |
| **Block-autoregressive (64 rows given the past)** | Allows N-sample selection or reranking per block, and a per-block latent or plan token | Needs a lookahead of about 6.6-13.5 s (64 rows) and N× compute (fits: 4 × 64 × ~5 ms ≈ 1.3 s per 6.6 s block) | Within-block AR reuses R2 (warm); the critic is small | Lifts #5; #7 with block latents |
| **Masked / iterative refinement or discrete diffusion over a window** | Bidirectional inside the window; revises uncommitted rows; **exactly the local-edit shape** (fixed past and future) | Fits with lookahead ≥ W (W = 64, 8 steps ≈ 8 window passes). Absorption across windows remains, so it still needs θ for F2. | **From scratch** (masked objective), about 8-10 Mac-hours. Legality is hard: holds and releases are a sequential state, so it needs constrained decoding or an LN representation as head plus length (which would also lift #9). | Lifts #5; #9 if re-represented; serves `local-edit` |
| **Whole-song non-autoregressive** | Best: repeats, choruses, implicit plan | **Violates the standing constraint** "chart decisions and publication stay causal" (RESEARCH.md, Vision) unless the human relaxes it. Also cannot respond to a player in the practice scenario. | From scratch, highest; legality repair over the whole song | Lifts #1, #5, #7; adds a constraint violation |

### 5. What enters the bake-off, at what cost, and how it is compared

**Tonight (unchanged core): B0, B1, B2, B3.**
- Optional **B4, two-rate encoder.** A zero-initialised slow path: a GRU over the 64-row TCN landmark states, plus the same history dropout as B2. No hand-built statistics.
- B4 against B2 tests a learned anchor against a hand-built one, with everything else equal: warm start from 56M, schedule, draw seed, panel, gap-closure rule G and guards.
- Cost: 1.25 h of training plus about 1.1 process-hours of evaluation. That makes the night about 7-7.5 h serial, so add B4 only if the first 10 minutes show that two trainings can run in parallel at ≥ 0.8× speed each.
- Early failure sign: the slow-path output weights are still near zero at 2M exposures.

**Round two, in priority order:**
1. **θ plus a residual z (CVAE, free bits 4-8 nats per chart, decoder history dropout) on the night-1 winner.** Run it only if regime C remains.
   - Cost: 1.5-2 h warm plus 1.1 h of evaluation.
   - Fair control: the winner continued for the same hours with no z.
2. **Block reranking with a learned critic** (real against own 64-row blocks, conditioned on the running chart statistics) on the winner.
   - Cost: critic about 0.5 h; evaluation about 4.4 process-hours on the full panel, or 1.5 on 1 seed.
   - Fair comparison: same training hours. Report generation compute separately, with a guard of ≤ 20 ms per row at k = 3,000 on 1 thread including the N×.
   - Compare against plain sampling and against training-free S5 (the hand-built Φ critic) to show what learning adds.
3. **Encoder family swap (LRU/S4 or a hierarchical encoder with block attention).** Only if B3 passes while regime C or F3 (X2 local) remains.
   - This cannot be compared fairly against warm-started arms, which inherit about 10 Mac-hours.
   - Protocol: train a TCN and the candidate **both from scratch**, with the winner's inputs and recipe and whole-chart scoring, at equal Mac-hours (about 3.5-4 h each, one night for the pair). Compare with G, fit_dev NLL and the latency guard.
   - I would not include a full-attention transformer: 5.5× the training cost for a benefit on F2 the evidence argues against.
4. **Masked-window refinement:** a separate track for the section-regeneration scenario, about 8-10 Mac-hours from scratch. It is not a remedy for collapse and is not a bake-off arm.
5. **625-way head residual:** a 20-minute screen, not an arm. It goes ahead only on a gain of ≥ 3 millinats against B1 and an X2 local verdict.

Two enabling changes, not arms. They are family-neutral and change cost, not the model's function:
- incremental sampler state, which removes the k-term (5 ms per step at k = 2,000);
- whole-chart scoring in training.

----- END addendum -----
