# R2 pattern collapse: synthesis and proposed core questions (2026-10-07)

Main-thread synthesis of three independent reports, written for [d-collapse-primary](r2-collapse-20261007.md#d-collapse-primary):
- [Fable](r2-collapse-fable-20261007.md): formalisation, with two addenda;
- [Opus](r2-collapse-opus-20261007.md): empirical diagnosis;
- [Astra](r2-collapse-astra-20261007.md): code and information-path audit, with probes.

Each report's load-bearing numbers were spot-checked against its output files; the checks are listed in each material. Everything below "Proposed" is for human discussion and is not decided. Nothing was trained.

<a id="s-collapse-agree"></a>
## What the three reports agree on (measured)

1. **The objective is satisfied while rollout fails.**
   - On real histories the model is calibrated: expected heads per row is within ±0.05 of realised in every band and position, and the LN response matches the data.
   - On its own histories the whole-song statistics fail.
   - Degenerate stretches start at the corpus rate (entry 0.059-0.069 against 0.059) but last about twice as long (stay 0.53-0.57 against 0.24) ([Opus D1](r2-collapse-opus-20261007.md)).
2. **More training does not help, and the regime flips between checkpoints while NLL stays flat.**
   - 24M falls into a sparse, looping regime. 40M collapses in band 2.
   - 48M → 64M is worse: degenerate windows go 0.11 → 0.14, and busy-lock runs of 30+ rows go 14 → 70 of 348 ([Opus D8](r2-collapse-opus-20261007.md)).
3. **There is a hard information boundary at 511 rows, and no chart-level variable on the natural path.**
   - Replacing history older than 511 rows doubled the cumulative head count but left the logits identical ([Astra](r2-collapse-astra-20261007.md)).
   - The cumulative prefix statistics (`cum_heads`, `cum_ln` and so on) are read only by the FiLM frames, never by `query_features` (main-thread check).
   - `memory: none`; there is no band input, chart latent or style summary.
4. **The model's long-range anchoring is a proxy that breaks on its own output.** On real prefixes it weights the cumulative LN share like the data (β_cum 0.51-0.53 against 0.57). On its own prefixes β_cum halves (0.23-0.25) and the last 64 rows take over (0.64-0.70) ([Fable addendum 2](r2-collapse-fable-20261007.md)).
   - The window share stands in for the chart's level only while the history holds a constant level, which real charts do and generated ones do not.
   - In real charts the causal cumulative statistic is worth only R² +0.05 over the recent window: a few millinats in CE, but the whole collapse in rollout.
5. **The failure covers all pattern statistics, not only LN.** First-to-last-third correlation is 0.49-0.64 in runs against 0.80-0.91 in sources, for held share, heads per row, chords, jacks and pattern entropy ([Opus §1](r2-collapse-opus-20261007.md)). Excursions outlast real ones in every component ([Fable addendum](r2-collapse-fable-20261007.md)).
6. **Collapse grows with song length, and our panels were blind to it.**
   - On long charts the end-of-song departure is 2× (48M) to 4.3× (64M) the sources'.
   - The trainer's free-run panel takes only charts with K ≤ 600 (`data.freerun_charts`, `max_rows=600`). Selection used 16 charts.
7. **Ruled out as causes or levers:**
   - the training window and start rule;
   - more exposure;
   - row temperature (T = 0.8 broke composition: 91% single-head rows; the test was inconclusive on collapse);
   - 8- and 16-bar phrase features (no concentration beyond 4-bar starts, which the model already sees);
   - R1-style fixed-lane runs (1-4 of 348 runs).

<a id="s-collapse-components"></a>
## Formalisation: three components of "pattern collapse"

Real charts are hierarchical: p*(D | S) = ∫ Π_k p*(a_k | s_k, θ) p*(θ | S) dθ, with a chart-level organisation θ drawn once.
- θ covers density, chord habit, LN level and length, lane vocabulary and jack regime.
- The skeleton identifies little of θ: R² 0.03-0.20 for chart means, 0.17 for LN share.
- R2 approximates the hierarchical model with q(a_k | last 511 rows, exact state, local skeleton). It re-infers θ at every decision from recent rows, in effect the last 64-128.

Three components follow:
- **F1, a wrong start draw (band-dependent).** From the start of a song, a run adopts a band-biased regime within about 64-192 rows, and the offset then stays flat. _Re-read 2026-10-08: measured against the band mean, which mixes rows with arrangement; against E[θ | rows] the start follows the rows ([c-band-reference](r2-representation-20261008.md#c-band-reference))._
  - Band 2 is too busy and too LN-heavy: "all four lanes busy" in 16-31% of windows against 1.3%. This is the natural-generation form of the human's "full 4 lane".
  - Bands 4-5 are too sparse and loop.
  - A 64-row real prefix removes about two thirds of the band-2 excess for the whole song ([Opus D3, D7](r2-collapse-opus-20261007.md)).
- **F2, no holding (drift and absorption).** Identity is not kept even after a correct start.
  - A real prefix's identity fades past about 512 rows: nh correlation 0.63 → 0.35 at 48M, while the source holds 0.78-0.92.
  - Variance about the early level grows with length.
  - Degenerate stretches absorb.
  - Mechanism, inferred and consistent with every measurement: a calibrated short-memory response compounds on the model's own output, with no convergent anchor ([Fable §1](r2-collapse-fable-20261007.md)).
- **F3, weaker local organisation (possibly separate).** Adjacent-row dependence is 0.18 bits below the sources, 90% CI [−0.24, −0.12]. The deficit is present from the start and does not worsen monotonically ([Astra](r2-collapse-astra-20261007.md)). Whether this is what the human perceives is unknown.

Today's LN-level wandering is F2 on one coordinate, plus F1's band bias in LN share. The stopped `ln_level` fine-tune addressed one coordinate of θ, and only through an external input.

**How it maps onto the recipe, architecture and information paths:**
- **Objective.** Per-decision CE on real histories prices anchoring at a few millinats, so it exerts no pressure toward stable rollouts.
- **Architecture.** A 511-row TCN with no cumulative statistics in the query, no chart latent and memory off.
- **Information.** θ is not identifiable from the skeleton. Section endings come from audio or mapper intent, not from the grid.
- **Selection.** Teacher-forced NLL and short-chart, 16-chart guard panels cannot see F1 or F2.

<a id="s-collapse-disagree"></a>
## Where the reports differ, and the main-thread reading

- **Which lever first?** Fable says a self-anchor (cumulative prefix statistics as inputs), Opus says an oracle chart latent z, Astra says no remedy before a horizon test. Main-thread reading:
  - A self-anchor targets F2 but would lock in whatever F1 drew. Fable's addendum 2 adds an identifiability risk: on real data the cumulative channel is redundant with the window, so CE may leave its reader at zero unless training breaks the redundancy, for example with history dropout.
  - An oracle z covers F1 and F2 together for the dimensions it carries, and CE will use it, because it predicts better than the window. Its risk is obedience in free runs, and generation needs a prior over z.
  - Astra's horizon test addresses F3 and costs no training.
- **Start or drift?** Both, by band: band 2 is mostly a start problem, while band 5 drifts back after about 512 rows ([Opus D7](r2-collapse-opus-20261007.md)).
- **Do the measures match what the human sees?** No agent can check this. Astra warns that rarity in the corpus does not certify playability. Every pass/fail below is a proxy until the human calibrates it.

<a id="p-collapse-questions"></a>
## Proposed core questions (for discussion; not decided)

- **Q0, measurement validity.** Which computable measure tracks what the human calls collapse? The candidates are the envelope-exit rate, the stay rate, first-to-last-third identity, the variogram on long charts, and adjacent dependence. This is a prerequisite for every pass/fail.
- **Q1, holding (F2).** Given a correct start, does the model hold chart identity over a long song when it can read a convergent anchor?
- **Q2, the start draw (F1).** Can the skeleton identify θ well enough, or must θ be drawn per chart from a prior?
- **Q3, local organisation (F3).** Does accumulated self-history worsen local organisation at a fixed song position, or is the deficit already local?

<a id="p-collapse-experiments"></a>
## Proposed experiments (for discussion; nothing launched)

Phase 0 needs no training, and its runs take about one hour of Mac time in all. X0 also needs the human's time.

| Id | Question | Manipulation / control | Metric | Pass / fail (fixed in advance) | Cost |
| --- | --- | --- | --- | --- | --- |
| X0 | Q0 | The human views or plays 8 full long-song generations (56M) and their sources, blind, and marks where collapse starts and what it is. | Agreement between the human's marks and each measure (window-level AUC) | A measure is usable if its AUC is ≥ 0.75 against the human's marks | ~20 min to render, plus the human's time |
| X1 | Q2 / F1 | 116 charts × 3 seeds at 56M, starting from a real prefix of 0, 16 or 64 rows (0 is the control) | Paired d_nh, d_held and `bus4` exits per band at rows 256-512 and 512-1k | "Start-dominant" if the 64-row prefix removes ≥ 50% of \|d_nh\| at both spans; "drift-dominant" if < 25% at 512-1k | ~25 min |
| X2 | Q3 / F3 | 8 long charts, 56M. Generate the 128, 512 or 1,024 rows before the same late endpoint, each from its real prefix; the 128-row arm is the control | Gap to the source in shuffle-corrected adjacent dependence over the same final 128 rows | Accumulation confirmed if the 1,024-row arm is worse by ≥ 0.05 bits with a 95% CI above 0 | ~15 min |
| X3 | Q2 | Data only: rich skeleton features (`fitskel.py`) → per-chart θ, on fit_train and fit_dev | R² per θ coordinate and band | R² < 0.3: θ must be drawn per chart. R² > 0.6: a skeleton summary input suffices | ~10 min |

Phase 1 is training, after phase 0 and discussion. It has one question, Q1, and is measured on continuations of real prefixes, which keeps F1 out of the test:

| Arm | Change | Purpose |
| --- | --- | --- |
| C (control) | 56M + 8M CE exposures, no new input | Separates "more fine-tuning" from the anchor |
| SA (self-anchor) | Cumulative prefix statistics of the committed chart (LN share, heads per row, jack rate, held share, lane histogram, log k) through a zero-initialised reader, plus history dropout (encoded history truncated to 64-128 rows on a fraction of windows) | Tests whether a convergent self-anchor stops F2 without any external input |
| Z (oracle latent) | The per-chart θ vector (whole-chart nh, c3, jack, held share, median LN length, pattern entropy, loop rate, hand balance) as an input, dropped to unknown 30% of the time | Tests whether CE uses a held chart-level input to keep identity; it later needs a prior over θ (Q2) |

- **Primary metric:** on long charts (K ≥ 1,500), continuing from a real prefix of the first third, the correlation of run identity with the source prefix's identity at rows 1k+, for nh and held share. The source holds 0.78-0.92; 48M without an anchor reaches 0.35.
- **Pass:** ≥ 0.70 in SA or Z, while C stays ≤ 0.45.
- **Fail:** the arm is within 0.10 of C.
- **Secondary:** stay rate (≤ 0.35), variogram V(8)/V(1) on long charts (≤ 1.25), and the own-prefix β_cum for SA (rising toward 0.5).
- **Cost:** about 1.2-1.8 h of Mac time per arm, plus about 1 h of evaluation.
- **If both anchors fail:** CE does not make the model hold any input over its own history. The lever then moves to the objective: training on its own rollouts (a prefix-rollout calibration or a DPO-style objective on its own continuations).

**Evaluation changes, at no training cost:** free-run panels must include long charts (K ≥ 1,500) and at least 25 charts per band × 3 seeds. Checkpoints are selected on the collapse measures, not on teacher-forced NLL or guard pass/fail alone.

**Controls (later):** LN share and length, difficulty and the five style concepts are all coordinates of θ. If Q1 passes with Z, a condition is a drawn or requested θ coordinate, and obedience is scored on whole-song free runs. Difficulty tilts push runs into the absorbing band-2 regime and stay on hold until the natural model holds identity. Per-chart style coordinates connect to the future "style baseline".
