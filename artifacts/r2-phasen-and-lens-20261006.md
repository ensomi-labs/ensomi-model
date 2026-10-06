# Phase-N values and the Lens data for phase C (Astra, 2026-10-06)

Two Astra jobs ran on the mac at the human's request ([private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-7), amended by [prompt-8](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-8)). Both used the fast tier at effort xhigh, under 2 h, from 12:58 to 14:44 UTC. Reports are in the code checkout, git-ignored and mirrored here: `artifacts/r2-phaseN-tune-20261006/report.md` and `artifacts/lens-phasec-20261006/report.md`. No tracked file was changed and no full training ran. The main thread read both reports and spot-checked the tables quoted below against them; it did not re-run anything.

<a id="s-phasen-tune"></a>**Phase-N values (job `20261006-125849-r2-phasen-tune`).**
- **Divisor:** n_bar = 898.916, relative SE 0.65% over 2,000 draws; n_bar_ln = n_bar_star = 0; key verified.
- **Device:** CPU with 4 threads, about 2,029 head decisions/s over the peak pilots. MPS ran at about 986/s with 1 thread, roughly half.
- **Learning rate:** pilots of 512 updates (458k heads, two weight seeds) gave mean natural NLL of:

  | Peak lr | Mean NLL |
  | --- | --- |
  | 1e-4 | 3.183 |
  | 3e-4 | 3.076 |
  | 1e-3 | 3.032 |
  | 3e-3 | 2.995 |

  1e-4 is worse than 3e-4. The three higher rates are tied under the two-SE rule; the trend favours higher, but that is post-hoc. Warm-ups of 5k, 20k and 50k are tied.
- **Proposed config** (`ce_v2_n.proposed.json`, accepted by `check_config`): lr 3e-4 (the lowest of the tied rates, by an operational rule), warm-up 20k, cosine to 3e-5, 64M exposures, checkpoints every 4M, `g3c_exposures` [].
  - The 64M budget is extrapolated from the v1 curve, whose minimum was at 61.4M.
  - The run would take about 13.6 h, of which about 3.9 h is evaluation; one full evaluation, mostly calibration, takes about 29 min.
  - The pilots stopped at 0.7% of the proposed budget, so the peak lr is not decided at full budget.
- **Proposed follow-up, not run:** 3e-4 against 3e-3 at 4M heads, with fresh seeds 173 and 174 and the 64M schedule clock. That is about 2.2 h of raw training.

<a id="s-lens-data"></a>**Lens annotation data (job `20261006-125845-lens-phasec`).**
- **The "latest three batches"** are read as the three newest completed deliveries of today's campaign, `labeler-025` to `labeler-027`: 75 4K charts with one section each (median 10 s, 76 head rows) and 360 supported labels. Gold is the current High-confidence human layer: 171 labels on 58 sections.
- **Join to the R2 cache:** 280 labels on 74 charts (66 fit_train, 8 fit_dev).
- **The larger published machine layers** (v2.1, v2.1 repair, v3) hold 6,801 labels on 1,434 sections and 656 charts. With High gold they join R2 on 5,400 labels and 535 charts.
- **Label form:** five independent concepts (Jack, Stream, Trill, Tech, LN coordination), each absent, present-supporting or present-prominent.
- **Coverage is sparse:** labelled time is a median 2-12% of a chart, at 1-3 sections per chart about 27-50 s apart. There are no dense within-chart trajectories.
  - First-order transition gains are at most 0.1 nats, and they are not order-sensitive.
  - The data do not identify HMM transitions or dwell durations.
- **Agent quality:** the latest batches share no section with gold, so current agent accuracy is not validated. Historical kappa against High gold was 1.00 for Trill, 0.93 for Jack, 0.79 for LN coordination, 0.73 for Stream and 0.48 for Tech.
- **Rare levels:** in 2-6★, the latest batches have no prominent Trill and only 2 prominent Tech cells.

<a id="a-phasec-method"></a>**Phase-C method, Astra's proposal (not decided).**
- **Properties** (LN share, difficulty: exact metrics):
  - frozen-base phase C first, with the approved star proxy;
  - an LN count-feedback decoding baseline kept as a comparison, under short-hold and feasibility guards. In v1, count feedback reached MAE 0.0007 but gave 2.8% holds ≤ 60 ms against 0 in the source;
  - an own-sample LN term stays the human's call (Q-I).
  - The report adds that a 40 ms floor alone does not enforce the ≤ 60 ms guard.
- **Styles:**
  1. Train a section scorer (per concept, three levels) and validate it on gold.
  2. Use it to test whether R2 already samples the requested styles (candidate support).
  3. Then try best-of-N or reranking inside the scope.
  4. Then scoped style conditioning for the concepts and levels with enough data.
- **Defer** HMM/HSMM and CRF transitions (no trajectory data; a persistence prior would impose a hand-chosen aesthetic) and DPO (Lens labels are not same-context preferences).
- **Search** (SMC or beam) must branch only on decisions in V; tails after b come from the base law.
- **Operating point:** the same frozen evaluator for every arm; the least-steering Pareto point that meets the adherence requirement.

This refines [h-decoding-style](decoding-stage.md#h-decoding-style). Decoding-time selection is supported as best-of-N or reranking with a validated scorer. A learned HMM over style states is not supported by the present data.

**For the human:**
- the phase-N peak lr: accept 3e-4, use 1e-3 (the v1 precedent, stable to 61M), or run the 4M follow-up first;
- the budget (64M, about 13.6 h);
- memory (plan v5 decision 11);
- the Lens training population: the latest three batches only, or the published layers plus High gold;
- frozen first for phase C.
