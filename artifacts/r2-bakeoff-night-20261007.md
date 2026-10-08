# R2 bake-off round 1: the night of 2026-10-07

Question material for the first system comparison on the collapse problem ([r2-collapse-20261007](r2-collapse-20261007.md)). The plan it executes: [r2-bakeoff-plan-20261007](r2-bakeoff-plan-20261007.md). Main session 28c38740.

<a id="o-x0-first-read"></a>**Observation (agent read of the sheet against the key, unscored), 2026-10-07 about 18:06 UTC.**
- The human marked 6 of 8 generated songs as collapsed and 1 of 4 real source foils: x0-08, a band-5 chart of 451 s, at high confidence ("repetitive patterns, poor hand balance, lack of stream style feeling").
- Not marked: generated x0-05 (band 3) and x0-10 (band 4).
- The human's descriptions name LN distribution and organisation, jack organisation (same-lane, mechanical repetitive jacks), repetition and lack of variance, hand balance, and difficulty diverging between parts.
- Most first marks fall at about 20-85 s, which the agent estimated (K / song length, not the row maps) at rows 170-500, inside the encoder's 511-row history.
- Inferred: what the human calls collapse is mostly local organisation (F3) and absorption, appearing early, rather than identity lost past 511 rows (F2). Round 1 as planned targets F1 and F2 only. To be checked by the scoring ([r-x0-scoring](#r-x0-scoring)).
- The human added: the marked ranges are not precise, and only parts of each chart were checked ([private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#answer-1)). Unmarked windows are therefore not reliable negatives.

<a id="r-x0-scoring"></a>**Delegated 2026-10-07 18:05 UTC: X0 scoring and the X1 analysis** (fresh general-purpose subagent; brief `~/ensomi/.sync/cp/scratch/r2-collapse/x0-score/brief.md`; outputs `artifacts/r2-collapse-20261007/x0-score/` on bings-mac).
- Window-level AUC per candidate measure, 64-row windows, primary label "half the window inside a mark", with sensitivities: any overlap; negatives only from unmarked items; windows after the last mark dropped; marks widened by ±3 s and ±6 s (added after the human's caveat). Item-bootstrap 90 % CIs. Differences under about 0.05 AUC are read as noise.
- Also: the false-positive source, the row of each first mark, and the type notes mapped to statistics.
- X1 per the fixed start-dominant or drift-dominant rule ([p-collapse-experiments](r2-collapse-synthesis-20261007.md#p-collapse-experiments)).

<a id="d-bakeoff-night"></a>**Decision (human, 2026-10-07 about 18:08-18:10 UTC, [private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#answer-1), [prompt-2](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-2)).**
- Build now and fix the arm list after X0 is scored: incremental sampler, evaluation module, long-chart panel, B1-B3 behind flags.
- Decode-time block selection runs tonight as an arm for the local defects. The on-policy fine-tune is not ruled out; it is not built tonight.
- For tonight only, the pre-launch check is handed to a fresh Opus subagent, which reviews and corrects Astra's code; then training starts without waiting for the human. The human is asleep until morning in Japan.
- Agent reading, within the plan: B4 (two-rate encoder) is not built tonight (not in the chosen option). The plan's early-kill rules only freed a slot for B4, so the mid-run mini-panels are dropped; reader weight norms are logged instead. Training length is equal wall-clock, 1.25 h per arm, with each arm's cosine schedule set from its measured throughput.

<a id="r-bakeoff-build"></a>**Delegated 2026-10-07 18:12 UTC: Astra builds round 1** (ens job `20261007-181225-r2-bakeoff-build`, fast tier, `--rw`; brief `~/ensomi/.sync/cp/scratch/r2-bakeoff/brief-astra-build.md`).
- Arms: configs `ce_v2_bo_b1`, `b2` (about 16 cumulative self-anchor channels through a zero-initialised reader, plus history dropout on 25 % of windows truncated to 64-128 rows), and `b3` (B2 plus θ, 10 standardised whole-chart coordinates with a known bit, noised, dropped on 30 % of windows; a donor prior drawn from the 32 nearest fit_train charts in skeleton space). All warm-start from 56M with LN level and length off.
- Enablers: an incremental sampler state with an equality test against the current sampler; `collapse_eval.py` with m1-m7, gap closure G against B0 and the guards; the long-chart panel (25 groups per band, K ≥ 1,000 in band 2 and ≥ 1,500 above, X0 charts excluded; BOS at three seeds capped at 2,560 rows, plus one prefix continuation).
- `d0`: decode-time selection on 56M, best of N = 4 candidates per 64-row block, with a pluggable block score (`phi`: corpus ridge log-density; `envelope`: distance outside the band's source envelope). The score and the G measure subset are fixed after X0 scoring.
- Night runner `artifacts/r2-bakeoff-20261007/night.sh`, written, not run: B1, B2 and B3 train in series; panels run alongside the next training.

<a id="o-x0-scored"></a>**Observation, 2026-10-07 about 18:16 UTC: X0 scored. No candidate measure passes AUC ≥ 0.75 robustly** (fresh subagent, [r-x0-scoring](#r-x0-scoring); report `artifacts/r2-collapse-20261007/x0-score/report.md` on bings-mac, mirrored; tables `auc_table.csv`, `fp_source.csv`, `typenote.csv`; script `~/ensomi/.sync/cp/scratch/r2-collapse/x0-score/score.py`. The main thread checked the AUC and X1 tables against the mirrored files).
- Window-level AUC, 64-row windows. Item-bootstrap 90 % CIs are 0.3-0.45 wide; differences under about 0.05 are noise.
  - Five LN-share measures pass under the primary label: LN share vs the band's source IQR 0.83 [0.57, 0.91]; Fable's held drift 0.82; `bus4` 0.79; Fable's max drift 0.78; lane-lock 0.78.
  - They fall to 0.58-0.72 under any-overlap labels, ±3 s or ±6 s widened marks, or without x0-02, which supplies 15 of the 26 positive windows.
  - Within marked items they separate marked from unmarked windows at 0.47-0.68. Inferred: they separate songs by LN level rather than locate the marked spans.
  - Fail under every label set: C1 envelope exits 0.55-0.58; stay rate (item AUC 0.57-0.66); identity drift in nh 0.32 (inverted); Astra adjacency 0.51-0.53.
- The source marked collapsed, x0-08 (band 5), is an LN chart: held share 0.37 against a band-5 source median of 0.004. The LN measures rank it the most collapsed of the 12. C1 has its own false positive: x0-01, a clean source, exits in 62 % of windows because it is sparse.
  - Inferred: the human's "collapse" includes some real LN-heavy charting, so LN measures need a reference such as the chart's own level, not the band median.
- First marks: in 5 of the 6 marked generated items, before row 512 (rows 121-472, 4-28 % of K). The exception is x0-07 at row 1,623 (low confidence). 31 % of positive windows lie wholly before row 512, against a 23 % base share.
  - The share of each song after its last mark is 17-80 %, so the marks are partial and may be biased early.
  - Inferred: the marked collapse is mostly inside the encoder's 511-row history, which weighs against F2 as what the human sees most.
- The type notes in the statistics (marked vs unmarked windows of the same item):
  - x0-11 ("repetitive jack"): jack +2.3 IQR, rep1 +7.8, hlock +2.9; within-item AUC about 0.98.
  - x0-09: rep1 +1.6, hlock +1.85 IQR.
  - x0-07 ("difficulty diverges"): window SD of nh is 1.87× the band source median.
  - x0-12 ("bad LN and jack organization"): *less* LN in marked windows, so the organisation meant is not LN share.
  - x0-08's hand balance and stream feel do not show in hmax, lmax or jack.

<a id="o-x1-scored"></a>**Observation, 2026-10-07 about 18:15 UTC: X1 (start against drift) analysed** (`x1ana.py` unmodified; `artifacts/r2-collapse-20261007/x0-score/x1_verdict.json`, `phase0-opus/x1/x1_table.csv`). Removal fraction of the signed band offset in d_nh with a 64-row real prefix, at rows 256-512 / 512-1k:

| Band | 256-512 [90 % CI] | 512-1k [90 % CI] | Verdict (fixed rule) |
| --- | --- | --- | --- |
| 2 | 0.55 [0.31, 0.74] | 0.55 [0.31, 0.77] | start-dominant |
| 3 | 0.10 [−5.6, 0.93] | 0.54 [0.03, 0.92] | undetermined (offset ≈ 0 at 256-512) |
| 4 | 0.61 [0.24, 0.91] | 0.58 [0.10, 0.92] | start-dominant, wide CI |
| 5 | 0.43 [0.15, 0.70] | 0.26 [−0.04, 0.51] | mixed, just above the 0.25 drift cut |

- On per-chart |d_nh|, bands 4-5 are drift-dominant (0.09-0.34 removed at 512-1k). Every band's 512-1k CI spans at least two verdict zones.
- Reading: the start (F1) carries most of the band offset in bands 2 and 4. That supports B3's drawn θ there. Band 5 leans toward drift.

<a id="d-night-measures"></a>**Decision (main thread, 2026-10-07 18:16 UTC; proposed for the human's morning review): measures and the `d0` score for tonight.**
- X0 did not validate a measure subset. The plan's rule "only the validated subset counts" therefore does not apply. The bake-off judges arms on the full pre-registered set m1-m7 ([p-bakeoff-round1](r2-bakeoff-plan-20261007.md#p-bakeoff-round1)) and reports the five LN-share measures separately. A human blind screen of the winner stays the final judge.
- `d0` runs in two variants on 56M, BOS seed 954:
  - `d0-phi`, the chart's own restoring force on nh, held, c3 and pent (the `phi` score);
  - `d0-env`, a penalty on local organisation outside the band's source envelope over jack, fjack, rep1, hlock, lock and `bus4`. These are the statistics that matched the clearest type notes and the best AUCs. LN share is left out, because of the x0-08 finding.
- The two variants split decode-time holding from decode-time local organisation, the defect the human marked.

<a id="d-decode-direction"></a>**Direction (human, 2026-10-07 about 18:52 UTC, [private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-6)): decode-time selection, done right.**
- Tonight's `d0` is naive but probably not wrong, and it stays.
- The right version ranks whole sequences with a stateful heuristic that is efficient to compute and adds up to one whole-sequence score. Ideas to borrow: HMMs, CRFs, semi-Markov CRFs.
- The final judge is the human's whole-song blind screen. The human's time ranges are imprecise, but bad patterns across a whole song are spotted reliably.
- Working assumption: the wanted charts are already in the model's distribution and are not sampled. To be tested.
- Reward hacking is a real risk.

<a id="r-decode-design"></a>**Delegated 2026-10-07 about 18:55 UTC: design round on a stateful whole-sequence score.** Fable and Opus work as fresh subagents, independently, from the same brief, `~/ensomi/.sync/cp/scratch/r2-decode/brief-design.md`. Due in 60 minutes.
- Asked for: a formal statement and a cheap test of the assumption; at least three stateful scores (HMM, linear-chain CRF, semi-Markov or explicit-duration), with what each sees, what it is fitted on, cost and decomposition; whole-sequence search (rerank, SMC with twisting, beam); reward-hacking safeguards; a human blind-screen protocol built on whole-song calls; and one recommended experiment with a fixed pass/fail rule.
- Probes on existing runs only, with 1 mac process; no generation while the night runs.
- Reports: `~/ensomi/.sync/cp/scratch/r2-decode/{fable,opus}/report.md`.

<a id="r-bakeoff-built"></a>**Astra's build ended, 2026-10-07 about 18:58 UTC, exit 0. Not committed, not launched** (job `20261007-181225-r2-bakeoff-build`; handoff `artifacts/r2-bakeoff-20261007/reports/build-handoff.md`, mirrored).
- Changed: `data.py`, `features.py`, `model.py`, `sampling.py`, `generate.py`, `train_ce.py`. Added: the θ, collapse-evaluation, panel and `d0` modules, three arm configs, six test files and `night.sh`.
- Tests: 352 passed, 4 skipped (Astra's count).
- Sampler at k ≈ 2,000: 6.89 → 1.44 ms per step. The output matched byte for byte on 3 cached charts × BOS and prefix × 600 rows.
- Throughput in exposures per second, with `total_exposures` set for 4,500 s: B1 2,665 (12.0M), B2 2,963 (13.3M), B3 2,964 (13.3M). History dropout makes B2 and B3 faster.
- The anchor has 13 named channels.
- Panel: band 3 has only 24 eligible groups at K ≥ 1,500 after the X0 exclusions. Main-thread choice: keep the threshold, so 99 charts.
- Pre-launch check: a fresh Opus subagent from about 19:01 UTC (brief `~/ensomi/.sync/cp/scratch/r2-bakeoff/brief-opus-review.md`, which adds the two `d0` variants and the m1-m7 subset).

<a id="s-decode-design"></a>**Decode design round returned, 2026-10-07 about 19:12 UTC: [Fable](r2-decode-fable-20261007.md), [Opus](r2-decode-opus-20261007.md). Main-thread synthesis, proposed for the human's review.**
- **Agree (measured by both on existing runs).** A row-level density score from an HMM is the wrong instrument.
  - Fitted on real charts alone, it is a rarity meter: sparse, LN-free, jack-free charts score as most real. Contrasted against the model's own runs, it learns the band offset (Fable) or ranks the human's marks the wrong way round (Opus: window AUC 0.24-0.31). The corpus holds jacks and LN in charts built around them, so without chart context the score rewards out-of-context jacks and LN.
  - Both point to scores relative to the chart's own running level (Opus's sID: item AUC 0.83, window AUC 0.60-0.66; exploratory, coordinates chosen after reading the type notes).
  - Both point to durations modelled per chart type: real charts that stay uniform exist (x0-01), so a global duration law would penalise them.
- **Measured, new:** the chance of leaving a degenerate stretch falls with its length in the model's runs (0.57-0.62 after one flagged window, 0.15-0.31 after 3-6) and is flat in sources (0.74-0.80). Selection has to act early in a stretch.
- **Agree on safeguards:** use the score for ranking or as a tilt only, keep the model's log-probability, keep held-out evaluators that are never selected on, guard variety across seeds, and make the human screen with real foils the final judge.
  - Fable adds a satisficing clip: pick at random among candidates inside the real charts' score range, so a score above q95 is suspect rather than better.
- **Differ on the search.**
  - Fable: whole-song rerank with N = 32, about 3.3 Mac-hours for 100 charts.
  - Opus: SMC with 4 particles over 64-row blocks, at `d0`'s compute. The argument: the whole-song acceptable mass falls with length (back-of-envelope from X0's 8 items, about 0.35 at 1,550 rows and 0.13 at 3,000), while the per-block mass stays near 0.96.
  - The main thread finds Opus's argument stronger for long songs. SMC also keeps the whole-sequence target and suits real-time play.
- **New and worth keeping (Opus): a source-oracle arm.** The same block search, scored against the real chart's same rows. It tests directly whether acceptable charts are in the model's proposals. If even the oracle cannot reach the source's stay rate and identity, the lever is training, not decoding.
- **Proposed next, not authorised:** `d1` = SMC, P = 4, scored by chart-relative identity plus a penalty on staying in a flagged stretch; plus the oracle arm. It reuses tonight's B0 and `d0` panel runs, about 75 Mac-minutes for 48 charts × 2 seeds. Then a human screen of 6 songs × (plain, `d0`, `d1`) + 2 real foils, about 1 hour, with Opus's fixed pass rule. Fable's satisficing clip can be added to `d1`'s weights.

<a id="d-review-amendments"></a>**Amendments before any result, 2026-10-07 about 19:20 UTC, after an external review the human shared** ([private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-7); the review is ChatGPT Pro's, kept privately).
- **Main-thread judgment:** the review is largely right. Continue B1/B2/B3/`d0`; correct the evaluation and the causal claims before results exist.
- **Applied (passed to the pre-launch check, or edited on the page):**
  - **G is two-sided:** g = |m_sys − m_src| / |m_B0 − m_src|. The signed version rewarded overshooting past the source, so a system with too much spread would score g < 0 and win.
    - Terms where B0 is within the source's bootstrap noise are dropped.
    - G is also reported without the band-relative terms (m1 envelope, m6 band offset). A natural generator that draws a different but legitimate organisation for a skeleton need not match the source chart's band.
  - **B3 in prefix mode** takes θ computed on the observed prefix, or "unknown" if that is not simple. A skeleton-only donor draw could contradict the prefix it is scored on preserving.
  - **A B3 channel-use diagnostic, if cheap:** shift held share or nh in θ by ±1 within-band SD at fixed prefixes, real and own, and report the realised slope.
  - **The winner rule becomes a shortlist plus a human blind comparison that includes B1**, on matched songs. A trained arm that wins is repeated with a second training seed before it becomes the base.
  - **Three claims are kept separate:** the input is used; the generated process improves; the charts improve.
  - **Causal wording corrected** on [r2-architecture-20261007.html](r2-architecture-20261007.html), and to be read the same way in [s-arch-constraints](r2-bakeoff-plan-20261007.md#s-arch-constraints) and the plan's outcome lines:
    - teacher forcing by itself is not the cause; sequence CE is a KL on whole charts. The present recipe gives weak pressure toward a persistent identity and does not train recovery;
    - explicit θ is an inductive bias, not a prerequisite;
    - "nothing beats B1" would not show that CE cannot hold any input;
    - θ coordinates are one candidate route for controls, not their definition.
- **Not applied tonight, kept for later:**
  - B2 can hold a wrong start, and truncating history to 64-128 rows may leave the shortcut. The diagnostic and the BOS-against-prefix split will show it.
  - For on-policy work, labels must be valid under the generated history (naive scheduled sampling is inconsistent), and there should be no generic anti-repetition objective.
  - For the next screen, reviewed passages early, middle and late, with "not inspected" kept separate from "acceptable".

<a id="r-night-launched"></a>**Pre-launch check passed; night launched 2026-10-07 19:32 UTC** (check: fresh Opus subagent, 19:01-19:30 UTC, GO; code `df4458c` on `r2/train`, not pushed; ens job `20261007-193157-r2-bakeoff-night`; runner `artifacts/r2-bakeoff-20261007/night.sh` on bings-mac, source copy `~/ensomi/.sync/cp/scratch/r2-bakeoff/night.sh`).
- The check fixed:
  - the anchor's pattern entropy, now normalised by non-empty rows;
  - θ's noise SD, which was √2 too large; θ values are unchanged;
  - m3-m5, now with each run as one observation, where a chart's runs had been averaged first;
  - `night.sh`, now running both `d0` variants with no `set -e` and at most 8 threads.
  - It applied the [amendments](#d-review-amendments): two-sided G with `G_core` (no m1 or m6) beside it; B3's prefix-mode θ computed on the prefix; the θ channel-use probe `theta_probe.py`.
  - It removed defensive code in `theta.py` and an unused oracle-from-source path.
  - Tests: 353 passed, 4 skipped. B3's training smoke: 420k exposures in 150 s, no NaN, both readers logged.
- Main thread, after the check: `d0-env`'s score summed raw distances, so `lock` and `hlock` (in rows) outweighed the share statistics. Each distance is now divided by its envelope width; tests passed (job `20261007-193129-bo-envscore-test`).
- Launch arguments: `d0` = `phi,envelope`; G = m1-m7; envelope statistics = jack, fjack, rep1, hlock, lock, bus4.
- Known limits, from the check:
  - the single-head and 4-gram guard can hardly fail;
  - `d0-*`, `b3-unknown` and `b3-oracle` are matched to B0 on seed-954 BOS runs only;
  - B2 and B3 draw different history-dropout masks;
  - schedules will likely stretch past 1.25 h with panels alongside. The check estimates the night at 4-4.5 h.
- Evaluation is separate from generation, so G or the guards can be recomputed on the saved runs in minutes.

<a id="o-bakeoff-r1"></a>**Observation, 2026-10-08 01:11 UTC: round 1 finished (exit 0). No arm passes the shortlist rule. Fine-tuning alone made free-running worse, θ brings it back to about B0, B3 barely follows θ, and decode-time selection gets the lowest G partly by removing LN.** (`artifacts/r2-bakeoff-20261007/comparison.md`, `b3-probe/probe.md` on bings-mac, mirrored; read by the main thread.)
- **Training:** all three arms finished with no NaN.
  - B1: 12.0M exposures in 7,124 s. B2: 13.3M in 5,686 s. B3: 13.3M in 6,029 s.
  - The panels running alongside stretched the planned 1.25 h, so the arms were matched on exposures, not wall-clock.
- **G** (two-sided; lower is better; B0 = 1), with G without m1 and m6 in brackets:

  | System | G | G_core | Runs matched to B0 |
  | --- | ---: | ---: | ---: |
  | B1 | 1.41 | 1.21 | 396 |
  | B2 | 1.29 | 1.20 | 396 |
  | B3 (prior θ) | 1.02 | 0.96 | 396 |
  | B3, θ unknown | 1.18 | 1.06 | 198 |
  | B3, oracle θ (diagnostic) | 0.90 | 0.90 | 198 |
  | `d0-env` | 0.87 | 0.88 | 99 |
  | `d0-phi` | 0.91 | 0.78 | 99 |

- **The bootstrap intervals are not interpretable.** The lower bounds sit near −2 for most differences, with point estimates at the upper edge (e.g. B0 − B1 −0.41 [−2.18, −0.33]). Inferred cause: the ratio's denominator |m_B0 − m_src| is re-estimated in each resample and nearly vanishes in some. As computed, every candidate's interval against B1 includes 0: B3 0.39 [−0.06, 1.03], oracle 0.43 [−0.02, 2.48], `d0-env` 0.25 [−0.83, 0.91]. So nothing passes the shortlist rule. The rule's interval needs a fixed scale before it can decide anything.
- **B1 is worse than B0 by 0.41 in G.** It has more LN and more all-four-busy rows in bands 2-3 (`bus4` 0.087-0.089 against 0.030-0.038 in sources) and longer lane locks. Per the plan's own rule, fine-tuning alone moves the regime, so any margin needs a second training seed.
- **B3 recovers to about B0, from B1's 1.41.** Its band 4-5 LN spread is closer to the sources than B0's (held IQR [0.069, 0.212] in band 5, against [0.064, 0.244] in sources and [0.077, 0.151] for B0). Bands 2-3 stay too LN-heavy. θ unknown is worse (1.18) and oracle θ better (0.90), so θ is read, but weakly.
- **Channel-use probe (B3), realised shift per requested shift:**
  - held share: 0.23 from real prefixes, 0.05 from the model's own prefixes;
  - nh: about 0 from both.
  - The plan's early-fail threshold was a slope below 0.6. Reading: B3 barely follows θ, and almost not at all on its own output, which is where it was needed.
- **Decode-time selection gets the lowest G, partly by gaming the statistics:**
  - `d0-phi` strips LN in bands 4-5: held IQR [0.009, 0.078] against sources [0.058, 0.247]; `bus4` 0.011-0.018 against 0.076-0.118; band-5 stay rate far below the source's (signed −27).
  - `d0-env` also lowers band-5 LN (held [0.026, 0.128], `bus4` 0.039 against 0.118).
  - `d0-env` is the only system that passes the head-lock and tail-exit guards, and it selects on those very statistics.
  - This is the LN-avoidance hack both decode designs predicted ([s-decode-design](#s-decode-design)).
- **Guards:** every system, B0 included, fails the head-lock-≥30 and tail-exit guards except `d0-env`. Legal export, NLL and the single/4-gram guard pass for all.
- **What this does and does not show:**
  - None of these measures is validated against the human (X0). The ranking is about distance from source statistics, not about charts the human would accept.
  - Under this recipe and budget, the three trained arms do not beat B0 in free-running statistics, and B3's θ is weakly used. Fine-tuning itself is a confound.
  - Decode-time selection moves the statistics most, with visible reward hacking.
