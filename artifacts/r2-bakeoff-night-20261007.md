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
