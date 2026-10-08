# Diagnosis round after bake-off round 1: the bigger picture (Fable, 2026-10-08)

Report of the fresh Fable subagent "fable-picture", one of three analysts launched at about 01:35 UTC after round 1 ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1)). Brief: `~/ensomi/.sync/cp/scratch/r2-diagnose/common.md` plus `brief-fable-picture.md`. Returned about 02:00 UTC, after 28 minutes.

The main thread saved the returned text with two edits: references to the human are made pronoun-neutral, and the return time is corrected. It spot-checked the reader norms against `artifacts/r2-bakeoff-20261007/reports/readers-training.md` and the pilot outputs exist.

Tags: **M** measured (a number from a file or job named here), **I** inferred from measured parts, **P** proposed.

- Jobs: `20261008-013842-fp-ln-audio` (pilot v1, 335 s), `20261008-014616-fp-ln-audio-v2` (pilot v2, 471 s). Mac budget held: 1 process, 2 threads, 13.4 min.
- Scripts: `~/ensomi/.sync/cp/scratch/r2-diagnose/fable-picture/ln_audio_pilot.py`, `ln_audio_pilot_v2.py`.
- Outputs: `artifacts/r2-diagnose-20261008/fable-picture/` and `…/v2/` (mirrored).
- The seed-noise table in section 2 was computed from the mirrored `artifacts/r2-bakeoff-20261007/b{0,1,2,3}/evaluation.json`; no script was kept.

## 1. Are we directionally right?

**Half right (I).** The information-path diagnosis is real and well measured: F1 start offset, F2 no holding, the 511-row boundary, β_cum halving on own prefixes. Two measured facts say it is probably not what the human calls collapse:

- **The marks are early and local.** The human's first marks sit at rows 121-472 in 5 of 6 marked items, inside the 511-row history (M, `o-x0-scored`). The type notes describe local texture: mechanical same-lane jacks, repetition, poor hand balance, LN placement, difficulty diverging between parts.
  - Within marked items, the LN-share measures separate marked from unmarked windows at 0.47-0.68. Stay rate, envelope exits and the adjacency gap are at chance (M).
- **Runs wander, they do not stick.** Under a real-fitted regime model, runs stay in one regime for *shorter* spans than sources: 2.7 against 3.3-3.5 blocks (M, decode-Fable probe B). "Absorption" is a property of the band envelope, not of the charts.

Round 1 therefore optimised what the measures could see, chart-level means and how they hold, while the human judges local organisation and section structure. The bake-off's null is consistent with that. B3 with oracle θ, the strongest possible chart-level input, reaches G 0.90, not a different kind of chart (M).

**What a strong researcher would say is missing (I; standard results in sequence and music generation, and mapping practice):**

1. **An objective with gradient on own histories.** Compounding per-step error on self-generated context is exposure bias.
   - Remedies: training on own rollouts (prefix rollout, then teacher-forced CE on the real continuation), scheduled sampling, sequence-level or preference objectives, or decode-time search against a critic.
   - The team wrote this as the fallback and bet the night on inputs.
   - "CE is a KL on whole charts" is a population statement. At finite data and 2.3M parameters the per-step error is not zero, and the 2× stay rate is the policy's sensitivity to its own error.
2. **Section structure.** Mappers write in sections, with consistency inside and contrast between. "Difficulty diverging between parts" is a section complaint.
   - The model has no section variable. The phrase test only looked for concentration at 8/16-bar starts.
   - Section boundaries come from audio or structure, not from the grid.
3. **A texture critic the human agrees with**, before any further training round. Without it, every arm is judged by proxies X0 already failed.
4. **Audio** is the long-run grounding for lane and LN texture. But the pilot (section 3) says the simplest mixture features add little for LN placement, so it is not the quick win for "weird LN distribution".

## 2. What the estimates did not cover

| Item | Status | Evidence |
| --- | --- | --- |
| Checkpoint-to-checkpoint and seed noise floor | **Never measured before margins were set**; now partly measured (M) | See the seed-spread findings below this table. |
| Selecting B0 on the same statistics | Winner's curse (I) | 56M was the best of five checkpoints on guards iv-v, so any retrain regresses toward the checkpoint mean. B1 > B0 in G is expected even if training does nothing. |
| Measures not validated against the human | Known before the night and overridden (M, `d-night-measures`) | X0 validated nothing, and the night was judged on m1-m7 anyway. `d0-phi` gets the best "G without m1, m6" (0.78) by stripping LN in bands 4-5 (held IQR [0.009, 0.078] vs [0.058, 0.247]), so the measure is gameable. |
| CE gives weak pressure on chart-level inputs | Predicted and confirmed; the kill rule never fired (M) | See the reader-norm findings below this table. |
| Musical grounding | Pilot says small (M, section 3) | Audio adds +0.01-0.02 AUC for LN presence over the skeleton. The chart's own LN share adds +0.15 pooled and 0 within chart. LN placement is chart-level allocation plus skeleton, not a mixture-energy cue. |
| B3's prior fixes variety by construction | Design artefact (I) | θ is drawn from the 32 skeleton-nearest fit_train charts, and the skeleton explains R² 0.03-0.20 of θ, so the draw is nearly a band draw. m5 improves whether or not θ is used. |
| Band-relative two-sided measures | Reward blandness (I) | G rewards the band mean, but real charts are multimodal (band-5 sample: LN share median 0.048, IQR 0.006-0.439). The decode-Fable chart-relative identity score pointed the right way (item AUC 0.83). |
| Long-chart panel | Style bias (I) | K ≥ 1,500 in bands 3-5 selects marathons and LN charts, while the envelopes came from 116 ordinary fit_dev charts. |
| Guards | Miscalibrated to long charts (M) | Every system fails the head-lock and tail-exit guards, B0 included. The thresholds came from the 16-chart K ≤ 600 panel. |

**Noise floor, measured.**
- Phase-N prefix-panel `sd_ratio` went 1.05 → 0.52 → 0.94 at 40.0M, 40.4M and 41.4M exposures, with NLL 2.071-2.085 (`r2-runs/r2-phaseN-20261006/evals.jsonl`).
- From the bake-off records (BOS, seeds 954-956, 25 charts per band, mean over bands of max − min across seeds), seed spread against arm minus B0:

  | Statistic | Seed spread per system | B1 − B0 | B2 − B0 | B3 − B0 |
  | --- | --- | ---: | ---: | ---: |
  | m1 | 0.025-0.062 | 0.078 | 0.052 | **0.013** |
  | d_nh | 0.044-0.080 | 0.116 | 0.081 | **0.052** |

  - d_held: seed spread 0.026-0.058, arm differences 0.008-0.020.
  - bus4: seed spread 0.014-0.044, arm differences 0.011-0.023.
- **B3 − B0 is inside the seed spread on every statistic. Only B1's busier regime (+0.12 nh, m1 +0.08) clears it.**
- B1 moved G by +0.41 with no design change. The 2026-10-02 rule "noise floor before criteria" was skipped.

**Reader norms, measured.**
- B2's anchor-reader norm is 0.041 after 13.3M exposures; B3's anchor 0.037 and θ 0.064 (`reports/readers-training.md`). The θ-probe slope is 0.05 on own prefixes.
- The NLL of B2 and B3 equals B1's (2.080-2.083 against 2.081): the channels bought nothing.
- The early sign "reader norm near 0 at 2M" had no reference scale.
- Inferred: dropout to 64-128 rows on 25 % of windows does not break the redundancy. On real charts the last 64 rows already give the chart level (F2's own mechanism), so the window stays the cheaper predictor.

## 3. Musical-grounding pilot (M)

**Setup.** 200 fit_train charts (50 per band 2-5, K 400-2,500, one per song group, seed 954), 230,450 head rows. Audio is located as the X0 pack did (`AudioFilename` next to the `.osu`), decoded with soundfile into a 64-band log-mel at 10 ms. Models: logistic and a 2×64 MLP, 4-fold by chart.

- **S, skeleton:**
  - gaps to the previous and next head in ms and beats;
  - beat phase, position, density at ±1 s and ±4 s;
  - time to the next and previous rest ≥ 2 beats;
  - row index, beat length, band, global density.
- **A, audio** (v1, 14 features):
  - onset strength at the head, peak energy;
  - energy decay at 100/250/500 ms, sustain share above −6 dB in 500 ms, decay time to −10 dB;
  - onset activity inside the gap and in the next 300 ms;
  - spectral similarity to the previous and next head frames;
  - centroid, low-band share, energy relative to the song median.
  - v2 adds 8: low/mid/high band decay at 250 ms, per-band persistence at 250 ms and at mid-gap, spectral stationarity 30 → 300 ms, accent contrast to the next and previous heads.
- **T, chart-level oracle:** chart LN share excluding the row, median log2 LN length, jack rate.
- **Ashuf, control:** audio rows permuted within chart.

v1 results (v2 in brackets where it differs by ≥ 0.005):

| Target | S | S+A | S+Ashuf | S+T | S+A+T | A alone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LN presence, pooled AUC (lin / mlp) | 0.743 / 0.739 | 0.754 / 0.763 | 0.749 | 0.898 / 0.900 | 0.898 / 0.901 | 0.639 (0.653) |
| LN presence, within-chart mean AUC | 0.756 / 0.729 | 0.762 / 0.747 | 0.750 | 0.753 / 0.757 | 0.767 / 0.776 | 0.652 (0.674) |
| LN length log2 beats, pooled Spearman | 0.585 / 0.576 | 0.606 / 0.583 (0.611) | 0.586 | 0.632 / 0.626 | 0.645 / 0.637 (0.650) | 0.282 (0.395) |
| LN length, within-chart Spearman | 0.427 / 0.415 | 0.453 / 0.434 | 0.424 | 0.427 / 0.413 | 0.451 / 0.437 | 0.252 (0.301) |
| LN extends past the next head (59 % of LN), pooled AUC | 0.615 / 0.619 | 0.636 / 0.625 (0.640) | 0.610 | 0.653 / 0.600 | 0.663 / 0.629 | 0.516 (0.568) |
| Jack vs not at gap ≤ 600 ms (21 % jacks), pooled AUC | 0.836 / 0.848 | 0.845 / 0.857 | 0.835 | 0.870 / 0.879 | 0.879 / 0.889 | 0.567 (0.615) |
| Jack, within-chart mean AUC | 0.793 / 0.807 | 0.810 / 0.822 | 0.791 | 0.793 / 0.808 | 0.813 / 0.826 | 0.590 (0.632) |

**Per band, pooled AUC:**
- LN presence, S → S+A (lin): band 2 0.694 → 0.718, band 3 0.738 → 0.733, band 4 0.759 → 0.765, band 5 0.760 → 0.778.
- Jack, S → S+A (mlp): 0.734 → 0.739, 0.793 → 0.804, 0.819 → 0.854, 0.873 → 0.896.

**Single audio features (v2).** Every one is within 0.39-0.61 AUC for LN presence.
- The strongest are negative: onset activity in the next 300 ms (0.41) and mid-gap persistence (0.39). LN heads sit where the gap is quiet.
- Mid-gap persistence is the best single cue for "LN extends past the next head" (0.57).

**Reading (I).**
- **Mixture audio adds real but small information beyond the skeleton:** +0.01-0.02 AUC for LN presence, +0.02-0.03 Spearman for LN length, +0.01-0.035 AUC for jacks (largest in bands 4-5).
  - All of these are above the shuffle control, which stays at S.
  - The band-wise sustain features (v2) change nothing when S is present, while A alone improves. So the new features are real but redundant with the skeleton.
- **The chart's own LN share dominates where LNs go** (pooled 0.74 → 0.90) and adds nothing within chart.
  - LN placement is first an allocation per chart, then a local skeleton decision. This is the θ story of F1 at row level.
  - So "weird LN distribution" is more likely wrong allocation and release organisation than missing audio.
- **Model class does not matter** (MLP ≈ logistic): the ceiling is the features.
  - Mixture energy is dominated by drums and bass, while mappers follow vocals and synth lines.
  - If audio is pursued, the next step is a source-separated or pitch-salience probe. A learned audio front end is beyond a pilot.
- **Failed:** 1 of 201 charts (a libsndfile error on one mp3, `dataset/0/675918/...Frost Crystal].osu`).

## 4. Capacity or kind of network?

**Evidence (M).**
- Phase-N dev NLL fell 2.84 → 2.07 nats per decision from 1.3M to 48M exposures, then stayed at 2.070-2.084 to 64M (about 5 passes over 12.7M decisions). B1-B3 stay at 2.080-2.090.
- Calibrated on real histories within ±0.05 heads per row in every band and position.
- Regimes flip between checkpoints at equal NLL (D8). The band-2 offset is set by row 64.
- The 35M teacher of the R1 lineage reached better NLL (1.61 vs 1.74), and had more fixed-lane runs at 30M than at 5-10M (16 cases, one seed).
- The hidden 128/192/256 sweep (`r2-tune-20261003-scale*`) ran 8 minutes each to 0.5-1.4M exposures, so it says nothing.
- F3: 0.18 bits below sources from the first third, not growing (Astra). Then X2: +0.075 bits at 1,024 vs 128 rows of own history.
- Not retrievable here: the train-side NLL (the mirrored `logs/resources-*.jsonl` hold memory only), so the train-dev gap is unknown.

**Reading (I): objective and information, not capacity.**
- A capacity-limited model is stably mediocre. This one is calibrated on real data, unstable on its own output, and flips regime with no NLL change.
- Capacity may still bound F3 and LN-release organisation. Nothing measured separates that from data noise.

**Cheap discriminating pilot (P, about 1 Mac-hour, no training).**
- Per-decision teacher-forced NLL on fit_dev, split by decision type (tap, LN head, release, chord size, jack), at checkpoints 16M-64M. Measure it on real prefixes and on own 512-row prefixes (the X2 material).
- Reading:
  - flat per-type NLL on real prefixes from 32M for every type, while own-prefix NLL keeps diverging: capacity is not binding, and the gap is the rollout objective;
  - LN-release or chord NLL still falling at 64M: exposure or capacity binds there.
- A hidden-256 run from scratch to 20M exposures (6-8 Mac-hours) is the only direct capacity test; rank it last.

## 5. If the 7 measures fail, which others are unvalidated proxies?

| Measurement | Rests on | Validation status (M) | Verdict (I) |
| --- | --- | --- | --- |
| Stay rate, degenerate windows (m1, m2) | Band 1 %/99 % envelopes on 64-row windows | X0 item AUC 0.57-0.66; regime durations under a real-fitted model shorter in runs than sources | Distance from the band mean, not stickiness; drop as a primary |
| β_cum anchor | OLS of the next-window LN response on cumulative vs recent share | Descriptive; real-prefix β uninformative by construction | A diagnostic of what the model reads, not of quality |
| X2 adjacency gap | A shuffle null that also breaks timing and state relations | X0 window AUC 0.51-0.53; 8 charts | Unvalidated; the null is too strong |
| Guards (head-lock ≥ 30, tail exits, iv-v) | Thresholds from the 16-chart K ≤ 600 panel | Every system fails on the long panel, B0 included | Miscalibrated; 56M was selected on iv-v |
| Teacher-forced NLL | Population fit on real histories | Flat while regimes flip | Valid for one thing: an unused input shows as zero NLL gain (B2/B3) |
| m3, m4 (thirds correlation, variogram) | Chart-relative, long charts | X0 nh identity drift inverted (0.32) | Right reference, wrong statistic (means over thirds) |
| LN-share measures | Band median | 0.78-0.83 under one labelling, 0.47-0.68 within items; the real LN chart x0-08 ranked most collapsed | Confounded by the chart's own level |
| G and its bootstrap | Per-resample denominator | Interval lower bounds near −2 | Not interpretable as built |

**A measure that agrees with the human needs (P):**
- **a chart-relative reference:** the chart's own first third or own level, never the band median;
- **the texture features the human named, gap-conditioned:** same-lane jack runs at speed, hand imbalance over 8-16 rows, release-into-next-head gaps, repetition of 4-8-row lane patterns beyond the chart's own base rate;
- **position and section:** the chart's density and complexity profile against the skeleton's density profile, which is where "difficulty diverging between parts" lives;
- **song-level aggregation:** the human is reliable at the song and imprecise in time, so use items with n ≥ 24 rather than 64-row windows;
- **calibration on the human's calls**, which exist for only 12 items so far.

## 6. How to tune the system: ranked plan (P)

**Stop:**
- CE fine-tune arms judged on m1-m7 or G;
- checkpoint selection by guards or NLL;
- bootstrap intervals on G as built;
- 64-row window statistics treated as truth;
- audio as a near-term lever for LN placement.

| Rank | Step | Cost | What it teaches whatever the outcome |
| --- | --- | --- | --- |
| 1 | **Noise floor.** Run the panel at seeds 957-958 on 56M, and seed 954 on 48M, 52M and 60M. Compare m1-m7, G and the LN tables between seeds and between checkpoints; the section-2 seed table is the start. | ~1.5 Mac-hours, no training | The margin any arm must beat. If the between-checkpoint SD of G is ≥ 0.3, round 1 had no result, and every future comparison needs 2 training seeds. |
| 2 | **Human calibration set.** 24 whole songs, blind: 12 B0 at two seeds on 6 skeletons, 6 real, 6 `d0-env`. Record a whole-song call, type tags and the first bad passage. Same-skeleton pairs give preferences. | ~30 min Mac, 1.5-2 h human | Ground truth for measures and for a preference objective. If both seeds pass on most skeletons, collapse is per sample and selection suffices; if both fail, it is per model. |
| 3 | **Measure validation.** Test the chart-relative texture candidates of section 5 against step 2's calls. Keep those with item AUC ≥ 0.75 and a CI lower bound ≥ 0.6. | ~1 Mac-hour | Which quantity to optimise. If none passes, the next round uses pairwise human preference only. |
| 4 | **Own-history objective (the night).** Prefix-rollout fine-tune from 56M: per window, roll out 64-128 own rows (no gradient, 1.4 ms per row), then teacher-force the real continuation given that own prefix. Add a preference term on same-prefix pairs if step 3 yields a score. Control: B1 at equal exposures. Judge by step 3's measure, a chart-relative stay rate and a 12-song human screen, against step 1's floor. | ~5-6 Mac-hours | Whether gradient on own histories changes rollout at all. A null says the lever is decoding or data; a pass says the channel work was premature. |
| 5 | **Forced channel use**, if B3 stays in play. Truncate history to ≤ 8 rows on 50 % of windows, with θ known there. Report the reader norm relative to the first query layer at 2M, and kill below 0.1. | ~1.5 Mac-hours | Whether CE can be made to use a held input at all, with a kill rule that can fire. |
| 6 | **Decode-time guards, not objectives.** Keep `d0-env`-style envelope satisficing on texture statistics as constraints. Add a chart-relative critic only after step 3. | none | Prevents the LN-stripping failure of `d0-phi`. |
| 7 | **Capacity**, only if steps 1-4 point at local organisation: the per-type NLL split of section 4, then at most one hidden-256 run. | 1 h, then 6-8 h | Whether F3 is a capacity floor. |

**Panel fixes alongside:** chart-relative references; item-level statistics with a fixed bootstrap denominator; three seeds; guards recalibrated on long sources.

## Failed and not done

- **Train-dev NLL gap:** not in the mirrored logs (only memory resources are mirrored); it needs the mac's job log.
- **Capacity sweep:** the existing scale runs are 8 minutes each and unusable.
- **Reader-norm reference scale:** not logged, so "near zero" cannot be judged against the layers the readers feed.
- **Per-seed m2 (stay):** the recomputation was wrong (the `stay` field is a count, not a rate) and is omitted. m1, d_nh, d_held and bus4 in section 2 are correct. Adjacency per seed was not recomputed.
- **Pilot limits:** mixture-level features only (no source separation, no pitch salience, no learned front end); one training-free probe; 200 fit_train charts; one seed of chart selection.
