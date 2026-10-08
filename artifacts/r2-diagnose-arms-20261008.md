# Diagnosis round after bake-off round 1: the trained arms and the evaluation (Opus, 2026-10-08)

Report of the fresh Opus subagent "opus-arms", 01:34-02:20 UTC, one of three analysts after round 1 ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1)). Brief: `~/ensomi/.sync/cp/scratch/r2-diagnose/common.md` plus `brief-opus-arms.md`. The main thread saved its returned text here unchanged in content, and checked the loop-gain numbers against `loop/loop_gain.json`. Mac budget respected: at most 4 threads.

Tags: **[M]** measured, **[I]** inferred, **[P]** proposed.

## Bottom line

1. **[M] Most of round 1's ranking is checkpoint luck.**
   - On the same 48-chart subset, phase-N checkpoints 48/52/56/60/64M have G = 0.78 / 1.55 / 1.00 / 1.26 / 1.15 against 56M: SD 0.29, with 56M second-best of five.
   - B1 at 4M/8M/12M is 0.94 / 1.11 / 1.20, and B2 1.22. Both sit inside the phase-N spread; the four non-selected checkpoints average 1.18.
   - "B1 worse than B0" is mostly regression to the mean from a checkpoint selected on LN guards.
2. **[M] The per-step model is calibrated and copies its own recent LN level almost one to one.**
   - Under teacher forcing, B1's marginals differ from B0's by only 0.2-4 %. Free-running, `bus4` in bands 2-3 differs by +59-70 %.
   - The predicted LN-head rate follows the previous 64 rows' rate with slope 0.85-0.97, on real and on own histories. Real charts have the same persistence (0.82-0.94).
   - Nothing pulls the level back towards a chart identity, so the free-run LN level integrates sampling noise, and tiny weight changes become large regime changes.
3. **[M] The θ and anchor readers carry almost nothing under teacher forcing.**
   - Zeroing either changes NLL by 0.0002-0.002 nats per decision, at any history length.
   - A ±1 within-band SD shift of θ moves the per-row marginals by about 1-3 %, with or without history.
   - [I] Teacher-forced CE gives them almost no gradient, because real histories already agree with θ.
4. **[M] No train/inference skew.** Along the real trajectory, sampler-path and training-path log-probabilities agree to ≤ 1.5e-5, and the anchor features are bit-identical.

## Jobs, scripts, outputs

Scripts are in `~/ensomi/.sync/cp/scratch/r2-diagnose/opus-arms/`. Mac outputs are in `artifacts/r2-diagnose-20261008/opus-arms/`, small files mirrored.

| Job id | Purpose | Output |
|---|---|---|
| 20261008-013736-oa-ckpt-panel | 48M/52M/60M/64M and B1 4M/8M on 48 charts (12 per band), BOS seed 954 + prefix | `ckpt/<name>/evaluation.json` |
| 20261008-015959-oa-swa-panel | Uniform weight average of 48-64M on the same panel | `ckpt/swa/` |
| 20261008-014652-oa-tf-probe | Teacher-forced marginals, θ response, reader ablation and norms, skew check | `tf/` |
| 20261008-015527-oa-loop-gain | History-following slope on real and own states | `loop/loop_gain.json` |
| 20261008-015606-oa-train-hd032 | Pilot: B3 + 2M exposures, history dropout to 0-32 rows on 50 % of windows | `train-hd032/` |
| 20261008-021438-oa-probe-hd032 | `theta_probe` on the pilot | `probe-hd032/probe.md` |
| 20261008-020912-oa-theta-tf-b3, -021439-oa-theta-tf-hd | Teacher-forced θ response for B3, B3-4M, pilot 1M and 2M | `theta-tf-*/` |
| control plane | Fixed-scale intervals, seed noise, guard calibration (`eval_noise.py`); checkpoint scoring (`ckpt_analysis.py`) | scratch JSON |

**Failed paths.** The subset G first came back null, because band-2 m5 (lnlen) is undefined on 12 charts; the defined terms are scored instead. A stray `tf-smoke/` directory was left behind.

## (b) Learning rate at 56M [M]

- Phase N ran cosine 3e-4 → 3e-5 over 64M: lr 6.96e-5 at 48M, 4.03e-5 at 56M, 3.26e-5 at 60M.
- B1-B3 re-warm to 1e-4, 2.5× the lr at 56M, which is phase N's lr at about 42M. B1 spends 8.8M of its 12M exposures above 4.03e-5.
- Natural-manifest NLL: 56M 2.0748; B1-4M/8M/12M 2.0882 / 2.0890 / 2.0811; B2 2.0828; B3 2.0829; phase-N 48M / 64M 2.0700 / 2.0837.

## Failure 1: B1 worse than B0 (G 1.41)

| Rank | Hypothesis | Prediction | Evidence | Tuning |
|---|---|---|---|---|
| 1 | **Checkpoint variance / winner's curse** | Neighbours of 56M as bad as B1; B1 inside their spread | [M] Phase-N G 0.78 / 1.55 / 1.00 / 1.26 / 1.15; B1 0.94 → 1.11 → 1.20. Share of runs in an LN-heavy regime (bands 2-3, `bus4` > source + 0.05): 48M 8 %, 56M 21 %, 52M 38 %, 60M 48 %, 64M 23 %; B1 23 / 38 / 31 % (binomial SE about 6 %). Phase N's own free-run panels: LN share minus source +0.046 / +0.063 / +0.099 at 48M / 56M / 64M | Never compare against one selected checkpoint |
| 2 | **Rollout amplification of tiny weight changes** (the mechanism behind 1) | Teacher-forced marginals barely differ while free-run statistics differ a lot | [M] Teacher-forced B1 − B0 on 48 real windows: LN heads +0.2 to +2.6 %, `bus4` +2 to +4 %, release given held ≤ +1.5 %; phase-N neighbours differ by the same few percent. Free-run `bus4` in bands 2-3: +59-70 %. History-following slope of the LN rate per 64 rows: 0.85-0.97 for B0, 48M, 64M, B1, B3, on real and own states, with model differences ≤ 0.03. [I] Near a unit root, a 2 % change in intercept or a 0.02 change in slope moves the stationary mean and variance by tens of percent | A decode-time feedback controller |
| 3 | **Recipe kick from the lr re-warm** | B1-4M (peak lr) worst, then recovery | [M] B1-4M is not worst (23 %, G 0.94); B1-8M is (38 %, 1.11). 60M and 64M show the same shift with no re-warm. The re-warm costs +0.006 to +0.014 NLL. [I] It moves the weights; where they land is the typical checkpoint distribution | Lower re-warm (≤ 4e-5) or continue the cosine |

Not supported [M]: SWA of 48-64M gives G 0.98 and a 25 % LN-heavy share, the same as an average checkpoint. Weight averaging does not remove the regime luck.

## Failure 2: B3 barely follows θ

| Rank | Hypothesis | Evidence | Tuning |
|---|---|---|---|
| 1 | **Under teacher forcing θ adds almost nothing beyond history and anchor, so CE barely trains the reader** | [M] Removing θ costs 0.00018 nats per decision with full history, 0.0009 at 64 rows, 0.0018 at 0 temporal rows; removing the history itself costs 0.035. θ-reader norm 0 → 0.27 → 0.36 → 0.40 → 0.41 at 0/4/8/12/13.3M, against 17.4 for `exact.0` | Train where θ disagrees with the history, or use θ at decode time |
| 2 | **The per-row effect is small and the loop integrates it slowly; 256 probe rows are too short** | [M] Teacher-forced per-row response to +1 SD: held → LN heads +0.0044 per row (about 1.3 %); nh → heads +0.0038 per row (about 2-3 % of the chart-level SD). Same at full, 64 and 0 history. [I] With b ≈ 0.9-0.95 the steady-state gain is 10-20×, reached only after 600-1,300 rows; the 256-row prediction of about 0.1-0.25 matches the measured 0.05-0.23 | Measure θ following over 1,000+ rows |
| 3 | **The anchor duplicates θ** | [M] Each alone costs ≤ 0.002 nats; the joint ablation was not run | A θ-only arm, if θ is kept |

Ruled out [M]:
- **Noise and dropout:** the noise SD is 0.24-0.57 of the within-band SD, an attenuation of at most about 0.8.
- **The donor prior misses the true θ** by an RMSE of 0.97-1.33 band-SD for held and 0.50-1.70 for nh, close to a random chart of the band. Prefix-mode θ misses by only 0.15-0.34.

**Training pilot [M]:** B3 + 2M exposures, history dropout to 0-32 rows on 50 % of windows, lr 1e-4 → 3e-5.

| Measure | B3 final | Pilot |
|---|---:|---:|
| θ-reader norm | 0.41 | 0.45 |
| Teacher-forced held → LN response | 0.0043 | 0.0049 (+12 %) |
| Teacher-forced nh → nh response | 0.0038 | 0.0047 (+24 %) |
| Probe slope, own prefix, held / nh | 0.052 / 0.027 | 0.145 / 0.132 |
| Probe slope, real prefix, held / nh | 0.233 / −0.022 | 0.123 / 0.144 |
| Natural-manifest NLL | 2.0829 | 2.0960 |

- The probe slopes come from 6 charts per band and one seed, so they are noisy.
- [I] θ use can be trained, but slowly at this pressure. Reaching slope ~1 needs roughly 20-40× the current per-row response.

## Failure 3: B2's effect is small

1. **Anchor reader unused.**
   - [M] Zeroing it changes NLL by −0.0007 to +0.0018 nats; its norm is 0.42.
   - [M] B2's teacher-forced marginals are within 1-2 % of B1's.
   - [M] Fixed-scale B1 − B2 = 0.12, 90 % CI [−0.05, 0.31]. On the subset, B2 1.22 vs B1 1.20.
2. **History dropout too weak.**
   - [I] Truncation to 64-128 rows on 25 % of windows still leaves j + 64-128 visible rows at window position j: a mean of about 224 rows vs 511 in those windows, and full history in the other 75 %.
   - [M] Even at 0 temporal rows, the anchor recovers only 0.0018 of the 0.035 nats the history is worth.
3. **The self-anchor is the wrong restoring signal.** [I] Cumulative statistics of the model's own output record the drift, so even if used they anchor to the drifted level.

Training-process noise is larger than the B1 − B2 gap. Tuning: drop B2. Any identity input needs training that creates disagreement.

## Failure 4: the evaluation

| Problem | Evidence | Fix |
|---|---|---|
| Bootstrap intervals not interpretable | [M] The cause is the denominator \|m_B0 − m_src\| re-estimated per resample. Held at its full-sample value, the CI SDs are 0.08-0.17. A same-model null (B0 seed 954 vs 955, BOS) gives −0.02 [−0.24, 0.30] | Fixed-scale intervals (`eval_noise.py::fixed_scale`) |
| Survive the fixed-scale 90 % CI | [M] B0 − B1 −0.41 [−0.55, −0.27]; B0 − B2 −0.29 [−0.48, −0.10]; B1 − B3 0.39 [0.26, 0.53]; B2 − B3 0.27 [0.09, 0.44] | |
| Do not survive | [M] B0 − B3 −0.02 [−0.17, 0.12]; B0 − oracle 0.10 [−0.08, 0.33]; B0 − unknown −0.18 [−0.37, 0.05]; B0 − `d0-env` 0.13 [−0.13, 0.40]; B0 − `d0-phi` 0.09 [−0.12, 0.40]; B1 − B2 [−0.05, 0.31]; oracle − B3 [−0.35, 0.05] | |
| Intervals cover chart and seed noise only | [M] Checkpoint SD of G ≈ 0.29 on 48 charts, about 0.24 after removing seed noise. B0 seeds 955/956 against 954 give G 1.02 / 0.93 on 99 charts and 1.24 / 0.87 on 48 | A multi-checkpoint baseline |
| "No head-lock ≥ 30" fails for every system | [M] Sources fail it too: 2 of 99 charts, 7 of 396 runs. B0 11, B1 15, B2 15, B3 11, out of 396 | A rate against the source, e.g. ≤ 1.5× |
| Tail-exit guard measures level, not tail | [M] B0's whole-run degenerate rate in band 2 is already 2.3× the source's (0.189 vs 0.081). The tail/whole ratio is 1.11 for B0 vs 1.33 for sources: no late collapse | Compare tail/whole ratios |
| G heavy-tailed, terms differ between systems | [M] Small B0 gaps as denominators (`d0-phi` band-5 m2 −27.2, `d0-env` 12.6). b3-oracle/unknown drop different terms. On identical runs (subset): oracle 0.77, prior 0.94, unknown 1.07 | Normalise by the source chart-bootstrap SD, not B0's gap |

## How much of round 1 is noise

- **[M] Chart and seed sampling:** with a fixed scale, B1 and B2 are worse than B0, and B3 is better than B1 and B2. Nothing else is resolved: not B3 vs B0, not the θ modes, not `d0` vs B0.
- **[M] Training process:** checkpoints 4M apart at low lr span G 0.78-1.55, and the LN-heavy run share spans 8-48 %.
  - [I] A single-checkpoint arm difference must exceed about 0.7 (2·√2·0.24) to be real. No round-1 difference does; the largest is B1 − B0 at 0.41.
  - B0 itself was selected on LN statistics and is about 0.2 G below its neighbours.
- **[I] Paired comparisons:** `d0` is paired on 56M's weights, so training noise cancels, but its CI still covers 0. B3's position, about 1 checkpoint-SD better than a typical checkpoint, is a lead only.
- **What the predictions neglected:** the selection effect, single-checkpoint training noise, CE giving identity channels no gradient, and run-level regime luck. For example, B0's band-2 `bus4` mean over 25 charts is 0.048 / 0.088 / 0.047 across seeds.
- **Capacity vs kind:** [M] teacher-forced calibration is within about 5 % of the data, and history persistence matches the data. [I] Local modelling is not the bottleneck, and more capacity will not stop the drift. What is missing is a chart-level state not inferred from the model's own output.

## Top three tuning actions [P]

1. **Rank arms against a checkpoint distribution.**
   - Score each arm at ≥ 3 checkpoints, with ≥ 2 seeds plus the prefix run, on the 48-chart panel.
   - Report fixed-scale CIs and the LN-regime share. Use phase N 48-64M as the reference.
   - Cost: about 3 min of mac per checkpoint at 3 threads (576 runs in about 19 min measured).
   - It fixes the evaluation, not the charts.
2. **A decode-time proportional controller on LN level and `bus4`.**
   - Bias the LN-head logits (and the busy-4 masks) by −κ·(running 64-row level − target). The target comes from the band, the prefix or prior θ.
   - It attacks the measured mechanism: 0.85-0.97 history-following and no restoring force.
   - It beats a constant per-band calibration. [I] A constant bias acts with gain 1/(1 − b) ≈ 10-20, but fixes only means (m6, `bus4`), not persistence or spread, and must be refit per checkpoint.
   - Cost: a scratch logit hook on the action log-probs (`head_mask_bias` covers only head masks) and a sweep of 4 κ × 48 charts, about 20 min.
3. **If identity must live in the network, train where history and θ disagree.**
   - For example, a corrupted, other-chart or own-generated history segment with the real target and the real θ.
   - Current CE gives θ ≤ 0.002 nats, and short-history pressure raised its use only 12-24 % in 2M.
   - Cost: ≥ 10M exposures (about 1.5 h). Untested.
   - A cheaper first step: feed θ to the controller as its target.

## Not done

- the joint anchor + θ ablation;
- the skew check beyond 3 charts × 256 rows;
- a rollout probe over more than 256 rows or more than one seed;
- a CI on the checkpoint spread, which has only 5 points.
