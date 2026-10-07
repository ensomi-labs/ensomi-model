# Fable report: pattern collapse in R2 long-range self-generation (formalisation and theory)

Report of the Fable subagent (fresh context) for [r-collapse-agents](r2-collapse-20261007.md#r-collapse-agents), copied verbatim between the markers. It was returned as text.

Main-thread checks:
- `artifacts/r2-collapse-20261007/fable/martingale.json` (mirrored) matches the n48 LN-share variogram and β values quoted in section 3.
- `features.py` confirms the claim that `cum_heads`, `cum_ln`, `cum_repeat` and `cum_held` are read only by `_interval_counts` and `star_proxies` (the FiLM frames), not by `query_features`.

Script: `~/ensomi/.sync/cp/scratch/r2-collapse/fable/martingale.py`. Mac job `20261007-161320-fable-martingale`.

----- BEGIN report -----

# Fable report: pattern collapse in R2 long-range self-generation (formalisation and theory)

Started 16:09 UTC, delivered ~16:30 UTC. One diagnostic run (job `20261007-161320-fable-martingale`, exit 0, 1 min); outputs `artifacts/r2-collapse-20261007/fable/martingale.json` on the mac, script `~/ensomi/.sync/cp/scratch/r2-collapse/fable/martingale.py`. No tracked file touched.

## 1. Formal problem statement

**Objects.** Skeleton S = (head times H, grid, T). Decisions D = (a_0..a_K), a_k one of 625 joint actions plus gap-release times. Corpus law p*(D | S). The data say p* is hierarchical: a chart-level organisation θ (LN level, LN-length style, lane vocabulary, chord-size regime, jack regime, hand use), drawn once per chart with a broad conditional p*(θ | S) (LN share: skeleton explains R² 0.17, so about 83 % of the level is the mapper's choice; the level is "a chart constant plus short local noise", demeaned 64-row ACF 0.37 at 64 rows and about 0 beyond 128), plus sections (local excursions that end on musical structure). So p*(D|S) = ∫ Π_k p*(a_k | s_k, θ) p*(θ|S) dθ with s_k the exact state.

**Model.** q_φ(a_k | h_k) where h_k = (TCN over the last 511 committed rows; exact lane state and ages; k/K, t/T, remaining; beat phase, bpm, meter; densities 1–32 beats ahead; next 16 gaps). Trained by teacher-forced MLE on real windows, so at the optimum q(a_k | h_k) = ∫ p*(a_k | s_k, θ) p*(θ | h_k) dθ: the **posterior predictive of θ given a 511-row window of real history**. Measured: calibrated on real history (LN report §3: the response to the last-64/last-511 share equals the data's; early-song mean P(LN) 0.069–0.072 against 0.0715).

**Generation.** a_k ~ q(· | own h_k): a *windowed posterior-predictive sampler*.

**Why collapse is not a necessary consequence of token CE (the key formal point).** By de Finetti/Pólya: if memory were unbounded and q exactly the posterior predictive, free running from BOS is exactly a draw from the hierarchical model: the early decisions implicitly draw θ, the posterior concentrates, and every running statistic converges almost surely (excursions damp as 1/k). "CE learns the average" is therefore half right: CE learns the mixture predictive, which is "average" only where the history is uninformative about θ. Drift and collapse come from three things the recipe and architecture add:

1. **Finite memory with no sufficient statistic.** The estimate of θ is a function of the last W_eff rows (data: the last 64 rows carry slope 0.42, rows k−511..k−64 almost nothing). The level then obeys E[level_{k+1} | past] ≈ windowed mean: a **windowed Pólya urn**, a bounded martingale with injected variance ≈ Var(a_k)/W_eff² per step. It has no restoring force toward the chart's own level and (martingale convergence) eventually absorbs at an extreme; with gain g = d E[next share]/d(history share) ≠ 1 it moves deterministically (g < 1: decay toward the zero mode, 48M; g > 1: creep up, 64M; measured gain maps 0.84 and 0.95/above identity).
2. **Information never present.** Real sections end on musical structure (audio, phrase boundaries, mapper intent). The model sees only the head skeleton 32 beats ahead. On its own output a chance excursion looks like a section and nothing ends it (LN report: "no section structure ends them").
3. **Error compounding (Ross & Bagnell 2010/2011).** Per-step error ε under the data distribution gives O(T·ε) to O(T²·ε) rollout divergence; here the error that matters is the gain's deviation from 1, which is second order in NLL (48M and 64M differ by 0.014 nats and drift in opposite directions) but first order in rollout.

**Three regimes of the same mechanism**, which together are "pattern collapse":

- **A. Diffusion** (gain ≈ 1): a chart-level level wanders; excursions last about W_eff; variance grows with lag. LN share today; within-chart chord-density drift 2× real in v1.
- **B. Positive-feedback absorption** (gain > 1 locally): runaway into a degenerate regime: all-LN (v1 0.41 vs 0.19), fixed-lane runs of 30+ rows (R1), quad walls and "full 4 lane" (C0 tilt). The data conditional "a long run predicts continuation" is right for real charts because the mapper ends the run; the model has no ending signal.
- **C. Averaging fixed point**: for high-dimensional θ (lane vocabulary, length style) the posterior from a window of mixture-sampled output never concentrates, so the predictive stays the mixture, the output stays mixture-like, and the history stays uninformative: a self-consistent "average chart". Measured in v1: lane-pattern entropy near maximal in every chart (SD 0.13 vs 0.38), chord density and jack rate across charts at 0.22×/0.17× real variance; today: per-chart median LN length SD 0.63–0.74× real, 1/8-beat-LN chart types missing, seed variance on one skeleton twice the variance across skeletons.

**Operational definition** (per chart, per 64-row block, versus the corpus in the same star band): block statistics Φ = (LN share, median LN length, heads per row, jack rate, lane-mask entropy and vocabulary size, longest same-lane run, 4-lane-held share, repeated-mask loop rate). Three measures:
- **M1 anchoring (regime A):** variogram V(l) = Var over charts of (Φ_{2+l} − mean of blocks 0–1), l = 1..8 blocks; report V_run(8)/V_run(1) against the sources' (flat ≈ 1), and the anchor-reversion coefficient β in ΔΦ_b = a + β(Φ_b − anchor) + β'(Φ_b − Φ_{b−1}).
- **M2 degeneracy (regime B):** rate of blocks beyond the corpus' per-band 99.5th percentile for each Φ component (run length, held share, LN share, quad share), as a ratio to the sources, by position third; and the exit hazard from such a block versus the corpus.
- **M3 commitment (regime C):** across-chart SD of per-chart Φ over source SD (0.8–1.25 good; <0.7 averaging; >1.4 over-dispersion) and within-chart mask entropy against the per-band source distribution.

**LN-level wandering is one instance of this** (regime A, the lowest-dimensional, best-measured one, and the dimension least anchored by the skeleton). It is not the whole: regimes B and C are present in the record (v1 numbers, C0 screen, length styles).

**Target, "playable and stable over a whole song":** per band, on a 100-chart × 3-seed panel: (a) M1: V(8)/V(1) ≤ 1.25 for LN share and heads/row (sources 0.84–0.99); (b) M2 ≤ 1.5× the corpus for every degenerate class in every third of the song, exit hazard ≥ 0.5× the corpus; (c) M3 within 0.8–1.25 for each Φ component; (d) legality and guards (i),(iii),(iv) v2; (e) a position-blind human screen (late thirds against early thirds) finds no difference. "Stable" = (a),(b) hold in the last third as in the first; "playable" = (b),(d),(e).

## 2. Causal map

| Link | Mechanism | For | Against | Strength |
|---|---|---|---|---|
| Teacher-forced per-decision CE | Learns the real-history posterior predictive; nothing constrains behaviour on own histories; the rollout-relevant gain is a near-flat direction of NLL | TF calibration measured (LN report §3–4, v1 0.191 vs 0.190); 48M/64M same NLL ±0.014, opposite drift; R1: capacity and exposure did not remove collapse (s-likelihood-vs-rollout) | Not "CE learns an average" per se: with an anchor the same CE would converge (de Finetti) | strong (mechanism from code + measurement) |
| 256-row loss windows, start rule | Loss window only; `R2Model.window` encodes the full prefix from row 0 (`window_hands`, N = stop−1); decision positions are covered ≈ uniformly | code | — | strong: **not a cause** |
| 64M exposures (≈5 passes over 12.7M fit_train decisions), cosine finished | NLL flat from 40M (2.070→2.084); more exposure moves the gain by second-order amounts only | measured NLL; v1 LN oscillating ×2.5 between checkpoints | — | strong: not the lever |
| Receptive field 511 rows, `memory: none`, **no cumulative prefix statistics in the natural query** | No sufficient statistic for θ; the windowed urn. `Derived.cum_heads/cum_ln/cum_repeat/cum_held` exist but enter only FiLM frames (conditions), not `query_features` | code; data: given the whole-song level the last 512 rows weigh 0.12 vs 0.90; landmarks unused under CE in v1 (lesion improved NLL: no CE pressure to use long memory because the last 64 rows already predict; beyond-RF slope 0.036) | the LN report calls the 511-row RF "not the cause" because the data's beyond-RF slope is small on real history: true for TF prediction, irrelevant for rollout anchoring | strong on mechanism, moderate on magnitude |
| Information never present: musical/phrase structure, mapper intent | Nothing ends an excursion or a run | No audio; only 32-beat lookahead and 16 gaps; real sections last ~64 rows then end | untested whether phrase position (bars mod 4/8) would carry it | moderate (inferred) |
| Row-head factorisation (hand scores + rank-16 coupling, mirror marginalisation) | Full 625 support; mirror symmetry forces a hand-symmetric natural policy and may wash per-chart handedness | — | no evidence | weak; not a priority |
| Sampling: native ancestral, legality mask, no temperature | Not a cause. Prediction: τ<1 slows A (variance ∝ Var(a)) but raises B (sharpening pushes gain >1) and worsens C (toward the mode); τ>1 the reverse | v1: temperature contraindicated | — | moderate (theory) |
| Difficulty tilt (C0) | Adds a bias on a positive-feedback statistic (chord size): drives regime B | screen: "quad walls", "full 4 lane" | — | moderate |

## 3. Diagnostic run (measured)

Anchored-versus-martingale test on the n48/n64 BOS free runs (116 fit_dev charts × 3 seeds = 348 runs per checkpoint) against their 116 sources; 64-row blocks; anchor = mean of blocks 0–1; blocks ≥ 2; cluster-robust SE by chart.

| statistic | group | β_anchor | β_recent | V(1) | V(8) | V(8)/V(1) |
|---|---|---|---|---|---|---|
| LN share | sources | −0.265 ± 0.047 | −0.162 | 0.032 | 0.027 | 0.84 |
| LN share | 48M runs | −0.138 ± 0.019 | −0.177 | 0.033 | 0.056 | **1.70** |
| LN share | 64M runs | −0.123 ± 0.014 | −0.150 | 0.040 | 0.078 | **1.92** |
| heads/row | sources | −0.252 ± 0.036 | −0.097 | 0.068 | 0.067 | 0.99 |
| heads/row | 48M | −0.191 ± 0.021 | −0.129 | 0.053 | 0.074 | 1.38 |
| heads/row | 64M | −0.172 ± 0.018 | −0.186 | 0.052 | 0.073 | 1.40 |
| jack rate | sources | −0.285 ± 0.033 | −0.074 | 0.0102 | 0.0113 | 1.11 |
| jack rate | 48M | −0.249 ± 0.019 | −0.148 | 0.0098 | 0.0102 | 1.04 |
| jack rate | 64M | −0.219 ± 0.023 | −0.219 | 0.0100 | 0.0122 | 1.22 |

Reading: real charts' LN share stays at its early level (flat variogram); the runs' variance about the early level grows roughly linearly with lag (+0.004–0.005 per 64 rows), and the reversion to the chart's own level is half the sources'. Extrapolated, a 2,500-row song reaches an SD of about 0.4 about its early level: the longer the song, the fuller the collapse (testable: drift by song length). Chord density diffuses mildly (it is largely pinned by the given head rows), jack rate not at all. This is exactly the theory's ordering: drift is largest on the dimension least determined by the input (LN share, skeleton R² 0.17). Caveat: β on the sources is −0.27 rather than −1 partly because block sampling noise biases β negative in both groups; the variogram is free of that bias and is the discriminating measure. Inferred, not measured: the W_eff and gain quantities.

## 4. Core questions, ranked

1. Does a chart-level anchor the model can read at every decision remove regime A, whether the anchor is its own cumulative output (self-anchor) or an externally drawn level?
2. Is the missing section-ending information cheap (phrase position from the grid) or does it require audio?
3. Does regime B occur in natural phase-N generation at all, and where (Opus' measurement M2 by position)?
4. Does an anchor leave regime C untouched (prediction: yes), so that per-chart style needs an explicit held latent?
5. Does an on-policy sequence objective on Φ trajectories buy anything beyond the anchor?

## 5. Experiments

**E1. Corpus-only diagnostic: phrase-boundary hazard** (no model; ~5 min Mac, 2 threads).
Question: in real charts, do regime changes (LN-section start/end; block Φ jump > 1 within-chart SD) concentrate at 4/8/16-bar boundaries relative to the first head? Manipulation: none; control: hazard at non-boundary bars. Metric: hazard ratio boundary/non-boundary. Pass ≥ 2 → phrase-position features (bar index mod 4, 8, 16) are the cheap information fix; run E4. Fail < 1.3 → sections are audio-driven; drop the feature idea, rely on anchors and guards.

**E2. Self-anchor fine-tune** (the principal test of the theory).
Question: does giving the natural query cumulative statistics of the committed prefix (cum LN share, cum heads/row, cum jack rate, cum held share, cum lane histogram, log k; 8–10 channels through a zero-initialised bias-free reader, exactly the `ln_level_reader` pattern) make free-run levels converge? Manipulation: warm-start 56M, 12M exposures, same schedule as `ce_v2_n_lnlevel_ft.json`. Control: identical warm-start fine-tune without the channels (isolates the fine-tune). Metric: M1 on the 116 × 3 panel, both checkpoints; NLL on the natural manifest. Pass: V(8)/V(1) ≤ 1.25 for LN share and β_anchor within 0.05 of the sources', NLL rise ≤ 0.005. Fail: ratio ≥ 1.6. Cost: ~1.8 h per arm at 1,820 decisions/s (a 15-min pilot of 1.6M exposures checks the reader moves and NLL does not degrade). Pass → regime A is the finite-memory urn; measure M3 next (predicted unchanged) and go to the style latent. Fail with the model using the channels on real prefixes (slope test) but not on own → distribution shift of own output in the cumulative statistics, i.e. exposure bias proper; go to E5.

**E3. External anchor (the stopped `ln_level` + `ln_length` prior-mode fine-tune), reframed.**
Question: with a drawn whole-song level held fixed, does the level stop diffusing and does the model obey? Control: E2's no-channel arm. Metrics: M1 on LN share in prior mode; obedience slope of realised share on the oracle level (pass ≥ 0.8); history weight (free-lane P(LN) regressed on last-64 share at fixed level; pass ≤ 0.15, the data value is 0.12). Pass/fail as E2 for M1. Cost: as planned (12M, ~2 h). The human's doubt is answered by the theory: an anchor converts the martingale into an anchored process at any length, *if obeyed*; the obedience slope is the thing to measure, not length-dependence. It covers only the conditioned dimensions.

**E4. Phrase-position features** (only if E1 passes): same protocol and metrics as E2, metric M2 added (exit hazard from excursions).

**E5. On-policy objective** (only if E2/E3 fail or regime C blocks acceptability): DPO on the model's own free-run pairs from a shared prefix, winner = the continuation whose block-Φ trajectory is nearer the corpus conditional given the prefix's own Φ (calibration, not typicality; the existing `train_dpo.py`, real pair file). Control: anchor-only model. Metrics: M1–M3, NLL ≤ +0.01, human screen. Cost: pair generation ~30 min (348 full songs at ~200 decisions/s) + 1–2 h training.

**E6. Temperature probe** (Astra): prediction recorded above (τ<1 reduces A, raises B and C). It is a diagnostic of the regime taxonomy, not a remedy.

Not an experiment: more exposure or parameters for the natural model (predicted null; R1 and phase N already show it).

## 6. The control problem

A condition is an anchor set from outside; phase C's interval FiLM is a time-varying anchor. If E2/E3 pass, LN control is the same mechanism as stability and should follow (the v1 LN track already reached slope 0.5 under CE because its frame carries the committed ratio). Regime C is the enemy of style control: a model at the averaging fixed point cannot hold a vocabulary from history, so the five style concepts need an explicit held latent (the "style baseline"), not a decoding trick. Difficulty tilts act on positive-feedback statistics and push into regime B; any tilt must run on an anchored model and be bounded by M2 as a hard guard.

## 7. Stop or do not do

- No more CE exposure, capacity or schedule changes for the natural model as a fix for collapse.
- No temperature, top-p or count-feedback controller as the fix (the latter produces fake short holds).
- No checkpoint selection on mean third-to-third drift alone (guard (v)): 48M has mean drift −0.020 and |Δ| 0.108; use the variogram ratio or mean |last − first| against the sources.
- Do not read a failure of the LN fine-tune as "CE cannot fix drift" without the obedience slope; and do not read its success as fixing regime C.
- No difficulty-measure work until M1–M3 pass (human's rule).
- Do not add hard rule residuals against runs/walls as the remedy (R1's path); acceptable only as a guard while E1–E4 run.

Literature anchors: exposure bias and compounding (Ross & Bagnell 2010; Ross, Gordon & Bagnell DAgger 2011; Ranzato et al. 2016), scheduled sampling and its inconsistency (Bengio et al. 2015; Huszár 2015), professor forcing (Lamb et al. 2016), global latent sequence models and posterior collapse (Bowman et al. 2016; Roberts et al. MusicVAE 2018; Chung et al. VRNN 2015), exchangeability and urn convergence (de Finetti; Pólya urn / Blackwell–MacQueen).

----- END report -----
