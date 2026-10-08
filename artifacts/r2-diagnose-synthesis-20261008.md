# Why round 1 came out as it did, and how to tune the system (synthesis, 2026-10-08)

Main-thread synthesis of three fresh analysts, asked by the human after round 1 ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1); request [private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-8)):
- [arms and evaluation (Opus)](r2-diagnose-arms-20261008.md);
- [decode-time selection (Opus)](r2-diagnose-decode-20261008.md);
- [the bigger picture (Fable)](r2-diagnose-fable-20261008.md).

Tags: [M] measured by an analyst (see its report), [I] inferred, [P] proposed. The plan in the last section is a proposal awaiting the human.

<a id="s-diagnose-answer"></a>
## The answer in short

- **Round 1 measured mostly noise.**
  - Neighbouring phase-N checkpoints differ by an SD of 0.24-0.29 in G, and B0 (56M) was second-best of five on the very statistics it was selected on.
  - A real single-checkpoint difference would need about 0.7; the largest observed was 0.41 [M].
- **Under that noise there is one mechanism, measured directly.**
  - The model copies its own recent LN and density level almost one to one: slope 0.85-0.97 per 64 rows, the same persistence real charts have [M].
  - In a real chart that persistence holds the level the mapper chose for the chart. The model has no such chart-level choice, so free-running the level drifts like a random walk.
  - Which regime a run lands in is then luck, per seed and per checkpoint. Weight changes too small to see in teacher-forced marginals (0.2-4 %) move free-run `bus4` by 59-70 % [M].
- **Inputs that should carry the chart level go unused.** Under teacher forcing the real history already reveals the level: θ is worth 0.0002 nats per decision, and the readers stay at 2 % of the main layer's weight norm [M].
- **Decoding can hold the level, if it is given a target.** A best-of-4-per-block search toward the real chart's statistics gets within-band identity r 0.94 (plain 0.40) at ≤ 0.64 nats per block [M]. So the right level is in the proposals.
- **The local texture the human complains about is not in the proposals.**
  - The texture: mechanical same-lane jacks, repeated patterns, hand balance.
  - Even that oracle closes only 17 % of the adjacency gap to real charts [M]. The gap is there from the first third of the song [M, earlier], and accumulation adds to it (X2).
  - None of round 1's arms targeted it.
- **So collapse is two problems with different fixes** [I]:
  1. **Level and identity drift.** Its mechanism is known, and it should be fixable at decode time with an external chart-level target. There is a training route as well.
  2. **Local texture.** Its cause is not yet separated. The candidates are diffuse per-step transitions, capacity, exposure to its own errors, and musical grounding. The last looks small for LN and jack placement with simple audio features [M].

## Each failure: hypotheses, verdict, tuning

| Failure | Most likely (verdict) | Second | Third | Tuning |
| --- | --- | --- | --- | --- |
| **B1 worse than B0** | Winner's curse on a selected B0, plus checkpoint variance. **Supported [M]:** phase-N G 0.78-1.55; B1's checkpoints inside that spread | Rollout amplification near a unit root, the mechanism of the first. **Supported [M]** | lr re-warm to 2.5× the lr at 56M. **Contributes, not needed [M]:** 60M and 64M shift without it | Never judge against one selected checkpoint; use a checkpoint distribution and a fixed-scale CI |
| **B3 ignores θ** | CE gives θ no gradient because history already agrees with it. **Supported [M]:** 0.0002 nats; reader norm 0.41 vs 17.4 | Per-row effect is small and the 256-row probe too short (gain 10-20× after 600-1,300 rows). **Consistent [I]** | Donor θ ≈ a random chart of the band (RMSE about 1 band-SD). **True [M]**, so prior θ could not help even if followed | Use θ as a decode-time target; or train where history and θ disagree. Stronger dropout raised use only 12-24 % in 2M [M] |
| **B2 small** | Anchor unused (0.0018 nats at most) [M] | Dropout too weak (about 224 visible rows on average in truncated windows) [I] | A self-anchor records the drift and would anchor to it [I] | Drop B2 |
| **`d0-phi` strips LN** | Self-anchor lock-in: its target follows the chart's own past (slope 0.948), freezing the LN-poor start. **Supported [M]** | Regression fixed point from between-chart confounds [M] | Pooled Gaussian on a skewed statistic [M] | Score distance to an external target, two-sided, with within-chart scales; satisficing pick |
| **`d0-env` inactive** | It kept the first candidate in 90-98 % of blocks [M] | | | Same as above; check selection activity in a smoke |
| **Lowest G, no visibly better charts** | G noise: plain seed vs plain seed gives 0.93-1.19 [M] | G rewards blandness and the statistics selected on [M] | G does not see what the human sees [M, X0] | G as a guard only; a human paired screen decides |
| **Evaluation broken** | Per-resample denominator in the CI [M] | Guards calibrated on short charts; sources fail them too [M] | Band-median references penalise legitimate multimodality [I] | Normalise by the source bootstrap SD; rate-vs-source guards; chart-relative references |
| **Two main-thread misreadings** | Signed ratio on a 0.007 denominator; a pooled B0 row compared with single-seed runs [M] | | | Read only matched rows; never read a ratio without its denominator ([c-bakeoff-r1-readings](r2-bakeoff-night-20261007.md#c-bakeoff-r1-readings)) |

## The human's questions

**Are we directionally right?** Partly [I].
- **Right:**
  - whole-song free-running evaluation;
  - the chart-level identity as the missing piece for level drift;
  - decode-time steering;
  - comparing complete systems.
- **Wrong:**
  - assuming teacher-forced CE would make the model use identity inputs. The precedents were there: R1's all-rows GRU and v1's landmarks went unused;
  - aiming at identity lost past 511 rows, when the human's complaint is early, local texture;
  - deciding on unvalidated aggregates over single checkpoints.

**What the estimates did not cover:**
- the noise floor (checkpoint SD of G 0.24-0.29; seed G 0.87-1.24);
- the selection effect on B0;
- that the level process is near a unit root, which explains the checkpoint flips;
- identifiability under teacher forcing;
- that the donor prior is close to a random band draw;
- that the human's complaint is mostly texture.

The 2026-10-02 failure list already had "noise floor before criteria" and "two training seeds" ([fm-margin-below-noise](agent-failure-modes.md#fm-margin-below-noise)). They were not applied.

**How much is network capacity or kind?**
- **For drift: neither** [I from M].
  - The model is calibrated per step, and its history-following matches real charts.
  - Any autoregressive family trained by CE on real histories will learn to read the level from history, so a family change does not add a chart identity.
- **For texture: open.**
  - Teacher-forced marginals are calibrated, but sampled transitions carry less structure than real ones from the start.
  - That is what a too-diffuse transition distribution would produce, whether from limited capacity or from the T = 1 tail.
  - Evidence from the R1 lineage: a 35M teacher reached NLL 1.61 against 1.74 for the 3.1M model. That is a different setup, so it does not transfer directly.
  - Two cheap tests separate these: the per-decision-type NLL split, and sampling sharpness with the level held by a controller. Both are in the plan below.

**If our 7 measures fail, could others be wrong too?** Yes.
- Unvalidated proxies: stay rate, the envelope exits, β_cum, the adjacency gap, the LN measures, the guards and G.
- NLL is valid for one thing: whether an input is used.
- What remains solid is model-internal facts, not quality: the oracle identity correlations, the loop gain, the teacher-forced ablations, the checkpoint spread.
- The human's whole-song calls are the only quality ground truth, and there are 12 of them.

<a id="p-diagnose-plan"></a>
## Proposed plan (for the human's decision; nothing launched)

Every step answers a question whatever its outcome, and the trained-arm comparisons stop.

| Step | What | Cost | What it answers |
| --- | --- | --- | --- |
| 1 | **Decode-time level controller.** Add a logit bias toward an external target on LN heads and on the chord/all-busy masks: −κ·(running 64-row level − target). The target is θ drawn from the donor prior at song start, or prefix θ when continuing. Sweep κ. Evaluate at three checkpoints (48M, 56M, 64M) × 2 seeds on the 48-chart panel | ~1.5 Mac-h, no training | Does an external restoring force remove the regime luck? Pass: the LN-heavy share and band offsets are at the source's across all three checkpoints, the between-chart spread is ≥ 0.8 of the source's, and no LN is removed. A fail means the level is not controllable at decode time and needs training |
| 2 | **Texture probe**, with the controller on: T ∈ {1.0, 0.9, 0.8} and top-p 0.95. Measure adjacency, same-lane jack runs, row repetition and hand balance against sources. Add the per-decision-type NLL split at 32M-64M | ~1 Mac-h | Is texture fixable by sharper sampling (diffuse transitions), or does it need training or capacity? |
| 3 | **Fix the evaluation:** fixed-scale CIs, normalisation by the source bootstrap SD, a checkpoint-distribution baseline, rate-vs-source guards, chart-relative references | minutes | Makes every later comparison readable |
| 4 | **Human screen, after 1-3:** 6 songs × (plain 56M, controller, controller + best sampling) + 2 real charts; a whole-song call and a ranking per song | ~45 min of the human's time | Whether the level fix is what the human hears, and whether texture remains. The calls become the first calibration set for measures |
| 5 | **Next training night, only after 4**, designed with the human and judged against a checkpoint-distribution baseline: texture training (capacity, or an own-history objective with labels valid under the generated history), or disagreement training for θ if step 1 failed | 1 Mac-night | |

**Stop:**
- CE fine-tune arms judged by G on single checkpoints;
- checkpoint selection by guards;
- self-anchored scores;
- band-median references;
- overnight decisions taken by agents alone.
