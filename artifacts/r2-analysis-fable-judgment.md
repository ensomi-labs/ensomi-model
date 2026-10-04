# R2 v1 analysis, Fable judgment (2026-10-04)

Judge of the three Astra reports under `ensomi-model/artifacts/r2-analysis-20261004/` (A `control/`, B `average/`, C `direction/`). Read-only; nothing re-run. Checkpoints judged: 30.72M, 39.49M, 43.88M (A), 48.27M (B); the run was at 57.0M of 158M (36%) when I looked (`r2-runs/r2-ce-overnight-20261004/run.json`).

## 1. What I checked

Every number below was read from the analysts' data files, not only their prose.

- **A**: `control/summary.json` (LN slope 0.213/0.449/0.504, MAE 0.294/0.229/0.168, star slope ≈ 0, switch DiD 0.045/0.206/0.226 per checkpoint; per-arm realised values), `extensions-summary.json` (guidance 43.88M gain 2/4 slope 0.902/1.037, MAE 0.080/0.056; 39.49M gain 4 slope 0.488; feedback slope 0.9989), `exposure.json` (active LN/star requests on 14.4 % / 24.7 % of scored rows), `star-prediction-summary.json` (R² 0.746/0.753), `headroom.json` (chart 0f6ab03b: real 2.66, single taps 1.87, jacks 3.00, two-chords 3.54, all-LN 3.14, four-chords 6.87), `inference-statistic-shifts*.json` (holds ≤60 ms: real 0.000, feedback @0.3 2.83 %, guidance w4 @0.3 1.40 % at 39.49M, gain 4 @0.9 2.01 % at 43.88M), `dose-response.png`.
- **B**: `average/tables.md` and `summary.json` (variance ratios 0.306/1.152/3.313 original, 2.280 ± 0.405 M1R; 30.72M contractions heads_per_row 0.225 ± 0.117, lane_overlap 0.165 ± 0.104; within-chart fractions; M1H shift 0.217 ± 0.050 then 0.012 ± 0.050; probe R² table; landmark lesion ΔNLL −0.001; creator 0.64 ± 0.07), `figures/average_decomposition.png`.
- **C**: `direction/probe/summary.json` (structural-residual ratios 0.789/0.812/0.762; retrieval 31.2/29.2/22.9 % vs 8.3 %; `preliminary_feasibility: false`); `persistence_update.md` agrees with B's tables.
- **Code** (branch `r2/train`): `conditions.py:15–91` (intervals and dropout), `data.py:70–83` (window drawn independently of the track), `features.py:264–304` (the LN frame carries committed `lns/heads` and `remaining`; the star frame carries no committed-difficulty quantity), `model.py:59–69,154–163,194–212` (FiLM, landmarks, full-prefix encoding), `train_dpo.py:268–282,666–671` (LN labeller uses a global `target`, ignoring the start state's own track).
- All reported numbers I checked match their files. No threshold was changed after the fact in any plan; post-hoc items are labelled as such in all three.

## 2. Report A (control)

**Sound and the most decision-useful of the three.** Headroom → teacher-forced sensitivity → dose response → cause audit → inference probes, all pre-registered, trend paired across three checkpoints.

Overclaim, by emphasis rather than by number:
- "Legal headroom is broad" (§1). The witnessed 4–13-star spans come from four-note chords on every row. From A's own constructions, the range any sane chart on skeleton 0f6ab03b can reach is about 1.9–3.5 stars around a 2.66 source; ±1-star requests sit at the edge of that, ±2 outside it. A says "acceptable headroom remains open"; the honest headline is "acceptable star headroom on a fixed skeleton is roughly ±0.8 star".
- "LN control is possible without a larger model or retraining" rests on four short skeletons (448–588 rows), one checkpoint, after failing at the previous one. A labels this correctly in the body; the bold summary reads stronger.

Over-defensive:
- Short holds. Real charts have 0.000 holds ≤60 ms; feedback at request 0.3 produces 2.8 %, guidance gain 4 at 0.9 produces 2.0 %. A calls these "distribution shifts, not established quality regressions". A defect class absent from the corpus appearing at 1–3 % is a quality signal and should be named as such: both methods partly satisfy the count with fake LNs.
- A never says plainly that star control as posed is ill-posed. Its own evidence says so: R² 0.75 of the label from head-row features, ±0.8-star acceptable range, star-value KL 2e-6 nats/row at every checkpoint, flat star lines on every chart in `dose-response.png`, and (my code read) no committed-difficulty counter in the FiLM frame, so star cannot be learned as the quota controller that LN evidently is. A's proposal 4 (residual star) is right but ranked fourth.

Important and verified: the exposure audit (active requests on 14 %/25 % of scored rows, mechanism in `data.py:70–83` + `conditions.py`) and the `train_dpo.py` labeller defect (global target 0.5 while the start state keeps its own drawn track). Both are things the main thread must act on.

## 3. Report B (average)

**Thorough; the right measurements; the wrong headline.**

Over-defensive / mis-headlined:
- The abstract says "it is not evidence that the architecture forces a single global-average chart" and leads with the 39.49M/M1R over-dispersion (2.28×). But the human looked at **30.72M**, and B's own 30.72M table is the human's observation: heads/row SD 0.134 vs real 0.287 (variance ratio 0.225, >2 SE), lane_overlap SD 0.052 vs 0.129 (0.165, >2 SE), mask_entropy 3.47 ± 0.13 vs real 3.23 ± 0.38 (every generated chart near-maximal lane-pattern entropy; real charts differ by restricting their vocabulary), LN share 0.41 vs 0.19 and wandering (drift 1.4×), chord-multiplicity variation 66 % within-chart vs 31 % real. That is "average patterns at all times": same chord density, same jack rate, every mask used, LNs everywhere. B has every number and labels them post-hoc (correctly) but does not say "yes, at the checkpoint you rendered, this is what you saw".
- The one within-chart signal that holds in all four panels, `drift_heads_per_row` higher (0.024 ± 0.011, 0.065 ± 0.019, 0.106 ± 0.017, 0.073 ± 0.017), and the chord-multiplicity within-chart fraction elevated >2 SE at all three original-panel checkpoints, are not headlined. Chord density wanders within a chart about twice as much as in real charts, at every checkpoint. That is the robust form of (b).
- The landmark lesion is the cleanest answer to the human's "memory" question and is stated in the softest possible way. Zeroing the landmark readout *improves* action NLL by 0.0015 ± 0.0003 and contributes nothing to prefix persistence (factorial interaction none). At 39.49M the long-range memory mechanism is not used by the policy.
- M1H (ranked first, correctly) is the most consequential mechanism in all three reports and is not named: teacher-forced LN forecasts are calibrated (0.191 vs source 0.190) while forecasts on the model's own histories sit at 0.409. This is exposure bias with positive feedback (history tokens carry LN counts; held state begets holds). It vanishes at 39.49M and LN is back to 0.49 at 48.27M; A's panel shows natural LN 0.50 → 0.23 → 0.20. The free-run first moment oscillates by a factor 2.5 between checkpoints 4–9M exposures apart.

Overclaim:
- "The state contains chart-wide arrangement information" (Q3). The probe baselines are timing-only. The right null is the descriptors of the real prefix actions themselves; B did not fit it. Its `no_history` arm (exact lane state and clocks only) already reaches R² 0.61 for whole-chart LN share, and real charts rarely change LN share, so hands at R² 0.72 is a running summary of what the history contained, which any sequence model has. It is not evidence of style memory, and the creator probe (64 % on 25 dev song groups; 48 % from lane state alone) has the same confound.

Correct and useful: the 256-row window is a loss window, not a context limit (`model.py:194–212` verified); sampling is native-probability Gumbel, so temperature is not a hidden cause.

## 4. Report C (direction)

**The best thinking; the thinnest evidence, by design.** The restated question ("what persistent choice selects a coherent family of acceptable arrangements conditional on the scaffold, and is it lost in prediction, memory or sampling") is better than the main thread's. The DPO/style division (§4), the pair-construction rules, the `Skeleton.from_chart`/`head_times`-fixed adapter trap (`case.py:196–200,265–272`), and "contrast against yourself is just temperature" are correct and worth keeping.

Overclaim: none of substance. P1 is reported with its failed gate and the shared-musical-content confound; retrieval at 3× chance is called identity, not style.

Over-defensive: C hedges guidance ("not ready remedies") on A's interim results; A's 43.88M repeat later passed. C's mechanism warning stands, and I think it is the explanation: the learned LN condition and its guidance work *because* the frame carries the committed ratio (`features.py:290–294`), i.e. the model learned a quota controller. Star has no such channel, so star guidance does nothing (A §5: slope 0.03 ± 0.03).

Local optimum: C's E3 (a persistent reference vector z, 2–6 h training pilot) is the big bet and is premature. The LN condition *is* a one-dimensional announced, persistent, interval-scoped z. From CE alone it reaches slope 0.5 at 28 % of the schedule and holds a half-song switch at DiD 0.23 (gate 0.30). Until a 1-D z is held over a song from training alone, a 32-D z adds dimensions, not capability. C half-says this ("the broader style programme should not postpone fixing a simple condition-interface failure") but orders its programme the other way.

## 5. Cross-report: is the frame right?

The main thread's question asks where "chart-level variation of the corpus is lost". B's data say variation is not lost; it is **uncommitted**. At 39.49M the within-skeleton seed variance (7.14) is twice the variance of seed means across skeletons (3.51); in real data the skeleton determines one chart. The model's output varies more by random seed than by which song it is writing, and chord density wanders inside a chart. The human's "average" is the perceptual trace of that: no chart has an identity. At 30.72M it additionally had a literal contraction in chord density and jack rate, plus the LN runaway.

Three consequences for direction:

1. **It is a dynamics problem before it is a representation problem.** Teacher-forced predictions are calibrated; free-runs drift and oscillate across checkpoints. Nothing proposed under Q4 (descriptor spaces, z, codes) addresses exposure bias. A larger model or longer loss window does not either (B is right that the window is not the limit). What does: finishing the schedule (lr 1e-3 cosine at 36 % is still a hot optimiser), and then an objective that sees self-generated histories. Sequence DPO from a fixed start state is exactly such an objective, so the planned DPO is not the enemy of coherence *if* its reward is calibration to the source chart's own descriptor trajectory rather than typicality. With the typicality evaluator as the only judge, C is right that it will tilt toward the common and make "average" worse.

2. **Star, as built, is ill-posed and should be dropped or redefined.** Fixed head rows determine 75 % of the label, the plausible range is ±0.8 star, there is no committed-difficulty channel, and the model shows 1e-6 nats/row sensitivity at every checkpoint. The human's "does difficulty take effect" has the answer "no, and it mostly cannot on this interface". Replace it with quantities R2 actually decides (chord density, jack/overlap rate, LN share) or with residual star (A's proposal 4).

3. **Inference fixes are hiding a training defect, and are the right tool only for count-type attributes.** The exposure audit shows the condition is supervised on one scored row in seven. Aligning intervals with the scored window is a few lines in `data.py` and is the cheapest experiment on the table. Guidance gain 4 and count feedback are legitimate decoding controls for LN share once the learned response is monotone, but they produce 1–3 % sub-60 ms holds (real: 0) and shift star by +0.6–0.8; they change nothing about chart identity or drift. Temperature is contraindicated (rarity is already excessive at 48.27M, +0.47 bits). Candidate selection by proximity to a reference (C's E1) is the only inference route that touches "away from average", and it needs an acceptability screen that does not yet exist.

Checkpoint vs design: none of the three can separate them yet, and all three say so; the oscillation is why. Every design conclusion above rests on a code mechanism, not a checkpoint number.

## 6. Answers for the human

- **Does difficulty take effect?** No. Star-value KL 2e-6 nats/row (A, 16 charts × 96 states, three checkpoints), generation slope ≈ 0 with every chart flat. Evidence strong. And it largely cannot on this interface (R² 0.75 from head rows; ±0.8 plausible range; no counter). LN share does take effect: slope 0.5 and improving at 28 % of training, 1.0 with guidance gain 4, exact with count feedback, at a cost in short fake holds.
- **What did you see?** At 30.72M, B's numbers: every chart with the same chord density and jack rate (variance 0.22× and 0.17× real), every chart using the full lane-pattern vocabulary (mask-entropy SD 0.13 vs 0.38), LN share doubled and wandering. By 39.49M the first two are gone; by 48.27M the model is over-dispersed, jack-heavy and LN-heavy again. The model is not in a stable regime; renders at any one checkpoint are a snapshot.
- **Does memory support clusters now?** No. The landmark memory contributes nothing to prediction (lesion improves NLL) or to prefix persistence at 39.49M; a 511-row real prefix stops helping by 256–512 rows at every checkpoint; the state carries a running summary of the past, not more. **Potentially?** The information path exists (full-prefix encoding, landmarks). What is missing is pressure to use it: CE on real histories with the local 511-row field already predicts the next row, so a long-range summary buys nothing. Parameters will not create that pressure; an informative persistent condition (the LN track today, a reference z later) or an objective on free-run continuations would. Evidence moderate (one checkpoint for lesions; three for persistence).
- **Proximity and its overlap with DPO.** C's division holds: DPO judges acceptability conditional on a target; a style input selects the target; "closer to target" as a preference is adherence and belongs to conditional CE. B's four descriptors are a sound diagnostic space; C's P1 shows passage descriptors carry chart identity at 3× chance after removing star and density, which is enough to measure persistence and shortlist references, not enough to call a cluster. No cluster test was run; do not assume clusters. The DPO prototype's labeller ignores the requested value (`train_dpo.py:268–282`); fix before use.
- **Pushing inference away from the average.** Guidance and count feedback move LN share; nothing tested moves chart identity. Temperature would make things worse. The honest inference experiment is C's E1: oracle selection among natural candidates by proximity to the source's descriptor trajectory, which tells you whether a coherent chart is even in the sample support.

## 7. Next experiments, in order

1. **Re-run A's and B's frozen panels at the end of the run and at two intermediate checkpoints** (both ship one-argument reruns: `control/scripts/probe.py`, `average/scripts/rerun.py`; about 25 min + 1 h on the Mac, two threads). Decide on: natural LN share trajectory, M1H history shift, chord-multiplicity within-chart fraction, LN dose slope, landmark ΔNLL. If natural LN still oscillates > 0.1 between adjacent checkpoints at the end, the CE recipe (schedule, exposure bias) is the problem and DPO waits. If it settles near 0.19 and LN slope ≥ 0.7, go to DPO with calibration pairs. If LN slope stays ≈ 0.5, do (2).
2. **Window-aligned condition draws** (A's proposal 2): draw intervals so they overlap the scored window with high probability; matched exposures, two training seeds, same panel and gates. If LN slope at matched exposure does not move, the sparse-signal hypothesis is dead and counterfactual contrasts (A's 3) or an on-policy objective are next. If it moves, the same recipe is how a style z would be supervised.
3. **Decision, no compute:** drop tiled star as a control for R2 v1, or redefine it as residual star / decided quantities.
4. **Before DPO:** fix the labeller, add the `head_times`-fixed adapter check, and build the first pair set as style-matched calibration pairs (same skeleton, same prefix, winner = closer to the source chart's own descriptor trajectory), with a typicality judge only as a defect filter. Measure B's drift and A's LN slope after, not only preference accuracy.
5. **Then** C's E3 (persistent reference z), only once (1)–(2) show a 1-D condition held over a song from CE alone.

## 8. What I could not check

- The renders themselves (agent reading is not a gate; I also cannot open the SVGs usefully from here). My account of "what you saw" is B's 30.72M statistics on a different 24-chart panel; the four preview charts are in A's panel, not B's.
- Acceptability of any guided, fed-back or natural chart; no evaluator or human gate was applied by anyone.
- Raw per-state tensors on the Mac, bootstrap implementations, receipt hashes; I took the analysts' SEs as computed.
- Checkpoints after 48.27M (57.0M exists now); whether the 30.72M contraction and 48.27M over-dispersion recur. Only the end-of-run rerun answers that.
