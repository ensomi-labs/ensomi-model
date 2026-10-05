# R2 v1: averageness, condition following and style as a target

Shareable. Started 2026-10-04 by main session `9a69a80d` (Claude, control plane), at the human's request to analyse R2 v1's strengths and weaknesses through Astra (max effort, `--no-hooks`) and to have a fresh Fable subagent judge the findings for overclaim, over-defensiveness and local optima inside a wrong direction ([private, local](private/human-inputs/9a69a80d-ed11-4442-9727-a483f42b589a.md#prompt-1)). Builds on [o-r2-first-look](r2-implementation.md#o-r2-first-look), [d-r2-conditions](r2-implementation.md#d-r2-conditions), [d-star-tiled](r2-implementation.md#d-star-tiled), [s-r2-dpo](r2-implementation.md#s-r2-dpo), [d-final-conditions](r2-ln-design.md#d-final-conditions); feeds the `control`, `proposal` and `rollout` nodes.

<a id="o-human-average"></a>
## Human observations, 2026-10-04

From the human's own viewing of the renders at checkpoint `ckpt-0030716993` (`artifacts/r2-preview/r2-ce-overnight-20261004-0030716993/render/` in the code checkout; natural mode, 4 fit_dev charts × 3 seeds): the model tends to generate "average" patterns at all times; LN share cannot be controlled; whether the star condition takes effect is in doubt. The human framed the rest as intuition and analogy, not method: whether the memory could hold distinct clusters of chart shapes (now, or with more parameters or a staged recipe) that a control embedding pushes generation toward, as a route to style control; how proximity is defined and measured and how that overlaps with the planned DPO; how inference can move away from the global average. The agent has not viewed the renders.

<a id="q-r2-style"></a>
## The question as reframed by the agent (proposed)

> Where, in R2's chain of model state, training signal, condition interface and sampling, is chart-level variation of the corpus lost, and what representation of "style" (a space with a proximity, at a stated time scale) would let a control input select a region of the corpus distribution, as a duty distinct from and compatible with preference optimisation that judges acceptability?

- **Q1 Control efficacy.** Does the conditioned policy follow LN-share and star requests within the headroom the given head rows allow; if not, why? Never measured before this session: the trainer's free-run report is natural mode only.
- **Q2 What "average" is.** Between-chart dispersion collapse, within-chart style drift (style resampled as generation proceeds), or a per-row preference for frequent local patterns. Each has its own measurement and its own fix.
- **Q3 State capacity.** Does a seed from a distinctive chart keep the continuation near it, and for how long; what do the hidden states encode beyond the given head rows.
- **Q4 Proximity and control vector.** Which space, at which time scale, under which criteria (structure in the corpus, stable within a chart, separating charts, mirror and tempo invariant, not star again); how a style input enters training so the model must use it, with a natural setting.
- **Q5 Overlap with DPO.** Acceptability versus where inside the acceptable set; whether sequence DPO narrows variety; when a style preference collapses into conditioning.
- **Q6 Inference.** Guidance between conditioned and natural predictions, contrastive decoding, committing to a style first, selection, temperature; their risks.

<a id="h-r2-shortcut"></a>**Agent hypotheses stated in the briefs as things to test, not assume (proposed).** (1) The star label is close to a function of the given head rows (density), so CE training gives the model little reason to read it, and with heads fixed the reachable star range may be narrow: "difficulty has no effect" may be partly a property of the interface. (2) Training labels are always the source chart's own values, so the model never sees two labels on one skeleton. (3) A CE policy with no chart-level latent must recover style from history; if the state does not hold it beyond the 256-row training windows, style is resampled as generation proceeds, consistent with the LN-share drift within songs.

<a id="r2-analysis-jobs"></a>
## Astra jobs, launched 2026-10-04 01:01 UTC

Three parallel jobs on the mac, `ens astra --rw --effort max --no-hooks`, default tier, each limited to 2 CPU threads beside the overnight CE run (4 threads; 10 cores). Briefs kept with the jobs (`~/ensomi/.sync/cp/jobs/<id>/brief.md`); each carries the shared question, the checked facts, the standing rules for briefs and a 3-hour budget; no tracked-file edits, no R2 training. Outputs under `artifacts/r2-analysis-20261004/` of the code checkout.

| Job | Owns | Output |
| --- | --- | --- |
| `20261004-010145-r2-analysis-A-control` | Q1 (headroom on fixed heads, label redundancy, policy sensitivity, conditioned dose response with an interval switch, ranked causes, guidance probe allowed) | `control/` |
| `20261004-010151-r2-analysis-B-average` | Q2, Q3 (definition of average, dispersion, drift, local mode preference, seed persistence, linear probes, landmark and seed lesions, trend over checkpoints) | `average/` |
| `20261004-010156-r2-analysis-C-direction` | Q4 to Q6, thinking first (restate the question, proximity candidates, control vector, DPO duty, inference ranking, research programme) | `direction/` |

Then: a fresh Fable subagent judges each report, read-only, for overclaim, over-defensiveness and local optima under a wrong direction; the agent records both here.

<a id="s-r2-analysis-astra"></a>
## Astra results, 2026-10-04 (from the jobs' final messages; not yet judged)

All three exited 0 and report no tracked-file edits and no processes left running. Checkpoints used (SHA-256 prefix): 30.72M `88af1521`, 39.49M `6e3a998a`, 43.88M `d677489f`, 48.27M `afa9fab9`, i.e. 19 to 31% of the schedule. Numbers below are the analysts'; the main thread has not checked them against the data, the Fable judgment does.

- **A, control** (`control/report.md`, 2 h 32 min). LN condition is read but under-followed in default sampling: requested-to-realised slope 0.504 ± 0.042, tracking error 0.168 ± 0.010 at 43.88M; tracking improves across checkpoints, so A ranks unfinished learning first. Inference-time guidance at gain 4 passes A's tracking gate on four preview charts (slope 1.037 ± 0.022, error 0.056); explicit count feedback on the committed counters reaches error 0.0007 with accurate interval switches. Acceptability of guided or fed-back charts untested. Star: no effective control at any checkpoint; headroom on fixed head rows is wide (LN share 0 to 1 on all eight skeletons, legal constructions spanning 4.15 to 13.32 star per skeleton, not acceptable-chart bounds); head-time features explain 75.3% ± 1.2 pt of the star label's variance. Causes for star (sparse active conditioning, redundancy with input and history) not separated; capacity limit unproven. 1,152 continuations legal.
- **B, average** (`average/report.md`, about 3 h, with an independent replication). "Average" is checkpoint-dependent and not monotonic: at 30.72M chord and lane variation narrows and LN passage drift exceeds real by 0.044 ± 0.021; at 39.49M generated charts show 2.28 ± 0.41 times real between-chart variance and rarer, not commoner, bigrams; at 48.27M LN bias and excess drift return. Seed persistence brief and inconsistent (511-row prefix helps through rows 128 to 255 at 30.72M, only the first 64 rows at 48.27M; no checkpoint shows benefit at 512 to 1023 rows). Hand vectors carry arrangement information beyond timing history (whole-chart descriptor RMSE down 0.280 ± 0.024; creator identity 64% ± 7.4 against a 20% baseline); this does not show discrete clusters. Ranked: rollout calibration instability, weak retention of an intended style, little useful landmark contribution at 39.49M; capacity not established as the bottleneck. Descriptor definitions in `average/descriptors.md`.
- **C, direction** (`direction/report.md`, 56 min; it read A's and B's directories while they were running, so its integration rests on partial results). Restated question: given fixed head times and requests, what persistent choice selects a coherent region of acceptable arrangements, and why does generation fail to retain it. Proposes reference-relative section targeting (a continuum suffices, clusters optional); proximity candidates: passage descriptor distributions with difficulty-aware distance, learned reference embeddings, prototypes or codes only if stable. Control: a descriptor or reference vector held through an interval, fed to action and release predictions, explicit absent mode, tested by reference swaps on a fixed prefix. DPO for acceptability inside the requested region, with pair construction that preserves equally acceptable alternatives and coverage measured apart from preference gain. Inference ranking: persistent target commitment, proximity-based selection with acceptability checks, modest guidance after condition efficacy, temperature or typical sampling, untargeted contrast last. Its real-chart probe found repeatable chart information but missed its own pre-declared threshold in one cohort. First experiment: same-skeleton candidate support (does a small sample pool already contain coherent acceptable continuations near the source).

Fable judgment started 2026-10-04 04:05 UTC (fresh subagent, read-only except its own file `r2-analysis-fable-judgment.md`).

<a id="a-r2-fable-judgment"></a>
## Fable judgment, 2026-10-04 (fresh subagent, read-only; full text [r2-analysis-fable-judgment](r2-analysis-fable-judgment.md))

It re-read the load-bearing numbers of all three reports from their data files and found them matching, with no plan threshold changed after the fact. Verdicts:

- **A (control).** Sound and the most useful of the three. Its "broad headroom" rests on four-note chords on every row; on skeleton `0f6ab03b` its own sane constructions span about 1.9 to 3.5 star around a 2.66 source. That is one skeleton, so ±0.8 star is an illustration, not a general bound. A is over-defensive in two places. It calls the 1 to 3% of holds under 60 ms produced under guidance and count feedback (real charts: 0%) a "distribution shift"; they are a quality signal. And it never says that star control as posed is ill-posed.
- **B (average).** Thorough, but with the wrong headline. The human rendered 30.72M, and B's 30.72M table is the complaint:
  - chord density and jack rate vary across charts at 0.22× and 0.17× the real variance;
  - every chart uses lane patterns near maximal entropy;
  - LN share is 0.41 against 0.19 in the source charts, and it wanders.

  The one signal present in all four panels is that chord density drifts within a chart about twice as much as in real charts; B does not headline it. Teacher-forced LN forecasts are calibrated (0.191 against 0.190) while forecasts on the model's own histories sit at 0.409: exposure bias with positive feedback. Natural LN share oscillates by a factor of about 2.5 between checkpoints 4 to 9M exposures apart.

  B overclaims that "the state holds chart-wide information": its probe has no prefix-descriptor baseline, and lane state alone gives R² 0.61. B also understates the landmark lesion: removing the landmark readout improves action NLL, so the long memory is not used at 39.49M.
- **C (direction).** The best thinking, with the thinnest evidence by design. Its DPO/style division, its pair-construction rules and the adapter trap are worth keeping. Its main bet, a persistent reference vector z, is premature: the LN track is already a one-dimensional persistent condition, and CE alone does not yet hold it over a song.
- **Frame.** Variation is not lost but uncommitted. At 39.49M the variance of generated charts across random seeds on one skeleton (7.14) is twice the variance across skeletons (3.51). On this reading it is a problem of dynamics (an optimiser still hot at a third of the schedule, plus exposure bias) and of training signal before it is a problem of representation.

<a id="s-r2-main-checks"></a>**Main thread's checks of two code claims, 2026-10-04.**

- `data.py:70-83` draws the scored 256-row window independently of the condition track. The track's 1 to 4 LN intervals of 8 to 64 beats are placed over the whole song, so they often miss the window. A's exposure audit puts active requests on 14.4% (LN) and 24.7% (star) of scored rows.
- `train_dpo.py` `LNShareLabeller` prefers branches nearer a fixed `target` of 0.5. Its start state carries the source's drawn track (`corpus_state`). DPO with this labeller would therefore reward ignoring the condition.

<a id="a-r2-reading"></a>**Agent reading (proposed).** The main thread accepts the judgment's frame over its own reframing ([q-r2-style](#q-r2-style)). What the human saw at 30.72M was measured: narrowed chord and jack variety, maximal pattern entropy, LN runaway. It is not a stable property of the model, because later checkpoints differ. Evidence strength:

- **Strong:** star has no effect (KL 2e-6 nats per row, three checkpoints) and is mostly determined by the given head rows (R² 0.75). LN responds and is improving (slope 0.21, 0.45, 0.50).
- **Moderate:** the landmark memory is unused (one checkpoint), and seed persistence fades by 256 to 512 rows.
- **Weak:** that guidance or count feedback give acceptable charts. No acceptability check was run, and holds under 60 ms appear.
- **None:** clusters of chart shapes exist. No cluster test was run. C's probe finds chart identity at about 3× chance after removing star and density, short of its own feasibility gate.

The memory question has this answer: the path exists, but CE on real histories gives no pressure to use it. More parameters would not add that pressure; an informative persistent condition or an objective on free-run continuations would.

<a id="q-r2-after-judgment"></a>**For the human (proposed, awaiting decision).**

1. Star as an R2 v1 control: drop it, or redefine it as residual star or as quantities R2 decides (chord density, jack and overlap rate, LN share). The judgment and A both point here.
2. Window-aligned condition draws (intervals overlapping the scored window with high probability) in the next CE run, compared at matched exposure. This needs a training run, so it waits until the overnight run ends.
3. End-of-run re-probe: the one-argument scripts `control/scripts/probe.py` and `average/scripts/rerun.py` on the final checkpoint and two intermediate ones (about 1.5 h on the mac). Decision rule from the judgment:
   - natural LN still oscillating by more than 0.1 between adjacent checkpoints: the CE recipe is the problem, and DPO waits;
   - natural LN settled near 0.19 with dose slope ≥ 0.7: go to DPO;
   - slope still about 0.5: do item 2.
4. Before any DPO: fix the labeller to use the start state's own track, and make the first pairs calibration pairs (same skeleton and prefix, winner nearer the source chart's own descriptor trajectory), with the typicality evaluator only as a defect filter.
5. A style vector z only after a one-dimensional condition is held over a song by CE alone.

<a id="r2-recipe-fixes"></a>
## Recipe fixes after R2 v1, 2026-10-05

The human judged that the R2 v1 results expose problems in the problem definition and the training recipe, and asked for the most obvious ones first ([private, local](private/human-inputs/0f5110e0-66a5-4e0c-97c0-cf4c78fcca9d.md#prompt-1)).

<a id="d-dpo-synthetic-off"></a>**Decision (human, 2026-10-05).** The DPO trainer's synthetic labellers, including the LN-share rule with a fixed target of 0.5, are an example of the mechanism and do not enter real training. They are removed from the real path and the code says why. DPO waits for real preference examples. This settles item 4 of [q-r2-after-judgment](#q-r2-after-judgment) one step further: not a fixed labeller but no synthetic labeller at all.

<a id="q-r2-ranges"></a>**Question (human, 2026-10-05): are the condition range, the memory range and the loss range misplaced relative to each other?** Check first. Known before the check: condition intervals are drawn over the whole song independently of the 256-row scored window ([s-r2-main-checks](#s-r2-main-checks)), and the landmark memory is unused at 39.49M ([a-r2-fable-judgment](#a-r2-fable-judgment)). Agent hypothesis, to test: the long memory may see little or no prefix in training windows, which alone would explain "unused". A fresh worker removes the synthetic labellers and audits the three ranges in training, fit_dev evaluation and generation, without changing the range code.

<a id="s-r2-run-stopped"></a>**The CE run stopped early (checked 2026-10-05 in `events.jsonl`).** `r2-ce-overnight-20261004` ended at 121,732,521 of 157,966,617 exposures (77% of the cosine schedule) on 2026-10-04 11:35 UTC: repeated `resource_limit` events (swap growth over 1 GiB) and then the supervisor's restart limit. The last checkpoints are `-safe` resumes; lr was not at its floor. The human treats training as finished.

<a id="r2-final-probe"></a>**Final-checkpoint probes, launched 2026-10-05 04:33 UTC** at the human's request ([private, local](private/human-inputs/0f5110e0-66a5-4e0c-97c0-cf4c78fcca9d.md#prompt-2)). Astra job `20261005-043328-r2-final-probe` (`--rw --no-hooks`, effort xhigh, default tier; brief kept with the job). It picks the checkpoint by the recorded rule (earliest within two SE of the best fit_dev CE), adds the last and two late intermediates, re-runs A's and B's frozen probes from the run's frozen code copy, checks reproduction on one earlier checkpoint first, and applies the decision rule of [q-r2-after-judgment](#q-r2-after-judgment) item 3 unchanged. No tracked-file edits (the range-audit worker is editing R2 code on the control plane). Output `artifacts/r2-analysis-20261005-final/` of the code checkout.
