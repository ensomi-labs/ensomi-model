# Astra report: pattern collapse in R2 long-range self-generation (code and information paths, probes)

Report of Astra, ens job `20261007-160853-r2-collapse-astra` (GPT-6 Astra, fast tier, `--rw`), for [r-collapse-agents](r2-collapse-20261007.md#r-collapse-agents). The final message (`~/ensomi/.sync/mac/jobs/20261007-160853-r2-collapse-astra/last.md`) is copied verbatim between the markers. Its links point into the mac tree.

Where things are: scripts and outputs on bings-mac in `artifacts/r2-collapse-20261007/astra/` (`information_probe.py/.json`, `temperature_probe.py`, `analyze_temperature.py`, `shuffle_diagnostic.py`, `metrics.json`, plus the generated action arrays). `information_probe.json` is mirrored; the main thread checked its model config and the parameter count (2,404,724).

----- BEGIN report -----

**The audit confirms a train–generation mismatch and missing persistent chart organization, but neither is yet established as the cause of collapse.** The new 56M probe found weaker local pattern dependence than the sources, already present early. It did not reproduce frequent severe repetition or show monotonically worsening organization.

**1. Formal problem statement**

Let \(S\) be the supplied head times, grid and song length; \(a_k\) the joint row action; and \(r_k\) its gap releases. R2 generates

\[
p_\theta(a,r\mid S)=\prod_k p_\theta(a_k\mid I_k)\,
p_\theta(r_k\mid a_k,I_k),
\]

where \(I_k\) contains recent encoded history, exact lane state and selected skeleton features. Training minimizes source-history negative log likelihood; generation recursively changes the histories supplying \(I_k\).

Operationally, distinguish:

- **Degenerate repetition:** excessive identical head-mask runs, fixed-lane attack runs, short loops or quad walls.
- **Loss of organization:** abnormal head-mask transitions, motif recurrence or dependence across rows. High entropy can be a failure too.
- **Organization drift:** chart properties or pattern regimes change more often, farther or for longer than in comparable corpus charts.

Measure these in 128-row windows, stride 64, with longer-window checks; calibrate against sources matched by skeleton rate and evaluation band. Report exceedance frequency, consecutive alarm duration, first onset, recovery and early/middle/late differences. A proposed collapse event is a corpus-tail excursion sustained across three adjacent windows—not an isolated unusual pattern.

LN-level wandering is an instance of **organization drift**. It establishes neither repetitive collapse nor unplayability by itself. The earlier 48M/64M analysis found first–last-third LN-share correlation around 0.61 versus 0.91 in sources; that is relevant evidence, not a complete pattern-quality diagnosis.

“Playable and stable over a whole song” should mean legal output, corpus-compatible distributions of sustained failures, preserved variation across charts, and human acceptance of complete songs including their endings. Corpus rarity alone cannot certify playability.

**2. Causal map and information paths**

The frozen model has **2,404,724 instantiated parameters**.

| Path | What actually reaches the decision |
|---|---|
| History | Two mirrored 100-feature views per row: lane actions, held state, release presence and timing offsets; preceding gap, elapsed time, beat phase, chord size and LN-head count. Both views include all four lanes. |
| Temporal encoder | Eight causal dilation levels; exactly **511 committed head rows** of raw-token dependence. |
| Exact query | Occupancy, hold ages, last-attack/release clocks and absence bits; remaining duration, time/row progress, BOS/EOS, BPM and meter. |
| Future skeleton | Next **16 inter-head gaps** and head counts over **1/2/4/8/16/32 beats**. Phase uses periods of **1/4/16 canonical beats**. |
| Missing organization | No audio, chart identity, persistent seed/style summary, sampled chart latent, whole-chart pattern statistics or learned full-skeleton representation. Landmark reads are disabled. |
| Output | Joint scores over 625 actions, using 25-way hand scores and rank-16 cross-hand coupling. Releases use a pointer conditioned on both hands, row codes, lane clocks and previously placed releases. |

These are verified in the frozen [features](/Users/l/ensomi/ensomi-model/artifacts/r2-runs/r2-phaseN-20261006/code/ensomi_model/r2/features.py:193) and [model](/Users/l/ensomi/ensomi-model/artifacts/r2-runs/r2-phaseN-20261006/code/ensomi_model/r2/model.py:237). Current `ln_level`/`ln_length` additions are absent from this checkpoint.

| Candidate link | Evidence for and against | Judgment |
|---|---|---|
| Teacher forcing → self-generation instability | All 64M exposures use real histories; no generated-prefix training or rollout objective. However, matched-history LN responses were largely calibrated in the prior investigation. | Mismatch confirmed; compounding mechanism plausible, not isolated. |
| Training windows → insufficient late-song learning | Starts are 12.5% BOS, 12.5% near EOS, 75% uniform. Full prefixes are encoded. Substantial loss exposure occurs late. | A 256-row memory cap or late-position coverage hole is rejected. |
| Finite history → loss of chart identity | Older pattern choices can become indistinguishable once recent tokens and exact state match. Against a universal explanation: older clocks survive, and prior LN analysis found little additional predictive value beyond the receptive field. | Information limitation confirmed; relevance to broader collapse unproven. |
| Insufficient timing/future information | There is substantial beat phase and local lookahead, but no explicit phrase plan or distant motif correspondence. | “No time coordinate” is false for R2; missing larger organization remains plausible. |
| Small/rank-limited row head → averaged patterns | Rank 16 constrains cross-hand interaction. But actions are joint, not independently sampled lanes; no controlled rank/capacity comparison exists. | Weak causal evidence. |
| Decoder → legal but degraded trajectories | Default sampling has no quality selector or repetition restriction. Train and decode share legality and release candidates. | No support mismatch found; legality does not imply playability. |

The recipe is uniform per-decision action-plus-release likelihood, with fixed normalization, AdamW and a cosine LR schedule. **CE fits a conditional distribution, not an arithmetic “average chart.”** Missing persistent information can make that conditional a mixture; whether it switches or becomes trapped depends on learned transition dynamics.

Frozen decoding is categorical sampling at **temperature 1**, without top-p/top-k. The mask enforces lane legality, heads at supplied times and final hold closure. **Rule L only limits access to scoped condition requests; it has no effect with an empty natural track.** There is no frozen 60 ms hold mask. See [sampling](/Users/l/ensomi/ensomi-model/artifacts/r2-runs/r2-phaseN-20261006/code/ensomi_model/r2/sampling.py:45) and [locality](/Users/l/ensomi/ensomi-model/artifacts/r2-runs/r2-phaseN-20261006/code/ensomi_model/r2/locality.py:1).

Corrected selection picks 56M, but its own-history calibration uses **64-row continuations**. Its LN-oriented guards do not establish song-long pattern stability.

**3. Diagnostics run**

**Information boundary, frozen 48M — measured.**

At decision 768:

- Raw-token gradients reached exactly rows **257–767**.
- Replacing older legal history changed cumulative heads **768→1,539**, while preserving recent tokens and exact query: **identical logits**.
- Moving an old release changed an exact clock and slightly changed logits. Old information is therefore not universally erased.
- Scoring through a 256-decision loss window versus a singleton gave identical logits. Actually cropping input history to 256 tokens changed the encoding; cropping to 511 did not.
- Dense versus online-cache logits differed by only **2.86×10⁻⁶** in the probe.

The 511-row horizon spans a median **61 seconds**, with position-pooled 10th–90th percentiles **34–85 seconds**, on the existing 116-chart panel. It is finite, but not uniformly short.

**Training coverage — analytically measured from the actual sampler/cache.**

Across 11,368 fit_train charts:

- **45.94%** of expected sampled head decisions occur at row 511 or later.
- Early/middle/late thirds receive **28.98% / 30.20% / 40.83%** of expected head-decision counts.

These are exposure/coefficient totals, not gradient magnitudes. Results and script: [information_probe.json](/Users/l/ensomi/ensomi-model/artifacts/r2-collapse-20261007/astra/information_probe.json), [information_probe.py](/Users/l/ensomi/ensomi-model/artifacts/r2-collapse-20261007/astra/information_probe.py).

**Temperature intervention, selected 56M — measured.**

Six deterministic fit_dev charts, 1,571–2,086 rows, two seeds, three temperatures: **36 whole-song runs, 611.5 seconds**. Only row temperature changed; release sampling stayed unchanged. All runs passed replay checks.

Repetition alarms used empirical 99th-percentile thresholds from **2,137 source windows**, stratified by head-row rate.

| Late-third measure | Source | T=0.8 | T=1.0 | T=1.2 |
|---|---:|---:|---:|---:|
| Repetition-alarm windows | 0% | 0% | 1.67% | 0.83% |
| LN share | 27.85% | 9.25% | 30.67% | 52.20% |
| Adjacent dependence above shuffle null, bits | .518 | .603 | .374 | .188 |

The fixed test is **inconclusive**: baseline alarm excess was below its predeclared five-percentage-point gate. No arm produced an eight-row quad wall. Lower temperature also failed composition guards, reaching **91.1% single-head rows late**, versus 69.6% in sources.

The secondary shuffle comparison found natural-T adjacent dependence below sources by **0.178 bits overall**, with a 90% chart-bootstrap interval **[−0.243, −0.117]**. Natural-T dependence was **.369/.330/.374** across thirds: weaker organization, but no monotonic decline. This statistic measures adjacent dependence; its shuffle null also disrupts timing/state relationships, so it does not isolate musical quality.

Results: [metrics.json](/Users/l/ensomi/ensomi-model/artifacts/r2-collapse-20261007/astra/metrics.json). Scripts and all generated action arrays are in the same directory.

**4. Core questions, ranked**

1. **Does increasing self-generated history cause the pattern deficit at a fixed song position, or is the deficit already present on short continuations?**
2. **Does remote chart organization supply useful predictive information beyond the available local context?**
3. **Can decoding improve stability without substantially changing composition?** The temperature probe provides no supported remedy.

**5. Strictly scoped experiments**

**A. Self-generation horizon — next experiment.**

- **Question:** Does accumulated self-history worsen organization independently of song position?
- **Manipulation/control:** At the same late endpoint on eight fixed long charts, generate the preceding **128, 512 or 1,024 rows**, starting each arm from its corresponding real prefix. Frozen 56M, T=1, two seeds; evaluate the identical final 128-row span. The shortest continuation is the control.
- **Primary metric:** Absolute source-relative gap in shuffle-corrected adjacent dependence.
- **Pass fixed now:** The 1,024-row arm worsens that gap versus 128 rows by **≥0.05 bits**, with a chart-bootstrap 95% interval above zero. Otherwise the accumulation hypothesis fails this screen.
- **Cost:** Estimated **10–15 Mac minutes**, two threads.
- **Meaning:** Positive supports a self-history mechanism. Negative directs attention toward an already-local modeling deficit; it does not establish that long-range generation is generally safe.

**B. Remote-information value.**

- **Question:** Does older organization add predictive information?
- **Manipulation/control:** Predict the next 128-row head-mask transition distribution using recent-511 summaries, exact state and skeleton features. Compare identical ridge predictors supplied either genuine older-prefix summaries or summaries shuffled between rate-matched charts. Split by song.
- **Primary metric:** Held-out mean squared error of the transition-frequency vector.
- **Pass fixed now:** Genuine remote summaries reduce error by **≥10%**, with a song-bootstrap 95% interval excluding zero.
- **Cost:** Estimated **5–10 Mac minutes**; no generator training.
- **Meaning:** Positive motivates persistent-information experiments. Negative argues against simply enlarging memory. Either conclusion is limited to the probe’s representation and predictor family.

**C. Row temperature — completed.**

- **Question/manipulation/control:** T=0.8 or 1.2 versus T=1, holding checkpoint, skeletons and seeds fixed.
- **Primary metric:** Late repetition-alarm fraction.
- **Pre-fixed pass:** Baseline excess over sources >5 points; reduction ≥10 points and ≥50% of excess; improvement on at least four of six charts; entropy-error increase ≤0.25 bits and chord-distribution TV-error increase ≤0.05.
- **Cost:** **10.2 Mac minutes measured.**
- **Outcome:** Inconclusive baseline gate; lower T additionally failed both composition guards. No temperature recommendation follows.

Before promoting any intervention, its numeric improvement must correspond to the human’s natural-generation failure. These screens are not substitutes for that check.

**6. The control problem**

LN share/length inputs can stabilize those quantities without preserving playable patterns. Difficulty optimization can likewise attain a scalar target through quad-heavy degradation. First establish natural whole-song stability; then test controls while retaining the same pattern-quality checks. Further difficulty-measure work is not the immediate dependency.

**7. What to stop or not do**

- Do not restart the LN-level fine-tune as the answer to general pattern collapse.
- Do not treat longer loss windows, more CE exposure, higher rank or enabled landmarks as established fixes.
- Do not promote temperature changes from entropy or repetition alone.
- Do not equate the 56M guard pass with playable whole-song generation.
- Do not carry R1 or R2-v1 failure measurements into phase N without checking them.

No training was run. No tracked files or research notes were changed. All scripts and outputs are under `artifacts/r2-collapse-20261007/astra/`; no report file was written.
----- END report -----
