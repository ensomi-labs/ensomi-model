<!-- Provenance: written 2026-10-03 by a fresh read-only Fable subagent of main session 10fd41cc, consulted at the human's request on four evaluator design blockers; copied unchanged from the session scratchpad. It was given only the usable parts of Astra's output (operator-properties.md#d-astra-usable). The main thread checked two code claims: SNAP_TOLERANCE_MS = 2.0 at 2044c2b:src/ensomi_model/evaluation/beats.py:29 (correct); "no legality checker in src/" holds for general .osu charts, but R1's row-level contract and output verifier exist (2044c2b:src/ensomi_model/research/bounded_typed_continuation/contract.py, verification.py), tied to R1's timing and row interface. Agent reading: evaluation-first.md#a-fable-review. -->

# Second mind on four evaluator design questions

Written 2026-10-03 by a fresh read-only reviewer (Claude, control plane), for main session `10fd41cc`. Nothing here is a decision. Where a statement rests on reasoning alone it is marked **[no evidence]**.

Read: `notes/artifacts/operator-properties.md` (whole), `notes/artifacts/evaluation-first.md` (whole), `notes/artifacts/lineage-review/synthesis.md` (whole), `notes/artifacts/agent-failure-modes.md` lines 140-152 (standing rules), the cited anchors in `notes/artifacts/r1-verdict.md` and `notes/artifacts/r2-ln-design.md`; code at `2044c2b` on `eval/corpus-beats`: `src/ensomi_model/evaluation/{case,beats,corpus,redlines}.py`, `src/ensomi_model/osu_core/hitobjects.py` lines 1-40, `data/r2-corpus.json`; Astra's `artifacts/eval-operators-20261003/math.md` lines 139-245 (section 4 only) and `gap_tables.md`. Not read, by instruction: Astra's `src/ensomi_model/evaluation/operators/`, its tests, `report.md`, `preregistration.md`, `calibration_table.md`, the rest of `math.md`. Notes paths below are relative to `/home/lkurisu/ensomi/ensomi-model/.git/research-relay/notes/`; code paths to `/home/lkurisu/ensomi/ensomi-model/`.

One limit up front: the control plane has no Parquet reader (`pyarrow`, `pandas`, `polars`, `duckdb` all absent), so I could not count fit charts per (star, canonical BPM) cell from `data/r2-corpus.parquet`. Sparse-strata statements use the star bands and fold counts in `data/r2-corpus.json` only.

---

## Q1. What "KL-like" means in practice

### Recommendation

Keep the agent's two readings, but pin the chain that makes them calibrated, and name the blind spot of the per-chart reading in a formula so nobody later claims it covers collapse.

1. **Passage surprisal under the model of normal.** For a scored passage with description vector d and key k: s = -log p̂(d | k), p̂ fitted on the fit split.
2. **Out-of-fit reference for s.** The corpus distribution of s at key k comes from the fit split *cross-fitted by song group* (K folds; each fit-split passage is scored by a model that never saw its group). The passage's score is its rank u in that reference. On the calibration split the ranks must be uniform; the divergence of the calibration rank histogram from uniform, per key stratum, **is the floor**. This is where "calibration is a divergence" ([d-no-cutoffs](artifacts/operator-properties.md#d-no-cutoffs)) becomes a number that should be near zero, not near a ceiling.
3. **Song score = goodness of fit of the song's rank set.** With n scored passages and ranks u_1..u_n, the song statistic is a tail-sensitive goodness-of-fit of {u_i} to Uniform(0,1) (Fisher's -2 Σ log u_i, or Anderson–Darling; exact choice open), whose **null is empirical**: the distribution of that statistic over cross-fitted fit-split songs with the same n at the same key. The song's final score is its rank among those. One corpus-typical rare passage (u ≈ 0.01) then moves the statistic by about one null standard deviation at n = 15 **[arithmetic on Fisher's statistic, not measured]**; two or three such passages, or one defect spanning several passages, accumulate. Rate and duration enter by construction; nothing is thresholded.
4. **Both tails, reported apart.** Low ranks are output mass where the corpus has little (defects). High ranks (every passage near the mode) are the per-chart face of collapse; D10's loop and any mode-seeking generator live there. Report the low-tail and high-tail statistics separately; do not fold them into one number.
5. **Population reading, both directions.** Over a generator's outputs at a key, report (a) the mean low-tail song score (Q→P flavour) and (b) a P→Q coverage: the share of held-out corpus passages (or corpus cells with mass ≥ ε) that the output population never reaches, estimated per family with the same estimator at **matched sample size**, with the corpus-against-corpus value at the same n as its floor. The missing-modes list ("no holds of 1-2 beats at any star", "never a 1/3 subdivision") is the interpretable form.
6. **Intervals.** Posterior intervals for p̂ (Dirichlet–multinomial for categorical descriptions, kernel posteriors for continuous ones) carried through to s, and song-group bootstrap for every reported number. A chart is "atypical" only when the interval of its song score clears the floor's interval. Sparse keys widen the interval; the point estimate borrows from neighbouring keys (Q2), never from a uniform prior.

### Reasoning

**Why Astra's floor sat at 0.586 of 0.693.** Its score was a Jensen–Shannon divergence between one song's *empirical histogram* of quantile-cell tokens over 16-beat passages and a smooth mixture ([s-astra-operators](artifacts/operator-properties.md#s-astra-operators)). A song has on the order of 10 to 30 such passages; the token vocabulary (four cells per numeric coordinate crossed with categorical state, over thirteen load channels) is far larger. The empirical histogram is then almost all zeros, and the JS divergence between a near-delta and a diffuse distribution approaches ln 2 regardless of how typical the song is. The floor is plug-in bias from finite n, not atypicality. The organisation score (0.315) being lower is consistent with a smaller vocabulary. Diagnosis **[not re-run; inferred from the description in the notes]**. The lesson generalises: any divergence between a single chart's empirical distribution and a population inherits a bias that depends on n and on the vocabulary, so the floor is never "near zero" and sensitivity drowns. Per-passage surprisal has no plug-in step; a rank against an out-of-fit reference is calibrated by construction. That is the reason to score passages, not histograms.

**The per-chart reading cannot see collapse, and the reason is a formula.** The mean surprisal of a generator's passages is the cross-entropy H(Q, P) = H(Q) + KL(Q‖P). Compared with the corpus's own mean surprisal H(P), the excess is KL(Q‖P) + [H(Q) - H(P)]. A generator with lower entropy than the corpus (a loop, a mode-seeker) *lowers* the excess and looks better than the corpus. Only an estimate with Q on the left of the log, or an explicit coverage term, sees it. The agent's proposal says this in words ("missing modes ... that a per-chart score cannot"); writing it as a formula in the frozen spec stops the lineage's habit of citing a component for what it does not do ([fm-claimed-not-done](artifacts/agent-failure-modes.md#fm-claimed-not-done)). The high-tail statistic in item 4 is the per-chart partial remedy; the coverage in item 5 is the population remedy.

**Why the reference must be cross-fitted.** If ranks are computed against the calibration split itself, the rank distribution on calibration is uniform in-sample and the floor is zero by construction; if against the fit split in-sample, the fit passages' surprisals are optimistic and calibration ranks skew low (false alarms). Cross-fitting on the fit split gives an honest reference and leaves calibration to *measure* the floor, as [p-op-evidence](artifacts/operator-properties.md#p-op-evidence) asks, and held-out untouched.

**Why the song null must be empirical.** Passages within a song are dependent (one mapper, one style, one density); the ranks of a normal song are not i.i.d. uniform even when the model is right. Keying by star removes part of it, not all. A theoretical χ² null would flag normal dense songs. The null distribution of the song statistic is computed from normal songs at the same key and length, which is what "evidence by song" ([p-op-evidence](artifacts/operator-properties.md#p-op-evidence)) demands anyway.

**Direction and estimator.** Per chart: Q→P in the sense that the output's passages are scored under P; it needs no density of Q and is usable with one chart. Population: both. For P→Q the output side needs either a density of Q estimated with the same code (a measurement on the generator's population, not a change to the model of normal, but see the owner's "corpus only" below) or a density-free coverage (nearest-neighbour recall in description space, per family). I recommend coverage first: it needs no second model and the missing-modes list is what a reader can act on. Estimators: smoothed histograms or small parametric densities per family (Q2); for population two-sample numbers, any bias-corrected estimator run at matched n with its corpus-vs-corpus floor; the estimator is open, the matched-n rule is frozen.

**On "corpus only."** [d-corpus-referenced](artifacts/evaluation-first.md#d-corpus-referenced) says the evaluator is fitted on the corpus and generated outputs supply no negatives. Estimating coverage or KL(P‖Q) uses generated outputs as the *measured* population, not as training data for the model of normal. I read that as compatible, but the distinction should be put to the owner in one line rather than assumed.

**Preferences and gaming.** A per-chart rank is calibrated by construction. As a preference signal it is gameable in one direction: optimisation pushes toward the mode, which the low-tail statistic rewards. Guards: (i) prefer only when the two continuations' song-score intervals separate (otherwise "no preference"), which keeps noise out of the labels ([le-noise](artifacts/operator-properties.md#failures)); (ii) exclude the high-tail and coverage statistics from the preference signal and keep them as the "view not used for preference" of [p-op-evidence](artifacts/operator-properties.md#p-op-evidence); (iii) version the evaluator and hold out defect families (Q3) so a generator that has found a blind spot is seen on the next report. DPO pairs from one committed state ([d-beyond-row-ce](artifacts/r1-verdict.md#d-beyond-row-ce)) help: the comparison is relative and most of the song is shared.

### Where I disagree with the proposal

- "Per-passage log-likelihood compared with the distribution of corpus passages' log-likelihoods" is right but under-specified in the two places that decide calibration: the reference must be out-of-fit (cross-fitted fit split), and the song aggregate must be a goodness-of-fit against an empirical null, not a mean. A mean reproduces the cross-entropy blind spot and a theoretical null reproduces false alarms on dense songs.
- The population reading needs the P→Q side made concrete now, not later; without it the first comparison of two generators will be won by the more collapsed one.
- Add the matched-n rule for every population divergence. Astra's floor shows what happens without it.

### Risks

- The empirical song null needs enough normal songs per (key, n) cell; at 6 stars and above the cell is thin (691 ranked or loved charts at 6, 237 at 7, 63 at 8 in the whole corpus, `data/r2-corpus.json` `ranked_loved_star_bands`; about 15% of groups are calibration). Pool n into bands and use the kernel over the key (Q2).
- Rank statistics lose the magnitude of a departure; keep s itself in the report for attribution (which passage, by osu! timestamp).
- Overlapping passages at several scales make the per-song statistic's null depend on the overlap scheme; freeze the scheme before computing any null.

### What would decide it

A one-family pilot on the mac, fit split cross-fitted, calibration split for ranks: (1) rank uniformity per star band (the floor); (2) the song statistic's null spread; (3) three constructed charts: a D10 loop, a mode chart (every passage the most common pattern at the key), a chart with one injected rare passage. Expected: the loop and the mode chart score *better* than normal on the mean surprisal and are caught only by the high tail or the song-scale recurrence family; the single rare passage sits inside the null. (4) Re-implement Astra's per-song histogram JS on the same family and show its floor against the rank floor; that tests my diagnosis of the 0.586.

### Freeze now / leave open

Freeze: the chain (passage surprisal, cross-fitted reference, rank, song goodness-of-fit with empirical null by key and n, two tails apart, population coverage with matched-n floor); song-group bootstrap intervals on everything; no number reported without its floor. Open: the goodness-of-fit statistic, the kernel bandwidths, the coverage estimator, how families combine into one headline (it must be calibrated against the same empirical null, never a sum of per-family thresholds).

---

## Q2. What the model of normal is built from

### Recommendation

Hand-designed description families with small conditional densities, as proposed, **plus one small fitted sequence family**: a smoothed n-gram (variable order, Kneser–Ney or Dirichlet smoothing) over canonical-beat row tokens within a passage, mirror-canonicalised, with LN state. About 10 to 14 families in total, each one to three dimensions, each with a declared clock, reads, judges, symmetries and lesion test. Joint densities only where an injection or lesion on the harness shows the marginals miss the defect. A full learned density over sequences (the costlier alternative) is not needed for Phase 1 and would cost interpretability, which the attribution requirement ([p-op-evidence](artifacts/operator-properties.md#p-op-evidence): spans as osu! timestamps) needs.

### Reasoning

**Why hand-designed families carry the attribution.** Every failure in the table ([failures](artifacts/operator-properties.md#failures)) names a quantity: anchors by length and rate, releases against other lanes, holds by length in beats, subdivision family, hand assignment, LN concentration by passage. A family per quantity lets the report say *which* departure and *where*. The lineage's evaluators failed on calibration and scope, not on being hand-designed ([s-eval-sensitivity](artifacts/lineage-review/synthesis.md#s-eval-sensitivity)); Control V3 failed as a generative player state, which says nothing about statistics as evaluators ([p-operator-properties](artifacts/evaluation-first.md#p-operator-properties), last bullet).

**Why one sequence family is needed anyway.** The must-flag checks of [p-op-felt](artifacts/operator-properties.md#p-op-felt) (lane permutation other than mirror, rows shuffled inside a bar, 1/4 moved to 1/3) and the failures `cv-no-transitions`, `cv-no-memory` are properties of the row sequence. Covering them with marginals means one family per transition type, and each new generator defect in sequence structure needs a new family (the lineage's "added after a complaint", `le-tuned`). A smoothed n-gram over row tokens in canonical beats is a few hundred lines, fits on 17k charts in minutes on CPU **[no evidence; standard for this vocabulary size]**, gives a per-passage surprisal like any other family, and catches D6, D7, D8 and bar-scale repetition without being designed for them. Its risk is the one Q1 names: the most gameable family under collapse, so it enters the preference signal only through the two-tailed statistic, or not at all.

**How to avoid re-encoding the star.** The key contains the requested star by decision ([d-star-key](artifacts/operator-properties.md#d-star-key)), so every density *should* depend on star; the danger is different: a chart whose realised difficulty misses the request shifts every family at once, and the report becomes "the star is off" fifteen times, drowning pattern defects (`le-star`, `le-label`). Two mechanisms: (i) a **star-shift row first** in every report: the realised star of the scored span against the corpus distribution of realised span stars at the key (corpus charts at star s have a spread of span stars; `le-label` puts a 16 s passage a median 0.54 below its chart); (ii) a diagnostic second view, **not the reference and not the score**, of every family at the realised star, so the reader sees "atypical at the request and at what it reached" (pattern defect) against "typical at what it reached" (difficulty miss). (ii) touches the owner's "never the star the output reached" and needs the owner's consent as a reading; I would ask rather than assume. The property-7 drop rule ("explained by star") becomes operational as: a family earns its place by detecting at least one injected family other than D9 at a dose where the realised span star stays inside the corpus spread at the key.

**Sparse strata.** Smooth over the key with a kernel in (star as a continuous covariate, log canonical BPM), bandwidths chosen by leave-one-group-out predictive likelihood on the fit split, effective sample size reported per key, and the posterior interval widening as ESS falls. Never a fallback to a neutral value. Two specifics from the code: the canonical BPM axis is **not circular**. A chart at 158 notated keys to 158 and one at 162 keys to 81 ([d-fold-per-segment](artifacts/evaluation-first.md#d-fold-per-segment); `src/ensomi_model/evaluation/beats.py:34-43`), and a 375 ms gap is one whole beat at the first key and half a beat at the second, so beat-clock families must not borrow across the 80/160 boundary, while seconds-clock families may. Songs near the boundary get fewer neighbours on one side; report that as a wider interval. Within a key the fold is consistent: at canonical 120 a notated 240's 1/4 and a notated 120's 1/8 are both canonical 1/8 at 62.5 ms, so the mixing of tempo octaves inside a key that one might fear does not occur in beat labels. Two thirds of charts fold by -1 (`data/r2-corpus.json` `grid.dominant_fold`: 14,581 of 21,779), so most of the mass sits at canonical 80-160 from notated 160-320. At 7 and 8 stars (237 and 63 charts) the reference is borrowed from 6 by the kernel and labelled as borrowed; the reporting population is 2 to 6 stars ([d-target-distribution](artifacts/evaluation-first.md#d-target-distribution)) so this is secondary.

**Which joints.** The agent's three (LN choice by local density, anchor by run length, chord size by gap) match `le-whole-chart`, `le-extremes` and `cv-pooled`. I would add release placement by (hold length in beats, hand relation of the next head) for `le-same-column`, and same-finger gap by run length because the corpus's shortest gaps appear to come in bursts ([t-lab-vs-corpus-gaps](artifacts/operator-properties.md#t-lab-vs-corpus-gaps)). But the rule should be evidence-driven, not a fixed list: a joint is added when the harness shows the marginals miss an injected or real defect, with a ledger row.

**Description content, as a checklist against the properties.** Organisation (beats): subdivision denominator and residual in ms per head (use `residual_ms`, see cross-cutting on the 2 ms snap), beat phase, row-token n-gram, same-lane run length, chord size, hand assignment under mirror canonicalisation, hold length in beats, recurrence at bar and 4-bar scale, within-song dispersion of the above at scope scale (the collapse detector). Load (seconds): same-finger gap, same-hand other-finger gap, other-hand gap, press-while-partner-holds, release against the next head on every lane, run length in notes, history on two declared scales (under a second; 10 to 30 s) per [p-defer-split](artifacts/operator-properties.md#p-defer-split). Twelve to fourteen families; more than that and the per-chart combination loses power to multiplicity.

### Where I disagree with the proposal

- Pure hand-designed marginals will leak on sequence defects and invite the lineage's add-a-family-per-complaint loop; the n-gram family is the cheap hybrid.
- The joint list should be an outcome of the harness, not an input.
- "How many descriptions" needs a ceiling now, because the song-level combination's power falls with the count; I would set a cap of about 14 with any addition requiring a ledger row and a drop of another or a measured gain.

### Risks

- The n-gram vocabulary (16 lane masks × LN state × subdivision slot) needs smoothing choices that are themselves tunable; freeze the selection rule (held-out predictive likelihood on fit, by group) so the implementer cannot tune it on calibration.
- Kernel smoothing across star can hide a real discontinuity (mappers change technique between 3 and 4 stars **[no evidence]**); check by comparing predictive likelihood of the smoothed against a per-band model on the fit split.
- Loved charts widen the normal; see cross-cutting.

### What would decide it

On the fit split (cross-fitted) and calibration: fit the n-gram family alone and the hand-designed organisation families alone; inject D6, D7, D8 and a graded bar-copy family; compare detected doses. If the n-gram detects them at lower doses, it stays; if the hand-designed families do, the n-gram is dropped as redundant under the property-7 rule. Separately, the star-shift row against D9 shows whether other families' detections collapse to a star miss.

### Freeze now / leave open

Freeze: the family list with clocks, reads and judges, the symmetry table per family, the cap, the key and its smoothing rule (continuous star, log canonical BPM, non-circular, selection by group-held-out predictive likelihood), the star-shift row, the rule for adding a joint. Open: smoothing priors, n-gram order, bandwidth values, whether the realised-star diagnostic view is shown (owner's call).

---

## Q3. Defect families for calibration

### Coverage against the failure table

| Family | Answers | Verdict |
| --- | --- | --- |
| D1 anchor rate | `le-extremes`, `cv-pooled`, `le-star` | Dose is one-dimensional (rate) where the complaint had two axes (rate and length; the 28-attack anchor of [s-ranked-contains-rejected](artifacts/lineage-review/synthesis.md#s-ranked-contains-rejected) is a length). Make the dose two-dimensional: run-length multiplier and run-count multiplier. Unit (notes, dimensionless) is right. |
| D2 short holds ≤ 1/8 beat | `cv-ln`, `le-ms` | Beat unit is right for organisation, but the lineage's separating defect was holds ≤ 40 ms, which no source chart had, while quarter-beat holds are as common in sources as in outputs ([s-ln-open-state](artifacts/lineage-review/synthesis.md#s-ln-open-state)). 1/8 canonical beat is 47 to 94 ms across the canonical range, inside what ranked charts contain, so low doses are ranked-plausible by design. Add a seconds-clock variant (holds ≤ 40 ms) for the load side. |
| D3 release crowding ≤ 40 ms | `le-same-column` | Matches the measured 6.9% against 1.0%. A fixed-ms *dose* is legitimate for a seconds-clock family; the failure-table warning (`le-ms`) is about ms *thresholds in the evaluator*. State that distinction in the harness. Stratify the target lane by hand relation (same hand, other hand), since property 3 separates them. Report detection per canonical-BPM band so one fast song cannot carry the result (the 210 BPM lesson). |
| D4 LN concentration | `le-whole-chart`, `sc-scope-ln` | Well posed; the one family that is purely passage-level with whole-song marginals held fixed. Keep. |
| D5 jitter σ ms | `cv-seconds`, `le-grid`, [s-no-time-coordinate](artifacts/lineage-review/synthesis.md#s-no-time-coordinate) | Unit right. Note σ = 2 ms sits on the snap tolerance (`beats.py:29`, `SNAP_TOLERANCE_MS = 2.0`): about a third of heads leave "on grid" at the smallest dose, so a snap-flag description shows a cliff while the residual shows a slope. Descriptions must use `residual_ms`, not the flag (cross-cutting). |
| D6 hand permutation | `cv-mirror`, `cv-pooled`, must-flag of [p-op-felt](artifacts/operator-properties.md#p-op-felt) | Good. Lanes 2 and 3 are different hands, so this also moves hand balance. |
| D7 row shuffle | `cv-no-transitions` | Good. |
| D8 1/4 to 1/3 | `cv-buckets` | Good; only where heads are free. |
| D9 density drift | `le-label`, [d-star-key](artifacts/operator-properties.md#d-star-key) | This tests the key, not a pattern; keep it, and use it as the control in the star-shift test of Q2. |
| D10 degenerate | sanity | The loop variant is the only collapse test; it needs a graded sibling (below). |

Must-not-flag list: right as far as it goes; additions below.

### Missing families

1. **Graded repetition**: copy one bar over a fraction of bars (`cv-no-memory`); the loop in D10 is its extreme and gives no dose-response.
2. **Population collapse**: a fraction of songs replaced by a "mode chart" (every passage the most common pattern at the key). Individually plausible, wrong as a population; the only test of the P→Q coverage of Q1.
3. **Never writing LN**: all LN to taps in a fraction of songs. Per chart ranked-plausible (many ranked charts have no LN); the defect is the population share at a star. Named in [p-op-divergence](artifacts/operator-properties.md#p-op-divergence) as a missing mode but absent from the families.
4. **Context break under partial scope**: under `Scope.continuation` or `Scope.edit` (`src/ensomi_model/evaluation/case.py:294-299`), splice the scored span from a *different* corpus chart at the same key. Normal in isolation, wrong against its given span (density, LN use, hand pattern). No current family exercises the given-span reading of [p-op-honest](artifacts/operator-properties.md#p-op-honest), and DPO pairs are continuations from a fixed state ([d-beyond-row-ce](artifacts/r1-verdict.md#d-beyond-row-ce)), so this is the use case.
5. **Chord-size shift, graded**: move a fraction of notes into existing rows (or split chords); D10's all-four-note-chords has no dose.
6. **Off-grid releases**: releases placed at non-musical fractions of a beat, where releases are free (always, after [d-no-release-input](artifacts/r1-verdict.md#d-no-release-input)); `release_on_grid` exists in the corpus summary but no family touches it.
7. **Near-duplicate heads**: two heads within under 20 ms on different lanes that are not a chord, where heads are free; the mechanism behind short holds in the fresh expert ([d-grid-cause](artifacts/lineage-review/synthesis.md#d-grid-cause)).
8. **Real rejected outputs**: the twelve V68 charts that can be tied to files ([s-eval-sensitivity](artifacts/lineage-review/synthesis.md#s-eval-sensitivity)). The only negatives nobody constructed. Conditions: they were generated under given release times and exported at a constant 120 BPM (`le-grid`), so each needs a reconstructed condition record and the source chart's grid; without those they are unusable and should be listed as such rather than silently dropped.

### Doses and units

Rule to freeze: a dose is stated in the clock of the description it targets (beats for organisation, ms for load) **and** reported as a multiple of the chart's own baseline rate of the injected thing, as D1 already does. Detection is reported per key stratum (star band × canonical-BPM band), never pooled, with the song-bootstrap interval. The smallest detected dose is meaningless without the next item.

### Ill-posed injections, made explicit

Several families at low dose produce charts that sit inside the corpus (D2 at 5%, D1 at ×1 or ×2, D3 at 5% against a 1.0% base). That is not a flaw in the family; it is [h-rate-not-presence](artifacts/evaluation-first.md#h-rate-not-presence) at work. The harness should measure, for every (family, dose, key), where the injected chart's targeted description sits in the corpus distribution at that key: the **corpus-plausible dose** is the largest dose still inside the central 90%. Sensitivity is then "detected dose / plausible dose", and a non-detection below the plausible dose is not a failure. This converts the ill-posedness objection into a reported number and stops both the implementer claiming sensitivity it does not have and a reviewer demanding sensitivity that would be a false alarm.

### Legality

"Keeping every chart legal" has no definition in `src/` outside Astra's unaccepted operators (grep over `src/ensomi_model/osu_core/*.py` and `src/ensomi_model/evaluation/*.py` for legality or overlap finds only the star calculator's hold-overlap flag at `difficulty.py:410-430` and the scope checks in `case.py`). `ManiaHitObject` (`hitobjects.py:15-19`) is a bare dataclass. Before any injection: a legality checker (no two objects overlapping in one lane, release after head, times finite and within the song) with its own tests, applied to every injected chart, and an injected chart that fails it is discarded and counted. This is a blocker for Q3 and belongs to the first deliverable of Q4.

### Conditions and scopes

Each family applies only under some conditions (D5, D8, D9 and near-duplicate heads only where heads are free; the context-break family only under partial scope). The harness enumerates (family × condition × scope) and marks "not applicable" cells, so "sensitivity measured" is a claim per cell and `evaluate` skipping an operator (`case.py:350-357`) is visible rather than silent.

### Stopping the evaluator being tuned to the families

- **Held-out families.** The ten above and my additions are development families the implementer may see. A second set is written *after* the design freeze by someone other than the implementer (the owner, or a fresh agent given the failure table and the format), kept out of the implementer's checkout, and run once by the reviewer on the held-out split at report time. Their commit timestamp must postdate the evaluator hash.
- **Sealed randomness.** Which passages, lanes and bars an injection touches is drawn from a seed fixed in the report job, not available during development.
- **Split discipline.** Development injections on calibration charts; the report on held-out charts with held-out families; the implementer never sees per-chart held-out results.
- **Real negatives last.** The V68 charts are scored once, by the reviewer.
- **Must-not-flag as gates, extended**: lane mirror; whole and per-segment renotation; shift of chart and grid; expressive red lines; **and** green lines and hitsounds, file order, the chart's own grid against the same grid supplied by the condition, and time-stretch of chart and grid together with the signed check of [p-op-clocks](artifacts/operator-properties.md#p-op-clocks) (beat families identical, seconds families changed in the stated direction).

### Freeze now / leave open

Freeze: the development family list with its additions, the dose rule, the plausible-dose measure, the legality definition, the (family × condition × scope) table, the held-out-family protocol. Open: the exact doses within each family, the hand-relation stratification of D3, which of my additions the owner wants.

---

## Q4. Who implements, and how the design is protected

### Recommendation

Separate three roles and build in this order: **harness first, model second, review by neither.**

- **Harness** (legality checker, injections, must-not-flag transforms, split access, cross-fitting scaffold, floor measurement, report schema, receipts): a Claude worker on the mac via `ens run`. Mechanical, well specified, and it must exist before any model so that the metric is not shaped by the model it measures.
- **Model of normal** (families, densities, scoring chain): Astra under a frozen spec, with the harness as its only metric. Its last job misread the purpose ([d-astra-not-accepted](artifacts/operator-properties.md#d-astra-not-accepted)), but the brief then fixed the families and left "KL-like" and "model of normal" open, and an agent that "follows explicit instructions closely and infers little" (`~/ensomi/AGENTS.md`, Astra section) filled the gap with histogram JS. With Q1 and Q2 frozen in the brief, with a **forbidden list** (no quantile cells, no per-song histogram divergence, no cost law, no thresholds, no fit on calibration), and with a comprehension step (restate the purpose and the scoring chain in its own words, in `last.md`, before writing code; the main thread reads it and kills the job if it is wrong), the risk is bounded. If Codex quota or trust is the constraint, a Claude worker can do this part too; the separation and the order matter more than the agent **[judgment, no evidence beyond the one job]**.
- **Review**: a fresh read-only agent recomputes the headline from raw artifacts and checks every "in the system" sentence against receipts ([p-claim-integrity](artifacts/evaluation-first.md#p-claim-integrity) item 6).

### Frozen before implementation, reviewed with the owner

1. The key: requested star (continuous), log canonical BPM (non-circular), condition name, scope shape; the smoothing rule and its selection criterion.
2. The passage unit per clock (cross-cutting item 1), overlap and stride, attachment to scored spans and reading of given spans.
3. The family list: for each, clock, reads, judges (aspect names from `case.py:34`), symmetries that must hold and those that must break, lesion test, marginal or joint, in or out of the preference signal.
4. The scoring chain of Q1 and the matched-n floor rule.
5. Split use: fit cross-fitted for densities and references; calibration for floor and development injections; held-out once, by the reviewer.
6. The harness contract: report schema; receipt fields (evaluator code hash, `data/r2-corpus.parquet` SHA-256 `49ab9e1b…`, split salt, condition, scope, seed, family set hash); the (family × condition × scope) table; the plausible-dose column.
7. The legality definition.
8. The development families and the protocol for held-out families.

### What the implementer may change

Estimator internals under the frozen selection rule (bandwidths, smoothing priors, n-gram order, numerics), code structure, performance. Adding a family or a joint needs a question-ledger row and lands in a "candidate" set that the headline excludes until the reviewer has seen its sensitivity and lesion results. Not changeable: the key, the passage unit, the scoring chain, the splits, the report schema, the family set, the receipts; there are no thresholds to change.

### Mechanical checks that fail a claim when part of it fails

- **Property tests on the evaluator object, not on functions.** Renotation, mirror, shift and expressive-line tests run the whole evaluator on a fixed set of charts and compare output hashes. The scope test is a lesion: alter objects outside given and scored spans (output identical), inside the given span (output may change, the given-span-read families must), inside the scored span (output must change). A family without a passing lesion test is not registered.
- **The report refuses to render a headline** when any registered family lacks: a floor value at every reported key, a sensitivity cell for at least one development family, a lesion result, and a receipt. A missing item prints "untested", never a blank.
- **"In the system" means "in the receipt."** The receipt lists the family set hash and each family's version; the reviewer matches sentences to receipts (standing rules 2 and 3, [standing-rules-for-briefs](artifacts/agent-failure-modes.md#standing-rules-for-briefs)).
- **Held-out access is logged.** One function reads the held-out split; it appends to a log committed with the report; the reviewer checks the log shows one access, at report time, by the report job.
- **Hash binding.** Calibration results are stored under the evaluator hash; a comparison across hashes is refused by the report code, not by convention ([fm-name-drift](artifacts/agent-failure-modes.md#fm-name-drift)).
- **Noise floor before criteria.** The first run after any evaluator change reports song-bootstrap spread on calibration; every sensitivity claim is in multiples of that spread (standing rule 6).
- **Pre-registered claims** in the brief: "at dose d of family F in stratum s, at least X% of injected songs score above the 95th percentile of calibration songs", judged after the run as stated; everything else "post hoc".

### First deliverable

Not the operators. The harness run end to end with a **trivial evaluator** (one family: head count per passage in canonical beats) on calibration charts: legality checker and tests; the ten plus added injections; the must-not-flag transforms with hash comparison; the cross-fitting scaffold and floor; the report with receipts and the (family × condition × scope) table. The trivial evaluator should detect D9 and little else; that the report says so, cell by cell, is the test that the harness can say "fail". Second deliverable: the Q1 chain on two families (anchor by run length; release against every lane) with the Astra-floor reproduction of Q1's deciding experiment. Only then the full family set.

---

## Cross-cutting concerns

1. **Passage unit per clock, undecided.** "Passages of the same length" ([d-star-key](artifacts/operator-properties.md#d-star-key)) has two readings: 16 canonical beats is 12 s at canonical 80 and 6 s at 159. Load families in seconds then see different durations at the two keys. Options: one unit in bars for both clocks (the key's BPM makes duration known; load statistics are rates per second inside the bar unit), or a beat unit for organisation and a seconds unit for load. I lean to the first for simplicity of attribution, but it needs a decision before any null is computed.
2. **A hidden cut-off in the coordinates.** `SNAP_TOLERANCE_MS = 2.0` (`beats.py:29`) decides `snap` (0 when off grid) and the corpus `head_on_grid` statistic (`corpus.py:82-83`). A description built on the snap flag is a hard rule at 2 ms, against [d-no-cutoffs](artifacts/operator-properties.md#d-no-cutoffs), and D5 at σ = 2 ms would show a cliff. `residual_ms` is continuous and available (`beats.py:164-201`); organisation families should model the residual distribution and the denominator jointly, with the flag used only for bookkeeping. Likewise `COMMON_DENOMINATORS` up to 16 is a vocabulary choice; a 1/32 pattern in a slow song reads as off grid.
3. **Fold boundary in key smoothing** (Q2): no kernel borrowing across 80/160 for beat-clock families; edge widening near the boundary.
4. **Whole-chart star as the key for partial scopes.** Corpus charts carry one star for the whole chart; a 30 s scored span at request 4 is compared with 30 s spans drawn from star-4 charts, which range widely in local difficulty (`le-label`). The reference is wide and the evaluator correspondingly lenient on partial scopes until per-interval difficulty conditions exist ([d-final-conditions](artifacts/r2-ln-design.md#d-final-conditions)). State it in the report rather than let a reader infer precision the key does not have.
5. **Loved charts widen the normal.** 2,632 of 21,779 ranked or loved charts are loved (`data/r2-corpus.json` `status`). By decision they are in the population; the report should still carry a floor row for loved and ranked calibration songs separately. If loved songs are systematically atypical under a model fit on both, the owner learns what "normal" absorbed; no decision changes.
6. **Multiplicity.** With about 14 families, "any family flags at 5%" has a false-alarm rate near 50% under independence. The per-chart headline must be a single statistic with an empirical null across families, not a count of flags.
7. **The evaluator as DPO label source will be optimised against.** [d-beyond-row-ce](artifacts/r1-verdict.md#d-beyond-row-ce) makes this certain. Version the evaluator; keep the high-tail, coverage and held-out families out of the preference signal as the view that shows whether gains are real; expect the first generator trained on it to find a blind spot, and plan the second family set then.
8. **Real rejected outputs need conditions and grids.** The V68 charts exported a constant 120 BPM (`le-grid`); the evaluator reads the grid from the condition record (`case.py:321-325`). Without a reconstructed record per chart they cannot be scored honestly; list them as "not scorable" if so.
9. **Within-song dependence** makes every theoretical null wrong; every null in this design is empirical and keyed by n. This doubles as the reason the floor needs the calibration split and cannot be read off a formula.
10. **Compute and sync.** Fitting and injection run on the mac only; fitted densities should be small text files (JSON histograms, under the 4 MB mirror limit in `~/ensomi/AGENTS.md`) so the control plane can read and commit them; a fitted model that only exists as a pickle on the mac is not reviewable here.
11. **What I could not check.** Cell counts per (star, canonical BPM, split) in the Parquet (no reader on this box); whether Astra's injection code already contains a legality checker (not read, by instruction); the actual passage counts per song, which set the song-null granularity.
