# R2: how the generator chooses long notes without given releases

Shareable. Started 2026-10-03 by main session `31cad85c` (Claude, control plane). Holds the human's R2 proposal of 2026-10-03 as the agent read it, the agent's counter-arguments, and the LN census launched to settle them. Source: [private, local](private/human-inputs/31cad85c-4707-42f3-ac95-deddaab38209.md#prompt-1); agent account, the original wording is not reproduced here. Everything except the human's proposal is `(proposed)` and not reviewed.

Follows from [d-no-release-input](r1-verdict.md#d-no-release-input) and the LN defects in [s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state); feeds the `proposal` and `time` nodes of the problem map.

<a id="p-r2-human"></a>
## The human's proposal (2026-10-03, for debate)

- Input: a heads-only skeleton (the head times of a chart) and its timing (BPM segments, i.e. the musical red lines and canonical grid of `src/ensomi_model/evaluation/`). No audio, no release times.
- Steps: after a head row the model either advances to the next head row, or emits `BEFORE_HEAD`, a release-only row at a time strictly inside the gap. Head rows may also release.
- At a head row, layered prediction: heads (tap or LN entry) first and releases after, or a joint prediction refined afterwards.
- So R2 needs a trained distribution over release time inside a gap between two head rows.
- First step named by the human: identify the LN patterns of the corpus and their distribution, so the model can be checked against each.

<a id="a-r2-args"></a>
## Agent counter-arguments (proposed)

<a id="a-coupling"></a>**1. A release in a gap is often decided by the next head row, so deciding it before that row is the wrong order.** Inverse and full-LN mapping hold a lane until just before the same lane is pressed again. In the `BEFORE_HEAD` order the release time is sampled before the model knows whether that lane is pressed at the next head row, so the coupling has to be learnt through a guess. Alternative (agent's, `B`): one step per head row; the row action covers, per lane, held: keep, release at this row, or release before this row (then none, tap or LN head); free: none, tap or LN head. A lane released before the row gets a categorical over canonical grid positions inside the gap, conditioned on the row's discrete action. This keeps the human's "joint, then refine" option, folds `BEFORE_HEAD` into the next row, and fixes the step count at one per head row, which also makes preference pairs over a fixed start state simpler. Joint space: at most 5^4 = 625 lane actions under a support mask; 4K is small enough that factorising the discrete part buys nothing. Whether it is worth it depends on the share of `E2 repress` releases in the census.

<a id="a-hazard"></a>**2. Keep-or-release at every step trains each step, never the length.** Under both the human's order and `B`, a hold's length is a product of per-step decisions; the lineage's 40 ms holds and near-head releases came from release and head hazards sampled separately on a millisecond clock ([s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state)). Given heads and a beat grid remove most of that, but the length marginal is still not a training target. Counter-option (`C`): choose the release at LN birth as an anchor (the k-th next head row) plus an offset before it on the grid. It plans occupancy ahead and trains the length directly, and handles `E5 free-end` naturally, but it cannot express "until just before I press this lane again" without a deferred anchor. Agent's current preference: `B`, with the hold's elapsed length and the head skeleton ahead as inputs, and the length distribution checked by the census measures; `C` if `E2` is small and `E1` with anchor above 1 or `E5` is large.

<a id="a-ln-mode"></a>**3. LN amount is a chart- and passage-level choice the row model cannot infer reliably.** Expected (to be measured, census item 7): LN share is bimodal across charts and switches between passages. Without a chart or passage LN variable, taken from the seed context or a request, a row model infers the mode from its own history, which is how the lineage drifted to LN shares near 0 or above 0.9 without its inference-time controller ([s-controls-order](lineage-review/synthesis.md#s-controls-order)). The passage case is the 09-28 rejection pattern: 27 to 47 LN in a passage that is all taps in the source ([s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity)).

<a id="a-no-audio"></a>**4. Without audio, a gap of two beats is either a rest or a hold.** The skeleton carries some of the answer (mappers leave fewer heads where they hold), the rest is a prior. Release-only times agree between mappers of the same audio at F1 about 0.09 against 0.79 for heads ([s-release-sensitivity](lineage-review/synthesis.md#s-release-sensitivity)); that mixes "whether there is an LN" with "where it ends". Census item 9 separates them: if two mappers who both hold from the same head also end together, releases follow the music and audio will matter later; if not, release placement is style and the skeleton plus an LN-mode variable is enough.

<a id="a-grid"></a>**5. Release positions on the grid, not in milliseconds.** A gap's release position as a categorical over canonical subdivisions, with bar position available, since musical holds often end on beats or bar lines. Census item 4 measures how many lengths are whole half-beats.

<a id="ln-census"></a>
## LN census (how it was run)

Run by a fresh Claude worker on the control plane through `ens run`, started 2026-10-03; brief kept at `/tmp/claude-1001/-home-lkurisu-ensomi-ensomi-model/31cad85c-4707-42f3-ac95-deddaab38209/scratchpad/ln-census-brief-v2.md` (session scratchpad, not durable; the same measures as the Astra brief at `~/ensomi/.sync/cp/jobs/20261003-081303-ln-census/brief.md`), scripts under `~/ensomi/.sync/cp/scratch/ln-census/`, outputs under `artifacts/ln-census-20261003/` of the code checkout on the mac. Two Astra launches (`20261003-081131-ln-census`, `20261003-081303-ln-census`) were killed within 90 s: both began acting as the research-relay main thread (reading the skill, running `relay status`), the second despite an explicit role section in the brief. Cause found: the mac's `ensomi-model/.codex/hooks.json` installs relay hooks for Codex (mac-local, not synced; the control plane's `relay status` reports Codex hooks as not installed), so every `ens astra` brief arrives with relay's autoresearch context. Not changed by the agent; for the human. Population: ranked and loved fit-split charts of the R2 corpus at 2 to 6 stars (calibration and held-out not read). Measures fixed in the brief before the run: where releases fall relative to head rows, a five-class release explanation in priority order (`E1 on-head` by anchor index, `E5 free-end`, `E2 repress`, `E3 gap-grid`, `E4 gap-off-grid`), positions in gaps by gap length, lengths, cross-lane release neighbourhoods, release groups, chart and passage LN share, up to ten named pattern families with examples, and optionally cross-mapper release agreement. Descriptive only; no architecture recommendation asked.

What each result would move: large `E2` favours `B` over `C` ([a-coupling](#a-coupling)); bimodal chart share and frequent passage switches make an LN-mode input necessary ([a-ln-mode](#a-ln-mode)); high cross-mapper agreement given a shared LN head makes audio a near-term need for releases ([a-no-audio](#a-no-audio)).

<a id="s-ln-census"></a>
## LN census result (2026-10-03)

**Observation, single run, deterministic (identical `ln.parquet` over three full runs).** Population: ranked and loved fit-split charts at 2 to 6 stars, 12,551 charts in 4,596 song groups; 11,492 with LN carry 3,631,889 LN. Calibration and held-out rows are filtered at read with an assertion (script line 1443). Script `~/ensomi/.sync/cp/scratch/ln-census/ln_census.py`, SHA-256 `129ff19c…`, run as `ens run` job `20261003-083641-ln-census-final2` on control-plane code `eval/corpus-beats` @ `2b554a0` (the mac's `.git` sits at `840fb09`, working tree current); corpus SHA-256 `49ab9e1b…`; `ln.parquet` (132 MB, mac only) SHA-256 `e57c365d…`; report, `summary.json` and figures in `artifacts/ln-census-20261003/` of the code checkout, mirrored to the control plane. Every number below is in `summary.json`; the main thread checked the headline ones against it. "Beat" is a canonical beat (segments folded to 80 to 160 BPM), so a quarter beat spans about 94 to 188 ms.

- Release classes (stated rule, pooled): `E1 on-head` 68.8% (per-chart median 83.0%, IQR 63.8 to 94.1), `E2 repress` 15.1%, `E3 gap-grid` 15.5%, `E4 gap-off-grid` 0.1%, `E5 free-end` 0.5%. E1 by anchor: next head row 48.6%, second 14.6%, third 3.1%, later 2.6%. E1 falls with stars: 84.0% at 2 to 3, 60.8% at 5 to 6. E4 is near zero because `snap != 0` admits 1/96 steps that a 2 ms tolerance nearly always catches; restricted to 1/1 to 1/16 (post-hoc) it is 0.9%.
- Positions in a gap (post-hoc measure, added after the stated quartiles all sat at 0.5): 65.5% of the 1,131,438 off-row releases are within 2 ms of the gap midpoint. On a head row, or at the midpoint, or 1/8 or 1/4 beat before the next row covers 93.7% of all LN (per-chart median 98.0%). The census cannot separate "midpoint" from "coarsest subdivision inside the gap".
- Repress (E2): release-to-repress gap 1/8 beat for 51.9%, 1/4 beat for 30.6%; median 88 ms, under 40 ms only 1,210 of 548,347.
- A head in another lane 1 to 40 ms after a release: 1.96% pooled, and strongly star-dependent: 0.05% at 2 to 3, 0.44% at 3 to 4, 2.12% at 4 to 5, 5.50% at 5 to 6, 9.83% at 6 to 8. 42.7% of E4 releases have one, 0.59% of E1.
- LN share per chart: median 0.129 (IQR 0.039 to 0.264); 14.9% of charts at or below 0.01, 0.6% at or above 0.9. Ranked is not bimodal (a steady decline with a small rise in the 0.95 to 1.0 bin); loved has a small full-LN cluster (22 of 1,540) and a lower median (0.029 against 0.143) with more E2 (21.0% against 14.4%). No formal dip test (package absent).
- Passages: 2.2% of adjacent 16-beat windows switch between under 10% and over 50% LN; 23.4% of charts have at least one switch.
- Length: holds of 60 ms or less are 2.7% of LN (per-chart median 0); holds of a quarter beat or less 57.9% (per-chart median 48.6%).
- Chords: of LN heads born together, 47.1% of groups release together and 46.7% all staggered.
- Same audio, different creators (778 pairs): 18.4% of LN heads have a counterpart at the same time; of those, 45.8% release within 2 ms. Same creator: 51.0% and 58.4%. `creator` names the set host, so same-creator pairs include guest difficulties. No chance baseline was computed.
- Ten pattern families with three clickable examples each are in the report; they overlap heavily (17.2% of LN in exactly one family), so they describe, they do not partition. 0.6% of LN fall in none.

Not reproduced: the lineage's 1.0% (ranked) and 6.9% (generated) near-head release figures, whose definitions are not recorded; the report gives two readings per star band.

<a id="a-after-census"></a>
## What the census changes (agent reading, proposed)

- **Release vocabulary is small.** A release is well described as (anchor head row, placement) with placement in {on the row, gap midpoint, 1/8 before, 1/4 before, other}. The human's worry, a distribution over release time inside a gap, reduces to a categorical of about five values. This holds for every candidate order of decisions.
- **[a-coupling](#a-coupling) stands, at 15%.** One LN in seven is released because its lane is pressed again at the next head row. Under `BEFORE_HEAD`-then-next-row, those are guesses; deciding the release with the row it precedes (the agent's `B`), or heads before releases at that row (the human's first option), makes them a consequence.
- **[a-hazard](#a-hazard) weakens.** With 63% of LN ending at the first or second head row and 2.7% at 60 ms or less, the length is mostly "how many rows", which a per-row keep or release represents directly. The risk that remains is the 40 ms holds the lineage made, which a minimum-length mask can remove without harming ranked charts (median 0 per chart).
- **[a-ln-mode](#a-ln-mode) refuted in its bimodal form.** Ranked LN share is a continuous chart-level level, widely spread and stable over passages, not two modes. What survives: the level varies from none to about a quarter between charts and rarely changes inside one, so a chart-level LN amount (from the seed or a request) is a natural condition; a model without it can still drift.
- **[a-no-audio](#a-no-audio) partly answered.** Whether a head is an LN is mostly the mapper's choice (18% shared LN heads between mappers of the same audio). Given a shared LN head, releases agree about half the time, against 58% for the same set host; with no chance baseline, how much of that is music is not settled. No case for audio in R2 from this.
- **For evaluation and the support mask.** A release 1 to 40 ms before another head is ordinary at 5 to 6 stars (5.5%) and nearly absent at 2 to 3 (0.05%). A hard mask on it would remove ranked behaviour; a rate judged within the star band fits [h-rate-not-presence](evaluation-first.md#h-rate-not-presence).

<a id="d-final-conditions"></a>**Vision, human, 2026-10-03. The final system's inputs and conditions.** Full audio is available; event timing is generated by the system, not supplied. Difficulty, style (the concepts of the Beatmap Lens foundation, sibling repository `../beatmap-lens`) and LN share are conditions, each applying to the whole song or to a fixed interval. Which of these conditions R2 takes is explicitly left open. [private, local](private/human-inputs/31cad85c-4707-42f3-ac95-deddaab38209.md#prompt-2). Agent reading: this is the `control` node's target stated as an interface; a scoped condition is a per-row input that changes at interval boundaries, so R2's decision structure can take it later without change, whichever conditions R2 starts with.

<a id="d-r2-row-decision"></a>**Direction, human, 2026-10-03, after the census. R2 makes one decision per head row, and that decision sets LN releases jointly with the row's own actions.** The human "inclines towards" it; read as the chosen direction, not a frozen specification. [private, local](private/human-inputs/31cad85c-4707-42f3-ac95-deddaab38209.md#prompt-2).

- Each head row's decision covers the releases that belong to it together with the row's taps and LN heads. This replaces `BEFORE_HEAD` from [p-r2-human](#p-r2-human) and matches the agent's option `B` ([a-coupling](#a-coupling)).
- Song end: LN still held after the last head row are released by a terminal decision at the end of the song.
- LN born together in one row may be released together or apart; the model must represent both (census: 47.1% of such groups release together, 46.7% all staggered, [s-ln-census](#s-ln-census)).
- A release position is a head row plus a beat subdivision, seen both from this row and from the previous row.

<a id="a-r2-spec-gaps"></a>**Agent reading: what is clear, and what an implementation still has to fix (proposed defaults).** The direction is clear enough to build the decision structure; six points are not fixed by it, and each changes code:

1. Window. "Decides LN release within" lacks its object. Default: row k decides every release in the half-open gap after row k-1 up to and including row k.
2. Candidate positions. The census covers 93.7% of LN with four placements (on the row, gap midpoint, 1/8 and 1/4 beat before the row), but long gaps and the remaining 6% need general positions. Default: a variable candidate set, every canonical grid position in the gap down to 1/16 and 1/12, each scored from features that include beats after the previous row, beats before this row, subdivision level and bar position. This is where "seen from both rows" enters.
3. Joint release of a chord. Releases of several lanes in one gap must be decided jointly, or one lane after another, not independently per lane, or "together" and "staggered" cannot both be learnt at their rates. Default: the discrete lane actions jointly (at most 5^4 masked), then release positions lane by lane in a fixed mirror-equivariant order, each conditioned on those already placed.
4. Held-lane state. Telling a chord born together from holds born apart needs, per held lane, its birth row and elapsed length in beats. Default: both are inputs.
5. Terminal window. How far past the last head row the terminal decision may place a release; without audio R2 does not know where the song ends. Default: up to 4 canonical beats after the last head row, on the grid, to be checked against the corpus before it is fixed (not measured by the census).
6. Minimum hold. Holds of 60 ms or less are 2.7% of ranked LN, per-chart median 0. Default: the support mask removes holds below a threshold set from the census table, not by hand.

<a id="r2-open"></a>
## Open, for the human

- Which conditions R2 takes ([d-final-conditions](#d-final-conditions) leaves it open), including whether a chart-level LN amount is one of them.
- Whether R2 stays a continuation from a seed prefix, as R1, which is where its style would come from.
- The six points of [a-r2-spec-gaps](#a-r2-spec-gaps), where the defaults are the agent's.
- Whether the mac's Codex relay hooks (`ensomi-model/.codex/hooks.json`, mac only) should stay, since they make every `ens astra` worker act as a research lead ([ln-census](#ln-census)).
