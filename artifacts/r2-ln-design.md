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
## LN census (running)

Astra job `20261003-081131-ln-census` on bings-mac, started 2026-10-03, read-write limited to `artifacts/ln-census-20261003/` of the code checkout; brief kept at `~/ensomi/.sync/cp/jobs/20261003-081131-ln-census/brief.md`. Population: ranked and loved fit-split charts of the R2 corpus at 2 to 6 stars (calibration and held-out not read). Measures fixed in the brief before the run: where releases fall relative to head rows, a five-class release explanation in priority order (`E1 on-head` by anchor index, `E5 free-end`, `E2 repress`, `E3 gap-grid`, `E4 gap-off-grid`), positions in gaps by gap length, lengths, cross-lane release neighbourhoods, release groups, chart and passage LN share, up to ten named pattern families with examples, and optionally cross-mapper release agreement. Descriptive only; no architecture recommendation asked.

What each result would move: large `E2` favours `B` over `C` ([a-coupling](#a-coupling)); bimodal chart share and frequent passage switches make an LN-mode input necessary ([a-ln-mode](#a-ln-mode)); high cross-mapper agreement given a shared LN head makes audio a near-term need for releases ([a-no-audio](#a-no-audio)).

<a id="r2-open"></a>
## Open, for the human

- Whether R2 may take an LN-mode or LN-amount input (from the seed context or a request), or must infer it.
- Whether R2 stays a continuation from a seed prefix, as R1, which is where its style would come from.
