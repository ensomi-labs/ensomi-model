# R2 implementation: settled design, defaults and open points

Shareable. Started 2026-10-03 by main session `cf834490` (Claude, control plane), at the human's request to start implementing R2 training code from R1 with the fixes already decided ([private, local](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#prompt-1)). Decisions are the human's; everything marked agent default or `(proposed)` is not reviewed. Written as a separate material because main session `4259953e` was editing `RESEARCH.md` at the time; the entry point is to link here once that session ends.

Builds on [d-no-release-input](r1-verdict.md#d-no-release-input), [d-beyond-row-ce](r1-verdict.md#d-beyond-row-ce), [d-r2-row-decision](r2-ln-design.md#d-r2-row-decision), [d-r2-spec](r2-ln-design.md#d-r2-spec), [d-final-conditions](r2-ln-design.md#d-final-conditions); feeds the `proposal`, `rollout` and `control` nodes.

<a id="d-r2-settled"></a>
## Settled before this session

- Inputs: head times only (no release times), BPM segments (musical red lines and the canonical beat grid of `src/ensomi_model/evaluation/`), song length. No audio.
- One decision per head row k: the row's taps and LN heads and every release in (row k-1, row k]; an end-of-song step admits no heads and only closes LN.
- A candidate release is described in ms and canonical beats from this row, the previous row and the LN's start; the decision commits one release time. Releases of several lanes are decided jointly. No minimum hold; 40 ms the candidate.
- Objective: cross-entropy plus DPO sequence optimisation from a fixed start state ([private, local](private/human-inputs/4259953e-714c-4628-8911-eb1ec9fafdd8.md#prompt-9) for the human's statement of R2 as CE plus DPO, conditioned only on head rows).

<a id="d-r2-conditions"></a>
## Decisions, human, 2026-10-03 (this session)

[private, local](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#answer-1), [answer-2](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#answer-2).

- **Conditions.** LN share and star rating are optional conditions scoped to intervals, which can be injected or changed during generation. A chart-level scalar fed to every row (the agent's option) was rejected. Each condition has a "natural" setting: no condition given, and the model still generates plausible charts. Agent reading: train with each condition independently dropped to a null value.
- **Injection form.** Not fixed in principle; the first build carries both an encoder (interval tokens the model attends to, which also shows upcoming changes) and FiLM from the active interval, behind one condition interface, compared by ablation on held-out CE and condition following.
- **Interval star labels.** Use the existing star algorithm (`src/ensomi_model/osu_core/difficulty.py`, the 2024-10-07 osu!mania port the corpus build already uses, equal to the API on ranked and loved charts to 4e-5), only on sections over a length x, after its sensitivity and response on short sections are reviewed. Study running, see [r2-star-sections](#r2-star-sections).
- **Seed.** Optional: trained with random seed lengths including none, so one model generates from scratch and continues a chart.
- **Scope of the first pass.** Data pipeline, model, CE training, generation and export, and a DPO trainer tested on synthetic preference pairs.
- **Where.** A new branch in the main code checkout (not a separate worktree), created only after the evaluation-framework cleanup of session `4259953e` is committed; the human says when.

<a id="p-r2-defaults"></a>
## Agent defaults stated to the human, not commented on (proposed)

- New package `src/ensomi_model/r2/`. It imports R1's temporal encoder (`temporal.py`), landmark memory (`long_memory.py`) and joint-head pattern, and `oracle_time_continuation`'s exact replay and `.osu` export, unchanged. R1's correction residuals (`routing`, `consequence`, `response`, `recovery`) and the `r1_restore` staging are not carried; DPO takes their role.
- Head-row decision: per lane, held: keep, release at the row, or release in the gap then none, tap or LN head; free: none, tap or LN head. The four lanes joint under a support mask, then gap release positions lane by lane in a fixed mirror-equivariant order, each conditioned on those already placed (an exact factorisation of the joint). EOS: same structure, no heads.
- Release candidates: the row, and canonical grid positions inside the gap at 1/16 and 1/12 beat. The 0.1 to 0.9% off-grid releases of the census are snapped to the nearest candidate in training and counted.
- Data: an R2 row cache built on the mac by parsing `.osu` with the evaluation package's `Chart` and `BeatGrid` (the corpus Parquet holds no hit objects).
- DPO: reference is the frozen CE checkpoint; a pair is two continuations of N head rows from one committed state, scored by exact sequence log-probability.

<a id="s-r1-code-map"></a>
## What R1's code offers (explorer reading, 2026-10-03)

A read-only explorer mapped `src/ensomi_model/research/bounded_typed_continuation/` at `eval/corpus-beats` (working tree). Main points, from its report (agent reading of code, nothing run):

- Everything that indexes candidates in R assumes one decision per supplied time with a role: `Timing`/`Schedule` (`contract.py`), the support mask (`support.py`), `TimingView.queries` lookahead (`features.py:135-166`), `SourceChart` and windows (`data.py`), `Rollout.step` (`generation.py:258-293`), and the exporter's time check (`otc/export.py:64`). These are rewritten for R2.
- Independent of the time set: the dilated causal encoder (`temporal.py`, 511-row field, mirror-shared hand stream), the landmark memory (`long_memory.py`, one landmark per 64 head rows), `JointHead` (`model.py:76-94`), seed pooling. Their input sizes follow from R1's feature constants, so the readout changes.
- `ExactReplayState`/`commit` (`otc/replay.py`) accept any increasing times; `export_osu` (`otc/export.py:87-143`) exports release-only rows at any time if each is materialised as its own row. R1 forbids a release and a head on the same lane at the same time (`contract.py:131-133`); agent default keeps that.
- R2 would depend on `oracle_time_continuation` (schema, replay, storage, runtime, export) and `chart.*`; not on `vacation_training`, which enters only through `r1_restore`.

<a id="r2-star-sections"></a>
## Star on short sections

Fresh Claude worker, 2026-10-03; brief in the session scratchpad (`star-sections-brief.md`, not durable), scripts `~/ensomi/.sync/cp/scratch/r2-star-sections/` (`run_sections.py` SHA-256 `c645cf22…`, `analyze.py` `0793d238…`), output `artifacts/r2-star-sections-20261003/` of the code checkout (`preregistration.md`, `sections.csv`, `dose.csv`, `summary.json`, `tables.md`, two figures, each with a receipt). Jobs `20261003-164806-star-sections-run` (exit 0) and `20261003-165546-star-sections-analyze`. `difficulty.py` and the corpus were hash-identical to `f6251c4` at every stage. Pre-registered: 240 fit-split ranked and loved charts at 2 to 6 stars, one per song group, 60 per star band; L in {4, ..., 90} s; reference `s_tile` = star of the section tiled to 240 s; provisional criterion in every band: median |e| ≤ 0.10, p90 ≤ 0.25, Spearman ≥ 0.95, dose ratio in [0.8, 1.25].

<a id="s-star-sections"></a>**Observation, single run (8,010 sections kept of 8,036).** Under the pre-registered criterion, x = none of the tested lengths. The main thread checked the headline numbers against `tables.md`.

- The section star is below its tiled reference in every kept section, roughly in proportion to the star: median relative error -46.8% at 4 s, -19.1% at 10 s, -10.3% at 20 s, -6.0% at 45 s, -3.6% at 90 s. At 90 s median |e| is 0.090, 0.125, 0.153, 0.180 in bands 2 to 5 (limit 0.10), p90 0.171 to 0.359 (limit 0.25).
- Ranking and response to content hold from short lengths: pooled Spearman ≥ 0.974 from 4 s; thinning dose ratio within limits from 10 s (0.842); compression from 20 s (0.826, marginal), 0.907 at 30 s. Within-band Spearman is lower (band 4: 0.895 at 10 s, 0.950 at 90 s); the pooled test is weak. Dose figures rest on 30 sections per L.
- Cause, in the code (decomposition post-hoc): the star is the sum of 400 ms strain peaks sorted and weighted by 0.9^i (`difficulty.py:348-355`), not normalised. (1) A section has about L/0.4 peaks, so uniform material reaches 1 - 0.9^N of its long value: dominant up to 8 s, gone by 30 s. (2) The weighted sum spreads weight over the top 20 or so peaks while a long chart repeating the material is carried by its top peaks: -0.14 star median at 90 s, and this is a length dependence of the algorithm for any chart shorter than 240 s, whole-chart corpus stars included. (3) Cold start and missing holds from before the section: small in median from 8 s. (4) The 400 ms grid's phase: small for the tile.
- Post-hoc options, not adopted: the tiled star as the label (zero error by definition; noise from grid phase, p90 ≤ 0.039); isotonic calibration per L (meets the error conditions only at 90 s); the closed form s/(1 - 0.9^N) (fails, removes only part 1).
- The worker could not write `report.md` (harness block on report files); its content is above and in `tables.md` and `summary.json`. Its pre-registration cites `difficulty.py` lines 255-257 for the clock-rate division; correct is 256-258.

<a id="a-star-label"></a>**Agent reading (proposed).** The algorithm's response to content is usable from about 20 to 30 s; its level is not a length-free quantity at any length. So "the star of an interval" needs a definition before it can be a label. The tiled star reads as "the star a chart made of this material would have", is free of length by construction, and gives song-level and interval-level conditions one scale if the song-level label is computed the same way (for songs of 240 s or more it equals the ordinary star). Raw section star with length as an extra input keeps osu!'s own number but makes a request's meaning depend on interval length. Question for the human.

<a id="r2-ml-design"></a>
## ML design by Astra (running)

At the human's instruction the mathematics, training recipe and sampling are decided before implementation, by Astra at max effort, fast tier ([private, local](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#prompt-2)). Job `20261003-164527-r2-ml-design` (`ens astra --rw --effort max --tier fast --no-hooks`; `--no-hooks` keeps the mac's relay hooks out of the job without moving them, [answer-3](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#answer-3)); brief at `~/ensomi/.sync/cp/jobs/20261003-164527-r2-ml-design/brief.md`; writes only `artifacts/r2-ml-design-20261003/` of the code checkout, no tracked files. Asked: exact factorised likelihood of a head-row decision and EOS, state and look-ahead, the interval condition interface (encoder and FiLM, natural setting by dropout, interval sampling, guidance), the CE recipe for one M5, the DPO loss from the KL-regularised objective (reference, beta, sum or mean over decisions, mask, CE or KL mixing, label noise, on- or off-policy pairs, segment length, alternatives), synthetic pairs with a known answer to test the trainer, sampling, and the check that fails for each component. The human's decisions and the census facts are stated in the brief as fixed; the agent defaults above are open to change with reasons.

<a id="o-r2-mirror"></a>
## Mirror equivariance of the whole decision (human remark, 2026-10-03)

The human pointed out that R1 scores a row with a mirror-equivariant head that flips the history and the row together, rather than scoring the row on its own, and asked that this be kept in mind for R2 ([private, local](private/human-inputs/cf834490-7d37-42c1-b587-3b7f1ba0dd93.md#prompt-3)). Agent reading (proposed): the requirement for R2 is p(M d | M s) = p(d | s) for the full head-row decision d (lane actions and gap release positions) and the state s (history, held lanes, look-ahead, conditions), with M the lane mirror. The agent default "release positions lane by lane in a fixed order" breaks this unless the order is itself mirror-invariant or the factorisation is symmetrised; a test that mirrors state and decision and compares log-probabilities to float tolerance fails if it does not hold. Passed to Astra as a follow-up on its thread.

Code: branch `r2/train` created in the main checkout from `eval/corpus-beats` at `17b73b9`, after the evaluation cleanup (`469d72e`, `17b73b9`).

<a id="r2-impl-open"></a>
## Open

- The definition of an interval star label and its minimum length, after [s-star-sections](#s-star-sections) (human).
- Whether the agent defaults above stand, after Astra's design; the design document, once reviewed by the agent and the human, becomes the implementation brief.
