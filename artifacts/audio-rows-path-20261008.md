# Head rows from audio: a separate path beside R2 (opened 2026-10-08)

Discussion session a6d111aa (worktree `~/wt/ensomi-model-audio-rows`, branch `docs/audio-to-rows` at `3692ca2`), started at the human's request beside the main session 6d71c094, which owns `RESEARCH.md`. This material records the human's decision on how the running system gets its head rows, the evidence it rests on, and what the decision implies for R2's inputs. Nothing has been built or run.

<a id="d-audio-rows-path"></a>**Decision (human, 2026-10-08 about 06:25 UTC, [private, local](private/human-inputs/a6d111aa-4123-4bf0-8743-94f00985c427.md#answer-1)): head rows come from audio on a path of their own, separate from R2 and outside the R2 plan.**
- **Purpose:** a running system end to end. The rows part does not need to be good yet.
- **Separate from R2.** It proceeds independently while the R2 work continues, and it is not part of the R2 plan.
- **One input for now: difficulty, scoped to sections.** It is not R2's difficulty input: the rows decide more of the difficulty than the arrangement does.
- **Route:** the whole song's beat grid first, then the heads. Heads are generated roughly in sequence. The exact form is open, and nothing requires one head per step. The constraint is speed: it must fit the realtime budget.
- **Representation:** grid positions with offsets, because R2 reads beat information and local BPM.
- **Grid:** the legacy BeatThis grid fitter. It has limitations but works in most cases; it may need fine-tuning later.

The decision answers five questions the agent put after reading the notes (the questions are in the private record). It settles, for this path, "whether head times stay given", which had been open since 2026-10-03 ([r1-verdict, open](r1-verdict.md#r1-open)): in the running system they are generated. R2 keeps training on human rows; R2 v2's scope exclusion of audio and head timing ([d-r2v2-scope](r2-v2-stage0-state.md#d-r2v2-scope)) is unchanged.

## What the decision rests on

- **The rows fix rhythm, density and most of the difficulty.** From the rows alone the star band is right 70 % of the time and star R² is 0.83 ([s-skeleton-achieves](r2-represent-skeleton-20261008.md#s-skeleton-achieves)). This is why the rows module carries its own difficulty input.
- **Rows without a difficulty choice lose it downstream.** On rows from the September timing pilots, R2 lost the chords and jacks of a band-5 song, and the rows of all four songs looked like band-3 rows ([o-skeleton-shift](r2-represent-skeleton-20261008.md#o-skeleton-shift), 4 songs, direction only). Those pilots were trained on 48 charts; whether the source density was supplied in those runs was not checked.
- **The same rows also carry several difficulties.** 69 % of chart pairs on near-identical rows sit in different bands, through chord density ([r2-represent-skeleton, table 1](r2-represent-skeleton-20261008.md#1-what-do-the-supplied-rows-carry)). R2 still owns that part. This fits the human's 2026-10-05 decision that R2's difficulty is a response relative to a baseline computed from the given skeleton ([d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss)): the rows set the base, and R2's difficulty moves relative to it.
- **Given density, the audio fixes most head positions.** Two human charts of the same audio agree at head F1 0.785 at 20 ms (584 pairs) and 0.94 when density is matched (151 pairs) ([lineage review, timing evaluation](lineage-review/astra/03-audio-timing-skeleton.md#2-attempts), from `artifacts/audio-skeleton/20260923-v1/distribution-audit/extended-summary.json` on bings-mac). Same-mapper pairs and copies were not separated there. The mapper's main row choice looks like density, which is what a difficulty input can steer [inferred].
- **Placement is a beat-tracking problem** ([d-placement-is-timing](evaluation-first.md#d-placement-is-timing), human, 2026-10-03), and the final system generates its timing from audio with difficulty scoped to the song or an interval ([d-final-conditions](r2-ln-design.md#d-final-conditions)).
- **A musical coordinate matters.** 99 % of ranked 3.5-4.5★ charts have at least 90 % of heads within 2 ms of a subdivision of their own timing lines. The September lineage wrote free milliseconds and its heads landed on the grid at chance ([s-no-time-coordinate](lineage-review/synthesis.md#s-no-time-coordinate)).
- **The legacy fitter is better than its headline error.** Median phase error 43 ms over 5,026 songs, mostly a constant lag of 29 ms behind the chart. With the lag removed, 53 % of songs start within 10 ms. Tempo is within 0.1 BPM on 73 % of single-tempo songs and mostly wrong on songs with several tempo sections (16 % of audio). Fitted grids for 5,050 songs survive ([s-pre-v3](lineage-review/synthesis.md#s-pre-v3)). Code: `5c56e28:src/ensomi_model/timing/` (branch `legacy/v2`); BeatThis at 50 Hz, then `GridFitter` to compact timing segments.
- **What September tried and why it does not count against this route.** The lineage moved from a frame pilot (heads F1 0.68-0.73 at 20 ms with source density supplied) to joint per-millisecond hazards for heads, releases and rows, with five ownership reversals and runs of 512-4,096 updates on one seed. The human closed it as unreliable and dropped its code ([lineage review, attempts](lineage-review/astra/03-audio-timing-skeleton.md#2-attempts), [audio-skeleton feedback index, timeline](audio-skeleton-human-feedback-index.md#2-timeline-of-the-loop)). Releases are no longer supplied to the generator ([d-no-release-input](r1-verdict.md#d-no-release-input)), so the release-only times, where two humans agree at F1 0.09, are no longer predicted. In September the human also set that the skeleton reads audio and earlier skeleton, not the row materialisation (feedback index, problem F); this path keeps that direction.

## What R2 reads from the grid (checked in code at `3692ca2`)

- R2 builds its grid from the chart's own timing lines: `musical_grid(red_lines, head_times)` (`src/ensomi_model/r2/cache.py`, line 106), stored as canonical segments (offset, canonical beat length, meter; `src/ensomi_model/r2/common.py`, `GridArrays`).
- Its queries read beat phase, local BPM as `g.bpm(t) / 120`, meter, beat distances to earlier rows, head density over 1-32 beat windows, and remaining beats to song end (`src/ensomi_model/r2/features.py`, lines 205-252 and 333-346). Release candidates and LN length in beats are built from the grid.
- The human's recollection is right: a generated grid feeds R2 directly. Grid positions with offsets give R2 the same coordinates it was trained on.

## What the fixed inputs imply (agent analysis, not decided)

Guardrail 4 of `.agents/skills/ensomi-research-guardrails/SKILL.md` asks what each fixed input lets the model learn and the evaluation see.
1. <a id="o-grid-shift"></a>**Observation, inferred, 2026-10-08: R2 is trained on human timing lines and will receive fitted grids.** The fitter's known errors (the 29 ms lag, wrong tempo on multi-section songs, possible half- or double-tempo choices) move R2's phase, BPM, beat-distance and release-candidate inputs. Canonical folding of beat lengths may absorb some tempo-multiple errors; not checked. Measurable without training: R2's teacher-forced NLL on source heads with the timing-line grid against the fitted grid, on fit_dev songs that have a surviving fitted grid.
2. **The section difficulty has to be defined on rows.** Corpus labels are chart-level star. R2 already has tiled-star interval labels (`src/ensomi_model/r2/labels.py`), but those are star of the full chart content, which includes the arrangement. Candidates for the rows module: head density per section, a rows-only star predictor (opus-skeleton's predictor reaches R² 0.83 at chart level), or a rows-only strain. Which one is the human's call.
3. **Offsets absorb grid error.** Human heads sit on their own timing lines; against a fitted grid the same heads carry the fitter's error as offsets. The offset distribution the model learns depends on which grid it is trained against: the chart's timing lines or the fitted grid.
4. **A whole-song grid needs the whole track before generation.** The Vision allows complete audio at inference. Whether ambient play has the whole track before it starts is still to confirm ([H4](lineage-review/synthesis.md#h4)).
5. **The difficulty input has to be shown to be used** (guardrail 3). R2's star input went unused because the rows already predicted star. Here the same audio supports many densities, so the input should carry information, but a teacher-forced ablation on trained weights has to show it.
6. **No measure of rows is validated** (guardrail 1). References exist: same-audio human charts (F1 0.785 overall, 0.94 at matched density), alignment to timing lines, the rows-only difficulty predictor. They characterise. Until one is checked against the human, the human's view of the running system decides.

## Open

- The form of the head generator: per-slot choices over grid positions with an offset, an event sequence with gap tokens, a bar at a time, or another form. Rough sequence and speed are the only constraints so far.
- The definition of the section difficulty (item 2).
- When and against what the fitter is fine-tuned; what to do on songs where it fails (multi-section tempo).
- How the rows' difficulty and R2's difficulty request are set together in the running system.

## Proposed first steps (not started; each needs the human's go)

1. Fitter check on existing outputs: the 5,050 surviving fitted grids against the charts' timing lines with the lag corrected, to size "works in most cases" and find the songs that need a fallback. Small mac job.
2. R2 grid-shift cost ([o-grid-shift](#o-grid-shift)): teacher-forced NLL with the timing-line grid against the fitted grid, on the same heads. Small mac job, no training.
3. A design study by a fresh subagent: head-generator forms that meet the realtime budget, with the difficulty definition options and how each would be evaluated against same-audio human charts.
