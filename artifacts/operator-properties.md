# Operator properties for chart evaluation

Shareable. Written 2026-10-03 by main session `fa69c029` (Claude, control plane), at the human's request to design the properties evaluation operators must hold, aimed at the failures seen before, legacy v2's Control V3 features first ([private, local](private/human-inputs/fa69c029-7080-4f45-8b54-be050497532a.md#prompt-1)). Everything here is `(proposed)`, not reviewed. It replaces the short list in [p-operator-properties](evaluation-first.md#p-operator-properties) and is what an Astra brief for operator design would carry ([d-operators-reserved](evaluation-first.md#d-operators-reserved)). Code it builds on: branch `eval/corpus-beats` at `2b554a0`, `src/ensomi_model/evaluation/` (`case.py`: `EvalCase`, `Operator`, `evaluate`; `beats.py`: `ChartEvents` with canonical coordinates).

## What Control V3 computed

<a id="s-control-v3"></a>**Observation, 2026-10-03, from a read-only catalogue of `origin/legacy/v2` (= `5c56e28`) by a fresh subagent of this session; the main thread re-read four of its citations (the kernel, the jack cut-off, confidence gating, the rhythm bucket) and they match.** Paths below are under `5c56e28:src/ensomi_model/`.

- Twenty channels, computed once over the whole chart on a 0.1 s grid as sums under a triangular kernel over ±L seconds divided by L, so every value is symmetric in time and per second (`features/control_v2.py:194-203`), then interpolated to 20 ms frames and cut into non-overlapping 2 s windows (`features/control_v3_targets.py:95-102`). L is 3 s for density, hand balance and repetition. No normaliser is fitted anywhere (`features/control_v3_artifact.py:270-277`).
- Channels: density level and burst (log rate of chord-weighted onsets per second, lanes pooled); hold occupancy (held time over four lanes, pooled); LN change rate (changes of the held-lane mask); chord ratio; jack excess and jack streak exposure (same-lane pairs under a fixed 0.22 s gap, against a random-lane null); hand balance, signed and absolute (lanes {1,2} against {3,4}); exact, shifted and motion repetition (concentration of transition tokens within ±3 s, with rhythm bucket round(2·log2(gap/beat))). Jack, chord, hand and repetition use onsets only.
- Seven confidences count kernel support (Kish effective number of events) and are multiplied into their values (`features/control_v2.py:375, 515-519, 599, 756`).
- Use: regression targets (smooth-L1 per frame, weighted by confidence) for an encoder that read 12 s of Mel, dense timing and star; later only `density_level` fed mapper v2.1's density loss.
- No calibration, test, false-alarm rate, injected-defect sensitivity or agreement with a human judgment was found; it was never a gate or an evaluator. The mapper's density loss compared a per-frame lane-press count with `density_level`, a log rate per second under a 3 s kernel, at scale 1 and bias 0; that eval loss stayed at 1.690 to 1.707 from step 1,250 to 43,750 while token loss fell from 1.42 to 0.96 (`5c56e28:artifacts/runs/stage2_mapper_v2_1/.../report.json`, as reported by the subagent; not re-read). The control-memory run collapsed on one held-out song, cause unattributed (`5c56e28:artifacts/reports/evals/mapper_v21_44000_decoder_postmortem_2026-05-20.md:301-303`).
- Ruled out: "window edges split a pattern" does not apply; targets were computed on the whole chart and then sliced. Only the repetition chain reset at gaps over min(3 s, 4 beats).
- Not determined: the 12k control run's results and channel distributions (bings-mac `artifacts/runs/stage2_control*`, `artifacts/features/`, not read); why the parameter values were chosen.

For evaluation, reading the future is not itself a defect: a finished chart is judged. What carries over is the rest.

## Failures answered

Each property below names the failures it answers by these ids.

<a id="failures"></a>

| Id | Failure | Source | Strength |
| --- | --- | --- | --- |
| `cv-seconds` | Rates per second and windows in seconds; one subdivision gets different values at different BPMs; no snap or beat phase | [s-control-v3](#s-control-v3); `features/control_v2.py:194-203` | checked-code; consequence inferred |
| `cv-fixed-gap` | Jacks counted under a fixed 0.22 s: a 1/2 jack counts at 200 BPM (150 ms), not at 120 BPM (250 ms) | `features/control_v2.py:137, 400-402` | checked-code; examples are arithmetic |
| `cv-buckets` | Rhythm quantised by round(2·log2(gap/beat)): 1/3 and 3/8 share a bucket, as do 1/6 and 3/16 | `features/control.py:439-448` | checked-code; collisions are arithmetic |
| `cv-pooled` | Lanes pooled: a [1,2] and a [1,4] chord score alike; no per-finger load or four-finger relation, only a left/right split | [s-control-v3](#s-control-v3); `8e5e7ad:README.md` line 182 | checked-code, doc-claim |
| `cv-smoothed` | A 6 s kernel for density hides short extremes; the jack-streak maximum was computed and discarded | `features/control_v2.py:125-127`; `features/control_v3.py:100` | checked-code |
| `cv-gated` | Values multiplied by support count: a sparse chordy passage reads as "no chords" or "balanced" | `features/control_v2.py:375, 515-519, 599, 756` | checked-code; effect inferred |
| `cv-ln` | Holds as pooled occupancy and mask changes; no release placement against other heads, no per-lane LN, no duration distribution; holds invisible to jack, chord and hand channels | [s-control-v3](#s-control-v3) | checked-code |
| `cv-mirror` | Signed hand balance flips under mirror; a mirrored repeat is a different token | [s-control-v3](#s-control-v3) | checked-code; second effect inferred |
| `cv-no-transitions`, `cv-no-memory` | No explicit transitions; no memory across windows | `8e5e7ad:README.md` line 182 | doc-claim |
| `cv-pointwise` | A multimodal outcome regressed per frame to a conditional mean | `models/control/loss.py:107-163`; `8e5e7ad:README.md` line 182 | checked-code, doc-claim |
| `cv-scale` | Arbitrary scales, no fitted normaliser; unanchored channels not identifiable | `dfb4618:docs/formulation/gameplay-state.md:992-1019` | checked-code, doc-claim |
| `cv-units` | A loss compared quantities in different units (press count per 20 ms frame against a log rate per second) at scale 1; it never moved | [s-control-v3](#s-control-v3) | checked-code; measured as reported, not re-read |
| `cv-uncalibrated` | "Diagnostic only": never validated | [s-control-v3](#s-control-v3) | checked (absence) |
| `cv-split` | Evaluation split by beatmap path, so sibling difficulties of one song fell on both sides | `training/common.py:497-502` | checked-code; leakage inferred |
| `sc-scope-ln` | The September LN label counted LN heads per 16 s scope, ignoring holds entering from before; no duration or release | `audio-joint-2026-09:src/ensomi_model/research/typed_audio_continuation/controls.py:103-120` | checked-code |
| `le-extremes` | Scorers referenced per-chart peaks (99th percentiles); blind to how often and how long; accepted 31- and 28-attack single-column windows and a 25 ms LN | [s-response-blind](lineage-review/synthesis.md#s-response-blind) | replicated |
| `le-whole-chart` | Gates on whole charts; a song met its LN ratio by putting 27 to 47 LN into one passage that is all TAP in the source | [s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity) | replicated |
| `le-tuned` | Each evaluator added after a complaint and tuned on its charts; development panel of five TRAIN songs reused; agent page reading as evaluator | [s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity), [fm-self-evaluation](agent-failure-modes.md#fm-self-evaluation) | replicated |
| `le-star` | The response state re-encoded the star rating (R² 0.965); the star rating scores a 16 s single-column 12 Hz run at 4.14 stars | [s-response-blind](lineage-review/synthesis.md#s-response-blind) | replicated |
| `le-hard-rule` | A fixed support rule (60/50/50) added after a complaint excluded real relations in 13.7% of ranked 4 to 5 star charts | [s-response-blind](lineage-review/synthesis.md#s-response-blind) | replicated |
| `le-same-column` | Release checks same-column only; releases 40 ms or less before a head in another lane (6.9% of generated LN against 1.0% ranked) never seen | [s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state) | replicated, audited |
| `le-ms` | Millisecond thresholds: the pooled "30% to 40% of LN at 80 ms" was mostly one 210 BPM song | [s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state), [s-no-time-coordinate](lineage-review/synthesis.md#s-no-time-coordinate) | replicated, audited |
| `le-vacuous` | An attribution flag ("could have kept the hold") true in every case, so it tested nothing | [s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state) | replicated, audited |
| `le-noise` | Seed noise about 0.2 star against margins of 0.1; a single-pair effect of -0.35 became -0.04 with 16 seeds | [s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity), [fm-margin-below-noise](agent-failure-modes.md#fm-margin-below-noise) | replicated |
| `le-label` | Whole-chart stars as labels while a 16 s passage sits a median 0.54 star below its chart; realised stars ran 0.57 above the request | [s-controls-order](lineage-review/synthesis.md#s-controls-order) | replicated |
| `le-grid` | The lineage exported a constant 120 BPM, so a grid read from the output file is meaningless | [p-eval-object](evaluation-first.md#p-eval-object) | checked |

## Shape of an operator (proposed)

A pipeline of typed stages, so each property is checked at the stage where it can fail:

1. **Read**: the events of the case on the case grid (`EvalCase.events`), the given context and the condition record; declared aspects only.
2. **Describe**: quantities per event, per relation between events, or per run, each with a declared unit: canonical beats or milliseconds.
3. **Segment**: musical windows at several scales; runs measured whole.
4. **Refer**: the same operator applied to corpus cases of the fit split with the same scope shape and condition, keyed by requested star, canonical BPM and the fixed aspects.
5. **Depart**: where the observed value lies in the reference (a conditional tail probability), never a distance to a point.
6. **Aggregate**: per song, the rate of departing spans and the length of departing runs, each span attributed to its time.

## Properties (proposed)

<a id="p-op-reads"></a>**A. What an operator reads**

1. **Declared reads, lesion-tested.** Each operator declares the aspects it reads (grid, head times, release times, lanes, given context, audio, requested star) and those it judges. One test per aspect: perturbing an undeclared aspect leaves the output unchanged on every held-out chart; perturbing a declared one changes it on some. Answers `le-star`, `le-grid`, [fm-tests-not-binding](agent-failure-modes.md#fm-tests-not-binding).
2. **Scope locality.** Output depends only on events in the scored and given spans; given events are read, never scored. A run or hold that starts in the given span and continues into the scored span is measured whole and attributed to the scored part. Check: replacing everything outside both span sets with another chart changes nothing; an anchor of n attacks in the given span extended by k in the scored span scores as n + k; a hold entering the scope is counted. Answers `sc-scope-ln`, `cv-no-memory` at the scope edge; follows [d-partial-scope](evaluation-first.md#d-partial-scope).
3. **Keyed by what was asked, at the scope's scale.** The reference key comes from the condition record: requested star, the grid's canonical BPM, the fixed aspects, the scope shape. Never the generated chart's own star or density, since those are judged: keying on realised star would hide a generator that runs 0.57 star too hard. A passage is compared with passages of the same length from corpus charts of that star, not with whole charts. For a generator without a star input, such as R1, the key is the source chart's star (agent reading). Answers `le-label`, `le-grid`.
4. **Conditioned on what is fixed.** When the condition fixes aspects, the reference is the corpus distribution of the free aspects given the fixed ones at the same scale: under a skeleton, chord sizes and jacks given the local head pattern. The operator then measures the generator's choices, not the condition. Check: on held-out corpus charts the departure rate is flat across strata of the fixed aspects (local head density, canonical BPM). Extends the skip rule in `evaluate`, which drops an operator only when it judges a fixed aspect outright.
5. **One code path for observation and reference.** The reference is the operator itself applied to corpus cases; no separately computed target, label or normaliser. Observed and reference values are then the same quantity in the same unit by construction. Answers `cv-units`, `cv-scale`.

<a id="p-op-units"></a>**B. Units and invariances**

6. **Musical units for structure, physical units for load.** Rhythm and arrangement in canonical beats, exact subdivisions and beat phase, with binary and triplet families kept apart; load (gaps a finger or hand must make, sustained rates) in milliseconds, keyed by canonical BPM. Each quantity declares its unit. Check: under a uniform time-stretch of chart and grid together, beat quantities are unchanged and millisecond quantities scale by the stretch. Answers `cv-seconds`, `cv-buckets`, `le-ms`.
7. **Must not flag.** Output is unchanged under lane mirror; renotation at half or double BPM, whole or per segment; a shift of chart and grid together; adding or removing expressive red lines and green lines; object order and encoding in the file. Recurrence counts a mirrored repeat as a repeat; hand balance is reported unsigned or as a pair. Check: identical output on every held-out chart. Answers `cv-mirror`.
8. **Must flag.** Output changes under transforms players feel: a lane permutation other than the mirror (it moves patterns between hands); shuffling or reversing rows inside a window; moving notes from 1/4 to 1/3 or 3/8 positions; moving a release relative to heads in other lanes; changing a hold's length in beats. Check: each transform at full dose changes the output on most held-out charts. Answers `cv-no-transitions`, `cv-pooled`, `cv-buckets`, `cv-ln`, `le-same-column`.

<a id="p-op-describe"></a>**C. What is described**

9. **Sequences, not bags.** The set includes transitions between successive rows, per hand and across hands, not only counts inside a window. Answers `cv-no-transitions`; checked by 8.
10. **Long notes as durations and placements.** A hold is described by its length in beats, its release's offset to the next head on any lane, the holds open on the same hand, and whether it sits on a dense or a sparse passage; held lanes enter jack, chord and hand descriptions as occupied fingers. Never by count, share or pooled occupancy alone. Answers `cv-ln`, `sc-scope-ln`, `le-same-column`, `le-whole-chart`.
11. **Hands and fingers explicit.** Four lanes are two hands of two fingers: jacks, trills, chord shapes, chords split across hands, hand balance and same-hand chains are described as such. A statistic pooled over lanes never stands alone. Answers `cv-pooled`, and the single-column run the star rating scores at 4.14 (`le-star`).
12. **Runs and recurrence over the scope.** Anchors, jack chains, streams and holds are measured at their full length; recurrence (how often a pattern returns, how far apart) is measured over the scope, not inside a fixed horizon. Answers `cv-no-memory`. Control V3 did not split patterns at window edges ([s-control-v3](#s-control-v3)), so the window-origin check (moving the origin by part of a bar changes no song-level result beyond a stated tolerance) is a robustness check, not an answer to a seen failure.
13. **Value and support apart.** A value is never multiplied by how many events support it. Sparse evidence widens the value's interval or moves it to a coarser scale; it does not pull the value toward neutral. Check: two chords in an otherwise empty bar read as chordy with a wide interval. Answers `cv-gated`.

<a id="p-op-compare"></a>**D. Comparison with the corpus**

14. **Distributions, not points.** A departure is the tail position of the observed description in the keyed reference; the reference may be multimodal; no distance to a mean, no regression target. Check: on a bimodal quantity such as a passage's LN share, a value between the modes departs and values in either mode do not. Answers `cv-pointwise`.
15. **Joint where marginals fail.** For each pairing known to fail, the set holds a joint description: LN choice by local head density, anchor by run length, chord size by gap. Check: a marginal-preserving defect (move LN between passages, keeping the song's LN share and head-density histogram) is detected. Answers `le-whole-chart`; LN ratio and duration marginals cannot fix LN interaction (`audio-joint-2026-09:docs/research/native_pattern_failure_analysis_zh.md:23`).
16. **Rate and length, not presence.** A song is judged by the rate of departing spans per minute and the length of its longest and total departing runs, each against its own keyed corpus distribution. One corpus-typical occurrence never flags. Check: ranked charts holding the rejected-looking examples (a 28-attack anchor, eleven LN of 80 ms or less) pass at the nominal rate; the same passage repeated at k times its corpus rate flags from a stated k. Answers `le-extremes`; follows [h-rate-not-presence](evaluation-first.md#h-rate-not-presence) and [s-ranked-contains-rejected](lineage-review/synthesis.md#s-ranked-contains-rejected).
17. **Several scales, no smoothing before departure.** The same description at row, beat, bar, phrase (several bars) and scope scale; a departure is taken at each scale before any averaging and reported where it appears, so a short extreme stays visible. Answers `cv-smoothed`, `le-whole-chart`.
18. **No fixed cut-offs.** Every threshold is a keyed corpus quantile; no fixed millisecond or count limit and no rule that excludes relations. Check: no operator flags above the nominal rate in any star band by canonical BPM band. Answers `cv-fixed-gap`, `le-hard-rule`, `le-ms`.

<a id="p-op-evidence"></a>**E. Calibration and evidence**

19. **Calibrated both ways before use.** Fitted on the fit split, thresholds set on the calibration split, reported on held-out, all split by song group ([p-split](evaluation-first.md#p-split)). False alarms within tolerance of nominal in every stratum (star band, canonical BPM band, condition, scope shape), not only pooled. A dose-response curve for each defect family, with the smallest detected dose. A flag's base rate on held-out corpus is recorded; one that fires on nearly every or no chart is not a measure. Degenerate witnesses (empty scope, one lane, all chords, the given context looped) must flag. An operator without measured sensitivity is dropped. Answers `cv-uncalibrated`, `cv-split`, `le-tuned`, `le-vacuous`.
20. **Defect families written before results.** Families come from the human's rejections and the review's findings, are fixed before operator results are seen, and are scored on songs the operator was not tuned on; an operator revised after seeing a family's failures is re-scored on fresh held-out songs. Answers `le-tuned`, [fm-self-evaluation](agent-failure-modes.md#fm-self-evaluation).
21. **Information beyond star and beyond the others.** Each operator's detection power is reported after conditioning on star and on the rest of the set; one explained by star or by another operator is merged or dropped. Answers `le-star`.
22. **Song as the unit of evidence.** Intervals by song bootstrap; generator comparisons carry seed spread; a difference below that spread is reported as none. Answers `le-noise`.
23. **Attributable and reproducible.** Every departure names its span as an osu! editor timestamp, its events and its reference quantile, so the human can open it; the output is a pure function of chart, condition, scope and reference version, and carries the evaluator hash ([p-claim-integrity](evaluation-first.md#p-claim-integrity) item 1).
24. **Preference use split.** Under [d-beyond-row-ce](r1-verdict.md#d-beyond-row-ce) the evaluator may supply preferences. The operators that do are a declared subset; a disjoint audit subset and the human's judgments never supply preferences and measure whether a gain is real; the preference subset's agreement with the human is measured on the same pairs. Replaces "never a training signal" in [p-operator-properties](evaluation-first.md#p-operator-properties). Needs the human.

## Out of scope

Whether the grid fits the audio, and musical placement beyond the grid ([d-placement-is-timing](evaluation-first.md#d-placement-is-timing)); player response as such ([response](../RESEARCH.md#response)); scroll speed and visual effects.

<a id="op-open"></a>
## Open, for the human

- Property 24: the evaluator as a preference source through a declared subset, with an audit subset kept out.
- Property 3: key on the requested or source star, never on the realised one; passages compared with same-length corpus passages.
- Property 6: load in milliseconds keyed by canonical BPM, structure in beats.
- Whether "failures we saw before" meant Control V3 only; the table also carries the lineage evaluators' and the September controls'.
