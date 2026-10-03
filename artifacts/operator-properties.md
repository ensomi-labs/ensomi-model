# Operator properties for chart evaluation

Shareable. Written 2026-10-03 by main session `fa69c029` (Claude, control plane), at the human's request to design the properties evaluation operators must hold, aimed at the failures seen before, legacy v2's Control V3 features first ([private, local](private/human-inputs/fa69c029-7080-4f45-8b54-be050497532a.md#prompt-1)). Revised the same day after the human's answer ([private, local](private/human-inputs/fa69c029-7080-4f45-8b54-be050497532a.md#answer-1)): the twenty-four properties of the first version (notes commit `44d820b`) converge into seven, around the human's decisions and priors below. Text not marked as the human's is `(proposed)`. It replaces the short list in [p-operator-properties](evaluation-first.md#p-operator-properties) and is what an Astra brief for operator design would carry ([d-operators-reserved](evaluation-first.md#d-operators-reserved)). Code it builds on: branch `eval/corpus-beats` at `2b554a0`, `src/ensomi_model/evaluation/` (`case.py`: `EvalCase`, `Operator`, `evaluate`; `beats.py`: `ChartEvents` with canonical coordinates).

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

## Decisions and priors, 2026-10-03

Source: the human's answer to the first version ([private, local](private/human-inputs/fa69c029-7080-4f45-8b54-be050497532a.md#answer-1)). Agent account; the original wording is not reproduced here.

<a id="d-failure-scope"></a>**Decision, human.** The failures to answer are Control V3's, the September controls' and the lineage evaluators', as in the table above.

<a id="d-evaluator-purpose"></a>**Decision, human. The evaluator's main purpose is to model the distribution of normal charts.** Supplying preferences is one use read off that model, not its purpose. The first version's "audit subset kept out of preference" was neither accepted nor rejected; it stays a proposal under property 7.

<a id="d-star-key"></a>**Decision, human (agreed to the agent's item).** The reference is keyed by the requested star, or the source chart's star for a generator without a star input, never by the star the output reached; a passage is compared with corpus passages of the same length from charts of that star.

<a id="d-golden-separate"></a>**Decision, human. Human-labelled golden parts in Beatmap Lens (`~/ensomi/beatmap-lens`) are for DPO rewards later, and work separately from the evaluator.** They do not fit, calibrate or select it.

<a id="d-no-cutoffs"></a>**Decision, human. No hard-coded cut-offs; calibration is a divergence between output and corpus, KL-like.**

<a id="h-body-priors"></a>**Priors, human, to be formalised and tested.**

- Chart organisation is in beats.
- Players play in physical seconds, and difficulty calculation is in seconds.
- Physical load comes from coordination, within a hand and between hands, and from strain.
- The body has a physical limit; as a gap in milliseconds approaches it, difficulty rises exponentially.
- Note types contribute different kinds of load, but an LN head and a tap are highly similar.

## Properties, converged (proposed)

Seven properties. Each names the first-version properties it absorbs (numbers at `44d820b`) and the failures it answers.

<a id="p-op-model"></a>**1. One model of normal, keyed by what was asked.** The evaluator is a conditional distribution of chart descriptions given the key: requested star ([d-star-key](#d-star-key)), the grid's canonical BPM, the condition's fixed aspects and the scope shape. Under a condition that fixes aspects, it is the distribution of the free aspects given the fixed ones, so it measures the generator's choices and not the condition. Observed and reference values come from the same code applied to corpus cases; there is no separate target, label or normaliser. Absorbs 3, 4, 5, 12. Answers `cv-pointwise`, `cv-scale`, `cv-units`, `le-label`, `le-grid`.

<a id="p-op-clocks"></a>**2. Two clocks.** Organisation is described in canonical beats: exact subdivision families (binary and triplet apart), beat phase, bars, recurrence. Load is described in seconds, because players play and difficulty is computed in seconds ([h-body-priors](#h-body-priors)). Each quantity declares its clock. Checks: under a time-stretch of chart and grid together, beat quantities are unchanged and seconds quantities change; output is identical under mirror, renotation at half or double BPM (whole or per segment), shift of chart and grid, expressive red lines and green lines, file order. Absorbs 6, 7. Answers `cv-seconds`, `cv-buckets`, `cv-mirror`, `le-ms`.

<a id="p-op-body"></a>**3. Load from the body.** Load is a function of presses and holds per finger and per hand: same-finger gaps, coordination within a hand (trills, chords, one finger held while the other presses), coordination between hands (simultaneity and alternation), and strain accumulated over time. Cost rises steeply, the prior says exponentially, as a gap approaches the body's limit; the limit and the rate are estimated from the corpus or stated as sourced priors, never set as a cut-off. A tap and an LN head enter as the same press, to be tested; a hold occupies its finger until release; a release is its own action, with its own load, placed against heads on every lane. Lanes are never pooled into one statistic. Absorbs 10, 11, part of 8 and 18. Answers `cv-pooled`, `cv-fixed-gap`, `cv-ln`, `le-same-column`, `le-hard-rule`, and the single-column run the star rating scores at 4.14 (`le-star`).

<a id="p-op-felt"></a>**4. What players feel stays visible.** Descriptions are sequences: transitions between successive rows, runs and holds measured whole (across window edges, and from the given span into the scored one), holds by length in beats and release placement, at row, beat, bar, phrase and scope scale, with no smoothing before the comparison. Must-flag checks: a lane permutation other than the mirror, rows shuffled inside a bar, 1/4 positions moved to 1/3, a release moved against another lane's head, a hold's length in beats changed. Absorbs 8, 9, 10, 12, 17. Answers `cv-no-transitions`, `cv-no-memory`, `cv-smoothed`, `cv-ln`, `sc-scope-ln`.

<a id="p-op-divergence"></a>**5. Divergence, not cut-offs.** Departure from normal is a divergence between the distribution of descriptions in the output and in the keyed corpus ([d-no-cutoffs](#d-no-cutoffs)), over passages, so how often and how long enter by construction and one corpus-typical occurrence costs nothing. Both directions count: output mass where the corpus has little (defects), and corpus mass the output never reaches (missing modes, such as never writing LN). Joint descriptions are kept where marginals are known to fail: LN choice by local density, anchor by length, chord size by gap. Sparse evidence widens the estimate's interval; it never pulls a value toward neutral. Absorbs 13 to 16, 18. Answers `le-extremes`, `le-whole-chart`, `le-hard-rule`, `cv-gated`, `cv-pointwise`; follows [h-rate-not-presence](evaluation-first.md#h-rate-not-presence).

<a id="p-op-honest"></a>**6. Honest reads and scope.** Each operator declares what it reads and what it judges, with one lesion test per aspect. Output depends only on the scored and given spans; the given span is read and never scored; a run crossing into the scored span is attributed to it. Absorbs 1, 2. Answers `le-star`, `le-grid`, `sc-scope-ln`, [fm-tests-not-binding](agent-failure-modes.md#fm-tests-not-binding).

<a id="p-op-evidence"></a>**7. Calibrated, and evidence by song.** Fitted on the fit split, calibrated on the calibration split, held-out untouched until reporting, all split by song group ([p-split](evaluation-first.md#p-split)). The divergence of calibration corpus against fit corpus is the floor, per stratum (star, canonical BPM, condition, scope). Dose-response for defect families written from the human's rejections and the review before results are seen; an operator without measured sensitivity, or explained by star or by the rest of the set, is dropped. Intervals by song bootstrap; spans attributable as osu! editor timestamps; output a pure function of chart, condition, scope and reference version, with its hash. When the model supplies preferences ([d-evaluator-purpose](#d-evaluator-purpose)), a view not used for preference shows whether gains are real (proposed). Absorbs 17 to 24. Answers `cv-uncalibrated`, `cv-split`, `le-tuned`, `le-vacuous`, `le-noise`, `le-star`.

## Out of scope

Whether the grid fits the audio, and musical placement beyond the grid ([d-placement-is-timing](evaluation-first.md#d-placement-is-timing)); player response as such ([response](../RESEARCH.md#response)); scroll speed and visual effects; Beatmap Lens annotations ([d-golden-separate](#d-golden-separate)).
