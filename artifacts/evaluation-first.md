# Evaluation first

Shareable. Written 2026-10-02 by main session `13236b40` (Claude, control plane). Holds the human's decisions of 2026-10-02 on H11, H1, H2 and H3 of the [lineage review](lineage-review/synthesis.md#for-the-human), and the agent's proposed design for the evaluation work the human put next. Everything under "Proposed design" is `(proposed)` and not a decision.

## Decisions, 2026-10-02

Source: the human's free-text answer to the agent's digest of section 5 ([private, local](private/human-inputs/13236b40-ac9c-4abe-a791-60fb6e93c03f.md#answer-1)). Agent account; the original wording is not reproduced here.

<a id="d-target-distribution"></a>**Decision, human, 2026-10-02. The generation target is the 2 to 6 star ranked and loved 4K distribution, matched as a whole.** Output should look like that population: mostly ordinary charts with the few outliers it contains. The two H11 examples ([s-ranked-contains-rejected](lineage-review/synthesis.md#s-ranked-contains-rejected)) are acceptable. This answers H11 and the definition half of H1. Not answered: whether the row half with real times or the whole audio-to-chart system is measured first.

<a id="d-accept-set"></a>**Decision, human, 2026-10-02. Every ranked chart is acceptable; rejected examples are to be constructed, after a stricter evaluation design.** The accept side needs no human labelling. How much judging the human will do is not answered.

<a id="d-response-priors"></a>**Decision, human, 2026-10-02. The player response state includes physiology; its only direct data is the corpus; its priors need care.** Closest to option (b) of [H2](lineage-review/synthesis.md#h2). Consequence, agent reading: with corpus-only data, what key 1 adds beyond key 2 lies in its priors, so the priors must be explicit, sourced and testable ([d-key1-vs-key2](lineage-review/synthesis.md#d-key1-vs-key2)).

<a id="d-eval-first"></a>**Decision, human, 2026-10-02. Evaluation comes next, in two parts.** (1) Measure generated charts against the target better: the human rejects in generated charts patterns that ranked charts also contain, so presence of a pattern is the wrong measure. (2) Restrict agent-written code and agent-run experiments so that a claim cannot pass while part of it fails, as when the response module was never in the system the human played ([s-response-not-in-loop](lineage-review/synthesis.md#s-response-not-in-loop)). The agent doing the work must reason about the system end to end while keeping the essential constraints. Unresolved: whether "end to end" was also meant of the generation model.

## How "accept all ranked" and "reject what ranked charts contain" fit

<a id="h-rate-not-presence"></a>**Hypothesis, agent, open, 2026-10-02.** A ranked chart with a 28-attack anchor is acceptable; a generated chart that does the same is rejected. Both hold if the defect is a matter of rate, length, placement and context: how often such passages occur per song, how long they last, whether they sit where the music or the rest of the chart calls for them, at which difficulty. The lineage's scorers caught extremes and were blind to how often and how long ([s-response-blind](lineage-review/synthesis.md#s-response-blind)); its gates measured whole charts while the human judged passages ([s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity)). A distribution-level target ([d-target-distribution](#d-target-distribution)) is the natural home for this: the reference supplies the rate of every unusual thing, and generated output is wrong when it departs from those rates, overall or within a passage. What it would not catch: a pattern at the right rate in the wrong musical place. That needs audio-conditioned comparison and is left for later.

## Proposed design (proposed)

Two parts, matching [d-eval-first](#d-eval-first). Neither is built; each item names the failure it answers.

<a id="p-chart-eval"></a>### Part 1. Chart evaluation against the target distribution

1. **Reference split, frozen.** 2 to 6 star ranked and loved 4K charts, split by song group into a calibration part and a held-out part. Neither part is ever used to train a model or tune a generator. A failure set is held out the same way. Answers: evaluators tuned on their own failures, a development panel of five TRAIN songs reused across dozens of experiments ([s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity)).
2. **Unit: the passage, conditioned on difficulty.** Windows of a few seconds as well as whole charts, compared with reference windows of matching chart star band and local density. Answers: whole-song LN ratios met by stuffing one passage.
3. **Measures: rates and lengths, not presence.** Per-window distributions of column runs and anchors, jack lengths, LN share and LN duration in beats, release-to-next-head gaps, chord sizes, head offsets from the song's grid. Compared by tail rates and quantiles with song-level bootstrap intervals. In addition, a classifier two-sample test: a fresh discriminator trained per evaluation on reference against generated windows; its accuracy says how distinguishable the output is, and its most confident windows are the queue for human inspection. The discriminator is evaluation only, never a training signal or selector, or it becomes the next thing to game.
4. **Calibrated both ways before use.** False-alarm rate: held-out reference against calibration reference must pass. Sensitivity: constructed negatives ([d-accept-set](#d-accept-set)) made by injecting each known complaint at controlled doses into held-out reference charts (anchors at k times the corpus rate, short LN and releases just before a head, LN on TAP passages, head jitter off the grid), plus the lineage's real rejected outputs where they can be tied to files. Report the smallest dose detected. An evaluator without both numbers is not used to accept anything.
5. **Reporting rules.** Every table carries rows for held-out reference, R1 with real times, and the candidate. At least the agreed number of songs and seeds, with song-level spread; a difference smaller than seed noise is reported as none ([s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity): seed noise about 0.2 star against promotion margins of 0.1).
6. **Human spot-check.** A small blind sample from the discriminator's queue, judged by the human, kept as files. It tests the evaluator, not the generator.

<a id="p-claim-integrity"></a>### Part 2. Claim integrity for agent work

Failures it answers, all from the review: the response scorer that was not in the played system ([s-response-not-in-loop](lineage-review/synthesis.md#s-response-not-in-loop)); "R1" naming a descendant that had dropped four of its components ([s-lineage-r1-is-not-r1](lineage-review/synthesis.md#s-lineage-r1-is-not-r1)); the agent's own reading of Lens pages standing in as an evaluator; expert advice recorded as adopted while ignored in substance ([opus/interpretation](lineage-review/opus/interpretation.md) section 5).

1. **Frozen evaluator.** Evaluation code and the reference split are a separate, versioned unit. Experiment jobs may run it and may not edit it; every report carries its content hash. Changing the evaluator is its own reviewed change, never part of an experiment.
2. **Pre-registered claims.** Before a run, the brief states the claim, metric, threshold, songs and seeds. The result is judged against that statement. A post-hoc metric may be reported only as such.
3. **Execution receipts.** Each generated output records the checkpoint hash, the resolved config and a trace of the modules that actually ran in the generation path. A claim that a component is "in the system" is checked against the trace, not the source code.
4. **End to end through the human's entry point.** A component result is reported only together with end-to-end output on the fixed panel, produced by the same path the human plays or qualification scores. "Works in a study script" is reported as exactly that.
5. **Essential-constraint checks on every output.** Legality, causality of decisions, the target distribution, the agreed support rules: checked automatically, and a violation is reported with the result, not after it.
6. **Independent verification before the human sees a claim.** A fresh, read-only agent recomputes the headline numbers from raw artifacts and labels each number checked or claimed, as the lineage review did ([README](lineage-review/README.md#state)).

## Open, for the human

- Loved charts: are they in the corpus today, and are they accepted individually like ranked ones, or only as part of the target population?
- How many windows per round the human will judge in the spot-check of Part 1, item 6, and whether played or viewed.
- Part 2: enforced in code (job launcher and Codex hooks on the mac refuse a report without receipts and an unchanged evaluator hash) or as brief and review rules only.
- "End to end" ([d-eval-first](#d-eval-first)): of the agent's reasoning only, or also of the generation model.

## Proposed first work, after review (proposed)

All on the mac, which was down on 2026-10-02.

1. Inventory: which 2 to 6 star ranked and loved 4K charts exist on the mac, with song groups; what of the lineage's instruments is reusable as measurement (star calculator, the 6,924-chart scan corpus, `gameplay_evaluation` observers; [opus/evaluation](lineage-review/opus/evaluation.md) section 9); which rejected outputs can be tied to files. Read-only.
2. Build Part 1 items 1 to 4 and report its false-alarm rate and dose sensitivity, before any generator is evaluated with it.
3. Only then: the first measurement of R1 with real times and of held-out reference under it.
