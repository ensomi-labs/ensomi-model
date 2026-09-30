# R1 foundation: independent review (Opus, control plane)

Written 2026-09-30 by a Claude reviewer (session `2a66b88f`, subagent), independent of the Astra
`lr-*` jobs: nothing under `artifacts/reports/lineage-review/`, `.sync/*/jobs/*lr-*`, the other
files in this `opus/` directory, `artifacts/private/` or `~/.codex/` was read.

Question: the `codex/audio-skeleton` lineage assumed that R1 is a sound row model that only lacks
event times from audio. Is that true, and what did the lineage take for granted about R1?

Evidence grades: **[code]** checked against source at `5c56e28` (or `audio-joint-2026-09` where
stated); **[artifact]** measured or read from raw files on bings-mac; **[doc]** a claim in a
lineage or baseline document, not re-checked. "Source" means the real chart whose times R1 was given.

## 1. What was read and measured

Read: `RESEARCH.md`, `lineage-review/README.md`, the feedback index (sections 2, 3, 5);
`docs/research/bounded_typed_continuation.md` (all), `r1_staged_restoration.md`,
`r1_training_distribution.md`, `vacation_training.md` at `main`; lineage docs
`r1_transfer_stability_audit.md`, `audio_conditioned_choreography.md` (parts),
`audio_joint_expert_question.md` (parts), `ranked_2to6_action_reference.md` (parts); agent note
`2026-09-23-audio-skeleton-r1-integration.md` (first 330 of 16,499 lines); code
`features.py`, `response.py`, `r1_restore/harvest.py`, `joint_audio_continuation/model.py`
(`initialize_from_r1`).

Measured (read-only, CPU, a few minutes each). Scripts are in the control-plane scratchpad
`opus-r1-foundation/` and in `/tmp/lineage-review/opus-r1-foundation/` on the mac, with outputs
`measure.json`, `stages.json`, `teacher.json`, `ranked_ref.json`:

| Set | What | Size |
| --- | --- | --- |
| A | Released R1 (response 6.75M, sha `4b3ec156`) on its restoration VAL screen, against the source `.osu` | 8 sources x seeds 17/23 = 16 charts, 26,400 suffix rows |
| B | The same 16 cases at each of the six restoration stages | 6 x 16 charts |
| C | The six real reconstructions the human was shown on 2026-09-21 (`artifacts/r1-real-reconstruction-20260921`), sources found by exact match of the event-time union | 6 songs (alone, odin, descent, epistrofi, lilith, slash-dot-slash), seed 17 |
| D | The 35M teacher (`~/Documents/Pulsefield/research-assets/teacher35m-20260920-v2`) at 1M/5M/10M/20M/30M exposures on the same 16 cases, plus both models' fixed likelihood readouts | 5 x 16 charts; 6,144 VAL + 6,144 TRAIN onsets |
| E | Frozen R1 with extra release-only candidates (lineage day 1, `artifacts/audio-skeleton/20260923-ms-sensitivity`) | 12 VAL charts x 3 conditions x 2 seeds = 72 |
| F | Ranked 2-6 star reference sample from the lineage's byte-verified ranked list | 319 charts, about 80 per integer star band 2, 3, 4, 5 |
| G | R1 training census joined to per-set osu! metadata (`dataset/0/*/metadata.json`) | all 11,563 TRAIN charts, 3,169 groups |

Metrics are simple and mine: rows and heads after the seed; same-lane head-to-head and
release-to-head gaps under 40/60 ms; LN duration in beats from the source's uninherited timing
points; releases followed by a same-lane head within a quarter beat; longest run of consecutive
head rows that all contain one lane; star rating with the repository's
`compute_mania_star_rating_20241007` (whole chart, 1.0x). They are locators, not quality labels.

## 2. Findings

### F1. The supplied times fix the rhythm, the density and most of the difficulty; R1 chooses lanes, chord sizes, TAP/LN and which release happens where. [code][artifact]

- R is the union of all source event times, including pure-release moments; H marks the rows that
  must carry a head [doc: `bounded_typed_continuation.md` "Conditions"]. Every head time and the
  number of head rows are therefore given (set A: generated rows = source rows in all 22 charts).
- R1 also reads, at every step, the next 16 candidate offsets, gaps and roles (head or
  release-only), counts of R and H within 0.25/1/4/16/64 s, the remaining H count and the relative
  position in the song [code: `features.py`, `TimingView.queries`, `LOOKAHEAD=16`,
  `COUNT_SPANS_MS`]. It sees where the next pure releases are before deciding which lane to hold.
- In the source, every release-only candidate carries a release: "original R contains no
  all-empty source rows" [doc]. R1 has never seen a release-only moment without a release.
  Generated charts use 86-100% of the supplied release-only moments (set A/C) [artifact].
- Chord density is R1's and tracks the source weakly: heads per row, source vs generated,
  correlation 0.42 over 22 charts; a 3.18-star Hard (Youma Yakou) goes from 1.22 to 1.54-1.59 heads
  per row and from 0% to 7-13% rows with three or more heads [artifact].
- Stars: source vs generated correlate 0.91 (rhythm dominates the rating), mean shift +0.31, mean
  absolute 0.52, largest +1.4 (3.18 to 4.56; 4.75 to 5.85; 2.27 to 3.19) [artifact, sets A+C].

So what R1 was shown to do is fill lane assignments into a real mapper's rhythm, with the real
mapper's LN tail moments marked. The human's "4-star" judgements concern decisions (density,
subdivision, where LNs end, breathing) that in the R1 setting came from the source chart.

### F2. R1's LN behaviour is driven by the release candidates it is offered. The lineage measured this on its first afternoon. [artifact][doc]

Set E, recomputed from the raw `generated.osu` files (whole chart, 24 charts per condition):

| Release-only candidates added | Mean LN share of heads | Median of per-chart median LN length |
| --- | ---: | ---: |
| none (real R) | 0.21 | 230 ms |
| midpoint of every 4th gap | 0.46 | 164 ms |
| midpoint of every gap | 0.86 | 82 ms |

Consistent with the agent note's own numbers (duration ratio 0.69/0.375, LN fraction +0.26/+0.69)
[doc]. The same note records that release-only times are close to arbitrary between human mappers
of the same audio: release-only F1 at 20 ms between alternative human arrangements 0.092 (0.169 at
similar density), versus 0.785 for head times; the audio timing pilot reached release-only F1 0.035
to 0.051, and 0 with default controls [doc, 584 same-audio pairs; pilot 770k params, 1,200
updates, 48 songs, one seed].

Consequence: the R1 interface places the most mapper-specific LN information (when tails end) on
the input side. Any upstream timing model that emits extra or misplaced release-only moments will
make R1 produce more and shorter LNs. This is a mechanism for problem B (fragmented short LN)
and for the recurring dispute over whether R or R1 owns releases (problem F) that exists before any
audio model is trained. The agent read it as a reason to learn sparse release opportunities and
adapt R1, not as evidence against the factorisation.

### F3. With real times, R1 already shows a milder form of several later complaints. [artifact]

Pooled set A (16 charts) and the 8 sources, per 1,000 suffix rows unless stated:

| Metric | Source | R1 base 4.5M | R1 release 6.5M | R1 response 6.75M |
| --- | ---: | ---: | ---: | ---: |
| Same-lane head-head under 40 / 60 ms | 0 / 0 | 0.27 / 1.63 | 0.11 / 1.93 | 0.23 / 2.50 |
| Release-to-head under 40 / 60 ms | 0.15 / 13.1 | 14.2 / 67.0 | 3.8 / 25.2 | 0.76 / 8.3 |
| Longest fixed-lane head-row run (max over charts) | 19 | 12 | 11 | 10 |
| LN share of heads | 0.19 | 0.50 | 0.29 | 0.20 |
| Mean abs. LN-share error per chart | 0 | 0.31 | | 0.07 |
| Star MAD / bias vs source | 0 | 0.49 / +0.31 | 0.59 / +0.37 | 0.58 / +0.35 |

Per complaint:

- **A, long jacks and sub-40 ms attacks.** Mostly absent at 2-5 star sources in the final model
  (max run 10, head-head under 40 ms 0.23 per 1k rows). Present in dense sources: descent (6.38
  stars) release-to-head under 40 ms 2 in source vs 144 generated, head-head under 60 ms 4 vs 46;
  tanasinn (5.77) head-head under 60 ms 0 vs 28-30. Intermediate stages show the collapse: fixed-lane
  runs of 77 (seed stage) and 49 (memory stage), removed by the routing and release corrections.
- **B, fragmented short LN, release right before the next head.** Present with real times on a
  TAP-leaning source: Youma Yakou [Hard] 3.18 stars, LNs at or below a quarter beat 0.7% in the
  source, 45-51% generated; release-to-same-lane-head within a quarter beat 7 vs 103-130 per 1k
  rows. Present from the base stage (23-32%) and not reduced by any correction. In the 09-21
  reconstructions: odin (0 LN in source) gets 29 LNs, 65% of them at or below a quarter beat;
  alone's LN median falls from 4 beats to 0.5. Fairness: set F shows short LNs are common in
  ranked charts (share at or below a quarter beat: median 0.23 at 4 stars, 90th percentile 0.57-0.62
  at 3-5 stars), so the defect is a style imposed where the rhythm and the source do not call for
  it, not short LNs as such. This matches the human's own wording in B ("a style, not a baseline").
- **C, no ordinary pattern / D, no breathing.** Not measurable with these locators. Chord
  inflation on easy sources and LN style drift (above) are the only measured signs.
- **Difficulty.** No input for it; outputs drift up to +1.4 stars from the source.

Seed variance is large before the last stage (base: LN share differs by up to 0.54 between seeds
17 and 23 on the same source) and small in the final model (mean 0.03) [artifact].

### F4. The evidence of quality R1 ever had is mechanical plus unblinded agent inspection. [doc][artifact]

Established: legality, exact replay, export/reparse, byte-for-byte reproduction [doc; not
re-run]. Everything about arrangement quality in `bounded_typed_continuation.md` is "machine
judgments calibrated to human examples", sealed at best after numeric summaries were seen, on
scopes chosen by the agent (16-60 scopes per comparison). No blind human comparison, no player
trial, no independent test set; TEST "remains unread" [doc]. The restoration ledger ends with
`quality_status: requires_longform_ln_tap_and_local_response_review` [artifact]. The restoration
doc states that completed computation does not establish that historical quality was restored
[doc]. The one human look recorded is the 09-21 reconstructions (index V01-V07), after which the
human asked for release and tagging; whether that was a quality judgement is not in the
shareable record (question Q1). Of those six reconstructions only one source is in the 2-4.5 star
target range (lilith, 4.16); alone is 1.63 and the other four are 4.7-6.4 stars.

Every lineage-era "R1 improved X" claim I found is a count on 16-48 outputs from 8-24 source
groups with two or three seeds, with reused development groups [doc]. The 4M to 4.5M continuation
with no model change moved native LN share from 18% to 49% on the same 28 groups [doc], which
means native behaviour is not pinned by the training loss and small comparisons are weak.

### F5. The "machine preferences" are three hand-written rules, fitted on 244 queries, validated by agent inspection only. [code][artifact]

| Stage | Rule that defines a bad sampled row | Pool | Validation |
| --- | --- | --- | --- |
| Head routing | a lane present in at least 28 of the previous 32 head rows, while the source's max is at most 24 | 80 queries, 8 groups | agent Lens inspection |
| Release routing | three holds unchanged through 12 heads with all attacks on the remaining lane | 72 queries, 4 groups | agent Lens inspection |
| Response (`frontier2`) | a head within 30 ms of the same lane's previous head or release, over the current row and the next two H, with optimistic futures | 92 queries, 9 groups | "all nine contexts inspected" by the agent |

[code: `r1_restore/harvest.py` lines 33-45 and 99, `response.py`; artifact: ledger pool counts
80/8, 72/4, 92/9]. Each residual is a zero-initialised scorer trained with the base frozen, source
CE plus preference loss at weight 0.25, 250k exposures (about 330 updates, under 4 minutes each)
[artifact: stage results]. Nothing links these rules to human judgement except that the source
chart at the same place does not contain the pattern. The response rule also rewards releasing
early (optimistic earliest release lowers the 30 ms cost); in set B the share of LNs at or below a
quarter beat rises from 0.27 (release) to 0.33 (response) while release-to-head under 60 ms falls
from 25 to 8 per 1k rows. That is one plausible route from a 30 ms rule to shorter LNs; with 16
charts it is suggestive, not shown.

`frontier2` is therefore a 27,648-parameter residual trained to avoid sub-30 ms same-lane gaps. The
lineage later treated "frontier" as the carrier of player response (index V36, key 1). The code
does not support that reading; the lineage's own transfer audit says so too [doc].

### F6. Capacity and exposure on the same objective improve likelihood and make free-running generation worse. [artifact]

| Model | Params | Exposures | VAL NLL/onset | TRAIN NLL/onset | Max fixed-lane run | Star MAD / bias | Head-head < 60 ms per 1k rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 base | 2.28M | 4.5M | 1.772 | 1.949 | 12 | 0.49 / +0.31 | 1.6 |
| R1 response | 3.08M | 6.75M | 1.739 | 1.928 | 10 | 0.58 / +0.35 | 2.5 |
| 35M teacher | 35.2M | 5M | 1.746 | 1.915 | 11 | 0.66 / +0.48 | 1.4 |
| 35M teacher | 35.2M | 20M | 1.613 | 1.690 | 177 | 0.66 / +0.22 | 3.3 |
| 35M teacher | 35.2M | 30M | 1.635 | 1.663 | 354 | 1.10 / +1.09 | 26.3 |

Same 16 cases, same fixed likelihood windows, native temperature 1, one run per model (one training
seed each). At 30M the teacher has three charts with fixed-lane runs of 65, 85 and 354 head rows
and one chart +3.3 stars over its source. The teacher has no correction residuals, so the
comparison is teacher vs R1 base plus corrections, not a clean capacity ablation. Still: the best
likelihood model in the project is the worst generator on this screen, and VAL NLL bottoms at 20M.
Teacher-forced likelihood on R/H-conditioned rows does not select for playable closed-loop output;
what keeps R1 usable are the rule-trained corrections. I found no committed document or note that
reports these 35M results, although the human asked about the finished 35M run on 09-23 (index V08)
and the lineage then started its audio work the same hour.

### F7. The corpus is mostly ranked, but not "ordinary ranked 2-6 star", and R1 has no difficulty input. [artifact]

Of 11,563 TRAIN charts (expected-draw share in brackets): byte-verified ranked 2-6 star 59.9%
(62.5%); ranked below 2 stars 20.2% (17.9%); loved 2-6 star 8.3% (8.2%); ranked with bytes
differing from the official checksum 7.2% (7.9%); loved above 6 or below 2 stars 2.3%; loved with
differing bytes 0.7%; ranked above 6 stars 0.4%; no metadata 0.7%; wip/pending/graveyard 0.3%. Per-beatmap status and stars from `dataset/0/<set>/metadata.json`
matched by MD5 [artifact]. So about 70% of exposure is ranked or loved 2-6 star; the rest is mainly
easy charts. The model is not told which difficulty it is producing; it infers it from rhythm density.

Scale: 6.75M onset exposures against 10.74M available post-seed onsets, about 0.6 passes, 32%
of onsets seen by 5M [doc: census]; 8,931 optimizer updates of four windows, 3,379 s total compute
on one CPU thread for all six stages [artifact: stage `result.json`]. R1 is a one-seed,
sub-epoch, one-hour CPU fit. Nothing in the record tests whether its failures are capacity, data
or objective; F6 is the only scale evidence and it points at the objective.

### F8. The lineage did not keep R1. By the evening of day 1 "R1" was a different model. [code][doc]

The released R1 was run frozen behind learned timing for about three hours on 09-23 (commits
`ff6471c` 08:19Z to `068988e` 08:52Z; joint training from `e2bca3e` 11:05Z). The joint model then
copied 2.44M of R1's parameters and dropped the seed residual, landmark memory, `frontier2`,
the exact-projection columns carrying future candidate roles, offsets, schedule counts and song
position (1435 to 529 inputs) [code: `joint_audio_continuation/model.py`,
`initialize_from_r1`; doc: transfer audit]. It generated from BOS without a real seed and was
trained first on 48 song groups / 121 arrangements, later 615 arrangements / 240 songs [doc], while
audio exists for all 14,689 indexed charts [artifact]. The lineage's own docs say this "does not
preserve the released R1 policy". Later attributions of failures to "R1" (for example the 09-28
finding that short LN tails came from "R1's choice at H moments") concern this descendant decoder.

So the assumption was never actually tested in the form stated. What was tested on day 1 (F2)
showed that frozen R1 depends on exactly the information audio does not supply; the response was to
replace R1's information path while keeping its name and weights.

## 3. Judgment on the assumption

**Case for "R1 is a sound row model that lacks only times".** With real times, the final R1 is
legal, stable across seeds, close to the source in pooled LN share (0.20 vs 0.19) and heads per row
(1.48 vs 1.51), and rarely produces sub-40 ms attacks or long fixed-lane runs at 2-5 star sources.
Its star ratings follow the source (r = 0.91). Most of the human's 09-26 to 09-28 complaints were made
about descendant models with generated times, not about R1 with real times. The exact contract,
replay and export machinery is well tested.

**Case against.** (1) The good pooled numbers are what real mapper rhythm plus marked release
moments give almost for free; R1 decides lanes, chord size, TAP/LN and release placement among
given moments, and that is where its errors are (chord inflation, up to +1.4 stars, LN style
imposed on a TAP source). (2) Its LN decisions are dictated by the release candidates (F2): R1 is
not a row model plus missing times; it is a row model whose inputs carry part of the arrangement.
(3) Its avoidance of long jacks and sub-30 ms gaps comes from rule-trained residuals fitted on 244
queries (F5), and the only scale experiment shows the base objective degrading with more capacity
and exposure (F6). (4) No human-validated quality evidence exists for it (F4). (5) Its training
target is a mix that is 70% ranked or loved 2-6 star with no difficulty signal (F7).

**Verdict.** The assumption is false in its strong form, confidence about 80%. R1 is a reasonable
*reconstruction-given-mapper-rhythm* model, not a row model awaiting times. Its interface puts
mapper-specific information (release-only moments, whole-schedule density and position) on the
input side, so replacing that input with audio predictions was bound to shift it off distribution,
and its visible good behaviour rests partly on hand rules. Where I am least sure: whether the
milder defects I measured (B on one of eight sources, A only at 5.8+ stars) would be judged
serious by the human at 2-4.5 stars. The sample is 8 sources, two seeds, one training seed.

## 4. Overlooked or never questioned

1. The information split of R/H: that release-only times are an input. The lineage measured the
   consequence on day 1 (F2) and the cross-mapper arbitrariness of release times (0.092 F1) and
   still kept a timing module that emits release moments (R) feeding a row model.
2. Whether R1's defects are objective-bound. The 35M result (F6) answered it and was not used.
3. That "frontier/response" was a 30 ms rule, not a player-response model (F5); the name carried
   meaning into keys the code does not have.
4. Name continuity: "R1" after 09-23 is a different model on about 1-5% of the corpus (F8), so
   "R1 is responsible" findings do not transfer to the baseline.
5. Difficulty as an input. R1 infers it from rhythm density; with generated rhythm there is nothing
   to infer it from. The human asked for difficulty control on 09-25 (V39).
6. That R1's quality was never measured against ranked charts of the same star band at the level of
   arrangement, only against its own source with agent-chosen scopes.
7. Small-sample instability as a reading hazard: 500k extra exposures moved native LN share from
   18% to 49% [doc]; comparisons of 16-48 outputs from one training seed are within that noise.

## 5. Worth keeping

- The contract, replay, support mask, exact state, verification and export (`contract.py`,
  `support.py`, `verification.py`, `osu_core/export.py`): solid and reusable whatever proposes rows.
- Released R1 as a *conditional reference*: given a real chart's times, it produces a legal,
  seed-stable arrangement near the source in pooled statistics. Useful as a baseline for any new
  proposal and as a probe of how much an arrangement is fixed by its rhythm.
- The staged restoration runner and its readouts, which made this review measurable.
- The measurement that release-only moments are mapper-specific (F2) and the sensitivity result:
  a design constraint for any timing/row split.
- The 35M teacher checkpoint and readouts as a scale data point (F6).
- The ranked 2-6 star byte-verified list (8,774 charts) and per-set metadata, for a corpus-grounded
  reference (set F shows how cheap per-band reference distributions are to build).

## 6. Inconsistencies and questions only the human can settle

- **Q1.** On 09-21 the human saw the six reconstructions and asked to publish. Was that approval of
  quality or of the restoration procedure? Five of the six sources are above 4.1 stars; descent
  shows 144 release-to-head gaps under 40 ms against 2 in the source; alone's LNs shrink from 4 to
  0.5 beats.
- **Q2.** Key 2 asks for "ordinary ranked 2-6 star first". R1's corpus is already about 70% ranked
  or loved 2-6 star plus 20% easier charts. Is the wanted change a narrower corpus, a difficulty
  input, or a different notion of ordinary (arrangement-level) than corpus membership?
- **Q3.** Is the RESEARCH.md position "R1 answers the half that chooses rows at supplied times"
  still acceptable given F1-F2, where the supplied times include release moments and future roles?
- **Q4.** Was the 35M teacher's native result reported to the human in chat on 09-23? It is not in
  any committed doc or note I found (grep of main, audio-joint, agent-notes and tagged relay notes).
- **Q5.** RESEARCH.md says the restored R1 was trained "to 6.75M source-onset exposures on one CPU
  thread" and lists what is established; it does not say that this is under one pass over the
  corpus and one hour of compute. Should the position state that?

## 7. Not verified, and how to follow up

- Blinded human judgement of R1 vs source at 2-4.5 stars: nothing exists. Smallest step: render
  set A's 2-4.5 star pairs (`.../response/readout/beatmap-lens-svg` exists for the response stage).
- Larger R1 screen: 8 sources is thin. The 28-group `difficulty-ln-longform` and 24-group cohorts
  were lost with `artifacts/bounded-typed-continuation/` (absent on the mac). Rebuild: condition
  files via `condition_hydra` from ranked 2-4.5 star VAL sources, then
  `generate_hydra checkpoint_file=<released> ... device=cpu cpu_threads=1 seed=17|23`
  (`bounded_typed_continuation.md`, "Portable condition and generation commands"), and rerun
  `/tmp/lineage-review/opus-r1-foundation/measure.py`'s `compare()` on each pair.
- R1 with timing from a different human chart of the same audio (same-audio pairs from the
  lineage's distribution audit): would separate "needs real times" from "needs this mapper's times".
  Not run; it needs generation, which this review did not do.
- Whether the response rule causes shorter LNs (F5): needs response vs release on a larger cohort.
- The 35M teacher's generations were read at one seed pair per case; its training log
  (`teacher35m-20260920-v2/run/teacher/segment-*/training.jsonl`) was not read.
- Historical (deleted) R1 vs restored R1: not comparable; historical outputs are gone.
- Star calculator: the repository's 20241007 port; not cross-checked against the official server
  here (the lineage reports a 0.000004-star match on 17 charts [doc]).
- I did not read the lineage's 94 research docs beyond those listed, nor the 16,499-line agent note
  past line 330.
