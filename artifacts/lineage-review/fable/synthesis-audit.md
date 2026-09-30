# Audit of the lineage-review synthesis

Shareable; no human wording. Written 2026-09-30 by an auditing subagent (Claude, control plane) of session `2a66b88f`. Subject: [synthesis.md](../synthesis.md) and the parts of `RESEARCH.md` rewritten from it, checked against the fifteen reports, the baseline code, git history and raw artifacts on bings-mac. Nothing here is a decision; the main thread applies or rejects each item. I edited no other file.

Evidence labels: **report** = read in a reviewer's report; **code** = I read the source in the exported trees or with `git show`; **artifact** = I computed it from raw files on the mac; **git** = commit or note history. My scripts and outputs: `/tmp/lineage-review/fable-audit/` on the mac (`teacher_check.py`, `cj_check.py`, `sourceh_check.py`, `expert_check.py`, `misc_check.py` and their JSON), copies and `baserate.py` in the control-plane scratchpad `fable-audit/`. Disposable locations. No model was loaded; every run took seconds.

**Overall.** The synthesis transcribes most numbers correctly and its findings on R1's interface, the training budgets, the undefined response target and the evaluators hold. It is weakest where it names causes: three headline sentences (the 35M teacher, grid alignment "at chance", the short-LN "signature") reproduce as pooled numbers and do not survive a per-case breakdown, and in each of those and two more places the synthesis took the stronger Opus wording over an Astra qualification that my checks support. Ten corrections follow, nine substantive.

## 1. Corrections, most consequential first

### C1. `s-likelihood-vs-rollout`, and the RESEARCH.md line "better likelihood gave worse free-running charts"

The sentence "the finished 35M-parameter teacher ... has the best held-out likelihood in the project (1.61 to 1.63 against R1's 1.74) and the worst generation on the same 16 cases: same-lane runs of 177 and 354 head rows, star bias +1.1" joins two checkpoints and compares against the wrong R1.

- The 177 run is one chart of 16 at 20M exposures; 354 and the +1.1 star bias are 30M (20M star bias is +0.22; report, opus/r1-foundation F6). At 20M the other 15 charts have a longest run of 15 or less; the median over charts is 6.5 against 6 for released R1 (artifact, S1).
- 30M is past the validation minimum: 1.613 at 20M, 1.635 at 30M, while the train monitor falls 1.690 to 1.663 (artifact, S1).
- The teacher has R1's seed and memory modules and no correction residuals (`row_consequence: none` in its run config). The matching R1 stages are "seed" and "memory", not the released model. On the same 16 cases those small stages already collapse: longest fixed-lane runs 77, 39, 37 (seed stage, 2.33M parameters) and 49, 31 (memory stage, 2.78M); R1's three residuals then bring the maximum to 10 (artifact, S1; the stage numbers are also in astra/01 Q6 and opus/r1-foundation F3). The teacher has no chart with a run of 30 or more at 5M and 10M, one at 20M, three at 30M.

What the data support: a 15 times larger model without the residuals keeps the collapse that small R1 shows before its residuals; scale did not remove it, and the last, over-trained checkpoint was worse. They do not show that better likelihood made charts worse.

Replacement body: "The 35M teacher (R1 with seed and memory, no correction residuals, one seed) reaches 1.61 validation NLL at 20M exposures against 1.74 for released R1, then 1.63 at 30M. On 16 fixed validation cases it has no fixed-lane run of 30 head rows or more at 5M and 10M, one at 20M (177) and three at 30M (354, 85, 65). Small R1 at the same module stage, before its corrections, has three (77, 39, 37) and two (49, 31). Capacity and exposure did not remove the collapse that R1's rule-trained residuals remove."

Replacement for the RESEARCH.md bullet: "In the one scale test on the row half, a 35M model without the correction residuals kept the fixed-lane collapse small R1 shows before its corrections; its last checkpoint, past its validation minimum, was worse (16 charts, one seed)." The key-2 line in section 4 ("the row half drifts on its own histories even when well fitted") should read "more fitting and capacity did not remove the uncorrected drift".

"This is the one place scale was tested" is also wrong: see C6 and omission O1.

### C2. `s-ln-open-state`: "30% to 40% of all LN last 80 ms or less (13% in their sources ...) ... has the signature of that representation"

The pooled numbers reproduce (artifact, S2: 39.5%, 38.7%, 30.7% for inherited, early, fresh; 12.6% in the five sources). The reading does not survive the breakdown.

- One song carries the pool. Operation Zenithfall (210 BPM, 5.87 star source) is 11 of 27 cases, 71% of generated head times and 57% of generated LN (inherited arm). Its quarter beat is 71 to 80 ms, so the 80 ms cut counts one-interval holds there and not in slower songs. By song the share is 51% (Zenithfall), 30% (Max Burning), 27% (Classic Pursuit), 10% (Blizzard Heights), 4.5% (STYX HELIX).
- In beats there is no excess. LN of a quarter beat or less: generated 40%, 36%, 26% (inherited, early, fresh); the five sources 71%, 43%, 16%, 15%, 0%, pooled 32%. Two of the five ranked sources near 4 stars (STYX 122 ms, Blizzard 89 ms) have a median LN of exactly one quarter beat.
- The excess that is unambiguous is at 40 ms: 4 to 5% of generated LN against 0 of 3,479 in the sources.
- "Most of those end exactly at a head time" (74 to 81%, reproduced) has a base rate the synthesis omits: 79 to 82% of all generated LN end at a head time, and 19% to 97% of source LN do, by song (pooled 58%).
- Supplying real head times to a late model of this line did not reduce short holds (S3).
- The one reference generator whose code I read for this represents an LN the same way (C8).

Replacement: "No model attaches a duration at LN birth; the alternative was named on day 1 and never tried. In the final clean joint outputs 4 to 5% of LN last 40 ms or less, against none in the five source charts. The pooled 80 ms share (30 to 40% against 13%) is dominated by one 210 BPM song; counted in beats, holds of a quarter beat or less are as common in the sources (32% pooled, 0 to 71% by song) as in the outputs (26 to 40%). Whether the representation causes the sub-40 ms holds was not tested." Drop "has the signature of that representation" from the title. The closing sentence of section 4 and H5 should present LN representation as a named, untested alternative, not as something the review "puts beside" the keys.

### C3. `s-no-time-coordinate`: "Generated heads of the late models sit at chance on that measure"

True for the pool, not for the songs. Pooled over the 4,096-update outputs: 16.0%, 16.3%, 15.5% within 2 ms of a 1/4, 1/6 or 1/8 position against 16.3% by chance (artifact, S2), matching the reviewer. But Zenithfall is 71% of the pool and sits below chance (14% against 18%). The other four songs are above chance in both arms I broke down by song (inherited, fresh); inherited arm: Blizzard 28% against 13.5%, Classic Pursuit 21.5% against 11%, Max Burning 19% against 15%, STYX 13.5% against 10%; at 5 ms, Blizzard 62% against 34%, Classic 51% against 28%. Sources are 90 to 100%. The heads are far off the grid; they are not at chance on four of five songs.

Dropped with it:

- A training trend. On the seven cases saved at 512, 2,048 and 4,096 updates, the fresh arm moves from chance to about 1.5 times chance on the non-Zenithfall songs (Blizzard 16%, 20%, 22%; Classic 12%, 18%, 18%; STYX 8%, 11%, 13%) and its head F1 at 20 ms against the source rises 0.37, 0.52, 0.62. The inherited arm is flat (0.68, 0.63, 0.64).
- "At a 10 ms tolerance one reviewer finds 75% against 99% on four songs" is the four-song fresh expert on its own training songs, not the clean joint line, and astra/03 warns beside the number that chance coverage is high at that tolerance. I get 39% to 80% by chance per song, about 60% pooled (artifact, `expert_check.py`).

Replacement: "Pooled, 16% of generated heads in the clean joint arms are within 2 ms of a subdivision, the chance rate; the pool is 71% one 210 BPM song. On the other four songs the share is 1.2 to 2.1 times chance (at most 28% against 13.5%); sources are 90 to 100%. The fresh arm's alignment and head F1 were still rising at 4,096 updates. Head F1 at 20 ms is 0.55 to 0.83 against 0.89 to 0.96 between human charts of one song."

"endorsed by the first expert reply" should read "the first expert reply advised against a fixed grid for the first round". astra/08 Q5: the advice "does not forbid an internal learned beat coordinate"; opus/interpretation section 5: the question sent to the expert had already fixed the no-redline design. I could not read the replies.

### C4. `d-locality-counts`: the two counts are on different denominators

"12 of the 13 upstream moves following a human or expert message" is a share of upstream moves. "20 triggered by the agent's own evidence" is a share of all 33 episodes, local ones included. Astra's coding file (artifact, `/tmp/lineage-review/08-codex-interpretation/diagnosis-coding.json`) gives the comparable split:

| Astra coding, 33 episodes | agent evidence only | human | expert | mixed |
| --- | ---: | ---: | ---: | ---: |
| upstream (19) | 7 | 8 | 2 | 2 |
| local (14) | 10 | 3 | 0 | 1 |

On Astra's own coding 10 of 19 upstream episodes follow a human or expert message (12 with the mixed ones) and 10 of 14 local ones are the agent's. The reviewers agree on the association and differ on its size. Replacement text is in section 3a. "I give the Opus count more weight" should go or state a reason; see 3a for why neither number is checkable.

### C5. `s-response-not-in-loop`: "No response scorer was in the system the human played or that qualification scored"

Too broad. The plain session applies a corpus-calibrated recovery preference to every row by default (code: `controlled_audio_continuation/generation.py`, `ControlledSession.__init__(..., recovery_preference=RecoveryPreference(head_pressure=4.))`, `prefer_rows`; `typed_audio_continuation/response_preference.py`). It subtracts up to 4 nats for a same-lane head gap under the ranked first percentile (107 ms at 4 stars), a release-to-head gap under 65 ms, the release of a hold younger than 53 ms, and a head-pressure term, on top of the hard `Recovery` support. It entered on 09-25 (`82436bd`, `764669e`). The demo server builds `ControlledSession(model, mel, duration, controls, seed=..., encoded_audio=..., max_seconds=...)` with those defaults (code: `git show refs/archive/heads/codex/stream-generation-benchmark:src/ensomi_model/serving/runtime.py`, line 179), and the clean joint fit plan records `recovery_preference: default`. opus/player-response lists it in its inventory as "Default native sampling".

Absent were the time-horizon scorers and the planner (code: `qualification.py` lines 229 to 230 and 260; `ResponsePlanner` appears only in `frontier.py`). Replacement title: "No time-horizon response scorer or planner was in the system the human played or that qualification scored; a per-gap rarity preference was." Consequence for [d-key1-vs-key2](../synthesis.md#d-key1-vs-key2): a corpus-rarity response on single gaps, including a penalty on releasing young holds, was in the loop for every late result, and the reported failures occurred under it.

### C6. `s-undertrained` "scale was never a variable"; `d-scale-or-objective` "no evidence either way on the audio side"

astra/05 Q4 lists what was varied, none of it controlled: data at matched budget (121 to 585 charts: held-out NLL better, rollout not); steps on the largest audio fit, read at 2,400 and 6,000 updates on 40 native cases; steps on clean joint. The second is on point and I confirmed it in the lineage document (`typed_audio_continuation.md`, lines 462 to 475): star error 0.553 to 0.959, LN of 40 ms or less 544 of 31,671 to 1,430 of 44,899, "the later fit is not selected despite its lower teacher-forced losses". astra/05 Q7 also has a case in the other direction, omitted: broader source fitting cut the longest same-column run in four Stream cases from 21, 16, 8, 17 to 8, 7, 7, 6.

Replacement: "scale was never a controlled variable", and in `d-scale-or-objective`: "on the audio side the evidence is thin and mixed: the largest fit got worse on its native panel between 2,400 and 6,000 updates (doc, 40 cases, one seed), a broader row fit improved native recurrence on one panel, and the fresh clean joint arm was still improving on timing and likelihood when it stopped."

### C7. `d-key1-vs-key2` and H2 in RESEARCH.md: "key 1 adds nothing to key 2" / "key 1 collapses into key 2"

This contradicts the synthesis itself. `s-likelihood-vs-rollout` says fitting the corpus does not keep free-running output on the corpus distribution; `s-response-not-in-loop` says a selector with a corpus-rarity cost cut that cost by 98% on an unchanged proposal. A corpus-calibrated acceptor at generation time is therefore not redundant with a corpus-trained proposal. The reviewer's question (opus/player-response section 12, item 3) is narrower: whether corpus rarity is acceptable as the meaning of comfortable, demanding, unplayable. Replacement: "If the ranked distribution is both the proposal target and the response reference, key 1 has no target of its own; it would still act as a rollout corrector." In the RESEARCH.md table: "If the reference is corpus rarity, key 1 has no target independent of key 2."

### C8. `s-outside` and the RESEARCH.md bullet: "Working generators differ ... in LN as one object"

astra/09 says the opposite for Mapperatorinator ("not one preselected-duration token") and concludes: "neither requires all raw timing to live on a beat grid, and neither universally represents an LN as an indivisible duration object." I checked (code, `ref-proj/Mapperatorinator/osuT5/osuT5/dataset/osu_parser.py`, lines 642 to 690 and 102 to 105): a hold is a `HOLD_NOTE` event at its start, optional sustain events, and a `HOLD_NOTE_END` event at its end, and events are sorted by time. That is an open hold closed by a later event in an autoregressive stream, the lineage's own form. Mug-Diffusion's body channel is the only one-object case. "Working" is also unestablished: astra/09 found no controlled 4K evaluation for either project, and neither ran here. Replacement: "Reference generators differ from the lineage in scale, in a time grid at output, and in output repair. Mug-Diffusion holds an LN as a body channel; Mapperatorinator as start and end tokens. Their quality on these songs is unknown."

### C9. RESEARCH.md bullet: "the sharper cause is that nothing ran at a scale where a failure could be told from under-training"

In the synthesis this is "The Opus reviewer's reading" (that reviewer's confidence: about 70%). astra/08 calls the hypothesis "partly supported, medium confidence" and names another sharper cause: open questions were recognised and not settled before the next step. RESEARCH.md states the Opus reading as the finding. Replacement: "The search was mostly local and upstream moves mostly followed the human. One reviewer reads under-training as the common cause, the other a gap between recognising a question and settling it."

### C10. Small errors of transcription or scope

| Anchor | As written | Source | Fix |
| --- | --- | --- | --- |
| `s-r1-interface` | "about 71 legal rows of 256 on average at a head row, n = 85,176 states" | astra/01 Q2: 70.8 is the mean over all 85,176 candidate states; at the 82,021 head rows it is 73.3 | "about 73 at a head row (n = 82,021)" |
| `s-r1-quality` title | "its quality was never judged by anyone but the agent" | opus/r1-foundation F4 and Q1: the human saw six reconstructions on 09-21 and asked to publish; whether that was a quality judgment is unknown (it is H7) | "no human judgment of its quality is on record" (RESEARCH.md already says this) |
| H8 | "ranked minima (same-lane head gap 37 ms, release-to-head 25, head-to-release 21)" | census `result.json`: 8 head-to-release gaps of 20 ms or less in one ranked chart (artifact); reports give the minimum as 19 ms; 21 is the lineage's support value | "37, 25 and 19 ms; the lineage's support used 37/25/21" |
| `s-not-ordinary` | "every joint recipe drew a quarter ... from 112 annotated style charts and a quarter from star-by-LN balanced cells" | `train_ranked.py` lines 95 to 97 (code): the largest fit drew 25% annotated, 75% natural; the balanced quarter is the clean joint recipe | "every joint recipe after 09-25 drew a quarter from 112 annotated charts; clean joint drew another quarter from balanced cells" |
| `s-eval-sensitivity` | "reused across 13 to 40 experiment directories" | opus/evaluation F6: 11 to 40 by directory name; astra/07 Q4: 13 to 16 by decoded plan | "11 to 40 by name, 13 to 16 by plan" |
| `s-eval-sensitivity` | "one chart met its whole-song LN ratio ..." | opus/evaluation section 3: four generated charts of one song | "four charts of one song" |
| `s-eval-sensitivity` | "a measured false-alarm rate on ranked charts" | opus/evaluation sections 3 and 11: 158 of the 200 charts were in the references' fitting pool; held out it is 1 of 42 | add "mostly in-sample" |
| `s-undertrained` | "about 30 times slower ... (full-song audio gradients at batch 2), which set the pilot sizes" | opus/proposal-recipe section 9: cause inferred, not profiled | "about 30 times slower; the cause was not profiled" |

### A pattern across C1, C3, C4, C8 and C9

In each the synthesis adopted the Opus wording and dropped an Astra qualification: "capacity, initialization, exposure, training recipe and corrections differ together" (astra/01 Q5), the chance-coverage warning (astra/03), the coding by trigger (astra/08), "not one preselected-duration token" (astra/09), "partly supported" (astra/08). In the four I could check against artifacts or code, the Astra qualification was right. [d-astra-tone](../synthesis.md#d-astra-tone) tells the reader to discount Astra's direction calls; my checks do not support that instruction. The Opus reports contributed more new measurements; the Astra reports were better calibrated about what a measurement shows.

## 2. Omissions

- **O1. The project already trained large, grid-conditioned audio-to-chart models.** astra/02 Q1 and Q2: the pre-V3 stack has a mapper of 106.9M parameters (plus a 15M control encoder; 11,000 to 12,000 steps) and one of 15M plus 15M at 44,000 steps on about 9,200 charts, conditioned on BeatThis grid features, exporting charts from audio. The reviewer's inventory file records the large checkpoint with 121.9M tensor elements, d_model 768, 8 layers (`/tmp/lineage-review/02-pre-v3-history/checkpoint_inventory.json`; I read the file, not the checkpoint). Recorded quality was poor (sparse, repetitive, off its own grid: 11.4% of starts within 5 ms against 97.3% for the reference, doc, one song). The synthesis says only "poor quality, and also off grid". For anyone reading `s-outside` as "scale and a grid are what was missing", this is the project's own counter-example, with the caveats that quality was judged on one or two songs and the output was not snapped.
- **O2. Optimisation.** astra/05, overlooked item 2: every one of the 4,096 logged gradient norms per clean joint arm exceeded the clip of 1 (medians 66, 66, 86; I confirmed from `fit-v1/*/losses.jsonl` and `worker.py` line 81, learning rate 1e-4). Adam dampens the effect of rescaling, so this proves nothing, but "budget used" is not "optimised", and it belongs next to check 1 in section 7 of the synthesis.
- **O3. Ranked charts contain what the human rejected.** Several reports say it and the synthesis folds it into H1 as a list of exclusions: a ranked 4.07 star chart with a 28-attack anchor in about 4 s (opus/player-response section 12 item 4); a source window with eleven LN of 80 ms or less, and "whether the complaints are separable at all by chart statistics" (opus/evaluation section 7); a ranked source with 67% of LN under 100 ms (astra/01 item 6); in the ranked 3.5 to 4.5 band, 10% of LN last 82 ms or less and the song-equal median is 176 ms (astra/06 Q6). My S2 adds that two of the five panel sources have a median LN of one quarter beat. This is a question for the human before key 2 has a target; see 4c.
- **O4. A saved causal substitution on the grid question.** The lineage ran one checkpoint with its own head times and with the source's (`20260928-action-response-frontier-v1/native-guided-v1` and `source-h-v1`). No reviewer used it for [d-grid-cause](../synthesis.md#d-grid-cause). Results in S3.
- **O5. The one case where seed noise was measured and a story retracted** (astra/07 Q4: 16 fresh seeds turned a single-pair LN change of -0.35 into -0.04 with an interval spanning zero). It supports "seed and song variance before promotion margins" in key 3 better than the 0.2 star estimate alone.
- **O6. Human annotation coverage for style controls** (astra/06 Q1): 289 cells on 112 charts; all five prominent Tech cells lie at 3 to 4 stars, prominent LN coordination has 8 cells. Relevant to whether any style control can be learned from this set.
- **O7. Control exposure gaps** (astra/06 Q9 items 2 and 3): the clean joint recipe had no style-known, LN-unknown Stream pieces although the panel requested that combination; progress supervision had zero overlapping LN requests and 28 of 11,398 late queries with a large error. These say the control failures were partly untrained conditions.
- **O8. The deadline atom came back** (astra/03 section 10 item 4): the joint-release mode skips the feasible-wait normalisation; 130 of 1,842 late Stream tails sit at non-terminal deadlines (doc).
- **O9. astra/02 on "V3" naming and the MIR probe numbers.** The synthesis has the probe in one clause; the numbers (0.19 nats, three seeds, 148 test audios) and the absence of any recorded decision to drop it are in astra/02 section 5 item 3.

## 3. The two open disagreements

### 3a. How local was the search

What the sources allow.

- The two codings agree in direction once put on one denominator (C4): local moves were mostly the agent's own, upstream moves mostly followed a human or expert message.
- The local to upstream ratio (27:13 against 14:19) is definitional. Astra codes an episode upstream when the diagnosis reconsidered data, target, ownership or initialisation; Opus codes the action taken and counts repeated repairs separately ("L x8"). Both are internally consistent.
- The trigger counts rest on different rules and neither is checkable as a number. Opus: a human message within about three hours before. Astra: the sources tie the change to the message. By time alone the Opus rule is weak: 121 of the 198 lineage commits of 09-23 to 09-28 (61%) fall inside or within three hours after a message window of the feedback index, and those windows cover 55% of the wall-clock time (git and index, `baserate.py`). Opus clearly also judged topic, which is not recorded.
- Astra's seven agent-only upstream episodes against commit and note times (git): D01 (first sensitivity probe) started 7 minutes after the human approved the plan; D29 (clean joint, fresh arm) was planned in note `e2d1c21` at 03:50Z, 78 minutes after the message that, per the index, allowed restarting modules, and the note does not cite it; D06 (latent intent) is an item from the expert's list that was coded and never fitted; D03 (wait coverage) and D20 (common-prefix outcomes) change training exposure or signal, which Opus's rule calls local; D13 (count factorisation at 22:34Z on 09-24, 9.5 hours after the last message, then typed planning) and D27 (sampling audit, 5.7 to 11.6 hours after the last message) are agent-initiated and upstream on any rule. D13 is the move the human reversed.

Conclusion. In six days the agent started two upstream reconsiderations on its own under a strict reading (ownership, reversed by the human; the sampling measure, which fed the clean recipe) and up to seven under Astra's. Of the structural changes that held, the eight I timed (joint model `09b919c`, H/R split `b5a664c`, row ownership restored `c5b7db8`, player state `7ea3b95`, memory `b130dfb`, joint wait and release `965d670`, clean joint `ef42095`, fresh expert `d6eba23`) were committed inside a message window of the index or within 80 minutes after one. Proposed replacement for `d-locality-counts`: "Put on one denominator, both codings say upstream moves mostly followed a human or expert message (12 of 13; 10 to 12 of 19) and local moves were mostly the agent's (about 25 of 27; 10 of 14). The ratio of local to upstream depends on whether the diagnosis or the action is coded. Neither count is reproducible from the record."

The single check that would settle it: one episode list (Astra's 33, which carry commits), a written codebook (upstream means the action changed factorisation, ownership, base weights, data regime or representation; human-triggered means the design note is the first after an on-topic message, using exact message times), two coders working blind, agreement reported. It needs the exact message times, which are in the private file only the main thread may read. About an hour.

### 3b. Missing time coordinate: cause or symptom

Measured facts that hold: no beat, tempo or subdivision variable after the first pilot (report, both reviewers; code for the loss and sampler read by opus/system-outside-view); generated heads far from the song's grid (S2); human charts on it.

Cause of which complaint.

- Short LN in the fresh expert: yes, through near-duplicate heads. Real head times remove 85% of holds of 40 ms or less (191 to 29; reproduced, `expert_check.py`). That model allowed 1 ms gaps and saw 64 s of chart, so representation and under-training are confounded, as both reviewers say.
- Short LN in the late inherited line: the saved substitution says no (S3). With the source's head times, 90 to 100% on grid, the same checkpoint and seed gave 7.4% of LN at 80 ms or less against 4.4% (STYX), 7.2% against 9.8% (Blizzard), 19.7% against 13.7% (Zenithfall, first 58 s); at 40 ms, 2, 5, 4 against 1, 6, 0. Three cases, one seed, one checkpoint with guidance and gate on.
- Long jacks: no evidence that timing matters. The source-head runs still exhaust the gate on the Stream case (report, astra/04 Q6), and a 3.0 star source became 4.85 stars at unchanged head times (doc, astra/03).
- Irregular subdivision: the measurement is the complaint restated; no human judgment separates a 5 to 20 ms scatter from wrong rhythm.

Symptom of under-training. Partly. The fresh arm's alignment and head F1 rose with training and had not flattened at 4,096 updates (S2). The inherited arm, whose head model descends from the longest fits, is flat at head F1 0.64 to 0.68 and at most 2 times chance on the grid, which looks like a ceiling of this recipe. A recipe ceiling and a representation ceiling are not separated: the head loss has zero tolerance per millisecond, no peak picking, a clip below every gradient norm, and 9% of one pass.

Relation to the two facts in `d-grid-cause`: both stand (the old fitter's 43 to 47 ms phase error, the old mapper off its own grid). Neither is about a learned coordinate.

What would settle each part.

- Whether grid-true head times fix the charts of the final clean joint checkpoints: run `20260928-action-response-frontier-v1/source_h.py`'s substitution on the three 4,096-update checkpoints over the 28 cases and compare holds of 40 ms or less, LN amount error and the qualification failures with the native runs. Inference only, existing script, no training. The three saved cases predict little change.
- Whether the head model's scatter is under-training or representation: needs training. Head times alone, no rows, ranked 3.5 to 4.5 star charts, scored on held-out songs by grid share against chance and head F1 at 5 ms, at three budgets, with and without a beat-phase input (the first pilot's frozen BeatThis features raised head F1 by 0.055 on six songs, report). No inference-free check exists; the trend in S2 is the most the saved files give.

## 4. Entry point

### 4a. The five Position corrections against `5c56e28`

All five are true of the baseline (code unless stated).

1. R and look-ahead: `data.py` builds the timing from every source row time with `roles = attack.any(-1)`, so R is heads plus release-only rows and H the head rows; `features.py` has `LOOKAHEAD = 16`, the next 16 offsets, gaps and roles, `COUNT_SPANS_MS` up to 64,000, remaining R and H counts to the end and the relative position.
2. No audio, no difficulty input: nothing in `model.py`, `features.py` or `generation.py` reads audio, stars or a request.
3. Correction residuals: `r1_restore/harvest.py` has the three rules as stated (28 of 32 with source maximum at most 24; three holds through 12 heads; a head within 30 ms of the same lane's previous head or release); restoration `ledger.json` has pools of 72 queries from 4 groups and 92 from 9 (artifact) and `quality_status: requires_longform_ln_tap_and_local_response_review`. I did not re-read the routing pool count (80 from 8; both reviewers report it).
4. Training: stage updates 5,958 + 661 + 1,317 + 333 + 334 + 328 = 8,931 (report, astra/01 Q6); 6.75M of 10.74M onsets; one model seed. "About 60% verified ranked 2-6 star" rests on one reviewer's join (59.9%; a second gets 64.6% with approved maps).
5. Teacher: finished at 30M exposures (artifact: `segment-00006`, `stop_after_checkpoint: 30000000`, readouts through 30M); `docs/research/vacation_training.md` describes the queue and reports no result; the three agent notes that mention the teacher are dated 09-19 and 09-20, before the run ended, and a grep for its outcomes in them finds only the plan.

Two wording points. "so rhythm, density and the moments where long notes may end come from the source chart": chord size is R1's and follows the source weakly (r = 0.42, report), so write "rhythm, the number of head times, rests and the moments where long notes may end". And the Position text links to `s-likelihood-vs-rollout`, which needs C1.

### 4b. The `time` node

Worth having; not yet well posed. "In what coordinate event times are proposed and judged" covers one of four things the reports separate (astra/03 section 10 item 1): (1) the output coordinate; (2) where tempo and phase come from, an audio inference problem of its own on which the pre-V3 fitter was 43 to 47 ms off; (3) the head objective and decoder (zero-tolerance per-millisecond loss, no peak picking, no refractory rule), which can produce scatter under any coordinate; (4) how timing is scored (source grid, self-fitted lattice, F1 against one chart). Proposed problem text: "How event times are represented, inferred from audio (tempo and phase included) and scored." The last column should carry C3's numbers and S3: "Ranked charts sit on their grid; generated heads do not (pooled at chance, up to 2 times chance by song); real head times did not reduce short holds in the three saved cases; no experiment isolates representation from training." Add `evaluation` to the nodes that need `time`, since timing was unscored after the first pilot.

### 4c. H1 to H10

Neutrality.

- H2 (table row): "key 1 collapses into key 2" is a conclusion and a contestable one (C7).
- H4: "is off-grid expression required for the first target" presents a grid as the price of expression. The node's own wording ("with residual offsets") and the formulation remove that trade. It also presumes the grid matters for chart quality, which S3 does not support for short holds. The second sub-question (does ambient play have the whole track) is already answered in the entry point: the Vision's standing constraint "Complete audio may be used in training and inference" and `notation.md` line 227. Ask it only as a confirmation.
- H5: presumes LN-as-object is what working systems do (C8) and that the short-LN evidence points there (C2). Reword as: "An LN with a duration at birth is an untested alternative; is it worth a test, given incremental release?"
- H3: the sub-question is phrased so that only one answer is reasonable. Neutral form: "Does an LN-amount request apply to the whole song only, or also to where the LN go?"
- H1, H6, H7, H8, H9, H10 are neutrally put. H8 needs the number fix in C10.

Missing, and needed before anything is built.

- **M1. Is the human's standard the ranked corpus, or stricter?** Would the human accept as ordinary the ranked source charts of the panel (two with a median LN of one quarter beat; one window with eleven LN of 80 ms or less) and the ranked 4.07 star chart with a 28-attack anchor? If not, "ordinary" is not corpus membership in any star band, the key-2 target is not the ranked distribution, and every corpus-quantile reference is aimed at the wrong thing. H1 lists possible exclusions; it does not ask this.
- **M2. Order of goals.** `s-controls-order` has no question. Are difficulty, LN-amount and style controls, scoped changes, real-time publication and the demo deferred until an ordinary proposal exists, or required of the first system? opus/interpretation section 8 lists their early request among the instruction-side causes.
- **M3. Compute and wall-clock budget.** H9 asks "small runs or scale" in the abstract. The decision needs: which machine, whether multi-hour or multi-day fits and overnight use are acceptable, whether any GPU is in scope. The reference systems are 157M to 219M parameters; the lineage's throughput set its pilot sizes.
- **M4. The human's time for judging.** H3 assumes a labelled set can be made. How many windows, in what form (played or viewed), is the human willing to judge?

## 5. Spot verifications

Picked because a wrong answer would change what is built first.

**S1. The teacher (scale on the row half; key 2's sufficiency).** `python3 teacher_check.py` on the mac. Inputs: `rows.jsonl` and `condition.json` of the 16 validation cases (8 sources, seeds 17 and 23) under `~/Documents/Pulsefield/research-assets/r1-restoration-20260920-v1/run/{base,seed,memory,routing,release,response}/readout/native/validation/` and `teacher35m-20260920-v2/run/teacher/readout-{1,5,10,20,30}M/native/validation/`; source `.osu` from `dataset/`; `likelihood.json` (`nll_per_onset`, 6,144 onsets). Metric: longest run of consecutive head rows containing one lane, suffix only.

| Model | val NLL | three longest runs over 16 charts | charts with a run of 30 or more | median |
| --- | ---: | --- | ---: | ---: |
| sources | | 19, 19, 10 | 0 | 4 |
| R1 base 4.5M | 1.772 | 12, 11, 8 | 0 | 5.5 |
| R1 seed 5M | 1.765 | 77, 39, 37 | 3 | 6.5 |
| R1 memory 6M | 1.727 | 49, 31, 12 | 2 | 5.5 |
| R1 routing 6.25M | 1.731 | 31, 20, 17 | 1 | 6 |
| R1 release 6.5M | 1.729 | 11, 10, 7 | 0 | 5.5 |
| R1 response 6.75M (released) | 1.739 | 10, 7, 7 | 0 | 6 |
| teacher 5M | 1.746 | 11, 11, 11 | 0 | 6 |
| teacher 10M | 1.644 | 16, 14, 13 | 0 | 9 |
| teacher 20M | 1.613 | 177, 15, 13 | 1 | 6.5 |
| teacher 30M | 1.635 | 354, 85, 65 | 3 | 12 |

Same-lane head gaps under 60 ms: released R1 66, teacher 20M 86, teacher 30M 695. The reviewers' raw numbers reproduce; the comparison they are put to does not hold (C1).

**S2. Grid alignment and LN in the clean joint outputs (the `time` node, H4, H5).** `python3 cj_check.py`. Inputs: `generated.osu` of every case with a reference in `artifacts/joint-audio/20260928-clean-joint-proposal-v1/native-{0512,2048,4096}-{inherited,early,fresh}/` (27 per arm at 4,096; 7 per arm common to all three steps) and the five source charts named in `plan.json`. Grid from the source's uninherited timing points, subdivisions 1/4, 1/6, 1/8; chance from 4,000 uniform times per chart; F1 by greedy one-to-one matching. Results are in C2 and C3. Source alignment within 2 ms: 90.4% (STYX), 99.6% (Zenithfall), 100% (three). `python3 expert_check.py` reproduces the four-song expert counts (holds of 40 ms or less 31, 47, 68, 45 native against 1, 9, 10, 9 with source heads; 152, 80, 154, 77 head gaps of 10 ms or less).

**S3. Real head times on a late checkpoint (does a grid fix short holds).** `python3 sourceh_check.py`. Inputs: `rows.jsonl` of `20260928-action-response-frontier-v1/native-guided-v1/` and `source-h-v1/` (checkpoint `ln-fragmentation-repair-v1/continuation-v1/step-80.pt`, same seeds, controls, guidance and gate; the plan calls it "one causal substitution"), three cases, compared up to the shorter coverage.

| Case | head source | heads on grid (2 ms) | LN | 80 ms or less | 40 ms or less | quarter beat or less |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| STYX s1 | model | 14% | 609 | 27 | 1 | 150 |
| | source chart | 90% | 734 | 54 | 2 | 271 |
| Blizzard s1 | model | 28% | 551 | 54 | 6 | 104 |
| | source chart | 100% | 573 | 41 | 5 | 102 |
| Zenithfall 271201, to 58 s | model | 11% | 357 | 49 | 0 | 35 |
| | source chart | 100% | 441 | 87 | 4 | 65 |

LN share of heads stays far from the request in both (0.69 and 0.72 for a 0.49 request on STYX).

**S4. What was in the generation path (key 1).** Code read: `audio-joint` tree `controlled_audio_continuation/generation.py` lines 25 to 40 and 94 to 104, `sampling.py` lines 13 to 24, `typed_audio_continuation/response_preference.py`, `gameplay_evaluation/qualification.py` lines 228 to 260; `git show refs/archive/heads/codex/stream-generation-benchmark:src/ensomi_model/serving/runtime.py`; `grep -rn ResponsePlanner` over `src`. Result in C5.

**S5. Position corrections.** Code and ledger reads listed in 4a.

Smaller checks: gradient norms and clip (O2); ranked-main sampling branches (C10); census gaps of 20 ms or less (C10; the same file has 8,960 cross-lane head gaps of 20 ms or less in 566 of 8,774 ranked charts, so near-simultaneous heads are not absent from ranked charts); clean joint validation row NLL (fresh 4.33, 3.54, 2.40, 1.95, 1.84; inherited 1.57, 1.65, 1.63, 1.58, 1.61); Mapperatorinator hold tokens (C8); `notation.md` lines 28 to 29 and 227.

## 6. What holds up

- R1's interface and what it reads ahead; the three rules and pool sizes behind the residuals; the training budget; the release manifest's open quality status.
- The release-candidate sensitivity result and its reading (two reviewers, one recomputation; not re-run by me).
- The lineage stopped using R1 within about three hours (git times agree: `ff6471c` 08:19Z, `e2bca3e` 11:05Z).
- The budgets in `s-undertrained`: 4,096 updates at batch 2, the fresh arm still improving, the four-song expert.
- "Ordinary first" was never run; the 112 annotated charts were a quarter of every late recipe.
- No time-horizon scorer or planner in the demo or in qualification.
- The scorers' blind spots and the undefined response target (two reviewers each; the 25 ms zero-cost hold was replayed by astra/04).
- The evaluator findings as stated, with the scope notes in C10. I did not re-run the battery.
- Both verdicts on the leads. `v-timing-defect` "split" is right and S3 strengthens its weak half.
- Section 4's key 1 and key 3 paragraphs, section 6 (worth keeping), section 7 (checks not run; add S3's substitution as the first, cheapest one).
- The pooled numbers quoted from the reviewers: every one I recomputed matched.

## 7. Not checked

- The two expert replies and all human wording (out of bounds); every statement about what the human said or saw rests on the paraphrased index.
- The evaluator battery on the V68 charts (7 of 12), the false-alarm run, the R² of 0.965, the 98% selector result, the 13.7% support exclusion: read in reports only.
- The corpus joins (60 to 65% ranked 2 to 6 star; 132,459 windows behind "6.2%" and "9%"); the arithmetic from those counts is right.
- The release-sensitivity recomputation, the routing pool count, the idle overnight windows, the Mug-Diffusion parameter count.
- Whether the step-80 checkpoint of S3 behaves like the 4,096 clean joint checkpoints; they share the inherited line, not weights.
- My metrics are simple locators written for this audit; one implementation, not cross-checked by a second person. Seed and song variance is not estimated anywhere: S1 is 8 songs, S2 is 5 songs, S3 is 3 cases.
- `opus/controls-ln.md` and `opus/pre-v3-history.md` appeared during this audit and were not read.
