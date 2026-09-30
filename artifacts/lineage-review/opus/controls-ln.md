# Controls and long notes in the audio-joint lineage: independent review

Reviewer: Claude Opus 5.5, control plane, 2026-09-30. Slice `06-controls-ln`. Written without reading the Astra, other Opus or synthesis reports.

Evidence grades: **doc** = a lineage document or commit says so; **code** = I read the code at tag `audio-joint-2026-09`; **artifact** = I recomputed it from saved files on bings-mac. All recomputation used my own stdlib `.osu` parser (`osu_stats.py`), not the lineage's parser. Where the lineage's own star code was needed (`osu_core/difficulty.py`, `ScopeStrainTrace`), I imported it read-only from `/tmp/lineage-review/trees/audio-joint/src`. Scripts and JSON outputs: mac `/tmp/lineage-review/opus-controls-ln/`, copies in the control-plane scratchpad `opus-controls-ln/`. No model was run.

## 1. What I read and checked

- Notes entry: `RESEARCH.md`, `lineage-review/README.md`, feedback index sections 2, 3 and 5.
- Pre-lineage (`main`): `scoped_style_probe_postmortem.md`, the start of `scoped_style_seed17_results_and_next_questions.md`.
- Lineage docs (`audio-joint`): `shared_arrangement_profiles`, `controlled_audio_continuation`, `typed_audio_continuation` (control, scoped-difficulty and feedback sections), `control_condition_learning`, `paired_scope_control_learning`, `balanced_condition_alignment`, `optional_control_training`, `action_segment_r1`, `ordinary_fourstar_rhythm_and_holds_zh`, `joint_r1_release_decisions` (attribution table), `r1_short_hold_acceptance_zh` (start). The openings only of `scoped_ln_allocation`, `contextual_ln_count_learning`, `ln_feedback_scope_ablation`, `ln_release_calibration`, `release_support_joint_learning`, `active_ln_audio_cues`, and the support-filter section of `native_pattern_failure_analysis_zh`. Agent note `2026-09-28-rh-r1-fragmentation-repair.md`.
- Code: `typed_audio_continuation/controls.py`, `difficulty_targets.py`, `controlled_audio_continuation/allocation.py` (`LnAmountFeedback`), `planned_audio_continuation/spacing.py` (release deadline), `gameplay_evaluation/ln_fragmentation.py`, `head_difficulty.py`, `joint_audio_continuation/evaluation.py`, the 09-28 `release_ownership.py` on the mac.
- Artifacts (mac, read-only): 4,037 `generated.osu` files dated 09-24..09-28 that have a sibling `result.json` (4,763 control ranges); the 1,973-chart ranked 3.5-4.5★ cohort from `20260928-coordination-corpus-v1/scan-v1/charts.jsonl`; `20260928-action-segment-r1-v1` (pilot steps, plans, six charts); `20260928-clean-joint-proposal-v1` (release-ownership output, 132 charts).
- Git: 202 commits `r1-restored-6.75m..audio-joint-2026-09`. By a crude keyword count on subjects, 104 mention controls or LN (55 controls only, 27 LN only, 22 both).

## 2. Where the control labels come from

| Control | Training label | Granularity | Grade |
| --- | --- | --- | --- |
| Difficulty | Whole-chart star rating from the repository's port of the osu!mania "20241007" algorithm (`compute_mania_star_rating_20241007`, multiplier .018). By default `source_schedule` gives every 16 s training scope the whole-chart value. Optional since 09-25: `ScopeStrainTrace.level`, a local proxy made from the same strain peaks per 400 ms cell, renormalised for short scopes, with real LN tails (offline only). | Whole chart by default; 16 s scopes with the proxy | code |
| LN amount | `LN heads / all heads` among heads inside each 16 s scope, computed from the source rows. A hold that enters the scope does not count. Durations and occupancy are not in it. | 16 s grid with random offset | code |
| Style | The pre-lineage human scoped-style cells: five concepts (jack, stream, trill, tech, LN coordination), absent/supporting/prominent, native spans. 11 TRAIN charts in the first small corpus; 289 cells on 112 charts in the broad one. After exclusions, prominent cells for fitting were 10 jack, 20 stream, 9 trill, **3 tech, 5 LN coordination**. | Annotated spans | doc (counts), code (wiring) |
| Arrangement profile (09-24 only) | 16 weighted medoids of three whole-chart descriptors: H rows/s, heads per H, LN-head fraction | Whole chart | doc |

Two provenance details matter. The ranked TRAIN manifest (`typed-contract-repair-v1/pairs.json`, 6,924 charts) filters on `metadata_stars`, the value stored with each chart, while the 09-28 corpus studies filter on stars recomputed with the port. I did not check the port against the official calculator. The docs call it "official 20241007". It predates the 2025 osu!mania rework and has known weaknesses on LN.

**Can whole-chart labels teach a scoped control?** Only for a different meaning. Trained this way, the model learns what a passage looks like somewhere inside a chart rated X. That is a real, learnable conditional, but it does not say how hard this passage is. The lineage's own census puts the gap on record (doc, `typed_audio_continuation.md`, "Bounded scoped-target result"): over 2,940 sixteen-second TRAIN scopes, the local proxy minus the chart label has median −0.54, and 24.3% of scopes sit more than one star below their label. So a request for "5★ over these 32 s" asks the model for an average passage of a 5★ chart, and it will under-deliver. Worse, ranked charts never switch difficulty mid-song, so a mid-song switch has no training example at all: its transitions are out of distribution. LN fraction, by contrast, is labelled locally and is identifiable from data. Style is local but far too sparse (3 to 20 prominent cells per concept).

**What the scoped-style postmortem had already said (09-14, before the lineage).** Even a classifier built for this task, trained on 382 human plus 2,818 machine cells and validated on 61 human cells from 18 groups, missed every human Trill positive and every LN-coordination positive at its threshold. Tech had one validation positive, and the combined architecture did not beat the original. The postmortem's own next step was to establish fit and exposure before any larger grid. The lineage then used the same human cells to condition a generator, with no working style readout to check the output against. Style control was unidentifiable from the labels and could not be evaluated from the start. The docs admit it ("Tech inconclusive", "weak immediate differences"). As far as I found, nobody said explicitly that the 09-14 result made this predictable.

## 3. Requested versus achieved

### 3.1 Per attempt (the lineage's own numbers: doc unless marked)

| Date, commit | Attempt | Size | Result as reported |
| --- | --- | --- | --- |
| 09-24 `10ddaa8` | Shared chart-level profiles (3 descriptors, 16 medoids) | 1,200 updates, 615 charts; 54 outputs, 9 audio/seed cases, 5 audios | Descriptor error −42.8%. LN-heavy request .734 realised .036 to .282. H-rate requests leak into chord width (.49) |
| 09-25 `ca3dc65`, `27b9526` | Typed resource plan with scoped stars/LN/style; skeleton fixes TAP/LN counts | 1,200 to 6,000 updates, 614 to 6,923 charts; 15 to 40 native cases on 3 to 8 audios | Star error .74 to .59 at 800 updates, **.96 at 6,000** (more training, worse control). Switched 70% LN scopes realised 78.5 to 82.5% |
| 09-25 `8687c35` | Scoped strain-proxy labels | +1,000 updates | Star error .488 → **1.644**; ≤40 ms LN 2.05% → 9.38% |
| 09-25 `466d470`, `c5b7db8`, `298c17c` | Inference-time LN amount feedback (typed counter, then R1 integral controller: gain 1/8, clipped ±2 log-odds, per new LN) | No training; 12 static cases, 3 audios × 3 seeds | LN error .078 → .038 (typed); .0017/.0089 (R1 integral). Star error unchanged at 1.06 and 1.44 |
| 09-26 `2c2b8af` | Paired low/high outcome learning on 12,288 difficulty weights | 96 updates, 12 charts; 60 continuations, 5 prefixes × 3 seeds | Error .809 → .756 (target −.20); high-minus-low .16 → .25 (target .40). **Fail** |
| 09-26 `2c2b8af` | Condition audit on 12 real identical-H pairs | 24 real scopes | R1 preferred the *lower* difficulty on 22/24 real histories |
| 09-26 `b2a5351`, `6520ba1` | Balanced exposure plus condition-contrastive alignment | 1,200 updates, 703 charts; 30 continuations per arm | Real-history ranking 14/24 → 20/24; generated error .784 → .716; gap .313. **Fail** |
| 09-27 `1434924` | Contextual LN-count readout | 256 windows | NLL better; native regressions |
| 09-27 `24061b9` | Controller on/off ablation | 36 generations | "Disabling feedback makes LN usage more extreme" |
| 09-27/28 `c0db49c` | Learned scoped LN allocation MLP | 6 native cases | Error .154 → .099; matched no-information arm .115; gain .016 < .02. Live override poor |
| 09-28 `1658123`, 23 | Style-known / LN-hidden training views | 14 factual intervals | Hiding LN moves expected LN share by +.26 points (median). Zero Stream-prominent pieces in the recipe had LN unobserved |
| 09-28 `7d31b1e` | Action-segment R1, K=1 vs K=4 latent plans | **32 updates, 146 s**, 158 segments; 6 source-H charts, 1 seed | LN .25 to .30 whatever the request; K4 picked code 1 in all 85 plans |

### 3.2 Lineage-wide census from saved charts (artifact)

Command: `census.py` over every `generated.osu` with a `result.json` carrying `controls` (4,037 files). Ranges are resolved as in `ControlSchedule.resolved_ranges`, keeping those with ≥32 heads. "Free-running" means `head_source` is `native` or `generated-audio`, i.e. H from audio rather than source H or a fixed prefix.

**LN amount.** Over all 4,220 ranges with an LN request (1,944 charts, 48 experiments), requested and realised shares correlate at .96; 82% are within ±.10. That overstates it: fixed-prefix and source-H experiments dominate, and requests below .15 mostly realise 0. On free-running generation (491 ranges, 362 charts), response is ordinal but amplified, and it holds only while the hand-built controller is on. The cleanest matched comparison is `20260927-controller-semantics-v1`: same weights, same seeds, 11 ranges per arm on the fixed panel.

| Arm | Within ±.10 | Examples, requested → realised |
| --- | --- | --- |
| actor, feedback off | 5/11 | .49 → .94, .93; switch .20 → .91; .60 → .96 |
| actor, feedback on | 9/11 | .49 → .50, .50; .60 → .58 |
| memory, feedback off | 2/11 | .22 → .07, .07; .49 → .60, .36 |
| memory, feedback on | 11/11 | all within .04 |

Without the controller the learned models are **bistable**: their LN share runs to near 0 or near 1. On 09-28 the new architectures (clean joint, joint release, action segment, continuous segment, fragmentation repair) could not carry the controller: joint release "requires flat rows with no count prior, scoped allocation or player adapter". LN control disappeared there. Correlation for clean joint is .04 (63 ranges); for the action-segment and fragmentation lines it is negative (n = 6 and 4). The late impression that "LN ratio did not respond" belongs to that change, not to the lineage as a whole.

**Difficulty** (`census_stars.py`: the lineage's star port for whole charts, `ScopeStrainTrace.level` for switched ranges; free-running only).

- Static whole-chart requests, 674 charts: median |error| .72★, 39% within .5, correlation .46, slope .48, mean signed error **+.57**. By request: 2.0 → median 3.47 (n = 48, minimum 2.76, never within .75); 3.0 → 3.94 (n = 45); 4.0 → 4.42 (n = 501); 6.0 → 5.44 (n = 50). Output is compressed toward about 4 to 4.5★ and skewed hard. Three quarters of requests were 4★, so per-experiment "within .5" rates on 4★ panels mostly measure that bias.
- Can H alone explain low requests failing? I applied the lineage's `head_timing_star_lower_bound_20241007` to the same charts (`hfloor.py`). For D2 requests the H-only floor exceeds 2.5 in 46% of charts (median floor 2.04), but realised charts sit a median 1.36★ above their floor. For D3 the floor never blocks, and the overshoot is 1.94★ above it. Most of the overshoot is row choice on top of H: chord width, LN.
- Switched ranges, 194: median |error| .71, 30% within .5. Within 65 charts that carry an override, realised contrast / requested contrast has median **.34**; the direction is right in 91%.

**Style.** No readout exists, so there is nothing to recompute. Four-star Stream requests with LN unspecified gave a median LN share of .068 across 157 free-running charts, but the same case produced .907 in the one clean-joint stage-2048 sample (section 5).

**Did any control ever work on free-running generation within a stated tolerance?** LN amount did, within ±.10, when the inference-time integral controller was on: 9/11 and 11/11 in the one matched ablation, on 5 to 6 songs with 1 to 2 seeds. Learned conditioning alone did not. Difficulty never met the lineage's own criteria (improvement .20, response gap .40) and is compressed to about half the requested range. Style was never demonstrated.

### 3.3 The two late claims (artifact)

- "LN ratio stayed 25-30% whatever was requested": **true for the six action-segment charts** (.245 to .303 for requests .485, .131 and .075; K1 and K4 alike). Those came from a 32-update, 146-second pilot on 158 segments, one seed each, with source H. It is not a lineage-wide finding: see 3.2.
- "A four-state controller chose one state everywhere": **true**, 85/85 plan selections were code 1 (`plans.json`). But the four states are a latent segment plan, not a user control. At step 32 the exact posterior already put .969 of its mass on code 1, with .10 nats of information (`pilot-v1/steps.jsonl`). The codes were never differentiated. Collapse after 32 updates is expected and says nothing about latent plans.

## 4. How a long note is represented through the lineage

| Stage | Representation | Where duration comes from |
| --- | --- | --- |
| R1 baseline | Head action in a row; release action in a later row at a supplied time from R (source heads plus release-only times) | The source chart: tails sit on real moments |
| 09-23/24 skeleton and planned | Open state; H and R are separate millisecond hazards from audio; R1 picks rows; release clock conditioned on full-hold waits | Survival of the R hazard plus R1's choice at H rows |
| 09-25 typed resource plan | Skeleton marks fix TAP/LN counts and release identities before R1 | Skeleton. Reverted the same day (`c5b7db8`) after the human objected |
| 09-25..09-27 controlled | R1 count families `(heads, new LN, releases)` + LN-amount integral controller + scheduler release window (earliest release, deadline = first H needing the key minus RH) | R hazard truncated by the deadline window; R1 at H rows |
| 09-26 hold audio cues | Each active LN retrieves the audio at its start | Same, with better cues |
| 09-28 joint release | R1 scores "wait" against every release subset at each native millisecond; R hazard derived from them | R1 end to end |
| 09-28 action segments | 4 s segments under a latent code; rows as before | Same as joint release |

A long note was **never an object with a duration** chosen at its head. In every form its length is the product of many per-millisecond or per-row "keep holding" decisions. What that makes easy and hard for complaint B:

- **Easy:** incremental publication (the tail need not be known when the head is published); the human agreed the client can carry that (V10). Real-time release reacting to later audio. Release-only moments exist naturally.
- **Hard:** a long hold needs a low hazard at every one of hundreds of queries, so any under-training shows up as short tails. The length distribution cannot be supervised directly, only through survival terms. "Hold this under that TAP pattern" is not one decision, so organising LN plus TAP must emerge from row-by-row choices. The release deadline in `spacing.py` is defined relative to the *next* head that needs the key. That is the mechanism that produces releases timed just before a head. The amount controller adds its own artefact: it rewards prefix balance, so after a TAP run it pushes LN starts in bursts (up to a factor e⁸ for four LN starts at the bound; doc and code). That is one plausible source of "irregular LN runs", which I did not measure (section 9).
- **The bistability in 3.2** (without feedback, LN share runs to ~0 or ~0.9) is what one expects when open-LN occupancy feeds back into the next row's LN decision through history. It is a representation-level symptom, and no lineage document named it as such.

A related support change: the 60/50/50 recovery profile in force from 09-25 to 09-28 dropped 2,313 of 131,611 real 8 s training windows that contain short real LN relationships (doc, `native_pattern_failure_analysis_zh.md`). The 60/25/21 profile restored them. After that, the minimum generated LN in the action-segment outputs was exactly 21 to 22 ms, the new floor (doc), which suggests mass piling up at the support boundary.

## 5. The 09-28 short-tail attribution, and my recomputation

**Method** (code, `clean-joint-proposal-v1/release_ownership.py`): replay each finished chart and give every release a role. It is "H" if its row contains a head, "R" if it is a release-only row. Two flags are also computed: `could_keep_given_heads` (support allowed a row with the same heads that keeps this hold) and "at forced deadline". It ran on nine stage-2048 charts; the documents quote two of them (D4 Stream Zenithfall with LN unspecified; D4 STYX with LN .485), one seed and one checkpoint each.

**Recomputation** (artifact, `osu_stats.py`, independent parser). The counts match exactly: Stream has 4,382 LN, 2,670 of them ≤80 ms, 2,524 ending at an H row and 146 at release-only rows; STYX has 84 short, 13 at H and 71 at R. Three points change the reading:

1. The flag carries no information: `could_keep_given_heads` is true for **every** H-role tail in all nine charts (2,524/2,524, 555/555, 769/769 …). Support almost never forces a release at an H row, so "every one could have continued" is close to a tautology.
2. In Stream, **2,463 of the 2,524 short H-role tails (97.6%) end at the very next head time after their own head**. Head spacing has median 69 ms, and **90.7% of heads are LN**. The short durations are the head spacing. The chart is 6.28★ against a 4★ request. With nine heads in ten starting an LN at 69 ms spacing, four fingers cannot hold much longer. The proximate "R1 chose to release at H" is set by allocation plus H density, and the lever is LN allocation, not release timing.
3. That Stream chart is an outlier. The same case gives LN share .495 at 512 updates, .253 and .351 at 4,096 updates (inherited arm), and .040 in the fresh arm at 2,048. The attribution ("mostly R1 at H" versus "mostly pure R", hence "a single-module fix misses half") therefore rests on one seed of one checkpoint at its worst point, and on a flag that cannot fail.

The method is sound as bookkeeping. As causal attribution it is weak, and the lineage said so itself ("proximate support attribution"). What it lacked: normalisation against a ranked baseline (next paragraph), the interior-head count, and the LN share at which the tails arose.

For comparison, in ranked 3.5-4.5★ charts only **32%** of ≤80 ms LNs end on head rows (16,552 of 52,277); 68% end at release-only moments. Across the 298-chart generated panel it is 77%.

## 6. Long notes in ranked 3.5-4.5★ charts versus generated charts (artifact)

Ranked: 1,973 charts, 1,614 song groups, 3,430,578 heads, 628,071 LN. My parser reproduces the 09-28 study's totals, band medians (196/162/132 ms) and tails-on-head (76.9/67.9/62.0%) exactly, so those lineage numbers can be relied on. Generated: 298 charts from the fixed `four-*` native panel on 09-27/28, with native H and ≥10 LN, across about a dozen models (`gen_panel.py`). Per-chart medians, with IQR:

| Statistic | Ranked 3.5-4.5★ (1,766 charts with ≥10 LN) | Generated panel (298) |
| --- | --- | --- |
| LN share of heads | .154 [.070, .294] (group-weighted mean .183; TAP-majority band 58% of charts) | .224 [.073, .519] |
| Median LN duration | 167 ms [145, 222] | 162 ms [121, 203] |
| Tails on a head time | .79 [.61, .92] | .85 [.75, .92] |
| LN with no head strictly inside ("one-interval hold") | .75 [.64, .85] | .69 [.57, .77] |
| LN ≤80 ms | .000 [0, .074] | .073 [.038, .145] |
| LN ≤40 ms | 0 [0, 0] | 0 [0, .009] |
| **Release followed by any head within 40 ms** | **0 [0, 0]** (pooled, weighted .8%) | **.035 [.017, .063]** |
| Release followed by any head within 20 ms | 0 [0, 0] | .015 [.006, .026] |
| Same-column release → head ≤80 ms | .002 [0, .049] | .055 [.030, .092] |
| Head spacing ≤20 ms (near-duplicate heads) | 0 [0, 0] | .005 [.002, .008] |

The medians look ranked-like: duration, tails on heads and one-interval holds all resemble the reference. What separates generated charts is **the tails**: releases 1 to 40 ms before another finger's head, which almost never occur in ranked 4★ charts; near-duplicate head times; and a heavier ≤80 ms tail. Releasing at the next head is ordinary ranked practice (75% of ranked LNs do it). The problem is how short that interval is and how close the release sits to the next event. The lineage's release-to-head checks were same-column (code, `joint_audio_continuation/evaluation.py`; the RH term of the 60/50/50 profile), and I found no metric for the cross-column case, in which a release lands just before another finger's head. Near-duplicate head times belong to the timing slice. A release-only hazard and a head hazard sampled on the same millisecond clock with no grid can place R a few ms before H, which is the structure of complaint B's "release right before the next head". Pooled over all LNs (`near_release.py`), 6.9% of generated LNs (11,547 of 167,157, 311 charts) are released 40 ms or less before some head, against 1.0% in ranked charts (6,255 of 628,071). In the generated set, 61% of these near-head releases are release-only (R) events, although R events make up only 20% of all generated tails, so R events are about three times over-represented among them. The other 39% end on an H row that is followed within 40 ms by another head, i.e. near-duplicate head times. Both mechanisms are timing-side: R and H hazards sampled independently on a millisecond clock.

## 7. Was there a fixed regression suite?

Only late, and partly. From 09-25 a three-audio panel (Zenithfall, Hysteric, Take) recurs in about 30 to 40 experiment directories. From 09-27 a fixed native panel (`four-styx-helix`, `four-blizzard-heights`, `four-classic-pursuit`, `four-max-burning`, `control-switch`, later `four-stream-zenithfall`) was reused by 10 to 13 experiments (artifact, directory census). `gameplay_regression_evaluation` and its executable qualification appear on 09-27 (`138a1f5`, `226725c`), the B80 red check on 09-28 (`e1fecbd`). The *criteria* still changed per attempt: descriptor error, star error means, ±.20/.40 control thresholds, .016 < .02 gains, B80 quantiles. Seeds were 1 to 3, and most static requests were 4★. The B80 check was built on a failure already known, as its note says. It passed runs the human would reject (fragmentation note: "Whole B80 can pass while local LN organisation remains poor").

## 8. Order of work: controls from 09-25 while free-running output was unplayable

The human asked for the controls: difficulty input, LN/TAP ratio, and style scoped to time ranges and switchable mid-song (V39, 09-25 03:23-03:49Z), and `ca3dc65` followed at 03:49Z.

**Strongest case that it was reasonable.** The target is "playable 2-6★". Without a difficulty input the generator samples a mixture of star levels, and playability at a given level cannot even be judged. A 4★ request is part of defining the unconditional target. LN amount likewise separates TAP-majority from LN-majority styles; a generator that mixes them mid-chart produces exactly the irregular LN runs complained about. Conditioning variables are cheap to add at the start and costly to retrofit. The human asked for the work explicitly and repeatedly (V39, V40, V43).

**Strongest case against.** A control response cannot be read on a generator whose samples are rejected: when "5★ here" fails, it cannot be told whether the control or the generator failed, and the lineage repeatedly could not tell. The work used up the evaluation apparatus: most native panels measured request error (star error, LN error), scalars that unplayable patterns satisfy. The scoped and mid-song part asked for a conditional the data never contains (section 2). Style had 3 to 20 prominent cells per concept and an earlier result showing the concepts were not even recognisable. The LN "control" that worked was a hand-built inference controller that masks the learned model's bistability. It added state (episodes, quota, schedule revision) that each later architecture had to carry, and when the 09-28 architectures dropped it, LN control was lost. About half the commit subjects concern controls or LN, while the ordinary 4★ generator never became acceptable.

**My call.** A static chart-level difficulty request (and possibly a chart-level LN band) as a conditioning variable from the start was reasonable. Scoped and mid-song controls, style, learned control objectives and the amount controller were premature. They should have waited until a 4★ generator produced output the human accepted across several songs and seeds without controls beyond the static request. Confidence about 70%. The main uncertainty is whether static difficulty conditioning is part of fixing ordinary output, in which case some of the early control work was necessary rather than premature.

## 9. What was never questioned

- The whole-chart star rating (a 2024 algorithm that is weak on LN) as the difficulty target, and a strain-peak proxy as local difficulty. Both were used as training targets without a check against human judgment.
- Whether an inference-time integral controller counts as a "control" under the standing constraint that no condition is offered before it works. The learned model on its own did not follow the request.
- LN amount defined as `LN heads / heads`. The 09-24 doc itself shows one scope with a single new head and 4.4 to 7 s of hold time. Hold duration and coverage were never controlled.
- The bistability of LN share without feedback, as a property of the representation and not of tuning.
- Long notes as an open state closed by later events, and never as one object with a duration (now open question H5).
- Short LN treated as a question of which module owns the release (R versus R1), rather than one of allocation density and head spacing. My recomputation points at allocation.
- The same 5 to 8 songs, mostly 4★, 1 to 3 seeds, for every conclusion. Difficulty control across 2 to 6★ was never tested with a spread of requests.
- Whether the human style cells could support any learned control after the 09-14 postmortem.

## 10. What is worth keeping

- The ranked 3.5-4.5★ LN census (09-28), reproduced here exactly, and the nine source-linked readings of LN-plus-TAP organisation in `ordinary_fourstar_rhythm_and_holds_zh.md`. These are solid.
- The same-audio, identical-H pairs of different arrangements (1,339 pairs over 424 audios with local difficulty differing by ≥.5; doc). This is real conditional variation for a difficulty study.
- The measured local-versus-whole-chart difficulty gap (median −.54★).
- The condition audit (score a real scope under the correct and a swapped condition). It is cheap, and it carries a lesson: 20/24 on real histories did not carry over to free-running generation.
- `short_ln_exposure` with explicit denominators and request-selected reference bands; `release_ownership.py` as bookkeeping; the profile-path crossover method (H plan versus downstream).
- The controller-semantics on/off ablation as the one clean test of learned versus controlled LN amount.

## 11. Inconsistencies, and questions only the human can settle

1. Does "control" permit an inference-time feedback controller when the learned model does not follow the request? It was the only LN control that worked.
2. What does a scoped difficulty mean: local strain, "like the hard part of an X★ chart", or the main-subdivision response the human described on 09-28 (V72)? Whole-chart labels cannot teach the first or the third.
3. Is style out of scope for the first target, given 289 cells and a classifier that could not recognise the concepts?
4. Which part of complaint B is the defect: short absolute duration, LN share, or a release landing just before another head? On my measurements the last separates generated from ranked most clearly. Is a chart that meets its LN ratio by holding across a TAP passage a success (already H3)?
5. Metadata stars (manifest filter) and recomputed stars (09-28 studies) are both in use and are never reconciled.
6. The feedback index calls the K=4 segment code a "four-state controller". It was a latent plan, not a control.

## 12. What I could not verify, and how to

- Whether the amount controller produces irregular LN runs. Command: compare LN run lengths in `20260927-controller-semantics-v1/{actor,memory}-feedback-{on,off}/*/generated.osu` (11 charts per arm) with the ranked cohort.
- The style probes' structural statistics (only the docs were read); per-attempt star errors in `typed_audio_continuation.md` (doc only; my census covers the same directories only in aggregate).
- Whether the star port matches the official 2024 calculator; whether `head_source=native` in every directory means H generated from audio (taken from `result.json` as written).
- The census groups repeated outputs. For example, `scoped-ln-allocation-v1/native-{context,progress}-v2` and `broader-full-row-learning-v1` hold identical LN shares on the Stream cases, so experiment-level n is overstated there.
