# Audio timing and skeleton lineage review

Slice: `03-audio-timing-skeleton`; review date: 2026-09-30.
Started 12:59 UTC; completed 13:22 UTC. Scope: H, R, audio representations, and timing/row ownership.

## 1. Scope and sources

The strongest result is a timing contribution to the late short-LN failure, with a remaining row/release contribution.
The evidence does **not** establish that millisecond resolution itself is defective, or that a beat grid would cure the problem.
The memory experiment rejects one small fitted candidate; it does not reject audio/history attention as an idea.

Evidence labels throughout are `doc-claim`, `checked-code`, and `checked-artifact`.
Judgments are my inferences from the cited evidence, not additional measurements.
`D` below denotes `/tmp/lineage-review/trees/audio-joint/docs/research/`; document filenames resolve there.
Code citations use `099cb66:src/...` for the lineage-end export that I inspected, unless another commit is stated.
Counts of song groups are not asserted to be counts of distinct recordings: the data audit found grouping/byte-identity discrepancies.

Read the baseline entry point `.git/research-relay/notes/RESEARCH.md`, feedback-index sections 2–3, workspace instructions, README, and formulation excerpts.
Read relevant portions of all sixteen documents named in the slice, including the lengthy choreography and typed-model documents.
Also read the owning sections of `audio_joint_playtest_v2.md`, `controlled_audio_continuation.md`, `joint_r1_release_decisions.md`, `action_segment_r1.md`, and `ordinary_expert_from_scratch_zh.md`.
The chronology is anchored to the read-only git log, saved as `/tmp/lineage-review/03-audio-timing-skeleton/commits.txt`.
No private human material, agent-session files, network sources, model generation, training, or accelerator execution was used.

The central checked implementation references are:

| ID | Evidence and exact location | What was checked |
| --- | --- | --- |
| C1 | `checked-code`, `099cb66:src/ensomi_model/research/audio_skeleton/data.py:10`, `model.py:20`, `training.py:27` in that package | 10-ms frames, two slots per role, offsets, BCE/offset loss, threshold decoder and matching. |
| C2 | `checked-code`, `099cb66:src/ensomi_model/research/joint_audio_continuation/model.py:30`, `timing.py:21`, `context_model.py:47` | Local encoder, ten hazards per bin, survival law, complete-song attention. |
| C3 | `checked-code`, `099cb66:src/ensomi_model/research/planned_audio_continuation/model.py:190`, `generation.py:85`, `features.py:102` | Separate H/R logits, native sampling, compulsory H rows, release support. |
| C4 | `checked-code`, `099cb66:src/ensomi_model/research/typed_audio_continuation/model.py:102`, `program.py:14` | Categorical/H-priority timing, typed marks, resource clocks and recovery limits. |
| C5 | `checked-code`, `099cb66:src/ensomi_model/research/planned_audio_continuation/spacing.py:26`, `spacing.py:64` | Four-head capacity and row-derived release windows; no global adjacent-head gap floor. |
| C6 | `checked-code`, `099cb66:src/ensomi_model/research/controlled_audio_continuation/joint_release.py:79`, `generation.py:56`; planned `session.py:231` | Row scores determine release hazard; late joint mode bypasses conditional waiting normalization. |
| C7 | `checked-code`, `099cb66:src/ensomi_model/research/audio_memory_continuation/audio.py:9`, `model.py:19`, `memory.py:18` | Audio pyramid, separate H/R/R1 readers, 64-s history, causal known-index cap. |
| C8 | `checked-code`, `099cb66:src/ensomi_model/research/segment_audio_continuation/bootstrap.py:19`, `model.py:134` | Fresh initialization, smaller encoder, unrestricted H residual, joint releases, persistent segment context. |

Raw evidence checked includes the first pilot results/freeze files, millisecond sensitivity summary, data/input audits, onset-rate validation, memory fit and qualification records, and final clean-fit records.
For the new timing audit I parsed twelve complete `.osu` files: four sources, four fully native charts, and four source-H charts from `20260928-ordinary-scratch-v1`.
Each source hash matches its recorded identity. Checkpoint inspection used CPU `torch.load(weights_only=True, mmap=True)` solely to count stored encoder tensors.
I did not visually inspect Lens images or listen to audio; historical visual verdicts remain `doc-claim` even when a reading receipt survives.

## 2. Attempts

**Question 1: timing spaces, resolution and minimum gaps.** “R” changes meaning across the lineage; treating it as one unchanged release module obscures the redesigns.
In R1's original contract, R is an externally supplied event schedule. In the audio papers it often means actual release-only events; releases may also occur on H rows.

| Stage | H output and conditioning | R output and minimum spacing | Evidence |
| --- | --- | --- | --- |
| 09-23 frame pilot, `135e181`, `273c296`, `068988e` | Two head-presence slots per 10-ms frame; learned ±5-ms offsets; local-peak/threshold selection. Initially fractional times, subsequently rounded/deduplicated to native integer ms. | Two analogous **release-only** slots, not all LN tails. At most two same-role events per frame; neighboring peak rule is a decoder restriction, not a physiological gap. Integer materialization permits 1-ms distinct times. | C1; fractional-to-integer decoder change checked in `git show 068988e -- src/ensomi_model/research/audio_skeleton/data.py`. |
| 09-23 flat joint, `09b919c` | No separate H hazard: one hazard for any nonempty event, followed by a complete row deciding head/release role. Reads row history and exact state. | Same shared event hazard; ten Bernoulli hazards for ten native-ms positions. Earliest next event is previous time +1 ms; true audio-end closure is forced. | C2; `D/audio_conditioned_choreography.md`, “First implementation”. |
| 09-23 optional thinning, `a5abc25`, `3a5cea9` | Same flat joint time/row law; row-dependent soft acceptance uses same-lane head age, then may be distilled into the model. | Scale 27 ms multiplies acceptance by `(age/27)^4` below 27 ms; it preserves positive support and imposes no hard minimum. | `checked-code`, `099cb66:src/ensomi_model/research/joint_audio_continuation/head_spacing.py:19`. |
| 09-24 full context, `ce25a7b` | Same clock; optional audio/LN-state base plus bounded row-history residual. | Same any-event law; 500-ms audio cells do **not** quantize output. No global comfort gap. | C2; `context_model.py:187`. |
| 09-24 planned H/R, `e04b349` | Separate next-H Bernoulli hazard at each ms; H-only finite history and future H preview. | Separate release-only hazard from H/R history, LN occupancy/age and next heads; no event without held lanes. Original full-hold law forces remaining mass at H−1 ms. H rows may also release. | C3; `features.py:119` and `features.py:137`. |
| 09-24 bounded H and conditional R, `88e5683`, `12d80eb` | Audio base plus residual bounded by `4 exp(−H_age/1000)`; same 1-ms support. | Conditional first-release distribution before necessary H replaces the original deadline atom; retains 1-ms possibilities, not a learned minimum gap. | C3; `planned_audio_continuation/release.py`; `D/release_wait_conditioning.md`. |
| 09-25 shared spacing, `7329b0e` | H support checks four-column capacity: fifth distinct head cannot precede first+HH. This is **not** adjacent-H spacing. | Earliest LN release and latest feasible release, plus same-lane HH/RH/HR row support. Positive `minimum_action_gap_ms`; zero preserves earlier law. | C5; `spacing.py:117`. |
| 09-25 typed, `ca3dc65` | Per-ms categorical `none/H/R-only`, plus mark `(TAP count, LN count, release-ID mask)`. | Initial resource HH/RH/HR =37/25/21 ms; later trial 60/50/50. Different resources can act 1 ms apart. | C4, `program.py:26`. |
| 09-25 typed head-stream variant, `4cda0e3` | H-priority hazard: `P(H)=h`, `P(R)=(1−h)r`; H history contains H gaps and counts and reads current resource state. | Same typed mark/support; changed R logit no longer directly renormalizes H probability. | C4, `model.py:108`. |
| 09-25 controlled row-owned, `c5b7db8`, `bbd1a7f` | Return to timing-only H, independent of learned row history. | Independent ms R preferences; row feasibility supplies release bounds. Typical support 60/50/50, later 60/25/21; these are run-specific, not universal constants. | C3/C5; controlled `model.py:78`; clean native `result.json`. |
| 09-27 memory; 09-28 clean joint | Same H/R timing spaces and support as their parent; memory changes information, clean fit changes learning/initialization. | Memory qualification uses 60/50/40; clean final qualification uses 60/25/21. | C7; raw fit/native configs. |
| 09-28 joint R1 releases, `965d670` | H remains timing-only. | R1 scores virtual wait and up to 15 nonempty release subsets at every ms; hazard derives from their odds. Nonempty rows alone enter history. Forced feasibility/terminal atoms remain. | C6, `joint_release.py:123`. |
| 09-28 segment and ordinary scratch, `7d31b1e`, `d6eba23` | Segment plan does not replace H. Fresh model uses direct unbounded H logits, 16-head preview and four-head capacity with HH=20. | Joint R1 wait/release law; fresh `Recovery(20,1,1)` explicitly allows 1-ms LN and release-to-head intervals. | C8, `bootstrap.py:26`. |

**Questions 3 and 5: dated experiment and dependency inventory.** The following is chronological, not a claim that every branch replaced the deployed checkpoint.
Five substantive direction/ownership reversals are numbered **1–5**; other entries modify information, support or training without reversing ownership.
Here “reversal” means moving a decision across the timing/row boundary or adding/removing learned row-history dependence. Counting every optional decoder flag would give a different, unhelpful total.
All rows below are single training-seed studies unless stated; **between-training-run variance was not recorded**. Generation seeds are not independent fitted replicas.
For unreported quantities, “not recorded” means unavailable in the sources inspected for that attempt, not proof that no file anywhere contains them.

| Date / commits; problems | Hypothesis and ownership change | Size, measurement and result | Interpretation at the time; next step |
| --- | --- | --- | --- |
| 09-23 `ff6471c`→`068988e`; B/F/H | Audio timing can drive frozen R1; first test sensitivity to extra optional non-H opportunities. | No fit; R1 3,084,432 parameters; 12 songs × seeds 17/23 ×3 schedules=72 outputs. Median LN-duration ratio .686 with every-fourth-gap insertion, .375 with every-gap insertion; LN shares +25.6/+68.6 percentage points. Runtime total not recorded. **checked-artifact**: `artifacts/audio-skeleton/20260923-ms-sensitivity/sensitivity/summary.json`. | Schedule is consequential; legality does not certify quality. Then learned timing pilot. |
| 09-23 `135e181`/`273c296`; E/H | Independent frame skeleton, optional frozen BeatThis features, then frozen R1. | 770,188 trainable parameters; 48 train charts/songs; six calibration and six assessment songs. Seed 172; 1,200 updates: local 107.70 s, BeatThis 202.36 s; selected updates 600/200. H F1@20 ms .677→.732 with supplied source density; R-only .035→.051. 200-update four-chart overfit diagnostic took 16.67 s. **checked-artifact**: pilot `training/*/result.json`, `freeze.json`. | Small transfer signal; not playable qualification. Frontend then deliberately changed. |
| 09-23 `09b919c`; E/F/J | **Reversal 1**: replace external timing→rows with next-event timing that reads earlier complete rows; joint audio/time/row learning. | 2,950,458 total, local encoder 463,392; 121 train charts/48 groups,12 validation songs. Seed 230923; memorization 300 updates/74.14 s; random 2,400/989.40 s; full-wait coverage 2,400/1,118.44 s. Fixed validation 48 queries: best joint NLL 6.071 random vs 6.091 coverage. **checked-artifact**: `20260923-v1/training/{memorize,random,coverage}-v1`. | Fits supervision; native short repeats remain. Expanded audio corpus and full context followed. |
| 09-23 `a5abc25`→`16209ea`; A/H | Soft marked thinning, then learn the corrected complete next-event distribution on native histories. | Same 2.95M model. Decoder probe 18 songs reduces<=10 ms TAP pairs 42→0 and<=20 ms 157→10. Matched 600-update continuations use five TRAIN songs for 640 correction examples/390 prefixes, sixth song 128/70 held-out examples;18 generated cases/arm. Corrected endpoint still has 2<=10 ms pairs, LN count 1,839→4,091 and median head-count ratio .688. Fit time/seeds not recorded here. **doc-claim**: `audio_conditioned_choreography.md`, “Learning the local correction”. | Failed preservation/transfer guards; one ten-head output stops near 7.8 s in 121 s audio. This directly motivated waiting-history investigation. |
| 09-24 `ce25a7b`, `98013ee`; D/E/H | Full-song attention plus bounded row-history timing to recover activity after silence. | Expanded 121/48→585/240; 36 validation songs. Four 1,200-update interval cells,4,800 intervals each; global/bounded 3,461,828 parameters. 42 audios×2 seeds per cell=336 outputs; <=20 ms same-key pairs 17/35/11/4 across local/global × original/bounded. Per-arm wall time not recorded here. **doc-claim**: `audio_joint_playtest_v2.md`. | Useful candidate, not general playability; source NLL nearly tied. Byte-identical source aliases later expand 585→615 charts without increasing 240 groups. |
| 09-24 `6031291`; C/D/E | Whole-song four-state latent intent conditions the shared audio coordinates of timing and rows; no clock or ownership change. | Adds 5,192 parameters to 3,461,828; full-audio prior, target-descriptor recognition model, one fixed code per song. The owning document records an implemented comparison but no completed training steps, time, chart/song counts or native result. **doc-claim**: `audio_persistent_intent.md`; **checked-code**: `099cb66:src/ensomi_model/research/joint_audio_continuation/intent_model.py:52`. | Experiment remains open in the document; do not call persistent arrangement conditioning never attempted. |
| 09-24 `b5a664c`, `e04b349`; F/A/B | **Reversal 2**: remove row-history dependence from H, separate H plan/R clock, restore future timing and `frontier2` to R1. | 4,245,188 parameters;615 train charts/240 groups,36 VAL;1,200 updates/991.64 s. Nine native cases, four premature H cessations; joint NLL/s 61.65→40.28. **doc-claim**: `head_wait_recovery.md`. **checked-artifact** input audit: two prefix interventions on one chart changed flat timing logits by .459/1.419 despite same LN state. | Refined interface is implemented, but new H process starves. Bound its historical veto next. |
| 09-24 `88e5683`→`7c316e6`; A/B/D | Bound H history; replace full-hold deadline mass with feasible conditional R. | H repair 4,247,438 parameters,1,200 updates/867.46 s on same corpus; three of four frozen prefixes pass 99% five-second H CDF, piano only 93.52%. R-only frozen-weight comparison: nine outputs, two changed, <=20 ms RH 1→0;31 s CPU. No added R parameters. **doc-claim**: `head_wait_recovery.md`, `release_wait_conditioning.md`; mechanism checked C3/C6. | Partial H recovery and one exact release repair; neither presented as full success. Joint refit follows. |
| 09-24 `9a4f1fe`; A/B/F | Train conditional R jointly; swap generated H across two materializers. | Same 615/240,1,200 updates/907.01 s,4,247,438 parameters; nine native cases, six fail LN stability. Two selected songs cross H/materializer at fixed seeds: Prom Queen LN 18.04→2.85% when only plan changes;17.11→2.03% under other materializer. **doc-claim**: `head_plan_row_response.md`. | Strong plan effect on selected outcomes; candidate consequences can still choose avoidable bad rows. |
| 09-24 `a2bce68`,`fd2ddfd`,`844de84`; D/F/I | Route density via H; decompose drift; exchange whole H plans between initial/continued materializers. | No new fit for decomposition/composition; eight audios×two profiles, one seed/audio;48 reproduced H streams,32 drift comparisons. Prospective composition total control error 7.340→7.385;LN error−6.9%, width error+13.8%.21 screened calls finish,22nd exhausts retries,10 unattempted. Param count/analysis wall time not recorded here. **doc-claim**: `head_factor_drift.md`, `head_materializer_composition.md`. | Mixed base/history causes, composition fails its guard. Counts/support moved into further proposals. |
| 09-25 `7329b0e`,`ca3dc65`,`4cda0e3`; A/B/F/I | **Reversal 3**: move chord size, TAP/LN counts and release identities into upstream typed resource plan; R1 assigns columns. Separate head phase later. | 614 train charts,36 VAL; paired-song count not recorded in typed fit summary.3,895,879 parameters;1,200 updates/413 s; LN-law continuation 400/147 s also changes scope sampling.15 native cases per arm/three audios; high-LN proportions improve, e.g. Zenithfall 21.7→74.8% for 70% request, but 173/3,005 LNs<=40 ms. Head-stream variant 4,301,393 parameters. **doc-claim**: `typed_audio_continuation.md`. | Controls improved with unresolved difficulty/fragmentation; both factorization and sampling changed. Ownership subsequently reversed. |
| 09-25 `c5b7db8`,`bbd1a7f`; F/I/A | **Reversal 4**: restore complete row choices to R1; retain timing-only H, pass row recovery bounds to R sampler. | Core 4,583,985 parameters;2,500 updates. Accepted train charts/groups and fit time not recorded in inspected summary. Three audios at 3-star/20% and 70% LN; amount feedback improves low-LN error .0548→.0017, star error still 1.0559. **doc-claim**: `controlled_audio_continuation.md`; C5 confirms actual row→R feasibility coupling. | Ownership repaired; timing and materialization still fail separately. |
| 09-26 `94c6968`,`c5fd73f`; A/E/H | Source-H counterfactuals and separate audio H-rate reference. | Source-H diagnostic: three TRAIN songs, one fixed seed each; singles 3.0★ becomes 4.85★ at unchanged 783 H. Rate readout 63,362 parameters;2,000 steps/83.95 s;5,188 sampled charts, song count not recorded.36 validation windows: absolute head-count error 28.32→11.68. **checked-artifact** for rate; **doc-claim** for source-H star comparison. | R1 can be wrong even with good timing; rate calibration is not rhythmic organization. |
| 09-27 `b130dfb`,`ec9c1ef`,`f22e936`; D/E/H | Add multiscale audio and H/R/R1 query memory; keep ownership. | 7,616,517 vs 4,583,985 parameters;384 matched updates,768 windows/417 charts/383 groups; baseline 1,227.90 s, memory 3,058.49 s including discarded attempts.28 cases/arm;84 exports. Memory pressure .38230 vs unfitted .04447 and fitted baseline .78553; star MAE1.33452 vs 1.15214/1.51711. **checked-artifact**: memory fit/config/preparation/comparison/verdict. | Native qualification fails; runtime passes. Later fixed-prefix probes distinguish H base drift from memory contribution. |
| 09-28 `de5d560`; E/I/H | Let control multiply encoded audio in H's base instead of only adding an audio-independent shift. | 512 steps,1,024 windows/736 charts/661 groups; H trainable 513,108 vs 588,372, other modules frozen;1,123.78 s pair.22 VAL windows; H NLL 32.019→31.654/31.623;28 cases/arm. D2 star MAE parent 1.385/additive 1.614/modulated 1.788. **doc-claim**: `head_audio_control_interaction.md`; implemented path checked. | Small interaction learned, low-difficulty qualification fails. Clean joint learning follows. |
| 09-28 `ef42095`→`6de65d2`; B/C/E/H | Three initializations, direct LN conditioning, joint audio/H/R/R1 learning; same factorization. | 4,675,633 each;4,096 updates/arm;8,192 windows,3,736 TRAIN charts/2,270 groups plus 22 validation identities.128 completed segments total 13,450.65 s across three arms, not per-arm time. Seed 280281. At 4,096,28 cases each complete;19/18/17 fail numeric checks. **checked-artifact**, scratch aggregation of `source-plan.json`, `fit-v1/*/result.json`, `native-4096-*/result.json`. | Document stops at 512; raw endpoint is later and remains failed, semantic review pending. No initialization is qualified. |
| 09-28 `965d670`,`a6c912f`; B/F/H | **Reversal 5**: R1 decides wait versus release before selecting R time; formerly it could only choose a nonempty mark after R fired. | Inherited 2048 parent plus joint R/cues/loss reweighting;16 updates/32 draws/91.34 s, then 64/128/321.31 s. Three known songs, one paired seed each; exact total parameters/charts/groups not recorded here. Stream median LN 73→191 ms, but 5.79★ forD4;130/1,842 tails at nonterminal deadlines vs 0/4,382 parent. **doc-claim**: `joint_r1_release_decisions.md`; C6 checked. | Short-tail prevalence improves while amounts, style and startup fail. Multiple interventions prevent attribution to R ownership alone. |
| 09-28 `7d31b1e`,`e4c4453`; B/C/E | Segment R1 plan persists up to 4 s; H remains separate. Categorical plan initially blocks context, then continuous path is added. | Matched 1/4-state pilot 32 updates,158 segments,3,960 target rows,160,831 R-risk clocks; train chart/song totals, parameters and wall time not recorded here. Six source-H outputs/three songs; all 85 four-state choices use code 1. Short 21–25 ms LN remain. **doc-claim**: `action_segment_r1.md`; C8 model path checked. | Failed small pilot, not a test of learned diverse segment plans. Continuous repair has no demonstrated native qualification here. |
| 09-28 `d6eba23`,`96f84fd`; A/B/C/E | Fresh ordinary joint H and segment-row model, smaller audio encoder and unbounded H; no inherited actor weights. | 4,782,754 parameters;four charts/four songs;16 four-second units;512 updates/235.73 s; train seed 290029, generation 290031.12 outputs: source-prefix/source-H/native; eight BOS outputs fail 40 ms LN regression. **checked-artifact**: ordinary capacity initialization/result/native records; C8. | Capacity probe explicitly unqualified; most whole-song positions were not trained. New timing audit below separates H contribution from remaining R1/R errors. |

**Question 3: encoder details.** Full-song input and causal output are different properties.
All encoders below use future audio; the prohibition is on future *committed chart* leakage.

| Encoder stage | Architecture / receptive field | Parameter evidence |
| --- | --- | --- |
| First frame pilot | 128→128 projection; six depthwise kernel 5 layers, dilations 1–32; parallel 100-ms pooled branch, four kernel 3 layers, dilations 1–8.16 s windows with 4 s halos. Optional 514-D frozen BeatThis embedding/beat/downbeat features at 20 ms, interpolated to 10 ms;30 s extraction chunks. Bidirectional. |770,188 for complete learned predictor; separate pretrained encoder size not recorded. C1; `audio_skeleton/audio.py:91`. Longest slow branch has 31 pooled cells before interpolation, not unlimited audio context. |
| Canonical local encoder |24 kHz mono,128 Mel bins,10 ms hop,40 ms Hann/FFT960, natural log floor 1e−5, no centering; frame centers 20+10i ms.128→96 plus six kernel 5 depthwise residual blocks.253 frames:2,560 ms waveform support, computed as 252×10+40. |463,392 checked in random-fit transfer receipt; `099cb66:src/ensomi_model/features/mel_base.py:52`, C2. |
| Full-song context, retained through planned/typed/controlled/clean | Local encoder plus learned 50-frame pooling, two width 128/four-head bidirectional Transformer layers.500-ms cells, sinusoidal **seconds**, complete-song receptive field. | Local 463,392 + context 419,712 =883,104 encoder parameters, excluding projections into H/R/R1. Checked checkpoint tensors; C2. |
| Audio/history memory | Adds mean/max 500-ms cells and .5/2/8 s pooled views; three bidirectional width 256/four-head layers. Separate width 128 H/R/R1 attention over last 64 s sampled in 500-ms cells. | Pyramid 2,550,272; H/R/R1 readers 169,852/188,312/124,096. Adds 3,032,532 total; audio-only encoder 3,433,376. Checked tensors and C7. |
| Fresh ordinary |128→64, four kernel 5 dilations 1/2/4/8:61 frames,640 ms waveform support. Global two-layer width 96/four-head attention remains full-song. | Local 143,168 + context 242,656 =385,824. Total model 4,782,754 includes segment/row components. Checked capacity checkpoint and C8. |

Encoder counts and receptive-field arithmetic are reproducible with `/tmp/lineage-review/03-audio-timing-skeleton/encoder_counts.py` and the formulas above.
Inputs are the baseline/memory `step-384.pt` and ordinary `step-512.pt`; three checkpoints, CPU only.
No missing cache was regenerated; `artifacts/cleanup-report-20260930.json` was read to distinguish retained milestones from deleted recovery weights. The original frame pilot's centered/log 10 frontend differs materially from the canonical frontend; its gain cannot be carried over as a controlled canonical-Mel comparison.

**Question 2: beat/tempo/subdivision representation.** No explicit BPM, beat phase or subdivision variable enters the checked joint/planned/typed/memory/segment model inputs, outputs or losses.
H-gap features, absolute time and positional seconds can implicitly encode periodicity; that is weaker than a persistent musical coordinate.
The early frozen BeatThis branch is an explicit exception: beat/downbeat logits and learned features enter as evidence, not a hard event grid (C1, `audio_skeleton/audio.py:100`).
Source timing points are used in **analysis**, not as training labels or model conditions: `distribution-audit/audit.py:42` parses redlines and `:51` computes subdivision residuals.
Source-free `.osu` outputs contain a documented constant 120-BPM display placeholder. It is not a predicted BPM; preserving a source header elsewhere also does not imply model conditioning.

`D/audio_rhythm_hierarchy_zh.md` proposes a persistent latent tempo/phase/main-layer plan, optional subdivision/ornament slots, skip decisions, native-ms offsets and a free branch.
A slot is consumed after one event; the proposal explicitly recognizes that repeatedly sampling a narrow intensity peak can duplicate heads.
It discusses marginalization/variational training and preserving phase across insertions/publication windows. It is marked **not implemented or trained**.
Only the descriptive lattice evaluator is implemented, at `099cb66:src/ensomi_model/research/gameplay_evaluation/rhythm_lattice.py:11`; it fits observed H, not audio or a learned rhythm prior.
Thus the timing representation was questioned, but the corresponding generation hypothesis was not experimentally tested before closeout.

**Question 4: timing evaluated without rows.** Yes. “Only downstream charts were scored” is false, although no sustained independent timing benchmark governs the whole lineage.
The following inventory distinguishes source comparisons, prediction tests and native diagnostics; they cannot be pooled into a quality score.

| Measurement | n / tolerance / result | Evidence and limit |
| --- | --- | --- |
| Alternative source arrangements |416 eligible charts,584 same-audio pairs/91 groups;20 ms median H F1 .7854, R-only .0915.151 density-matched pairs: H .9410, R-only .1695. | **checked-artifact** `artifacts/audio-skeleton/20260923-v1/distribution-audit/extended-summary.json`; source/source, not generated quality. |
| Source redline alignment |418 charts with H,345 with R-only; denominators 1,2,3,4,6,8,12,16,24,32,48,64,96,192;2 ms tolerance. Median off-1/48 fraction 0; maxima .537 H/.582 R-only. | **checked-artifact**, **checked-code**, same owner `charts.json`, `audit.py:51`, `augment.py:29`. Residuals under a chosen subdivision family; they do not establish bad metadata or bad charts. No generated beat-grid test here. |
| Frame-pilot P/R/F1 | Six assessment songs, five with R-only targets; six separate calibration songs. At 20 ms supplied density: local H P/R/F1=.607/.774/.677; BeatThis=.689/.796/.732. R-only=.0459/.0288/.0354 vs .0354/.0932/.0513. | **checked-artifact**, `training/{local,beat}-pilot-v1/result.json`; one train seed; source density is an oracle-like supplied control. |
| Other pilot tolerances/controls | H F1@10/20/40/70 ms: local .467/.677/.732/.757, BeatThis .546/.732/.781/.795. Default-density 20 ms H .619/.698; both R-only F1=0. | Same six songs; full P/R/F1 at all tolerances saved in scratch `artifact-metrics.json`. These original results precede integer-ms materialization correction. |
| Event-time likelihood | Flat joint 48-query validation panel plus two full-gap probes; later interval H/R NLL and 22-window H-modulation validation. | **checked-artifact** early results; **doc-claim** later H probe. Measures teacher-forced probability, not onset matching or rhythmic regularity. |
| H waiting and drift | Four starving prefixes; later 48 reproduced H streams/32 count decompositions; eight audios×two profiles for composition H-rate error. | **doc-claim** `head_wait_recovery.md`, `head_factor_drift.md`, `head_materializer_composition.md`; fixed prefixes/profiles, not held-out timing accuracy. |
| Separate H count predictor |36 validation chart-windows; absolute count error 28.32→11.68, predicted total 886.94→1,861.54 for source 1,894; NLL/s−.523→−2.077. | **checked-artifact**, `20260926-onset-activity-v1/fit-2000/validation*.json`; counts on actual window extents, not onset-location accuracy. Negative objective includes omitted constants. |
| Memory next-event sensitivity |12 generated prefixes from three failed songs plus three human prefixes; three models;339 input laws and 150 component exchanges. | **doc-claim** `audio_memory_joint_fit.md`; censored waiting distributions, not reference-onset precision/recall. |
| Late lattice coverage |21 fixed scopes: three source/native comparisons across four conditions plus nine source examples.40–2000 ms candidate units,±3 ms. STYX92.3% source vs 30.9% inherited 2048; Blizzard 100% vs 36.4%; Zenithfall 97.6% vs 37.0%. | **doc-claim** hierarchy document; evaluator `gameplay_evaluation/rhythm_lattice.py:11` checked. Eight/nine extra sources meet 12-H eligibility. Fitted grid ignores redlines and is not held-out tempo inference. |
| New matched-grid/gap audit | Four songs; source/native/source-H, twelve files.10 ms redline-subdivision tolerance; strict <10/<20/<40 ms gaps; results in section 3. | **checked-artifact**, independently computed for this review, not a metric used to select lineage models. |

No systematic late generated-H precision/recall/F1 comparison across a reserved multi-song panel was found in this scope.
No lineage-wide generated-head share on the *source redline grid* was found; the early source-only redline audit and late redline-free lattice test are distinct.
Musical alternatives make exact source matching incomplete, but do not make count, phase continuity, duplicate-onset and local cadence checks dispensable.

## 3. Commentary

**Question 7: direct test of the timing-defect lead.** I used the ordinary capacity endpoint because all four sources and matched generated/source-H outputs survive.
This is a four-song capacity study, not representative corpus sampling: Dawn, Sulyvahn, mumei, Kill The Beat; generation seed 290031 throughout.
The model saw only sixteen four-second training units, including some inspected contexts. Whole-song inference substantially exceeds that coverage.

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /tmp/lineage-review/03-audio-timing-skeleton/chart_timing_audit.py`.
Inputs: `artifacts/joint-audio/20260928-ordinary-scratch-v1/capacity-data-v1/data.json`, the twelve files listed in scratch `chart-timing-audit.json`, and `capacity-native-512-v1/{tap,held_stream,mixed,ln_body}-{native,source_H}-290031/generated.osu`.
The script checks source SHA-256, parses actual note heads/tails and uninherited timing points, and counts distinct global H times; chords count once.
“1/4,1/6,1/8” means beatLength divided by 4,6,8. Active redline offset/beatLength defines phase; before the first redline its grid extends backward.
“Extreme short LN” here means duration **<=40 ms**, matching the retained late regression; `<` for head gaps and `<=` for LN duration are intentionally different.
“Tail at next H” means exactly the first distinct global head after that LN's start, regardless of lane. These are descriptive coordinates, not proposed legality rules.

| Source / fully native | H / adjacent gaps | Gaps <10 / <20 / <40 ms | Native median gap | Native <=40 ms LN / all LN | Short tails exactly at next H |
| --- | --- | --- | --- | --- | --- |
| Dawn |1,542/1,541 →1,231/1,230 |0/0/0 →130/208/220 |128 ms vs 129 ms source |31/270 |26/31 (83.9%) |
| Sulyvahn |865/864 →773/772 |0/0/0 →63/120/138 |140 ms vs 150 ms |47/263 |39/47 (83.0%) |
| mumei |995/994 →1,128/1,127 |0/0/16 →121/224/256 |162 ms vs 157 ms |68/395 |54/68 (79.4%) |
| Kill The Beat |625/624 →603/602 |0/0/0 →44/106/111 |125 ms vs 125 ms |45/365 |36/45 (80.0%) |
| Pooled four-song source |4,027/4,023 |0%,0%,0.398% |Per-chart values above |0/1,296 |Undefined: no extreme short LN |
| Pooled fully native |3,735/3,731 |9.60%,17.64%,19.43% |Per-chart values above |191/1,293 (14.77%) |155/191 (81.15%) |
| Pooled BOS + source H |4,027/4,023 |0%,0%,0.398% |Exact source H |29/1,302 (2.23%) |3/29 (10.34%) |

These pooled percentages weight events, not songs; n remains **four songs**, not thousands of independent observations.
Same-lane <40 ms attack pairs are 0 in sources,51 in native outputs and 4 with source H. They are different from the global-H gap counts above.
Only 4/191 native extreme short tails end 1 ms *before* next H. The dominant late phenomenon is release **on** the next close H, not the old H−1 atom.
Holding H to source lowers extreme short-LN count 191→29, but row histories, RNG consumption and release trajectories diverge; this is total system effect of H substitution, not an isolated decoder coefficient.

| Grid test, pooled H | Within 10 ms of 1/4 | Within 10 ms of 1/6 | Within 10 ms of 1/8 | Union of three |
| --- | --- | --- | --- | --- |
| Source on its own redlines, n 4,027 |94.81% |75.07% |96.87% |99.06% |
| Native on matching source redlines, n 3,735 |66.96% |57.75% |71.35% |74.54% |
| Source-H output on matching source redlines, n 4,027 |94.81% |75.07% |96.87% |99.06% |

Literal use of each generated file's **own** timing points yields union shares 49.39%,49.03%,40.16%,26.70% in the four native outputs.
Those numbers describe the arbitrary 120-BPM export placeholder. They are not evidence of musical tempo estimation; the source-grid comparison is the meaningful paired measurement.
A10 ms tolerance against a union of fine grids can have high chance coverage, and mumei already has source offsets outside the union. Neither metric is a universal “good rhythm” criterion.

**Lead verdict: supported as a contributor, undetermined as a representation-level cause.** Native H duplication/irregularity materially increases opportunities for extreme short LN in this endpoint.
The nearly unchanged median gaps and even lower pooled H count show why average density and median spacing miss the local defect.
Against the stronger claim:29 short LNs remain under source H; the ms hazard has explicit history/age inputs capable of representing suppression; earlier support restrictions already prevent some close same-finger actions.
The saved comparison cannot separate lack of metrical structure from insufficient training, weak audio discrimination, repeated firing around an onset, or a row policy that chooses to release when continuation is legal.
A matched trained comparison changing only timing representation/slot use, with same data, budgets, seeds and row policy, is absent; that missing evidence would settle the stronger attribution.

The early opportunity-insertion result is unusually solid evidence for a **schedule effect**: required H and seed are fixed while LN duration changes substantially across 24 pairs per intervention.
Its interpretation was correctly bounded: altered RNG consumption and timing features are part of the intervention. It does not identify a release-hazard mechanism, because frozen R1 did not have that module.
The one-millisecond full-hold release case is a stronger localized mechanism: the declared survival-to-deadline rule itself creates the probability atom.
At the reproduced Airborne state, the document reports 34.4447% last-clock mass versus .1364% after conditioning, and paired native output changes 1 ms RH to 63 ms. I checked the law, not a fresh replay.

The original flat joint distribution is mathematically legitimate: `p(time|history,audio) p(row|time,history,audio)` is not intrinsically a bad factorization.
The later **information restrictions**, obligatory H plan and small finite consequence horizon are substantive assumptions beyond writing that factorization.
The local thinning/distillation attempt already showed that fixing very short repeats can increase LN prevalence and produce silent tails; its own failed guards were appropriate.
The input audit establishes row-content sensitivity that violated the revised interface; it does not establish that this sensitivity caused bad play. Removing it was an ownership decision with a testable cost.
The planned model then develops starvation, and bounding its history fixes three witnesses while leaving a fourth and introducing concern about loss of useful rhythmic memory after waits.

Typed resource planning has a defensible benefit: it guarantees that the chosen mark has a realizable column assignment under the declared recovery envelope.
Its cost is equally concrete: R1 cannot change counts, LN births or release identity after the typed mark is fixed. A better geometric scorer cannot repair every bad typed plan.
Later restoration of row-owned decisions is not evidence that every typed abstraction is wrong; it does show that the experiment moved responsibilities the stated interface intended to keep together.
Support improvements are mechanical facts. Absence of<=40 ms LNs under HR=50 is imposed, not learned; subsequent lowering toHR21/HR 1 reopens that failure family.

**Question 6: what the memory failure means.** The checked raw record is 384 updates, not a sustained training study:768 thirty-two-second examples from 417 charts/383 groups.
It adds 3.03M parameters, initializes their outputs at zero and trains all old/new modules, with learning rates 3e−5/3e−4; fit seed 274100, one run per arm.
Memory reduces mean Stream pressure from continued-baseline .78553 to .38230, but the common unfitted checkpoint is .04447; it also misses the star-error and restored-control guards.
The nine Stream cases are three songs×three seeds. All 84 exports complete; maximum memory startup 1.591 s and two-second service 1.141 s pass the recorded cached-Mel timing limits.
The failure is excessive native activity/control response, **not** inability to run attention or meet those measured deadlines. Zenithfall H count 3,582→5,514 is a documented witness while rows are already thinning chords.
Joint fitting also changes the audio encoder and H base. The matched baseline's regression and later base-substitution diagnoses prevent attribution solely to the attention readers.
One reader-null diagnostic shifts median wait 8.2% on 12 prefixes; that is not an ablation showing memory is harmful over whole charts.
The result is strong enough to reject this checkpoint under its frozen guards, too small/confounded to reject the idea or prove that more capacity or training would solve it.

## 4. Direction

| Attempt group | Strongest case for the direction | Strongest case against | Call / confidence |
| --- | --- | --- | --- |
| Separate learned skeleton into R1 | Reuses a trained row prior; schedule sensitivity and source-H interventions can localize failures. | Compulsory timing can place the downstream model in a region where no acceptable realization remains; inherited R1 evidence does not transfer automatically. | Sound diagnostic bridge, weak evidence for a permanent architecture. **High**. |
| Joint hazard + rows + full audio | Exact normalized waiting law, native timing support, shared supervised audio and direct row audio; no need to declare one source chart uniquely correct. | Teacher-forced row loss at source times does not train sampled H choices to preserve native organization; millions of negative clocks and useful rhythmic phase share one mechanism. | Reasonable baseline whose quality remained unestablished. **High**. |
| H-only history and bounded residual | Exact ownership isolation; removes reproduced long-wait veto without forcing activity through real rests. | Forgets useful timing organization with the same decay; cannot adapt H pacing to different committed TAP concentration under identical H history. | Useful failure-specific intervention, not demonstrated general solution. **Medium**. |
| Typed planner / restored R1 / joint releases | Each identifies a real responsibility: resources, complete action consequences, and waiting before release. | Repeated boundary moves change representable decisions, support, losses and data together; few controlled native comparisons isolate the intended cause. | Restoring decisions that must interact is well motivated; superiority of the final law remains unproven. **Medium**. |
| Memory and segment plans | Complete-song relationships and persistent held/TAP roles require information beyond immediate rows; implemented memory actually receives gradients. |384 updates for memory and 32 for segment planning are weak architecture tests; segment code collapse removes the hypothesized communication channel. | Keep hypotheses open; reject those endpoints, not the concepts. **High**. |
| Clean/fresh ordinary learning | Direct supervision of ordinary examples attacks proposal quality rather than only downstream penalties. | Fresh four-song fit is 512 updates on 64 seconds total; clean 4096 changes initialization under one recipe, not all architecture assumptions. | Relevant direction; evidence supports “not yet learned,” not “ordinary learning failed in principle.” **High**. |

**Question 8: what skeleton determines at the end.** There is no single qualified final generator; the tag retains independent-R, joint-R and segment variants.
Native H is produced lazily in a rolling preview ahead of rows; it is not necessary to compute a whole-song skeleton before the first row.
In the late H→R1 family, H fixes every head-bearing timestamp and requires at least one attack there. R1 controls chord size, columns, TAP/LN and H-coincident releases; it cannot skip or move H (C3, `features.py:140`).
Independent R additionally fixes that some release must happen at each chosen R time. Joint-R changes that: row wait/release energies determine whether R happens, so pure releases are no longer an immutable upstream schedule (C6).
H nevertheless bounds possible density, rest structure, available fingers and attainable difficulty. Sixteen-head preview is not permission for R1 to rewrite H.
Documents eventually acknowledge this explicitly: low-request necessary-H floors above 3.0 and failed screened publication under dense H refute universal downstream recoverability (`head_audio_control_interaction.md`, `head_materializer_composition.md`).
Some earlier work operationally attempted recovery by row resampling, penalties or composition while keeping H fixed. The later documents do **not** uniformly claim R1 can repair arbitrary skeleton errors.

## 5. What was overlooked or never questioned

**Question 9: candidate assumptions, ranked by consequence.** Most were eventually questioned in prose; the gap is usually an absent decisive comparison, not total conceptual blindness.

1. **No measured trained contrast isolates musical timing structure from free-ms recurrence.** C3 has H age/history but no persistent metrical coordinate; hierarchy proposal explicitly unimplemented. New four-song audit shows close duplicates despite normal-looking median gaps. This matters more than whether the export clock is integer milliseconds.
2. **Finite ownership constraints were treated as design answers before their quality costs were measured.** The contract restricts H from row-content history while row recovery later re-enters through R windows. C5 and C6 establish necessary coupling; no broad comparison establishes how much committed action response H should read. Problems F/G.
3. **Training coverage was repeatedly too small to adjudicate architecture.** Memory 384 updates, segment 32, ordinary 512/four songs; clean 4096 has broader 3,736-chart exposure but only one run/arm. No convergence or learning-curve scaling study with independent training seeds establishes a representation ceiling. Problems J/K.
4. **Shared fitted rhythm was measured late relative to repeated density/control tuning.** Earliest tests had P/R/F1 and source grid audits, so “no timing evaluation” is false. But the late lattice diagnostic is only 21 selected scopes, and my redline comparison was not a selection criterion. Problems C/E/H.
5. **No matched comparison of timing alternatives under the same audio encoder and budget.** BeatThis helped a 770k frame predictor on six songs; canonical joint models then change frontend, objective, decoder and row coupling. The recorded evidence does not decide pretrained timing evidence versus learned joint audio. Problem E.
6. **No evidence that conditional source likelihood identifies the desired H base separately from historical residual.** The drift decomposition and memory base swaps explicitly expose compensation; their sums fit source data while native density rises. The architecture's “audio base” name should not be read as calibrated musical activity. Problems D/E/H.
7. **Local timing and action ownership were not enough to preserve LN organization.**155/191 extreme late tails occur at the next close H, yet 29 persist with source H; the known-failure Stream/STYX ownership split also differs. A single H-only or R-only diagnosis leaves a documented counterexample. Problems A/B.
8. **Meaning of ordinary remained only partly sampled.** The four-song fresh pilot includes held-stream, mixed and LN-body examples but not a trained ordinary population; a controlled source-H output can still raise chord density. That prevents either timing or R1 from being acquitted by the other's failure. Problems C/K.

The **millisecond clock without grid** was explicitly discussed from the start and challenged by the late hierarchy proposal; it was retained to preserve irregular/technical support, not accidentally overlooked.
**Strict next-event sampling** is narrower than `5c56e28:docs/formulation/notation.md:226`, which permits joint provisional futures and prefix commit. However, finite H lookahead, forked row continuations and segment plans were tried; “no planning” is also false.
What remains untested is a learned jointly revised future timing-and-row proposal against the otherwise matched late baseline; the hierarchy/ordinary-fusion documents are proposals, not that experiment.
**No external beat tracker** is refuted by C1/BeatThis artifacts. It was deferred after the initial pilot, with no canonical-frontend matched reprise found.
**Timing judged only downstream** is refuted by section 2's inventory. What was missing was a validated timing evaluation that distinguishes good alternative rhythms from native duplicates and loss of shared phase.

## 6. Worth keeping

- **Exact timing/row probability and replay distinctions** (`checked-code`, C2/C3/C6): survival is not an empty physical row; H versus pure R versus H-coincident release are distinct. These survive a baseline restart as tested semantics, without endorsing the architecture.
- **The schedule-sensitivity experiment** (`checked-artifact`):24 paired runs per intervention demonstrate that optional event opportunities alter LN duration and amount. It is stronger than intuition that R1 will harmlessly ignore extra times.
- **The separate conditional-release diagnostic** (`checked-code` plus bounded `doc-claim`): a specific deadline atom has a calculable mass and a paired repaired witness. Preserve the phenomenon and the distinction between raw and conditioned waiting laws.
- **Source-H and source-prefix controls** (`checked-artifact` ordinary exports; earlier studies `doc-claim`): they expose failures remaining after timing is supplied. The four-source hash-verified audit and its parser are reproducible without a model run.
- **Full-audio versus chart-causality tests and causal memory index cap** (`checked-code`, C2/C7): complete future audio is permitted; a hazard-bin anchor must not admit target chart events into history.
- **Honest failed qualification records** (`checked-artifact` memory and clean endpoints): passing runtime, legality and export does not overwrite pressure/control failure. The native/checkpoint identities are worth retaining with their limits.
- **Separate source grid, fitted lattice and audio correspondence observations** (`checked-code` and `checked-artifact`): useful diagnostics when their scope, tolerance and lack of quality semantics stay explicit. None should be promoted alone to a playability oracle.

## 7. Claims worth re-verifying

| Claim / missing evidence | Exact follow-up location or command | Why unresolved |
| --- | --- | --- |
| Original first-pilot benefit survives integer materialization and canonical Mel | `artifacts/audio-skeleton/20260923-v1/training/{local,beat}-pilot-v1/result.json`; `099cb66:src/ensomi_model/research/audio_skeleton/training.py:52` | Original F1 predates materialization correction; no matched canonical-feature rerun located. Requires new fit/generation, not permitted here. |
| Early conditional release fixes the claimed exact Airborne state | `artifacts/joint-audio/20260924-head-recovery-v1/release-law-audit/` and `artifacts/joint-audio/20260924-feasible-release-v1/`; inspect freeze/result and observer script | Code supports mechanism; historical probabilities/native hash identity not independently replayed in this review. |
| Memory failure persists with adequate fitting or isolates attention | `artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/{preparation-final-v2,baseline-384,memory-resumed-384,comparison}/` | Single fit pair and three-song Stream panel cannot settle general architecture; would require new matched runs, not a missing descriptive statistic. |
| Fragmentation contribution beyond four sources / another seed | `python3 /tmp/lineage-review/03-audio-timing-skeleton/chart_timing_audit.py`; inputs explicitly embedded | Reproduces this audit only. Additional saved clean endpoint charts can be paired through its source plan; no broad causal estimate supplied here. |
| Final clean endpoint quality versus published 512 summary | `python3 /tmp/lineage-review/03-audio-timing-skeleton/artifact_metrics.py`; `artifacts/joint-audio/20260928-clean-joint-proposal-v1/native-4096-*/{result,cases}.json` | Numeric failures checked; semantic review still pending. No model execution needed to review remaining artifacts. |
| Exact reusable encoder inventories | `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python /tmp/lineage-review/03-audio-timing-skeleton/encoder_counts.py` | Counts surviving checkpoint tensors without running models. Frozen BeatThis parameter count/receptive-field internals were not independently inspected. |

The scripts above have no hidden downloads or writes outside scratch. No executable model-run command was attempted or deferred after a partial launch.
I have not invented a training command for an unimplemented hierarchy; the missing item is an implementation/comparison, not a recoverable CLI invocation.

## 8. Cross-slice notes

Evaluation slice: the extreme<=40 ms tail regression catches failures that the 80 ms aggregate burden accepted; source-grid matching alone would also miss many onset-adjacent duplicates.
Row/response slice: H substitution leaves 29 short LNs and four same-lane<40 ms pairs in the fresh endpoint; this review does not assign all responsibility upstream.
Data slice: the checked 09-24 audit has 11,136 paired-eligible TRAIN charts/3,081 groups,490 missing-audio charts across TRAIN/VAL, and one encoded-audio identity crossing split groups. Pilot group counts should not be called verified independent songs.
Memory/runtime slice: the 384-update fit survives resumed segments and passes cached-Mel publication guards; the quality failure is separate from resource implementation.
Ownership slice: late joint R directly reads full row context, unlike the earlier skeleton-preference restriction. Compatibility flags preserve both laws, so “the final R” needs an explicit checkpoint/mode.

## 9. Failed paths and unfinished work

Completed all nine numbered questions within the bounded slice. The report is an evidence review, not a new architecture or roadmap.
An initial artifact listing respected gitignore and returned nothing; corrected with `rg --files --no-ignore` inside named experiment directories only.
Two metadata readers initially assumed the wrong nesting (pilot assessment conditions, source-plan entry wrappers); corrected before using their output.
Checkpoint counting initially stopped at RNG tensors and omitted attention `_weight` names; corrected to descend into the state dict and include attention projection weights. Final counts reconcile with the 3,032,532-parameter memory addition.
The cleanup log removes legacy/shared caches and intermediate clean-fit recovery weights; the milestone checkpoints used here survive. No pivotal claim was blocked solely by a confirmed deletion in this review.
I did not exhaust every one of 202 commits' diffs, every typed-control intermediate, all per-experiment Agent Notes, or the 84 memory Lens/export trajectories.
Raw late charts and decisive training/qualification records received priority. Historical visual/audio claims remain unverified; some smaller-attempt training populations/wall times remain explicitly not recorded.
No files outside this report and `/tmp/lineage-review/03-audio-timing-skeleton/` were created or changed; no git mutation or subagent was used.

## 10. Inconsistencies and items for the human

1. **“No grid/minimum gap” combines three different claims.** `checked-code`: no explicit metrical model in late H, but per-lane recovery and four-head capacity exist; fresh HR/RH are deliberately 1 ms. `checked-artifact`: source-H still leaves 29 extreme short LNs, so timing representation alone is not an adequate attribution.
2. **No external tracker / no attention are inaccurate descriptions of the lineage.** `checked-code` and `checked-artifact`: BeatThis features are present in the first pilot and complete-song attention from 09-24. The unanswered question is usefulness and training sufficiency, not existence.
3. **The release information contract and late joint-R mode describe different dependencies.** `checked-code`: C6 reads full row history before R time selection, whereas the earlier contract excludes learned row history from skeleton preferences. This is an intentional new law, so checkpoint/mode identity must accompany claims about ownership.
4. **Repairing the early deadline atom did not remove deadline atoms permanently.** `checked-code`: joint-R skips feasible-wait normalization and forces the last eligible hazard; `doc-claim`:130/1,842 late Stream tails hit nonterminal deadlines. This deserves separate interpretation from the old 1 ms counterexample, not silent credit for a universally solved release problem.
5. **The clean-joint document is behind the surviving experiment.** `doc-claim`: the document presents 512 updates and calls 4,096 future work. `checked-artifact`: all three arms reached 4,096 and completed 28 cases each, with 19/18/17 numeric failures and semantic review pending.
6. **The strongest architecture rejection is weaker than the endpoint rejection.** `checked-artifact`: memory 384 updates and ordinary 512 on four sources; `doc-claim`: segment 32 updates. All are bounded failures. Whether ordinary quality is constrained mainly by representation or training remains unresolved by these budgets and single fitted seeds.
7. **Display timing points cannot answer the musical-grid question for generated files.** `checked-artifact`: native outputs have a 120-BPM placeholder, while the source-H output can preserve every correct H yet score poorly on that placeholder. This report uses paired source redlines and labels the literal own-header statistic separately.
8. **Ordinary-versus-expressive priority needs a human criterion, not an implicit change of legal support.** `doc-claim`: early requirements emphasize irregular technical support; later feedback asks ordinary 2–6★ first. The lineage never measures an agreed tradeoff between preserving that support and reliably selecting ordinary rhythm, and this review does not resolve the preference on the human's behalf.
