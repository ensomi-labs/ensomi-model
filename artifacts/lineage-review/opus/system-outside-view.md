# Lineage review, outside view: time representation and what working generators consist of

Independent reviewer (Claude Opus, control plane), 2026-09-30. Written without reading `artifacts/reports/lineage-review/` or any `lr-*` job. Scope: the `codex/audio-skeleton` lineage (`r1-restored-6.75m..audio-joint-2026-09`) seen from the whole system and from outside, centred on how event time is represented. Not a design and not a roadmap.

Evidence grades: **doc-claim** (lineage document, not rechecked), **checked-code** (read in the tree export at tag `audio-joint-2026-09` or `main`), **checked-artifact** (computed or read by me from raw files on bings-mac), **external** (URL).

## 1. What I read and checked

- Notes: `RESEARCH.md`, `artifacts/lineage-review/README.md`, `artifacts/audio-skeleton-human-feedback-index.md` (sections 2, 3, 5).
- Lineage docs (`audio-joint` tree, `docs/research/`): `audio_conditioned_choreography.md`, `audio_skeleton_information_contract.md`, `audio_rhythm_hierarchy_zh.md`, `clean_joint_proposal_learning.md`, `ordinary_expert_from_scratch_zh.md`, `audio_joint_expert_question.md`, parts of `typed_audio_continuation.md`, `joint_r1_release_decisions.md`. Formulation `main/docs/formulation/notation.md`.
- Code: `audio_skeleton/{model,audio,config}.py`; `planned_audio_continuation/{model,generation,spacing,buffering,intervals}.py`; `segment_audio_continuation/{model,bootstrap}.py`; `joint_audio_continuation/timing.py` (`sample_hazards`); `osu_core/difficulty.py` (port of the official strain model); `gameplay_evaluation/head_difficulty.py`; pre-V3 `src/ensomi_model/timing/` (listing and headers only); git `dfb4618` (legacy boundary).
- Raw artifacts on bings-mac (read-only): `artifacts/joint-audio/20260928-ordinary-scratch-v1/capacity-native-512-v1/` (4 songs x {native, source_H, source_prefix}, seed 290031, checkpoint `f7b2037c...`) and `artifacts/joint-audio/20260928-clean-joint-proposal-v1/native-{0512,2048,4096}-{inherited,early,fresh}/` (7-24 cases each), their source `.osu` files in `dataset/0/`, and a random corpus sample from `artifacts/oracle-time-review/20260915-adfb1ee/catalog.json`.
- Reference projects on the mac: `ref-proj/Mapperatorinator` (HEAD `4d2e548`, 2026-05-30) README, `osuT5/osuT5/event.py`, `inference.py`, `inference/postprocessor.py`, `calc_fid.py`; `ref-proj/Mug-Diffusion` (HEAD `c2903d4`, 2023-06-06) README, `configs/mug/*.yaml`, `mug/data/{convertor,utils}.py`. Web: official osu!mania strain code, Dance Dance Convolution, arXiv 2311.13687.

Measurement scripts (scratch, not in the repo): `/tmp/lineage-review/opus-system-outside-view/{timing_audit,clean_joint_batch,offset_batch,tol_batch,f1_batch}.py` on the mac, copies in the control-plane scratchpad. Run from `~/ensomi/ensomi-model` with system `python3`, stdlib only, each under 30 s. Method: parse `.osu`; H = distinct times with a TAP or LN head; grid = the **source chart's own uninherited timing points** (the generated `.osu` carries a constant 120 BPM placeholder); a head is on-grid if within ±2 ms of a k/d beat position, d in {4, 6, 8}; chance = the same share for uniform random times over the span (4,000 draws).

## 2. Timing in the lineage

### 2.1 Output spaces, resolution, minimum gap

| Stage (date) | H output space | R output space | Minimum gap enforced | Grade |
| --- | --- | --- | --- | --- |
| `audio_skeleton` pilot (09-23) | Frame classifier on 10 ms Mel frames: up to 2 head slots per frame (cumulative sigmoid) plus a tanh in-frame offset; threshold decoder | Same, 2 release slots | None beyond 2 per frame | checked-code |
| Flat joint model (09-23..24) | One discrete-time hazard for every event at integer ms (10 logits per 10 ms audio frame), sampled with a carried Exp(1) threshold; row sampled given time | Same hazard; row chooses heads/releases | "No peak NMS, no fixed slot cap, no subdivision whitelist" | doc-claim (`audio_joint_expert_question.md` l.95-97) + checked-code (`sample_hazards`) |
| Planned / controlled / clean-joint (09-24..28) | Per-ms Bernoulli hazard; optional bounded mode = audio base + history residual bounded by 4 and gated by exp(-dt/1000 ms) | Per-ms release-clock hazard between heads from LN state, audio, H preview; later (`r1_joint`) derived from R1's wait/release scores | Per lane only: HH/HR/RH recovery (20 ms default; 60 ms in the clean-joint 4096 `result.json`); across lanes only "any five consecutive H span >= gap" (`next_head_earliest`, `check_head_capacity`). Two distinct H 1 ms apart are legal | checked-code (`spacing.py`, `generation.py`, `buffering.py`) |
| Typed resource plan (09-25) | Native-ms event with TAP count, new-LN count and release mask | Inside the event mark | Recovery envelope | doc-claim |
| From-scratch ordinary expert (09-28) | Unbounded per-ms hazard (`bounded_head=False`) | `r1_joint` | Per lane 20 ms | checked-code (`bootstrap.py`) |

The H loss is a per-millisecond binary cross-entropy (`interval_losses`, checked-code): a prediction 1 ms from the true head counts as a miss at one bin and a false alarm at another. There is no temporal tolerance in the objective and no peak suppression at sampling; refractoriness after a sampled head must be learned from the "time since last H" clock.

### 2.2 Is a beat, tempo or subdivision grid represented anywhere?

- **Inputs.** Only in the 09-23 pilot, as an optional frozen BeatThis embedding (default off). With it, head F1 rose 0.677 to 0.732 on six songs (doc-claim, `audio_conditioned_choreography.md` l.89). Every joint model after it states it uses no redline, BPM or BeatThis (doc-claim, `audio_joint_expert_question.md` l.37; consistent with the code I read).
- **Outputs, losses:** none. Export writes `0,500,...` "Constant 120 BPM for editor and scroll only" (checked-artifact, `generated.osu` headers).
- **Evaluation:** one descriptive self-fitted lattice (`gameplay_evaluation/rhythm_lattice.py`, 09-28) applied to three 6 s windows and nine references, not to redlines and not in the regression suite (doc-claim + checked-code header).
- **Stated reasons:** "redlines are not timing truth", "no redline grid is required", preserve Tech, high fractions, 1 ms cross-lane offsets and dumps (`audio_skeleton_information_contract.md` l.16, l.317; `audio_conditioned_choreography.md` table l.204-215). The contract (`notation.md` l.28) *allows* internal beat coordinates and lattices. Leaving them out was a choice of the lineage, not of the formulation.
- **Earlier in the project** a full grid pipeline existed: audio -> Mel -> BeatThis -> GridFitter timing grid -> dense timing features -> mapper decoder, with an osuT5-style vocabulary (checked-code: `src/ensomi_model/timing/`, `models/mapper/v2_1/vocab.py`; system path deleted from README in `dfb4618`, 2026-09-01). The commit declares it off-limits as a V3 reference and records no reason.

### 2.3 Was timing ever scored on its own?

Once: head F1 on six songs in the 09-23 pilot. After that, H was judged through NLL, head-count ratios, same-lane gap counts, star floors, LN statistics and Lens pages. The agent's own audit found head F1 of 0.785 at 20 ms between alternative human charts of one song and used it to argue that F1 against one reference cannot define quality (doc-claim, l.75). That number could instead have served as a yardstick. I did that scoring here (2.5).

### 2.4 Reversals of the timing/rows dependency (doc-claim, from the feedback index timeline and docs)

1. 09-23: audio -> H/R -> frozen R1 (timing first). Result: legal but not playable; human reset target to playability.
2. 09-23..24: flat joint hazard; timing residual reads row history (rows -> timing). An audit found layout changes moved timing logits (contract doc table). Result: millisecond repetition storms, silences.
3. 09-24: information contract; H independent of rows; R reads LN state only; rows read H (timing -> rows), on the human's direction (V29-V31).
4. 09-25: typed plan puts TAP/LN counts and release identities upstream in timing (timing owns part of rows). Human corrected (V41); restored to R1 (`c5b7db8`).
5. 09-25..: R's release window supplied by R1's feasibility (rows -> R).
6. 09-28 RC: `r1_joint`, release hazard derived from R1's scores (R folded into rows).
7. 09-28: ordinary expert, fresh joint H + R/R1; H still independent of rows.

Each reversal changed who owns LN ends and counts. None changed the H output space (a per-ms hazard with no grid). The one representation proposal that would have changed it (`audio_rhythm_hierarchy_zh.md`: slot-based H with main layer and ornaments, at most one H per slot) was written on 09-28 and never implemented. Its own text names the defect: sampling a narrow peak repeatedly "produces multiple H near the same peak".

### 2.5 Measurements (checked-artifact)

**Corpus baseline.** A random TRAIN sample, star from `metadata.json`. 3.5-4.5 stars: n = 141 charts, 148,581 H. 2-6 stars: n = 200 charts, 171,632 H.

| Corpus | median on-grid 1/4-1/6-1/8, ±2 ms | charts with >= 90% on-grid | chance (median) | consecutive-H gaps <= 10 ms | LN <= 40 ms | LN tail at the next H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.5-4.5 stars | 1.000 (q01 0.931) | 99.3% | 0.144 | 12 / 148,440 | 12 / 26,851 | 54.7% |
| 2-6 stars | 1.000 (q01 0.978) | 100% | 0.143 | 42 / 171,432 | 17 / 26,633 | 58.7% |

**From-scratch ordinary expert** (512 updates; same model and seed; `source_H` = real H from BOS, `native` = model H):

| Song (source stars) | Arm | H | gaps <= 10 ms | min gap | on-grid ±2 ms | chance | LN <= 40 | LN <= 80 with next H <= 40 ms | head F1 @20 / @5 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dawn (4.00) | source / source_H / native | 1542 / 1542 / 1231 | 0 / 0 / 152 | 42 / 42 / 1 | 1.00 / 1.00 / 0.25 | 0.19 | 0 / 1 / 31 | 0 / 0 / 31 of 46 | - / - / 0.67 / 0.45 |
| Sulyvahn (3.99) | same | 865 / 865 / 773 | 0 / 0 / 80 | 75 / 75 / 1 | 1.00 / 1.00 / 0.26 | 0.17 | 0 / 9 / 47 | 0 / 0 / 43 of 57 | 0.63 / 0.39 |
| mumei (3.99) | same | 995 / 995 / 1128 | 0 / 0 / 154 | 39 / 39 / 1 | 0.96 / 0.96 / 0.14 | 0.08 | 0 / 10 / 68 | 0 / 4 / 59 of 72 | 0.56 / 0.30 |
| Kill The Beat (3.77) | same | 625 / 625 / 603 | 0 / 0 / 77 | 62 / 62 / 1 | 1.00 / 1.00 / 0.24 | 0.09 | 0 / 9 / 45 | 0 / 0 / 41 of 72 | 0.62 / 0.41 |

The LN <= 40 counts reproduce the document's table exactly (0/1/31, 0/9/47, 0/10/68, 0/9/45), and the native total is the 191 extreme short tails of the tagged notes. Native: 463 of 3,731 gaps (12.4%) are <= 10 ms, against 0.008% in the corpus. At the same row model and seed, real H cuts LN <= 40 ms from 191 to 29 (-85%). Of native LNs <= 80 ms, 174 of 247 (70%) have their next H within 40 ms; under real H, 4 of 137 do. The row model still adds LN under real H (Sulyvahn 368 LN against 172 in the source).

**Clean-joint three-arm run, 4,096 updates** (24 cases per arm, 5 songs; min per-lane gap 60 ms in `fresh`):

| Arm | H | gaps <= 10 ms | on-grid ±2 ms (H-weighted) | chance | LN | LN <= 40 | LN <= 80 | of those, next H <= 40 ms | LN tail at next H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inherited | 49,804 | 145 (0.29%) | 0.167 | 0.157 | 24,019 | 1,320 | 9,555 (40%) | 1,690 | 60% |
| early | 50,931 | 138 (0.27%) | 0.170 | 0.158 | 24,798 | 1,276 | 9,667 (39%) | 1,788 | 57% |
| fresh | 59,573 | 153 (0.26%) | 0.157 | 0.157 | 16,122 | 655 | 4,834 (30%) | 778 | 44% |
| sources (5 charts) | 7,465 | 0 | 0.90-1.00 per song | 0.10-0.18 | 3,479 | 0 | 439 (13%) | 4 | 12-76% |

On-grid share against the 1/4 beat alone, for the s0 case of each song in all three 4,096-step arms (15 charts). At ±2 / ±5 / ±10 ms, generated heads reach 0.05-0.27 / 0.14-0.59 / 0.31-0.84; the sources reach 0.73-1.00 at every tolerance; chance is 0.03-0.07 / 0.07-0.16 / 0.13-0.31. Searching a constant offset of ±100 ms raises native on-grid share only to 0.15-0.33 against 0.10-0.21 for random times. Zenithfall (Camellia, 210 BPM, 10 redlines) is at chance even at ±10 ms. Head F1 against the source, @20 ms / @5 ms: generated 0.55-0.83 / 0.13-0.60. Human alternative mania difficulties of similar density score 0.886-0.96 / 0.886-0.96 (Classic Pursuit Insane and Great Escapades, three Blizzard Heights difficulties, Max Burning Lv.12/14/16; n = 8 charts). Generated heads land near the right musical events but scattered by 5-20 ms. Human charts that differ in arrangement share the grid exactly.

Commands: `python3 timing_audit.py pair <source.osu> native=<gen.osu> source_H=... source_prefix=...`; `python3 clean_joint_batch.py`; `python3 tol_batch.py`; `python3 offset_batch.py`; `python3 f1_batch.py`; `python3 timing_audit.py corpus <catalog.json> 300 3.5 4.5` (and `200 2.0 6.0`).

## 3. Verdict on the lead

Lead: part of the fragmented-LN and short-attack complaint is a timing-representation defect (near-duplicate head or release times from a millisecond hazard with no grid and no minimum gap), not a row-model or response defect.

- **Supported, strongly, for the from-scratch ordinary expert** (09-28, V79-V81). Near-duplicate H are 12% of gaps. 70% of its short LNs close on an H that follows within 40 ms. Replacing only the H stream with real times, at the same weights and seed, removes 85% of LN <= 40 ms. The code permits the mechanism: cross-lane H 1 ms apart are legal, and no NMS or refractory rule exists. Caveat: 512 updates on 16 four-second units, so an undertrained hazard and a defective representation are not separated.
- **Weak for the clean-joint line**, the lineage's main end (4,096 updates, 3,736 charts). Near-duplicate H fall to 0.3% of gaps. Only 18% of LN <= 80 ms have the next H within 40 ms. Their LN tail-at-next-H share (44-60%) is at the corpus rate (55-59%). The excess of short LNs (30-40% of LN against 5.6% in the corpus) comes from placing one-interval holds in dense passages, for example Zenithfall at 71 ms per 1/4. That is the row model's and release law's doing, as the lineage's own replay attribution also found. Evidence against the lead lives here.
- **What the lead understates.** The larger timing defect is not near-duplicates but missing grid alignment. In every generated chart measured, head times sit 5-20 ms off the song's own subdivision, near chance at ±2 ms, while 99% of ranked charts are on it. This bears on complaint C ("subdivision too irregular", V70), which was never measured on its own. It may also bear on the row model: R1-derived row models were trained on snapped gaps, and at inference they receive jittered ones. That distribution shift is a hypothesis; I did not test it.

## 4. What working generators consist of

| | R1 (baseline) | Lineage end (clean-joint / ordinary expert) | Mapperatorinator (v32) | Mug-Diffusion | Dance Dance Convolution |
| --- | --- | --- | --- | --- | --- |
| Chart / time representation | Complete 4-lane rows at supplied times | Complete rows at sampled integer-ms times | Event tokens; time shift quantized to 10 ms; SNAPPING token per event; TIMING_POINT, BEAT, MEASURE tokens; HOLD_NOTE / HOLD_NOTE_END | Per-column channels (start, start offset, holding, end offset) on ~46.4 ms note frames with a continuous in-frame offset; latent 16 x 512 | Step placement per 10 ms frame, then step selection |
| Where timing comes from | The source chart (snapped) | Per-ms hazard from audio and H history; no grid | The model generates redlines first (optionally "super timing": 20 whole-song passes averaged), then resnaps every event to the model's snap divisor | Joint denoising; post-hoc `gridify` fits one BPM and offset and snaps to 1/1..1/32 | Peak-picking on a Hamming-smoothed onset curve (suppresses double peaks), threshold; selection gets beat phase to 1/16 and beats since/until |
| Audio encoder | None | Mel 10 ms; ~0.46M-param local TCN plus coarse 500 ms full-song attention | Whisper-style encoder over Mel frames | Mel (22.05 kHz, 128 bins) scale encoder into a UNet with attention and S4 | CNN + LSTM on spectrogram |
| Model / data / compute | 3.08M; 11,563 TRAIN charts; 6.75M onset exposures on 1 CPU thread | 4.7-4.8M; 16 units (expert) or 3,736 charts / 8,192 windows / 4,096 updates (clean joint); minutes to hours on an M5 | 219M; all ranked maps (HF `project-riz/osu-beatmaps`); ~5,700 GPU-hours over 261 runs | UNet 128 ch, mult [1,2,3,4]; dataset list published; size not verified | 35 h of charts, ~350k steps |
| Conditioning | Seed chart; none requested | Stars, LN fraction, style, scoped in time | Difficulty (SR), mapper ID, year, descriptors (user tags), keycount, hold-note ratio, CFG with negatives | osu! SR, Etterna MSD overall and per skillset (stream, jumpstream, chordjack, stamina, tech...), LN ratio, rank status | Difficulty |
| LN representation | Row actions with occupancy | Head, then a separate release hazard or R1 wait/release law | Hold-note start and end tokens, end snapped | Holding channel: one object with a duration, denoised jointly | n/a |
| Causality | Causal continuation | Causal; publish-ahead with head lookahead | 16.4 s windows, 90% overlap, sequential; each token sees >= 4 s of past tokens and 3.3 s of future audio | Whole song (<= 190 s) at once | Offline |
| Evaluation | Mechanical legality, replay, export | NLL, gap and LN counts, star floors, Lens pages, human inspection | `calc_fid.py`: FID on classifier embeddings, rhythm precision/recall/F1 against the real map, BPM MSE, SR, self-similarity-matrix RMSE | Not stated in the README | Placement F1 at ±20 ms, AUC, selection perplexity |
| Grade | checked (notes, code) | checked-code / checked-artifact | checked-code, README (external: github.com/OliBomby/Mapperatorinator) | checked-code, README (external: github.com/Keytoyze/Mug-Diffusion) | external: https://ar5iv.labs.arxiv.org/html/1703.06891 |

Also relevant: arXiv 2311.13687 (Yi, Lee, Lee, ISMIR 2023 LBD) calls tempo-informed, beat-aligned preprocessing "integral for a successful training" (external). The lineage cited it and set it aside because its "fixed beat-relative support ... do[es] not cover all required dense/irregular cases" (`audio_conditioned_choreography.md` l.282).

### Ingredients the project lacked, rejected, or never considered

| Ingredient | Status | Where the reason is recorded |
| --- | --- | --- |
| Musical time coordinate (generated redlines plus snap, or post-hoc gridify) | Rejected | Contract doc l.16 and l.317, expressivity table l.204-215, expert question l.34-38. Reason: expressivity of off-grid Tech and dumps. Not tested against the corpus, where 99% of 2-6 star charts are on grid |
| Beat features as input (BeatThis) | Tried once (+0.055 F1), then dropped when the Mel frontend was fixed | l.57-58: "optional later comparisons, not prerequisites". The human's V20/V21 choice of a simple, interpretable encoder is the context |
| Peak suppression or a refractory rule on H | Explicitly rejected ("no peak NMS") | Expert question l.96. A 27 ms same-lane factor was added as an optional decode prior, "not a general filter" |
| Timing scored alone (onset F1, on-grid) | Only in the first pilot | None; replaced by downstream metrics |
| Scale: parameters, all ranked data, GPU training | Never a variable | Expert question: "model parameter expansion has no necessity evidence yet" (l.142) |
| Whole-song joint generation (diffusion or windowed AR with future audio) | Partly: the full song is available as coarse audio context; the decision process is strictly causal per event | Contract: causal publication. The formulation allows provisional future rows, which the lineage did not use |
| LN as an object with an end | Rejected in favour of head plus a release clock or R1 release law | Contract doc; human keys V31 and V62 on release ownership (later dropped as a key) |
| Conditioning on a pattern-aware difficulty model (Etterna MSD) | Never considered, as far as I found | None |
| Reference generator on the same songs as a baseline | Never done in the lineage | Docs repeat "no matched comparison" (`oracle_time_m3_validation.md` l.210). On the mac, three Mapperatorinator inference logs from 2026-05-31 on one song (`dataset/0/1069336`) and an empty `artifacts/evals/issue110_beatthis_final0_100/mapperatorinator_v29_timer1_outputs/`; no Mug-Diffusion weights present (`models/ckpt` missing); the artifact index has no reference outputs |

### Which complaints the references address by construction

- **A (long jacks, same-lane attacks under tens of ms):** Mug-Diffusion allows at most one start per column per 46 ms frame and conditions on MSD chordjack and stamina. Mapperatorinator and DDC rely on learned statistics, snapping and peak suppression. None guarantees difficulty-appropriate jack length. A is the complaint least addressed by construction.
- **B (fragmented LN):** Mug-Diffusion's holding channel and Mapperatorinator's snapped end tokens make an LN one object whose end is decided with its start or in the same denoising pass. Grid snapping removes the near-duplicate-head route to micro-LNs. By construction, B is addressed only in its timing-driven part.
- **C (not ordinary, irregular subdivision):** addressed by construction in all three references (redlines plus snap; post-hoc gridify; beat-phase features).
- **D (no breathing across scales):** partly. Mug-Diffusion denoises the whole song jointly. Mapperatorinator has song-position tokens and 16 s windows, and measures self-similarity RMSE. Neither guarantees it.
- **E (weak use of audio):** addressed by scale (encoder-decoder attention over the window; a large UNet) rather than by a specific mechanism.

## 5. The player response state as novelty

**Where it holds.** No reference uses a player state inside generation. None scores or accepts candidate continuations by a causal, per-candidate response computed over real time from the committed prefix. None has a profile-parameterised response that controls could target. Mapperatorinator and Mug-Diffusion use whole-chart labels (SR, MSD, descriptors) only as conditioning. The formulation's frontier over legal continuations is a different object from theirs.

**Where it does not hold.**
1. The official osu!mania star rating is already a causal stimulus-response state. It keeps a per-column strain decaying as 0.125^(dt/1 s), an overall strain decaying as 0.30^(dt/1 s), a 1.25 factor for notes under an active hold, and a logistic release-timing term with a 30 ms threshold. The value is the highest column strain plus overall strain, over 400 ms sections (external: https://github.com/ppy/osu/tree/master/osu.Game.Rulesets.Mania/Difficulty). That is most of what problem G asks for: per-finger strain with decay and recovery. The repository has a port (`osu_core/difficulty.py`, checked-code), and the lineage used it only as a whole-chart scalar and a mandatory-H star floor.
2. Etterna's MSD is a pattern-aware, per-skillset difficulty model, and Mug-Diffusion conditions on it.

The novelty is using such a state online, per candidate, with a calibrated target. That is unimplemented and unvalidated: the formulation leaves the target response undefined (README observation `o-response-undefined`). Nothing measured shows that references fail on A or B for want of it, because none was run on the same songs.

## 6. The strongest case for ensomi's different design

Ambient play needs a chart that follows music already playing. It must publish settled coverage ahead of playback within a latency budget, accept mid-song scoped control changes that affect only the future, keep committed LN starts while their ends are still open, and later support local edits between a fixed past and a fixed future. Mug-Diffusion generates a whole song at once and has no notion of a committed prefix or a scoped live control. Mapperatorinator is closer: windowed, sequential, 3.3 s of future audio, a prefilled decoder, `add_to_beatmap` and `start_time`/`end_time` for partial remaps. But it is 219M parameters and GPU-oriented, and its timing is generated for the whole song first. Neither offers exact row legality, deterministic replay or an incremental LN protocol. A small, exact, causal row generator with a state that can be audited is a reasonable bet for that product. R1's exact contract, replay and export are real assets (checked by the baseline's tests, per RESEARCH.md).

This case does not extend to the time representation. The standing constraints say "complete audio may be used in training and inference; chart decisions and publication stay causal". A whole-song grid pass (beat tracking plus redline fitting, as the pre-V3 stack or Mapperatorinator's timing stage do) is a computation over audio, not a published chart decision. The formulation explicitly allows internal beat coordinates. The real-time vision therefore does not require a grid-free millisecond hazard.

## 7. The two lists

**Assumptions of the lineage that look unusual relative to working systems**

1. Event time with no musical coordinate: no redline, beat or snap in input, output, loss or export. Every reference and DDC has one, and 2311.13687 calls it integral.
2. A per-millisecond Bernoulli hazard with a zero-tolerance loss, no peak suppression and no cross-lane minimum between distinct H.
3. Timing not scored on its own after the first pilot; no onset F1 or on-grid metric in the regression suite, although the corpus has an unambiguous grid signature.
4. Scale held tiny while architecture conclusions were drawn: 3-5M parameters, 512-4,096 updates, 16 units to 3,736 charts, against 219M parameters on all ranked maps and ~5,700 GPU-hours.
5. Row models initialised from R1, which was trained on snapped supplied times, then fed jittered generated times; the input shift was never measured.
6. LN as a head plus an independently timed or sequentially chosen release, rather than an object with an end.
7. Quality handled by optional decode priors, thresholds and acceptance factors added after each failure, rather than by the learned distribution plus a structural time prior.
8. No reference generator run on the same songs, so no external yardstick for "ordinary".

**Assumptions that follow justifiably from the real-time vision**

1. Causal commitment of rows and no-row decisions, with incremental LN publication (end undecided at head publication).
2. Exact legality, replay and deterministic RNG streams, so scheduler order and query partitioning cannot change content.
3. Bounded per-step latency on local hardware; small models are justified for the serving path (not necessarily for a teacher or offline timing pass).
4. Scoped controls that change only the future and can switch mid-song.
5. A timing stream that runs ahead of rows (head preview), so rows can be decided with lookahead. The two-stage split itself is standard (DDC).
6. Full-song audio available before generation, which the vision grants. This point argues *for* a precomputed grid, not against it.

## 8. Inconsistencies and questions only the human can settle

1. The ordinary 2-6 star target (key 2) against the lineage's expressivity requirement (1 ms cross-lane offsets, high-fraction Tech, no grid). The corpus puts >= 90% of heads on 1/4-1/6-1/8 in 99-100% of charts. Is off-grid expression a requirement for the first target, or a later style?
2. Was the V20/V21 choice of a "simple interpretable audio encoding" meant to exclude beat evidence? BeatThis was dropped right after it helped in the pilot.
3. Does ambient play guarantee the full track before generation, via recognition and retrieval? If so, a whole-song timing pass is within the vision.
4. Why was the pre-V3 grid pipeline (BeatThis -> GridFitter -> mapper) excluded on 2026-09-01? The commit gives no reason; slice 02 may have the history.
5. Key 1: is the intended player response state the official strain model's kind of per-column decaying state (problem G), or something the formulation calls a canonical profile, which excludes individual capacity and fatigue?
6. The May 2026 Mapperatorinator runs: was a judgement made on their output?

## 9. Not verified, and what a follow-up needs

- Whether grid-snapped H alone would remove most of the fragmentation in the clean-joint line. That needs inference, which was excluded here. The cheapest test: rerun `planned_audio_continuation.generation.rollout(..., head_times=<native H resnapped to the source redlines at 1/4-1/6-1/8>)` on the 4,096 checkpoints, same seeds, and compare LN <= 80 ms and the on-grid share. The inputs are in `artifacts/joint-audio/20260928-clean-joint-proposal-v1/native-4096-*/*/rows.jsonl` and the sources listed in `clean_joint_batch.py`.
- Whether jittered H shift R1-derived row choices: compare `source_H` against `source_H` plus ±5-10 ms jitter on the ordinary expert (`capacity-native-512-v1`) with the same seed.
- Mug-Diffusion parameter count and data size: its weights are not on the mac. It would need the bundled checkpoint (download, not done) or the dataset list at mugdiffusion.keytoix.vip/dataset.html.
- Mapperatorinator's mania quality on these songs: `python inference.py audio_path=<song> gamemode=3 keycount=4 difficulty=4 ...` on the five clean-joint songs and the four ordinary-expert songs, scored with the scripts above and `calc_fid.py`. Running generators was out of scope here.
- Early joint models (09-23..24) were not measured; only the late outputs were.
- The corpus sample depends on `metadata.json` version matching (141 of 300 requested matched in the 3.5-4.5 band). The human alternatives used for F1 are eight mania charts from three songs.
- Every lineage number I did not recompute (pilot F1, lattice coverage, 27 ms corpus minimum, dataset sizes) is doc-claim.
