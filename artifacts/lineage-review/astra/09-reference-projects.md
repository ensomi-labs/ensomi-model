# Lineage review: 09-reference-projects

Review date: 2026-09-30. Slice: the outside view, reference generators, and the scope of player-response novelty.

## 1. Scope and sources

The strongest outside-view finding is a missing comparison, not a demonstrated winning architecture.
The inspected references combine substantially larger learned audio/chart models with explicit conditioning and practical output repair.
Neither proves that ordinary 2–6★ charts follow automatically from a representation, nor that Ensomi's response formulation is unnecessary.

Evidence grades used below:

- `doc-claim`: an author or experiment document reports it; this includes public READMEs, model cards, and the formulation's intended semantics.
- `checked-code`: implementation or configuration was read; this establishes a mechanism, not generated quality.
- `checked-artifact`: a raw result, ledger, manifest, or locally computed inspection was read; no training or generation was rerun.
- Judgments and negative search findings are explicitly inferences from those grades.

| Source | Scope checked |
| --- | --- |
| `ref-proj/Mapperatorinator` | README; V32 training/inference configs; model construction; metadata/timing/LN parsing; sequential generation; resnapping; evaluation code. Clone revision `4d2e5481d4394a67f67633340828eacf1d9caffa`, 2026-05-30. |
| `ref-proj/Mug-Diffusion` | README and Chinese README; `mug_diffusion.yaml`; feature labels; converter; audio/U-Net/autoencoder constructors; DDIM; WebUI; grid and minijack repair; dataset split. Clone revision `c2903d4e3d7216d94ef512a9872bc77423c6e0ac`, 2023-06-06. |
| Baseline and lineage | `5c56e28` and `/tmp/lineage-review/trees/audio-joint`; formulation, R1 restoration/distribution, audio choreography, memory fit, ordinary-expert documents and relevant code. |
| Historical claims | All 45 reference/BeatThis mention matches across the four requested exports, plus their relevant surrounding passages. Search saved in `/tmp/lineage-review/09-reference-projects/reference-mentions.txt`. |
| Raw evidence | R1 release metadata; both 1,200-update skeleton pilots; memory preparation/final/comparison reports; ordinary capacity-fit/data reports. |
| Human problem definitions | `.git/research-relay/notes/RESEARCH.md` and feedback-index sections 2–3. No private material was opened. |

Web search was available and used for official project pages, model/dataset cards, BeatThis's paper, and osu!'s rating documentation/source.
No dedicated Mapperatorinator or Mug-Diffusion research paper with controlled evaluation was located; neither inspected README supplies one.
Mug's linked dataset page failed over both HTTP and HTTPS, so its chart/song totals and released checkpoint identity remain unverified.
The official star-rating implementation inspected is current upstream source, not necessarily the `20241007` implementation used in Ensomi's experiments.

For compact citations below, `M/` means `ref-proj/Mapperatorinator/`; `U/` means `ref-proj/Mug-Diffusion/`.
These are local source paths, not assertions that the references govern Ensomi's design.
Raw numeric extraction and ratio calculations are reproducible with `/tmp/lineage-review/09-reference-projects/summarize_evidence.py`; output is `evidence-summary.json` in that directory.

## 2. Attempts

**Q1 — Side-by-side system inventory.** “Lineage end” is a family of incompatible experimental checkpoints, not a single accepted release.
The core, memory branch and fresh ordinary pilot therefore have separate sizes/budgets below.

| Ingredient | Mapperatorinator V32 | Mug-Diffusion inspected config | R1 restored | Audio-skeleton lineage end |
| --- | --- | --- | --- | --- |
| Chart representation | Autoregressive sparse event tokens: times, note types, columns, metadata, timing/SV. | 16-channel 4K note tensor, compressed by VAE to 16-channel latent; diffusion predicts the latent. | One complete four-lane row at supplied candidate time; joint categorical row scores. | H/R event times plus complete row choices; late experiments jointly score wait/release alternatives. |
| Time and source | Window-relative 10 ms tokens; audio predicts timing points/snapping; supplied timing optional; output resnapped to predicted divisors. | Fixed physical-time frames plus learned start/end offsets; configured note frame ≈46.44 ms; optional post-hoc BPM/grid inference. | Real source chart supplies all R candidate times and required H subset, in milliseconds. | Audio-conditioned waiting/event probabilities at native millisecond support; no required external beat grid. |
| Audio encoder | Jointly trained Mel→modified Whisper-family encoder, 128 Mel bins, 16 kHz/128-sample hop; encoder-only count not recorded. | Learned multiscale 1D Mel CNN, 49,392,192 parameters in inspected config; audio features enter several U-Net scales. | None. | Canonical 24 kHz/128-bin/10 ms Mel; small learned CNN; full-audio context and later attention/history variants. Encoder-only count not recorded here. |
| Total parameters | README reports 219M; HF advertises about 0.2B. Released checkpoint not loaded. | 156,536,384 configuration parameters, CPU-counted; release `model.yaml` unavailable, so not a verified release count. | 3,084,432 in release metadata. | Core2500 4,583,985; memory 7,616,517; fresh ordinary 4,782,754 (`doc-claim` sizes). |
| Training data | Ranked/approved filter, all four modes; dataset card has 213,068 charts/46,386 audio tracks, but exact retained V32/4K counts not recorded. | 4K; ranked/loved/graveyard and Malody-style controls; exact charts/songs not recorded in reachable sources. | 11,563 eligible TRAIN charts/3,169 song groups; distribution document reports 9,533 charts actually consumed in its 5M exposure audit. | H pilot 48 charts/songs; memory fit 417 charts/383 groups; fresh ordinary fit 4 charts/songs, 16 short units. These are fits, not the entire available corpus. |
| Training compute | V32 config: 700,000 updates, batch 32, accumulation 2; actual checkpoint wall time not recorded. README's 5,700 GPU-hours/261 runs is project-wide development, not one model. | Two-stage AE/diffusion recipe; batch 48 and 1,000 diffusion noise levels configured; optimizer steps, wall time, GPU training budget not recorded. | Six stages, 6.75M source-onset exposures, CPU1; model seed 172/sampler 471; optimizer-step total and actual cumulative wall time not established here. | Small CPU/MPS pilots; concrete budgets in attempt table. No aggregate lineage compute total recovered. |
| Conditions and training | Difficulty, mapper, year, mode, keycount, LN ratio, song position/length, SV, tags; metadata-prefix dropout plus supervised next-token CE. | SR, rank status, LN ratio, RC/HB/LN flags, Etterna skill values/labels; feature embeddings/dropout, diffusion denoising loss and CFG. | Observed seed and chart history; source CE plus separately trained preference residuals. No user difficulty/style request. | Scoped difficulty/LN/style controls and historical state; joint source likelihood, native recovery and actor/control variants. Effectiveness differs by checkpoint. |
| Sampling | Token sampling, V32 temperature .9/top-p .9, optional CFG; sequential overlapping windows; output repair. | DDIM with CFG; WebUI defaults 100 denoising steps, CFG 5, four samples; grid/minijack repair. | Sample from support-masked joint row probabilities, with exact replay. | Sample event waits and rows under legality/recovery support; some variants screen candidate continuations. |
| Causal or whole-song | Autoregressive tokens in overlapping windows; complete audio is loaded; normal pipeline assembles/postprocesses a chart before export. | Whole-song tensor jointly denoised; WebUI adapts latent length to audio. | Chart-history causal, with all source times known. | Chart-history causal, complete audio available; provisional lookahead and measured publication traces in some variants. |
| LN representation | Start, sustain, end events with lane information; not one preselected-duration token. | Start/offset, holding-body mask, end-offset channels per lane; decoder scans body until next start/end. | LN_OPEN/LN_CLOSE row actions and exact occupancy. | Same occupancy with changing release ownership/scoring; no universal duration-object replacement. |
| Authors' evaluation | Showcase/timing claims; code computes validation token CE/accuracy. No inspected controlled 4K quality benchmark, sample count, or variance report. | Showcase and author runtime claim: four 3-minute charts in about 30 s on 3050Ti/4 GB. No controlled quality benchmark found; split overlap discussed below. | Mechanical/reparse evidence, native panels, six release examples; release metadata explicitly leaves quality review open. | F1/NLL, native panels, pressure/star/LN proxies, export checks and Lens inspection; multiple pilots explicitly fail qualification. |

Inventory evidence: `checked-code` M/configs/train/v32.yaml:19, M/configs/model/default.yaml:32, M/osuT5/osuT5/model/modeling_mapperatorinator.py:55 and :74, M/osuT5/osuT5/dataset/data_utils.py:494, M/osuT5/osuT5/dataset/osu_parser.py:642, M/osuT5/osuT5/inference/postprocessor.py:588, M/osuT5/osuT5/utils/train_utils.py:247.
For Mug: `checked-code` U/configs/mug/mug_diffusion.yaml:1, U/mug/data/convertor.py:218, U/mug/diffusion/diffusion.py:23, U/webui.py:339 and :603, U/configs/mug/mania_beatmap_features.yaml:1.
For Ensomi: `checked-code` `5c56e28:src/ensomi_model/research/bounded_typed_continuation/generation.py:213`; `099cb66:src/ensomi_model/research/planned_audio_continuation/model.py:199`; `099cb66:src/ensomi_model/research/audio_memory_continuation/model.py:34`.
Dataset totals are `doc-claim`, not a verified V32 training manifest: [official dataset card](https://huggingface.co/datasets/project-riz/osu-beatmaps).
Model size/availability: [official V32 model card](https://huggingface.co/OliBomby/Mapperatorinator-v32), [checkpoint file listing](https://huggingface.co/OliBomby/Mapperatorinator-v32/tree/main).

The Mug parameter count uses four constructors from one YAML, with no weights, forward pass, training, sampler or accelerator.
`count_mug.py` omits unavailable audio-I/O imports and substitutes NumPy contractions for unavailable `opt_einsum` during construction; this is configuration arithmetic, not runtime compatibility evidence.
Its output is 102,015,888 U-Net + 49,392,192 audio + 5,086,192 autoencoder + 42,112 condition parameters.
All counted parameter storage is float32; the configured diffusion wrapper adds no learned log-variance (`U/mug/diffusion/diffusion.py:126`).

**Relevant attempts in time order.** All effects below lack repeated-training-seed estimates of variation unless expressly stated.

| Attempt and problems | Hypothesis/change | Scale, comparator and measurement | Result, contemporary interpretation, next step |
| --- | --- | --- | --- |
| R1 baseline, before lineage; A–C/G/H | Seed/history plus row consequence residuals could improve continuation on real timing. | 3.084M; 11,563 charts/3,169 groups eligible; 6.75M exposures; CPU1; actual steps/wall time not recorded here. Six release examples seed 17, larger development panels documented elsewhere. | `checked-artifact` release metadata says restoration complete, quality still requires long-form LN/TAP/local-response review. It becomes the frozen timing-transfer receiver. |
| 09-23 frozen BeatThis pilot; E/H | Contextual beat features improve event timing beyond local Mel. Commits `135e181`, `68a48aa`, `2ae8e3a`; `audio_skeleton`. | Both heads 770,188 parameters, plus frozen BeatThis only in one arm; 48 TRAIN songs/charts; 6 calibration + 6 assessment; 1,200 updates, batch 12, seed 172. Local 107.70 s, BeatThis 202.36 s on MPS. | `checked-artifact`: head F1@20 ms .67713→.73186; release-only .03541→.05130, with source densities. Interpretation supported transfer, not playability; native R1 integration followed. |
| 09-23 integration/materialization; A/B/H | Learned H and release opportunities can drive frozen R1. `2ae8e3a`→`068988e`. | 48 cases per arm across 6 songs, seeds 17/23, timing/control variants; not 48 independent songs. Training inherited above; integration wall time not extracted. | Documents claim mechanical passes, then Lens exposed fractional timestamps; native-ms materialization corrected them. Earlier F1 remains fine-offset evidence, not corrected end-to-end quality. |
| 09-23 joint canonical-Mel route; E/F/J | Learn audio, timing and complete-row probabilities together. `9538d7d`, `e2bca3e`, `09b919c`. | New canonical frontend and architecture/objective; initial pilot counts/steps/wall time not independently recovered for this entry. No matched broad-pretrained-encoder comparison. | `doc-claim`: BeatThis/larger encoders deferred, not disproved. Seed residual, landmark memory and consequence module initially omitted, so it did not preserve the released R1 policy. Later context/control modules were added. |
| 09-27 complete-audio/history memory; A/D/E/H | Audio pyramid and separate H/R/R1 history attention recover phrasing. `b130dfb`→`f22e936`. | Core 4.584M vs memory 7.617M; 384 updates, identical 768-example ledger from 417 charts/383 groups; ledger seed 274000. 1,227.90 s baseline vs 3,058.49 s memory including discarded attempts. 28 cases/arm, 84 exports; principal Stream comparison 3 songs × 3 generation seeds. | `checked-artifact`: mean pressure .04447 unfitted, .78553 continued baseline, .38230 memory. Memory beats continued baseline but regresses from initialization; rejected. Publication checks pass in tested cached/loaded setting. This does not reject attention generally. |
| 09-28 rhythm-coordinate proposal; C/E | Learn shared musical phase/unit and subdivision organization without a mandatory redline grid. `a6c912f`; `audio_rhythm_hierarchy_zh.md`. | Proposal, no training; document gives 21 fixed rhythm-lattice inspections, not a trained architecture comparison. Params/steps/wall/seeds not applicable. | `doc-claim`: rhythmic relation failure recognized; mechanism explicitly unimplemented. Thus metrical reasoning was considered but never tested as the proposed learned hierarchy. |
| 09-28 ordinary expert from scratch; A–E/H/K | Remove inherited proposal bias and fit ordinary arrangements jointly from BOS. `d6eba23`. | 4.783M; four songs/charts, sixteen 4 s training units; 512 updates, 235.73 s CPU four threads in the historical run; eval 12 outputs on the same four songs, seed 290031. Broader 1,602-chart TRAIN split was prepared, not consumed by this fit. | `checked-artifact`: completed fit, qualification false. `doc-claim`: source NLL improves, native output retains tiny LNs/duplicate H; rejected as capability pilot, not a completed ordinary-corpus experiment. |
| Reference generator baseline; A–E/H/K | Use an external generator to anchor musical/playability comparisons. | No matching run found; n, seeds, training/inference wall time and effect size not recorded. | Early documents explicitly retain osuT5/Mug as future comparison targets. No retrieved lineage result closes that comparison. |

Raw anchors: `artifacts/hf-pulsefield-r1-restored/release.json`; `artifacts/audio-skeleton/20260923-v1/training/{local,beat}-pilot-v1/{freeze,result}.json`; `artifacts/audio-skeleton/20260923-v1/integration-ms/beat-pilot-v1/summary.json`.
Memory anchors: `artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/preparation-final-v2/result.json`, `baseline-384/result.json`, `memory-resumed-384/result.json`, `comparison/result.json`.
Ordinary anchors: `artifacts/joint-audio/20260928-ordinary-scratch-v1/capacity-data-v1/result.json` and `capacity-fit-v1/result.json`.
Interpretation anchors: `099cb66:docs/research/audio_conditioned_choreography.md:37`, `audio_memory_joint_fit.md:13`, `ordinary_expert_from_scratch_zh.md:169`, `audio_rhythm_hierarchy_zh.md:3`.

**Q7 — Scale ratios and their limits.** Values are computed by `summarize_evidence.py`; inputs are the named release/configuration reports, one Mapper README size, and one public dataset card.

| Denominator | Mapper 219M / denominator | Mug configured 156.536M / denominator |
| --- | ---: | ---: |
| R1 3,084,432 | 71.00× | 50.75× |
| Core2500 4,583,985 | 47.78× | 34.15× |
| Memory 7,616,517 | 28.75× | 20.55× |
| Fresh ordinary 4,782,754 | 45.79× | 32.73× |

The public Mapper dataset totals are 18.43× R1's eligible chart count and 14.64× its song-group count.
Those are corpus-supply ratios, not same-mode training ratios: the card includes all modes/years, the V32 config filters them, and audio hashes differ from Ensomi's grouping definition.
Mug's data ratio cannot be determined because its linked dataset list was unreachable and no local training list was found.
No honest compute ratio can equate R1 CPU exposures, 512 CPU pilot updates, 384 MPS updates, and Mapper's project-wide GPU-hours.
In particular, 700,000 configured updates is not evidence that a released model actually completed that count, and 1,000 diffusion timesteps are not optimizer updates.
The defensible conclusion is an order-of-magnitude capacity difference and very short lineage interventions, not a measured scaling law.

**Q4 — Was a reference generator run on the same songs?** No such comparison was found; confidence medium, not proof of nonexistence.
The code/docs search found external-quality aspirations and legacy attribution strings, not a benchmark runner/result.
The artifact content search used `rg -l --no-ignore -i 'mapperatorinator|mug.?diffusion|osut5' artifacts` over JSON/MD/log/YAML/OSU files ≤4 MB, excluding this review directory.
Its only two matches were older beat-structure/beat-chunk experiment cards discussing token analogies; filename search returned no reference-named outputs.
Three sealed legacy timing locations were unreadable, and files over 4 MB, unnamed outputs and deleted runs remain outside this negative evidence.
See `/tmp/lineage-review/09-reference-projects/{repo-reference-mentions,artifact-reference-files,artifact-reference-names}.txt`.
`doc-claim`: `5c56e28:docs/research/oracle_time_m3_validation.md:210` explicitly says matched comparisons are unestablished.

| What a later same-song run needs | Mapperatorinator | Mug-Diffusion |
| --- | --- | --- |
| 4K support | Yes: mode 3/keycount 4; HF listing includes a mania-specific folder. | Yes: native supported mode; no other key count in inspected converter. |
| Checkpoint | No local weight file found in clone. Official V32 listing offers ~866 MB base weights; separate mode model may add another similar file; entire listed repository 3.48 GB. | No local weight/config pair found. README points to bundle containing `models/ckpt/model.ckpt` and `model.yaml`; link existence is not a successful availability/download check. |
| Size | 219M claim; runtime memory/latency on this Mac not measured. | Config counts imply ~626 MB raw FP32 parameter storage; release/bundle size not verified. Runtime can be much larger. |
| Dependencies | Existing environment has Torch/Torchaudio 2.11, Transformers 4.57.3, nnAudio, accelerate, datasets, torchcodec, peft and slider. Exact slider-fork/API compatibility not tested. | Gradio, audioread, opt-einsum and MinaCalc distribution absent; reamber and Lightning installed. Older Lightning imports also need compatibility checking. |
| Device path | Explicit CPU/MPS/auto selection exists (`M/inference.py:79`); use non-CUDA attention on this Mac. No backend run performed. | WebUI chooses CUDA or CPU, not MPS (`U/webui.py:101`); script loader unconditionally calls CUDA (`U/scripts/mapping.py:38`). CPU is the available documented fallback path; MPS would require adaptation. |
| Inputs/output | Same audio, pinned checkpoint/config, SR/LN/style/seed, output paths; no source timing unless declared as a distinct condition. | Same audio, pinned release config, control prompts/seed, explicit snapping and minijack settings; these settings change the comparison. |

No downloads, installs, reference forward passes, training or generation were performed.

## 3. Commentary

**Q2 — Features bearing on complaints A–E.** Categories describe the particular mechanism, not a proof that the entire complaint is solved.
“Addresses by construction” can mean a partial/default postprocessor; exceptions are stated.

| Complaint | Mapperatorinator | Mug-Diffusion |
| --- | --- | --- |
| A: long jacks/close same-lane attacks | **Addresses by data or scale:** rank/difficulty/mapper/tag conditioning learns lane patterns. No explicit strain veto or same-column recovery guarantee found in inspected generation path. | **Addresses by construction**, partially: default 90 ms minijack repair moves/removes close attacks. It exempts stream endings, avoids moving LNs, and is not a long-run load model. Etterna labels also address patterns by data. |
| B: fragmented LN | **Addresses by data or scale:** jointly learned start/end/sustain and LN-ratio conditioning; representation itself imposes no ordinary-duration preference. | **Addresses by data or scale:** joint holding-body tensor and LN/style controls encourage learned durations. Scanning a body makes an LN object, but short bodies and close end/start offsets remain possible. |
| C: irregular subdivision/no ordinary organization | **Addresses by construction**, for timing only: learned snapping divisor plus timing resnap. Ordinary TAP/LN organization still depends on data; off-grid events can survive when no snapping is specified. | **Addresses by construction**, for optional grid repair only; default UI enables it. Ranked/RC and skill conditioning address organization by data. A grid does not enforce phrase or LN/TAP roles. |
| D: no breathing/variation | **Addresses by data or scale:** overlapping audio/chart context and song-position/length metadata can learn variation. No explicit breathing evaluator or guarantee located. | **Addresses by data or scale:** whole-song denoising and multiscale audio expose long relationships. No explicit temporal breathing target or validated evaluator located. |
| E: weak audio/music use | **Addresses by construction:** audio encoder conditions autoregressive decoder at every window; timing/SR/style interaction remains learned. Presence of audio is not proof of appropriate use. | **Addresses by construction:** separate learned audio pyramid conditions U-Net at several scales across the song. Audio alignment still fails on some music according to author issue #6. |

`checked-code` anchors: M/configs/train/v32.yaml:19, M/osuT5/osuT5/inference/postprocessor.py:588; U/mug/data/convertor.py:232; U/mug/data/utils.py:110 and :142; U/webui.py:401 and :586.
There is no need to force a “does not address” label onto a whole complaint when relevant learned pathways exist.
Both systems **do not address** independently validated canonical player-response semantics by construction; neither gives a guarantee against sustained single-finger overload or poor musical phrasing.

The references weaken two easy diagnoses: neither requires all raw timing to live on a beat grid, and neither universally represents an LN as an indivisible duration object.
Conversely, the minijack and resnapping code makes it unsafe to attribute their displayed quality solely to learned generation.
A fair comparison must identify output repair; removing repair would measure a different system.

The BeatThis result is a reasonable transfer pilot and its positive head-F1 result is supported by raw reports.
The improvement is 0.05473 absolute F1 on six assessment songs, while release-only improvement is 0.01589 and default-density release F1 remains zero.
Source density, one training seed, prior-pretraining overlap not audited, and a changed frontend block a stronger conclusion about musical sufficiency.
The documents largely acknowledge these limits; the evidence does not support saying “BeatThis failed, therefore pretrained encoders are the wrong direction.”

The memory comparison usefully retains the unfitted parent, preventing a weaker continued-training control from manufacturing an improvement claim.
Its native rejection is supported. Its 384 updates and three main songs cannot decide whether richer audio context is unnecessary.
Adding several audio/history readers together also leaves the responsible component unidentified.
The official result is qualification not established; some semantic guards remain pending in the raw comparison report.

The ordinary pilot is too small to judge the ordinary-data direction: 64 seconds of selected training units across four songs is a capability probe.
Evaluating full songs mostly outside those units changes the task sharply; good source NLL and bad native charts isolate a gap, not a general impossibility.
The original document explicitly states this restriction, so criticism should target any later overgeneralization, not invent one in this result.

**Q3 — What replaces a player-response state?**
Mapperatorinator has causal token hidden states, difficulty/style metadata and a learned chart distribution; these are not declared player-load variables.
Its supervised CE and surprisal-based MaiMod can express typicality, but no independent continuation-response target or physical-load rollout was found (`checked-code`, M/osuT5/osuT5/model/modeling_mapperatorinator.py:130; `doc-claim`, M/README.md:190).
Mug uses explicit difficulty/skill conditioning and practical repair. Its S4 state-space layers are neural sequence machinery, not player-response state.

Mug's Etterna pipeline is a substantive precedent for load-related conditioning.
`checked-code` U/scripts/prepare_beatmap_features.py:106 calls MinaCalc on sorted `(start, column)` pairs and stores stream/jumpstream/handstream/stamina/jackspeed/chordjack/technical scores.
Those labels summarize a whole chart; LN durations are not passed in that particular call, and the generator does not expose the calculator's evolving state during sampling.
A visible code concern: boolean `stamina` is assigned from the `technical` score at line 159, although `stamina_ett` uses the actual stamina score. This is a reference limitation, not evidence of Ensomi behavior.

Official osu!mania difficulty **does** maintain temporal load-like state.
`checked-code`: per-column strain decays with base .125 per second, aggregate strain with .30; notes add difficulty contributions and chord handling takes the hardest individual strain.
Hold overlap/awkward release relationships contribute through dedicated evaluators; section peaks are aggregated, not a constant chart-density count.
Sources: [Strain.cs](https://raw.githubusercontent.com/ppy/osu/master/osu.Game.Rulesets.Mania/Difficulty/Skills/Strain.cs), [individual evaluator](https://raw.githubusercontent.com/ppy/osu/master/osu.Game.Rulesets.Mania/Difficulty/Evaluators/IndividualStrainEvaluator.cs), [overall evaluator](https://raw.githubusercontent.com/ppy/osu/master/osu.Game.Rulesets.Mania/Difficulty/Evaluators/OverallStrainEvaluator.cs).
The shared skill uses 400 ms sections and descending-peak weighting .9: [StrainSkill.cs](https://raw.githubusercontent.com/ppy/osu/master/osu.Game/Rulesets/Difficulty/Skills/StrainSkill.cs).
The [official rating description](https://osu.ppy.sh/wiki/en/Beatmap/Star_rating) describes the resulting difficulty estimate; it does not establish individual physiology or a full playability judgment.

Ensomi's distinction is its **intended counterfactual semantics**: the same committed history must support comparisons among legal future continuations, with horizons and mapper-defined distinctions, linked to style/demand controls.
That explicit contract is absent from the inspected generators. It is not new merely to maintain decaying strain, condition on difficulty, or use a latent recurrent state.
Nor is demonstrated scientific novelty established: `5c56e28:docs/formulation/gameplay-state.md:53` explicitly leaves target quantities/scales/comparison rules undefined (`doc-claim`).
The same document at :25 excludes individual capacity, physiological fatigue and subjective pain; describing the implemented system as a physiological player simulator would exceed its contract.
My call: distinct formulation relative to these two implementations, high confidence; validated useful novelty or novelty across the wider literature, not established.

**Q8 — References' own limitations.**
Mapper's author warns that missing year/difficulty can produce inconsistent style/difficulty, conflicting music/style requests may not be followed, and supplied timing improves speed/accuracy (`doc-claim`, M/README.md:175).
These bear on C/E and control reliability; they are not measured failure rates. Super timing still sometimes needs manual adjustment (`M/README.md:289`).
Mug's author opened [issue #6](https://github.com/Keytoyze/Mug-Diffusion/issues/6) for missing/unreasonable placement on piano sounds (`doc-claim`), directly matching C/E.
Its UI recommends grid snapping mainly when BPM does not change, and its default minijack repair is concrete evidence that close same-lane problems need handling (`checked-code`, U/webui.py:586).
Neither accessible project document supplied quantitative LN-fragmentation or breathing failure rates; absence of a report is not absence of the failure.

The references are also not strong evaluation authorities.
Mug's shared config gives train/validation the same file list; `OsuTrainDataset` returns the entire list while validation takes its final 10% (`checked-code`, U/mug/data/dataset.py:277).
Thus this config does not establish held-out validation. It does not prove the released checkpoint used that exact split.
Mapper's inspected evaluation is teacher-forced token loss/accuracy, not a public matched 4K playability trial; both projects' “high quality” descriptions remain author claims in this review.

## 4. Direction

The strongest case for the lineage's direction is real: exact replay, recoverable state, explicit publication semantics and freedom from a mandatory musical grid matter for an interactive generator.
Its own raw results sometimes rejected checkpoints despite improving proxy metrics; the memory and ordinary results are examples of scientific restraint.
The strongest case against is that these engineering requirements never established that a small event-first proposal, trained in short pilots and repeatedly corrected downstream, adequately covered ordinary chart organization.
My call: the outside view supports that concern, medium confidence; it does not isolate which architecture component caused the failure.

For small jointly trained encoders, the favorable argument is cheap, interpretable iteration on task-relevant supervision without assuming speech/music pretraining transfers.
The unfavorable argument is 20–71× less capacity than the inspected reference configurations and weak exposure to music/arrangements in decisive pilots.
Call: sensible prototype strategy, weak basis for declaring an audio-representation or organization limit; high confidence in this evidential limit, low confidence about the best encoder choice.

For avoiding hard grids/hand rules, the favorable argument is preserving legitimate tech, tuplets and LN edge cases rather than making them illegal.
The unfavorable argument is that the references operationalize ordinary timing/close-jack repair, while Ensomi often treated global expressiveness as the immediate default learning problem.
Call: preserving expressive support is justified; the inference that ordinary rhythmic bias or bounded repair is thereby unsuitable was not tested. Confidence medium.

**Q6 — Real-time and causality.** Mug's current WebUI denoises and decodes an entire song before postprocessing/export; it has no tested prefix-publication interface.
Its reported faster-than-song-duration inference is not first-window latency or a deadline guarantee. Whole-song denoising can revise any chart position until sampling ends.
Mapper is a closer analogue: `M/osuT5/osuT5/inference/processor.py:305` generates sequentially, reusing prior output in overlapping audio windows.
But its supplied application loads full audio, may generate a complete timing pass first, resnaps after generation and exports the completed result (`M/inference.py:449`).
No measured publication-ahead-of-playback or immutable-row commit service was found for either reference.

The strongest fair argument for Ensomi is therefore preserving **settled chart coverage under a compute deadline**, including open holds and live scoped requests.
The argument is weaker if stated as “the references look at future audio”: Ensomi explicitly allows complete audio (`5c56e28:docs/formulation/notation.md:10`).
It also allows joint reasoning over provisional future rows and prefix commit (:227–268), so local autoregressive event sampling is not dictated by the formulation.
Neither complete-audio context, chunk proposals nor provisional future chart reasoning conflicts inherently with that contract; production latency and immutability would need separate evidence.

**Q9 — Unusual assumptions relative to the two references.**

- Treating audio-to-H transfer into a frozen row policy as a short bridge to chart quality: unlike references that learn timing and lane/type organization within one chart model. Evidence: pilot/integration raw reports; M parser/model and U joint note latent (`checked-code`, `checked-artifact`).
- Strong event-level probability/legality machinery before a broad ordinary conditional proposal was demonstrated. Evidence: R1 source-timing contract and four-song fresh pilot, compared with reference conditional objectives (`checked-code`, `checked-artifact`).
- Relying on a supplied seed and no difficulty/style request at R1 baseline. Both references generate directly from audio plus metadata (`checked-code`, generation paths above).
- Reading local fixes as possible architectural lessons after hundreds of updates. Larger reference configurations do not prove scale is sufficient, but leave under-training as an unexcluded alternative (`checked-artifact` budgets; counted/configured capacities).
- No located same-song external baseline despite explicit comparison aspirations. This deprives system-level criticism of a concrete quality anchor (`doc-claim` plus bounded negative search).

**Q9 — Differences justified by the real-time vision.**

- Exact occupancy/replay and immutable committed rows, including no-row intervals and open holds: needed for consistent publication (`doc-claim` contract; `checked-code` R1 generation).
- Bounded service accounting, resumable state and scrutiny of first playable coverage: neither reference's file-oriented interface supplies these measurements (`checked-artifact` memory comparison; `checked-code` reference applications).
- Scoped mid-song condition changes without rewriting earlier output: legitimate additional requirement beyond song-wide metadata prompts (`doc-claim` formulation; `checked-artifact` memory switch probe).
- Starting with a small model for local-device deadlines: a reasonable resource choice, though neither three million parameters nor strict one-event sampling follows logically from the requirement (inference from checked mechanisms).

## 5. What was overlooked or never questioned

**Q5 — Rejected, deferred, considered but untested, and not found are different statuses.**

| Ingredient | Recorded status and reason | Assessment |
| --- | --- | --- |
| Pre-V3 mapper/timing/control/runtime | Baseline README:43 forbids using it as a V3 design/correctness/implementation reference; retains legacy checkpoint ownership. | Explicit scope rule, not a matched experiment against every ingredient used by the old stack. |
| Mandatory finite beat grid | `099cb66:docs/research/audio_conditioned_choreography.md:207` rejects finite support without a residual/general-time path; expert question :34 rejects redlines as timing truth. Reason: irregular/expressive timing coverage. | Does not reject beat coordinates, uncertainty, residual timing or learned subdivision. |
| Beat-aligned representations generally | Choreography document :282–295 discusses a beat-aligned generator and soft coordinates with residuals; late rhythm hierarchy remains unimplemented. | Considered, not absent; no matched learned-coordinate result found. |
| Larger pretrained audio encoder | Choreography document :57 and :266 explicitly defer BeatThis/MERT/MusicFM while establishing joint task learning. | Deferred for scope/supervision, not empirically ruled out; positive BeatThis timing transfer survived. |
| Whole-song generation/future reasoning | No blanket empirical rejection found. Complete audio and provisional future rows are explicitly permitted in the formulation. | Prefix deadlines make an offline application insufficient, but do not settle the proposal architecture. |
| Reference-specific repair and metadata recipes | No mention found of Mug's minijack algorithm, its default snapping, Etterna label extraction, or a matched Mapper overlap/metadata recipe in the requested lineage mention set. | “Not found in searched sources,” not proof nobody considered them. |

Ranked omissions, with concrete unanswered questions:

1. **An external quality anchor (A–E/H/K).** No located same-song reference outputs means there is no observed gap to explain: perhaps references fail similarly, perhaps simple repair dominates. Evidence: Q4 bounded search and explicit unfulfilled comparison claims.
2. **A defined independent response target (G/H).** The intended novel object remains semantically unspecified; no located comparison shows it distinguishes unacceptable continuation from legitimate style beyond rating/repair heuristics. Evidence: formulation :53 and reference strain/Etterna mechanisms.
3. **Capacity and training exposure as rival explanations (C/E/J).** Four songs/512 updates and three main assessment songs after 384 updates cannot settle ordinary-distribution or full-audio architecture questions. Evidence: raw preparation/fit/comparison reports and configuration counts.
4. **What generation alone contributes versus output repair (A/C).** Reference demos include mechanisms absent from a pure model comparison. No matching repaired/unrepaired comparison in this project was found; Mug's default 90 ms setting makes the omission consequential.
5. **Ordinary support versus expressive support (B/C).** The records carefully defend irregular timings, but the larger ranked ordinary split was not the data used by the last capability fit. Evidence: ordinary expert document :60 versus raw `capacity-data-v1/result.json`.
6. **Where streaming actually constrains design (D/E/J).** Full audio and provisional multirow inference are allowed; no located matched comparison establishes that the millisecond next-event factorization uniquely meets publication requirements. Evidence: formulation :227 and Mapper's windowed autoregression.

These are bounded missing measurements or distinctions, not a proposed architecture or roadmap.

## 6. Worth keeping

- **The positive, scoped BeatThis transfer finding.** Raw scores establish a head-timing improvement on the six-song development assessment; release-only failure prevents overselling it. Keep the original frontend identity and source-density caveat (`checked-artifact`).
- **Three-arm memory comparison.** Retaining unfitted, continued-baseline and memory outputs exposed regression that a two-arm summary could hide. The checked raw comparison agrees with the document's rejection (`checked-artifact`).
- **Separation of mechanical validity, source likelihood and native quality.** Release metadata and ordinary qualification explicitly keep these apart; this survives any restart (`checked-artifact`).
- **Complete-audio plus committed-prefix contract.** It admits substantially more proposal choices than one-event causal sampling, while preserving publication semantics (`doc-claim` formulation, not validation of an implementation).
- **Reference mechanism inventory.** Metadata dropout, multiscale audio, overlapping generation, explicit hold bodies, resnapping and jack repair are concrete comparison ingredients, not authorities about what Ensomi must implement (`checked-code`).
- **The novelty distinction.** Counterfactual continuation semantics is a clearer differentiator than the mere existence of strain or a neural state. Actual utility remains to be shown.

## 7. Claims worth re-verifying

1. **Released Mug architecture and training population.** Obtain the exact release's `models/ckpt/model.yaml` and checkpoint metadata from the README-linked bundle; compare with `U/configs/mug/mug_diffusion.yaml`. Current 156.536M is a configuration count, not identification of the shipped model; chart/song/steps/GPU-hours and independent validation remain missing.
2. **Mapper checkpoint count and actual training.** Pin a revision of `OliBomby/Mapperatorinator-v32`, including `gamemode=3`, then count state-dict tensors and recover its training manifest/log. README 219M, rounded HF size and ~866 MB listing do not give encoder-only size or a complete training ledger.
3. **Reference same-song quality.** After separately obtaining weights, the concrete Mapper invocation is `python inference.py -cn v32 gamemode=3 keycount=4 difficulty=4 year=2024 seed=17 device=cpu model_path=LOCAL_PINNED_CHECKPOINT audio_path=SAME_SONG output_path=REFERENCE_OUTPUT` from its clone. Replace named input paths with a frozen song manifest; no reference run was authorized/performed here.
4. **Mug same-song quality.** With the pinned checkpoint/config and compatible dependencies, `python webui.py` is the inspected CPU-capable entry; record seed, SR/LN/skill conditions, 100 DDIM steps, CFG, and both `auto_snap`/minijack interval. The separate mapping script is not a safe Mac CPU command unchanged because its loader calls CUDA.
5. **Reproduce this review's size arithmetic.** `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/lineage-review/09-reference-projects/mpl XDG_CACHE_HOME=/tmp/lineage-review/09-reference-projects/cache OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python /tmp/lineage-review/09-reference-projects/count_mug.py`; n=4 constructors, input one YAML plus referenced feature YAML/source. Then run `.venv/bin/python /tmp/lineage-review/09-reference-projects/summarize_evidence.py` for ratios/raw report extraction.
6. **Version-matched strain behavior.** Compare current upstream `Strain.cs` and its evaluators with Ensomi's `20241007` implementation before transferring numeric stars. This review establishes the existence of temporal strain, not equality of those implementations or physiological validity.

No experiment requiring a model forward pass or >5-minute CPU command was needed for the report.

## 8. Cross-slice notes

The response/evaluation slices should distinguish semantic novelty from strain aggregation and from Etterna-derived global labels.
The timing slice should retain the difference between compulsory grid support and soft beat coordinates; the lineage explicitly discusses the latter.
The ordinary-data slice should separate the prepared 1,602-chart split from the four-song capacity run.
The release slice should not assume either reference eliminates fragmented LN by a duration token: their actual encodings differ and still learn duration organization.
These are pointers only; this review did not reassess their full experiment inventories.

## 9. Failed paths and unfinished work

All nine numbered slice questions are answered; unknowns are retained rather than filled by estimates.
The public Mug dataset page and two GitHub comment-API pages were inaccessible; no corpus-size or undocumented training claims were inferred from them.
The first artifact search accidentally respected ignore rules; it was rerun with `--no-ignore`, and only the corrected results support Q4.
Three sealed legacy timing locations remained unreadable; no permission bypass was attempted.
Mug constructor imports initially failed on missing audioread and opt-einsum; the documented constructor-only adaptation recovered counts, not runnable generation.
One intermediate memory report described an early 10-update resource stop; the completed resumed owner and common comparison were then checked, avoiding misidentification as the final fit.
No published checkpoint was downloaded, no audio was played, no generated chart was manually judged, and no direct reference-quality ranking was established.
Exact reference training counts, reference run-to-run variation, R1 cumulative training wall time, and released Mug model identity remain unfinished evidence items.
No tracked repository source, configuration, Git state, notes branch or another reviewer's report was changed.

## 10. Inconsistencies and items for the human

1. **“Player response state is novel” needs a narrow meaning.** `checked-code`: official mania has per-column/overall strain with decay, and Mug conditions on Etterna skills. `doc-claim`: Ensomi proposes a stronger continuation-response contract but leaves its target undefined; novelty should attach to that intended semantics and demonstrated use, not temporal state alone.

2. **Physiology language exceeds the frozen canonical profile.** `doc-claim`: `5c56e28:docs/formulation/gameplay-state.md:25` excludes individual capacity and physiological fatigue, while feedback problem G asks for physically grounded recovery/load. The human must decide whether the intended key stays mapper-canonical or changes scope; this review does not resolve that choice.

3. **Real-time does not imply causal audio or one-event proposals.** `doc-claim`: notation :227 permits complete audio and joint provisional future rows; `checked-code`: Mapper already generates overlapping windows sequentially. The unresolved requirement is settled publication under deadlines, not a general prohibition on future context.

4. **Large pretrained encoders were deferred, not refuted.** `checked-artifact`: BeatThis raises head F1 on the six-song assessment; `doc-claim`: MERT/MusicFM and soft beat coordinates are explicitly discussed. Also, `checked-code` Mapper constructs its backbone from configuration with an empty default pretrained-weight path, so its Whisper ancestry is not proof of pretrained speech-weight transfer.

5. **Reference quality includes hand-authored repair.** `checked-code`: Mug defaults to 90 ms minijack rearrangement and optional enabled grid snapping; Mapper resnaps predicted timing. If the human means purely learned ordinary generation, comparison criteria must say how these full-system outputs count.

6. **Neither reference supplies a ready-made trustworthy quality evaluator.** `checked-code`: Mug's inspected train list contains its validation tail, and its boolean stamina label uses technical score; Mapper evaluates token predictions. These limits prevent treating their claimed success or their scalar controls as independent playability truth.

7. **The last ordinary experiment was not the ordinary-corpus trial described by its prepared split.** `checked-artifact`: capacity training used 4 charts/16 units/512 updates and qualification is false. `doc-claim`: the owning document acknowledges this, so a rejection of ordinary-first learning would be a stronger conclusion than its evidence.

8. **No located baseline leaves the main outside-view question empirical.** Bounded search found no same-song reference outputs (`checked-artifact` search records); prior documents say the comparison was unestablished (`doc-claim`). This report cannot tell the human that either reference solves A–E on Ensomi's songs—only which mechanisms and resources a comparison would actually test.
