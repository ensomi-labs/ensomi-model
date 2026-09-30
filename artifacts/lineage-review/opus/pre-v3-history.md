# Pre-V3 history: what was set aside and why (independent review, Opus)

Slice `02-pre-v3-history`. Written 2026-09-30 by an Opus 5.5 reviewer on the control plane, without reading the Astra, Opus or Fable lineage-review reports or the synthesis. Evidence grades: **doc-claim** (a document, report or commit message says so), **checked-code** (read in source at the named ref), **checked-artifact** (read or measured in a raw artifact). Nothing was trained, inferred or installed; the only computation was parsing JSON and `.osu` text.

## 1. What I read and checked

- Notes entry point, lineage-review README and the feedback index (sections 2, 3, 5), as instructed.
- `git log main` (133 commits from 2026-05-17), all refs under `refs/archive/`, the unmerged branch `cleanup/main-r1` (09-30), and READMEs at `02adb53` (06-01), `65d8f7f` (07-10), `8e5e7ad` (08-31) and `dfb4618` (09-01, where the ban appears).
- Tracked-then-deleted artifacts recovered from history: `4ee0212^:artifacts/{evals,reports,runs}/...` (mapper v2.1 run report, decoder postmortem, PR2 Riria sweep with a generated `.osu`) and `a35dbc6:artifacts/evals/inference_bundle_mps_profile/` (bundle profile with a generated `.osu`).
- Code in the `main` export: `timing/`, `models/mapper/`, `data/control_windows.py`, `inference/session_runtime.py`, `evals/mir_anchor_*`, `features/mir_backbone.py`, `osu_core/beat_representation.py`; imports of the V3 and lineage research packages.
- Archive branches: `research/beatmap` (139 commits on 06-16/17), `research/token-LN-BPE`, `research/beatmap-structure`, `exp/mel-reconstruct` (timing v3 docs), `diffusion-test` (gold diffusion).
- On bings-mac, read-only: `artifacts/runs/*`, `artifacts/gold_diffusion/train-20260907T151812Z`, `artifacts/evals/mir_anchor_probe_750_150_150_3seed`, `artifacts/reports/timing/*`, `artifacts/reports/audits/*`, `artifacts/local/timing_v3`, `artifacts/cleanup-report-20260930.json`, two dataset song folders, the BeatThis weights location.
- Measurements I made: (a) the two surviving pre-V3 audio-to-chart outputs against the reference charts of the same songs (`scratchpad/opus-pre-v3-history/measure.py`); (b) the per-song distribution of the timing stack's error over the full corpus (`timing_stats.py`, run on the mac over `timing_v3_v2_baseline_full5050_v1.jsonl`).

## 2. Paradigms, in time order

Sizes are from run reports and configs (checked-artifact) unless marked.

| # | Paradigm | Dates | Chart and time representation | Audio input | Size and training actually done | How far it got | Where now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Timing stack (BeatThis + GridFitter + dense timing features) | before 05-17 to 06-11 (migrated in; ramp detection #4) | Piecewise-constant BPM segments (offset, beat length, meter), rendered as 4 dense channels (beat pulse, local BPM, phase sin/cos) | BeatThis `final0` frame probabilities, 50 Hz | Pretrained BeatThis; the fitter is hand-written search | Ran end to end in inference; evaluated on all 5,026 corpus songs (section 5) | `main:src/ensomi_model/timing`; mac `.venv` has `beat_this` 1.1.0, weights in `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` |
| P2 | Control encoder and Control V3 targets | before 05-17 to 08-31 | Hand-authored window statistics of the chart (Control V3), predicted from audio | Mel (16 kHz, 80 bins) + dense timing | Control 12k run, step 8,000 (100 MB ckpt); control-demo global d384 l3, step 2,000 (180 MB ckpt), used to initialise the mapper; parameter counts not recorded | Its encoder fed the mapper; its targets were judged unsuitable (section 4) | `main:src/ensomi_model/{features/control_v3*,models/control}`; mac `artifacts/runs/stage2_control*`, `artifacts/features/control_v3_timeseries_*.parquet` |
| P3 | Mapper v2 (complete 4-lane tuple tokens) | before 05-17 | Tuple EVENT tokens plus millisecond time-shift tokens | Via control memory | d768 l8, batch 1, `max_steps` 12,000 (config); step reached unknown | Superseded by v2.1 before this repository began | mac `artifacts/runs/stage2_mapper_v2/..._d768_l8_b1/checkpoint.pt` (1.34 GB) |
| P4 | Mapper v2.1 (sparse lane-action tokens, osuT5 shape) | before 05-17 to 07-15 | `TAP/HOLD_START/HOLD_END` per lane plus time-shift tokens `TS_10..TS_100` (10 ms steps), `TS_200..TS_1000`, `TS_2000..4000`; every time on a 10 ms clock (checked-code `models/mapper/v2_1/vocab.py`, `shared/vocab.py:26`) | Mel + dense timing + control memory | 29,971,825 parameters; 44,000 of 120,000 planned steps at batch 2 = 88,000 of 172,649 train windows (0.51 passes) from 9,142 charts / 3,625 audio, 2-6 star; eval token loss still falling (0.963 at 42,500, 0.957 at 43,750); seed 1337, one run | **End to end from audio** (section 3) | `main:src/ensomi_model/models/mapper/v2_1`; mac checkpoints step 44,000 and 44,250 (240 MB each) |
| P5 | Real-time WebSocket service | 06-13 to 07-15 | Protocol tokens over `pulsefield-protocol` | Full pipeline | n/a | RTF 0.38, first token 2.2 s, 8 s windows in at most 3.5 s on one 47 s song, M5 (doc-claim, `a35dbc6` result log) | `main:src/ensomi_model/inference` |
| P6 | Tokenisation research (beat representation, duration-LN, C3 context-adaptive fallback codec) | 06-03 to 06-17 | Beat-relative converter (canonical 80-160 BPM, 1/48 snap); C3 compression codec | none | No model; compression audits | C3 lowers charged bits/event by 0.38 on test (4.04 vs 4.42), zero reconstruction mismatches (doc-claim, route synthesis on `research/beatmap`) | converter on `main:osu_core/beat_representation.py`; audits on mac `artifacts/reports/audits/` |
| P7 | Event-group mapper "v3" | 06-16 to 06-17 (one ~22 h run of ~120 agent commits) | One token per complete 4-lane event group, still millisecond time shifts (checked-code `research/beatmap:models/mapper/v3/vocab.py`) | same as P4 | Gates of 200 to 500 updates; the last "full4k" run was 8 updates (doc-claim, `target_grammar_v3_delta_event_factor_target_full4k_training_result_report.md`) | Never trained beyond smoke scale; a 32-pair rollout gate got worse with an auxiliary loss | `refs/archive/heads/research/beatmap`; 17 tiny checkpoints on mac |
| P8 | Timing v3 (tempo-change detection) | 08-11 to 08-20 | Phase-continuous absolute-beat schema with constant and jump sections | BeatThis frontend, ACF | Small boundary models (65 KB to 300 KB) | 29 experiments; phase-continuous schema kept; real tempo changes not solved; paused awaiting a human-labelled reference | `refs/archive/heads/exp/mel-reconstruct` (docs); mac `artifacts/local/timing_v3`, `artifacts/reports/timing` |
| P9 | Mel frontend metamer test | 08-18 to 08-19 | n/a | 16 kHz/80/25 ms vs 24 kHz/128/40 ms | 6 waveform optimisations, 1 excerpt | Owner listening rejected the old frontend; the new one became `MUSIC_MEL_CACHE_CONFIG` | `main:docs/research/mel_frontend_metamer_result.md` |
| P10 | MIR anchor probe (audio-information probe) | 08-22 | Choose the true next chart row time among 16 same-gap controls | Mel + novelty (N), PLP (P), tempogram (T) | 270,549 parameters; 750/150/150 audio split; 3 seeds | Result exists, never written up (section 6) | `main:src/ensomi_model/evals/mir_anchor_*`; mac `artifacts/evals/mir_anchor_probe_750_150_150_3seed` |
| P11 | Gold fixed-placement diffusion | 09-07 to 09-10 | Masked discrete diffusion over complete rows at supplied source times, LN close times supplied | 24 kHz Mel patches ±200 ms | 1,165,443 parameters; 113 train scopes (138 gold in all) from beatmap-lens human labels; early stop at update 1,100, best 300; one seed | Test row accuracy 0.127, lane accuracy 0.49 (14 scopes); 72 samples never reviewed | `refs/archive/heads/diffusion-test`; mac `artifacts/gold_diffusion` |
| P12 | Scoped style probes, source-action, oracle-time | 09-13 to 09-18 | Style classifier; source-action block prediction; continuation at supplied source times | none (oracle-time explicitly drops audio) | See slice 01 for oracle-time and R1 | Became R1's parents; R1 imports parsing and storage helpers from them | `main:src/ensomi_model/research/*` |

P3, P4 and P1 predate this repository: the first commits "Migrate osu core modules" and "Migrate timing modules" (05-17) bring them in, and the v2 checkpoint is dated 05-11. Why v2 became v2.1 is **not recorded here**; the earlier history is outside the repository.

## 3. Did anything produce a chart from audio end to end, and how good was it

Yes: P1+P2+P4, the pre-V3 system, went audio → mel → BeatThis → GridFitter → control → mapper v2.1 → `.osu` (doc-claim `8e5e7ad:README.md` "System path"; checked-code `inference/session_runtime.py:218,242,711`). Two outputs survive, both only in git history. I measured them against the ranked reference charts of the same songs (checked-artifact; heads within 3 ms of a grid line of the reference's red timing point; "any" means any of 1/1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16).

| Chart | Objects | Notes/s | LN share | Head gaps (top) | On reference grid, any ≤1/16 | Same-lane gap min | Longest same-lane run of consecutive rows |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Riria reference, 2.36 star | 489 | 5.66 | 45.8% | 200 ms ×267, 400 ×63 | 100% | 100 ms | 4 |
| Riria v2.1 step 44,000, "best" policy | 696 | 8.00 | 3.7% | 160 ms ×245, 150 ×207, 170 ×46 | 11.2% (1/4: 0.6%) | 150 ms | 2 |
| Oyasumi reference Insane, 3.32 star | 495 | 11.38 | 3.4% | 150 ms ×224, 75 ×21 | 100% | 75 ms | 4 |
| Oyasumi v2.1 bundle, greedy, requested 4.0 star | 1,615 | 34.29 | 0% | 100 ms ×471 (all) | 33.3% (1/4: 0%) | 100 ms | 472 (lane 4 in every row) |

- Riria (held-out song, audio and map disjoint from training; `4ee0212^:.../mapper_v21_44000_decoder_postmortem_2026-05-20.md`): with greedy decoding the same checkpoint emitted six lane actions in 87 s. The listed chart came from a sweep of about 30 decode policies (flat and length-scaled time-shift penalties, temperature, top-p, 2 seeds) on this one song, reranked by closeness of its lane-action count to the reference (`..._decode_policy_report.md`). Selection used the evaluation song's own reference. The agent's own drift audit already said the chart "does not fit the song timing well" (doc-claim, confirmed by my measurement).
- The supplied grid was nearly right: BeatThis/GridFitter gave 800 ms at offset 1020 ms against the chart's 800 ms at 1000 ms. The mapper did not follow even its own supplied grid: 2.0% of heads on its own 1/4 lines (checked-artifact). Its gaps of 150-170 ms (about 1/5 beat) are compositions of 10 ms time-shift tokens that match no subdivision.
- Oyasumi (`a35dbc6`, a throughput benchmark): the author's log itself records "all 471 adjacent intervals are exactly 100 ms; all 1,615 hitobjects are TAPs; lane 4 active at every timepoint" and that "the throughput guards did not detect" it (doc-claim, confirmed). The fitted grid also switched from 199.9 to 100.1 BPM at 29.3 s, an alias change the reference (a single 200 BPM section) does not have.
- n = 2 songs, one checkpoint, no human judgment on record. The README of 07-10 summarises the stack as "sparse output, repetitive patterns, and rhythm placement that can be structurally valid while remaining musically off-grid" (doc-claim, `65d8f7f`). The two surviving outputs agree with that summary and are far below playable.
- Train/inference mismatch (checked-code, not traced through every override): training windows render timing features from the chart's own red lines (`data/control_windows.py:124,1011`, `timing/providers/oracle.py`); inference uses BeatThis+GridFitter. Section 5 shows these differ by about 29 ms on the median song.
- Fairness: the mapper had seen half an epoch at batch 2 and its loss was still falling. The outputs judge that checkpoint and its decoding, not the token-mapper paradigm at convergence.

No other pre-R1 line generated from audio. Gold diffusion and oracle-time fill supplied source times; the event-group v3 ran only short rollouts on eval slices.

## 4. Why each line was left, and where that is recorded

| Line | Reason as recorded | Kind | Where | Grade and my reading |
| --- | --- | --- | --- | --- |
| Mapper v2.1 system | "The mapper remains the main quality bottleneck" (07-10); then superseded by the V3 formulation and the ban (09-01) | Result (2 songs) plus a change of problem definition | `65d8f7f:README.md`; `dfb4618` | doc-claim, confirmed on the 2 outputs. No scale test was run: 0.51 passes, one run. Leaving the checkpoint was reasonable. That the paradigm itself was refuted is not shown. |
| Control V3 | "handcrafted, future-dependent window statistics describe completed-chart outcomes rather than a persistent generative state, collapse a multimodal chart problem into pointwise targets..."; "failure hypotheses rather than isolated root causes; the measurements remain diagnostic only" | Theoretical argument, openly unmeasured | `8e5e7ad:README.md`; `docs/formulation/gameplay-state.md` near line 1016 | doc-claim. I found no quantitative control result in the repository. A preference backed by argument, stated honestly. |
| Timing stack (P1) | No reason to leave it is recorded. The timing README still claims it is "good enough and fast enough to provide a real-time timing prior" | None: it fell under the ban | `main:src/ensomi_model/timing/README.md` | doc-claim for the claim; section 5 checks it. This is **undocumented**. |
| Event-group v3 / C3 (P6, P7) | The last report recommends a "longer full4k or production-config card"; the branch stops at 06-17 20:53 and `main` resumes 06-21 with refactors | No recorded reason | `research/beatmap` head `f411d25` | Cannot determine. The June route synthesis records that a lower loss made rollouts worse. |
| Timing v3 (P8) | Synthetic tempo-warp proxy killed as final acceptance (08-17); real tempo changes unsolved; paused for a human-labelled reference; commit "add failed timing v3 experiment" | Result plus a missing evaluation reference | `exp/mel-reconstruct:docs/research/timing_v3_research_timeline_2026-08-16.md`, `..._decision_2026-08-17_...md`, `..._phase_1_completion_audit.md` | doc-claim; artifacts exist. Well documented. It targeted multi-BPM songs (about 16% of audio), not the constant-tempo phase offset. |
| Old Mel frontend | Owner listening: the 16 kHz/80/25 ms frontend lost high-frequency attacks | Human qualitative judgment, 1 excerpt, 3 seeds | `main:docs/research/mel_frontend_metamer_result.md` | doc-claim. A reasonable, openly bounded choice. |
| MIR anchor probe (P10) | None; never documented | None | Only code and artifacts | Cannot determine |
| Gold diffusion (P11) | None; the commit is "stash" | Probably the data limit (138 labelled scopes); not recorded | `diffusion-test:761e1e5` | Cannot determine |
| Audio input as such (09-15 onward) | "At this stage no audio is input" | A staging choice | `main:docs/research/oracle_time_expert_question.md` | doc-claim; slice 01 covers R1 |

Where recorded, the reasons are about what was measured (mapper, timing v3) or are openly argued preferences (Control V3, the Mel frontend). For the timing stack, the MIR probe, C3, event-group v3 and diffusion, no reason survives. Several of these runs were produced by autonomous agent loops (the 06-16/17 burst of "gate/route" commits; the Codex annotation markers in the timing v3 timeline), so part of the pre-V3 record, like the September lineage, was written by agents.

## 5. The timing stack and its accuracy

What it does (checked-code): BeatThis `final0` gives beat and downbeat probabilities at 50 Hz. `GridFitter` searches tempo candidates, resolves half and double tempo, and splits or merges sections into a compact list of constant-BPM sections. `render_dense_timing_v2` turns that list into four frame channels. It predicts a beat grid, not onsets or note times.

Recorded claims (doc-claim, timing README): mean phase error 33-50 ms on 20- and 100-map slices; 81% recall of 58 web-audited real multi-BPM songs, with a 29.9% false-positive rate on 97 mapper artifacts; about 2.1-2.4 s per song on MPS.

What I measured (checked-artifact; `artifacts/reports/timing/timing_v3_v2_baseline_full5050_v1.jsonl`, 5,026 songs, first chart per song as comparator):

| Measure | All songs | Chart has one timing section (n = 4,243) | Several sections (n = 783) |
| --- | ---: | ---: | ---: |
| Mean phase error, median / p90 | 43 / 74 ms | 40 / 72 ms | 54 / 82 ms |
| Tempo within 0.1 BPM (alias-aware) | 62.6% | 73.2% | 5.2% |
| Drift over the song ≤ 20 ms | 55.4% | 64.7% | 4.9% |
| Signed offset at song start, p10 / median / p90 | -52 / **-29** / +8 ms | -49 / -29 / 0 ms | -83 / -27 / +111 ms |
| Start within 10 ms of chart grid, as is | 6.5% | 6.5% | 6.6% |
| Start within 10 ms after subtracting the global median (29 ms) | 53.0% | 56.3% | n/a |
| Within 10 ms at start and ≤ 20 ms drift, after that one correction | 38.8% | 45.4% | 0% |

- The headline "about 42 ms phase error" is mostly **one constant offset**: 2,665 of 5,026 songs fall between -40 and -20 ms. The value is predicted phase minus chart phase (checked-code, `exp/mel-reconstruct:src/pulsefield_model/timing/evaluation/drift.py` lines 66-70), so a negative value means **the fitted grid lags the chart**, by a median 29 ms. Riria (+20 ms later) agrees. The median is the same for OGG (-28.5 ms, n = 623) and MP3 (-29.0 ms, n = 3,620) sources among single-section charts, so MP3 decoder delay does not explain it. It is either a BeatThis/fitter convention or a mapper convention. I found no document that notices this offset. Timing v3 checked for fixed-time compensation only for tempo-change seams on synthetic warps.
- Tempo is usually right on songs with one section. Songs whose charts have several sections are mostly wrong. They are 16% of audio and include the dense and jump strata.
- `.osu` red lines are an imperfect reference. Timing v3's 08-17 decision says so explicitly and proposes note-onset grid evidence instead (doc-claim).
- Usable today? The components exist on the mac: timing code in the working tree, BeatThis weights, `beat_this` in `.venv`. Fitted sections for all 5,050 corpus songs survive in the JSONL above (`fit.predicted_segments`). The BeatThis frame cache was deleted on 09-30. As an inference-time grid for a new song it is plausible for single-tempo songs once the constant offset is understood. As a training-time grid it is unnecessary, because chart red lines exist for every training song. As a baseline for the `time` node it offers a ready evaluator (`timing/diagnostics/compare_to_oracle.py` and the full5050 harness). Needed first: explain or calibrate the -29 ms, which is a one-hour analysis on existing outputs; measure against note onsets rather than red lines; decide on multi-section songs.

## 6. Assets that V3 and the September lineage did not use (existence checked 2026-09-30)

V3 and the lineage import only `osu_core.difficulty`, `features.mel_base` and `features.audio` outside their research packages (checked-code, relative imports in both trees). Everything below is unused by them.

| Asset | Exists | Location | Note |
| --- | --- | --- | --- |
| Mapper v2.1 checkpoints, step 44,000 and 44,250 | yes | mac `artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoints/` | Its training caches (`artifacts/cache/...`) were deleted on 09-30 |
| Mapper v2 d768 l8 checkpoint | yes (1.34 GB) | mac `artifacts/runs/stage2_mapper_v2/..._d768_l8_b1/checkpoint.pt` | Step unknown |
| Control encoders | yes | mac `artifacts/runs/stage2_control/.../checkpoint_step_008000.pt`, `stage2_control_demo/.../checkpoint_step_002000.pt` | |
| Event-group v3 and C3 checkpoints (≤ 1,000 updates) | yes, 17 dirs | mac `artifacts/runs/stage2_mapper_v2_1/c3_*` | Low value |
| BeatThis weights and package | yes | mac `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt`; `.venv/.../beat_this` 1.1.0 | `cleanup/main-r1` drops the dependency |
| Corpus-wide fitted grids, 5,050 songs | yes (147 MB) | mac `artifacts/reports/timing/timing_v3_v2_baseline_full5050_v1.jsonl` | Frame cache deleted |
| Timing v3 labels, inventory, small boundary models | yes; 3 sealed dirs unreadable | mac `artifacts/reports/timing/timing_v3_*`, `artifacts/local/timing_v3/` | |
| Beat-relative chart converter and its audit | yes | `main:src/ensomi_model/osu_core/beat_representation.py`; mac `artifacts/reports/audits/beat_representation/` | The audit (9,242 charts, 0 failures) shows conversion fidelity, not grid adherence: the snap bound holds by construction |
| MIR anchor probe results | yes (features deleted 09-30) | mac `artifacts/evals/mir_anchor_probe_750_150_150_3seed/multi_seed_report.json` | Test, 148 songs, 3 seeds: audio over chart history −0.90 nats per choice (95% CI 0.84-0.97); MIR teacher over plain Mel −0.19 (0.15-0.23, primary, p < 1e-4); novelty carries it, tempogram about 0. Undocumented. The only pre-V3 measurement of how much audio says about where rows go. |
| Control V3 feature tables and window indexes | yes (3.2 GB features) | mac `artifacts/features/`, `artifacts/indexes/` | |
| WebSocket service, protocol, streaming | yes | `main:src/ensomi_model/inference/` | Measured latency on one song (section 2, P5) |
| Reamber render wrapper | yes | `main:src/ensomi_model/evals/mapper_render_reamber.py` | |
| Gold diffusion run and 72 unreviewed samples | yes | mac `artifacts/gold_diffusion/train-20260907T151812Z/` | |
| Two pre-V3 generated charts | yes, git only | `0e63b0e`, `a35dbc6` | Section 3 |

Pending change: the pushed, unmerged branch `cleanup/main-r1` (`d8082d0`, 09-30) moves everything above that is in `main` to `legacy/v2` (= `5c56e28`) and removes `beat-this` from the dependencies. Its commit message states "The legacy boundary warning goes away with the legacy code".

## 7. The README ban

Text (added in `dfb4618`, 2026-09-01, in the same commit as the V3 formulation): "Do not use mapper v2/v2.1, the pre-V3 timing stack, Control V3, or the training, inference, configuration, protocol, and test code built around them as design, correctness, or implementation references for Ensomi V3." The next sentence: "That limited ownership does not make their tokenization, timing representation, control targets, model interfaces, or runtime structure part of the V3 contract."

- By its wording it is a **statement about authority and contract**: the formulation, not the old code, defines V3. It is not a quality verdict.
- Quality reasons exist for two of its three named parts: the mapper (bottleneck, README 07-10) and Control V3 (argued, README 08-31 and formulation §9). For the **timing stack no quality reason is recorded**; its own README calls it a usable prior. Grade: doc-claim, with no contrary record found.
- The commit message ("update README and extend formulation") gives no reasoning. Who decided and why is not in git. The human's pre-September conversations are outside what I could read (see section 11).
- It does not forbid measuring against the old stack as a baseline. Whether it forbids reusing BeatThis as an input or a comparator is ambiguous, and it was read that way. In September the lineage said the joint model uses "no redline, BPM or BeatThis" and pointed at this boundary (`audio-joint-2026-09:docs/research/audio_joint_expert_question.md` lines 35-37 and 306-307). Meanwhile the human's vision message of 09-23 listed "timing/BeatThis" as part of the target system, and on 09-28 asked for local BPM/phase (feedback index, section 2 rows 09-23 07:23 and 09-28 06:14; paraphrase, private V09, V74).

## 8. Lessons recorded before V3 that the September lineage repeated

Each has a citation on both sides.

1. **Millisecond time tokens without a musical coordinate give off-grid rhythm.** Before: the 05-20 drift audit ("object times are exactly aligned to a 10 ms grid, but that is not the same as aligning to the chart's beat grid", `0e63b0e`); the 06-01 roadmap ("move core map representation away from raw seconds and toward beat-relative positions", `02adb53:README.md`); the 07-10 README ("musically off-grid"). The converter was built (06-03) and never trained on; the June event-group v3 kept time shifts. September: integer-millisecond events with no grid by design (`audio_joint_expert_question.md` 35-37; `native_pattern_failure_analysis_zh.md` line 47; `planned_audio_continuation.md` line 18); the human's complaint that subdivision was irregular (feedback index, problem C, V70). Strong.
2. **Lower loss is not better free-running output; rollout gates decide.** Before: "Stable auxiliary-loss training is not enough; rollout gates remain decisive. The loss can improve while generated timing structure worsens" (06-17, `research/beatmap:.../target_grammar_v3_post_time_shift_full32_route_synthesis_result_report.md`); the 05-20 postmortem (EOS only forced by grammar, an empty chart from a trained checkpoint). September: near-equal NLL across three arms and selection moved to generated charts (feedback index row 09-24); the human called NLL a proxy (V27, V28). Moderate to strong.
3. **Mechanical and throughput guards pass on degenerate charts; output sanity must be a guard.** Before: "The visualization exposes a qualitative failure mode that the throughput guards did not detect... Treat output-quality sanity as a prerequisite guard" (07-15, `a35dbc6` result log). September: 48 charts passed legality and export, then the human reset the target to playability (rows 09-23 08:10 and 08:45, V10-V12); the demo checkpoint showed a long-jack regression after benchmark work (09-26, V46-V47). Strong.
4. **Window statistics of finished charts as control targets.** Before: the Control V3 critique (08-31). September: difficulty, LN fraction and style as scoped request values (`audio-joint-2026-09:docs/research/controlled_audio_continuation.md` lines 38-40); measured response stayed flat (problem I; LN ratio 25-30% whatever was requested). Plausible, moderate: I did not check how the September labels were computed. Slice 06 should confirm.

A pattern that was never recorded as a lesson: architecture was judged from tiny runs. Examples are the v2.1 mapper at 0.51 passes, the June v3 gates at 8-500 updates, gold diffusion at 300 updates, and the September pilots. The June reports repeat "does not prove convergence" without turning it into a rule.

## 9. What is worth keeping

- The **measurement harness and data** of the timing stack: the corpus-wide grid comparison, the 5,050 fitted grids, and the finding that the error is mostly a constant offset. It is a cheap first baseline for the `time` node.
- The **beat-relative converter**, which puts any chart into beat coordinates at 1/48 resolution, and the timing v3 **phase-continuous section schema**. Both answer "in what coordinate are times proposed" and have never been used for training.
- The **MIR anchor probe** as a ready audio-information measurement: it says novelty features tell where rows go beyond history and Mel.
- The **two pre-V3 outputs and the v2.1 checkpoint** as a floor baseline of an audio-to-chart system on the same songs. They are not a design reference. The checkpoint's weak outputs are what a 30M token mapper at half an epoch does, not evidence against token mappers.
- The **streaming service latency numbers**, as the only measured real-time path in the repository.
- Leaving these was reasonable: the mapper checkpoint and its decoding; Control V3 as a state representation; synthetic tempo warps as acceptance; the 16 kHz Mel frontend.

## 10. Inconsistencies and questions only the human can settle

1. Does the ban cover the timing stack as a component or a baseline, or only as a design reference? Its wording supports the narrower reading. No quality reason for banning timing is recorded, and in September the human asked for BeatThis and local BPM/phase. Relates to H4 and H6.
2. Should `cleanup/main-r1` be merged before deciding (1)? It removes `beat-this`, the timing code and the MIR probe from `main`.
3. Was the ban's intent quality (mapper output was bad) or independence (a fresh formulation not anchored on old interfaces)? Git does not say.
4. The beat-relative roadmap of June (owner README) became "Instead of collapsing music directly into a fixed beat grid..." on 08-26 (`a341e04`), and the formulation allows internal beat coordinates. Which one is the current intent? H4 asks the same thing.
5. Was any pre-V3 output ever played or judged by the human? Nothing is on record.
6. Should the -29 ms offset be investigated before any timing comparison? It would change every "phase error" figure quoted so far.

## 11. What I could not verify, and how a follow-up would

- History before 2026-05-17: why v2 became v2.1, the v2 checkpoint's step, the human's reasoning for the ban. Sources not in git; the human, or `~/.codex/history.jsonl` on the mac (entries to 09-19), which I was not allowed to read.
- Whether the -29 ms offset is BeatThis's or the fitter's convention (for example peak picking on 20 ms frames) or a mapper convention: MP3 delay is ruled out (same median for OGG). A follow-up needs BeatThis's raw beat times on a few songs against chart note onsets, which means running BeatThis (inference, not allowed here).
- Mapper v2.1 quality at scale: not measurable without retraining; its caches are deleted.
- Gold diffusion samples were not measured. Run `measure.py` from my scratch directory on `artifacts/gold_diffusion/train-20260907T151812Z/samples/test/*/*.osu` against their sources; times are supplied, so only lanes, LN and same-lane gaps are meaningful.
- The September control labels (lesson 4): `audio-joint-2026-09:src/ensomi_model/research/controlled_audio_continuation/`.

Reproduction: `scratchpad/opus-pre-v3-history/measure.py <ref.osu> <other.osu>...` (charts copied from mac `dataset/0/{1942086,1086533}` and `git show 0e63b0e:... / a35dbc6:...`). `timing_stats.py` reads the full5050 JSONL on the mac (under a minute).
