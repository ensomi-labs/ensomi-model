# Pre-V3 lineage review — 2026-09-30

## 1. Scope and sources

This review answers slice `02-pre-v3-history`, May through mid-September 2026.
It distinguishes a runnable system, a successful bounded experiment, and a validated
chart generator. Those are different achievements. Later R1 and audio-skeleton
work is cited only to identify reused assets or repeated lessons.

Evidence grades used throughout:

- **doc-claim**: a contemporary document, note, or commit records the statement.
- **checked-code**: the implementing source was read, with revision/path/line below.
- **checked-artifact**: saved JSON, checkpoint metadata, index, or chart was read.
- Judgments and confidence are this reviewer's inferences from the graded evidence.

I read the research entry point and feedback-index sections 2–3; the complete
`main` commit list and archive-ref inventory; branch-specific commits on all eight
requested branches; baseline README/formulation; and selected reports and source.
The branch logs were inspected as `main..refs/archive/heads/<name>` after the shared
`main` history. Commit dates date repository records, not necessarily experiment
execution: May begins with migrated code; diffusion ran September 7 but was saved
in commit `761e1e5` on September 10. No original pre-migration history was recovered.

Source shorthand, used only to shorten repeated citations:

| Id | Grade and source actually inspected |
| --- | --- |
| D1 | doc-claim: `dfb4618:README.md`, `8e5e7ad:README.md`, and `5c56e28:README.md`; the boundary, its introduction, and the preceding Control V3 criticism. |
| D2 | doc-claim: `125e6d8:docs/research/timing_v3_research_timeline_2026-08-16.md`, `timing_v3_phase_1_completion_audit.md`, `timing_v3_decision_2026-08-17_ground_truth_and_osu_note_grid.md` in the same directory. |
| A1 | checked-artifact: surviving mapper/control checkpoints and mapper `report.json`, enumerated exactly by `/tmp/lineage-review/02-pre-v3-history/checkpoint_inventory.json` and `training_audit.json`. |
| A2 | checked-artifact: `a35dbc6:artifacts/evals/inference_bundle_mps_profile/summary.json` and its generated `beatmap/oyasumi_bundle_v2_1_sparse_diff4_seed0.osu`; also `0e63b0e:artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_best_low_temp_seed0.osu`. |
| A3 | checked-artifact: `f411d25:artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json`; and `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json` at that revision. |
| A4 | checked-artifact: `artifacts/reports/timing/timing_v3_v2_baseline_full5050_v1_summary.json`, `timing_v3_exp022_pilot42_authoritative_summary_v1.json`, and `timing_v3_exp025_jump20_postfreeze_audit_v1.json`. |
| A5 | checked-artifact: `artifacts/local/timing_v3/beatthis_frontend_fulltrack_boundary_oof256_v1/{result.json,contract.json,fold_models/fold_0.pt}`; all eight fold checkpoint files exist. |
| A6 | checked-artifact: `artifacts/gold_diffusion/train-20260907T151812Z/{manifest.json,metrics.jsonl,status.json,evaluation.json,validation_samples.json,test_samples.json,best.pt}`. |
| A7 | checked-artifact: `artifacts/evals/mir_anchor_probe_750_150_150_3seed/{multi_seed_report.json,multi_seed_inference.json,resolved_probe.yaml,runs/seed_0/summary.json}` and existence of three `probe_state.pt` files. |
| A8 | checked-artifact: `artifacts/scoped-style-modeling/{probe-b-v1,pilot-c-v1}/summary.json`; `artifacts/source-action-relation-matching/pilot-20260915-01/summary.json`. |

All calculations were CPU-only, with at most two Torch threads and no model forward
passes. Scripts and extracted sources live only in the assigned scratch directory.
I did not inspect private feedback, run training/inference, listen to reconstructed
audio, or perform a new human-quality review. Missing files mean unavailable here;
they do not prove deletion by the September 30 cleanup specifically.

## 2. Attempts

**Question 1 — paradigms in chronological order.** The table includes auxiliary
representations and probes because several were never complete generators.
“Not recorded” means absent from the inspected evidence, not necessarily absent
from every historical file. Training populations are not counts of examples seen.

| Paradigm; dates and problems | Representation, clock, audio | Scale, data, execution | Reached; present source/results |
| --- | --- | --- | --- |
| Tuple mapper v2; imported May 17–18, retained through August; C/E/I | Autoregressive complete four-lane event tokens plus time shifts; 10 ms grid. Packed legacy Mel, dense beat features, difficulty, frozen control memory. | Surviving large mapper: 106,913,126 mapper parameters plus 15,004,803 frozen-control parameters; 11,000/12,000 steps, MPS, seed 1337; 64,381 train/512 eval windows. Chart/song counts and wall time not recorded for this checkpoint. | Trained checkpoint and runnable legacy inference path; no independently checked v2 generated-quality result. `5c56e28:src/ensomi_model/models/mapper/v2/`; A1. |
| Sparse mapper v2.1 and decode repair; May 18–July; A/B/C/H/I | Time-shift tokens plus explicit lane TAP/start/end tokens; NONE omitted; same 10 ms clock and audio/control chain. | 14,967,022 mapper + 15,004,803 frozen control = 29,971,825 total; report at 44,000 steps, surviving checkpoint 44,250; seed 1337, MPS. 172,649 train/1,866 eval windows; underlying filtered index 9,242 charts/3,641 audio paths, not verified unique songs or training exposure. Wall time not recorded. | Audio-to-chart export and streaming worked; density, rhythm, LN and repetition quality remained poor. A1/A2; `models/mapper/v2_1`, `inference`, `training` retained at baseline. |
| Control V3 and global control encoder; present at May import, rejected as gameplay state August 31; D/G/I | Twenty completed-chart statistic/confidence channels; 20 ms target grid, symmetric time support; learned from legacy Mel/timing/difficulty, with a global-memory extension. | Base 8,288,276 parameters at 8,000 steps; global 15,004,803 at 2,000 additional-stage steps; seed 1337, MPS. 706,731 train/1,036 eval windows; source index 9,242 charts/3,641 audio paths. Wall time not recorded. | Trained and used by legacy mapper; no calibrated player-response state established. A1; `5c56e28:src/ensomi_model/features/control_v3.py`, `models/control/`. |
| BeatThis + GridFitter timing; May–June, August baseline audit; E/H | Beat/downbeat probabilities at 50 Hz to piecewise-constant BPM/offset grid, then dense features. Predicts a musical grid, not selected note heads or LN releases. | Pretrained BeatThis `final0`; parameter count/training data/compute not recorded in repository evidence inspected; fitter is non-neural. Full baseline: 5,050 audio identities, 14,617 successful chart comparisons. | Runnable timing component; broad weak-comparator measurements survive in A4. `5c56e28:src/ensomi_model/timing/`. |
| Beat representation, motifs/LN tokenization, CASF/C3; June 3 and 16–17; B/C/F/J | Beat coordinates, default 1/48-beat snapping; corpus motifs plus lossless chart-local LZ fallback side stream. No audio in codec audit; later auxiliary mapper conditioning/targets. | Codec corpus: 9,242 charts/3,573 mapsets; train 7,350/2,849, validation 927/362, test 965/362. No neural parameter count for codec. Seed 17; raw wide-sweep runtime 819.045 s; 200 mapset-bootstrap samples. | Solid compression/reconstruction evidence, unsuccessful or undeployable tested mapper integrations. `refs/archive/heads/research/{beatmap,token-LN-BPE}`; A3; some auxiliary weights survive. |
| Local event-group mapper “v3”; June 17; C/H/J | Reuses complete four-lane event alphabet, 10 ms time shifts; optional delta/kind/signature/end-gap factor targets. Audio/control-conditioned legacy model, distinct from September V3. | Representative matched timing-loss arms: 500 updates each; 32 rollout pairs; parameter count, wall time, distinct charts/songs and seed variation not recorded in inspected gate. Final full-index factor smoke: width 32, one layer, CPU, seed 20260703, 8 updates/73.273 s, 174,335 train/180 eval windows; parameter count not recorded. | Reversible grammar, bounded learning and runtime smokes; no replacement-ready generator. Last commit `f411d25` recommends longer training, not rejection. Source remains archive-only; selected checkpoint/cache paths absent. |
| Timing v3; August 11–20 archive; E/H/J/K | Continuous absolute beat axis; constant/jump/ramp candidates, source-only audio evidence, later ACF and learned boundary detection. No chart arrangement generator. | Many analytic gates; 42-song and 20-jump diagnostics. Later 47,393-weight boundary model, 256 synthetic jump routes, eight source folds, 50 epochs, seed 20260816, 980.013 s CPU. Separate fresh128/384-route external run unfinished. | Useful representation/diagnostics and synthetic boundary success, no natural-tempo product acceptance. `125e6d8:src/pulsefield_model/timing/v3/`; A4/A5; D2. |
| Mel metamer frontend; August 18–19; E/H | Direct waveform optimization to match log-Mel; legacy 16 kHz/80 bins/25 ms versus 24 kHz/128 bins/40 ms; both 10 ms hop. Not a generator. | One six-second excerpt, seeds 0/1/2 per frontend; no learned model parameters. 938–1,240 Adam steps; 18.561 s total reported CPU time. | Matching-error audible comparison led to frontend preference; no downstream chart test. `5c56e28:docs/research/mel_frontend_metamer_result.md`; raw notebook summary checked. |
| MIR anchor-position probe; August 22; E/H | Conditional choice between a real next-row location and 16 within-map controls at the same elapsed gap; acoustic, novelty, tempo/phase features plus history. Millisecond targets; 5 ms Mel/novelty, 20 ms tempogram. | 270,549 parameters; seeds 0/1/2, 20 epochs. Effective train/validation/test audio 740/149/148; test 155,533 cases. Optimizer update count, wall time, and resolved device not recorded in inspected summaries. | Positive held-out audio-feature result with seed variation; no autonomous timing or complete chart rollout. A7; `14a441a`; baseline `evals/mir_anchor_*`, `features/mir_backbone.py`. |
| Gold fixed-placement diffusion; September 7 run/September 10 archive; B/C/I | Masked categorical denoising over 255 nonempty rows; supplied exact source timestamps and entry/exit LN occupancy; local ±200 ms music Mel, human scoped style label, source history. | 1,165,443 parameters; 113 train scopes from 94 charts/92 song groups; validation 11 charts/11 groups, test 12 charts/11 groups/14 scopes. Seed 73; MPS, batch 4; 1,100 updates, last logged training elapsed 130.91 s. | Exported scoped substitutions; best step 300; stopped on validation patience. No recorded human judgment in 72 sample records. `761e1e5:src/pulsefield_model/experiments/gold_diffusion/`; A6. |
| Scoped style classifier; September 13–14; G/H/I | Noncausal chart event/relation encoding, five independent ordinal style labels; physical and beat-relative features in later probes; no audio generation. | Historical 2,701 updates per arm, seed 17, 3,526 machine training cells; 79 human validation cells/26 groups; 4,821.50/4,614.41 charged seconds. New B: 272 updates; C: 423; 106,423–175,941 parameters; 3,200 mixed cells; 61 human validation cells/18 groups. | Useful diagnostics, weak/unstable semantic detectors. Transition into source-action learning, not an audio-chart system. A8; baseline `research/scoped_style_modeling/` and postmortem. |
| Source-action prediction and relation composition; September 14–15; C/G/H | Conditional hidden-block action prediction retaining multiple representation levels; original action times, explicit local support; no audio. Oracle-time causal continuation begins September 15. | Query pilot: 340,913 versus 345,009 parameters, 300 paired updates, seed 17, 964.619 s training/1,309.469 s total MPS. Sampled 2,061 charts/1,690 groups of 11,564/3,169; 32-group/192-block validation. | Exact prediction/replay prototypes and controlled no-gain result for query matching. `9afa8a0`; baseline `research/source_action_modeling/`; A8. Later R1 is outside this slice. |

Parameter counts above separate frozen control from mapper capacity. The trainer
attaches the control encoder, while its generic count includes all parameters:
**checked-code**, `5c56e28:src/ensomi_model/models/mapper/shared/model.py:97`
and `5c56e28:src/ensomi_model/training/mapper_runner.py:549,612`. Counts were checked from five checkpoint
files and meta-device construction using the scratch scripts; no forward pass ran.

**Question 2 — did anything already generate charts from audio end to end?**

| System | What is established | Evaluation and quality boundary |
| --- | --- | --- |
| Mapper v2.1, May Riria | Yes: real audio → BeatThis grid → control encoder → sparse mapper → `.osu`. Archived raw export has 696 objects, 26 holds and 529 distinct head times (A2; own parser, n=1 chart). | Original greedy run emitted only six lane actions and ten empty windows of eleven. Decode sweep recovered density: selected seed-0 low-temperature run 722 lane actions versus reference 713. Same-song policy tuning, seeds 0/1 in the low-temperature comparison; no training-seed replication. The report explicitly calls it a candidate, not a musical-quality solution. **doc-claim**, `0e63b0e:artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_decode_policy_report.md`. |
| Mapper v2.1, July Oyasumi | Yes: A2 raw summary verifies six completed windows and stable output hashes over three measured repeats, one warmup, one diagnostic run; one 47.151 s song, greedy seed 0, requested difficulty 4. | Mean request compute 17.829 s, first protocol token 2.227 s; warm Mel cache, delivery pacing excluded. Raw chart has 1,615 TAPs, no holds, 472 head times. These are throughput/completion results, not evidence of requested difficulty or ordinary organization. No scored human-quality judgment found in this record. |
| Earlier tuple v2 | Code implements the audio-to-chart path and a trained checkpoint exists. | Could not verify a particular end-to-end exported chart or quality evaluation for v2 itself from inspected records. It must not inherit v2.1's results. |
| June event-group v3 | Contemporary full-pipeline report claims bounded real-audio runtime rollouts. | `f411d25:artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_pipeline_replacement_summary.json` verifies the smoke route; later 32-pair generated-state gate failed. This is bounded rollout evidence, not a verified full-song quality result. |
| Diffusion, style, source-action, timing, MIR | No complete audio-only chart generation established by these experiments. | Diffusion supplies source times/history/boundaries and retains the surrounding chart; MIR supplies conditional choice sets; timing supplies beat grids. Their successes cannot be counted as end-to-end chart generation. **checked-code** citations below. |

For the newer style arms, A8 records charged seconds 951.61/967.17 for B
and 1,325.60/1,209.63/1,313.71/1,299.10 for C0/CT/CM/CTM.
These include preparation and diagnostics, not just optimizer time; no seed
variation was measured. Training chart/song counts are not separately recorded
in these inspected summaries. Their denominators are annotation cells/groups.

The Riria drift report further records only 11.4% of generated starts within 5 ms
of its own 1/16 grid, versus 97.3% for the reference; correcting a 20 ms offset did
not solve it. This is a **doc-claim**, not a newly recomputed musical alignment
result: `0e63b0e:artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_timing_drift_audit.md`.

**Question 3 — what justified leaving each direction?**

| Direction/decision | Recorded reason and subsequent action | Assessment of the decision's scope |
| --- | --- | --- |
| v2 → v2.1 | Sparse lane vocabulary rationale is in `5c56e28:src/ensomi_model/models/mapper/v2_1/vocab.py:21`; migration history starts after v2 already exists. | **checked-code** representation difference; no recorded matched quality or compute comparison establishing why v2 should be abandoned. |
| v2.1 decode changes | May density failure led to time-shift penalties/sampling; July profile targeted mapper overhead. August README still reports sparse/repetitive/off-grid quality. | **doc-claim**, May reports and `dfb4618^:README.md`; reasonable symptom repairs, no demonstrated cure of learned chart organization. |
| v2.1 further training | MPS continuation 44,000→44,250 consumed memory; root-cause report distinguishes retained heaps from unexplained process-footprint growth. | **doc-claim**, `5c56e28:docs/research/mapper_v2_1_mps_memory_root_cause_report.md`; resource problem is real in the account, but no explicit record says it alone caused the later paradigm reset. |
| C3 as pooled conditioning | Full-pipeline audit identifies target-chart sidecar unavailable during generation and kills promotion of that input path. | **checked-code**, `f411d25:src/pulsefield_model/data/mapper_sparse_windows_v2_1.py:212`; lookup uses target beatmap/window. Strong contract reason. Codec itself explicitly retained. |
| C3 auxiliary-target variants | Reduced kind heads after 1,000 updates trail unigram recall@20; subsequent factor/context probes have diminishing returns. | **doc-claim**, `f411d25:artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_c3_diminishing_returns_route_synthesis_result_report.md`; supports deferring these integrations, not rejecting chart-local repetition. |
| June mapper v3/time-shift distance | After NaN row-filter repair, token loss improves slightly but full32 rollout starvation and rigidity worsen. Other factor-target work continues to the eight-step full-index smoke. | **checked-artifact** A3; justified negative for that objective. The final branch has no recorded global rejection of event-group representation. |
| Timing v3 analytic search | Runtime failure, selector/candidate failure, and lack of generalization beyond two exposed mechanism examples; later broader failure and truth-proxy retraction. | A4 **checked-artifact**, D2 **doc-claim**. Evidence supports stopping particular searches and withholding product claims. It does not establish that continuous beat axes are wrong. |
| Legacy Mel | One listener preferred high-frequency attack retention at matched reconstruction error on one excerpt. | **doc-claim** listening; raw convergence **checked-artifact**. Preference informed a candidate frontend selection, not an isolated causal finding about sample rate or mapper quality. |
| MIR probe | Positive three-seed held-out result remains in A7. | No recorded rejection or explicit reason for not integrating it found. It is a conditional evidence probe, so absence from generation is not itself an error. |
| Control V3 → formulation | `8e5e7ad` says future-dependent statistics are not a persistent generative player state and lack transitions/durations/finger relations/memory. `dfb4618` introduces the blanket boundary. | **doc-claim** conceptual mismatch, partially **checked-code** by symmetric kernels. No isolated intervention proves each proposed cause of quality failure. |
| Diffusion | A6 says `validation_patience`, best 300, last 1,100. Archive commit subject is merely `stash`. | No recorded architectural rejection or human-quality failure in checked samples; observed validation deterioration is evidence for the stop rule, not a verdict on diffusion. |
| Scoped classifier → source-action | Postmortem rejects broad success claims; representation-directions document argues five labels cannot exhaust source-action organization. | **doc-claim** conceptual broadening plus bounded failures; no evidence that source-action pretraining had already fixed style semantics. |
| Query matching → additive reference | Gain −0.00023594, group-bootstrap interval [−0.00051673, +0.00000713], misses declared 0.02 gain. | A8 **checked-artifact**; sound defer-adoption decision at one seed/300 updates. No general rejection of query-dependent attention. |

## 3. Commentary

**Question 4 — what did the timing stack actually do, and is it usable today?**
BeatThis returns beat and downbeat logits, converted to probabilities at 50 Hz;
GridFitter searches and splits timing segments and handles tempo aliases.
The session renders those segments into conditioning features. It does not decide
which subdivisions deserve note heads, nor which keys or LN endpoints to output.
**checked-code**: `5c56e28:src/ensomi_model/timing/providers/beatthis.py:81`,
`5c56e28:src/ensomi_model/timing/grid_fitting/fitter.py:51`,
`5c56e28:src/ensomi_model/inference/session_runtime.py:242`.

The broadest checked pre-V3 timing result is A4: 5,050 fits succeeded, 5,026 audio
identities had usable comparators, and 14,617 chart comparisons succeeded; 24 audio
identities lacked comparators. Audio-weighted mean phase error was 46.566 ms;
median per-audio mean error 43.000 ms; p90 of per-audio mean error 74.039 ms.
Mean alias-aware BPM MAE was 1.662, versus 33.167 without alias handling.
These are agreement with chart timing, not beat-annotation accuracy.
A4 reports cache-backed fitter p50/p90 2.102/4.217 s; BeatThis feature extraction
is excluded. The 3,031.574 s run wall time is not the sum of individual fit times.

Earlier narrower claims were stronger than their scope: 100-map phase error
48.69→42.54 ms and likely-real multi-BPM recall 47/58 with 29/97 artifact false
positives. These remain **doc-claim** in `5c56e28:src/ensomi_model/timing/README.md`.
The June ramp “100% recall, 0/500 false positives” result was recognition of parsed
redline shapes, explicitly not audio inference (`1717517:artifacts/evals/bpm_ramp_timing_detection/result_log.md`).
It must not be used as evidence that gradual tempo change in audio was solved.

My call: the baseline fitter is a usable **research comparator**, conditional on
restoring its pinned local BeatThis assets and agreeing what it measures.
Runtime/source exist, but I did not execute them; legacy cache directories needed
for exact historical replay are absent in sampled paths. Restoring the pipeline
requires dependency/checkpoint availability, relative-path repair where needed,
and an independent natural-audio timing reference for any accuracy claim.
A beat grid could be compared with a head scheduler only through explicitly
separated tasks; one cannot silently stand in for the other. Confidence: high
on task distinction; medium on current runnability without a smoke run.

**Question 6 — the “failed timing v3” and diffusion experiments.**
Timing v3 was not one failed neural model. It combined phase-continuous schemas,
projection/search, candidate retention/selection, weak-label audits, synthetic
warps, and later learned boundaries. Exp022's 42 outputs were all marked accepted,
but p90 runtime was 11.724 s; Exp025 classifies 20 jump cases as 12 not generated,
5 pruned, 2 retained but ineligible, 1 selected. These **checked-artifact** outcomes
locate different failure mechanisms; “accepted” was an execution status.

The later boundary model deserves separate treatment: A5 reports 217/256 within
1 s, median error 107.098 ms, p90 2.520 s, maximum 116.137 s, across eight source
folds. The contract fixes 50 epochs and seed 20260816; raw fold weights survive.
That is positive evidence on injected jumps, with a serious tail. It is neither
natural-song validation nor a joint constant/jump/ramp solution. D2 records the
fresh external run stopping after 118/384 frontend caches, before final predictions
or truth opening, and the August 17 decision withdrawing synthetic product acceptance.
Thus “failed” combines failed algorithms, an unfinished external test, and a
corrected measurement target. It does not erase the positive bounded result.

Diffusion is likewise a bounded task: fixed placement, source history, and scoped
style; legal-path sampling conditions on original entry/exit occupancy.
**checked-code**: `761e1e5:src/pulsefield_model/experiments/gold_diffusion/model.py:78,102`
and `761e1e5:src/pulsefield_model/experiments/gold_diffusion/run.py:97`. A6 shows validation CE 3.616 at step 300 rising to 4.072 at 1,100;
test CE at selected checkpoint 3.558, row accuracy 0.1274, lane accuracy 0.4917.
There is no alternative architecture comparator, no training-seed variation,
and all 72 recorded human-style judgments are null. This run cannot establish
that diffusion failed at audio-to-chart generation, because that task was not tested.

Other inference limits matter. C3 compression gains are measured on a much broader
corpus than its small neural integration tests. A successful codec is not a learned
proposal distribution. Conversely, neural integration failures do not nullify its
lossless reconstruction result. The style postmortem already makes this distinction
well: its added capacity and limited updates do not isolate inductive bias, and
one Tech-positive validation cell cannot establish robust Tech recognition.
Sources: A3/A8 **checked-artifact**; scoped-style postmortem **doc-claim**.

## 4. Direction

**Question 7 — why the README ban, and how strong is that reasoning?**
The exact blanket instruction appears in `dfb4618`, after `8e5e7ad` criticized
Control V3 and `a341e04` changed the research focus. The inserted explanation says
legacy compatibility does not make tokenization, timing, control targets, interfaces,
or runtime structure part of the V3 contract. D1 is a **doc-claim** about authority.
The preceding README also documented poor mapper quality and described Control V3's
failure explanations as hypotheses. No matched all-stack rejection experiment is
cited in the boundary text. The best-supported reading is contract separation,
with dissatisfaction as background; an empirical ban on every component is unproven.

Both sides of the principal direction judgments:

| Direction | Strongest case for it | Strongest case against it | Call and confidence |
| --- | --- | --- | --- |
| Leave legacy mapper/control as V3 authority | The desired response-state contract differs materially; generated density and grid agreement did not ensure organization (D1/A2; `5c56e28:src/ensomi_model/features/control_v2.py:194`). | No broad human-quality comparison proves a new generator superior; older end-to-end runtime and weights were already real assets. | Good authority boundary, insufficient evidence for wholesale technical invalidation. High on contract, medium on strategic quality judgment. |
| Stop decode-only/time-shift-loss repairs | A3 directly shows loss improvement with worse native starvation/rigidity; repeated symptoms justify declining promotion. | Small/short training and fixed evaluation cases leave undertraining and generalization unresolved. | Stop that objective/decoder recipe, retain representation uncertainty. High. |
| Retain C3 as a codec, defer tested model integration | Corpus-scale exact reconstruction and charged savings; target-conditioning provenance blocks deployment. | Compression and side-stream sparsity do not establish easier generation; auxiliary recall remained poor. | The recorded narrow split decision is sound. High. |
| Stop broad Timing v3 promotion | Natural-tempo truth absent, 42-case generalization/runtime failure, synthetic-product proxy retracted. | Continuous coordinates, a large weak-comparator baseline, and a positive 256-route learned boundary result survive. | Withholding product claims was justified; calling the entire timing effort useless is not. High. |
| Select richer Mel / retain MIR evidence | Matched-error reconstruction and three-seed MIR test both show audio information worth examining. | One listener/excerpt for Mel; MIR's contrived candidate sets differ from native scheduling. | Keep bounded information evidence, withhold generator claims. High. |
| Move from style labels to source-action prediction | Classifier failures and five-label bottleneck motivate a richer task. | Small human cohorts, short training, and no native chart test do not isolate representation as the cause. | Plausible direction, not empirically established replacement. Medium. |
| Leave gold diffusion | Small gold-only training deteriorated on validation. | It received about two minutes of logged training, one seed, no architecture comparator or recorded human sample assessment. | Stopping that run is supported; rejecting diffusion as a paradigm is not. High on evidence boundary, low on its ultimate quality potential. |

## 5. What was overlooked or never questioned

Ranked by consequence; these are evidence-bounded gaps, not an architecture plan.

1. **A common chart-quality comparison across paradigms was never established in
   the inspected records.** May/july generated charts, codecs, timing, MIR, style,
   and diffusion optimize different tasks. A2–A8 contain no shared blinded chart
   test. Consequently “left behind” cannot be equated with “beaten.” Confidence high.
2. **Whether failures were representation failures or insufficient learning remained
   unresolved.** Eight-update factor training, 272/423-update style probes and
   one-seed diffusion are bounded evidence. The oracle-time expert question itself
   explicitly raises underexposure as an alternative (`5c56e28:docs/research/oracle_time_expert_question.md`).
   That is a documented unresolved question, not a newly discovered proof of undertraining.
3. **What real audio information was already available was not carried forward as a
   comparative result.** A7's primary MIR gain is 0.191227 nats, audio-cluster 95% CI
   [0.148336, 0.232629], between-seed SD 0.014446 on 148 test audios. No research-package
   references to `mir_anchor` were found at `audio-joint-2026-09`; the assay remains
   unused there. This supports preserving a comparison, not assuming MIR fixes generation.
4. **A declared player-response target still lacked independent validation.** The
   August Control V3 critique says chart statistics are not that state; September's
   formulation still leaves response quantities/comparisons to mapper evidence.
   Sources: D1 and `5c56e28:docs/formulation/README.md`, both **doc-claim**. Merely
   changing the state representation cannot demonstrate the missing semantic contract.
5. **Selection and proposal failures needed distinct measurements.** Exp025's
   12/5/2/1 lifecycle buckets show why repeated retention/selector repairs could
   not fix many missing proposals (A4). This is stronger evidence than a universal
   statement that increasing candidates never helps.

**Question 8 — recorded lessons that recur in the audio-skeleton lineage.**
These are repeated observable evaluation problems, not claims about an agent's intent.

| Earlier lesson | Later citation | Bounded inference |
| --- | --- | --- |
| June: token loss −0.003974 while rigid cases rise 10→13 and starvation 5→8 over 32 pairs (A3, checked-artifact). | `/tmp/lineage-review/trees/audio-joint/docs/research/r1_transfer_stability_audit.md`, “Joint initialization comparison”: 1,200-update arms have NLL/s 40.40358/40.05814/40.06002 but native screen incomplete; feedback index H records rejection despite proxy passes (doc-claim). | H/J recur: likelihood ranking still did not establish native chart quality. The later audit acknowledges the limitation; it should not itself be portrayed as hiding it. |
| May: recovered action count leaves off-grid rhythms and only 26 holds versus 224 in reference (A2 counts; drift report doc-claim). | `/tmp/lineage-review/trees/audio-joint/docs/research/native_pattern_failure_analysis_zh.md`, overview evidence table; feedback index A–C/H: improved metrics coexist with long single-lane attacks and fragmented LN (doc-claim). | A/B/C/H recur: totals and legality do not identify ordinary arrangement. This is analogous failure, not proof of an identical cause. |
| August: two exposed timing mechanism examples fail to generalize to 42; D2 explicitly warns against local repair extrapolation (A4 plus doc-claim). | `/tmp/lineage-review/trees/audio-joint/docs/research/ordinary_expert_from_scratch_zh.md`, §7: four songs/16 four-second units/512 updates, explicitly a capacity-fit pilot (doc-claim). | J/H: narrow pilots remain too weak for broad architectural verdicts. The later text explicitly preserves that limit; I found no basis here to accuse it of claiming a corpus result. |
| August 31: handcrafted completed-chart summaries do not define a persistent response state (D1); symmetric support verified in `5c56e28:src/ensomi_model/features/control_v2.py:194`. | `/tmp/lineage-review/trees/audio-joint/docs/research/native_pattern_failure_analysis_zh.md`, “definition or algebraic conclusions”: attack-count-only responses cannot distinguish release organization; feedback G (doc-claim). | G/H recur: a readout cannot recover distinctions absent from its inputs/targets. This does not prove that all later response modules repeat Control V3's exact implementation. |

I did not find sufficient paired evidence for a claim that the June C3 target-input
leakage bug itself recurred in the audio-skeleton line. Similar words about
conditioning or teacher forcing are not enough to establish the same bug.

## 6. Worth keeping

**Question 5 — assets that exist, and their actual reuse status.** File existence
was checked directly; it is not a guarantee of successful loading or good output.
No broad checkpoint scan outside named or source-cited directories was performed.

| Asset | Disk/source check | Relationship to V3/audio-skeleton |
| --- | --- | --- |
| Large tuple mapper | `artifacts/runs/stage2_mapper_v2/stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt` exists, 1,343,757,335 bytes; metadata loaded CPU (A1). | No mapper-v2 import found in lineage research packages. Historical baseline candidate, not a V3 contract. |
| Sparse mapper | `artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoint.pt` and `checkpoints/checkpoint_step_044000.pt`, `checkpoint_step_044250.pt` exist (A1). | Legacy inference comparison remains possible after dependency/config verification. |
| Control encoders | `artifacts/runs/stage2_control/stage2_control_12k_mps_d384_l3_4x/checkpoints/checkpoint_step_008000.pt`; `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt` exist (A1). | No Control V3 conditioning reuse found in the newer research path; do not call them player-state labels. |
| C3 learned auxiliary probes | `artifacts/runs/stage2_mapper_v2_1/c3_kind_heads_reduced_sidecar_1000step/checkpoints/checkpoint_step_001000.pt` exists; sampled conditioning/auxiliary runs also retain weights. | Associated code is on `f411d25`, not the baseline's default mapper; target-side probes only. |
| C3 codec and beat tokenizers | `f411d25:src/pulsefield_model/osu_core/c3_side_stream_tokenization.py` and beat representation/audit sources available through Git. Referenced `artifacts/audits/beat_chunk_patterns/beat_chunk_pattern_cache_le3.parquet` absent. | Broad checked compression result survives in committed JSON; rebuilding the absent corpus cache is required for recomputation. |
| Timing-v3 learned boundaries | Eight `artifacts/local/timing_v3/beatthis_frontend_fulltrack_boundary_oof256_v1/fold_models/fold_*.pt` exist, each 195,161 bytes; saved predictions, contract and `run.py` exist. | No import of this archived experiment found in lineage research. Synthetic-boundary mechanism comparator only. |
| Timing datasets and reports | `artifacts/reports/timing/timing_v3_{inventory,labels}_v1.jsonl` and A4 summaries exist. `artifacts/local/timing_v3/{stable256_causal_v2,fresh128_causal_final_v2}` directories exist; sealed contents not opened. | Useful provenance and stress-test assets; directory existence does not verify all WAVs or authorize treating synthetic truth as natural tempo. |
| MIR probe | Three `artifacts/evals/mir_anchor_probe_750_150_150_3seed/runs/seed_{0,1,2}/probe_state.pt`, manifest, per-audio tables and A7 summaries exist. | Strongest unused multi-seed audio-information result found; task-specific probe, not a head scheduler. |
| Gold diffusion | `artifacts/gold_diffusion/train-20260907T151812Z/{best.pt,last.pt,data_snapshot.pt}` plus source snapshot and sample manifests exist. | Archive-only experiment; neither baseline nor audio-joint research imports it. Gold scope labels remain an asset, but may share sources with later Lens-derived cohorts. Independence not established. |
| Filtered corpus/index/control features | `artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet`, matching `stage2_control_windows_...parquet`, and `artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet` exist. | The specific legacy window/target recipe is unused in inspected new paths. The broader `beatmap_index_4k.parquet` is reused by source-action; the dataset itself is not an unused asset. |
| Legacy evaluation/runtime tools | Baseline `src/ensomi_model/evals/mapper_v21_decoder_profiler.py`, timing diagnostics, tokenizers, inference bundles and tests survive; July benchmark JSON/charts survive in Git although their live directory is absent. | Portable measurement ideas and baselines, subject to the README's contract boundary. Current compatibility tests were not run. |
| Mel frontend/metamer | `notebooks/mel_frontend_metamer_experiment.ipynb`, `artifacts/mel_metamer_notebook/summary.json`, source WAV and reconstruction directories exist. | **Already reused**, so not a lost asset: audio-joint `joint_audio_continuation/data.py:20,230` imports `MUSIC_MEL_CACHE_CONFIG`. Initial skeleton also uses BeatThis features (`audio_skeleton/audio.py:1`). |

The strongest retainable findings are narrower than complete designs: C3 chart-local
repetition with exact reconstruction; distinction between musical grid and chart
placements; MIR conditional audio benefit; timing proposal-lifecycle diagnostics;
and scoped classifier reports that expose concept-specific failures instead of
relying on aggregate loss. A3–A8 support these claims within their recorded tasks.

## 7. Claims worth re-verifying

The following checks would change consequential conclusions; they are not a roadmap.
All commands below are read-only CPU analysis unless explicitly marked model work.

- To reproduce this review's checkpoint inventories and model-size decomposition:
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/lineage-review/02-pre-v3-history/checkpoint_inventory.py`, then `training_audit.py`; run `model_sizes.py` with `PYTHONPATH=/tmp/lineage-review/trees/main/src`.
  Inputs are the five paths in `checkpoint_inventory.json`; n=5 checkpoints, four model constructors, one timing-fold checkpoint. These counts exclude buffers from parameter totals.
- To reproduce chart/index counts: run `/tmp/lineage-review/02-pre-v3-history/chart_asset_audit.py` with the same Python/thread settings. Inputs are the two A2 Git charts and two named legacy Parquet indexes; n=2 charts and 9,242/707,767 rows. Audio paths are not deduplicated songs.
- To reproduce gold cohort counts and archived summary extraction: `python3 /tmp/lineage-review/02-pre-v3-history/artifact_audit.py`. Inputs and sample denominators are explicit in that script; n=138 gold scope records and 72 generated sample records. These are counts, not new quality scores.
- For actual natural-tempo accuracy, A4 cannot settle it. Needed missing source: independent unwarped beat/downbeat annotations and their exact evaluation cohort. The old 200-row “web-audited” comparator judgments are not substitutes for that reference.
- For the best late timing model, inspect `artifacts/local/timing_v3/beatthis_frontend_fulltrack_boundary_oof256_v1/{contract.json,oof_evaluated_rows.jsonl}` and D2's pause state before any external rerun. Fresh truth was not opened here. Model execution would require restoring the documented feature contract; no new natural-song claim follows from finishing synthetic fresh128 alone.
- For end-to-end legacy quality, archived raw charts are available in A2, but no multi-song scored human comparison was located. The exact prior benchmark command is in `a35dbc6:artifacts/evals/inference_bundle_mps_profile/result_log.md`; running it is model work and was intentionally left unexecuted.
- For why v2, MIR, or diffusion stopped as research directions, a contemporaneous decision record is missing from the sources inspected. Commit subjects and later absence cannot settle motivation. No private conversations were consulted.

## 8. Cross-slice notes

The earlier source-action and oracle-time documents already distinguish supplied
future timing from learned organization. The latter expert question also questions
strict nonempty rows and underexposure; reviewers of R1 should preserve those
qualifications rather than attribute every constraint to the formulation.

The legacy frontend was not wholly discarded: richer Mel and initial BeatThis
features flow into the audio lineage. Conversely, full timing-v3 boundary models,
MIR probe weights and gold diffusion do not appear in its research imports.
This is checked source usage, not proof that no person consulted their results.

## 9. Failed paths and unfinished work

The report covers all eight questions, with bounded sampling rather than an
exhaustive reconstruction of every June or August micro-experiment.
Several initial combined reads truncated output; pivotal tables were subsequently
read through targeted fields or saved JSON. Two shell glob searches failed on
nonexistent wildcard matches and were rerun against explicit directories.
PyArrow reported sandbox hardware-discovery warnings; the small index reads and
counts completed successfully. No permission change or bypass was attempted.

Historical live Riria/July evaluation directories, C3 chunk cache, v3 factor
checkpoint, mapper-record cache and control-teacher cache were absent at the
specific inspected paths. Git recovered committed charts/reports where available.
I did not independently recompute C3 compression, replay all 5,050 timing fits,
load every auxiliary checkpoint, inspect every gold chart visually, or establish
cross-cohort source overlap. No model-backed command or new quality test was run.

## 10. Inconsistencies and items for the human

1. **The blanket reference ban is stronger than the recorded empirical comparison.**
   D1 explicitly establishes contract authority, while A2 establishes older end-to-end
   execution. Whether the intended ban also forbids controlled legacy baselines is a
   human policy question; the evidence does not show every older component was beaten.
2. **“V3” names incompatible historical objects.** June event-group mapper, August
   timing fitter and Control V3 are distinct from September's formulation (**checked-code**
   vocab/provider/kernel citations above). Any retrospective verdict needs the component
   and revision, otherwise it attributes one experiment's failure to another.
3. **A successful ramp detector was not an audio ramp solution.** The 562-row June
   report's perfect classification is explicitly over parsed timing grids (**doc-claim**).
   D2 later says natural ramp support never passed production gates; these statements
   are compatible only when their different tasks remain visible.
4. **Timing “accepted” does not mean quality accepted.** All 42 Exp022 rows have
   `v3_accepted` while the experiment failed advancement/runtime gates (A4,
   **checked-artifact**, D2 **doc-claim**). Status vocabulary should not be read as
   independent accuracy or release approval.
5. **“Failed diffusion” has no demonstrated paradigm-level meaning in this record.**
   A6 records validation patience and null human judgments, while code supplies source
   times and LN boundaries (**checked-artifact/checked-code**). The human can decide
   whether an unrecorded inspection motivated leaving it; this report cannot infer one.
6. **Positive pre-V3 audio evidence exists without a documented disposition.** MIR's
   three-seed, 148-audio conditional result survives (A7, **checked-artifact**), as does
   timing's synthetic boundary result (A5). Neither proves generation, but calling all
   earlier audio representations unsuccessful would contradict these bounded results.
7. **Latest file names are not always the evaluated state.** Sparse mapper `report.json`
   says 44,000 steps, while `checkpoint.pt` contains 44,250 (A1, **checked-artifact**).
   The old C3 prose quotes 430.970 s but its later wide-sweep JSON at `f411d25` records
   819.045 s (A3); freeze revision/run identity before combining their numbers.
8. **State semantics were rejected before replacement semantics were measurable.**
   August's Control V3 critique names conceptual failures, while the September
   formulation leaves target-response quantities open (D1/formulation, **doc-claim**).
   This supports reconsidering the contract; it does not by itself validate a new
   player-state representation or establish which old features were causally harmful.
