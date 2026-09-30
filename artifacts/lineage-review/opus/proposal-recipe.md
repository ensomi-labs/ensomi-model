# Independent review (Opus): proposal distribution and training recipe

Written 2026-09-30 by an independent Claude reviewer (control plane, read-only on both machines). Scope: the human's key 2, "learn ordinary 2-6 star charts first". Not read: `artifacts/reports/lineage-review/`, `.sync/*/jobs/*lr-*`, `artifacts/private/`, `~/.codex/`.

Evidence grades: **doc** = a lineage document or commit says so; **code** = I read the source; **artifact** = I read raw plans, logs, ledgers or checkpoints on the mac, or computed from them. Numbers marked (mine) come from scripts I ran; they are in `/tmp/lineage-review/opus-proposal-recipe/` on the mac.

## 1. What I read and checked

- Notes entry point, `lineage-review/README.md`, the feedback index (sections 2, 3, 5).
- Docs: `r1_training_distribution.md`, `clean_joint_proposal_learning.md`, `action_segment_r1.md`, `ordinary_expert_from_scratch_zh.md`, `common_prefix_outcomes.md`, `ranked_2to6_action_reference.md` (head), grep of `native_pattern_failure_analysis_zh.md`, Codex's `relay-notes-tagged/artifacts/clean-joint-and-current-state.md`, agent notes on the 35M teacher and the vacation run.
- Code: `segment_audio_continuation/{model,bootstrap}.py`, `typed_audio_continuation/program.py` (`Recovery`), the experiment scripts `clean-joint-proposal-v1/prepare-v2.py`, `typed-contract-repair-v1/{prepare_ranked,pair_inventory}.py`.
- Mac artifacts: a sweep of every training plan, config and result under `artifacts/joint-audio/*` and `artifacts/audio-skeleton/*` (100 directories, `sweep.json`); the R1 restoration and the 35M teacher under `~/Documents/Pulsefield/research-assets/` (training logs, validation readouts, native outputs); the clean-joint source ledger (8,192 draws); the ranked corpus manifest; `artifacts/indexes/beatmap_index_4k.parquet` joined with `dataset/*/*/metadata.json`.
- Not done: no audio listening, no Lens pages opened, no generation or training.

## 2. Run table

"Passes" = examples consumed / examples available in the population it drew from. Ranked paired corpus (from 09-25): 6,923 TRAIN charts, 132,459 eight-second windows, 7.25M rows, 286.5 h (artifact, mine). R1 corpus: 10.74M post-seed onsets (doc, R1 census).

| Run (mac dir) | Params (trainable) | Init | Data | Updates x batch | Passes | Wall | Selection | Converged? | Conclusion drawn |
|---|---|---|---|---|---|---|---|---|---|
| R1 restored (`research-assets/r1-restoration-20260920-v1`) | 3.08M | fresh, then 5 staged forks | R1 TRAIN, 11,563 charts, all stars | 8,931 x 4 (base 5,958) | 0.63 nominal; 39% of onsets ever seen | 56 min CPU, 1 thread (base 29 min) | fixed stage plan | No. Constant LR 3e-4; base train NLL 1.97 -> 1.91 over its last 1.5M; val 1.77 -> 1.73 | The trusted baseline |
| 35M teacher (`research-assets/teacher35m-20260920-v2`) | 35.18M, R1 arch, width 512, no audio | fresh | same | 39,694 x 4 | 2.8 nominal; 77% unique | 23.7 h CPU | none | Val plateau: 1.61 at 20M, 1.63 at 30M | **None. I found no doc or note that reports or uses it** |
| First skeleton pilots (`audio-skeleton/20260923-v1`, `joint-audio/20260923-v1`) | small (not checked) | not checked | 48 train charts (23 ranked 2-6) | 200-1,200 | - | 17-202 s | - | - | Early timing-to-R1 pilots |
| `20260924-expanded-v1/coverage-240` | R1 + audio/timing | R1 6.75M | 585 charts (64% ranked 2-6) | 2,400 x 16 queries | - | 1,386 s | best val NLL, which was the last step | No | Joint model baseline |
| Planned/profile/lineage fits (`alias-restored`, `shared-profile`, `profile-routing`, `typed-resource-plan`) | 3.9-4.4M | chained | 585-615 charts | 1,200 each | - | 6-16 min | - | - | A dozen module variants |
| `typed-contract-repair-v1/ranked-main` | 3.95M | repair-main step 1200 | 6,923 ranked 2-6; 75% natural, 25% from 112 annotated charts | 6,000 x 2 eight-second windows | 9% of one pass; 4,641 charts touched | 47 min | none | No. H/R/row loss 30.5/8.2/6.2 -> 26.6/6.5/5.5, still falling | Largest audio fit. Ancestor of the audio encoder and H used after 09-25 |
| `common-prefix-outcomes/actor-128`, the model the human demoed (`a99519c`) | 4.60M (2.75M) | row-owned chain | 47 fixed generated prefixes, REINFORCE on star distance plus LN penalty; all 768 targets TAP-only | 128 | - | 20 min | none | - | "Calibration improved". The human then found a 4-star long jack |
| `audio-memory-joint-fit` | 7.62M (memory new) | chain + fresh memory | 768 windows | 384 x 2 | 0.6% | 51 min | final only | - | "Failed native qualification" |
| `full-row-history-views`, `broader-full-row`, `release-support`, `scoped-ln-allocation` | 0.07-2.75M trainable | chain | 256-1,024 windows | 128-512 | <=0.8% | 5-22 min | - | - | Five module/recipe decisions on 09-27/28 |
| `clean-joint-proposal-v1` (3 arms) | 4.68M each | inherited / early / fresh | 8,192 windows: 50% natural, 25% star x LN balanced, 25% from 112 annotated charts; 474k rows; 3,736 charts | 4,096 x 2 | 6.2% | 4.7 h for three arms | none | **Fresh: no.** Val row NLL 4.33 -> 2.40 -> 1.95 -> 1.84 at 0/512/2048/4096. Inherited: flat, 1.57 -> 1.61 | Failed 19/18/17 of 28 panel cases; "no initialization selected" |
| `action-segment-r1-v1/pilot-v1` | 5.9M (1.96M; new decoder, 30 tensors) | ln-fragmentation step 80 + fresh decoder | 158 four-second segments, 3,960 rows | 32 x 4 | 0.06% | 146 s | - | No | Code collapse to one state; LN share .25-.30 whatever the request. Reported to the human |
| `continuous-segment-context-v1` | same family | chain | 1,024 examples from the clean-joint ledger | 256 planned, died at 112 (MPS) | - | - | judged at 64 | No | Next design step |
| `ordinary-scratch-v1/capacity-fit-v1` | 4.78M | fresh | **4 songs, 16 four-second units (64 s of chart), 3.77-4.00 stars** | 512 x 1 (32 epochs) | - | 236 s | - | **No, even on its own 16 units.** H NLL/s 14.9 -> 13.0 over the last 128 steps; gradient norm rising 49 -> 173 | "Ordinary expert failed", which led to the response-architecture proposal |

All rows are artifact-checked (plans, results, loss logs), except the planned/profile block, which comes only from `result.json`/`config.json` in the sweep.

## 3. Findings

**F1. No audio model in the lineage was trained enough to judge its architecture.** (artifact) The largest audio fit consumed 9% of one pass over the ranked corpus. The clean-joint comparison, the basis of the late "initialization" and "architecture" discussion, consumed 6.2%. Its fresh arm was still improving steeply when stopped: validation row NLL fell 0.11 between 2,048 and 4,096 updates, and it ended at 1.84 against the inherited arm's 1.61. Every run after 09-25 was 32-512 updates, except clean-joint. The longest per-model training in the lineage was about 1.6 h. The 35M teacher run just before the lineage took 23.7 h, so long runs were not ruled out by policy.

**F2. Much of the small budget was self-inflicted by throughput.** (artifact, mine) The joint audio models trained at about 84 rows/s per arm: 474k rows per arm, 16,957 s for three arms. R1 trained at about 2,550 onsets/s on one CPU thread, 30x faster, with a similar parameter count. The difference follows from recomputing full-song audio with gradients on every update at batch 2 (`clean_joint_proposal_learning.md`: "current-weight full-song coarse audio"; code in `common.py`). In the action-segment pilot, memory footprint grew linearly to 16.7 GB in 32 updates, against a 17 GB cap. These engineering costs set the pilot sizes, and the pilot sizes set what could be concluded.

**F3. The training measure was not the ordinary distribution, even after the corpus was.** (artifact) From 09-25 the corpus was ranked 2-6 only: 6,923 charts, with status from official metadata; the admission code and `pair_inventory.py` were read. But every joint recipe drew 25% of its windows from the 112 human-annotated style charts, chosen for salient jack, stream, trill, tech and LN coordination. In clean-joint that branch was 2,050 draws, up to 29 per chart, and 30% of all training rows. Another 25% came from uniform star x LN cells, where LN>=.5 charts are one cell in three but 5% of the corpus. The 5-6 star share was 18% of draws, against about 8% under natural sampling. Only 45% of rows were natural draws. Controls were known on 80-100% of draws, and half the star labels were whole-chart and half local 16/32/64 s. The lineage never ran a natural-only recipe.

**F4. R1's own data is only two-thirds ordinary, and R1 is the ancestor of every inherited arm.** (artifact, mine; joined index plus set metadata) R1 TRAIN: 64.6% of charts are ranked or approved at 2-6 stars (67.7% of expected draws). 21.7% of charts are ranked below 2 stars, and 11% of expected draws are loved maps (loved maps at 4-6 stars: 759 charts). The ranked 3.5-4.5 band is 18% of charts. This fits the human's "not the ordinary distribution" only in part: two-thirds is ordinary. The larger distortions are the lineage's sampling branches (F3) and the objectives layered on top (F6).

**F5. No model was ever trained only on one ordinary star band long enough to fit it and then judged on its own output.** (artifact) The only single-band attempt is `ordinary-scratch`: 4 songs, 64 s of chart, 512 updates. It did not even fit its training units; the H NLL was still falling. It was then asked to generate whole songs from BOS on the same four songs, and most of each song was never trained on. The script's own `purpose` field says "capacity diagnostic, not generalization", and the Chinese doc is candid about this. The data for the real experiment exists: 1,972 ranked 3.5-4.5 charts, 2.39M rows (artifact, mine).

**F6. The model the human demoed was optimized toward a scalar that long jacks can satisfy.** (doc + commit check) `a99519c` is the common-prefix outcome actor. It had 128 REINFORCE updates on 47 fixed prefixes, with cost `max(|D-D_req|-.25,0)^2` plus `100*max(rho-.03,0)^2`, and every training target was TAP-only (doc, `common_prefix_outcomes.md`). The feedback index records that the star proxy scored a 16 s single-column run at about 4.1. The long-jack regression at 4 stars (problem A) is what this objective permits. It is a recipe failure, not evidence about module layout.

**F7. Hard support settings changed between runs, which confounds comparisons.** (code, artifact) `Recovery(hh, rh, hr)` was 60/25/21 in clean-joint, 60/50/40 in the memory fit, and **20/1/1 in the fresh ordinary expert** (`bootstrap.py`: `Recovery(config.minimum_action_gap_ms,1,1)`, `minimum_action_gap_ms=20`, `bounded_head=False`). The fresh expert's support therefore allowed 1 ms LNs and 1 ms release-to-head gaps. Its H hazard had its bounded-history correction removed, and its H was under-fitted (F5). Its reported failures follow from those three facts: 5-6 ms LNs, 14 H pairs within 10 ms in 6 s, and 155 of 191 short tails ending at the next H (doc). They say little about the row model. This supports the timing-representation reading (lead b in the README, slice 03's question). Caveat: the relaxation followed the human's "no hard-coded rules" instruction (V60-V67), so it was deliberate. But nothing else in the model could learn the ranked minima from 64 s of data.

**F8. More teacher-forced fitting does not by itself make native output more ordinary. This is evidence against "it is only budget".** (artifact, mine) I compared 16 fixed validation continuations (8 songs x 2 seeds; real times supplied, R1-style seeded continuation) across R1 and the 35M teacher's checkpoints. Metrics are over the generated suffix; "jack" = share of consecutive head rows sharing a column; "max4s" = most same-column attacks in 4 s; "s40" = LNs of 40 ms or less.

| Model | Val NLL/onset | Stars | Jack | Max4s | LN share | s40 |
|---|---:|---:|---:|---:|---:|---:|
| Source charts | - | 3.85 | .159 | 19.3 | .189 | 0.0 |
| R1 base 4.5M | 1.77 | 4.16 | .114 | 17.3 | .447 | 5.2 |
| R1 response 6.75M (released) | 1.74 | 4.20 | .180 | 20.3 | .180 | 6.5 |
| Teacher @10M | 1.64 | 4.70 | .326 | 21.9 | .229 | 1.4 |
| Teacher @20M | 1.61 | 4.07 | .134 | 19.3 | .384 | 1.7 |
| Teacher @30M | 1.63 | 4.94 | .320 | 24.2 | .295 | 2.3 |

Held-out NLL improved by 0.1 nats per onset. Native difficulty and jack share did not move toward the source and varied strongly from checkpoint to checkpoint; song val-14 went from 5.77 stars to 9.1 at 30M. All models inflate stars above the source even with real times given. The teacher has no native-trained residuals, so the fair comparison is with R1 base: better on LN, worse on jacks and difficulty. Limits: 16 cases, one sample per seed, crude statistics, no Lens reading. This is the strongest piece of evidence I found that the rollout problem (the `rollout` node) is real and not fixed by scale alone. It concerns the continuation half, which gets real times, not audio.

**F9. Some small-pilot conclusions are legitimately structural.** (code, doc) A tiny pilot can show a missing path, and several did:
- The segment decoder deletes `row_consequence` and ignores its `local`/`timing` arguments (code, `segment_audio_continuation/model.py` lines 144-173). The 4-window zero-change ablation is valid at any training size.
- A one-state categorical prior cannot carry varying inputs; that is true by construction.
- Count-only updates cannot change layout given counts. The measured KL of 5.8e-14 matches the algebra.
- H-only star floors: on the inherited and early H plans for Zenithfall D2, no row realization can reach the requested band. The trained H fixes a difficulty floor.
- Outcome targets were all TAP, which disqualifies LN claims.

These are sound, and they are the lineage's most durable results. What they cannot support is the step from "this path is missing" or "this checkpoint fails" to "this architecture cannot learn ordinary charts".

**F10. The evaluation panel was mostly control extremes, not ordinary charts at their natural difficulty.** (artifact) The fixed 28-case panel has 8 D2/D6 cases, 8 LN-amount cases up to .838, 6 style requests and a live switch, on about 6 songs. The Zenithfall source is 5.87 stars and was asked for 2 stars. The 8-case interim panel is 6/8 extremes. The recipe was never measured on ordinary output at a natural request against held-out ranked charts of that band. Fresh, the least trained arm, failed fewest (17/28): the long chain of inherited fine-tunes bought nothing on this panel.

## 4. The lead "late architecture conclusions rest on very small pilots"

Numbers in the README, checked against logs:
- Clean-joint: 4.68M parameters, 4,096 updates, 8,192 windows, 3,736 charts, 16,957 s. Confirmed (artifact). The README's 3,736 is correct for distinct charts; `preparation-result.json` says 3,758 identities, which includes validation.
- Ordinary expert: 512 updates, 235.73 s, sixteen 4-s units from four songs. Confirmed.
- 155 of 191 short tails ended at the next head, and pairs 10 ms apart or less appeared in native H: doc, consistent with F7. Not recounted.

Verdict: **mostly confirmed, with a correction to its framing.** The lineage documents themselves mostly hedge; `native_pattern_failure_analysis_zh.md` lines 7, 41 and 1396 say capacity and recipe must be tested separately. The problem is in the decisions. After each 32-512-update pilot the next step was a new module (segment mixture, continuous context, fresh expert, response redesign). Longer training of an existing recipe was never tried. One explicit stopping rule rules it out ("more source learning only improves NLL ... not enough to continue with the same recipe", line 1464), on the basis of 128-512-update fits. The strongest misreading is outside the lineage. The feedback index's falsifier for key 2 says a from-scratch ordinary expert "on the corpus also stays red (the 09-28 attempt did, which points at architecture)". That attempt was not trained on the corpus. It should not count against key 2.

Evidence against the lead: F9 (structural findings valid at any scale) and F8 (scale and budget on the MLE objective did not fix native behaviour in the one place scale was tested).

## 5. Judgment

Question: do the failures speak about architecture, or only about data, budget and recipe?

For "only data, budget and recipe":
- F1: no audio model was trained past about 9% of one pass.
- F5: fresh models never fitted even their own data.
- F3: training measures were less than half natural, and 25% came from 112 style charts.
- F6: the demoed model was pushed toward a scalar that jacks satisfy.
- F7: support settings changed between runs.
- F10: panels judged control extremes.
- Scale was never a variable inside the lineage.

For "architecture and objective matter too":
- F8: 11x parameters and 4.4x exposures bought 0.1 nats held-out but no systematic move of native output toward source. Autoregressive MLE sampling drifts toward harder and jackier output even with real times.
- F9: specific missing information paths and a millisecond timing representation without refractory structure are real defects.
- The R1 residuals trained on native trajectories changed LN behaviour (LN share .447 -> .180) where base training did not, so objectives on native states matter.

My call:
1. The lineage's evidence cannot discriminate between architectures. Confidence about 85%. Nothing it trained was in the regime where that question can be asked.
2. Key 2 ("ordinary first") is the right data and recipe decision, and it was never actually executed. Confidence about 75%.
3. It is not sufficient on its own. F8 suggests that even a well-fitted model will drift on its own histories unless rollout is addressed, and that needs its own test. Confidence about 60%; the sample is small.

The timing half (H from audio) is where the grossest late failures originate (F7, star floors), and it is the least trained component of all.

## 6. Overlooked or never questioned

1. Training budget and learning curves. No fresh model was trained to a plateau; no run doubled budget to see what changes; no scale sweep. The completed 35M teacher was never evaluated or mentioned after 09-21. Its checkpoints and native readouts are sitting on disk.
2. The sampling measure. The 25% annotation branch and 25% balanced branch were carried into every recipe after 09-25 and never ablated against natural-only.
3. Throughput as a design constraint (F2). Batch 2, per-update full-song audio gradients and a memory leak shaped every pilot. Nobody asked what a 30x cheaper recipe (frozen or cached audio features, larger batches) would allow.
4. Warm-start palimpsest. The "inherited" checkpoint is the end of about fifteen short fine-tunes with changing objectives, modules and support settings. Its validation row NLL did not improve in 4,096 further updates. The chain was never reset to test what it retained.
5. Held-out ordinary evaluation. Native output was judged on about 6 songs, mostly at extreme requests, rather than on many held-out ranked charts of one band at their own difficulty.

## 7. Worth keeping

- The ranked 2-6 paired corpus: `typed-contract-repair-v1/ranked-corpus` (6,923 train charts, 2,573 groups, Mel cache, grouped split), plus the relation census in `ranked_2to6_action_reference.md` (HH, HR and RH minima by band).
- The deterministic, pre-published draw ledgers and chunked supervisor. They make every fit auditable, as this review shows.
- The structural findings of F9.
- R1's staged artifacts and the 35M teacher checkpoints, validation readouts and native outputs. The teacher is a ready-made scale data point for the continuation half.
- The 3.5-4.5 star split by song and audio component (1,602 / 361 / 10) from `ordinary-scratch-v1/scout-v3`.

## 8. Inconsistencies and questions only the human can settle

- The key-2 falsifier in the feedback index (section 5) cites the 09-28 fresh expert as trained "on the corpus". It was not. Strike it or reword it?
- What "ordinary" means: a star band only (for example ranked 3.5-4.5), or also excluding loved maps, LN-majority and tech charts? Should the 112 annotated style charts be excluded from an ordinary recipe?
- Does "ordinary first" apply first to the row half given real times (R1, no audio), or to the whole audio-to-chart system? The evidence says the two halves fail differently.
- "No hard-coded rules" (V60-V67) versus the ranked census minima (HH 37 ms, HR 19 ms, RH 25 ms). Should these be treated as support, as learned targets, or as evaluation only? The fresh expert's 1 ms settings are what the literal instruction produced.
- The 09-23 question about whether the 35M run improved the base (V08): the answer is not in any document. What was the human told?

## 9. Not verified, and how to follow up

- The native comparison in F8 is mine: crude statistics, 16 cases, one sample each. To firm it up, rerun with several seeds per case, on held-out 3.5-4.5 star songs, with Lens reading. Script: mac `/tmp/lineage-review/opus-proposal-recipe/native_cmp2.py` (run with `.venv/bin/python` from `~/ensomi/ensomi-model`); the checkpoints are `research-assets/teacher35m-20260920-v2/run/teacher/segment-00006/checkpoint.pt` and the R1 release.
- The chain of about fifteen fine-tunes behind the "inherited" arm, from R1 to `scoped-ln-allocation-v1/fit-v2/progress.pt`, is inferred from configs I spot-read, not traced link by link. Follow `initial`/`checkpoint`/`parent` fields in each `config.json` listed in `sweep.json`.
- The per-chart LN share by status in R1 TRAIN, to see whether loved maps drive R1's LN behaviour: not computed. It needs a parse of 11.5k `.osu` files (about 2 min, `parse_osu_file`).
- The throughput cause in F2 is inferred from doc and config (full audio gradients, batch 2, MPS, three arms interleaved); not profiled.
- The 28-case panel's per-case results at 4,096 were counted, not read case by case.
- The decisive experiment is cheap and has never been run: train the R1 architecture (no audio, real times) from scratch on the 1,602 train charts at 3.5-4.5 stars (about 2M rows) with natural sampling to a validation plateau. Then compare its native continuations with R1 and the teacher on the 361 held-out charts using the same metrics plus Lens. At R1's CPU throughput that is hours, not days. It tests key 2 for the row half directly, and F8 predicts whether fitting alone is enough.
