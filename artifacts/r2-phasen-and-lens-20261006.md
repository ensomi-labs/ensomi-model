# Phase-N values and the Lens data for phase C (Astra, 2026-10-06)

Two Astra jobs ran on the mac at the human's request ([private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-7), amended by [prompt-8](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-8)). Both used the fast tier at effort xhigh, under 2 h, from 12:58 to 14:44 UTC. Reports are in the code checkout, git-ignored and mirrored here: `artifacts/r2-phaseN-tune-20261006/report.md` and `artifacts/lens-phasec-20261006/report.md`. No tracked file was changed and no full training ran. The main thread read both reports and spot-checked the tables quoted below against them; it did not re-run anything.

<a id="s-phasen-tune"></a>**Phase-N values (job `20261006-125849-r2-phasen-tune`).**
- **Divisor:** n_bar = 898.916, relative SE 0.65% over 2,000 draws; n_bar_ln = n_bar_star = 0; key verified.
- **Device:** CPU with 4 threads, about 2,029 head decisions/s over the peak pilots. MPS ran at about 986/s with 1 thread, roughly half.
- **Learning rate:** pilots of 512 updates (458k heads, two weight seeds) gave mean natural NLL of:

  | Peak lr | Mean NLL |
  | --- | --- |
  | 1e-4 | 3.183 |
  | 3e-4 | 3.076 |
  | 1e-3 | 3.032 |
  | 3e-3 | 2.995 |

  1e-4 is worse than 3e-4. The three higher rates are tied under the two-SE rule; the trend favours higher, but that is post-hoc. Warm-ups of 5k, 20k and 50k are tied.
- **Proposed config** (`ce_v2_n.proposed.json`, accepted by `check_config`): lr 3e-4 (the lowest of the tied rates, by an operational rule), warm-up 20k, cosine to 3e-5, 64M exposures, checkpoints every 4M, `g3c_exposures` [].
  - The 64M budget is extrapolated from the v1 curve, whose minimum was at 61.4M.
  - The run would take about 13.6 h, of which about 3.9 h is evaluation; one full evaluation, mostly calibration, takes about 29 min.
  - The pilots stopped at 0.7% of the proposed budget, so the peak lr is not decided at full budget.
- **Proposed follow-up, not run:** 3e-4 against 3e-3 at 4M heads, with fresh seeds 173 and 174 and the 64M schedule clock. That is about 2.2 h of raw training.

<a id="s-lens-data"></a>**Lens annotation data (job `20261006-125845-lens-phasec`).**
- **The "latest three batches"** are read as the three newest completed deliveries of today's campaign, `labeler-025` to `labeler-027`: 75 4K charts with one section each (median 10 s, 76 head rows) and 360 supported labels. Gold is the current High-confidence human layer: 171 labels on 58 sections.
- **Join to the R2 cache:** 280 labels on 74 charts (66 fit_train, 8 fit_dev).
- **The larger published machine layers** (v2.1, v2.1 repair, v3) hold 6,801 labels on 1,434 sections and 656 charts. With High gold they join R2 on 5,400 labels and 535 charts.
- **Label form:** five independent concepts (Jack, Stream, Trill, Tech, LN coordination), each absent, present-supporting or present-prominent.
- **Coverage is sparse:** labelled time is a median 2-12% of a chart, at 1-3 sections per chart about 27-50 s apart. There are no dense within-chart trajectories.
  - First-order transition gains are at most 0.1 nats, and they are not order-sensitive.
  - The data do not identify HMM transitions or dwell durations.
- **Agent quality:** the latest batches share no section with gold, so current agent accuracy is not validated. Historical kappa against High gold was 1.00 for Trill, 0.93 for Jack, 0.79 for LN coordination, 0.73 for Stream and 0.48 for Tech.
- **Rare levels:** in 2-6★, the latest batches have no prominent Trill and only 2 prominent Tech cells.

<a id="a-phasec-method"></a>**Phase-C method, Astra's proposal (not decided).**
- **Properties** (LN share, difficulty: exact metrics):
  - frozen-base phase C first, with the approved star proxy;
  - an LN count-feedback decoding baseline kept as a comparison, under short-hold and feasibility guards. In v1, count feedback reached MAE 0.0007 but gave 2.8% holds ≤ 60 ms against 0 in the source;
  - an own-sample LN term stays the human's call (Q-I).
  - The report adds that a 40 ms floor alone does not enforce the ≤ 60 ms guard.
- **Styles:**
  1. Train a section scorer (per concept, three levels) and validate it on gold.
  2. Use it to test whether R2 already samples the requested styles (candidate support).
  3. Then try best-of-N or reranking inside the scope.
  4. Then scoped style conditioning for the concepts and levels with enough data.
- **Defer** HMM/HSMM and CRF transitions (no trajectory data; a persistence prior would impose a hand-chosen aesthetic) and DPO (Lens labels are not same-context preferences).
- **Search** (SMC or beam) must branch only on decisions in V; tails after b come from the base law.
- **Operating point:** the same frozen evaluator for every arm; the least-steering Pareto point that meets the adherence requirement.

This refines [h-decoding-style](decoding-stage.md#h-decoding-style). Decoding-time selection is supported as best-of-N or reranking with a validated scorer. A learned HMM over style states is not supported by the present data.

**For the human:**
- the phase-N peak lr: accept 3e-4, use 1e-3 (the v1 precedent, stable to 61M), or run the 4M follow-up first;
- the budget (64M, about 13.6 h);
- memory (plan v5 decision 11);
- the Lens training population: the latest three batches only, or the published layers plus High gold;
- frozen first for phase C.

<a id="d-phasen-memory"></a>**Decisions (human, 2026-10-06, [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-9)).**
1. **Phase-N config accepted** as proposed. It is now in the tracked `configs/ce_v2_n.json` on the working tree, uncommitted: lr 3e-4, warm-up 20k, cosine to 3e-5, 64M exposures, checkpoints every 4M, `g3c_exposures` [], n_bar 898.916 with its key.
2. **Long-range memory off.** `memory='none'` in all three v2 configs, so phases N and C must match; this settles plan v5 decision 11. The landmark modules are still built (65,920 parameters) but unused under `none`: the parameter count stays 2,404,724 (mac job `20261006-152557`).

`check_config` accepts the config with memory off.

<a id="c-style-pool"></a>**Correction (human): the style training pool.** About 4,000-5,000 labelled sections are available for style training, possibly from three annotation provenances; the human's own reading is unsure. The Lens job above counted only 75 sections from today's 1,000-chart campaign as "the latest three batches", which was a misreading. Its method conclusions that rest on that count (sparse trajectories, "narrow feasibility study") need re-checking against the real pool. The inventory is part of the next Astra job.

<a id="q-style-module"></a>**Question (human): what architecture should the style scorer have?** It must work with the phase-N generator during generation, not just classify finished sections, because classification differs from generation. Should it be based on phase N? Astra job `20261006-152705-r2-style-module` (default tier, effort xhigh, under 2 h) inventories the pool and compares the options:
- a prefix discriminator on generator states (FUDGE-like);
- a class-conditional generative discriminator (GeDi-like);
- a section energy or reward on the frozen phase-N trunk;
- phase-C style adapters;
- combinations.
It recommends one design with an exact interface. Output: `artifacts/r2-style-module-20261006/` in the code checkout.

Whether to start the phase-N run now was raised as a "maybe"; the main thread asks the human to confirm.

<a id="r-phasen-run"></a>**Phase-N run launched (2026-10-06 15:36 UTC, human's choice "Commit, then start", [private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#answer-1)).**
- **Code:** R2 v2 committed on `r2/train` as `3954031`; not pushed.
  - Two tests in `tests/r2/test_phases.py` were updated for the accepted config. The suite gives 256 passed, 4 skipped (job `20261006-153104-r2-phase-tests-3`).
  - A 16-window pilot with memory off ran at about 3,400 decisions/s.
- **Run:**
  - id `r2-phaseN-20261006`, job `20261006-153613-r2-ce-r2-phaseN-20261006`;
  - directory `artifacts/r2-runs/r2-phaseN-20261006/`;
  - frozen code sha256 `06cb051d…`, git head `3954031`;
  - config `ce_v2_n.json`: memory none, lr 3e-4, 64M exposures, CPU with 4 threads.
- **Health at 200 s:** 594k exposures, about 3,300 decisions/s, RSS about 1.0 GiB, gradient norm about 12, all finite.
- **Outputs to expect:** a checkpoint every 4M exposures; selection by `select.py` after the run.

<a id="s-style-module"></a>**Result: style-module job (Astra `20261006-152705-r2-style-module`, default tier, effort xhigh, 15:27 to 16:38 UTC, exit 0).** The report is in the code checkout, git-ignored and mirrored here: `artifacts/r2-style-module-20261006/report.md`, with `inventory.md`, `synthesis.md` and `feasibility.md`. No tracked file changed.
- **Pool ([c-style-pool](#c-style-pool)):** 4,717 sections on 1,807 chart versions carry at least one explicit level. After admission (current Foundation, human precedence, three conflicting cells masked) there are 4,714 sections with 15,948 concept cells, of which 2,523 sections have all five. The exact-byte R2 join covers 2,995 fit_train and 273 fit_dev sections. Human labels cover 230 sections. The median section lasts 5.5 s. This matches the human's 4,000-5,000. The job reads "three provenances" as three method fingerprints (v2.1 initial, v2.1 repair, v3) or as three views (v2.1, v3, today's campaign) and confirms neither.
- **Recommendation (proposed, untested):**
  - Five per-concept FiLM adapters (28 → 128 → 128 → 256, zero-initialised output, one shared LayerNorm; 266,496 parameters). They are added to the kept property FiLM at the existing hand and release-query sites, gated by the rule-L read bit. They train in a frozen-base phase C by source-decision CE.
  - A separate prefix critic on frozen phase-N features (input 1,366, 179,643 parameters; 2,850,863 in total with the adapters). The job's own review found that the first version (1,114 inputs) could not see EOS gap releases, because EOS never enters the TCN; a 252-channel final-decision block repairs that. It forecasts each concept's finished-section level per tentative row action or release candidate, and is trained by masked label CE on real prefixes.
  - FUDGE-form guidance from the critic stays off until it is calibrated on generated histories and checked against independent judgments. GeDi, section energy with SMC, and HMM transitions are deferred.
- **Probe (preregistered, exploratory):** on the 458k-exposure memory-on pilot checkpoint, 389 train and 90 dev machine-labelled sections, state + features scored a macro F1 of 0.577 against 0.587 for features alone. The gap is −0.010, 95% interval −0.060 to +0.057, against the registered +0.03, so not met. State alone scored 0.580 (majority 0.254).

<a id="a-style-module"></a>**Agent reading (main thread, checked against the report and code, not re-run).**
- **Rule L.** The read bit is computed once per decision and concept, before the action, and shared by all actions, both orientations and every candidate. A concept that cannot be read contributes no frame, loss or critic potential. This matches `locality.reads`.
- **Identity gate.** An empty request returns z unchanged before any adapter MLP runs. Mixed batches use `where(read, …)`.
- **Frozen phase C.** Only `film.*` trains. The critic is fitted separately and sends no gradient into the generator.
- **No self-evaluation.** The critic is a guide and a diagnostic, never the sole judge. Adherence goes to fresh blinded human judgments or a scorer that shares no parameters.
- **Parameter arithmetic** recomputed: 5 × 53,248 + 256 = 266,496; 2·1,366 + 1,367·128 + 129·15 = 179,643.
- **What it does not show.** Whether frozen phase-N features carry the distinctions, whether source CE changes generated organisation, and whether the critic transfers to sampled alternatives are all untested. The probe is weak evidence on a superseded checkpoint.
- **Before any style training (for the human):** the admission scenario; which concepts and levels come first; a fresh human evaluation set; phase-C and critic budgets, label balancing and selection; whether the first use is live or offline. The style module is not decided.

<a id="w-architecture-page"></a>**Architecture page finalised for Lele Liu (2026-10-06, local HTML only, [r2-architecture-20261006.html](r2-architecture-20261006.html)).**
- Added a goal-and-status block (confirmed, under test, open).
- Drew the landmark read as switched off: 65,920 parameters built but unused; total still 2,404,724; 2,194,548 trained in phase N.
- Recorded the phase-N settings and marked phase C as open.
- Added §5 conditioning capacity for seven kinds and §10 the style module (proposed).
- Kept internal process out of the page.

<a id="o-phasen-restart"></a>**Observation: the phase-N run restarted once (main thread, from `events.jsonl` and the resources log).**
- At 15:45:58 UTC, 1,338,023 exposures, RSS rose from 1.08 GB to 3.78 GB between two 10 s samples. The guard ("RSS grew above 2.0 GiB since the last checkpoint") wrote `ckpt-0001338023-safe.pt` and the trainer exited with code 4.
- The launcher resumed at 15:46 (restart 1 of at most 5 in 6 h) and ran a full evaluation (`evals.jsonl`, 1,430 s).
- Training continued from 16:10 at about 1.0 GB RSS. It was a single jump, not gradual growth. The cause is not identified.
- Each further restart costs one evaluation.
- The handoff's "healthy at 1.34M" was written just after this event.
- Separately, the sync session `meta-ensomi-model` had been re-sending two 45 MB and 15 MB logs of the v1 run every cycle. It is fixed in workspace commit `d773084`. Per-step `logs/train-*.jsonl` of R2 runs no longer mirror; read them on the mac.

<a id="o-phasen-rss-cause"></a>**Observation, 2026-10-07 (main thread; supersedes "the cause is not identified" above): the RSS jumps are per-window peaks, and the run stopped on a supervisor bug.**
- **Stop:** eight resource stops between 15:45 and 23:58 UTC on 2026-10-06; the supervisor quit at 41,380,351 of 64M exposures (`supervisor_stop`, `restart_limit`). The restart count never cleared because each resource stop points `checkpoints/latest.json` at its `-safe` checkpoint, which `launch.latest_checkpoint` skipped; restarts 6 and 7 still counted five in the window although regular checkpoints 20M to 40M had been written in between.
- **Cause of the jumps:** no leak. Per segment, RSS sits near 1 GiB and spikes by 1-3.7 GiB for one 2,000-exposure log interval, then falls back; every stop fired on such a spike. The temporal encoder runs with gradients over the whole chart prefix up to the window's end, so a window late in a long chart keeps activations in proportion to the prefix: peak rise 0.70, 1.41, 2.85 and 4.76 GiB at 5k, 10k, 20k and 43,661 positions (the longest fit_train chart). About 4 windows per 1M exposures have a prefix over 10k (20k replayed draws). Probe job `20261007-034504-r2-rss-window-probe` (bings-mac), script `~/ensomi/.sync/cp/scratch/r2-rss-window-probe.py`.
- **Fixes (human's instruction, [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#prompt-1)):** supervisor reads regular checkpoints from the directory, and `rss_growth_limit_gib` 2 → 6 in the phase N and C configs (`ensomi-model` `7885133`); per-block activation checkpointing of the TCN, same values and gradients, peak rise 1.74 GiB at 43,661 (`3520b71`, [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#prompt-3)). Neither is pushed.
- **Run resumed** 2026-10-07 03:41 UTC (job `20261007-034113-r2-phaseN-resume`) from `ckpt-0041380351-safe.pt`, with its frozen `launch.py` replaced by the fixed one and its config limit at 6 GiB (event `operator_patch` in `events.jsonl`). The run keeps its frozen trainer, so it does not use the activation checkpointing. At 04:37 UTC: 47.0M exposures, about 2,000 decisions/s, RSS peaks up to 4.1 GiB.

<a id="d-decision-unit-kept"></a>**Decision (human, 2026-10-06, [private, local](private/human-inputs/da66cd9e-81c2-4ad1-a989-4fe6ef7c81fe.md#prompt-4)): the decision unit stays.**
- [q-decision-unit-rethink](r2-v2-stage0-state.md#q-decision-unit-rethink) is closed.
- A decision that straddles a scope boundary is a boundary effect. It touches at most one head row per boundary, which the human judges negligible.
- The decision unit is not redefined, and rule L stays as built.
- The open item was removed from the architecture page (§6 and the status block).

<a id="d-page-published"></a>**Decision (human, 2026-10-06, [private, local](private/human-inputs/da66cd9e-81c2-4ad1-a989-4fe6ef7c81fe.md#prompt-7)): the architecture page is published on claude.ai as it stands.** The earlier local-only rule for this page is lifted. Link: https://claude.ai/artifact/Tsmjq6BYdrwMevfVjJcStm. It is private until shared from its Share menu. The local file stays the source; republishing it keeps the link.

<a id="r-page-hosted"></a>**Record, 2026-10-06.** The claude.ai copy could not be made public from here. The human published the page on their own site (Cloudflare) at https://research.sed-i.org/r2-architecture; that is the link given to Lele Liu. The local file stays the source; updating the site is the human's step.

<a id="o-phasen-finished"></a>**Observation, 2026-10-07: the phase-N run finished (main thread, from `run.json`, `events.jsonl`, `evals.jsonl` on bings-mac).**
Ended 08:18 UTC at 64,000,573 exposures, exit 0 (`supervisor_done`), with no stop since the 03:41 resume. 8 restarts in
all, no NaN events, 1,820 decisions/s over the whole wall time. fit_dev natural-manifest NLL per decision by full-eval
checkpoint: 2.130 (24M), 2.089 (32M), 2.071 (40M), **2.070 (48M)**, 2.075 (56M), 2.084 (64M), flat from 40M with a slight
rise at the end.

<a id="o-phasen-no-selection"></a>**Observation, 2026-10-07: `select.py` selects no phase-N checkpoint; every candidate fails guard (iv)** (ens job
`20261007-082126-r2-phaseN-select`, `select.py` at `7985cf9`; phase-N guards are legal, (i), (iii), (iv)).
- Guard (iv) has two parts: holds ≤ 60 ms at most 0.5% of holds, and releases 1-40 ms before another head at most the
  source panel's rate (0.42%). Values at the full-eval checkpoints:

  | Checkpoint | holds ≤ 60 ms | releases 1-40 ms before a head | (i) | (iii) |
  | --- | ---: | ---: | --- | --- |
  | 16M | 2.7% | 1.2% | fail | pass |
  | 24M | 1.4% | 1.3% | pass | pass |
  | 32M | 4.4% | 5.2% | pass | pass |
  | 40M | 3.0% | 1.5% | pass | pass |
  | 48M | 1.9% | 2.5% | pass | pass |
  | 56M | 2.0% | 1.2% | pass | pass |
  | 64M | 1.4% | 1.6% | pass | pass |

  Both parts fail at every checkpoint, by 3-9× on short holds and 2-12× on near-head releases. No downward trend
  after 24M. Real charts have no holds of 60 ms or less ([decoding-stage](decoding-stage.md#p-min-hold-mask)). The
  same defect class appeared at 1-3% under R2 v1's guidance and count feedback
  ([r2-analysis-fable-judgment](r2-analysis-fable-judgment.md)); here it appears with no request at all.
- **Diagnostic, not the rule:** with guard (iv) dropped, the rule would select `ckpt-0048000198.pt` (3-checkpoint
  mean NLL 2.0722, the minimum; ens job `20261007-082229-r2-phaseN-select-noiv`, scratch script
  `~/ensomi/.sync/cp/scratch/r2-phaseN-select-noiv.py`). The C0 measurements used `ckpt-0052000205.pt`, which had no
  full evaluation.
- This is the redirect condition in `RESEARCH.md` ("a selected phase-N checkpoint that fails its natural guards").
  Phase C's frozen base is therefore undecided. The choice is the human's: decode-time masks on a chosen
  checkpoint ([p-min-hold-mask](decoding-stage.md#p-min-hold-mask)), a minimum hold in the action contract and a
  retrained phase N, or a relaxed guard.

<a id="d-phasen-base-mask"></a>**Decision (human, 2026-10-07, answer to the agent's question, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#answer-1)): mask at decoding.**
Take `ckpt-0048000198.pt` (lowest NLL, passes guards (i) and (iii)) and add a 60 ms minimum-hold mask to sampling,
on both the pointer candidates and the row codes. Then measure guard (iv) again on free runs. Agent reading: 48M
is the provisional phase-C base only if the mask brings guard (iv) within its limits. If it does not, the next step
is to propose retraining with the minimum hold in the action contract, not to launch it unasked. Delegated to
Astra job `20261007-082909-r2-minhold-alloc` (part A; outputs `artifacts/r2-minhold-20261007/` on bings-mac;
brief in `~/ensomi/.sync/cp/jobs/20261007-082909-r2-minhold-alloc/brief.md`). The brief does not touch the 1-40 ms
release rate beyond measuring it, overall and by star band.

<a id="q-ln-length"></a>**Question (human, 2026-10-07, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#prompt-2)): what causes phase N's short holds, and how is it removed in training?**
The human wants the guard (iv) defects rare by the model itself, without decoding-time tuning. Three questions:
does the release representation cause them; does R2 repeat the R1 mechanism recorded in
[s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state); does the lack of training on the model's own history
over whole songs (exposure bias) cause or amplify them. Agent reading: the decode mask
([d-phasen-base-mask](#d-phasen-base-mask)) continues as a measurement and fallback, not as the answer. Delegated
2026-10-07 about 08:50 UTC to a fresh Opus subagent (Claude, control plane). It is read-only on tracked files while Astra
edits the working tree, imports the run's frozen code `artifacts/r2-runs/r2-phaseN-20261006/code`, puts its scripts in
`~/ensomi/.sync/cp/scratch/r2-ln-length/` and its outputs in `artifacts/r2-ln-length-20261007/` on bings-mac, and does no
training. The brief asks for: the generation path of each defect; source rates and representation (candidates, snapping);
the model's teacher-forced mass at source states against its free-run rate on its own history, and that rate along
the song; R2 v1 against v2; the R1 comparison; ranked training-side remedies with costs.

<a id="h-keep-underlearned"></a>**Hypothesis (human, 2026-10-07, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#prompt-3)): the keep decision is poorly learned, so LNs end early.**
On a held lane, code 0 (keep) may be under-learned against the release codes, which would bias the model toward
releasing early. Sent to the running subagent of [q-ln-length](#q-ln-length) as a test with four parts:
1. model against source release hazard by elapsed hold length (ms and beats), gap, density and star band, at
   teacher-forced states;
2. the same hazard on the model's own free-run history;
3. hold-length survival curves;
4. from the code: whether the row decision sees the hold's age, and how keep decisions are weighted in the loss.

<a id="o-minhold-mask"></a>**Observation, 2026-10-07: the 60 ms minimum-hold mask removes generated short holds, but 48M still fails guard (iv) on near-head releases** (Astra job `20261007-082909-r2-minhold-alloc`, part A; code committed as `ensomi-model` `01aacba`; report `artifacts/r2-minhold-20261007/report.md` on bings-mac; final message `~/ensomi/.sync/mac/jobs/20261007-082909-r2-minhold-alloc/last.md`; read by the main thread).
- With the mask off, the evaluator reproduced the numbers in `evals.jsonl` exactly, and sampling stayed
  byte-identical. Each arm has 96 legal runs and 71,022 generated decisions. The fallback never fired.

  | Arm | holds ≤ 60 ms | releases 1-40 ms before a head (source 0.42%) | guard (i) mean diff |
  | --- | ---: | ---: | ---: |
  | 48M, no mask | 1.88% | 2.45% | -3.2 pp |
  | 48M, 60 ms mask | 0.40% | 1.63% | -2.4 pp |
  | 64M, no mask | 1.35% | 1.57% | +1.7 pp |
  | 64M, 60 ms mask | 0.34% | 1.38% | +0.9 pp |

- With the mask, the generated part has no short holds at all. The remaining 0.34-0.40% are holds inside the
  fixed source prefixes of the prefix-natural panel. **The cache's source charts do contain short holds: 179 in
  the evaluated cache.** The notes' "real charts have no holds ≤ 60 ms" is therefore wrong for the cache, or
  true only after some filter. The native `.osu` files were not rechecked.
- Near-head releases by star band, source / 48M no mask / 48M masked: band 2 0.11 / 0.53 / 0.49%; band 3 0.00 /
  0.28 / 0.25%; band 4 0.14 / 0.38 / 0.33%; band 5 1.75 / 9.55 / 6.26%. The excess is 2-5× in every band. Band 5
  carries most of it in absolute terms.
- LN share overshoots the source (14.6% pooled): 17.9% at 48M and 22.5% at 64M. Natural-from-BOS runs exceed the
  source by +4.6 pp at 48M and +9.9 pp at 64M, so LN share grows with training. Guard (i), on the prefix panel,
  still passes.
- Against the decision [d-phasen-base-mask](#d-phasen-base-mask): 48M with the mask fails guard (iv) by 1.2 pp
  on the near-head-release part (365 events against at most 94 allowed). Per the agent's reading of that
  decision, the next step is a proposal to the human, not a launch. The cause investigation
  ([q-ln-length](#q-ln-length)) is still running.

<a id="o-lnlen-cause"></a>**Observation, 2026-10-07: what causes the short holds and near-head releases** ([r2-ln-length-20261007](r2-ln-length-20261007.md), fresh Opus subagent, measured on bings-mac with the run's frozen code; no training).
- **Cause: where LNs are placed, not how they end.** From the start of a song, the model's per-chart LN level is
  unrelated to the source chart's: r = 0.00 at 48M and -0.16 at 64M over the 16 panel charts. Given the source's first
  third, r rises to 0.84-0.85. On its own history, the LN-birth rate exceeds the same chart's teacher-forced rate by
  41% at 48M and 78% at 64M, and at 64M it rises along the song. On dense, LN-light 5-star charts this fills rows
  40-60 ms apart with LNs. One such chart has 30 source LNs; the model gives it about 530 per seed. Three of the 16
  panel charts carry 86-89% of the failures
  ([s-lnlen-gen](r2-ln-length-20261007.md#s-lnlen-gen), [s-lnlen-exposure](r2-ln-length-20261007.md#s-lnlen-exposure)).
- **Release decisions are calibrated.** At teacher-forced states, the model's mass on short and near-head
  outcomes is 14-15% below the source's. On its own history, matched for context, its release, short-hold and
  near-head rates are at or below the teacher-forced ones. Generated hold-length distributions match the sources'
  (median 162-167 ms against 175 ms).
- **Representation: not the cause (strong).** Raw `.osu` and the cache agree on all 358,953 fit_dev holds. Snapping
  removes about 1.5% of short holds and creates almost none. Candidates allow both outcomes everywhere, but the
  pointer does not over-use them ([s-lnlen-repr](r2-ln-length-20261007.md#s-lnlen-repr)).
- **[h-keep-underlearned](#h-keep-underlearned) refuted (strong):**
  - The release hazard by elapsed hold length is 0.81-1.06 times the source's above 30 ms (59,251 held-lane
    states), and lower than the source's at 30-150 ms.
  - Hold age is an input of the row decision (`lane_query`), and keep is not weighted differently in the loss.
  - The only excess is a tail under 0.1% of holds ([s-lnlen-keep](r2-ln-length-20261007.md#s-lnlen-keep)).
- **Exposure bias amplifies, it does not create.** The drift is in the LN level only, a channel guard (iii) (64
  decisions) does not see.
- **R1:** R2 avoids R1's two timing mechanisms. Release-only events are 72% of generated near-head releases against
  74% in sources, and holds of 40 ms or less occur at the corpus rate. It repeats R1's third mechanism in milder form:
  too many LNs on dense rows, with an LN level fed back through history ([s-lnlen-r1](r2-ln-length-20261007.md#s-lnlen-r1)).
  [a-ln-mode](r2-ln-design.md#a-ln-mode) predicted this on 2026-10-03: "LN amount is a chart- and passage-level
  choice the row model cannot infer reliably".
- **R2 v1 against v2: no regression.** On the same 16 charts, v1 at 48M is worse: 5.3% short holds against 1.7%, and
  an LN-birth rate on own history 2.4 times its teacher-forced rate. v1's "at most 0.3%" came from an easy 4-chart
  panel, on which phase N is also at 0.26% or less ([s-lnlen-v1](r2-ln-length-20261007.md#s-lnlen-v1)).
- **On a random panel, 48M passes guard (iv) from BOS:** 0.40% short holds and 0.59% near-head releases, against
  the sources' 0.98%. This is 48 other fit_dev charts, one seed; there the model's free LN level is below the sources'.

<a id="c-no-short-holds-claim"></a>**Correction, 2026-10-07: "real charts contain no holds of 60 ms or less" is false for the corpus.**
That claim, in [o-phasen-no-selection](#o-phasen-no-selection), in [p-min-hold-mask](decoding-stage.md#p-min-hold-mask)
and in the brief behind [d-phasen-base-mask](#d-phasen-base-mask), holds only for the 8-chart panel it was measured
on. In the corpus, 2.72% of holds are 60 ms or less, rising to 7.0% at 5 stars; 14% of charts have at least one.
The 16-chart selection panel's own sources are at 5.8%: one 5.1-star chart has 171 holds of exactly 60 ms. Releases
1-40 ms before another head are 1.96% of corpus holds, rising to 5.5% at 5 stars
([s-lnlen-repr](r2-ln-length-20261007.md#s-lnlen-repr)). Consequences:
- Guard (iv) as built cannot be met by a model that matches the data. Its 0.5% ceiling is below the panel sources'
  own 5.8%. Its 0.42% near-head reference pools a mostly LN-light panel, whereas plan v4 specified the band's rate.
  It also counts holds in copied source prefixes ([s-lnlen-guard](r2-ln-length-20261007.md#s-lnlen-guard)).
- A 60 ms mask counting `<= 60` forbids 2.7% of real holds (7.0% at 5 stars), whole deliberate patterns included.
  The human chose the mask on a premise that is now false.

<a id="a-lnlen-reading"></a>**Agent reading (main thread, 2026-10-07; report spot-checked against the mirrored outputs, nothing new run).**
- The evidence supports the subagent's ranking. The training-side lever is a chart-level LN commitment in the natural
  model: phase N has no way to know a chart's LN level from the start, and its own history then amplifies whatever
  level it drew ([p-lnlen-remedies](r2-ln-length-20261007.md#p-lnlen-remedies), R1). A long-horizon LN-drift
  measure in selection costs nothing and belongs with it (R2a).
- Guard (iv) needs correcting before any checkpoint can be judged by it. The candidates are per-band references
  from fit_train, model-made holds only, and `< 60` or ≤ 40 ms. This is an evaluation change and the human's call.
- Not established: the causal step. The prefix contrast is observational: the prefix carries the LN level and also
  the rest of the chart's style. Also open is whether band-5 near-head releases stay above the corpus rate once the
  level is right. That cell is 4.3% against 2.4% on the random panel, from 20 events.

<a id="d-lnlen-next"></a>**Decisions (human, 2026-10-07, answers to the agent's three questions, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#answer-3)).**
1. **A fine-tune test first.** Add a whole-song LN-level input to phase N: zero-initialised, dropped out to
   "unknown" in training, and drawn from a data prior when generating from scratch. Fine-tune from 48M for about
   8-16M exposures. A full retrain follows only if the test works.
2. **Correct guard (iv).** Per-star-band reference rates from fit_train; only holds the model made count;
   a short hold is under 60 ms (strict); a whole-song LN-drift check is added. Then rerun selection on phase N.
   The tolerances are the agent's proposal and remain revisable.
3. **The 60 ms decode mask is off by default.** The option stays in code. This withdraws the mask premise of
   [d-phasen-base-mask](#d-phasen-base-mask). Whether 48M becomes phase C's base waits on the corrected selection
   and the fine-tune.

<a id="r-guard-lnlevel-job"></a>**Delegated 2026-10-07 12:32 UTC: Astra job `20261007-123218-r2-guard-lnlevel`** (`--rw`; brief in `~/ensomi/.sync/cp/jobs/20261007-123218-r2-guard-lnlevel/brief.md`). It implements [d-lnlen-next](#d-lnlen-next) items 1 and 2.
- **Part A, the corrected guard.**
  - Guard (iv) v2: only model-made holds count; a short hold is under 60 ms (strict); per-band references
    from fit_train. For each part, the observed count may be at most 1.25 × the band-expected count.
  - Guard (v): |mean first-to-last-third LN-share drift| ≤ 0.05 over `natural_bos`.
  - The 1.25 and 0.05 are agent choices, revisable.
  - Selection is rerun on the regenerated natural panels of 16M-64M.
- **Part B, the LN-level input (code and a smoke run only).**
  - A flag `ln_level`: query channels for a known bit and the whole-song LN share, zero-initialised.
  - Training dropout to "unknown" with probability 0.3.
  - A strict phase-N `warm_start` from 48M.
  - A fit_train prior P(level | band, density tercile), and level modes unknown, oracle, prior and fixed in
    sampling and evaluation.
  - Config `ce_v2_n_lnlevel_ft.json`: 12M exposures, peak lr 1e-4 cosine to 3e-5, full evaluation every 4M.
  - The main thread launches the fine-tune after reviewing the code.

<a id="q-lnlen-hintfree"></a>**Question (human, 2026-10-07, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#prompt-6)): why does phase N not learn a normal LN distribution without a whole-song hint?**
The human hopes the natural model learns a normal LN distribution without being told the chart's LN share, and asks
for a deeper cause.
- Agent reframing: without a hint, the model cannot know a given mapper's level. "Normal" means three things: a level
  consistent with the skeleton, with the data's spread and bimodality; kept along the song; and placed within the
  level like real charts, dense rows included.
- Ruled out in reply: under-trained song starts. Training windows start at BOS with p 0.125 (`data.py`).
- Four mechanisms proposed:
  1. how far head times reveal the level;
  2. under-commitment of the level (the spread ratio 0.43 at 48M);
  3. self-misidentification on own history (the +41-78% drift);
  4. placement within a level.

<a id="d-lnlen-hint-diagnostic"></a>**Decisions (human, 2026-10-07, [private, local](private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md#answer-6)).**
1. **Investigate now.** A fresh Opus subagent tests the four mechanisms on 48M and 64M, without training. It
   imports the frozen phase-N code, reuses the [r2-ln-length-20261007](r2-ln-length-20261007.md) tables, writes to
   `artifacts/r2-ln-level-20261007/` on bings-mac, and keeps scripts in `~/ensomi/.sync/cp/scratch/r2-ln-level/`.
2. **The whole-song LN-level input is a diagnostic only, not the fix.** It supersedes item 1 of
   [d-lnlen-next](#d-lnlen-next) in that role.
   - Its code (Astra job [r-guard-lnlevel-job](#r-guard-lnlevel-job), part B) stays behind an off-by-default flag.
   - Its oracle mode may be used once, as a probe of placement within a level.
   - The shipped natural model gets no hint.
   - Agent reading: the 12M fine-tune is not launched until the investigation reports.

<a id="d-lnlen-distribution-goal"></a>**Decisions (human, 2026-10-07 about 14:10 UTC, [private, local](private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md#prompt-2), [answers 1-2](private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md#answer-1)): the goal is corpus-like LN distributions; a self-drawn level is not a hint.**
1. **Goal.** The natural model should produce LN share and LN length *distributions* like the real corpus. The
   target is distributional, not a per-chart match to a source chart.
2. **What counts as a hint.** A chart-level LN variable that the model samples itself at generation (for example
   the `ln_level` input in prior mode, or a per-chart latent drawn from p(z | skeleton)) may ship in the natural
   model. A level copied from an existing chart (oracle mode) is a hint and stays a diagnostic. This narrows "the
   shipped natural model gets no hint" in [d-lnlen-hint-diagnostic](#d-lnlen-hint-diagnostic) item 2 to "no level
   supplied from outside". It reopens remedies 2 and 3 of the partial report
   ([r2-ln-level-20261007-partial](r2-ln-level-20261007-partial.md), section 5), and the `ln_level` code in prior
   mode becomes a candidate fix again, not only a probe.
3. **Timing.** No fine-tune yet: wait until the analyses of the n48 and n64 runs
   (`20261007-125618-lnlev-run-n48`, `20261007-130636-lnlev-run-n64`) give enough results, then start. Agent
   reading of "both analysis"; it may instead mean that investigation plus Astra's guard and selection job.
- Agent notes, not decisions:
  - Training can feed the chart's true level (it is a function of the chart), so an explicit level needs no
    unsupervised latent. Astra's prior is P(level | band, density tercile); a skeleton-conditional prior is
    sharper (R² 0.17 against about 0).
  - A drawn level fixes the across-chart spread only if the model obeys it and holds it against its own history
    (drift +41-78% earlier, [o-lnlen-cause](#o-lnlen-cause)). Oracle mode tests that.
  - The level carries share, not length. LN length needs its own corpus comparison. If length is also mostly a
    per-chart choice, the level could carry a length summary; check that in the data first.
  - Evaluation then scores, per band against fit_dev: the SD of LN share, the fractions below 0.025 and at or
    above 0.3, the PIT within the skeleton cell, LN length quantiles and the short-hold rate.

<a id="o-lnlen-hintfree"></a>**Observation, 2026-10-07: phase N draws a nearly data-like LN level at the start of a song but does not hold it; it does not misread its own history; LN length is right on average but lacks per-chart styles** (final report [r2-ln-level-20261007](r2-ln-level-20261007.md), fresh Opus subagent, analysis of the n48 and n64 runs with no training; key numbers rechecked by the main thread against the mirrored JSON). Answers [q-lnlen-hintfree](#q-lnlen-hintfree), scored against [d-lnlen-distribution-goal](#d-lnlen-distribution-goal).
- **Holding fails (strong).**
  - The first 128 rows are close to the data: SD of share 0.86-0.93× the sources', and 28% LN-free starts against 30% at 48M.
  - First-to-last-third correlation is 0.61-0.62, against 0.91 for the same charts' sources.
  - Excursions within a run last 2-4× longer and are 1.4-1.9× larger than in real charts.
  - Half of the LN-free starts acquire LNs; in the sources 81% stay LN-free.
  - Drift runs down at 48M (−0.020) and up at 64M (+0.043).
  - Not the cause: the 511-row receptive field, early-song calibration, or recency weighting.
- **No own-history misreading (strong).**
  - At matched history share, band and gap, own-history P(LN) is lower than on real history at 48M (0.053 against 0.062) and equal at 64M.
  - Model-made and dense-born LNs get no extra response.
  - The subagent's reading (inferred): calibrated one-step responses compound on the model's own chance excursions.
- **Per-band shape, not compression.**
  - Over 116 skeletons × 3 seeds, the single-run SD of share is 0.77× (48M) and 1.00× (64M) the sources'.
  - The zero mode is short: 15.5% / 10.6% against 23-24% expected.
  - The level barely follows the skeleton: slope 0.36 / 0.18 against 1.19.
  - Per-band shifts depend on the checkpoint. 48M's pooled KS of 0.055 hides opposite errors by band.
- **Placement within a level is fine on average (moderate):** no excess on dense rows, and fewer short holds than real charts at matched share.
- **LN length.**
  - At matched share and skeleton, median length equals the corpus' (×1.00 at 48M, ×1.06 at 64M), and the within-chart spread matches.
  - Chart-to-chart length styles are compressed: SD 0.63-0.74× the corpus in every band. Charts built on 1/8-beat LNs (bands 4-5) are missing.
  - Short holds are spread thinly over many charts. In band 2, 16-22% of generated charts have a hold under 60 ms, against 0.3% of real ones.
  - Length style is held only with r 0.20-0.36 along the song.
  - In the data, 16-34% of length variance is between charts. Skeleton and share predict 54-61% of the chart mean; a residual chart style of about 0.3 log units remains.
- **Proposed by the subagent (inference, for human review):**
  - a prior-mode `ln_level` fine-tune from 48M, with a length channel (median LN length in beats) drawn jointly with the share from the same prior chart;
  - the skeleton-neighbour prior as an upgrade of P(level | band, density tercile);
  - an evaluation that adds a fixed-level sweep and one oracle pass for obedience;
  - confirm and refute criteria in report section 8.1. Cost about 2.5-3 h plus small code.
  - Re-ranked remedies are in section 8.2: a prefix-rollout objective second, a latent third; BOS calibration and long memory demoted.

<a id="c-lnlen-cause-revised"></a>**Corrections to earlier readings (2026-10-07, from [o-lnlen-hintfree](#o-lnlen-hintfree)).** The original text of [o-lnlen-cause](#o-lnlen-cause) and the partial report stays as written.
- "Amplified by the model's own history" ([o-lnlen-cause](#o-lnlen-cause)): the +41-78% own-history LN-birth excess compared a run with its *source* chart. That is a difference of level between run and source. At matched history the model does not over-respond. The level wanders instead of being amplified.
- "SD ratio 0.43" ([r2-ln-level-20261007-partial](r2-ln-level-20261007-partial.md)) is the SD of seed-averaged levels on the 16 panel charts. Single runs on 116 skeletons give 0.77-1.00. The failure is the per-band shape and the holding, not a uniformly compressed draw.

<a id="d-lnlevel-ft"></a>**Decision (human, 2026-10-07 about 15:20 UTC, [private, local](private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md#answer-3)): fine-tune with a share and a length channel.**
- Order:
  1. Review and commit Astra job [r-guard-lnlevel-job](#r-guard-lnlevel-job).
  2. A new Astra job adds a length channel to `ln_level`: the median LN length in beats, drawn jointly with the share from the same prior chart.
  3. Review that code.
  4. Run the 12M fine-tune from 48M.
- Evaluation:
  - prior mode, on per-band share and length scorecards (`lnlength.py`);
  - a fixed-level sweep and one oracle pass, for obedience;
  - the confirm and refute criteria of [r2-ln-level-20261007](r2-ln-level-20261007.md) section 8.1.
- The prior (band × density tercile, or the skeleton-neighbour upgrade) is used only at sampling, so it can change without retraining.

<a id="d-night-20261007"></a>**Decision (human, 2026-10-07 about 15:20 UTC, [private, local](private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md#answer-4)): run the LN fine-tune and then the difficulty pilot tonight, scheduled automatically.**
- **LN.** The share and length fine-tune of [d-lnlevel-ft](#d-lnlevel-ft) is the only LN training tonight. Phase C's scoped LN-share request waits.
- **Difficulty pilot (phase C, C1-type).** It runs after the LN fine-tune finishes.
  - Defaults, the agent's option as selected:
    - the head-mask tilt of C0 ([o-c0-tilt](r2-phasec-c0-20261007.md#o-c0-tilt));
    - a small controller that learns the analytic NB teacher by KL on source states;
    - requests inside the natural r range per scope length (about p1-p99) and at least about 20% from natural ([o-c0-screen-scored](r2-phasec-c0-20261007.md#o-c0-screen-scored));
    - scopes of 2-16 s;
    - evaluation by held-out free runs against the analytic controller and natural.
  - Agent reading: the base is the fine-tuned checkpoint, frozen, with the level drawn in prior mode. 48M is used only if the fine-tune leaves no final checkpoint.
- **Scheduling.** Code is written now. The runs chain without a human check tonight, at the human's request. The usual practice is that the human checks before a run.
- **Astra.** The next Astra jobs use the fast service tier.
- **No invented settings.** Any setting that the plan materials do not fix is listed in the job reports.

<a id="o-selection-v2"></a>**Observation, 2026-10-07: the corrected selection picks 56M** (Astra job `20261007-123218-r2-guard-lnlevel`, code `827e306`; report `artifacts/r2-guard-v2-20261007/report.md` on bings-mac; final message read by the main thread, table not re-derived).
- Guard (iv) v2 counts only model-made holds, a short hold is strictly under 60 ms, references are per band, and the ratios must be at most 1.25. Guard (v) bounds |drift| at 0.05. All seven regenerated checkpoint totals match `evals.jsonl`.
- 56M passes everything: (iv) 0.454 / 0.746, (v) +0.034.
- 24M also passes but lies outside the two-SE NLL plateau.
- 48M fails (iv) on near-head releases (1.355), as do 32M (1.389 / 3.016) and others; 16M, 40M and 64M fail (v).
- Per band, the band-2 near-head ratio is 12-14 at 48M and 64M (diagnostic).
- `ln_level` smoke, 0.2M exposures from 48M:
  - training runs at 14.45 min per million exposures;
  - a three-mode full evaluation takes 116 min;
  - oracle and prior modes pass all guards; unknown fails the near-head check (1.56).

<a id="d-base-56m"></a>**Decision (human, 2026-10-07 about 15:25 UTC, [private, local](private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md#answer-5)): tonight's fine-tune starts from 56M, the selected checkpoint.** It replaces "from 48M" in [d-lnlevel-ft](#d-lnlevel-ft) and [d-night-20261007](#d-night-20261007), and the difficulty pilot uses the fine-tune as its base.
- Main-thread choice, not a human decision: no full free-run evaluation inside the training run, for wall time. Astra estimated 8 h 40 min with three full evaluations. The chain evaluates the final checkpoint instead.
- Delegated: Astra job `20261007-151553-r2-lnlevel-length` (fast tier) adds the length channel; brief in `~/ensomi/.sync/cp/jobs/20261007-151553-r2-lnlevel-length/brief.md`.

<a id="r-night-20261007"></a>**Running, 2026-10-07 from 15:46 UTC: the night chain** ([d-night-20261007](#d-night-20261007), [d-base-56m](#d-base-56m)).
- **Length channel.** Code `8bd2cdd`, from Astra job `20261007-151553-r2-lnlevel-length`.
  - `ln_length` adds a known bit and the median log2 hold length in grid beats, undefined below 10 holds.
  - Share and length drop to unknown together.
  - Prior v2 draws share and length from one fit_train chart (11,368 observations, 1,488 without a defined length).
  - `tests/r2`: 284 passed, 4 skipped.
  - The 0.2M smoke from 56M trained in 84 s.
  - Main-thread review: the code is again heavy with validation of internal values (`checked_length`, closure and gap contract checks). It was committed as is so the run could start tonight. Removing that bulk is a morning item under the new rule in `~/ensomi/AGENTS.md`.
- **Fine-tune.** Run `r2-lnlevel2-12m-20261007`, `ens` job `20261007-154601-r2-lnlevel2-12m`, launched 15:46 UTC.
  - Config `ce_v2_n_lnlevel2_ft.json` with `--set ln_prior=artifacts/r2-lnlevel2-20261007/ln-level-prior-v2.json`.
  - Teacher-forced evaluation only at 4M, 8M and 12M.
  - Expected to end about 17:15-17:30 UTC.
- **Astra job `20261007-154633-r2-night-chain`** (fast tier) writes three things; brief in `~/ensomi/.sync/cp/jobs/20261007-154633-r2-night-chain/brief.md`:
  - the LN evaluation of the final checkpoint, Part A;
  - the difficulty-controller pilot, Part B;
  - the chain script, Part C.
  - Main-thread choices in that brief, not human decisions:
    - teacher KL cap κ = 1.0;
    - on source states the base reads the source's own share and length;
    - the final 12M checkpoint is used without selection, with 56M as the fallback;
    - a runtime budget so the chain ends by about 23:00 UTC.
