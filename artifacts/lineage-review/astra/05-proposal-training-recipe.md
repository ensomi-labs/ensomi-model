# Proposal training recipe: lineage review, 2026-09-30

## 1. Scope and sources

Scope: slice 05, proposal training data, budget, recipe, and limits of architectural inference, from R1 through `audio-joint-2026-09`.
The strongest conclusion is that the tested endpoints failed; the evidence does not establish that their architecture families cannot learn ordinary charts.
The late records usually state this limit themselves. The lead about small pilots is supported as a limitation, not as proof that every historical interpretation overclaimed.

Evidence labels: **checked-artifact** means a saved run record or source dataset was read, not an experiment reproduced; **checked-code** means implementation was inspected; **doc-claim** means only a historical account supports the statement.
Numerical census and exposure results computed for this review are checked-artifact, with scripts and inputs named below.
All paths beginning with `J/` abbreviate `artifacts/joint-audio/`; `D/` abbreviates `/tmp/lineage-review/trees/audio-joint/docs/research/`; `S/` abbreviates `/tmp/lineage-review/trees/audio-joint/src/ensomi_model/research/`.
These are citation abbreviations, not additional files.

Read the relay entry point, feedback-index sections 2–3, baseline training-distribution/restoration documents, all named slice documents, and selected supporting experiment documents.
Checked corpus indexes and per-set metadata; manifests, frozen source ledgers, fit configs, initialization receipts, loss/step logs and final qualification receipts; and the bounded R1, joint and segment training paths.
The two recovered relay summaries were treated as doc-claims until matched against local records.
No private human text, accelerator run, new generation, training, install, or Git mutation was used.
The historical 5M R1 distribution owner named by its document was absent at the cited repository path; its detailed exposure figures remain doc-claims here.

**Q1 — corpus census.** Run `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/census.py` from the repository root.
Inputs are the four `artifacts/indexes/beatmap*.parquet` files, their indexed `dataset/<shard>/<beatmap_path>` files, and per-set `metadata.json`; output is scratch `census.json`.
No timing filter was reimplemented: the filtered-index membership is counted as stored. All census figures below are checked-artifact.

| Indexed population | Charts | Beatmapsets | Distinct indexed audio paths | Charts with audio now on disk |
| --- | ---: | ---: | ---: | ---: |
| All 4K | 14,689 | 4,201 | 5,050 | 14,689 |
| No timing anomalies | 14,617 | 4,178 | 5,026 | 14,617 |
| No anomalies and inclusive rounded-index 2–6★ | 10,977 | 4,158 | 4,742 | 10,977 |
| Also dense/local-BPM/unique≤3 filtered | 9,242 | 3,573 | 3,641 | 9,242 |

Beatmapsets and audio paths are not deduplicated song groups. The parquet schema has no `group_id`; catalog/manifest groups are reported separately, not silently equated with sets.
Using the exact metadata grouping functions from `S/source_action_modeling/local_corpus.py:22` with only index rows gives 3,978/3,958/3,940/3,395 connected song components for the four table populations, respectively (checked-code and checked-artifact).
Reproduce with `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/group_census.py`, n=14,689/14,617/10,977/9,242; it joins set, beatmap and normalized artist/title identities, without extra annotation-group links.
The canonical TRAIN/VAL catalog has 3,169/435 groups and 11,564/1,652 entries; its annotation-linked split groups are a different census from index-only components.
The unfiltered rounded-index inclusive 2–6★ count is 11,047. Exactly 70 of those disappear in the no-anomaly index.
The legacy stage2 window index has 707,767 rows; its existence is not evidence that V3 trained on it.

| Rounded index stars, half-open bins | [0,1) | [1,2) | [2,3) | [3,4) | [4,5) | [5,6) | [6,7) | [7,8) | [8,9) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 4K | 130 | 3,117 | 3,649 | 3,405 | 2,727 | 1,257 | 321 | 70 | 13 |
| No timing anomalies | 130 | 3,115 | 3,638 | 3,376 | 2,705 | 1,249 | 321 | 70 | 13 |

Nine entries in the inclusive no-anomaly 2–6★ index have rounded rating exactly 6.00; they belong to the [6,7) histogram bin.
Per-beatmap metadata, rather than the parquet itself, records 12,761 ranked (86.8745%), 1,782 loved (12.1315%), 25 WIP (0.1702%), 14 graveyard (0.0953%), five pending (0.0340%), and 102 missing metadata (0.6944%).
Of 9,536 ranked, native unconverted 4K charts with official unrounded rating in [2,6], 762 fail the source MD5 check; the verified reference is 8,774 charts in 3,387 sets, all with audio present now.
Its official star bins [2,3)/[3,4)/[4,5)/[5,6] contain 3,238/2,906/2,047/583 charts.
This independently reproduces the principal membership counts of `J/20260925-ranked-2to6-reference-v1/result.json`; it does not call checksum-mismatched charts bad.

Training populations differ materially: first joint = 121 TRAIN charts/48 groups/48 audio identities; expanded = 585/240/240; alias-restored manifest = 615/240/240; ranked = 6,923/2,573/2,629.
Validation populations are respectively 12, 36, 36, and 36 charts, each one chart per group. Counts come from each owner's `manifest.json` (ranked under `20260925-typed-contract-repair-v1/ranked-corpus`).
The typed pilot document says 614 eligible TRAIN charts, one fewer than the alias manifest population; eligibility and manifest inventory must not be conflated.
R1's admitted TRAIN subset is 11,563 charts/3,169 groups, versus the subsequent paired audit's 11,564 TRAIN catalog entries/3,169 groups (checked-artifact `J/20260924-data-audit-v1/{r1-validation-audit,audit}.json`). The R1 audit records 10,338 actually visited source charts, all TRAIN, and 6,750,000 onset exposures.
That historical audit has 11,136 eligible paired TRAIN charts/3,081 groups and 1,589 paired VAL charts/427 groups, with 490 missing-audio charts overall.
Current disk existence does not retroactively enlarge those training populations or prove decoded content equivalence.

## 2. Attempts

**Q2 — training inventory and what it can support.**
`A` = checked-artifact, `D` = doc-claim. Each table entry explicitly grades its numeric recipe; conclusions in the final column are historical doc-claims unless marked A.
`NR` means **not recorded in the inspected sources**; it does not assert that no other historical log exists. Paired arms share a row only where their separate budgets/results are given.
`C1` = 121 charts/48 song groups; `C2` = 585/240; `C3` = 615/240 manifest members; `T3` = 614 eligible members after the explicit HH exclusion; `R` = 6,923 ranked charts/2,573 groups.
C1/C2/C3/R validation pools are 12/36/36/36 charts. These are eligible populations, not counts of distinct charts actually visited, unless explicitly stated.
`B4` means two song microbatches × two arrangement/interval draws; `B2` is two windows per update. Intervals are eight seconds unless specified.
All passes are **NR, stochastic draws rather than corpus epochs**, except the explicit fixed-ledger or repeated-unit cases below. No row demonstrates converged native quality.
All listed fits use a single training-seed trajectory per arm; no independent training-seed variance estimate was found. Multiple generation seeds are not training replications.
Sources for A entries are the named run's `config.json`, `freeze.json`, `result.json` and/or final `losses.jsonl`; the extraction script `run_inventory.py` preserves selected records in scratch `run-inventory.json` and `run-summary.txt`.
For the early skeleton runs the owner is `artifacts/audio-skeleton/20260923-v1/training/`; subsequent owners are under J. Small software/resource preflights are not counted as separate quality experiments.

| Date, executable revision; run(s) | Modules, parameters, initialization | Data and update recipe | Endpoint, evaluation and historical reading |
| --- | --- | --- | --- |
| Before lineage; R1 six stages | D: 2,281,104 → 2,330,384 → 2,777,232 → 2,917,008 → 3,056,784 → 3,084,432; scratch then staged transfer, last three residuals alone | D: 11,563/3,169; 4.5/5/6/6.25/6.5/6.75M cumulative source-onset exposures, B4/micro2, CPU1; updates/wall NR; seeds 172/471 | A: final audit records 10,338 actually visited TRAIN sources and zero overlap with 36 later VAL sources. D: eight songs × two seeds per restored stage; completion did not establish playability. |
| 09-23 `273c296`; skeleton local-overfit | A: 770,188 fresh timing parameters; frozen R1 downstream | A: four charts; 200 updates, B8, 16.67 s MPS, seed 172; six assessment charts | A: step 200 calibration score .75819; capacity diagnostic, not architecture selection. |
| 09-23 `68a48aa`/`2ae8e3a`; skeleton local/beat pilots | A: 770,188 each; beat-feature input versus local features | A: 48 TRAIN song cases (D for corpus count), 1,200 updates each, B12, 107.70/202.36 s MPS, seed 172; six calibration/six assessment charts | A: calibration picks steps 600/200, scores .35337/.37116. D: 48 downstream exports legal; inspection rejected legality as sufficient. Problems A, C, E, H. |
| 09-23 `09b919c`; `20260923-v1/training/memorize-v1` | D: 2,950,458 joint timing/rows; selective R1 transfer | A: six-group restriction, 32 fixed queries, 300 updates B8, 74.14 s MPS, seed 230923 | A: development-NLL step 300. D: TRAIN 9.619→.00124, VAL 10.494 → 45.349; millisecond repeats. Overfit capacity, not full-song learning. |
| 09-23 `09b919c`; `training/random-v1` | D: same 2,950,458 joint model, R1 initialization | A: C1, 2,400 updates B16, 989.40 s MPS, seed 230923 | A: development-NLL step 2000. D: long waits were incompletely supervised; superseded by complete-wait recipe. |
| 09-23 `608f092`; `training/coverage-v1` | D: same joint model, fresh R1 transfer | A: C1, 2,400 updates B16, 1,118.44 s MPS; 38,400 logical/40,554 physical queries; 39 coverage examples | A: selected step 2000, VAL NLL6.090735; its exposure is 32,000 logical queries (D), not all 38,400. D: useful structures and residual native failures; led to outcome correction. |
| 09-23 `3a5cea9`; `training/native-distill-v2/{control,student}` | D: same 2.95M, initialized from coverage winner; source-only versus native KL | A: 600 updates, 246.90/303.22 s MPS; D: source B16 plus four native contexts; 640 correction examples/390 prefixes from five songs, 128/70 held-out examples/prefixes on one song | D: 18 full songs per endpoint, one matched seed; short TAP repeats 11 → 2 versus control, but held-out KL .2012→.1029 still worse than parent .0530. Both rejected. |
| 09-24 `f25508e`; `20260924-expanded-v1/training/coverage-240-v1` | D: same 2.95M initialization/normalizer | A: C2, 2,400 updates B16, 1,385.83 s MPS; 38,400 logical/40,235 physical queries, seed 230923 | A: NLL-selected step 2400. A: additional-24 NLL6.48517 → 5.48827 in matched endpoint comparison; D: three near-silent outputs among 84. More data helped likelihood, did not solve rollout. |
| 09-24 `b5c5ee3`; expanded `context-training/paths-{local-fused,global-fused,local-bounded,global-bounded}-v1` | A: 2,950,458/3,402,938/2,992,964/3,461,828; R1 common initialization, all-module interval fitting | A: C2; each 1,200 updates B4; 359.26/393.92/367.46/602.79 s MPS, seed 230928; D: 4,800 intervals, 339,728 heads each | D: 42 songs×two generation seeds per cell; all 336 complete. No3% likelihood gain; global/bounded short-attack totals4 versus 17 local/fused. Selected as inspected candidate with composition caveats. |
| 09-24 `4db2335`; alias `context-training/lineage-main-{plain,memory,release}-v1` | A: 3,461,828 each, initialized from different R1 stages; no size change | A: C3; each 1,200 updates B4; 559.08/569.82/555.17 s MPS, seed 230941 | D: nearly equal likelihood; native behavior rather than NLL intended to choose. Transfer comparison omits some R1 modules and is not restored-policy parity. |
| 09-24 `e04b349`; alias `planned-training/planned-main-v1` | A: 4,245,188 audio/H/R/R1, restored candidate-consequence path; R1 transfer | A: C3,1,200 updates B4,991.64 s MPS, seed 230941 | Fixed endpoint; D: head starvation and release deadlines remained, motivating bounded-head and release conditioning. A, B, F. |
| 09-24 `88e5683`; `planned-bounded-head-main-v1` | A: 4,247,438; R1-based planned model with bounded head history | A: C3,1,200 updates B4,867.46 s MPS, seed 230941 | Fixed endpoint; D: repaired head activation but did not settle release probability. |
| 09-24 `7c316e6`; `planned-feasible-release-main-v1` | A: 4,247,438, matched full-hold conditioning | A: C3,1,200 updates B4,907.01 s MPS, seed 230941 | Fixed endpoint; D: improved release conditioning did not remove short-attack failures. |
| 09-24 `10ddaa8`; alias `shared-profile-{base,cond}-main-v1` | A: 4,247,438/4,250,174; common planned parent plus shared profile | A: C3,1,200 updates B4 each,964.39/920.74 s MPS, seed 230941 | Fixed endpoints; D: profiles changed descriptors but retained native control/playability failures. I, C. |
| 09-24/25 `a2bce68`; alias `profile-routing-{shared,routed}-main-v1` | A: 4,250,174 each; same parent, changed downstream density input | A: C3,1,200 updates B4 each,906.86/1041.46 s MPS, seed 252801; D: 4,800 matched intervals | D: eight audios×two requests, one seed; descriptor error6.1967 → 10.5420, routed worse on every audio. Input-removal arm rejected. |
| 09-25 `4567d87`; alias `count-layout-{flat,factor}-main-v1` | A: 4,250,174/4,416,513; common shared-profile parent; R/R1 train, audio/H frozen | A: C3,1,200 updates B4,799.51/910.98 s MPS, seed 253001; D: 4,800 intervals | D: 16 native cases/arm; width+LN error6.3338 → 4.9092, but short attacks/failures worsen. Factorization not promoted. |
| 09-25 `ca3dc65`; `20260925-typed-resource-plan-v1/joint-main` | A: 3,895,879; planned-flat parent, joint typed model | A: T3; 1,200 updates B2 (checked-code `train.py:108`); last-update clock411.43 s MPS, seed 251925 | D: reported total 413 s; 15 native cases on three audios; raw type/resource planning still failed quality. |
| 09-25 `27b9526`; same owner `ln-base-fit` | A: 3,895,879 from joint1200; explicit LN base | A: T3; 400 updates B2 (checked-code same worker); final-update clock145.57 s MPS | D: total 147 s; 15 cases, changes both factorization and scope sampling, not a clean attribution. |
| 09-25 `4cda0e3`; `20260925-head-owned-clock-v1/{baseline,owned}` | A: 3,895,879/4,301,393 from LN-base400 | A: T3,800 updates each; final-update clocks279.82/340.78 s MPS; B NR | D: separated head-history/resource phase, but lifetime and recovery failures required contract repair. |
| 09-25 `4adcdd9`; `20260925-typed-contract-repair-v1/repair-main` | A: 3,947,227 from typed joint1200 | A: T3,1,200 updates; final-update clock448.96 s MPS; B NR | Fixed endpoint; D: bounded clocks/unrestricted local LN repaired mechanisms; broader ranked fitting followed. |
| 09-25 `c348ca2`; same owner `ranked-main` | A: 3,947,227 from repaired small-corpus model | A: R; 6,000 updates,4,641 distinct charts visited,2,812.50 s MPS, seed 251927; checked-code B2 `train_ranked.py:179` | D: 40 native cases on eight audios at steps 2400/6000. Later star MAE .553→.959, short-LN share1.72%→3.18%; retain2400 despite lower source loss. |
| 09-25 `8687c35`; `20260925-scoped-control-target-v1/joint-1000` | A: 3,947,227 from ranked2400; scoped-target recipe | A: R; 1,000 updates,511.82 s; B/device NR in inspected config | D: effective-range regression; scope-specific quantities did not become reliable control. |
| 09-25 `54adf6b`; `20260925-ln-group-coupling-v1/{baseline,coupled}-1000` | A: 3,947,227 each, ranked2400; change conditional LN group law | A: R; 1,000 updates each,483.28/513.16 s; baseline1,395 distinct charts; D: B2 MPS | D: conditional source loss improved; native control remained mixed. Not size evidence. |
| 09-25/26 `d0aff11`,`7c122d4`,`c5fd73f`; demand/per-field/onset owners | A: 36,866/63,362/63,362 auxiliary demand models; parent/fresh details NR | A: 2,000 updates B8 each,20.70/88.88/83.95 s CPU; seeds 251930/251930/261010; exact window n NR | D: activity/count reference tools changed feedback, not proof of learned ordinary organization. Separate from actor architecture. |
| 09-26 `c5b7db8`; `20260926-row-owned-restoration-v1/main-2000` | A: 3,700,881 trainable; D: 4,583,985 total; typed parent with rows returned to R1, audio frozen | A: R; stage64 → 2000,3,872 windows/2,286 charts,1,025.89 s MPS B2, seed 260926; initial64 updates43.41 s | Fixed endpoint; D: row ownership corrected but controls and sustained pattern quality remained open. |
| 09-26 `08b8337`; same owner `frontier-2500` | A: same 3,700,881 trainable; continue2000 | A: 500 added updates B2,269.34 s MPS; D: 1,000 windows/787 charts | D: release feasibility improved but did not qualify playability; common parent for later controls. |
| 09-26 `b7a6c88`; `20260926-r1-composition-baseline-v1/{continued,prior}-1000` | A: 4,583,985/4,719,808 total; 2,737,331/2,873,154 trainable R1; core2500 parent | A: R eligible; 1,000 updates B2,457.61/482.26 s MPS, seed 261210; actual charts/windows NR | D: prior/history split did not repair native quantity drift; multiple paths still under source supervision. |
| 09-26 `c1260d3`; `20260926-active-ln-audio-cues-v1/{continued,cues}-1500` | A: 4,583,985/4,657,905; 3,187,773/3,261,693 trainable R/R1; core2500 parent | A: 1,500 updates B2,789.72/841.68 s MPS, seed 261230; sampled chart/group counts NR | D: active-LN origins available, but no qualified control repair. Additional parameters have specific new inputs, not a size sweep. |
| 09-26 `c3befb2`; `20260926-range-outcome-r1-v1/{continued-128,outcome-128-v2}` | A: 4,583,985 total/2,737,331 R1 trainable; core2500 parent | D: 12 ranked charts/12 audios; A: 128 updates,27.64/1065.57 s MPS, seed 261240; D: 384 sampled candidates | D: seven source-H cases/arm; singles star error1.272 → 1.337 versus source-only. Native-H expansion withheld. |
| 09-26 `2c2b8af`; `20260926-paired-scope-controls-v1/{source,paired}-96` | A: 12,288 trainable difficulty-projection parameters, same inherited actor; total NR | A: 96 updates,36.26/635.69 s MPS, seed 261270; data count NR | D: restricted conditional path did not supply reliable difficulty separation; led to balanced source learning. |
| 09-26 `b2a5351`; `20260926-balanced-condition-r1-v1/{balanced,aligned}-1200` | A: 4,583,985 total/3,187,773 R/R1 trainable; common inherited parent | D: 2,400 examples/703 charts; A: 1,200 updates,819.02/906.02 s MPS, seed 261410 | D: five reserved common-prefix cases×two requests×three seeds; no sufficient control gain. Natural supervised sources did not isolate control from history. |
| 09-26 `3cf169c`; `20260926-trajectory-kernel-r1-v1/{continued,kernel}-128` | D: shared inherited row actor; exact trainable count NR | A: 128 updates, three anchors/update,151.82/1241.32 s, seed 261510; chart/group n NR | D: physical-trajectory matching had gains and regressions; kept separate from semantic judgments. |
| 09-26 `24fc4f1`; `20260926-layout-modulation-r1-v1/modulated-128` | D: add condition/history interaction,4,600,369 total; same kernel-study parent | A: 128 updates, three anchors/update,1209.38 s, seed 261510; data n NR | D: some paired conditional response improved; full quality still failed. Structural path changes, not size-only. |
| 09-26 `0591f90`; `20260926-scoped-style-discrimination-r1-v1/{source,aligned}-400` | D: modulated inherited actor; actual trained count NR | A: 400 updates, two anchors/update,254.84/303.95 s, seed 261700 | D: source/style discrimination did not establish calibrated native expression; no independent architecture conclusion. |
| 09-26 `e48e4ba`; `20260926-common-prefix-outcomes-r1-v1/{source,actor}-128` | A: 4,600,369/2,753,715 R1 trainable; modulation parent; audio/H/R frozen | D: 47 fixed generated prefixes/eight reserved,384 factual anchors+768 outcome trajectories; A: 128 updates,116.18/1184.00 s, seed 261810 | D: reserved MAE .51248→.38894 versus initialization but weak separation; deployed later as a research candidate, followed by human long-jack/breathing failure. A, D, H. |
| 09-27 `ba406ef`; `20260927-paired-response-r1-v1/{independent,paired}-128` | A: same total/trainable count, actor128 parent | A: 128 additional updates,1884.34/2055.56 s, seed 271010; D: same 47/8 prefixes,384 anchors+768 trajectories/arm | D: reserved MAE .38894→.55270/.53281; paired gap MAE worsens; some native ranges improve. Both primary comparisons fail. |
| 09-27 `94d0b08`; `20260927-player-state-r1-learning-v1/{source,response}-384` | A: 4,612,657/2,766,003 R1 trainable, actor128 plus player-state projection | D: 768 factual draws and 1,152 four-second futures in response arm; A: 384 updates,401.50/1492.60 s, seed 272710 | D: nine Stream cases/arm (three songs×three seeds); pressure .027516 parent→.144229/.107160. Response better than continuation, worse than parent. |
| 09-27 `baed4d7` baseline/`7fdad44` resumed memory; `20260927-audio-memory-joint-fit-v1` baseline/memory | A: 4,583,985/7,616,517 all trainable, core2500 parent | D: 768 32-second draws/417 charts/383 groups; A: 384 updates,1227.90/3058.49 s MPS (memory includes resource attempts), fixed ledger | D: 28 cases/arm plus parent; Stream mean pressure .04447→.78553/.38230. Does not reject memory as a family; rollout objective/coverage remain alternatives. |
| 09-27 `1434924`; `20260927-contextual-ln-fit-v1` reference/contextual | A: 4,600,369 total,143,011 count parameters only, actor128 parent | D: 256 windows/224 charts/214 groups; A: 128 updates (resumed at 64); final 64 paired segment121.83 s; total NR; D: B2 MPS | D: 22 VAL windows and native comparisons; contextual advantage .000088 nats/row, LN disappearance. Cannot directly change within-count layout/release identity. |
| 09-27 `0882315`; `20260927-full-row-history-views-v1` full/missing | D: 4,600,369 total/2,753,715 R1 trainable; actor128 parent | D: 256 windows/224 charts/214 groups; A: 128 updates per arm,290.01 s pair supervisor; B2 MPS, one ledger pass, seed 279812 | D: 14 native cases/arm, two seeds in main LN/Stream cases; full-source NLL improves but missing-history auxiliary worsens pressure relative to full arm. |
| 09-28 `0f5ec10`; `20260928-broader-full-row-learning-v1` | D: same 4.60M/2.75M, full128 parent | A: 1,024 draws/748 charts/651 groups; 512 updates B2,893.52 s MPS, one ledger pass, seed 280028 | D: 22 VAL windows,20 native cases; recurrence improves strongly, three new LN failures; retained candidate, not promoted. |
| 09-28 `c0db49c`; `20260928-scoped-ln-allocation-v1/fit-v2` context/progress | A: 69,953 new trainable each; D: 4,600,369 inherited frozen | D: 512 reused windows/425 charts/392 groups; A: 256 updates B2,344.93 s pair; MPS, one pass, seed 280031 | D: partial LN-quantity improvement; inherited unknown-LN behavior preserved, hold-role and short-articulation problems remain. |
| 09-28 `2653580`; `20260928-release-support-learning-v1` R/R1/joint | D: 3,274,110/4,670,322 trainable from scope parent | A: 1,024 draws/736 charts/661 groups; 512 updates B2 each,1338.75 s pair supervisor MPS, one pass, seed 280033 | D: 28 native cases/arm; LN MAE improvement but four/three new failures. Explicit support-only comparator distinguishes law from fitting. |
| 09-28 `de5d560`; `20260928-head-audio-control-interaction-v1` additive/modulated | D: 513,108/588,372 H parameters train; audio/R/R1 frozen, scope parent | A: 512 updates,1123.78 s pair; D: same 1,024 draws/736 charts/661 groups, B2 MPS; seed 281100 | D: 22 VAL/28 native cases per arm; H NLL improves, D2 mean error worsens1.38534 → 1.61354/1.78828. Missing interaction repaired without qualified quality. |
| 09-28 `ef42095`; `20260928-clean-joint-proposal-v1` inherited/early/fresh | A: 4,675,633 all trainable each; same architecture, distinct initialization | A: 8,192 draws/3,736 charts/2,270 groups; 4,096 updates B2, one pass; 13,450.65 s summed fitting segments/16,957.51 s supervisor; MPS, seed 280281 | A: fixed 512/2048/4096 endpoints; 22 VAL,28 final cases/arm,19/18/17 failures; final semantic review pending. Detailed Q3 below. |
| 09-28 `965d670`; `20260928-ln-fragmentation-repair-v1` pilot+continuation | D: joint release law+hold cues+release loss×2 from inherited2048; parameter count NR | A: 16+64 updates B2,32+128 ledger draws,91.34+321.32 s CPU2, seed 280929 | D: three fixed native witnesses; short-LN burden improved but LN amount/Stream difficulty failed. Several changes at once; step 80 retained for diagnosis. |
| 09-28 `7d31b1e`; `20260928-action-segment-r1-v1/pilot-v1` one/four-state | A: 5,881,252/5,882,215 total,~1.964M trainable, step 80 parent | A: 128 draws/126 charts/groups,158 segments; 32 updates B4,146.18 s MPS, one draw pass, seed 280930 | A: six source-H cases; D: all 85 four-state choices code 1; LN amount .25–.30 for very different requests. No convergence or native-H qualification. |
| 09-28 `4c8463a`; `20260928-continuous-segment-context-v1` code-only/continuous | Parent one-state32; parameters NR; conditional path and supervision-scope repair | A: prepared1,024 general+256 BOS,817 charts,44 validation; planned256 updates B4 MPS; 112 durable,121 logged before abort; wall NR | A: MPS channel-mismatch assertion, terminal failure; cannot count as completed256 or architecture rejection. Step 64 diagnostics only. |
| 09-28 `d6eba23`; `20260928-ordinary-scratch-v1/capacity-fit-v1` | A: 4,782,754 fresh audio/H/R1, continuous plan context, no imported actor | A: four songs/16 four-second units; 512 updates B1,32 passes/unit,235.73 s CPU4, seed 290029 | A: fixed 512; 12 training-song completions, one seed 290031. Source-fit losses improve; native ordinary quality fails. Detailed Q3/Q8 below. |

Chronology/interpretation: the first experiments answered E/H by showing conditional timing learning and then A/B through local native correction; the latter reduced repeats without preserving composition or held-out correction quality.
The context and lineage-transfer studies then tested information paths rather than assuming the restored R1 policy survived migration. The omission of seed/memory/consequence modules is an architectural fact, while its share of the native failure is not identified (D `r1_transfer_stability_audit.md`).
Profiles, count factorization and typed controls pursued I/F; their early successes were descriptor changes, repeatedly followed by native failures. Fairly, the documents often withheld promotion instead of claiming a solved system.
From 09-26, progressively narrower R1 outcome objectives pursued A/G/I, but their fixed-prefix/source-H tasks left the whole BOS/audio-timing rollout incompletely trained.
The late source-learning comparisons pursued B/C/K and established that recipe and parameter ownership could move real organization; they did not establish that the available ordinary distribution had been fitted.
The final fresh expert was expressly documented as a capacity probe. The strongest criticism is insufficient evidence for choosing among broad architecture directions, not that the documents secretly claimed convergence.

## 3. Commentary

**Q6 — the sampling mismatch is measurable, but is not a complete causal explanation.**
Run `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/exposure.py`; inputs are scratch `ranked-verified.json`, the ranked manifest above, and the three named `source-plan.json` ledgers below.
The script reads every draw, joins its source identity, and separately sums actual window heads. All figures in this table are checked-artifact.
“High LN” means whole-chart LN-head fraction ≥.5, not a semantic label for unusual or bad charts.

| Population/actual draws | n | [2,3)/[3,4)/[4,5)/[5,6] draws or charts | High-LN chart exposure | Mean source-chart LN fraction |
| --- | ---: | --- | ---: | ---: |
| Verified ranked 2–6, chart equal | 8,774 | 3238/2906/2047/583 | 5.09% | .1722 |
| Ranked TRAIN manifest, chart equal | 6,923 | 2558/2300/1611/454 | 4.95% | .1729 |
| Broader full-row actual windows | 1,024 | 238/258/259/269 | 26.37% | .2655 |
| Release-support actual windows | 1,024 | 205/272/263/284 | 27.05% | .2717 |
| Clean joint actual windows | 8,192 | 2220/2339/2150/1483 | 13.17% | .2154 |
| Clean joint natural branch | 4,100 | 1470/1278/1026/326 | 6.22% | .1877 |

The corresponding owners are `J/20260928-broader-full-row-learning-v1`, `J/20260928-release-support-learning-v1`, and `J/20260928-clean-joint-proposal-v1`.
Actual LN heads/all heads are 20,815/85,132=.2445, 22,837/85,246=.2679 and 144,265/656,166=.2199; these are window-weighted quantities, unlike the last column.
The ranked TRAIN equal-song-then-chart high-LN expectation is 6.106%; the release-support population branch yields 260/767=33.898%, matching the historical audit's 33.90% claim.
Across that recipe's deliberately hidden-LN draws, 66/217=30.415% still came from high-LN charts. Independent condition dropout did not restore the natural marginal distribution.
Clean joint restricts deliberate numeric dropout to natural draws: hidden-LN high-LN exposure becomes 59/820=7.195%. Balanced/human branches deliberately hide neither numeric request.
These are direct evidence of overexposure relative to the ranked population and of a later recipe correction. They do not establish that balancing caused short tails or that rare-style coverage should be removed.

R1 samples a uniform song group, a uniform chart within it, and a uniform 128/256-onset crop, with seed-stratum probability .125 (`S/bounded_typed_continuation/corpus.py:25`, `:46`, checked-code).
`data.py:179` constructs factual source history and targets; `data.py:264` normalizes full factor loss by source onsets. `train_run.py:62` accumulates microbatch loss against effective batch onsets (checked-code).
Its historical census says 80.08% of consumed onsets came from 2–6★ and only 1.39% above 6★, with consumed LN-head fraction .1796 (doc-claim, baseline `r1_training_distribution.md`).
Consequently “R1 mostly learned high-star charts” is contradicted by the available census account; no comparable semantic census establishes how much of its ordinary organization it actually learned.

The first joint sampler uses song/group choice and BOS/event-prefix/uniform-time/outro queries, followed by complete waiting-interval supervision (`S/joint_audio_continuation/training.py:64`, `:103`, checked-code).
The later fixed-duration recipes use per-second likelihood, so equally likely dense windows contribute more row terms; this is a valid event-likelihood objective with a different measure from macro nats/row validation.
Clean joint code explicitly draws 50% natural, 25% balanced star/LN cells and 25% human charts, and restricts dropout by branch (`J/20260928-clean-joint-proposal-v1/prepare-v2.py:37`, `:52`, `:72`, checked-code).
A claim about dense-gradient dominance remains unmeasured: more row terms do not prove proportionally larger gradient norms.

**Q4 — was scale isolated?**

| Variable | Actual comparison | Result and causal limit |
| --- | --- | --- |
| Data size | Same 2.95M initialization/normalizer, same 2,400-update/B16 budget:121 charts/48 groups versus 585/240 | A `J/20260924-expanded-v1/evaluation-coverage-v1/{initialization-identity,likelihood-comparison}.json`: old 12 NLL6.35422 → 6.21280; additional24 6.48517 → 5.48827. No repeated fit seeds. Wider coverage helped prediction at fixed compute, not qualified autonomous quality. |
| Steps | Ranked typed model's same run at 2,400 and 6,000 | A run reaches6,000/2,812.50 s; D native40-case comparison worsens MAE/short-LN share while likelihood improves. This refutes “more of this recipe monotonically improves this panel,” not scaling in general. |
| Steps | Clean joint512 → 2048 → 4096, common frozen draw order | A final native failures fall from 7/8 each at 512 to19/18/17 of 28 at 4096, but panels differ; failure proportions are not a paired improvement estimate. A likelihood curves still move. More steps also reveal more ledger windows, so this is budget continuation, not fixed-data repeated-epoch isolation. |
| Steps | Source-only control arms in native-distill, outcome, trajectory, player-state and broader-source studies | Extra updates were explicitly controlled in several paired interventions; their source-only controls sometimes regressed. These local continuations do not constitute a converged learning-curve study or a general scaling law. |
| Model size | No width/depth/parameter-count-only sweep found | Memory4.58 → 7.62M, count 4.25 → 4.42M, head-ownership3.90 → 4.30M and modulation additions change information paths or factorization. R1 stage additions change both recipe and modules. They cannot isolate capacity. |

The separate35M vacation teacher is a different pre-lineage training program, not a matched larger version of this audio lineage (D baseline `r1_staged_restoration.md`).
It cannot supply the missing controlled size comparison here. No comparison estimates effect size against independent training-seed variance.

**Q5 — distribution-level comparisons that actually exist.**
The broadest saved comparison is `J/20260925-ranked-2to6-reference-v1`:8,774 verified ranked sources versus 94 complete saved generated exports, with two incomplete exports separately retained (A `result.json`, `generated-result.json`).
Only 85 generated exports fall in achieved2–6★; profiles are density/width/LN requests, not matched star requests. Each arm uses eight audios, two profiles and one seed per audio; policy variants reuse those audios.
Source HH≤20ms is0/12,918,427 eligible intervals; sources contain eight HR≤20ms and zero RH≤20ms. In-range raw-flat14 charts have HH/HR/RH counts2/85/48; raw-factor15 have24/16/16 (A saved census).
The current/preview variants remove the RH≤20 counts but retain short-HR cases. This is solid evidence of specific departures from the reference, not a population-wide failure percentage or physiological limit.

A direct read-only reaggregation checks LN share and chord widths by achieved star band: run `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/distribution_compare.py`.
Inputs are that owner's `result.json` and `generated-result.json`; n=8,774 sources and 29 in-range raw exports, across two dependent model arms. Output is scratch `distribution-comparison.json`.
The following entries are A; values after n are median per-chart LN fraction and pooled multihead-H-row fraction. Source and generated ratings use the documented20241007 convention, but sources/requests are not randomly matched.

| Achieved stars | Ranked reference: n / LN median / chord-row fraction | Raw flat | Raw factor |
| --- | --- | --- | --- |
| [2,3) | 3238 / .1324 / .3400 | 2 / .2004 / .3894 | 3 / .1940 / .3530 |
| [3,4) | 2906 / .1298 / .4063 | 7 / .0967 / .4111 | 5 / .1964 / .3802 |
| [4,5) | 2047 / .1162 / .4277 | 4 / .6272 / .5778 | 4 / .0577 / .4712 |
| [5,6] | 583 / .1288 / .4122 | 1 / .1397 / .7648 | 3 / .0355 / .6265 |

The 4–5★ arms differ strongly in opposite LN directions; “the entire model family always overproduces LNs” is too coarse even here. With four outputs per arm, these are descriptive witnesses, not distribution-estimation precision.
Later comparisons inspect actual four-star LN roles, duration, RH gaps, beat-relative placement and group endings against1,973 ranked TRAIN charts/1,614 groups; A `J/20260928-ranked-fourstar-arrangement-v1/{charts.jsonl,summary-v2.json}`.
That census has 134 no-LN,1,140 TAP-body,636 mixed and 63 LN-body charts. Song-equal shares are6.40/57.65/32.89/3.06%; TAP-body LN median196ms versus 132ms for LN-body (A summary; historical semantic reading remains D `report.md`).
Nine selected source contexts/31 pages provide explanatory examples, not random quality labels. Many generated contrasts then reuse STYX/Blizzard/Classic and a few other songs, often only one or two seeds.
The earlier recurrence reference is1,972 TRAIN charts/1,613 groups using metadata [3.5,4.5), versus the later recomputed closed [3.5,4.5] census1,973/1,614; the denominators are intentionally different (D `full_row_learning_and_difficulty_response.md`; A later census).
No broad, independently held-out, same-star-and-style matched distribution test of the final 84 clean-joint exports was found. Final receipts explicitly leave semantic review pending.

**Q7 — source histories versus own histories.**
Most imitation updates, including memory, full-row, clean-joint, action-segment and fresh-ordinary fits, use factual source histories. Missing-history views hide observations of those histories; they are not generated-prefix training.

| Own-history attempt | How its signal is constructed | Transfer to native generation |
| --- | --- | --- |
| R1 head/release/frontier2 recovery | D: machine preferences on generated TRAIN trajectories, targeting repetition cores, blocked held lanes and short-clock consequences; source CE anchor, inherited parameters frozen | D: actual restored16-output screen improves response-stage RH<30ms67 → 7, while HH<30ms2 → 5. Useful conditional correction, not complete playability. |
| Native distillation600 | Checked-code `S/joint_audio_continuation/distillation.py:57` clears source targets; `:140`–`:158` uses a detached corrected time/row/censor law on generated prefixes | D: training KL improves, held-out KL worsens versus parent; short repeats improve but head/LN composition drifts and one output nearly stops. No successful broad transfer. |
| Range outcome128 and restricted paired control96 | D: sampled row trajectories rescored under the deployed law; source anchors retain genuine histories; outcome costs are difficulty/LN scalars | D: range model fails its singles criterion even with source H. Restricted projection does not establish control separation. |
| Trajectory-kernel128 and layout-modulation128 | D: generated outcomes compared using physical trajectory distances alongside scalar controls, source anchors retained | D: some matched-prefix physical/conditional gains, remaining native failures; geometric closeness is not semantic ordinaryness. |
| Common-prefix actor128 | Checked-code `J/20260926-common-prefix-outcomes-r1-v1/train.py:55`–`:99`: three genuine anchors; six new futures from one of 47 fixed generated states; score-function advantage with other-draw baseline | D: reserved absolute calibration improves .51248→.38894, weak separation; all 768 training outcomes TAP-only. The LN-tail credit path was not exercised and later full-song complaints remained. |
| Independent/paired response 128 continuation | D: same 47-state bank, new current-policy futures; paired gap objective plus causal trajectory KL | D: both worsen reserved absolute calibration; paired arm improves some full-song ranges/Stream reproductions. No uniform carryover. |
| Player-state response 384 | D: 1,152 private four-second futures,445 positive costs,162/384 nonzero-response updates; ongoing LNs retained at finite horizons | D: better than source-only continuation, worse than initialization on the nine-case whole-song Stream panel; targeted signal did not cover new reached failure locations. |

A factual counterexample to “source improvement never helps”: broader full-R1 learning reduces maximum same-column consecutive-H runs21/16/8/17 → 8/7/7/6 in four Stream cases (A `J/20260928-broader-full-row-learning-v1/numeric-comparison-v1.json`).
Its two Zenithfall attack-excess values fall .011005→.0000281 and .023828→.001567; whole-star MAE changes only .630285→.622187 and LN controls regress (A comparison for local metrics; D aggregate interpretation).
Thus source learning can improve native routing, but an NLL gain is neither sufficient nor uniformly harmful. The missing-history arm, ranked6000, memory384, and ordinary512 provide contrary examples to any simple monotonic claim.
Freezing H/R during outcome learning is a legitimate way to isolate R1 derivatives; it is also a different training problem from joint BOS generation on native H. Fixed-bank prefixes are not a census of the current policy's visited states.

## 4. Direction

For broader source learning, the strongest case is that larger paired coverage improved held-out conditional likelihood and full-R1 learning improved actual native recurrence. The strongest countercase is that these gains coexisted with quantity/pressure regressions and no convergence evidence. My call: useful learning evidence, insufficient architecture selection; high confidence.
For successive scalar/control fine-tunes, the strongest case is that they used executable generated trajectories and real source anchors to address specific observed failures. The countercase is that repeated optimization on small prefix banks and narrow costs never established ordinary autonomous organization. My call: reasonable bounded diagnostics that did not resolve the system-level objective; high confidence.
For information-path changes, the strongest case is that an absent input or hard-excluded action cannot be repaired by more updates. The countercase is that adding a path and observing an undertrained failure does not show that the new formulation is wrong. My call: retain demonstrated path/support facts; leave family-level performance judgments open; high confidence.

**Q3 — three-arm clean joint comparison: lead supported for rejecting a family-wide architectural conclusion; finite-budget initialization comparison supported.**
The logs verify 4,675,633 parameters per arm, 4,096 updates, batch two, 8,192 fixed draws from 3,736 TRAIN charts, 2,270 song groups and 2,300 encoded-audio identities.
There are 22 separate validation identities: the preparation receipt's 3,758 unique identities includes them. The lead's 3,736 training count is correct.
All weights in audio/H/R/R1 train; inherited/early/fresh initializations share one architecture, direct LN law and recipe. It is not a three-architecture comparison.
A full pass means one traversal of this draw ledger, not one epoch over every chart or all source events. Model seed 280281 is fixed; no training-seed replication is recorded.
The MPS supervisor finishes in 16,957.51 s including its orchestration/evaluation, not a per-arm training time. The 128 fitting-segment receipts sum to 13,450.65 s across the three arms.
Reproduce the complete 4,096 unique update IDs, curves, summed times and ordinary-unit multiplicities with `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/late_audit.py`; inputs are both owners’ fit plans, step logs, diagnostics and segment result receipts. Final receipts contain 28 completed cases per arm on five distinct audio identities, 19/18/17 failures, no promotion, semantic review pending. The final `plan.json` uses fourteen seed integers across distinct conditions; main paired conditions typically have two seeds, not 28 independent songs.
Evidence: owner's `fit-plan.json`, `source-plan.json`, `fit-v1/0000-0032/initialization.json`, `supervisor-v1/result.json`, and three `native-4096-*/result.json` (checked-artifact); `worker.py:59` checks batch/update execution (checked-code).

For the direction: the matched ledger and common law make initialization sensitivity worth testing, and the completed native failures justify withholding these checkpoints.
Against a broad conclusion: fresh validation H NLL/s still improves 39.0281 → 34.4115 → 32.7612 at 512/2048/4096 updates, and row NLL 2.4043 → 1.9538 → 1.8423; convergence is not demonstrated.
Inherited and early row NLL get worse between 2048 and 4096 while H improves, showing a tradeoff rather than one converged quality optimum (`validation-*/result.json`, checked-artifact).
Call: architecture adequacy undetermined; endpoint rejection supported, high confidence. The main document explicitly calls short pilots gradient/resource tests and stops at intermediate 512; the recovered summary correctly records completion to 4096.

**Q3 — action-segment R1: under-training concern supported; a specific information-path limitation is also supported.**
The matched one-/four-state fit made only 32 updates, batch four, on 128 source draws from 126 charts/groups, expanded to 158 factual segments, 3,960 target rows and 160,831 release-risk clocks.
Trainable parameters are 1,963,814/1,964,777 with 3,917,438 frozen in each arm; totals 5,881,252/5,882,215. Parent is the fragmentation-repair step-80 actor; audio/H stay frozen.
One seed (280930), MPS, 146.18 s for the pair; no convergence demonstration. Six native source-H outputs use three audios, one generation seed per arm/audio.
Evidence: `J/20260928-action-segment-r1-v1/{fit-plan.json,source-data.json,pilot-v1/initialization.json,pilot-v1/result.json}` (checked-artifact).
For: comparing an exact one-state decoder and four-state mixture can expose broken or unused conditioning at low cost.
Against: a 32-update partially fresh decoder cannot establish capacity failure, nor can a collapsed latent prior prove that all segment formulations collapse.
All 85 four-state selections chose code 1 (checked-artifact saved `metrics.plan_counts`, summed by `generation_audit.py`:26+23+36). Only one segment had actual BOS supervision (doc-claim `D/action_segment_r1.md`).
A separate ledger recount finds 36 of 60 human-branch views outside any recorded style span, a later-corrected exposure defect; section 5 gives the code and reproduction.
A constant one-state code mathematically cannot carry changing plan input; that information bottleneck is independent of more training. It does not imply all audio is absent: direct row audio and preview remain.
Call: this pilot failed and was too limited to reject segment planning, high confidence; absence of a varying categorical information path is a narrower legitimate architectural finding.

**Q3 and Q8 — ordinary expert from scratch: lead verified; ordinary-only broad training was not completed.**
The capacity fit uses four ranked sources (Dawn, Sulyvahn, mumei, Kill The Beat), sixteen four-second units, 4,782,754 entirely trainable parameters, CPU four threads historically, and 512 updates in 235.73 s.
Its fit plan repeats each unit 32 times with batch one; initialization has no parent checkpoint, seed 290029, draw seed 290030. The fixed step-512 endpoint is used, not an early-stopping winner.
Evidence: `J/20260928-ordinary-scratch-v1/{capacity-fit-plan.json,capacity-fit-v1/initialization.json,capacity-fit-v1/result.json}` (checked-artifact; exact draw multiplicities checked separately).
For: an inspected tiny fit can demonstrate local conditional capacity and expose startup/rollout bugs before expensive training. Genuine source-prefix versus BOS and source-H versus native-H comparisons address real deployment differences.
Against: 64 seconds of selected targets, including sparse entry/quiet blocks, cannot certify learning four complete songs, much less the 1,602-chart ordinary TRAIN scout population.
The twelve completions reuse the four teaching songs; this is a capacity/rollout probe, not held-out generalization. They have one generation seed (290031), no run-to-run variance estimate.
The artifact diagnostics verify fit-unit mean H NLL/s 31.01186 → 10.72888 and row NLL/s 23.07761 → 2.29225. The document says Sulyvahn's intended held structure was not yet learned and most generated positions were outside fit units; source loss reduction therefore does not establish that even this tiny task was fitted adequately.
Raw-export recount verifies source-H/native-H LN≤40ms counts 1/31,9/47,10/68,9/45 for the four songs; all eight BOS variants contain such tails. Reproduce with `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/generation_audit.py`, which verifies saved SHA-256 and parses all twelve ordinary and six segment exports; output is `generation-audit.json`.
Call: rejection of this endpoint supported; rejection of fresh ordinary learning undetermined, high confidence.
**Q8 answer:** no adequately fitted model trained only on a broad ordinary-ranked narrow-star population and then qualified on native output was found. The four-song attempt is a real, incomplete exception to “ordinary training was never tried.”

## 5. What was overlooked or never questioned

**Q9 — ranked omissions, distinguishing “never tested” from “never noticed.”**

1. **Adequate fitting of the intended ordinary population was never established.** Ranked status,2–6★ eligibility and a balanced sampler did not define ordinary default organization; the final scout explicitly says its 1,973 candidates are not approved ordinary labels. Only four teaching songs entered the fresh fit (checked-artifact scout/fit records). This is the largest missing premise behind architectural judgments about C.
2. **Optimization adequacy was not isolated from structure.** No matched LR/clip/epoch or width-only sweep appears in the inspected conclusion-bearing runs. In the 4,096 clean-joint updates,100% of logged gradient norms exceed clip 1, with medians 65.63/66.29/85.82; in ordinary512,95.31% exceed clip 5, median71.92 (checked-artifact `late-audit.json`; checked-code clean `worker.py:81`, ordinary `capacity_worker.py:92`). Clipping can be intentional and Adam is adaptive; these figures do not prove it caused failure, but “budget exhausted” is not “optimized sufficiently.”
3. **Training request combinations lagged the public controls.** The clean recipe repaired unknown-LN marginal exposure yet retained no effective Stream-known/LN-unknown training pieces in the published audit (doc-claim `D/optional_control_training.md`:235 pieces overall,74 near4★, zero LN-unknown). This was eventually noticed; it was not settled by a qualified matched fit. The tiny teacher-state probe's small effect is evidence against assigning it all native collapse.
4. **Style-centered crop selection could silently lose the style.** Recounting the segment ledger finds36/60 human-branch views with no overlap with any recorded style span, despite sampling that branch for style coverage (checked-artifact `segment_exposure.py`, n60 views from 128 draws). The original sampler chooses any partition (`J/20260928-action-segment-r1-v1/prepare.py:41`); later `S/segment_audio_continuation/segments.py:38` filters for actual style supervision. This weakens interpretation of the 32-update failure; it does not invalidate its factual row targets.
5. **Local target semantics versus whole-song difficulty were not isolated.** Recipes mix whole-song ratings and 16/32/64-second strain proxies; ordinary512 keeps whole-song ratings across quiet/dense units. That semantic change could affect breathing or uniform pressure, but the four-song run also changes architecture, initialization, data and support, so no contribution is identified (doc-claim recipe comparison; checked-artifact ordinary plan).
6. **Distribution fidelity was repeatedly replaced by a selected-state or selected-scalar test.** Source likelihood, a47-prefix bank, whole LN totals and low-dimensional pressure all answer narrower questions than autonomous ordinary organization. The lineage did identify this problem, but no broad held-out endpoint distribution test resolves it; all three clean final receipts say semantic review pending (checked-artifact).
7. **Support and weighting altered what the model could learn before an optimizer step.** The 60/50/50 support excluded real ranked articulation; later support audits found457 affected TRAIN charts and 64/64 recovered test windows (doc-claim `D/release_support_joint_learning.md`). Density-normalized loss, resampling and rare-condition balancing must be distinguished from the nominal corpus; the direct exposure census verifies the latter mismatch. This is a recipe issue as well as an inference-law issue.
8. **Uncertainty from training and repeated development was not quantified.** Matched seeds and frozen ledgers improve paired comparisons, but nearly all causal claims use one fit seed and repeatedly inspected songs. No measured between-fit variance calibrates small NLL differences or selects a universal best direction. This remains a gap, not proof that the observed within-panel effects are noise.

The record did question data size, source/native mismatch, missing controls, loss normalization, frozen audio, support and initialization. Calling these subjects “never questioned” would misrepresent its explicit audits.
What remained unasked in an experimentally decisive form was whether an ordinary distribution could be learned to a defensible fit criterion under a stable recipe before the next module-level change.

## 6. Worth keeping

- The indexed/raw corpus census and immutable source identities: actual ranked eligibility, status, MD5 matches, available audio, and group-versus-chart weighting are reusable evidence (checked-artifact census and manifests).
- The observed exposure ledgers, including rejected draws and control missingness. They make the 33.90% versus 6.11% mismatch auditable and show that a later natural branch corrected it substantially (checked-artifact source plans and `exposure.json`).
- Complete-wait/event likelihood and truthful source-history handling. The checked training and distillation code explicitly distinguishes observed survival, source targets and generated-prefix teacher distributions; this survives rejection of the particular networks.
- Matched controls for additional training, initialization, support relaxation and full versus missing history. They prevent assigning every common regression to the newly added module, provided the small-sample limits remain attached.
- The broader-row result as a positive learning observation: ordinary TAP movement/recurrence can improve through source fitting, even when controls regress. It is evidence against “only architectural impossibility,” not a qualified checkpoint (checked-artifact numeric comparison; historical image interpretation remains doc-claim).
- Ranked LN relationship measurements, including conditional duration/RH gaps, shared endings and held-role/TAP organization. Corrected `summary-v2.json` treats no-opportunity cases as null instead of zero; this is better measurement hygiene than a global short-LN average (checked-artifact receipt and summary).
- The explicit failed qualification records and preserved source-H/BOS/native-H exports. They substantiate failure without converting training loss into acceptance. This review independently checked hashes and short-LN counts of 18 saved exports via `generation_audit.py`.

No architecture from this slice is recommended for adoption. The strongest reusable result is the ability to distinguish available source data, actual optimizer exposure, conditional prediction and autonomous behavior.

## 7. Claims worth re-verifying

| Claim or missing measurement | Exact follow-up evidence/command | Why it still matters |
| --- | --- | --- |
| Full R1 consumed-distribution accounting | Recover `artifacts/bounded-typed-continuation/training-distribution-20260920-v1/`, especially its readout and per-chart tables named in baseline `r1_training_distribution.md`; inspect `J/20260924-data-audit-v1/r1-validation-audit.json` meanwhile | The 5M census is documentary here; the later6.75M ownership/overlap audit is readable. Do not substitute one checkpoint's population for another's exposures. |
| Ordinaryness of final clean-joint outputs | `J/20260928-clean-joint-proposal-v1/native-4096-{inherited,early,fresh}/{plan.json,cases.json,result.json}` and their saved `generated.osu`; 84 cases total on five audios | Final numerical rejection is checked; a complete independent semantic review was not present and was not performed here. |
| Whether the fresh model fitted its teaching relations | `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/late_audit.py` and `generation_audit.py`; inspect `capacity-data-v1/data.json`, `capacity-review-512-v1.json`, `capacity-interpretation-512-v1.json` under the ordinary owner | Existing data establish loss reduction and native failure, not convergence or adequacy. Longer training would be new evidence, not replication of a completed historical result. |
| Effective style-known/LN-unknown exposure count 235/74/zero | Read `D/optional_control_training.md` against `J/20260928-clean-joint-proposal-v1/source-plan.json` with effective interval intersections, not chart-level style existence | This review checked the scheduler code and related segment exposure but did not independently rebuild the 235-piece census. |
| All per-run data/epoch and checkpoint-selection details marked NR | `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/run_inventory.py`; follow its exact run paths to frozen configs/losses and training scripts | Some older fits lack a final receipt or explicit epoch accounting; a label such as“1200” alone does not identify actual visited charts or convergence. |
| Same-star final distribution comparison independent of development songs | Existing comparator: `.venv/bin/python /tmp/lineage-review/05-proposal-training-recipe/distribution_compare.py`; its inputs and n are in section3 | This reproduces the early descriptive comparison only. A final unseen-song benchmark was not run and cannot be recovered from these receipts. |

All listed commands are read-only except for replacing this slice's scratch summaries. No model run was required to verify the pivotal completed-budget claims.
No command is offered as if it could retrospectively establish an unrun convergence, human-playability or independent-generalization result.

## 8. Cross-slice notes

- The segment decoder deletes `row_consequence` and accepts candidate `local`/`timing` arguments without using them (`S/segment_audio_continuation/model.py:144`, `:159`–`:174`, checked-code). More training cannot learn through that absent path; other real-time inputs remain. This belongs to the information-flow/frontier reviews.
- The first audio transfer copied selected R1 tensors while omitting seed/memory/frontier2; the full restored policy was therefore not the tested downstream policy (doc-claim `D/r1_transfer_stability_audit.md`, mechanism partly corroborated by checked segment/joint ownership code). Attribution among omissions and the new timing task is outside this slice.
- Whole-star range and short-hold burden can pass while organization fails. The final ordinary exports provide an especially small, reproducible evaluator counterexample; qualification/physiological validity is the evaluation slice's question.
- Audio identity is not identical to song-group identity: the ranked TRAIN manifest has 2,573 groups but2,629 byte identities. The early audit also records one cross-split audio identity outside the selected paired cohorts; no perceptual deduplication guarantee is inferred.

## 9. Failed paths and unfinished work

The corpus census, three late exposure ledgers, late budget/curve audit,18-export short-LN recount, and segment style-overlap recount completed on CPU without training or accelerator work.
Initial `rg --files` did not show ignored artifacts; rerunning with `-uu` resolved discovery. A guessed `joint_audio_continuation/train_run.py` did not exist; the actual entrypoints are `training.py`, `context_training.py` and historical per-experiment workers.
A guessed `configuration.json` was absent; the surviving file is `config.json`. These lookup failures did not become negative scientific evidence.
PyArrow emitted sandboxed CPU-cache-query warnings; reads and census calculations completed successfully. No packages were installed.
The historical R1 distribution owner at the cited repository path was unavailable; its detailed original figures remain doc-claims, supplemented by the surviving6.75M audit.
The training table covers the conclusion-bearing families found in the inspected documents/owners; it is not a guarantee that every unpublished one-off fit is recovered. Preflight-only native-distill-v1 and unfinished resource trials are not counted as quality comparisons.
Exact distinct-song/window counts, per-arm timing in some paired workers, and full optimizer histories remain NR where the inspected artifacts do not establish them. No deleted tensor was silently recreated or its metadata inferred from a filename.
I did not rerender/visually inspect all historical Lens pages, listen to audio, play charts, or rerun model probabilities. Their semantic accounts remain doc-claims; saved numeric reports and the 18 checked exports have the stronger grades stated above.
No final 84-output semantic review, independent training-seed replication, convergence study, or new ordinary-population training was performed. Those omissions are limits of the historical conclusions as well as of this review.
Only this report and files under `/tmp/lineage-review/05-proposal-training-recipe/` were created or changed.

## 10. Inconsistencies and items for the human

1. **Intermediate document versus completed run.** `D/clean_joint_proposal_learning.md` ends at 512, while the supervisor and 84 final qualification records reach4096 and all fail (checked-artifact). Any summary based only on the main document understates completed compute; the recovered relay account is correct on this point.
2. **“Three-arm architecture comparison” is a misdescription.** The three clean arms have the same 4,675,633-parameter architecture and differ in initialization under one new recipe (checked-artifact initialization receipts). They test finite-budget initialization, not three competing architectures.
3. **Nominal balanced coverage versus the default distribution.** Hidden-LN samples retained30.41% high-LN exposure under the release-support recipe, against6.11% natural equal-song expectation (checked-artifact recount). The clean recipe repairs this dimension, but the effective Stream-known/LN-unknown audit still reports no examples (doc-claim); these are separate issues, not conflicting counts.
4. **Segment style sampling versus actual supervision.**36/60 style-selected views miss every style annotation (checked-artifact); later code requires an observed style in each retained piece (checked-code `segments.py:38`). The original 32-update fit is valid source imitation with weaker semantic coverage than its branch label suggests.
5. **New information path versus learnability failure.** Segment code removes candidate-consequence scoring and one-state plans cannot transmit a changing code (checked-code); fresh validation curves nevertheless keep improving and no converged ordinary fit exists (checked-artifact). The former establishes a structural limitation, the latter leaves overall capacity unresolved.
6. **Current paired audio versus historical availability.** All 14,689 indexed audio paths exist now, while the 09-24 audit recorded490 missing pairings (checked-artifact both). These refer to different times/path-resolution states; current availability must not be used as historical training exposure.
7. **“Ordinary” remains a human target, not a completed corpus label.** The scout calls1,973 charts candidates and trains only sixteen units from four songs (checked-artifact; doc-claim on intended semantics). The human must settle which ranked organizations define the default and which are expressive requests; neither LN amount nor ranked status alone answers that.
8. **Stopping at native red lights versus inferring architecture failure.** The documents generally reject checkpoints and explicitly disclaim convergence; the fresh fit's loss reduction plus severe short-tail output is real (checked-artifact). Whether the remaining uncertainty warrants more optimization or a changed formulation is a human research decision; this slice does not identify a uniquely justified architectural verdict.
