# 01-r1-foundation — What R1 was shown to do

## 1. Scope and sources

The assumption that R1 was a sound row generator needing only audio-derived event times is too strong.
R1 had credible execution guarantees and bounded evidence of improved arrangement, but no completed broad playability acceptance.
The restored checkpoint still produces rapid repeated heads and excess short LNs on real timing; its published status explicitly requires quality review.
The evidence also contradicts the stronger accusation that timing sensitivity and transfer risks were never questioned: both were investigated early in the audio lineage.
These are review judgments grounded in the graded evidence below, not claims that every generated chart is poor.

Review begun 2026-09-30 12:58 UTC on bings-mac. Scope is the pre-loss R1, staged restoration, separate 35M teacher, and the first R1 transfer/sensitivity checks.
No training, model inference, accelerator use, downloads, Git mutations, private wording, or subagents were used.
CPU computations inspect saved charts and JSON, with a two-thread ceiling. No source artifacts were edited.

Source abbreviations below expand to these exact locations:

| Alias | Source and inspection |
| --- | --- |
| `M` | `/tmp/lineage-review/trees/main`; baseline `5c56e28`. |
| `B` | `M/src/ensomi_model/research/bounded_typed_continuation`; inspected contract, support, data, corpus, features, model, generation, consequence, response, recovery, routing, verification and nearby support tests. |
| `D` | `M/docs/research`; read bounded-typed results, staged restoration, training distribution and vacation queue; skimmed oracle-time continuation, expert question and source-action stage 2. |
| `J` | `/tmp/lineage-review/trees/audio-joint`; transfer audit and relevant initialization code, early audio research account. |
| `N` | `/tmp/lineage-review/trees/agent-notes/artifacts/agent-notes/proposed`; read relevant 2026-09-20 long-form, recovery, stage assessment and teacher notes selectively. |
| `$ASSETS` | `Documents/Pulsefield/research-assets`, relative to the user's home. External files were read-only. `$READOUT` means `$ASSETS/r1-restoration-20260920-v1/run/response/readout`. |
| `S` | `/tmp/lineage-review/01-r1-foundation`, containing this review's scripts and computed receipts. |

`checked-code` means implementation read with a cited file/line; `checked-artifact` means a saved output/log/metadata or fresh artifact recount was read; `doc-claim` means the underlying historical result was not independently recovered.
An old report's claimed human-calibrated agent judgment remains a doc-claim, not a new human verdict.
Sources also include `.git/research-relay/notes/RESEARCH.md` and the paraphrased feedback index, sections 2–3; no private links were followed.
Read-only Git logs and the named archive tags establish commit chronology; they do not establish experimental success.

The historical `artifacts/bounded-typed-continuation/` owners cited by the old documents were absent at the checked paths.
This prevents independent reconstruction of most pre-loss comparisons and the original corpus census.
Surviving primary evidence includes the restoration ledger, six published example outputs, 16 restoration validation outputs, teacher training/results/readouts, and the September 24 transfer audit JSON.
`artifacts/hf-pulsefield-r1-restored/release.json` names checkpoint `4b3ec156…` and `quality_status=requires_longform_ln_tap_and_local_response_review` [checked-artifact].

## 2. Attempts

**Q1 — The actual inference contract.** R is the sorted union of all source head and LN-release times, including release-only times.
H is a Boolean flag on R, true exactly where at least one source TAP/LN head occurs; its construction reads every source row, not audio [checked-code `B/data.py:43–68`].
The seed consists of complete initial rows reaching at least 30 heads; chords can make that number exceed 30.
Typed seeds additionally supply the exact future endpoints of holds still open at the seed boundary [checked-code `B/contract.py:86–110`; `B/data.py:45–49,116–118`].
At H, at least one head is mandatory; outside H, every head is forbidden and the row may be empty.
Nothing in H specifies head cardinality, head lanes or TAP versus LN. New LN endpoints are unknown until R1 chooses a release; release-only source times are opportunities, not obligatory releases or preserved pairings [checked-code `B/contract.py:137–162`; `B/data.py:116–129`].
Thus R supplies the permissible endpoints and many useful articulation opportunities, but does **not** supply the lengths of suffix-born LNs.
A generated LN can end at a different source head/release time, and unused release-only candidates can disappear.

R1 also sees much more than the next timestamp: 16 future candidate offsets/gaps and roles, timing counts over 0.25/1/4/16/64 seconds, future long-gap descriptors, remaining R/H counts and relative schedule position [checked-code `B/features.py:135–166`].
Exact occupancy and action clocks, a 511-row local representation, persistent observed-seed conditioning, and landmark memory condition its row scores.
The final policy adds head-mask, release-mask and candidate-consequence residuals, masks illegal rows, and samples at temperature one; it does not roll out competing learned futures [checked-code `B/model.py:267–304`; `B/generation.py:226–275`; `B/consequence.py:26–85`].
No audio, requested star rating, requested LN fraction or validated style control belongs to this baseline contract.

**Q2 — How much is already decided?** H fixes the exact onset rhythm, silence between head events and number of head-bearing instants.
R fixes the available event grid and terminal closure; the seed fixes the introduction and crossing holds.
With four free lanes at an ordinary H there are 80 nonempty choices from EMPTY/TAP/LN_START; often one all-LN choice is removed because the next H would have no free lane.
An occupied lane can only hold or close, and cannot close and restart at the same instant. A release-only candidate with no holds has one legal choice, the empty row.
These restrictions are checked in `B/support.py:43–55` and independently expressed in `B/contract.py:137–162` [checked-code].

A deterministic sample of 100 indexed TRAIN charts yielded 85,176 post-seed source-history candidate states; zero sampled charts failed admission.
Selection: intersect `artifacts/indexes/beatmap_index_4k.parquet` with the TRAIN catalog, sort by SHA256 of `r1-support-20260930:` plus source SHA, take the first 100 valid charts.
Actual support masks were evaluated in batches and checked against scalar support at each chart's first/last suffix state [checked-artifact `S/support_metrics.json`; command in section 7].

| State population | n candidate states | Mean legal rows / 256 | Exactly one legal row | Observed legal-row counts |
| --- | ---: | ---: | ---: | --- |
| All suffix candidates | 85,176 | 70.756 | 43 (0.0505%) | 1, 2, 3, 4, 7, 8, 15, 16, 25, 26, 31, 32, 51, 52, 79, 80 |
| Required-head H | 82,021 | 73.325 | 1 (0.00122%) | 1, 3, 7, 8, 15, 16, 25, 26, 31, 32, 51, 52, 79, 80 |
| Release-only R | 3,155 | 3.965 | 42 (1.331%) | 1, 2, 4, 8, 15, 16 |

The dominant counts were 79 choices at 65,570 states and 51 choices at 11,317 states.
This is a support-size census of sampled **source histories**, not generated-history entropy or a percentage of musical authorship.
Most row content remains free; nevertheless all rhythmic head placement and much future timing context have already been authored.

**Q3 — Inventory of evaluations and how the conclusions changed.** The following includes every R1 quality/proxy comparison located in the prescribed documents and relevant notes.
Except where explicitly marked checked-artifact, historical numbers are doc-claims from `D/bounded_typed_continuation.md` at the stated section or `N` note.
Corpus runs use the eligible 11,563 TRAIN charts/3,169 groups; “coverage” below counts charts/groups actually encountered, not independent repetitions.
Unspecified optimizer steps, wall time, training initialization, star range or seed values are **not recorded in the inspected result**; a single initialization has no measured training-run variance.
Reported group-bootstrap intervals quantify sampled-group uncertainty, not uncertainty across independently trained models.

| Order / commit / problem | Hypothesis and intervention | Training and evaluation scale | Result, interpretation then, next action |
| --- | --- | --- | --- |
| 09-18, `a178bcf` / H,J | Bounded R0/R1/O1 can learn their likelihood factors. | R1 2,281,104 parameters; 16 TRAIN groups, 2,048 unique onsets; 128 updates/32,768 exposures, seed 171/shuffle 271, 65.25 s training. | R1 NLL 4.410→2.442 on the training slice. Positive-only LN gate failed; post-hoc proper binary scores showed type learning. Pipeline evidence, not quality acceptance. |
| 09-18, `21475e6` / A–C,H | Run those short-fit models freely. | 16 TRAIN charts per arm, seed 17; R1 generation 30.0 s; two selected visual scopes, source star range not recorded. | R1 LN share 21.36%; a seven-head 33–37 ms same-lane run occurred. Agent found independent LN organization in selected scopes; further corpus training followed. |
| 09-18/19, `1693d62` / A–C,H | More source exposure may resolve early instability; compare R0/O1. | 2M exposures, seed 171; 6,315 charts/3,078 groups covered. R1 1,501.43 s; 24 VAL groups × 3 generation seeds at 250k/1M/2M. | R1 pooled NLL 1.9158 at 2M; O1 1.9068; group difference CI included zero. R1 had 28 head pairs <40 ms; O1 498, mostly occupancy-related. No stable representational winner. |
| 09-19, `50dda55`→`23a91b0` / A–C,H | Continue unchanged R1 from 2M to 4M. | Two training seeds 171/172, same draws, 2.281M parameters; 8,799 charts/3,167 groups covered; extra ~25 min/model. 24 reused groups × 3 seeds/model, 144 outputs; 16 scoped comparisons. | NLL fell 4.70%/3.48%; head-pair counts 10→3 and 43→2. LN shares 25.34→11.86% and 53.86→10.60%; 12 visual ties, three limited 4M preferences, one 2M preference. Additional fitting helped, with retention questions. |
| 09-19, `ca28511` / A,B,H,K | Check 4M difficulty and long-form coverage on new groups. | 28 new VAL groups, seven/source band 2–3/3–4/4–5/5–6; two training initializations × 3 generation seeds =168 outputs; eval 2,056 s. | Seed-172 model failed output-band coverage and had 14 <40 ms head pairs in nine outputs. Only 14/60 primary scopes reviewed. Three-held-lane traps motivated consequences. |
| 09-19, `4a98c02`→`e525bda` / A,B,G,H | Add action-only versus exact next-H consequence residuals, against continued plain R1. | Same 4M parent, +500k exposures; 2.281M plus 26,912 parameters for either residual; 9,178 charts/3,168 groups covered. 28 reused groups × 3 seeds × 3 arms=252 outputs; eval 2,464 s; train steps/time not recorded here. | Frontier <40 ms union rate improved only 9.18% vs plain, below 25% gate; 90% CI included zero. All arms became much more LN-heavy. Failure did not establish that consequence features cannot work. |
| 09-19, `cefe3de` / B,H,J | Replace generated prefix with source prefix at fixed weights/timing. | Eight selected reused charts × 3 seeds × 2 histories; 512-H continuations, 48 windows; 168 s; no training. | First-bin LN-fraction error 0.351→0.116, 66.82% reduction; 90% paired group CI for difference excluded zero. Whole-state dependence established; memory versus occupancy versus intent not isolated. |
| 09-19, `a64ac0a`→`67107af` / B–E,H | Persistent observed seed versus zero-vector residual and unchanged model. | +500k to 5M, 661 updates/arm; 2.281M or 2.330M parameters; 9,533 charts/all 3,169 groups. Three arms train 1,696 s total; 28 reused groups × 3 seeds/arm; eval 2,341 s. | LN-proportion error improved 14.93% vs none, missing 20% gate; CI vs none included zero. Outputs 1.567–7.882 stars for observed; short-gap/structure concerns remained. Observed-seed branch nevertheless became the next parent. |
| 09-19, `342fa15` / A,B,G,H | Decompose short heads into inherited-state and current-choice burdens. | Replay 252 generated charts and 28 source references, 55.2 s; no training. | Roughly 47–48% of generated <40 ms heads were at states with no recovered lane; 36–39% exceeded the same-cardinality minimum. Source had six events, none state-forced. This was a bound using TAP alternatives, not an equal-style repair. |
| 09-20, `ed1b9b9` / A–D,H | Add full-history landmark memory. | 2,777,232 parameters; +1M exposures, 1,317 updates, 1,877.21 s; eight fixed VAL sources × seeds 17/23, 16 outputs, eval 275.20 s. | <40 ms union 152→69, but one 152-H run contained 143 quad TAP rows over ~18 s. Another locked activity under three holds. Memory candidate failed long-form gate; native recovery followed. |
| 09-20, `fe69830`→`004c98d` / A–C,H | Train all inherited weights with native anti-repetition preferences. | 80 queries/eight TRAIN groups, +250k exposures, 333 updates, 729.79 s; 2.777M parameters. Same eight VAL groups × two seeds; 16 outputs, 281.51 s. | Longest fixed-lane run 152→11; <40 ms events 69→3. Inspected independent LN relations disappeared. Retention gate failed; this branch was rejected. |
| 09-20, `8cf31e8`→`c1cd14c` / A–C,H | Freeze parent and train head-mask routing only. | +139,776 parameters (2.917M total); same 80/eight pool, +250k exposures, 333 updates/299.93 s. Sixteen reused outputs plus 16 fresh outputs from eight groups across 2–6 stars. | Reused run maximum 152→22; gap count 69→68. Fresh sample had 242 events vs 26 source events and a 26-H confinement under three fixed holds. At those states routing cannot alter release conditionals; release adapter followed. |
| 09-20, `1e5f5da`→`dfdbc75` / A,B,G,H | Train release-mask residual after targeted transition harvesting. | 3,056,784 parameters; 72 queries/four TRAIN groups from first adequate 103-group collection prefix; +250k exposures, 148 s. 16 VAL groups × seeds 17/23=32 outputs, 293 s; 27 visual contexts. | Three-hold maximum 26→4, fixed-lane maximum 26→14; selected LN relations retained. <40 ms union worsened 310→366; generated stars 2.137–6.717. Passed scoped retention/recovery, not general playability. A further eight-group/16-output confirmation fed the later 48-output comparison. |
| 09-20, `9da6725`→`db0a9b8` / A,B,G,H | Train frozen-parent frontier2 on composition-preserving response preferences plus source CE/KL. | 3,084,432 parameters; 92 queries/nine of 32 selected TRAIN groups; +250k exposures, 328 updates/235.35 s. 24 VAL groups × seeds 17/23=48 outputs, 461 s; 50 candidate/11 parent visual scopes. | <30 ms union 160→70; <40 ms 442→178; longest fixed-lane run 14→18. Generated stars 2.142–6.128. Working candidate retained as REFINE, with no player trial or full-chart human review. |
| 09-20, `N/2026-09-20-r1-stage-quality-assessment.md` / A–C,H,K | Confirm frozen response candidate on unused groups. | Eight further VAL groups, two per 2–6-star band, new seeds 31/47; 16 outputs/89 s; 24 fixed early/middle/late scopes. Combined with above: 32 groups/64 outputs. | Fresh sources had zero <20/30/40 ms events; outputs had 4/21/146. Longest fixed-lane run 20H; pooled 64-output star differences across two seeds median 0.224, maximum 0.840. Agent called long-form stage nearly complete while leaving playability gaps. |
| 09-20, `6d0837b` / A,G,H | Two-coefficient composition-conditional response calibration. | Implementation and engineering tests; no prepared TRAIN cache, coefficient fit or new generation recorded in `N/2026-09-20-r1-longform-structure.md:1645–1674`. | No measured quality gain; not part of restored R1. Proposed closing step was still unrun in the inspected handoff. |
| 09-20, `cdbc687` / H,J | Reconstruct staged R1 after asset loss. | Six stages, model seed 172; 6.75M exposures/8,931 updates; measured 3,379.47 s cumulative training, 4,567.60 s whole queue. Eight VAL groups × seeds 17/23 at each stage; six extra published examples at seed 17. | Completed method and exported bytes; final review status pending. New cohort cannot establish equivalence with historical 64 outputs. Detailed raw results below [checked-artifact]. |
| 09-20–21, separate teacher / A–C,H,J | Scale clean seed/memory R1 to 35,178,768 parameters for later distillation. | Engineering preflight: two four-crop cycles/device, CPU/MPS, plus CPU recovery checks. Actual CPU1 training seed 20260925 reached 30M exposures/39,694 updates, 85,204.51 s compute, 11,531 charts/3,169 groups covered; five eight-group/two-seed readouts. | Teacher stage completed despite queue-level incomplete status from audio anomalies. NLL improved; late native lane repetition worsened. No distillation result located [checked-artifact; Q5 below]. |
| 09-23, `ff6471c`, `068988e` / B,F,H | Change only supplied non-head opportunities for frozen R1. | 12 songs × two seeds=24 pairs/intervention; no fitting; native-ms rerun. | Every-fourth-gap insertion gave median LN-length ratio 0.686 and +25.6 pp LN share; every-gap insertion 0.375 and +68.6 pp. Timing features and RNG also changed. This actively tested, and weakened, simple plug-in timing compatibility [doc-claim `J/docs/research/audio_conditioned_choreography.md:101–113`]. |
| 09-24 transfer audit / F,H,J | Check actual restoration stages, omitted modules and neural seed/memory ablation. | Eight actual restoration groups × two seeds; zero seed/memory readouts only in 6.5M parent, 16 matched ablations; no fitting, 62.71 s. | Mean absolute LN-fraction change 6.88 pp, below declared 10 pp gate, but regional changes large; no uniform long-form deterioration. Transfer explicitly omitted frontier2 and did not preserve R1 policy [checked-code/checked-artifact below]. |

**Q6 — Restored versus lost R1.** `D/r1_staged_restoration.md:29–34` explicitly claims reconstruction of a method, not deleted weights, trajectories or quality.
The historical final checkpoint SHA is `195f1b01…` [doc-claim]; the released restored SHA is `4b3ec156…` [checked-artifact release manifest].
There is no paired same-condition comparison between those bytes. Original training coverage and stage recipes were reused; the fixed monitor was newly selected.
The restoration did reconstruct pools with 80/eight, 72/four and 92/nine queries/groups and frozen-parent audits, but matching counts do not prove matching native states or behavior [checked-artifact `$ASSETS/r1-restoration-20260920-v1/run/ledger.json`].
The exact original replay/quality equivalence question therefore remains open; calling the restored model “equivalent” is unsupported.

Restoration ledger counts and measured stage durations below come from `S/audit_assets.py`, n=6 stages; time is training start to evaluation start, excluding harvest/readout.
The validation diagnostic columns are the saved independent audit at `artifacts/joint-audio/20260924-r1-lineage-audit-v1/result.json`; final counts were independently recounted from saved `.osu` files.
Every stage uses 16 outputs from the same eight groups; NLL uses 24 fixed VAL windows/6,144 onsets, plus an equally sized TRAIN monitor.

| Stage | Parameters | New updates; train wall s | Cumulative chart/group coverage | VAL NLL/H | Mean chart LN share | <30 ms head/head; release/head | Max fixed-lane H run |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| Plain 4.5M | 2,281,104 | 5,958; 1,762.47 | 9,178/3,168 | 1.7715 | 43.12% | 6; 306 | 12 |
| Seed 5M | 2,330,384 | 661; 210.38 | 9,533/3,169 | 1.7652 | 31.81% | 3; 87 | 77 |
| Memory 6M | 2,777,232 | 1,317; 843.77 | 10,061/3,169 | 1.7273 | 26.35% | 2; 35 | 49 |
| Head routing 6.25M | 2,917,008 | 333; 169.09 | 10,176/3,169 | 1.7311 | 27.42% | 2; 68 | 31 |
| Release routing 6.5M | 3,056,784 | 334; 178.33 | 10,257/3,169 | 1.7290 | 27.67% | 2; 67 | 11 |
| Response 6.75M | 3,084,432 | 328; 216.66 | 10,338/3,169 | 1.7386 | 19.25% | 5; 7 | 10 |

The final correction sharply reduces release→head events while head→head events increase 2→5; it should not be described as uniformly improving rapid repeated presses.
No training-run replication or uncertainty interval exists for this restoration comparison. Sampling seeds are paired but sequential trajectories diverge after changed choices.

**Q4 — What the three machine preferences prefer.** They label candidate **actions at selected generated states**, not two complete trajectories judged by a player.
Head routing selects a lane present in at least 28 of the preceding 32 H rows when the corresponding source's maximum is at most 24; a 1,000 ms rest clears that witness.
Alternatives omit a core lane or release a blocking hold [checked-code `M/src/ensomi_model/research/r1_restore/harvest.py:32–69`; `B/recovery.py:23–38`].
Release routing selects three unchanged holds across 12 H, all heads on the remaining lane, while the source uses at least three head masks and no lane more than nine times.
Its preferred actions release a blocking lane; targeted sparse-to-dense selection was necessary after an ordinary 32-chart collection supplied zero qualifying queries [same code; doc-claim `N/2026-09-20-r1-longform-structure.md:827–850`].
Response preference counts heads with either a previous-head or previous-release gap <30 ms, including an optimistic minimum over the next two H.
That future assumes one TAP per H and earliest possible release of unknown holds; it does not preserve future chord/LN organization or sample likely continuations [checked-code `B/response.py:20–59`].
Among strictly cheaper legal alternatives, it minimizes head-count change, then LN-count change, then response cost, then action Hamming distance, retaining ties.
The loss increases total preferred mass, conditional on sampled/preferred composition families for response; it uses source CE alongside two native queries/update at weight 0.25 [checked-code `B/response.py:62–89`; `B/recovery.py:41–53,162–180`].
The response collector excludes intervals where the matched source has a <30 ms witness [checked-code `M/src/ensomi_model/research/r1_restore/harvest.py:72–101`].
Source contrast therefore validates selected discrepancy witnesses, not the universal preference ordering. Agent inspection used human source examples for calibration; no direct human labels on the candidate-action preferences, blinded generated preference trial, or calibrated player-response validation was located.

**Q5 — The 35M teacher finished, but did not establish better chart quality.** It was a clean larger R1 with observed seed and landmark memory, without the three correction modules, intended for subsequent distillation [doc-claim `N/2026-09-20-r1-teacher35m.md`; checked-artifact final run configuration].
The actual teacher ran from September 20 to 21, reached 30M onsets, and produced the surviving 425,898,941-byte checkpoint `ee4b2200…`.
Its teacher stage is `completed`; the enclosing vacation queue is `finished_with_incomplete_stages` because the audio stage is `completed_with_anomalies`, with independent stress disabled.
Wall time including teacher readouts/pauses was 94,095.50 s; cumulative training compute was 85,204.51 s. This is not the 23-second engineering preflight described in the old teacher note.
Evidence: `$ASSETS/teacher35m-20260920-v2/run/ledger.json`, `teacher/segment-00006/result.json` and `run-config.json`, summarized by `S/audit_assets.py` [checked-artifact].

| Teacher exposures | Same fixed VAL NLL/H | Outputs | Pooled LN/head | Head pairs <60 ms | Longest fixed-lane H run | Longest single-lane-only H run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 2.09337 | 16 | 2.07% | 59 | 30 | 3 |
| 5M | 1.74621 | 16 | 25.64% | 36 | 11 | 7 |
| 10M | 1.64413 | 16 | 22.10% | 86 | 16 | 10 |
| 20M | 1.61307 | 16 | 41.33% | 86 | 177 | 177 |
| 30M | 1.63466 | 16 | 28.77% | 695 | 354 | 62 |

These are fresh counts on 80 saved outputs, not newly generated samples [checked-artifact `S/teacher_audit.json`].
The 30M NLL is lower than restored R1's 1.73858 on the same windows, but capacity, initialization, exposure, training recipe and corrections differ together.
At 20M, First Storm seed 23 repeats only lane 1 for 177 successive H, 184514–209699 ms; the source's whole-suffix maximum is four.
At 30M, Youma Yakou seed 23 repeats the lane-2/3 pair at all 354 H from 185873–229905 ms; source maximum is three.
Neither run is forced by entering holds: there are none at either start, and no other lanes attack in those intervals [checked-artifact `.osu` witnesses; `S/supplement.json`].
Their generated/source stars are 3.809/3.084 and 5.431/3.182 respectively under the pinned calculator.
Thus more capacity/source fitting improved a proxy but did not remove the native repetition failure. This is one teacher trajectory and eight songs, not an architectural impossibility result.
No fitted student, paired distillation result or human teacher preference evaluation was found in this slice's sources.

**Q7 — Source versus restored output, measured directly.** Six published examples (one seed 17 each) and eight monitor charts (seeds 17/23) were recovered: 22 outputs, 14 distinct VAL source charts.
Every source was matched to its exact R/H and seed; the monitor sources also match presentation SHA. The six example titles were resolved through the local index, not guessed from generated metadata.
Examples span source 1.627–6.382 stars, so are not a 2–6-star acceptance panel; monitor sources span 2.125–5.768 stars and outputs 2.153–5.702.
The largest monitor mismatch is Youma Yakou seed 17: 3.182→4.560 stars. These are descriptive calculator values, not playability labels [checked-artifact `S/supplement.json`].

Measurement definitions: primary counts exclude heads at/before the last seed row; same-lane head gaps retain seed predecessors.
The gap denominator is heads having a prior same-lane head. LN statistics count suffix-born LNs only.
“Run” means consecutive distinct head-bearing rows containing a fixed lane; simultaneous chords create no arbitrary lane-order tie. A separate single-lane-only run is retained in JSON.
“Near” means a release strictly before the next strictly later head in any lane by at most 40 ms; simultaneous head/release is recorded separately, not treated as a positive gap.
Pooled source totals repeat each monitor source twice to match its two output seeds; this does not create 16 independent songs.
`S/analyze_r1.py` reads the original `.osu` files; `S/chart_metrics.json` contains exact input paths, SHA digests, full-chart and suffix metrics, gap/length quantiles and denominators.
No seed/output selection was made after seeing the numbers. The 16-output head/LN totals (38,967/7,753) independently match the September 24 audit.

| Chart / seed | Heads S→G | Minimum gap ms S→G | Gap % <40/<60/<100: S → G | Longest run S→G | LN/head % S→G | LN <100 ms % S→G | Median LN ms S→G | Near release % S→G |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| alone | 479 → 663 | 308 → 154 | 0.000/0.000/0.000 → 0.000/0.000/0.000 | 4 → 4 | 2.30 → 8.30 | 0.00 → 0.00 | 2466 → 309 | 0.00 → 0.00 |
| descent | 3252 → 3339 | 53 → 21 | 0.000/0.123/11.839 → 0.240/1.378/10.662 | 7 → 3 | 42.25 → 66.88 | 67.32 → 78.33 | 80 → 61 | 41.48 → 40.93 |
| epistrofi | 1417 → 1297 | 81 → 54 | 0.000/0.000/1.059 → 0.000/0.154/3.084 | 6 → 4 | 100.00 → 74.17 | 66.83 → 64.97 | 81 → 81 | 15.53 → 19.13 |
| lilith | 3092 → 2984 | 74 → 74 | 0.000/0.000/0.065 → 0.000/0.000/0.101 | 6 → 7 | 34.51 → 30.63 | 5.44 → 4.70 | 148 → 111 | 0.37 → 0.98 |
| odin | 2171 → 1998 | 61 → 62 | 0.000/0.000/6.264 → 0.000/0.000/4.955 | 5 → 4 | 0.00 → 1.45 | — → 65.52 | — → 46 | — → 0.00 |
| slash-dot-slash | 4989 → 5381 | 68 → 68 | 0.000/0.000/0.100 → 0.000/0.000/1.319 | 14 → 5 | 7.96 → 17.78 | 84.89 → 71.89 | 68 → 68 | 0.00 → 1.04 |
| val-08-s17 | 940 → 925 | 150 → 150 | 0.000/0.000/0.000 → 0.000/0.000/0.000 | 3 → 4 | 7.34 → 12.11 | 0.00 → 0.00 | 300 → 300 | 0.00 → 0.00 |
| val-08-s23 | 940 → 959 | 150 → 150 | 0.000/0.000/0.000 → 0.000/0.000/0.000 | 3 → 4 | 7.34 → 17.83 | 0.00 → 0.00 | 300 → 300 | 0.00 → 0.00 |
| val-09-s17 | 1210 → 1439 | 150 → 100 | 0.000/0.000/0.000 → 0.000/0.000/0.000 | 4 → 7 | 44.46 → 24.11 | 2.97 → 5.19 | 300 → 300 | 0.00 → 0.00 |
| val-09-s23 | 1210 → 1505 | 150 → 150 | 0.000/0.000/0.000 → 0.000/0.000/0.000 | 4 → 6 | 44.46 → 23.46 | 2.97 → 4.53 | 300 → 300 | 0.00 → 0.00 |
| val-10-s17 | 1811 → 1703 | 157 → 78 | 0.000/0.000/0.000 → 0.000/0.000/0.352 | 4 → 3 | 4.58 → 11.22 | 0.00 → 1.57 | 316 → 316 | 0.00 → 0.00 |
| val-10-s23 | 1811 → 1845 | 157 → 52 | 0.000/0.000/0.000 → 0.000/0.054/0.108 | 4 → 5 | 4.58 → 4.66 | 0.00 → 0.00 | 316 → 316 | 0.00 → 0.00 |
| val-11-s17 | 2388 → 3096 | 125 → 93 | 0.000/0.000/0.000 → 0.000/0.000/5.006 | 3 → 7 | 11.98 → 26.32 | 0.70 → 51.29 | 188 → 94 | 0.00 → 0.00 |
| val-11-s23 | 2388 → 3007 | 125 → 93 | 0.000/0.000/0.000 → 0.000/0.000/2.793 | 3 → 5 | 11.98 → 24.94 | 0.70 → 44.53 | 188 → 125 | 0.00 → 0.00 |
| val-12-s17 | 3460 → 3181 | 102 → 76 | 0.000/0.000/0.000 → 0.000/0.000/0.723 | 7 → 6 | 7.57 → 7.70 | 12.98 → 25.31 | 153 → 153 | 0.00 → 0.00 |
| val-12-s23 | 3460 → 3206 | 102 → 51 | 0.000/0.000/0.000 → 0.000/0.031/1.029 | 7 → 6 | 7.57 → 9.64 | 12.98 → 22.98 | 153 → 153 | 0.00 → 0.00 |
| val-13-s17 | 3357 → 3142 | 78 → 59 | 0.000/0.000/0.268 → 0.000/0.032/1.687 | 19 → 7 | 20.11 → 18.84 | 6.81 → 4.56 | 118 → 166 | 0.00 → 0.17 |
| val-13-s23 | 3357 → 3108 | 78 → 58 | 0.000/0.000/0.268 → 0.000/0.097/0.740 | 19 → 6 | 20.11 → 13.93 | 6.81 → 3.46 | 118 → 176 | 0.00 → 0.00 |
| val-14-s17 | 3170 → 2947 | 86 → 29 | 0.000/0.000/0.252 → 0.034/1.018/3.156 | 4 → 4 | 7.95 → 11.84 | 31.75 → 33.52 | 115 → 115 | 1.19 → 6.59 |
| val-14-s23 | 3170 → 2998 | 86 → 29 | 0.000/0.000/0.252 → 0.100/0.934/3.069 | 4 → 5 | 7.95 → 15.68 | 31.75 → 48.51 | 115 → 110.5 | 1.19 → 9.79 |
| val-15-s17 | 3562 → 2928 | 75 → 75 | 0.000/0.000/0.056 → 0.000/0.000/1.195 | 10 → 7 | 44.81 → 42.49 | 34.02 → 41.24 | 150 → 150 | 5.64 → 5.95 |
| val-15-s23 | 3562 → 2978 | 75 → 25 | 0.000/0.000/0.056 → 0.067/0.067/1.612 | 10 → 10 | 44.81 → 43.18 | 34.02 → 51.24 | 150 → 75 | 5.64 → 5.21 |

S=source, G=generated; — means no LN denominator, not zero. Each row is one source/seed pair.

| Pooled cohort / scope | n outputs | Heads S→G | Min gap ms S→G | Gap % <40/<60/<100: S → G | LN/head % S→G | LN <100 ms % S→G | Median LN ms S→G | Near release % S→G |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| examples/suffix | 6 | 15400 → 15662 | 53 → 21 | 0.000/0.026/3.526 → 0.051/0.306/3.633 | 27.70 → 32.88 | 53.14 → 60.66 | 84.5 → 80 | 18.61 → 21.69 |
| examples/full | 6 | 15583 → 15845 | 53 → 21 | 0.000/0.026/3.580 → 0.051/0.303/3.685 | 27.90 → 33.01 | 52.89 → 60.33 | 85 → 81 | 18.45 → 21.51 |
| validation/suffix | 16 | 39796 → 38967 | 75 → 25 | 0.000/0.000/0.095 → 0.015/0.169/1.660 | 18.90 → 19.90 | 19.17 → 32.00 | 150 → 150 | 2.47 → 2.72 |
| validation/full | 16 | 40284 → 39455 | 75 → 25 | 0.000/0.000/0.094 → 0.015/0.168/1.643 | 19.07 → 20.06 | 18.93 → 31.51 | 150 → 150 | 2.42 → 2.67 |

The six examples have 15,400 source versus 15,662 generated suffix head pairs; <40 ms counts are 0→8 and <60 ms 4→48.
The monitor has 39,796 matched-source versus 38,967 generated pairs; counts are 0→6 and 0→66 respectively.
Its <100 ms pair share rises from 0.095% to 1.660%; LN lengths <100 ms rise from 19.17% to 32.00%, although the pooled median stays 150 ms.
The clearest localized short-LN change is Youma Yakou: source 0.70% of LNs <100 ms, versus 51.29%/44.53% at seeds 17/23; median 188→94/125 ms.
By contrast, source Epistrofi already has 66.83% of its LNs <100 ms. A universal short-LN ban would reject a large part of that real source.
The positive-gap near-release rate changes little in the monitor (2.47→2.72%); simultaneous release/head share changes 67.38→75.51%.
There is evidence for A's rapid gaps and B's short-LN excess, but not a blanket confirmation of every later complaint.
Final restored long-lane runs are **shorter** than source maxima in both cohorts (14→7 examples, 19→10 monitor); these particular outputs do not establish stream-to-long-jack collapse.
C's ordinary-pattern/LN-plus-TAP organization is not decidable from these counts. No new semantic labels or playtesting were performed.

**Q8 — Training population and windows.** It is not established as a curated distribution of ordinary ranked 2–6-star arrangements.
`D/r1_training_distribution.md` reports a full census of 11,563 eligible TRAIN charts, 3,169 song groups and 10,735,674 suffix onsets; the original census artifacts are absent [doc-claim].
At 5M exposures, 9,533 charts and all groups were encountered, with 3,459,305 unique onsets; the restored ledger independently matches that coverage [checked-artifact].
Consumed-onset shares were 18.526% below 2 stars, 80.080% from 2–6, and 1.394% above 6; source range 0.670–8.778 [doc-claim].
Within 2–6, consumed exposure was 23.360/24.154/22.841/9.725% in 2–3/3–4/4–5/5–6 respectively; this is not a predominantly high-star training explanation.
Consumed heads were 17.957% LN; only 5.831% of exposure came from charts with ≥50% suffix LN heads, 1.271% from ≥75% LN.
The seed-to-quarter LN-share discrepancy averaged 16.01 percentage points under consumed-exposure weighting [doc-claim].
These are the original **5M** draw census, not silently a recomputed 6.75M restoration or 30M teacher distribution.

Groups are sampled uniformly, then charts uniformly within group; a horizon of 128 or 256 H is chosen, with 12.5% dedicated initial-suffix windows and otherwise a uniform valid interior start.
Short charts and exposure milestones truncate windows. Sampling is not uniform over notes or seconds [checked-code `B/corpus.py:25–77`].
Source CE is teacher-forced; full-history memory adds past-source encoding but does not make the supervised target a complete generated chart [checked-code `B/data.py:179–235`].
Admission pins TRAIN identity, seed eligibility and row cache, but contains no ranked-status, “ordinary organization” or 2–6-star filter [checked-code `B/corpus.py:81–116`].
The inspected catalog and index have no ranked-status column. **Ranked share: not recorded / could not determine from these sources.** This is not evidence that they are all unranked.
Human-table coverage was 149 TRAIN charts (1.289%); that audit joined source identities only, not semantic labels or generated preferences [doc-claim `D/r1_training_distribution.md`].

## 3. Commentary

The strong result is a viable conditional row-learning task with reliable state accounting, rather than a completed playable chart prior.
R/H remove head-time discovery, supply pauses and density changes, and constrain LN endpoints; R1 still has a substantial lane/type/cardinality problem to solve.
The support audit refutes “almost everything is forced,” while the timing feature code refutes “R1 merely gets the next onset.”
These distinctions matter for A–E: success on the easier conditional task cannot be credited as learning all rhythm, phrasing or audio correspondence.

The early exposure experiments were reasonable causal probes. Two initializations and matched draws provide stronger evidence than one tuned rollout, and the documents explicitly preserve failed gates.
The 2M→4M gains show that inadequate fitting mattered at that stage; the 4M→4.5M native LN surge shows that lower teacher-forced loss was not a monotone path to stable composition.
It would be unfair to dismiss source CE altogether, or to describe these short fits as a decisive architectural refutation.
Old numerical results remain doc-claims here because their exact artifact owners are missing.

Prefix replacement provides evidence of dependence on generated history **as a whole**. It simultaneously changes neural history, occupancy, clocks and composition.
It does not isolate a missing seed representation, long-memory deficit or individual causal variable.
Persistent seed and memory were plausible interventions, but their primary gates failed or new collapse appeared; carrying those modules forward was exploration, not a demonstrated quality improvement.
This is where an accumulated module stack could be mistaken for an accumulated set of established remedies (B–D,H,J).

Head routing identified a precise, real representational limit: when all legal actions have the same head mask, its score shift cannot change release probabilities.
Release routing was consequently aimed at a mechanism, not merely at a bad score. Frozen-parent conditional preservation is a useful guarantee **at the same state**, not over a changed future trajectory.
The observed 2→31→2 longest three-hold run across actual memory/head/release restoration stages makes that distinction concrete [checked-artifact September 24 audit].
The full-parameter recovery failure was appropriately rejected when fewer diagnostic failures came with loss of independent LN relations.

Response recovery is a defensible bounded symptom correction. Source contrast, composition-family normalization and KL anchoring reduce obvious ways to satisfy its loss by deleting all difficult content.
They do not validate its <30 ms union as a player response or its optimistic two-H future as the future R1 is likely to generate.
Head/head and release/head actions have different meanings; a single merged count can hide an increase in one behind a decrease in the other.
The restored result's 67→7 release/head improvement alongside 2→5 head/head worsening is exactly such a case (A,B,G,H).

The early collector failed when it equated emitted rows with all consumed timing candidates, despite optional empty R decisions.
The recorded repair retained every decision, required schedule completion and independent replay, and reran the collection; already-complete trajectories were byte-identical [doc-claim `N/2026-09-20-r1-longform-structure.md:363–405,820–835`; checked-code `B/recovery.py:87–123`].
I found no basis to invalidate the final published pool on that bug. It does show why a raw “completed generation” count needs its exact completion definition.

Restoration reproducibility is not historical behavioral equivalence. The six release examples demonstrate saved-output reproducibility according to `verification.json`; they are not a six-chart comparative quality result.
Publication and the human's request to tag the restored baseline establish adoption/provenance, not a new blinded preference test [doc-claim paraphrased feedback index, first timeline row].
The final monitor's aggregate LN amount is close to source while Youma Yakou has a large short-LN and difficulty change; global fraction agreement does not validate arrangement (B,C,H).

The completed teacher is particularly informative about the foundation assumption: its existence cannot be inferred from the old engineering note alone.
Its raw logs prove completion, and saved outputs show that better NLL can coexist with much stronger same-lane or repeated-pair persistence.
Its lack of recovery residuals and different training seed/exposure prevent attributing the problem specifically to width.
There is no demonstrated benefit to the released small model unless actual distillation/adoption evidence is recovered.

## 4. Direction

| Attempt group | Strongest case for the direction | Strongest case against / unresolved alternative | Call and confidence |
| --- | --- | --- | --- |
| Typed R/H continuation (Q1–Q3) | Separates row organization from timing; eliminates R0's requirement to invent a head at every source release when no hold exists; exact supports make failures interpretable. | Supplied rhythm, end time and release opportunities substantially simplify the target; no generic timing distribution or audio-free BOS competence follows. | Sound research isolation and conditional baseline; insufficient evidence of a sound deployable row prior. **High**. |
| Additional CE and seed/memory | Broad corpus exposure improved NLL and some burdens; two initialization comparisons exist; old-history information can plausibly help form. | Primary seed/consequence gates failed, memory collapsed, and architecture was bundled with additional fitting. Source intro may not represent later arrangement. | Worthwhile bounded hypotheses; cumulative quality adoption not justified by those tests alone. **High** about limits, **medium** about mechanism. |
| Frozen residual corrections (Q4) | Targets actual native-state failures, preserves conditional distributions, uses TRAIN-only preferences and real-source contrasts; final short-gap and occupancy diagnostics improved. | Small pools of 4–9 contributing groups, hand-set predicates, optimistic future and narrow score do not define physical/player or ordinary-pattern quality. | Reasonable repairs of identified failure modes, not evidence that the whole ordinary-chart problem is solved. **High**. |
| Staged restoration (Q6) | Rebuilds auditable module order, source draws, pools and freeze guarantees; yields a usable pinned baseline. | Deleted original weights/cohorts leave equivalence untested; quality status was never closed by the worker. | Restoration succeeded computationally, quality equivalence remains unknown. **High**. |
| 35M teacher (Q5) | Capacity/optimization scale was tested on the actual machine; 30M training completed and likelihood improved. | Single run, unmatched recipe and severe saved native recurrence; no teacher preference or distillation benefit. | Engineering success, no established quality or downstream gain. **High** for observed results; **low** for architecture-wide inference. |
| First audio transfer (Q9) | Timing-support sensitivity, integer-ms repair and the later parameter audit explicitly challenge plug-in reuse; reused weights are a reasonable initialization. | Whole seed, memory and frontier2 behavior was not preserved, and original quality was already unresolved. | Treat as a different research model; original R1 quality cannot certify it. **High**. |

## 5. What was overlooked or never questioned

**Q9 — Properties carried forward without the needed check, ranked by consequence.** “Unchecked” below means the relevant claim lacked a sufficient check, not that nobody wrote a caveat.

1. **A completed ordinary/playable row prior.** `N/2026-09-20-r1-stage-quality-assessment.md` calls the long-form stage nearly complete and describes remaining gaps as relatively local, based on 64 development outputs and scoped agent judgments.
   The released artifact still carries a pending-quality status; the new recount finds rapid gaps and large source-relative short-LN changes. No broad human acceptance of restored outputs was located (A–C,H,K; doc-claim plus checked-artifact).
2. **Behavioral continuity after restoration.** Historical 48+16 response results concern `195f1b01…`, not restored `4b3ec156…`.
   The baseline adoption sequence in the feedback index does not supply a same-condition old/new comparison; the restoration document explicitly leaves it open. The missing check is functional equivalence, not whether stages finished (H,J; checked-artifact/doc-claim).
3. **Response preferences as a sufficient acceptance signal.** The three learned repairs optimize a handful of recurrence/short-gap predicates, while `B/response.py:20–89` supplies no validated player-response target.
   The historical stage assessment's proposed two-parameter calibration treats residual local burden as the next closing step. It does not test whether C's ordinary LN-plus-TAP relations or D's pressure development are captured by that target (G,H; checked-code/doc-claim).
4. **The seed/corpus distribution as “ordinary ranked 2–6 star.”** `B/corpus.py:81–116` has no such admission or weighting contract; the old distribution census does not report ranked share or ordinary-pattern labels.
   It measures mostly 2–6-star exposure, which is evidence against a predominantly high-star explanation, but not evidence of ordinary arrangement distribution (C,K; checked-code/doc-claim).
5. **Whole-chart semantic stability from sparse scoped inspections.** The historical 4M coverage screen reviewed 14 of 60 primary scopes; later reviews increased scope coverage but remained agent judgments on selected contexts.
   Neither completing 17.5 minutes nor getting a typical aggregate LN fraction measures the uninspected remainder. Restored monitor songs are approximately three to five minutes, not a repeat of the original longest-chart evidence (C,D,H; doc-claim/checked-artifact).
6. **A generally improved model from lower source NLL.** The teacher's fixed eight-song readouts now supply a concrete counterexample: 20M/30M produce long unforced runs despite lower NLL than small restored R1.
   Capacity, fit duration and policy corrections were not separated, and no quality evidence was found for distillation into R1 (A,H,J; checked-artifact).

Two alleged assumptions deserve an explicit correction rather than inclusion in that list.
**Timing-support invariance was tested:** the September 23 insertion experiments found large LN-duration/amount changes; source R/H were never empirically interchangeable with arbitrary predicted candidates.
**Transfer parity was later explicitly rejected:** `J/src/ensomi_model/research/joint_audio_continuation/model.py:215–252` omits seed, landmark and consequence modules, slices the exact projection and documents a different model [checked-code].
The audit reports 2,444,688 copied parameters, 523,776 omitted-module parameters and 115,968 discarded projection parameters; released 6.75M and release 6.5M yield identical transferred tensors [checked-artifact `artifacts/joint-audio/20260924-r1-lineage-audit-v1/result.json`].
The correct criticism is failure to establish a new row-policy acceptance basis, not a claim that these differences were never noticed.

Concrete missing measurements: generated-chart human preference/play trials; calibration of preference labels against physical response; ranked and ordinary-pattern coverage; old/restored same-condition equivalence; broad restored whole-chart/seed generalization; and teacher-to-student benefit.
A definition of acceptable LN duration/placement at matched ordinary difficulty was not fixed by R1's short-gap counters. Source Epistrofi's genuine short-LN distribution shows why that omission matters.
These are bounded evidence gaps, not a proposed architecture or roadmap.

## 6. Worth keeping

- **The exact conditional task and executable counterfactuals.** `B/contract.py:137–165`, `B/support.py:21–57` and `B/verification.py:20–89` make legality, future-head room and endpoint visibility inspectable [checked-code]. The 100-chart recount confirms substantial remaining choice; the mask is a constraint, not a playability evaluator.
- **A pinned small-model baseline with honest scope.** Published checkpoint/hash, conditions, decisions, rows and `.osu` outputs survive. Six published files match the reconstruction output hashes, and the released checkpoint matches its manifest [checked-artifact]; original real timing must remain part of its stated task.
- **Native-state failure attribution.** The head-mask conditional invariance and its inability to release a blocking hold are exact mechanisms. Separating inherited occupancy from immediate avoidable choices remains useful even when the <30/40 ms predicate is only a diagnostic [checked-code `B/routing.py`; `B/response.py`; historical scope doc-claim].
- **Failure-preserving records.** Rejected full-parameter recovery, failed seed/consequence gates and the explicit REFINE disposition preserve evidence against optimistic interpretation [doc-claim]. They should survive as negative results, not become silent endorsements of the final module stack.
- **Surviving teacher and restoration readouts.** These are concrete counterexamples to equating source likelihood or global LN fraction with native arrangement stability [checked-artifact]. The teacher is a useful scale/trajectory comparison even though no distillation benefit is established.
- **Recountable output diagnostics.** `S/chart_metrics.json`, `support_metrics.json`, `teacher_audit.json`, `restoration_audit.json` and `supplement.json` retain paths, denominators, scopes and witnesses. They diagnose gaps and allocation; they do not replace an ordinary-pattern or player judgment.

## 7. Claims worth re-verifying

**Historical response quality and restored equivalence.** Recover original `artifacts/bounded-typed-continuation/row-response-recovery-20260920-v1/response-6750k/checkpoint.pt` (SHA `195f1b0109696302addc1aa62bf896ca399d2a1656c15ba4fe27b4a47c8171ce`), its `evaluation-v1/readout.json`, and `row-response-confirmation-20260920-v1/evaluation-v1/readout.json` / `semantic-review.json`.
These are the missing sources for the historical 48+16 claims; the restored checkpoint and new eight-song monitor cannot substitute for them. No model rerun was attempted.

**Original training-distribution census.** Missing owner: `artifacts/bounded-typed-continuation/training-distribution-20260920-v1/`, specifically `results-v1/readout.json`, `results-v1/charts.jsonl` and `independent-audit-v1/audit.json`.
Their recorded hashes begin `c4f93fe8`, `bd77030a` and `c5076a25`. The reported full recount took 405.99 s and source census 783.14 s, beyond this review's five-minute command ceiling; neither was repeated.
Ranked status needs a source-identity-joined status record: neither these documented quantities nor the inspected parquet schema supplies it.

**Teacher benefit outside this monitor.** Exact surviving owner: `$ASSETS/teacher35m-20260920-v2/run/teacher/`, `segment-00006/result.json` and `readout-{1000000,5000000,10000000,20000000,30000000}/`.
The 30M checkpoint survives; no student artifact or matched human quality comparison was located. A broader conclusion needs the missing evaluation, not merely proof that training finished.
The final teacher and restored likelihood files have identical 48 source/window descriptors (24 TRAIN, 24 VAL); the NLL comparison is on matched windows despite unmatched training recipes.

All new computed numbers in this report are reproducible from the three CPU analysis scripts below; run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/01-r1-foundation/analyze_r1.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/01-r1-foundation/audit_assets.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/01-r1-foundation/supplement.py
```

`analyze_r1.py`: inputs are the catalog/index named in section 2, six reconstruction run directories, `$READOUT/native/validation/*`, and the 14 corresponding source `.osu` paths recorded in JSON; n=22 output pairs plus 100 TRAIN charts. The initial full analysis took 3.73 s.
`audit_assets.py`: inputs are the two external ledgers, teacher segment results and 80 completed teacher `.osu` outputs across five readouts; n=6 restoration stages and 5 teacher checkpoints. It follows the completed resumed segment at 10M rather than omitting that case.
`supplement.py`: inputs are the 22 source/output pairs and two identified teacher witnesses; computes stars with `M/src/ensomi_model/osu_core/difficulty.py`, native 4K, no mods, rate 1, retaining original hit-object file order. It also records the long-run endpoints, lane counts and entering holds.
No command samples a model, changes evidence inputs, or writes outside `S`.

## 8. Cross-slice notes

Timing work should distinguish absence of audio alignment from the substantial real-chart rhythm information already supplied to R1. The early non-head insertion result is relevant to release modeling, but was not re-run here.
The transfer branch drops modules and whole-future timing inputs; this review does not assign a causal percentage of later audio failures to those omissions.
The formulation's target-response outputs/scales remain undefined (`M/docs/formulation/gameplay-state.md:44–55`), while `frontier2` is a learned short-horizon row residual. The name does not establish the formulation's gameplay frontier [checked-code/doc-claim].
Strict next-event sampling is an implementation choice; `M/docs/formulation/notation.md:239–277` permits provisional branches and prefix commit. This observation does not prescribe a replacement architecture.
Requested difficulty/style/LN controls (I), audio correspondence (E), and the later ordinary-expert/red-evaluation work belong to other slices; R1's source-star coverage is not evidence that those controls existed.

## 9. Failed paths and unfinished work

All nine slice questions are answered, including explicit uncertainty where primary evidence is absent. The requested source/generated and support-count CPU analyses completed.
Historical artifact owners were absent at their documented repository paths; it cannot be established from this review whether each was lost before restoration or removed in the September 30 cleanup.
An initial title lookup missed “Slash Dot Slash (Slim Boy Fat)”; exact R/H/seed matching resolved it. All 22 source identities subsequently matched the catalog, and the six example outputs matched their published copies.
The star parser rejects minimal generated headers lacking AudioFilename. The analysis instead constructs the same raw hit objects in original file order and calls the pinned calculator; it does not edit charts or invent audio.
One teacher 10M case completed in a second segment; the initial segment-only count of 15 was corrected to all 16, giving 80 total teacher outputs. The final receipt includes the completed segments only.
PyArrow emitted sandbox CPU-cache discovery warnings but completed its reads. No package installation or permission escalation was needed.
No new Lens semantic inspection, listening, player test, corpus-wide ranked census, exhaustive old-note history search, model inference or training was performed.
Therefore C's ordinary-pattern judgment, historical/restored quality equivalence, direct preference validity and unobserved teacher/student results remain unresolved, with missing sources identified above.

## 10. Inconsistencies and items for the human

1. **Method restoration versus quality restoration.** The restoration document explicitly disclaims equivalence, yet historical candidate quality can easily be attached to the restored R1 name. Those are different checkpoint identities and different evaluation cohorts [doc-claim plus checked-artifact; section 2/Q6].
2. **“Teacher unfinished” versus completed training.** The vacation queue's incomplete status does not mean the teacher failed to reach 30M: the teacher stage completed, while audio reported anomalies. The raw ledger and `M/src/ensomi_model/research/vacation_training/run.py:185–187` resolve the apparent contradiction [checked-artifact/checked-code].
3. **Short-gap improvement versus repeated-press improvement.** Restored response reduces release→head <30 ms from 67 to 7 but increases head→head from 2 to 5; describing only the union conceals the distinction. These cannot be treated as interchangeable player costs without an independently specified response target [checked-artifact/checked-code].
4. **Stable overall LN amount versus changed local organization.** The monitor's source/generated pooled LN shares are 18.90/19.90%, while Youma Yakou's short-LN distribution changes sharply. The saved audit's generated 19.25% is a chart-mean, not a conflicting pooled estimate [checked-artifact].
5. **“R1 backend” versus changed conditional model.** The flat joint initialization drops seed, memory and frontier2, and 6.5M/6.75M yield identical transferred weights. Behavioral continuity is explicitly disclaimed by the code and audit; the baseline's acceptance cannot be inherited by naming [checked-code/checked-artifact].
6. **Which real examples count as ordinary?** Epistrofi's source already has mostly sub-100 ms LNs, and the six published examples include sources outside 2–6 stars. Whether these are expressive references or ordinary-target exemplars requires the human's intended standard; this report does not infer it from chart legality or stars [checked-artifact].
7. **Publication versus playability acceptance.** The paraphrased index records a request to restore, render, publish and tag R1; the release still declares pending quality review. Whether there was an additional unrecorded play test or quality acceptance can only be settled from that evidence, not inferred from publication [doc-claim/checked-artifact].

Review closed 2026-09-30 13:22 UTC; no further work was started after report completion.
