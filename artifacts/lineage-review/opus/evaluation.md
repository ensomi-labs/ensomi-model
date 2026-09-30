# Lineage review, slice 07 (independent): evaluation

Claude Opus reviewer, control plane, 2026-09-30. Not anchored on the Astra reports: nothing under `artifacts/reports/lineage-review/` or `.sync/*/jobs/*lr-*` was read.

Evidence grades used below: **doc-claim** (a lineage document or commit says so), **checked-code** (read in the `audio-joint-2026-09` export), **checked-artifact** (read from, or recomputed on, mac artifacts). Numbers I computed carry command, inputs and n.

## 1. What I read and checked

- Notes: `RESEARCH.md`, `lineage-review/README.md`, feedback index sections 2-5.
- Docs (audio-joint tree): `gameplay_regression_evaluation.md`, `fresh_audio_system_evaluation.md`, `ranked_2to6_action_reference.md`, `ordinary_expert_from_scratch_zh.md`, `head_audio_control_interaction.md`, `sustained_response_planning.md` (head), formulation `gameplay-state.md` ("Target response and frontier", "Evaluation questions").
- Code: `gameplay_evaluation/{qualification,qualification_config,ln_fragmentation,tap_organization,rhythm_lattice,head_recurrence,report,witnesses,temporal}.py`, `player_response/envelope.py`, `controlled_audio_continuation/outcomes.py:scoped_difficulty`.
- Mac artifacts: `20260927-gameplay-evaluation-v1` (positive guards), `20260928-response-blindspot-v1` (probe, quality replay), `20260928-coordination-corpus-v1/scan-v1` (6,924-chart ranked scan), `20260927-sustained-response-planning-v1/calibration.json` (envelope), qualification `cases.json` of `20260928-head-audio-control-interaction-v1` and `20260928-clean-joint-proposal-v1`, lens script `render_review.py`.
- Git (read-only): commit history of the evaluator files.
- Not read: `native_pattern_failure_analysis_zh.md` (1,513 lines) beyond what other docs cite; most of the 65 lineage docs.

## 2. Evaluator inventory

"Validated vs ranked" means a false-positive rate on real ranked charts was measured; "vs human" means agreement with the human's judgments was measured. Neither column was ever filled by a labelled human set; see F1.

| Evaluator | Measures | Threshold and its origin | Validated vs ranked | Validated vs human | Used to claim |
| --- | --- | --- | --- | --- | --- |
| Legality, replay, export/reparse | Mechanical contract | Exact | Yes (by construction) | n/a | 09-23 "48 charts pass" (human reset to playability, V12) |
| Teacher-forced NLL (H NLL/s, row NLL, release Brier) | Fit to source paths | None | n/a | No | Arm parity 09-24; H-only fit gains 09-28; scratch expert 31→10.7 nats/s |
| Whole-chart SR `compute_mania_star_rating_20241007` | Official difficulty | Request ±1.0 star (`QualificationConfig.difficulty_error_limit`, a startup default with no stated source) | Yes, calculator agrees with official ratings (F7) | No | Star MAE per arm, "within 2-6★" |
| `scoped_difficulty` proxy | Same strain model on a scope | ±1.0 | Agrees with SR on whole charts (F7) | No | Scoped control claims |
| H-only star lower bound | Timing feasibility | Proof-based | Math, not data | n/a | "H must be part of the repair" (sound) |
| Same-column attack < 20 ms | Impossible jacks | 20 ms from the 2-6★ census minimum and the human's "under 20 ms always bad" (V33) | Census: 0 of 8,774 ranked ≤ 20 ms (doc-claim) | No | "0 short attacks" in every qualification |
| Sustained attack envelope excess (`player_response/envelope.py`) | Per-column attack rate vs q99 of ranked TRAIN maxima per star band | q99, song-weighted; 3 ranked positive guards | In-sample by construction; 3 guards | No | Planner excess .0275→.0004 (also its selection objective, F4) |
| Action-response "work" budgets (excess, quadratic potentials) | Recovery work over 0.5-8 s | q99 of ranked; ≤ 2.5% held-out rejection declared | Yes, held-out 19-22/1,368 and 15-17/1,368 rejected (checked-artifact) | No; sensitivity only on 16 agent-labelled charts | Frontier acceptance 09-28 |
| Short-LN prevalence (burden ≤ 80 ms, later ≤ 40 ms) | Short LN heads / all heads, whole chart | q99 of ranked 3.5-4.5★ burden per LN band | Yes, held-out 1-7 of 145-208 (checked-artifact) | No | "Short-LN metric improved" (V75-78) |
| LN amount error | Realized vs requested whole-scope LN head fraction | ±0.10, startup default | n/a (control) | No | Control claims |
| Profile descriptor MAE (H/s, heads/H, LN fraction) | Chart-level control | None | No | No | Fresh-audio control section |
| Agent Lens reading ("reading receipts") | Agent's visual judgment of rendered pages | None | No | No, and V68 shows divergence | "Agent-confirmed regression witness", "209 pages read" |
| Audio correspondence (CKA vs circular shifts), rhythm lattice, TAP organization, head recurrence, hold interactions | Descriptors | None, explicitly "not a BAD rule" | No | No | Diagnostics only (the docs say so; checked-code) |
| Publication, startup, service | Latency | 2 s | n/a | n/a | Realtime claims (measured, fine) |

## 3. What I ran

Inputs and scripts: `/tmp/lineage-review/opus-evaluation/{manifest.py,battery.py,analyze.py,rows.jsonl}` on bings-mac; copies in this session's scratchpad `opus-evaluation/`. Command: `PYTHONPATH=/tmp/lineage-review/trees/audio-joint/src:. ~/ensomi/ensomi-model/.venv/bin/python battery.py 0 411` then `analyze.py`; one process, one thread, 21.5 s CPU. No model was loaded.

Charts (n = 411):
- `ranked`: 200 charts at official 3.5-4.5★, one per song group, sampled with `random.Random(20260930)` from the lineage's own 6,924-chart ranked scan. 42 of them are in the lineage's held-out groups; the rest were in the fitting pool of the q99 references, so their false-positive rate is in-sample.
- `V68-lens-set`: the 12 generated charts behind the 40 Lens pages the human reacted to in V68 (additive/modulated × Classic Pursuit, STYX Helix, Blizzard Heights × 2 seeds; identified from `render_review.py`), plus their 3 ranked sources.
- Other charts from runs the human rejected in aggregate: head-audio other cases (42), clean joint proposal native 512/2048/4096 × early/fresh/inherited (123), ordinary scratch (12), action-segment R1 (6), LN repair (6). The human did not look at each of these.
- `R1-reconstruction`: the 7 restored-R1 real-time reconstructions of 09-21 (osu copies given an `AudioFilename` line so the parser accepts them).

Each lineage evaluator run with its own code and fitted reference: the qualification gates (below-20 ms, |SR − request| > 1, |LN − request| > 0.1), short-LN prevalence at 40 and 80 ms (references refitted with the lineage's `fit_fragmentation_reference` on the scan's non-held-out charts; they equal the quality-replay references where both exist), envelope excess > 0 at the requested stars, work budgets at 4 s (the probe's own decision window), and the 4 s and 8 s single-column counts > 23 and > 43 from the ordinary-expert doc. Bracketed columns are my reference descriptors, not lineage evaluators. Cell = charts flagged.

| Group | n | Qualification gates fail | shortLN40 | shortLN80 | env>0 | work 4 s (ex/quad) | col4s>23 | Any lineage evaluator | [any LN ≤ 40 ms] | [same-lane gap < 60 ms] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ranked 3.5-4.5★ | 200 | 0 | 1 | 1 | 1 | 0 / 1 | 0 | 5 | 2 | 0 |
| ranked, held-out groups | 42 | 0 | 0 | 0 | 0 | 0 / 1 | 0 | 1 | 0 | 0 |
| V68 sources | 3 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| **V68 lens set (rejected)** | 12 | 5 (all LN amount) | 6 | 0 | 0 | 4 / 4 | 0 | **7** | 12 | 0 |
| head-audio other | 42 | 13 | 20 | 0 | 23 | 15 / 11 | 24 | 38 | 38 | 0 |
| clean joint | 123 | 83 | 74 | 23 | 54 | 64 / 66 | 49 | 116 | 107 | 0 |
| ordinary scratch | 12 | 5 | 9 | 0 | 8 | 9 / 8 | 3 | 12 | 11 | 7 |
| action-segment | 6 | 2 | 6 | 0 | 5 | 1 / 0 | 2 | 6 | 6 | 0 |
| LN repair | 6 | 2 | 1 | 0 | 2 | 4 / 3 | 2 | 4 | 5 | 0 |
| R1 reconstructions | 7 | n/a | 3 | 2 | 1 | 2 / 2 | 3 | 5 | 4 | 2 |

Other numbers from the same run (checked-artifact):
- Ranked 3.5-4.5★: minimum same-lane attack gap over all 200 charts is 70 ms (p5 of per-chart minima 77 ms); maximum 4 s single-column count 23.
- Recomputed SR equals the scan's stars within 4.6e-6 (n = 203); the scan itself recomputed SR and matched the official metadata within 5.2e-6 on all 6,924 charts (`stars` vs `metadata_stars` in `scan-v1/charts.jsonl`).
- Inside the 6 s Lens windows the human saw in V68: the STYX source window has 65 heads and 0 LN; the four generated STYX windows have 27-47 LN out of 39-62 heads, while their whole-chart LN fraction (0.50-0.52) matches the request (0.49) and passes the LN-amount gate. The Classic Pursuit source window itself holds 11 LN ≤ 80 ms (median LN 54 ms); the generated windows hold 0-2.
- Seed noise (HACI additive/modulated, 24 same-arm s0/s1 pairs): mean |ΔSR| = 0.223, so per-chart σ ≈ 0.20 star; standard error of a 4-case arm mean ≈ 0.10, of a difference of two such means ≈ 0.14 (unpaired). Across all 60 seed pairs in HACI and clean joint: mean |ΔSR| 0.275, mean |ΔLN fraction| 0.091, mean |Δ LN ≤ 40 count| 12.0.

## 4. Findings

**F1. No evaluator that carried a claim was ever validated against the human's judgment; only its false-positive rate on ranked charts was controlled, and that is 1% by construction.** Every thresholded evaluator is a q99 of ranked maxima or a fixed default, so it flags about 1% of ranked charts: 5 of 200 in my run for all eleven together, 1 of 42 on held-out groups (checked-artifact). What was never measured, before the 09-28 blind-spot probe, is sensitivity: how many charts the human rejected each evaluator catches. On the V68 charts, the only set I can tie to one human message, the numeric qualification gates in force at the time pass 7 of 12, and the other 5 fail only the LN-amount control gate, not a quality gate. All eleven lineage evaluators together, including those written after V68, still pass 5 of 12. The short-LN-80 evaluator introduced right after V68 (e1fecbd, 09-28 07:14Z) passes all 12. The blind-spot probe (`20260928-response-blindspot-v1`) is the lineage's own evidence of the same gap: the "excess" work potential gives a 23 ms LN zero cost and accepts 32 same-finger presses at 8 Hz (checked-artifact). Grade: checked-artifact. Confidence high.

**F2. The lineage evaluated whole-chart quantities, while the human judged local organization.** Whole-song LN fraction, whole-chart SR and whole-chart short-LN prevalence were gated; the human looked at 6-8 s windows. The STYX case is exact: the model hit the whole-song LN request (0.50-0.52 vs 0.49) by filling a passage that is pure TAP in the source with 27-47 LNs, so the gate passed on the property the human rejected. The lineage saw the point in prose (`full_row_learning_and_difficulty_response.md`: "whole-song quantity improvement does not establish a better local interpretation") but never made it a check. The agent's own diagnostic scored a 16 s single-column run at about 4.1★ (feedback index, problem A). Grade: checked-artifact for STYX, doc-claim for the SR example. Confidence high.

**F3. Each evaluator was written after a human complaint and calibrated on the failure that triggered it, so the final battery's separation is in-sample.** Timeline (git, UTC): envelope 09-26 18:16 (after the long-jack report at 16:59); temporal evaluation 09-27 03:47 (after V56); qualification 09-27 10:02; short-LN 80 ms 09-28 07:14 (after V68 at 02:29 and V70-74); action work 09-28 09:24-10:33; the 40 ms short-hold version 09-28 14:31. By the end the battery flags 116 of 123 clean-joint charts, but those failure families were the design input. No held-out set of failures, from new songs or from later human rejections, was ever kept. Each next human inspection found a mode the current battery missed: LN placement after the jack evaluators, 21-36 ms attacks after the 20 ms gate, 40 ms LNs after the 80 ms burden. This is the local-optimum pattern of the modelling work, repeated in the evaluation. Grade: checked-code/git for timing, checked-artifact for counts. Confidence medium-high; the "calibrated on the triggering failure" reading is mine.

**F4. Several thresholds came from the wrong population or had no source.**
- *20 ms gate.* It is the 2-6★ census minimum plus the human's "under 20 ms is essentially always bad". At the target 3.5-4.5★ the ranked minimum is 70 ms in my sample (n = 200) and 37-55 ms per band in the census (doc-claim). The gate therefore passed the 21-36 ms attacks the human then rejected (V37-38), and the ordinary-scratch charts with gaps of 20-40 ms (min 20 ms, 5 of 12 below 40 ms) still pass it. A 60 ms HH support mask was added later in generation, not in the gate.
- *±1.0★ and ±0.10 LN.* These are startup defaults in `qualification_config.py` with no stated derivation. ±1★ spans most of a band: a 4★ request passes anything from 3 to 5★.
- *q99 of a rare-event prevalence.* In a band where about 99% of charts have zero short LNs, the q99 of the burden lands on the few ranked outliers: 21.7% of heads for mixed-LN charts at 80 ms. At 80 ms the check passed charts with 31-68 LNs ≤ 40 ms (`ordinary_expert_from_scratch_zh.md` §4, §7; my run: shortLN80 flags 0 of 12 V68 and 0 of 12 scratch). A plain "any LN ≤ 40 ms" count flags 12 of 12 V68 charts against 2 of 200 ranked. I found that descriptor after the fact from the complaint wording, so it is not a validated alternative either: 2 ranked charts in my sample contain 6 and 12 such LNs, and the human calls short LN a style (V77).
Grade: checked-code and checked-artifact. Confidence high.

**F5. Optimized quantities were reported as improvements.** The sustained-attack excess was both the planner's selection cost and the reported gain (.0275 → .0004); the lineage doc says so itself ("The cost was optimized, so it cannot independently establish musical improvement"). The human then rejected the output as rigid, without breathing (V53-54). NLL improvements on factual windows went with worse native short tails (scratch expert; `ln_release_calibration.md` records better total row NLL with worse release cardinality). Grade: doc-claim, consistent with the code. Confidence medium-high.

**F6. Hygiene: one small development panel, drawn from TRAIN songs, reused for selection for four days, with seed noise never set against the criteria.**
- *Panel reuse.* Case directories named after Zenithfall appear in 40 of the 101 `joint-audio` experiment directories; STYX 16, Blizzard 17, Classic Pursuit 15, Max Burning 11 (`ls -d */*/*<song>*`, checked-artifact).
- *TRAIN songs.* The native panel sources carry `"split": "train"` in the plans (STYX checked). The calibration plan states "not unseen song generalization of the generator". The fresh-audio panel of 8 songs is the exception, curated and held out.
- *Seeds.* Most comparisons use 1-3 seeds, paired across arms. The per-chart seed σ of about 0.20★ (section 3) exceeds several declared criteria. For example, HACI's "D6 non-regression allowance .10" is set on a 4-case mean whose standard error is about 0.10; that test would fail often between two identical models. HACI's D4 MAE differences of .02-.05 over 19 cases are within noise; to its credit, the doc does not claim them.
- *Uncertainty statistics.* Bootstrap intervals appear in 13 baseline-era research docs and in 1 of the 65 docs the lineage added (`continuation_state_dependence.md`; the other hit is a module named `bootstrap.py`). The formal uncertainty practice of the R1 era was dropped.
Grade: checked-artifact and grep counts. Confidence high on the facts; the claim that the noise invalidates specific decisions is medium, since the declared criteria mostly failed by larger margins.

**F7. Solid: the star calculator is right, and the scoped proxy agrees with it.** The recomputed SR matches official metadata within 5.2e-6 on 6,924 ranked charts, and the scoped proxy on whole charts is within ±0.02 of SR (p5-p95) (checked-artifact). What failed was the use of SR, a whole-chart difficulty scalar, as a quality or local-pressure check (F2), not its computation.

**F8. The agent's own Lens reading acted as an evaluator and was more lenient than the human.** Docs cite page counts and reading receipts as evidence ("All 209 new time pages ... were read"; "agent-confirmed regression witness"). On the same 40 pages that led to V68, the agent wrote that mixed TAP/LN and anchors "remain possible" and that Blizzard is almost all LN. The human judged the LN "still badly off". The agent tended to check whether a structure could still occur; the human judged whether the chart was right. No agreement rate was ever measured. Grade: doc-claim plus the feedback index. Confidence medium.

**F9. Lead: the restored R1 baseline already shows the short-LN defect with real times, and no evaluator was run on the baseline before building on it.** In 4 of 7 restored-R1 real-time reconstructions there are LNs ≤ 40 ms: epistrofi 68 among 1,328 heads (4.5★), descent 500 (6.4★). Two have same-lane gaps below 60 ms (min 21 ms). Part of the fragmented-LN failure blamed on audio, H or R modules may be inherited from R1's row choices at supplied release times. Not verified: I did not compare with the source charts of these reconstructions, and n = 7. Grade: checked-artifact for the counts, inference for the attribution. Confidence low-medium.

**F10. The formulation left the evaluation target undefined, so the lineage built proxies.** `gameplay-state.md` names the target response and leaves its quantities and comparison rules to be set from mapper evidence. Its "Evaluation questions" table has no plain question like "is this generated chart an acceptable ordinary chart at its difficulty?". Every lineage evaluator stands in for an undefined target; the envelope docs say "not physiological capacity" and the code says "not a BAD rule". The honest disclaimers are real, but they meant no check could ever say a chart was good, so the human's eyes remained the only positive evaluator, applied sparsely. Grade: checked (formulation text). Confidence high.

## 5. Metric gain followed by human rejection

| When | Metric that improved or passed | Human reaction | Why the metric missed |
| --- | --- | --- | --- |
| 09-23 | 48 learned-timing charts pass legality and export | V12: target is playability, not legality | Mechanical only |
| 09-24 | Three 1200-step arms at near-equal validation NLL; restoration metrics | V27-28: NLL is only a proxy | Teacher-forced; not native rollouts (F5) |
| 09-24/25 | Zero same-column attacks < 20 ms after screening | V37-38: 21-36 ms still unacceptable | Threshold from the wrong population (F4) |
| 09-26 | a99519c: held-out spread error better, absolute difficulty error worse (doc-claim) | V46-48: long jack at 4★ under stream control; no breathing | Whole-chart SR (F2); no local jack check yet (F3) |
| 09-27 | Planner envelope excess .0275 → .0004 | V53-54: rigid, no breathing, weak audio echo | Optimized metric (F5); measures only peak attack rate |
| 09-27 | Memory H-base: excess .38 → 0, star MAE 1.33 → 1.16 | Self-rejected (H counts, startup) | Gains moved the failure elsewhere (doc-claim) |
| 09-28 | V68 set: 7/12 pass all numeric gates, 12/12 pass 20 ms, LN amount met on STYX | V68: short LN, release before head, irregular LNs; long jacks | Whole-scope LN gate; no LN placement check (F2) |
| 09-28 | Short-LN (80 ms) metric improves after repair | V75-78: arrangement still fails; ultra-short LN persist | q99-of-prevalence threshold too loose (F4) |
| 09-28 | Scratch expert NLL 31.0 → 10.7 (H), 23.1 → 2.3 (rows) | V79-81: all red, not fused | NLL on 16 training units; native output fails the 40 ms check |

## 6. Judgment

For the prosecution: over four days no evaluator could say a chart was good. Each was a q99 fence against one failure type, added after a complaint and tuned on that complaint's charts. Gates checked whole-chart quantities that the model could satisfy while getting the local organization wrong. Promotion criteria were tighter than seed noise, and the development panel was a handful of TRAIN songs reused dozens of times. The measured result: on the one set of charts tied to a specific human rejection, the evaluation in force passed 7 of 12 on quality grounds, and even the final battery passes 5 of 12.

For the defence: the lineage's evaluation code is careful and honest. Scopes are exact, LN tails resolve correctly, the star calculator is verified, disclaimers are explicit, and no single score is pooled. From 09-28 the q99 references were checked against held-out ranked groups, and the blind-spot probe found its own evaluator's holes. The battery does separate the late failure families (jacks, sub-40 ms LN) from ranked charts at 1-2.5% false positives. Much of the complaint-driven reactivity was the human asking for exactly that ("red eval first", V73). The agent often declined to promote and wrote down its tradeoffs.

Net: the evaluators are trustworthy as specific detectors with a measured false-positive rate (≈1% each on ranked). They are not trustworthy as evidence that a model improved: sensitivity to what the human rejects was never measured, and the decisive failures were local-organization failures that none of them represents. Confidence: high on the measured parts (F1, F2, F4, F7), medium on the causal reading (F3, F8), low-medium on F9.

## 7. Overlooked or never questioned

- A labelled set: the human's accepted and rejected charts or windows, kept as a held-out test for every evaluator. The 84 human messages were never turned into labels.
- Running the evaluators on the baseline R1 before building on it (F9).
- Seed and song variance measured before setting promotion criteria (F6).
- Songs outside TRAIN for native evaluation. The fresh-audio panel existed but the late work used TRAIN panels.
- A local-organization reference: "given this source passage, what do ranked charts at this difficulty do here", which F2 shows the gates lacked. `coordination_frontier_hypotheses_zh.md` and the ordinary 4★ study describe it as analysis, not as a check.
- Whether the human's complaints are separable at all by chart statistics: the Classic Pursuit source window itself holds eleven LNs ≤ 80 ms, and STYX's full source has 134 of 544 releases 1-40 ms off an H. Duration or gap rules will misfire on real charts; the human judges context.

## 8. What was missing, tied to failures (not a design)

- Sensitivity on human-rejected charts, measured and reported next to the false-positive rate on ranked charts (F1; V68 passed 7/12).
- Checks at the unit the human judges, local windows against the ranked distribution for that passage and difficulty, rather than whole-song totals (F2; STYX).
- Thresholds conditioned on the target difficulty band (F4; the 20 ms gate at 4★).
- A failure set held out from the evaluator's own calibration (F3).
- Seed replication and song-level spread set before any promotion criterion (F6).
- A baseline row: the same checks on R1 and on ranked charts in every comparison table (F9).

## 9. Worth keeping

- `osu_core/difficulty.py` star calculator (verified) and the H-only star lower bound (sound feasibility argument).
- The scan corpus `20260928-coordination-corpus-v1/scan-v1/charts.jsonl`: 6,924 ranked charts with per-chart LN, release, gap and proxy statistics, song groups and a held-out rule. It is a ready reference population.
- `gameplay_evaluation` observers (exact scopes, resolved tails, head recurrence, hold interactions) as instruments, together with their disclaimers.
- The blind-spot probe method: evaluate the evaluator on ranked held-out charts and on known failures, with a declared rejection budget.
- The census result that 3.5-4.5★ ranked charts have no same-lane gap under 40 ms (my sample: none under 70 ms).

## 10. Questions only the human can settle

- Is a chart that matches its whole-song LN ratio but moves LNs into a TAP passage a control success or a failure? The gates scored it a success.
- Is "short LN is a style" (V77) compatible with any duration check? Ranked 4★ charts do contain rare short LNs (2 of 200 in my sample).
- The formulation's response target is undefined. Should evaluation wait for it, or should a labelled accept/reject set of the human's own judgments come first as the ground truth?
- The feedback index notes that V35 may or may not cover release-to-head gaps; the 20 ms RH screen rests on that reading.

## 11. Not verified, and how to follow up

- F9 attribution. Compare each R1 reconstruction with its source chart: find sources via `artifacts/r1-real-reconstruction-20260921/conditions/*.json` (timing only; source path not found in `run-config.json`), then run `battery.py` on the source charts.
- Per-window evaluator scores on the V68 windows (the qualification harness supports scopes): add `Scope(name, a, b)` for the windows in `render_review.py` to `battery.py`.
- a99519c demo charts: no file or seed was retained (V47), so that rejection cannot be replayed.
- Lens images: I read none; my claims about what the human saw rest on the window counts above and the feedback index. `lens-main-v1/*.png` on the mac.
- The held-out false-positive rate of my ranked sample is in-sample for 158 of 200 charts; for a clean estimate, restrict `manifest.py` to `held_out == True` groups and sample more (e.g. 300).
