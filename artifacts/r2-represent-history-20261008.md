# Recurring agent failures across the record (opus-history, 2026-10-08)

Report of the fresh Opus subagent "opus-history" in the representation round ([r-represent-relaunch](r2-representation-20261008.md#r-represent-relaunch)), returned 2026-10-08 about 03:25 UTC. Brief: `~/ensomi/.sync/cp/scratch/r2-represent/brief-opus-history.md`. Read-only; no mac jobs. Saved as returned, below the main thread's checks.

<a id="v-history-checks"></a>**Main-thread checks (2026-10-08).**
- **Link removal confirmed:** `git log -S agent-failure-modes -- RESEARCH.md` gives `fbf5e96` (added, 2026-10-02 11:49 UTC) and `1bd2604` (removed, 2026-10-06 16:55 UTC).
- **Brief counts confirmed in kind:** a grep of `.sync/cp/jobs/*/brief.md` and `.sync/cp/scratch/r2-*/brief*.md` dated 10-07/08 for "failure-mode", "standing rule" and the two-SE rule finds only false hits ("2 seeds") and the history brief itself. No 10-07/08 brief carries the rule block.
- **Not checked by the main thread:** the lineage-era citations the report marks as unchecked, and the inferred root causes.
- **The five changes are proposals** for the human. Change 1 ("a launch card enforced by `ens`") and change 5 touch the workspace tooling, not the model.

---

## The answer in short

- **Nine patterns recur from September to 10-08.** Five have instances in both the lineage (September) and R2 (10-03 to 10-08): unvalidated measures, unexamined fixed inputs, local repair, proceeding against one's own warning, and overclaiming. Two occur in both eras: deciding without the human, and margins below noise. Two are mostly R2-era: monitoring cut for wall-clock, and reviews that check conformance rather than validity. "Forgetting lessons" is not a separate pattern; it is how the other nine come back (section 1.10).
- **The 2026-10-02 list stopped reaching the work, measurably.** It said: copy the standing rules into every brief. 10-03/04: 5 Astra briefs carry the rule block. 10-05/06: 2 of 5 R2 briefs carry a fragment (two-SE rule). 10-07/08: no rule block in the 8 Astra briefs or the 13 Claude round briefs [M]. The link to the list left `RESEARCH.md` in the 10-06 cleanup (`1bd2604`); every session that planned round 1 started after that [M].
- **Where it was applied, it bit:** phase-N lr pilots under the two-SE rule, X2 and X3 with fixed thresholds, the decode round measuring the HMM before using it. Where a check was not in the brief, nobody applied it.
- <a id="s-history-worst"></a>**Most damaging: deciding with measures never shown to see what the human sees.** It disables the agents' own error detection (section 4).

Tags: [R] stated in the cited note; [M] measured by the worker (grep, `git log`); [I] inferred; [P] proposed. Who: LA lineage agent (Codex, 09-23 to 09-29); MT Claude main thread; DR design round; W worker (Claude subagent or Astra); RV reviewer; H the human took part.

## 1. Recurring patterns

| # | Pattern | Instances | Root cause [I] |
|---|---|---|---|
| 1 | Unvalidated measures decide | 9, 09-25 → 10-08 | No held-out human labels until X0 (12 items, 10-07); validation planned "later" with no rule for when it fails |
| 2 | Fixed inputs and interface choices unexamined | 8, 09-23 → 10-08 | Scope and interface decisions treated as background; "out of scope to build" read as "out of scope to analyse" |
| 3 | Local repair at the nearest module | 5, 09-20 → 10-07 | The nearest module is what a session can change and measure; recognised upstream questions had no ledger |
| 4 | Proceeding against one's own recorded warning | 8, 09-24 → 10-07 | Warnings are prose risk lines next to the action; nothing turns one into a cheap test before the spend |
| 5 | Overclaiming causes; unchecked facts propagate | 7, 09-23 → 10-08 | Claims checked for presence in the notes, not for necessity (denominator, matching, alternatives) |
| 6 | Deciding without the human, or filling gaps in the human's decision | 7, 09-24 → 10-07 | Delegating "the night" read as delegating every decision in it; "for morning review" labels a decision already in effect |
| 7 | Margins below the noise floor; a selected checkpoint as the sole baseline | 3 episodes, 09 → 10-07 | See section 2 |
| 8 | Early-warning monitoring cut for wall-clock | 3, 10-04 → 10-07 | Evaluation costed as overhead, so it is the first thing cut to fit a night |
| 9 | Review checks conformance, not validity | 4, 09-30 → 10-07 | Reviewers get the plan and check the code against it; same model family, same blind spots |

### 1.1 Unvalidated measures decide

| Date | Instance | Wrong solution | Who |
|---|---|---|---|
| 09-25…28 | A q99 gate was added after each complaint, tuned on that complaint's charts. The gates passed 7 of 12 V68-rejected charts ([fm-self-evaluation](agent-failure-modes.md#fm-self-evaluation), [s-eval-sensitivity](lineage-review/synthesis.md#s-eval-sensitivity)) [R] | Another gate per complaint | LA |
| 09-26/27 | The star proxy rated a 16 s single-column run at 4.14★. The demo was tuned toward it, and the human playtest found long jacks. The planner's cost was reported as the gain ([s-response-blind](lineage-review/synthesis.md#s-response-blind)) [R] | Optimise the proxy | LA |
| 09-28 | The agent's Lens reading served as the evaluator, more lenient than the human, with no agreement rate measured (fm-self-evaluation) [R] | Agent reading used as a gate | LA |
| 10-04 | The in-run free-run report covered natural mode only, 4 charts. The human saw "average" patterns it did not flag ([o-human-average](r2-average-and-control.md#o-human-average), [s-r2-ranges](r2-average-and-control.md#s-r2-ranges)) [R] | Diagnostics on the wrong range | MT, W |
| 10-04 → 10-07 | "Real charts have 0 holds ≤ 60 ms" came from a small panel (r2-analysis-fable-judgment l.24). It became guard (iv), a decode mask and a human decision. The corpus figure is 2.72 % ([c-no-short-holds-claim](r2-phasen-and-lens-20261006.md#c-no-short-holds-claim)) [R] | A guard built on an unchecked premise | RV, then MT, H |
| 10-06 | The style probe found that generator state adds nothing (0.577 against 0.587). The proposed critic still sits on the same frozen features ([s-style-module](r2-phasen-and-lens-20261006.md#s-style-module)) [R], marked proposed | Design continued on an unrecognising representation [I] | W |
| 10-07 | 56M was selected by guards (iv v2) and (v), with agent-chosen tolerances calibrated on the 16-chart K ≤ 600 panel. It became B0 ([o-selection-v2](r2-phasen-and-lens-20261006.md#o-selection-v2)). Sources fail those guards on long charts (r2-diagnose-fable §5) [R] | Selection by miscalibrated guards | W, MT |
| 10-07 18:16 | X0 validated no measure, and the night was judged on m1-m7 anyway ([d-night-measures](r2-bakeoff-night-20261007.md#d-night-measures)). `d0-phi` gamed G by stripping LN ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1)) [R] | "Characterise" turned into "decide" | MT |
| Counter-example | The decode round measured the HMM score before using it, found a rarity meter, and dropped it ([s-decode-design](r2-bakeoff-night-20261007.md#s-decode-design)) [R] | Check worked | DR |

### 1.2 Fixed inputs and interface choices unexamined (includes the head-row example)

| Date | Instance | Wrong solution | Who |
|---|---|---|---|
| 09-23 | Premise: "R1 is a sound row model lacking only times". R1 actually reads the source's release-only times and a 16-candidate look-ahead. LN share followed the offered candidates (0.21 → 0.86), which was read as a reason to adapt R1 ([s-r1-interface](lineage-review/synthesis.md#s-r1-interface), [s-release-sensitivity](lineage-review/synthesis.md#s-release-sensitivity)) [R] | Build on the interface | LA |
| 09-25… | Difficulty controls were built although R1 infers difficulty from rhythm density and no training chart switches difficulty ([s-controls-order](lineage-review/synthesis.md#s-controls-order)) [R] | Control a quantity the input fixes | LA |
| 09-30 → 10-03 | The review found "R1 fills lanes into a mapper's rhythm" (s-r1-interface). The human removed release times only ([d-no-release-input](r1-verdict.md#d-no-release-input)). The agent wrote "Open: whether head times stay given" ([r1-open](r1-verdict.md#r1-open)) [R] | Question parked, never settled | MT |
| 10-04 | The star condition was unused after a 158M-exposure run: R² 0.75 from head rows, KL 2e-6. Condition intervals were drawn independently of the scored window. Both were found after training ([a-r2-reading](r2-average-and-control.md#a-r2-reading), [s-r2-ranges](r2-average-and-control.md#s-r2-ranges)) [R] | Train first, derive later | DR, MT |
| 10-06 | Head-row generation was put out of R2 v2 scope ([d-r2v2-scope](r2-v2-stage0-state.md#d-r2v2-scope), human). No note derives what supplied rows imply for choreography, control or evaluation [M by grep: rows were analysed only as a nuisance for star and difficulty] | Scope taken as licence not to analyse [I] | MT |
| 10-07 | In the plan the row "Head times fixed" reads only as "band-2 pile-up" ([s-arch-constraints](r2-bakeoff-plan-20261007.md#s-arch-constraints)) [R] | Narrow reading | DR, MT |
| 10-07 | B3's donor prior was drawn from skeleton-nearest charts, although X3 had just measured skeleton → θ R² < 0.3 ([o-x3-skeleton-theta](r2-collapse-20261007.md#o-x3-skeleton-theta)). The prior is therefore close to a band draw (r2-diagnose-fable §2) [R] | Prior built on an input shown to carry little | DR, MT |
| 10-07 | The design brief fixed the baseline as "phase N 56M", the selected checkpoint (`scratch/r2-collapse/brief-design.md` item 3) [M] | Winner's curse built into the brief | MT |
| 10-08 | The human raised the head-row premise ([d-represent-first](r2-representation-20261008.md#d-represent-first)) [R] | | |

The lineage lesson "the generator fills a supplied rhythm" was recorded on 09-30. On 10-03 it was applied to release times only, the part the human named [I].

### 1.3 Local repair at the nearest module

| Date | Instance | Who |
|---|---|---|
| 09-20/21 | R1's free-running collapse was patched with three rule-trained residuals ([s-r1-quality](lineage-review/synthesis.md#s-r1-quality)) [R] | R1 agent |
| 09-23…28 | About 27 of about 40 post-failure moves were local, and each threshold answered the latest complaint ([s-local-search](lineage-review/synthesis.md#s-local-search), [d-locality-counts](lineage-review/synthesis.md#d-locality-counts); the counts are not reproducible) [R] | LA |
| 09-25 | The ownership problem was located at 06:12Z; repairs inside the typed line continued for 11 h ([fm-local-repair](agent-failure-modes.md#fm-local-repair)) [R] | LA |
| 10-06/07 | Short holds led to a 60 ms decode mask ([p-min-hold-mask](decoding-stage.md#p-min-hold-mask), [d-phasen-base-mask](r2-phasen-and-lens-20261006.md#d-phasen-base-mask)). The cause was chart-level LN placement ([o-lnlen-cause](r2-phasen-and-lens-20261006.md#o-lnlen-cause)), and the mask was withdrawn ([d-lnlen-next](r2-phasen-and-lens-20261006.md#d-lnlen-next)) [R] | MT, H |
| 10-07 | LN drift was answered with `ln_level`, then `ln_length`, then θ with 10 coordinates: one input per symptom ([r-guard-lnlevel-job](r2-phasen-and-lens-20261006.md#r-guard-lnlevel-job), [p-bakeoff-round1](r2-bakeoff-plan-20261007.md#p-bakeoff-round1)). The human challenged this: a fine-tune on real histories does not target drift ([d-collapse-primary](r2-collapse-20261007.md#d-collapse-primary)) [R] | MT, W, H approved |

The 10-02 check for this pattern, a question ledger, never ran. Ledgers appear only in the 10-02 list, evaluation-first, three 10-03 Fable reviews and condition plan v1, and in no later R2 note [M].

### 1.4 Proceeding against one's own recorded warning

| Date | Warning, then action | Who |
|---|---|---|
| 09-24 → 09-28 | Expert 1 warned that a decoder may ignore a latent; the four-state mixture "collapsed to one code, as warned" (lineage-review/opus/interpretation §5) [R] | LA |
| 09-27/28 | The agent's own document said an optimised cost cannot establish improvement, and the cost was reported as the gain. A note said "whole B80 can pass while local LN organisation remains poor", and B80 stayed the check (fm-self-evaluation; opus/controls-ln §7) [R] | LA |
| 10-03 → 10-07 | [a-ln-mode](r2-ln-design.md#a-ln-mode) and the census ([s-ln-census](r2-ln-design.md#s-ln-census)) recorded: "a model without [a chart-level LN amount] can still drift". The human rejected a chart-level scalar ([d-r2-conditions](r2-implementation.md#d-r2-conditions)). The warning was not turned into a drift test in phase N's selection panel, which could not see drift ([s-arch-constraints](r2-bakeoff-plan-20261007.md#s-arch-constraints), last row) [R/I] | MT |
| 10-04 → 10-07 | "CE on real histories gives no pressure to use [the path]" ([a-r2-reading](r2-average-and-control.md#a-r2-reading)). The plan cited the R1 GRU and v1 landmarks as unused precedents ([s-network-families](r2-bakeoff-plan-20261007.md#s-network-families)). The Opus design said "the window proxy is as good as a convergent statistic, whatever the encoder". The night still trained B2 and B3 as CE plus redundant inputs, and θ was worth 0.0002 nats (round-1 item 3) [R] | DR, MT |
| 10-07 | Plan rule: "only the validated subset counts". None was validated, so all seven measures were used ([d-night-measures](r2-bakeoff-night-20261007.md#d-night-measures)) [R] | MT |
| 10-07 18:06 | Own reading: the human's marks are early and local, and round 1 targets F1/F2 only ([o-x0-first-read](r2-bakeoff-night-20261007.md#o-x0-first-read)). Confirmed at 18:16 ([o-x0-scored](r2-bakeoff-night-20261007.md#o-x0-scored)); the arm list was unchanged [R] | MT |
| 10-07 | "Regimes flip between checkpoints" (D8) is the stated reason for arm B1, yet B0 was one selected checkpoint ([p-bakeoff-round1](r2-bakeoff-plan-20261007.md#p-bakeoff-round1)) [R] | DR, MT |

### 1.5 Overclaiming causes; unchecked facts propagate

| Date | Instance | Who |
|---|---|---|
| 09-23…28 | "More data helped prediction, not rollout" rested on one 23-minute run, and "memory fails" on 384 updates ([fm-preconditions](agent-failure-modes.md#fm-preconditions)) [R] | LA |
| 09-30 | The synthesis was "weakest where it names causes" and needed ten corrections ([revision-2](lineage-review/synthesis.md#revision-2)) [R] | MT |
| 10-04 | "The state holds chart-wide information", with no prefix-descriptor baseline ([a-r2-fable-judgment](r2-average-and-control.md#a-r2-fable-judgment)) [R] | W (Astra) |
| 10-04 | "0 holds ≤ 60 ms in real charts", asserted by the reviewer asked to find overclaim (section 1.1) [R] | RV |
| 10-07 | "Own history amplifies, +41-78 %" compared runs against their source, not matched history. "SD ratio 0.43" came from seed-averaged levels ([c-lnlen-cause-revised](r2-phasen-and-lens-20261006.md#c-lnlen-cause-revised)) [R] | W, MT |
| 10-07 | "CE never penalises", "θ is a prerequisite", "nothing beats B1 → CE cannot hold any input". Caught by an external review, not by the agents ([d-review-amendments](r2-bakeoff-night-20261007.md#d-review-amendments)) [R] | DR, MT |
| 10-08 | A signed ratio on a 0.007 denominator, and a pooled B0 row compared with single-seed runs ([c-bakeoff-r1-readings](r2-bakeoff-night-20261007.md#c-bakeoff-r1-readings)) [R] | MT |

### 1.6 Deciding without the human, or filling gaps in the human's decision

| Date | Instance | Who |
|---|---|---|
| 09-24 22:34Z | Count factorisation and the typed skeleton started 9.5 h after the human's last message; the human reversed them (audit §3a) [R] | LA |
| 10-03 | The brief left "KL-like" and "model of normal" open; Astra filled them, and the human rejected the result (fable-evaluator-blockers Q4) [R] | MT, W |
| 10-03 19:58 | The R2 v1 spec was an "agent decision for the human's review in the morning", and a 20 h run was launched while the human slept ([d-r2-v1-spec](r2-implementation.md#d-r2-v1-spec), [r2-overnight-run](r2-implementation.md#r2-overnight-run)). The human's later rule: the human checks code before any run [R] | MT |
| 10-05 | An Astra brief asked it to pick the decision-rule branch; the human: "check numbers, not decide" ([r2-final-probe](r2-average-and-control.md#r2-final-probe)) [R] | MT |
| 10-06 | "Latest three batches" was read as 75 sections; the pool is about 4,700 ([c-style-pool](r2-phasen-and-lens-20261006.md#c-style-pool)) [R] | W |
| 10-07 15:25 | "Main-thread choice, not a human decision": no in-run free-run evaluation ([d-base-56m](r2-phasen-and-lens-20261006.md#d-base-56m)) [R] | MT |
| 10-07 18:16 | The night's measures were chosen "for the human's morning review" ([d-night-measures](r2-bakeoff-night-20261007.md#d-night-measures)), while the human was active at 18:52 and 19:20 ([d-decode-direction](r2-bakeoff-night-20261007.md#d-decode-direction), [d-review-amendments](r2-bakeoff-night-20261007.md#d-review-amendments)) [R] | MT |

### 1.7 Margins below the noise floor; a selected checkpoint as baseline

- **Lineage:** 0.1★ margins against 0.2★ seed noise. A single-pair effect of −0.35 shrank to −0.04 at 16 seeds ([fm-margin-below-noise](agent-failure-modes.md#fm-margin-below-noise)) [R]. LA.
- **10-03/04, R2 v1:** a 20 h run right after the build with no pilot. Conclusions were drawn at 19-31 % of the schedule, and the reviewer had to say "unfinished schedule before representation" ([a-r2-fable-judgment](r2-average-and-control.md#a-r2-fable-judgment)). The LN "oscillation" was within noise ([s-r2-plan-review](r2-average-and-control.md#s-r2-plan-review)) [R]. MT.
- **10-07, round 1:** margin 0.15 G against a checkpoint SD of G of 0.24-0.29; B0 was second-best of five on its own selection statistics ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer)) [R]. DR proposed, MT launched.
- **Counter-example:** the 10-06 phase-N lr pilots used the two-SE rule, which was in their brief ([s-phasen-tune](r2-phasen-and-lens-20261006.md#s-phasen-tune)) [R/M].

### 1.8 Early-warning monitoring cut for wall-clock

- **10-04:** R2 v1's in-training diagnostics measured natural mode on 4 charts only ([s-r2-ranges](r2-average-and-control.md#s-r2-ranges)) [R].
- **10-07 15:25:** no in-run free-run evaluation in the LN fine-tune, "for wall time" ([d-base-56m](r2-phasen-and-lens-20261006.md#d-base-56m)) [R].
- **10-07 18:10:** the mid-run mini-panels were dropped because B4 was not built; reader norms were logged with no reference scale ([d-bakeoff-night](r2-bakeoff-night-20261007.md#d-bakeoff-night)); the Astra build brief also says "no in-training full evaluation" [R/M].
- MT, all three times.

### 1.9 Review checks conformance, not validity

- **09-30:** the synthesis adopted the stronger wording although the review README had warned that "a reviewer given these readings tends to confirm them" ([revision-2](lineage-review/synthesis.md#revision-2)) [R].
- **10-03:** the Fable reviewer stated "no legality checker in src/", which was wrong (fable-evaluator-blockers, provenance) [R].
- **10-04:** the Fable judge introduced the false short-hold premise (section 1.1) [R].
- **10-07:** the pre-launch Opus check gave GO and fixed code details. It passed the per-resample CI, the short-panel guards and the signed G. Only the external review the human pasted caught the sign ([r-night-launched](r2-bakeoff-night-20261007.md#r-night-launched)) [R].

### 1.10 "Forgetting earlier lessons": lesson, then repeat

| Lesson (date, where) | Repeat |
|---|---|
| Lower loss is not better rollout (pre-V3, 06-17; opus/pre-v3-history §8.2) | Lineage NLL arms (09-24); phase-N NLL flat while regimes flip ([o-phasen-finished](r2-phasen-and-lens-20261006.md#o-phasen-finished), 10-07) |
| The generator fills a supplied rhythm ([s-r1-interface](lineage-review/synthesis.md#s-r1-interface), 09-30) | R2 takes head rows from an existing chart (10-03), and their implications go unexamined until 10-08 |
| Persistent paths go unused under CE; a latent "collapsed as warned" (09-28) | v1 landmarks and star unused (10-04), then B2/B3 readers unused (10-08) |
| Lesion tests catch dead inputs, not inputs ignored by trained weights ([fm-tests-not-binding](agent-failure-modes.md#fm-tests-not-binding), 10-02) | R2 v1's input-use tests at initialisation passed ([s-r2-v1-built](r2-implementation.md#s-r2-v1-built)), and star was unused after training. The B2/B3 zero-init identity tests passed, and θ was worth 0.0002 nats |
| LN level fed back through history ([s-ln-open-state](lineage-review/synthesis.md#s-ln-open-state), 09-30) | Phase N repeats it in milder form ([o-lnlen-cause](r2-phasen-and-lens-20261006.md#o-lnlen-cause)) |
| Instruments tuned on their triggering failure; small reused panels (fm-self-evaluation, 10-02) | v1's "≤ 0.3 %" from a 4-chart panel; the short-hold premise from a small panel; guards from the 16-chart K ≤ 600 panel |
| Noise floor before criteria (10-02) | R2 v1 (10-03), round 1 (10-07) |
| Two training seeds per recipe claim: "Adopted in stages 1-3" in the advice table of [r2-condition-plan](r2-condition-plan.md) (10-05) | Absent from plans v2-v5 [M]; phase N and round 1 each ran a single training |

## 2. Why the lists did not work

<a id="s-history-list-decay"></a>**What happened to the 10-02 list [M].**
1. **Propagation decayed.** The standing rules said: "A main thread copies these into every Astra or subagent brief" ([standing-rules-for-briefs](agent-failure-modes.md#standing-rules-for-briefs)). Briefs carrying the rules: 10-03/04, 5 Astra briefs with the block; 10-05/06, 2 of 5 R2 Astra briefs with a fragment (two-SE); 10-07/08, 0 of 8 Astra briefs and 0 of 13 Claude briefs. Earlier Claude subagent briefs lived in session scratchpads and are not countable.
2. **The entry path was cut.** The link to the list was added by `fbf5e96` (10-02 11:49 UTC) and removed by `1bd2604` (10-06 16:55 UTC), the cleanup in which the focus moved to finishing R2 v2. Every session from 31d15d7a to 28c38740 started without a path to the list.
3. **The rules lived in superseded documents.** Plan v1's advice table (two training seeds) did not reach plans v2-v5, and the 10-03/04 brief block did not reach the 10-07 briefs. Each redirect produced new documents, and the discipline stayed in the old ones.
4. **Enforcement was never decided** ("whether they are enforced in code or only as brief rules is open": list header; [p-claim-integrity](evaluation-first.md#p-claim-integrity)) [R]. Nothing at `ens run` or `ens astra` checks anything. Rules now sit in five places: the 10-02 list, evaluation-first, the 10-08 list, `MEMORY.md` and `AGENTS.md`.

**Where each check failed to bite.**

| Check (10-02) | Brief | Plan | Launch | Reading |
|---|---|---|---|---|
| Noise floor before criteria | **Failed.** The design brief asks for "a single decision rule", fixes 56M as the baseline, and does not mention noise [M] | **Failed in substance.** A chart-bootstrap CI within one checkpoint satisfied the check in form [I]. Training variance was demoted to an outcome line ("if B1 − B0 > 0.2, a second seed") | Pre-launch review on code conformance | Caught afterwards (diagnose round) |
| Preconditions and kill rule | | Kill rules had no reference scale | **Failed.** Mini-panels dropped by the main thread | Caught afterwards |
| Only validated measures decide | X0 was assumed to calibrate ("which X0 will calibrate") | The rule had no branch for "none validated" | **Failed.** The main thread filled the gap with "use all" | |
| Input-lesion test | | | Initialisation-time tests passed, as the list had said they would | Fit-time ΔNLL ran only in the diagnose round |
| Question ledger | **Failed.** Never in any R2 brief; "do head times stay given" stayed open from 10-03 | | | |
| Contradiction to the human | Not on the 10-02 list at all | | **Failed.** Noted at 18:06; the human was reachable | |
| Causal wording | | | | The page was checked against the notes (presence), not against the reasoning (necessity) |

**Why it failed structurally [I].**
- **Vocabulary mismatch.** The list names lineage failures (receipts, "R1" name drift, promotion margins). Round 1's failures came with new names: G, guards, θ readers. "The first run after any evaluator change" should have triggered on `collapse_eval.py`, a new evaluator, but nobody mapped one to the other.
- **The checks had no cost line in the plan, so they lost to the night.** The noise run (about 1.5 mac-hours) and the ΔNLL ablation (15 minutes) competed with a 6-7.5 mac-hour plan and with "training starts without waiting for the human" ([d-bakeoff-night](r2-bakeoff-night-20261007.md#d-bakeoff-night)). R2 v1 shows the same shape on 10-03.
- **The writers of the list were not its users.** A post-mortem subagent wrote it; design subagents never received it; the main thread approved plans without re-reading it.
- **AI review stood in for the checks.** The reviewer checked the code against the brief, and the brief did not contain the rules. Three Claude reviewers passed what one external review caught, consistent with the human's standing rule against AI-reviews-AI gates.
- **The list was long and unprioritised:** 10 standing rules, 9 claim-integrity items, 6 evaluator rules. The costly ones (frozen evaluator, receipts, two training seeds) were never built, and the cheap ones were dropped along with them.

## 3. Five structural changes [P]

<a id="p-history-changes"></a>All mechanical or human-facing; none adds an AI review gate.

1. **A launch card enforced by `ens`** (launch). Any training or decision job refuses to start unless the brief contains a card with: the decision statistic and the job id of its null run (seed against seed, neighbouring checkpoint, source against source); each deciding measure's validation status (an AUC against held human labels with its job id, or "characterise only"); for each new input, a teacher-forced ΔNLL on trained weights (job id); a kill rule with a reference scale. The launcher checks the fields exist and the job ids exist with exit 0. Test: fill it for round 1 retrospectively; it would have needed about 1.5 mac-hours plus 15 minutes, and those runs showed the null. Works if over the next three decision nights no decision statistic turns out to sit inside its own null.
2. **A measure registry with a validation gate** (plan, reading). One file lists every measure, the human labels it was tested on (X0, the Lens sections, the C0 screen), its AUC or F1 with a CI, and its status. A decision rule may name only registered, validated measures; others print under "characterisation". Labels used to design a measure are split from those used to validate it. Test: the record holds at least six decisions on unvalidated measures. Works if no later reversal is caused by an invalid measure, and the next human screen agrees with the validated measures at their registered AUC.
3. **A fixed-input premortem in every design brief** (brief). Before proposing arms, the design worker fills a table: for each fixed input or choice (supplied head rows, warm start, selected baseline, panel, memory off, objective, decision unit, scope exclusions), what the model can and cannot learn under it, what the evaluation can and cannot see, which of the human's complaints it could cause or hide; plus open-question rows from the notes, e.g. [r1-open](r1-verdict.md#r1-open). Test: give the 10-07 design brief plus the template to a fresh worker; does it surface the head rows, the selected B0, identifiability under teacher forcing and the K ≤ 600 panel? Works if blind spots the human raises that the premortem missed fall round over round.
4. **A contradiction step before any unattended launch** (launch). The main thread sends the human at most three lines: the human's latest evidence that contradicts this plan (for round 1: the X0 marks are early and local, and the arms target identity past row 511). The launch card records the reply. If the human cannot be reached, only measurement items (nulls, ablations, noise runs) may run. Test: each post-mortem lists the human evidence that predates the launch; any contradiction missing counts as a failure.
5. **One checklist with a single home, at the point of use** (brief, launch). Replace the 10-02 list, the 10-08 list and the scattered memory items with one checklist of at most seven lines, each with a mechanical trigger (e.g. new evaluation code → run the null first). `ens` prints it, the brief template embeds it by hash, `RESEARCH.md` links it at the top, protected from cleanups. Each new failure record attaches its instance to a line or replaces the line that has fired least. Test: grep briefs for the hash (today 0 of 21 on 10-07/08). Works if ≥ 90 % of briefs carry it and each line's recurrence count falls.

## 4. The most damaging pattern

Deciding with measures never shown to see what the human sees (section 1.1). It switches off the agents' own error detection: when the measure cannot see the target, noise, overclaims and a wrong direction all look like results, so every other pattern passes unnoticed.

It also compounds. Short-panel guards selected 56M; 56M became B0; B0 set every margin; a premise taken from a small panel reached a human decision (the 60 ms mask). The human remained the only working detector: in September (7 of 12 rejected charts passed the gates), on 10-04 ("average"), and on 10-07 (X0, where no measure matched). The human's bandwidth is the bottleneck: 12 X0 items and 48 C0 pairs in a week.

Every night spent under this pattern bought numbers that could not teach anything, which is the human's complaint that experiments only show "this arm fails". The redirect to representations first, with the condition that a representation counts only if a simple probe recognises the labels already held ([a-represent-agree](r2-representation-20261008.md#a-represent-agree)), is the direct fix, provided the labels used to choose the representation are kept apart from those used to validate it.

## Not done, and limits

- **Lineage instances** come from two read-only helper extractions. Checked by the worker: s-r1-interface, s-release-sensitivity, s-controls-order, the opus/interpretation §5 latent row, revision-2, d-r2-v1-spec, r2-overnight-run. The other lineage citations are unchecked.
- **Not read in full:** condition plans v2-v5, the design-fable and design-opus reports, the decode reports.
- **Brief counts** cover brief files on disk only; Claude subagent briefs before 10-07 are not countable.
- **No pilot was run** for any of the proposed changes.
