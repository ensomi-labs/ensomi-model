# R2 v2 implementation plan (condition plan v4)

Shareable. Written 2026-10-06 by an Opus subagent (Claude, control plane). It supersedes [r2-condition-plan-v3](r2-condition-plan-v3.md); a reader needs neither v1, v2, v3 nor the reviews. It serves [d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss), [d-dpo-synthetic-off](r2-average-and-control.md#d-dpo-synthetic-off) and the human's decisions recorded as [d-formulation-answers](r2-style-formulation-check.md#d-formulation-answers), [d-target-strength](r2-style-formulation-check.md#d-target-strength) and [d-formulation-answers-2](r2-style-formulation-check.md#d-formulation-answers-2). The formulation in `docs/formulation/` on branch `docs/style-formulation` at `bdbbbde` (`style-conditions-and-control.md`, `notation.md`, `gameplay-state.md`, `README.md`) is authoritative and takes precedence over this plan; the record of how it was written, with its open questions, is [style-formulation-rethink](style-formulation-rethink.md). Open questions q2, q3 and q7-q10 of that record are followed at their written defaults unless section 15 asks. Draft PR #16 (`r2/train`) states the branch's scope in prose; this plan is its detailed form. Nothing here is adopted until the human decides the questions in section 15.

Code read at `r2/train` `7d9640a` (paths relative to `src/ensomi_model/r2/` unless stated) and the formulation at `bdbbbde` in the worktree `~/wt/ensomi-model-formulation`. No code, formulation document or other note was edited, no job was launched and no `ens` command was run. Mirrored artifact files on the control plane were read where a number is tagged [data].

<a id="v4-amendments"></a>**Amendments after v4 (main thread, 2026-10-06; they override the text below where they differ).** The human answered §15 ([d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers)). The formulation document is now `docs/formulation/style-conditions-and-control.md`, titled "Style conditions and control", at `8da2bda`. Line numbers cited below as `style-conditions-and-control.md` lines refer to its earlier name `style.md` at `bdbbbde`.

- **Q-B: option (a).** Training LN scopes include runs of 2-8 pieces (p 0.25) and the whole song (p 0.10).
- **Q-F: the learning-rate schedule is Astra's task.** It may change between training phases, informed by R1's staged recipe. The schedule is fixed before any arm comparison and is identical across arms. It replaces options (a) and (b).
- **q10: no cancellation once a scope has started.** Once the committed frontier reaches a request's start, the request can be neither cancelled nor changed. Before then it may be withdrawn or replaced, and the replacement is a new request under g < a. Where H4, R4, T-C, the §5.2 request set, the §12 rows at lines 631-632 and stage-0 item 0.12 allow cancellation inside a scope, they change accordingly. T-C becomes: withdrawing before the start leaves generation bit-identical to never adding the request; cancelling or changing at or after the start raises.
- **Q-E: the condition terms are on in the new recipe.** λ = 1 for every kind present; there is no separate emphasis arm. Stage 1 compares A0, the current recipe on stage-0 code, with A1, the new draw plus the condition terms. Which of the two changes helped is not separated.
- **Q-G: option (B).** Stage 2 includes a term that pushes the realised star of generated sections toward the request: the relaxed-proxy term of §7.3, with true F3 as its calibration measurement. This is the human's OK for own-sample training for difficulty. Q-I for LN is not answered, so the LN own-sample term of stage 3 is not built.
- **Process.** No working-day estimates; build costs in §13 are void, and mac compute times stay. No AI review gates. Implementation follows this plan and the decisions only. Where the plan leaves a training detail open, the implementation keeps the plan's stated default behind a config switch and reports it; it invents nothing. No training run starts before the human has checked the code.

Evidence tags: **[code]** checked at the cited line for this plan; **[data]** read for this plan from the named file; **[data, v3]** a number plan v3 read from a file not re-opened here (listed in section 16); **[inferred]** reasoning; **[to measure]** a number a stage-0 job must produce.

**Standing rules (bind every stage):**

1. Before any run, state the claim, metric, threshold, songs, random seeds, training seeds and stop rule. Any other number is post-hoc.
2. Every output carries an execution receipt. A component is "in the system" only if the receipt lists it; a field the system cannot honour is rejected at input, never accepted and ignored.
3. Every claim names the test or evaluator that fails if it is false. Mechanism tests are engineering, not evidence. Every model forward has an input-lesion test per named input.
4. No conclusion from a run below its stop rule. A difference under two standard errors is "none". Tables carry baseline rows and the spread across random seeds and songs.
5. Evaluators are frozen and hashed, and never scored on what they were tuned on. The ledger of recognised questions (section 14) is kept with the plan.

**Terms.** A **random seed** selects sampling randomness (the `seed` of `sampling.py:29` [code]; 954-956 in the panels). A **training seed** selects weights and draws (`seed_weights`, `seed_draws`, `train_ce.py:75-76` [code]; 171/471 in the first run [data, `r2-runs/r2-ce-overnight-20261004/config.json`]). A **chart seed** is a committed prefix with its fixed-through time g0 (`prefix_actions`, `prefix_gap` of `sampling.py:28` [code]). The three are never written as "seed" alone. A **property** is a scoped measurement of the result (LN share, section difficulty); a **property target** asks for one value of it under a declared ν. A **style directive** asks for organisation (named attributes from stage 4; references and edits later). A **request** carries a scope S = [a, b), at least one directive, and η (priority and transition intervals; the default η has neither). The **request set** 𝒰 holds the requests in effect, each with the committed boundary g_u at which it was added and, if cancelled, the boundary g'. The **effective track** is the per-kind interval list the model reads. The **visible set** V_k is the set of effective-track intervals decision k may read (rule L, section 4). The **onset decision** of a scope is the first decision whose row time is at or after a; its **exit decision** is the first decision whose row time is at or after b. A **readout** measures the actual chart with the same ν as the targets. The **record** (receipt) lists intended against realised. The **baseline style ρ** is reserved (section 5.6). The **default operating point** is the generator the selection rule picks (section 9.6); it is what "default strength" means for R2 v2.

---

## 1. The plan in one paragraph

Requests become objects: a song-time scope, at most one exact target per property (LN share; absolute section difficulty) under the declared ν, strength fixed at the default, the default η (no priority, no transition interval), and from stage 4 named-attribute style directives. A validator rejects a request added at or after its start, any two directives on the same quantity whose scopes overlap, and every reserved field set away from its default; different-quantity overlaps all apply and their deviations are recorded. One `properties` module declares ν for labels, frame counters, readouts and validation. R2's decision k decides the closes in the gap before its row together with the row, so the formulation's two locality conditions are exact only if a decision reads a request when every row it can produce lies in the scope (rule L). Rule L removes the birth-role frame, masks each scope's exit decision and, when a hold is open there, its onset decision, and makes loss ownership equal to visibility: a condition's term scores exactly the factors that can read the condition. The loss is a uniform per-decision cross-entropy (fixed divisor, no window mean) plus per-condition emphasis terms on the governed factor, with weight zero in stage 1, so that stage tests one thing: a draw that places scope onsets inside the scored window (dropout first, natural decisions weighted back to the current distribution, spans of every length the interface promises). Stage 1's primary is in-distribution span following at onsets, two arms differing only in the draw, two training seeds, 50M exposures, thresholds relative to the control arm. The default strength is the operating point the frozen selection rule picks (checkpoint, guards, decoding), named by hash in every record, with its adherence reported per property as a deviation distribution. Stage 2 makes difficulty a residual against a frozen skeleton baseline, requested in absolute star; the readout keeps hold tails decided after the scope, so stage 0 measures how much of a section's star those tails carry. The baseline style ρ is a reserved slot and its own stage after stage 2; DPO waits for real pairs.

---

## 2. Facts the plan rests on

### 2.1 Code [code]

| Fact | Where |
| --- | --- |
| A track is a tuple of `Interval(kind, a, b, value)`, half-open [a, b) in song milliseconds; kinds 0 LN share and 1 tiled star; intervals of one kind never overlap; an empty track is natural mode | `conditions.py:3-5`, `features.py:256-261` |
| Decision k < K produces the close rows of gap k at times strictly inside (t_{k−1}, t_k), then the row at t_k; decision K (EOS) produces closes inside (t_{K−1}, T) only and has exactly one legal action (held lanes code 2, free lanes 0) | `common.py:7-18`; `state.py:52-54, 76-100` |
| Every head row places at least one head | `state.py:55-57` |
| Release candidates of gap k are built from t_{k−1}, t_k and the grid only, strictly interior, on gaps of at least 2 ms; every lane of a gap shares them | `features.py:72-78`; `candidates.py:58-61, 81, 86-89` |
| LN spans are single partition pieces of 8/16/32/64 beats, bounds at beat times; 1-4 per song whatever K, consecutive with p 0.5; the value is the source's own share over the piece, computed inline | `conditions.py:20-23, 26-31, 34-56` |
| Star spans: cached cells (30 s, 60 s, start by a hash of the heads-only inputs, up to four consecutive per length) or the whole song with p 0.10; 1-3 per song; the same cells every draw | `labels.py:37, 56-75`, `conditions.py:59-75` |
| Dropout 0.20 all, 0.25 per kind, 0.20 per interval, on the per-song track | `conditions.py:83-90` |
| `validate_track` rejects nonpositive length, LN share outside [0, 1], star < 0 and overlap within a kind | `conditions.py:94-103` |
| `replace_interval` clips a start to just after the frontier, keeps the part of an old interval before the edit, merges equal adjacent values; no caller outside its test | `conditions.py:106-134`; `tests/r2/test_conditions.py:55-65`; `git grep replace_interval 7d9640a` |
| The window start never reaches the track draw | `data.py:68-83` (`draw_track` at `:82` sees no `start`) |
| Start rule: BOS p 0.125, last 256 rows p 0.125, else uniform in [0, K]; stop = min(start+256, K+1) | `data.py:74-81` |
| Loss = window mean of decision NLL, averaged over the batch; the DPO anchor reuses it | `train_ce.py:266, 269`; `train_dpo.py:190-191` |
| "Conditioned" statistics keyed on a non-empty track | `train_ce.py:145` |
| In-run free-run is natural only from BOS; `continue_chart` takes a prefix, one fixed track, a random seed and a stop | `train_ce.py:317`; `sampling.py:28-29` |
| Sampler: Gumbel maximum on the masked log-softmax (temperature 1), a fair orientation, the pointer one lane at a time; one generator per call seeded by the random seed | `sampling.py:21-24, 43, 64-79` |
| Chart seed: prefix decisions replayed for legality; generation resumes at the next decision, so g0 is the time of the last prefix decision | `sampling.py:35-42, 56` |
| Presence bit set for a kind on every row when any interval of that kind exists in the track | `features.py:291`; `DEVIATIONS.md` item 5 |
| Activity test is half-open, `times >= a` and `times < b`; EOS is queried at T, so a whole-song interval [0, T) is inactive at the EOS row time | `features.py:49, 293` |
| Row frame per kind (16 channels): value, offsets to the bounds (8), progress, log1p heads, log1p LNs, ratio, log1p remaining, active, presence; star value normalised by /4; LN value 2v − 1 | `features.py:275-276, 296-304` |
| Frame counters count head objects of decisions before k whose head time lies in [a, b) (the labels' ownership) | `features.py:264-272` |
| The token conditioner gives every query one token per interval of the track, with value, offsets, `active` and `started`, before the interval starts and after it ends | `features.py:308-330`; `tests/r2/test_conditions.py:25-35`; `DEVIATIONS.md` item 6 |
| FiLM input is 3 roles × 2 kinds × 16 channels; the same FiLM weights modulate the hand vectors (row role only) and the pointer query (row, candidate and birth roles) | `model.py:62, 168, 177-184, 265-278, 288` |
| Release factors read frames at the row time, at every candidate time and at the held LN's start (birth); the pointer base is built from the conditioned hand vectors and the decision's codes; its softmax runs over all candidates of the factor | `model.py:253-263, 265-281, 283-291, 309-319` |
| History tokens carry no condition features; conditioning enters only at the queried row and the pointer | `features.py:182-199`; `model.py:171-175` |
| FiLM output layers zero-initialised: natural mode is learned FiLM(0) | `model.py:62-64` |
| Code table: per lane, free-before 0 nothing / 1 tap / 2 LN head; held-before 0 keep / 1 release on row / 2 gap release / 3 gap release + tap / 4 gap release + LN head; 625-way joint action under a support mask | `common.py:7-18`; `state.py:49-57` |
| Decision log-probability = action log-probability + the two-orientation release mixture | `model.py:4-9, 236-251` |
| Tiled star: objects whose heads lie in [a, b), tiled to 240 s with a 1 ms seam rule, scored by `compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)`; ≥ 30 s; described by a version string; untrimmed tails | `labels.py:35-39, 78-110` |
| Tiled star of a scope that owns no object returns 0.0, not undefined | `labels.py:108-109` |
| Four separate definitions of LN share: the draw (`conditions.py:42-52`), `labels.ln_share` (`labels.py:42-45`), the free-run summary by object start time (`report.py:46-48`), and A's probe (`artifacts/r2-analysis-20261004/control/scripts/probe.py:177-184`) | as cited |
| Labels read the original `.osu`, not the cache representation | `labels.py:116-118` |
| Manifest reused when only `star_conditions` matches; free-run charts K ≤ 600 | `data.py:121-129, 95, 111` |
| Safe checkpoints written at resource trips are evaluated on resume and enter `evals.jsonl` at irregular exposures | `train_ce.py:368-369, 411` |
| Resource guard trips on system-wide swap growth > 1 GiB since the trainer's own start; restart budget 5 for the run's life | `research/oracle_time_continuation/runtime.py:31, 95, 114, 119, 129-133`; `launch.py:27, 106-112` |
| Training receipt: entry point, config, code identity, cache hashes, training seeds, device; free-run entries carry the random seed only; no request, chart-seed or readout fields | `train_ce.py:331-339`, `receipts.py:14-39`, `train_ce.py:314-325` |
| Mirror: `maxStagingFileSize: "4MB"`; pattern exclusions cover `/reports/**/*.jsonl` only; the comment says the limit does not stop a transfer and retries every cycle | `~/ensomi/mutagen.yml:161, 179-182` |
| Design rules: no LN-share balancing of the draw without inverse-probability weights; the evaluation manifest is 128 groups, 24 per star band | `artifacts/r2-ml-design-20261003/design.md:303, 535` |
| Input-lesion test pattern: tiny model, one forward, each named input changes the likelihood | `tests/r2/test_inputs_matter.py:31-60`, `tests/r2/helpers.py:87-94` |
| An explicit LN 0 frame differs from no frame | `tests/r2/test_conditions.py:19-22` |

### 2.2 Data

- Recheck, ckpt-0061432779 (61.4M exposures, one training seed), `artifacts/r2-recheck-20261005/control-table.md` [data]: whole-song LN slope 0.715 ± 0.045, MAE 0.115 ± 0.011 (A's gate slope ≥ 0.7, MAE ≤ 0.15: met); half-song switch DiD/2 0.272 ± 0.034 (gate 0.30: not met); natural LN share 0.312 ± 0.038, whole-song request 0 realised 0.134 ± 0.020; star slope 0.014 ± 0.015, star value KL 1.0e-5 nats/row; 0 legality violations in 272 rows. Real share on those 8 charts 0.187 and teacher-forced Δ expected LN fraction for request 0→0.9 0.060 ± 0.010 [data, v3].
- `average-table.md` [data]: generated-minus-real-history LN forecast shift 0.129 ± 0.044 (0.217 at 30.7M, 0.012 at 39.5M); natural LN share 0.315 against 0.190 real (24 charts) and 0.298 (48 charts × 3 random seeds); landmark-by-prefix interaction −0.41 ± 0.32 (none).
- `r2-runs/r2-ce-overnight-20261004/` [data]: config lr 1e-3, lr_min 3e-5, weight decay 0.01, 158M-exposure cosine, checkpoints every 4.388M; `run.json` 2,177 decisions/s wall, stopped by the resource guard at 121.7M after 5 restarts; `evals.jsonl` 32 entries, fit_dev action NLL minimum 1.9753 at 61.4M, last 2.0612 at 121.7M (a `-safe` checkpoint). Checkpoint-to-checkpoint NLL noise 0.02-0.03 and the adjacent-checkpoint LN-share deltas (1 of 24 beyond 2 SE after 35.1M under an unpaired SE, 0 of 24 chart-paired) [data, v3].
- `artifacts/r2-analysis-20261004/control/star-prediction-summary.json` [data]: ridge on 23 head-time features, fit_dev R² 0.746 ± 0.013, RMSE 0.528 ± 0.014; kNN-32 R² 0.753 ± 0.012 (difference under 2 SE: none); real dev star mean 3.59, SD 1.05; 74,907 train and 7,862 dev intervals.
- `artifacts/r2-cache/v1/labels/star_summary.json` [data]: 82,773 labels (4 invalid) in 472 s on 4 workers (≈ 23 ms per label per worker). `summary.json` [data]: 12,531 charts, 13.97M head rows; fit_train 11,368 charts (the train set of `star-prediction-summary.json` [data]), in 4,167 groups [data, v3].
- Recheck `execution.json`: 264 continuations in 9 min 50 s on 2 threads [data, v3].
- Range audit (`r2-range-audit.md`, 3,000 draws) [data]: LN active on 13.5 % of scored heads, star 23.9 %; informative scored onset rows 0.18 %; median LN piece 77 rows; star interval median 278 rows; a window spans a median of 38 s; 3.8 rows per beat; 15 % of active rows within 16 rows of their onset.
- Fable judgment: at 39.49M the variance of generated charts across random seeds on one skeleton (7.14) was twice the variance across skeletons (3.51); seed persistence fades by 256-512 rows ([a-r2-fable-judgment](r2-average-and-control.md#a-r2-fable-judgment)) [data, v3].

### 2.3 The formulation, as it binds R2 (paraphrase; `bdbbbde`)

- **Request** (`style-conditions-and-control.md` "Scoped requests", lines 125-164): u = (S, style directive, property targets, η), at least one directive. S is one interval [a, b) in song time with 0 ≤ a < b ≤ T; when b = T the scope also contains T. A row at a is in scope, a row at b < T is not. At most one target per property; a target names the property, its ν, one value (never a range) and an optional strength. η holds the priority and any transition intervals; the default η has neither.
- **Validity, expiry, cancellation** (lines 166-193): a request is valid only if the committed boundary g at which it enters 𝒰 precedes its start (g < a), and also precedes any transition interval before a; adding one at or before g is rejected. Expiry comes when g reaches b. Cancellation at g' inside the scope ends governance at g' and the record keeps the committed part's readout marked cancelled. A change is a new request that starts after g'. Steering later rows of an active scope toward its section target is not compensation.
- **Locality** (lines 195-210), for the default η and every window: (1) the law of rows before a is the same with and without the request; (2) given the rows before b, the conditional law of rows at or after b is the same. Both include no-row decisions; (2) covers the close of a hold started inside S even when ν counts that hold in S, and the record lists such objects.
- **Overlap** (lines 218-245): two directives on the same quantity with intersecting scopes (the same property under any ν; two style directives specifying the same attribute) make the request set invalid. Directives on different quantities all apply; an optional priority orders them when they cannot all be met, and a shortfall is declared in the record. Adjacent scopes do not overlap.
- **Unspecified, zero, absent** (lines 247-264): an unspecified property is free; an explicit 0 is a target (a headless scope does not meet LN share 0, because an undefined readout is not zero).
- **Strength** (lines 266-333): unspecified means the default level, which is the generator's primary trained and validated operating point, balancing playability, style and control; it is not "target met exactly". Default adherence is a measured property of the generator, reported per property. Higher levels are ordered, not scaled; none below the default; no zero.
- **ν** (lines 95-123): declares counting and boundary assignment, the evaluator with its context and profile, when a readout is undefined, the deviation measure and resolution, and the mirror behaviour. A readout depends only on chart content and ν.
- **Generation inputs** (lines 72-93; `notation.md` lines 205-244): ρ is always in effect and is not chart state; 𝒰 is retained across continuation calls; neither is derived from (H, g).
- **Seeds, natural continuation, identity, record, evaluation** (lines 335-460): the chart seed is history and evidence for ρ, not a request; a free property is not held, compensated or pulled; the record lists the baseline, the seeds, every request with its fields and cancellation boundary, each target's readout and deviation (undefined kept undefined), shortfalls, and objects crossing a scope end; the evaluation table states the comparisons (section 12).
- **Controls** (`gameplay-state.md` lines 347-398): demand requests have no interface yet; strength, tendency, adherence, demand and temperature are separate quantities.

---

## 3. Rulings and change log

Ruling: **adopt** (plan changed as stated), **amend** (adopted with a change, argument given), **reject** (with evidence), **stands** (a v3 ruling unchanged).

<a id="v4-rulings"></a>
### 3.1 The human's decisions of 2026-10-05/06

| # | Decision | Ruling | Argument and where it lands |
| --- | --- | --- | --- |
| H1 | Targets only: a property directive is an exact target, never a range; drop (lo, hi) and Q-R; strength is defined but not built, its slot reserved, any non-default value rejected ([d-target-strength](r2-style-formulation-check.md#d-target-strength)) | adopt | `style-conditions-and-control.md` lines 152-155. The schema carries one scalar `value` per target and `strength = 'default'`; any other strength raises (§5.2, §5.5). The frame has one value channel per property kind (§6.8). v3 §3.7 and Q-R are withdrawn; the range flag, the half-width lesion and the hit-rate readout go. T-R tests the rejection. |
| H2 | Default strength is the primary trained and validated operating point; adherence is measured and reported per property; it is not "target met exactly" ([d-formulation-answers-2](r2-style-formulation-check.md#d-formulation-answers-2)) | adopt, with its consequence stated | `style-conditions-and-control.md` lines 273-280; `README.md` lines 53-56 give the operating point research status. For R2 v2 the default is defined by the training recipe, the frozen selection rule with its guards, and the decoding settings (temperature 1, the two-orientation pointer, rule L, presence `none`). §9.6 is rewritten as "The default operating point"; its rule is hashed with the evaluators and the hash enters every record. The guards are the plan's choice of operating point, not a pass threshold on adherence (`style-conditions-and-control.md` evaluation table, first row); adherence at the selected checkpoint is reported per property as the deviation distribution with undefined readouts kept apart. Changing the rule or the decoding defines a new default whose adherence is measured again. |
| H3 | Same-property overlap is invalid input and the set is rejected; the written default extends this to the same style attribute and to the same property under another ν; priority only orders different-quantity directives that cannot all be met, and is optional | adopt; amend for priority | `style-conditions-and-control.md` lines 218-245; q9 written default (a). The validator rejects instead of clipping; no priority between same-quantity directives (§5.2). Amended: R2 v2 rejects any priority value, because it has no mechanism by which a higher-priority directive is followed first and a lower one is never met at its expense; accepting a priority it cannot honour would list a component that is not in the system (rule 2). Different-quantity overlaps without priority are accepted (q8 written default (a)) and each target's deviation is recorded as the declared shortfall (§5.3, §5.7). |
| H4 | A request is valid only while the committed frontier g is before its start a; mid-scope activation does not exist; `replace_interval`'s clip becomes a rejection; cancelling an active request is allowed (q10 written default) and a change is a new request starting after the cancellation boundary | adopt | `style-conditions-and-control.md` lines 168-193. The validator rejects a ≤ g_u (§5.2). `replace_interval` (`conditions.py:106-134`), which clips at `:108-109` and merges at `:126-131`, is retired; its test `test_conditions.py:55-65` is replaced by T-A and T-C (§4.6). The request set keeps each request's addition boundary across continuation calls, because validity is checked at addition and `continue_chart` today takes one fixed track (`sampling.py:28-29`). Cancellation at g' = t_j removes the request from V_k for k > j (§4.4) and marks the record. |
| H5 | A hold closed after its scope does not see the request; ν may still count it in the scope; the record lists it. This removes v3's ownership by birth (N5, T-O, the birth-role frame of an expired scope) | adopt and extend | Checked against both locality conditions path by path (§4.2). Two further reads of a scope outside it exist besides the birth role: the onset decision's joint action when a hold is open, and the exit decision's candidate frames inside S. Per-candidate masking cannot remove them exactly (§4.3). Rule L, a decision-level rule, does (§4.4). Consequences: the birth-role frame is removed; the candidate-role frame stays, exact under rule L; ownership equals visibility (§4.5, §6.1); the difficulty term no longer owns tails decided after b, while ν keeps counting them (§7.5); T-O becomes T-V and T-P1 to T-P3 are restated, with free-run tests T-L1 and T-L2 for the two conditions (§4.6). |
| H6a | Presence `none` | stands | Necessary but not sufficient for condition 1 (§4.2). `DEVIATIONS.md` item 5 rewritten. |
| H6b | A released or unspecified property is free | stands; its test amended | The formulation's comparison is with natural continuations from the same post-scope history (`style-conditions-and-control.md` evaluation table, "Is a released property free?"). Under rule L that comparison is an identity, verified by T-L2 and by an identity check on every selected checkpoint (§9.7 (a1)). v3's Δ_post against the run without the override measures persistence through history, which the formulation permits; it is reported, not a guard (§9.7 (a2)). |
| H6c | Transitions only by explicit η, only `immediate` built | stands, renamed | v3's `immediate` is the formulation's default η: no priority, no transition intervals. v3's reserved `announce` and `ramp` are a lead-in interval before a and a release interval after b; both are rejected until built (§5.3). |
| H6d | The baseline style ρ reserved for a stage after stage 2 | stands | §5.6; stage R's return-to-natural test amended to the formulation's comparison (§13). |
| H6e | Song-time scopes | stands, amended at T | A scope with b = T contains T: the activity test closes at T (fixes `features.py:293` for EOS) and ν says so (§5.1). |
| H6f | A condition's loss acts only where the condition is present | stands, made exact | "Present" is V_k: Ω_I is the set of factors of decisions with I ∈ V_k (§4.5, §6.1). |
| H6g | Difficulty as a residual against a skeleton baseline, requested in absolute terms | stands | §7.1-7.2 unchanged. |
| H6h | DPO waits for real pairs | stands | Deferred list (§13). |

<a id="v4-reconcile"></a>
### 3.2 "Points the plan revision must reconcile" ([style-formulation-rethink](style-formulation-rethink.md#plan-points))

| # | Point | Ruling | Argument and where it lands |
| --- | --- | --- | --- |
| R1 | §3.2 resolver clips the lower-priority interval on a same-kind overlap; reject instead | adopt | As H3; §5.2 step 3. |
| R2 | Priority only orders different-kind directives; v3's "across kinds nothing is resolved" is compatible if the record keeps any priority and declares shortfalls | amend | Priority is rejected in R2 v2 (H3); the record lists co-active targets and each deviation (§5.7). If q8 were answered (b), every overlapping LN and difficulty pair would be rejected until priority exists (§15). |
| R3 | N5 and T-O: a release factor of an LN born in S and decided after S's end must not read S; ownership by birth goes; ν may still count the hold | adopt and extend | H5; §4. |
| R4 | §3.2 mid-generation edits, `replace_interval`, §3.3 "next decision after the frontier": adding at or before the frontier is rejected; cancelling inside a scope remains pending q10; a change starts after the frontier | adopt | H4; §5.2, T-A, T-C. |
| R5 | Default strength: the trained and validated operating point; adherence measured and reported; checkpoint selection is part of the default | adopt | H2; §9.6. |
| R6 | §3.7 (lo, hi) encoding: ranges are gone | adopt | H1. |
| R7 | §3.2 "the whole song is [0, T)": the formulation's whole-song scope contains T | adopt | §5.1, §5.2; the EOS row-role frame reads a scope with b = T; test T-E. No head lies at T, because the EOS gap is at least 2 ms (`candidates.py:60-61`), so the labels are unchanged. |
| R8 | Undefined readouts are not zero; v3 already treats "unreadable" so | adopt, with a code finding | `labels.py:108-109` returns 0.0 for a scope that owns no object; `properties` returns undefined (§5.1, T-U). |
| R9 | Presence `none` and the token conditioner as an announce form (N1) agree with condition 1 | amend | Presence `none` is necessary, not sufficient (§4.2). The token conditioner breaks condition 2 as well: after b its token still carries the value with `started = 1` (`features.py:319-329`). It raises under the default η (§5.3) and is kept as an instrument for a future lead-in policy. |
| R10 | Difficulty with untrimmed tails: the readout counts closes decided after the scope, which generation does not steer | adopt | The label keeps them (ν unchanged, `labels.py:39`); the term does not own them (§6.3); stage 0 measures their share of the readout and a stated rule decides whether a trimmed ν is put to the human before stage 2 (§7.5). |

### 3.3 Rows added by this revision

| # | Point | Ruling | Argument and where it lands |
| --- | --- | --- | --- |
| V1 | The brief's case "a release at or after b decided at a row inside S" does not occur in R2 | correction | Decision k decides the gap before its row (`common.py:12-15`; `features.py:77`; `state.py:76-100`), so a decision at a row in S produces rows before b only. The boundary cases are the mirror ones: the onset decision may close holds before a, the exit decision may close holds inside S (§4.1). |
| V2 | Per-candidate masking | reject as the rule | It leaves the pointer normaliser and the joint action carrying S across the boundary (§4.3). Rule L is per decision. |
| V3 | v3's G3(a) compared the override run with the run without the override and made "no compensation" a binding guard | amend | §9.7 (a1)/(a2); guard (v) leaves §9.6; T-L2 on each run's frozen code is a stage-1 engineering test. |
| V4 | v3's resolver merged adjacent intervals with equal values | reject | Each target concerns its own section (`style-conditions-and-control.md` lines 161-164); merging changes the frame's bounds, progress and counters, and so the section the model is asked about. Adjacent requests stay separate (§5.2). |
| V5 | v3's schema accepted `{'named': ...}` style directives before any were trained | amend | Style directives are rejected until stage 4 builds the named-attribute path (§5.2); otherwise a request would be accepted and silently ignored (rule 2). |
| V6 | Own-sample terms (relaxed proxy, score function) on generated sections | amend | They score only Ω factors; decisions after b enter neither the score-function sum nor the relaxed-proxy gradient. Otherwise a request-dependent reward would train decisions that cannot read the request, which teaches holding or compensating through history (§7.3, stage 3). |
| V7 | A headless LN scope was rejected in v3 | amend | Accepted and flagged "unattainable: readout will be undefined", the formulation's declared shortfall (`style-conditions-and-control.md` lines 259-263; `notation.md` lines 237-242). Difficulty below 30 s stays rejected: ν is undefined there (§5.2). |
| V8 | No readout test that unspecified and explicit zero differ | adopt | §9.7 (f), with a baseline row at 61.4M (0.134 against 0.312 [data]). |
| V9 | No request set across continuation calls | adopt | §5.2, §11 item 5: the CLI and `continue_chart` take a request set with addition and cancellation boundaries, so a continuation keeps a request whose start lies before the new frontier (`style-conditions-and-control.md` lines 90-93) and T-L2 can be run. |
| V10 | A formulation chart seed may end inside a gap; R2's ends at a head row (g0 = t_{s−1}) | out of scope | An implementation support limitation (`notation.md` lines 329-340); recorded in the receipt's chart-seed field. |
| V11 | v3 ran its stage-0 baseline rows with requests on ckpt-0061432779, which was trained with presence `anywhere` and the birth role | amend | Natural rows use any code; rows with requests are generated by that run's frozen v1 code (`r2-runs/r2-ce-overnight-20261004/code`) and read by the v4 `properties` module, labelled "v1 conditioning" (§13, stage 0). |
| V12 | Decoding belongs to the default | adopt | Temperature 1 and the sampler are part of the operating point (`sampling.py:21-24`); a strength comparison would hold them fixed (`style-conditions-and-control.md` lines 308-310). Recorded (§5.7). |
| V13 | Stage R's return-to-natural test compared with pre-override statistics (the main thread's note in [s-plan-v3](r2-style-formulation-check.md#s-plan-v3)) | adopt | Compare with natural continuations under the same ρ from the same post-override history (`style-conditions-and-control.md` evaluation table) (§13, stage R). |
| V14 | FiLM width and the token path | adopt | Removing the birth role leaves 2 roles × 2 kinds per FiLM input; the token conditioner raises under the default η (§5.3, §6.8). |
| V15 | PR #16's prose says "explicit priority for overlapping requests" and lists "release without compensation" as a diagnostic | for the main thread | Both change here (H3, V3). The PR text is outside this plan. |

### 3.4 v3's rulings carried

| v3 row | v4 status |
| --- | --- |
| C1 property and style directives differ in kind | stands (§5.2) |
| C2 one `properties` module with ν | stands, extended by R8 and by the partial-materialisation, deviation, resolution and mirror declarations (§5.1) |
| C3 difficulty requested in absolute star, residual internal | stands (§7.2) |
| C4 natural and between-scope rows free | stands; its G3(a) test amended (V3) |
| C5 presence `none` | stands; not sufficient alone (R9) |
| C6 resolver by priority | superseded by H3 |
| C7 song time | stands; b = T contains T (R7) |
| C8 (lo, hi) reserved | superseded by H1 |
| G1 ρ reserved | stands; stage R amended (V13) |
| G2 η `immediate`; `announce`, `ramp` reserved; no `hold` | renamed (H6c): default η; transition intervals reserved; priority reserved and rejected |
| G3 diagnostics (a)-(e) | (a) amended (V3); (f) added (V8); (b), (c), (e) stand |
| G4 tagged style field | stands; style directives rejected until stage 4 (V5) |
| A1-A3 answers 1-3 | stand |
| A4 ranges unclear | closed by [d-target-strength](r2-style-formulation-check.md#d-target-strength) |
| N1 token conditioner is an announce form | amended (R9) |
| N2 seeds separated | stands |
| N3 readouts take any object list; provisional branches out of scope | stands |
| N4 explicit zero against unspecified (T-Z) | stands; readout diagnostic added (V8) |
| N5 ownership by birth | reversed (H5) |
| N6 cancellation keeps the pre-frontier part | replaced by H4: cancellation ends governance at g'; nothing before g' is rewritten because it is committed |
| N7 demand field reserved | stands |
| N8 no-chart-seed baseline folded into stage R | stands |
| N9 section-level target semantics | stands |
| N10 G3(e) for stage 4 | stands |

The adversarial-review rulings v3 carried (P1-P10, W1-W16 as ledger rows L41-L54) stand with one change: P6 (release factors read conditions across span boundaries) is now resolved by rule L, not by ownership by birth. Summary so this plan stands alone: whole-song LN requests are out of distribution, so the primary is in-distribution span following at onsets (P1); dropout comes before alignment, with inverse-probability weights (P2); the LN term is the governed split of the action factor, the base CE is uniform and emphasis is additive with weight 0 in stage 1 (P3); difficulty's direction-2 reading is put narrowly as Q-G with costs, the baseline is refitted on every interface length (P4); arms differ in the draw only, read at 50M on the mean of the last three checkpoints (P5); D3 is thresholded before the simulation, with a binomial z-criterion (P7); decisions belonging to the human are questions (P8); "CE recipe first" rests on the own-history shift and the natural LN bias, not on checkpoint oscillation (P9); the sync, manifest-reuse and panel-deviation fixes (P10).

<a id="v4-changes"></a>
### 3.5 Change log from v3

| Where | What changed |
| --- | --- |
| Title, header | "R2 v2 implementation plan (condition plan v4)"; authority moves to `docs/formulation/` at `bdbbbde`; decisions cited by shareable anchors only. |
| Terms | Request set with addition and cancellation boundaries; visible set V_k; onset and exit decisions; default operating point. |
| §2 | New code rows: decision geometry, candidates, half-open activity at T, FiLM layout, pointer normaliser, token conditioner after b, tiled star 0.0 on an empty scope, `replace_interval` unused, sampler, chart-seed boundary. Data rows re-read where the files are mirrored. Formulation summary added (§2.3). |
| §4 (new) | Locality analysis against both conditions; rule L; ownership equals visibility; T-V, T-L1, T-L2, T-C, T-E, T-A; v3's T-O removed. |
| §5.1 | ν adds: b = T contains T; undefined on an empty scope (tiled star) and on an open S-headed hold; deviation measure, resolution and mirror behaviour declared. |
| §5.2 | Validator replaces the resolver: rejection of same-quantity overlap (any ν), of a ≤ g_u, of every reserved field; no clipping, no merging; style directives rejected before stage 4; headless LN scopes flagged, not rejected; request set across calls; cancellation. |
| §5.3 | η: default only; priority and transition intervals rejected; token conditioner raises. |
| §5.5 | Strength: default only (replaces v3 §3.7 ranges). |
| §5.7 | Record fields per the formulation's generation record: strength, η, addition and cancellation boundaries, deviation, undefined, co-active targets, crossing objects, decisions in S that rule L masks, decoding, operating-point hash. |
| §6 | Ownership by visibility; difficulty term owns visible decisions only; frame with one value channel and no birth role. |
| §7.5 (new) | Tails after the scope: kept by ν, not owned by the term; stage-0 measurement and a stated rule. |
| §8 | "Natural decision" means V_k empty; the onset row of the draw is the first visible decision. |
| §9.6 | "The default operating point": selection rule, guards and decoding define the default; adherence report per property; guard (v) removed. |
| §9.7 | (a) split into an identity check (a1) and a persistence report (a2); (f) unspecified against explicit zero added. |
| §12 (new) | Formulation-to-plan map. |
| §13 | Stage 0 itemised, ≈ 3.3 working days of code (v3: 2.5); runs re-costed at ≈ 8.1 h (v3: 7.6 h); stage 1 ≈ 32.5 h; stage-1 thresholds drop G3(a) and add the locality tests on the frozen code; stage 3 restricted to Ω; stage R's return test amended. |
| §14 | L18, L32, L41, L45, L56, L59, L64, L65 updated; L66-L77 added. |
| §15 | Q-R withdrawn (settled); Q-B, Q-E, Q-F, Q-G kept, with Q-E and Q-G restated under rule L; q10 added as a non-blocking question; q2, q3, q7, q8, q9 listed at their written defaults. |

---

<a id="v4-locality"></a>
## 4. Locality under the formulation

### 4.1 What one decision produces [code]

Decision k < K produces, in time order, the close rows of the lanes it gap-releases at times strictly inside (t_{k−1}, t_k), then the complete row at t_k (`state.py:76-100`; code table `common.py:7-18`). Its release candidates C_k are built from t_{k−1}, t_k and the grid only (`features.py:72-78`, `candidates.py:58-109`), so they are known before the decision and are the same in teacher forcing and in sampling. EOS (k = K) produces only the closes of the lanes still held, inside (t_{K−1}, T), and has one legal action (`state.py:52-54`). Every head row places at least one head (`state.py:55-57`). Hence:

- After decision j is committed the boundary is g = t_j: the gap after t_j belongs to decision j + 1. A chart seed of s decisions has g0 = t_{s−1} (`sampling.py:35-42, 56`).
- The times D_k can produce are R_k = {t_k} ∪ C_k when some lane is held entering k, and {t_k} otherwise. R_K = C_K when a lane is held at EOS; with no lane held EOS has no choice and no factor.

A decision at a row inside S therefore cannot place a release at or after b: its releases precede its own row. The boundary cases are:

- **B1, onset decision** (t_{k−1} < a ≤ t_k): its gap may hold closes before a, which condition 1 protects, while its row lies in S. When a falls on a head row (LN piece bounds are beat times, `conditions.py:41`), every candidate of that gap precedes a.
- **B2, exit decision** (t_{k−1} < b ≤ t_k): its gap may hold closes in [a, b), inside S, while its row and its later candidates lie at or after b, which condition 2 protects.
- **B3**: a decision after the exit decision that closes a hold born in S (the decision of 2026-10-06).
- **B4**: EOS when b = T.

### 4.2 Each conditioning path against both conditions

| Path | Condition 1: rows before a | Condition 2: rows at or after b, given the rows before b | v4 |
| --- | --- | --- | --- |
| Row-role frame at t_k, applied by FiLM to the hand vectors, which feed the 625-way action head and the pointer base (`model.py:171-175, 177-184, 253-263`) | Holds for t_k < a: the frame is inactive (`features.py:293-295`). Fails at B1 when a lane is held and C_k has a time before a: the joint action reads S, so the probability that the held lane takes a gap-release code changes (`model.py:216-219, 233`), and with it the law of closes before a. | Holds: t_k ≥ b gives an inactive frame. At B4 the half-open test leaves EOS blind to a scope with b = T (`features.py:49, 293`), a missing read, not a leak. | Rule L (§4.4); the activity test closes at T when b = T. |
| Candidate-role frame at every candidate time, applied by FiLM to the pointer query per candidate (`model.py:275, 283-291`) | Holds before a. At B1 the frames at candidates in [a, t_k) are active; the failure is the row role's. | Fails at B2: candidates in [a, b) read S. The pointer normalises over all candidates of the factor (`model.py:309-319`) and its query carries the decision's codes (`model.py:259, 263`), so both the mass at or after b and the conditional law of the row at t_k given the closes before b change. | Rule L: the exit decision reads nothing of S. Kept inside visible decisions, where every candidate lies in S. |
| Birth-role frame at the held LN's start (`model.py:276`) | Holds: a hold closed before a was born before a. | Fails at B3 (v3's N5) and at B2: a hold born in S is closed by a decision that reads S through its start time. | Removed (H5). Under rule L it would carry S only inside visible decisions, where it repeats the row role plus the LN age the pointer already has (`features.py:215`, `candidates.py:52-55`). |
| Presence bit (`features.py:291`) | Fails: set on every row once any interval of the kind exists. | Fails likewise. | `none` by default (v3's C5). |
| Token conditioner (`features.py:308-330`; `model.py:185, 281, 290`) | Fails: each query sees every interval's value and offsets before it starts (`test_conditions.py:25-35` asserts it). | Fails: after b each token keeps the value with `started = 1`, `active = 0` (`features.py:319-320, 324-329`). | Raises under the default η (§5.3). |
| Committed counters (`features.py:264-272`) | Holds: read only inside an active interval. | Holds. | Kept. |
| History tokens and landmarks (`features.py:182-199`; `model.py:154-163, 188-192`) | Holds: no condition features; the request reaches them only through realised decisions. | The channel condition 2 allows. | Kept. |
| Query look-ahead (`features.py:241-248`) | Holds: skeleton only. | Holds. | Kept. |

### 4.3 Why per-candidate masking is not enough

At B2, mask the candidate frames at τ ≥ b and keep those in [a, b). For a held lane the pointer probability is Q(τ ∣ A) = exp s(τ, A) / Σ_{τ'∈C_k} exp s(τ', A). The scores at τ < b still depend on the request, so the mass left at τ ≥ b depends on it. And given a close at τ* < b, P(A ∣ τ*) ∝ P(A) Q(τ* ∣ A), where A sets the row at t_k ≥ b. Since s depends on A through the pointer query (`model.py:259-263`), the conditional law of the row at t_k given the closes before b depends on the request: condition 2 fails. Masking the candidates in [a, b) as well removes every read of S from the exit decision, which is the decision-level rule.

At B1 no candidate rule helps. The 625-way action (`model.py:216-219`) chooses "gap release" for a held lane jointly with the heads at t_k. If the decision reads S anywhere, the probability that the lane closes before a changes, and condition 1 fails. An exact alternative would split the boundary decision: first the closes before a without S, then the rest with S. That is a change of factorisation, deferred with the trigger in §4.4.

### 4.4 Rule L

**Decision D_k reads interval I, in every role and through every path, if and only if every time D_k can produce lies in I** (R_k ⊂ I, with I = [a, b) when b < T and [a, T] when b = T). Otherwise D_k receives no channel of I. V_k denotes the set of intervals D_k reads.

Equivalently, for k < K: I ∈ V_k iff a ≤ t_k < b (t_k ≤ T when b = T), and either no lane is held entering k or min C_k ≥ a. For EOS: I ∈ V_K iff a lane is held and C_K ⊂ I, which for b = T means min C_K ≥ a and for b < T needs every EOS candidate before b. V_k depends on the skeleton, the scopes, the cancellation boundaries and the occupancy entering k, so it is computed before the decision. A request cancelled at g' = t_j is in no V_k for k > j.

**Why the two conditions then hold.** Condition 1: decisions with t_k < a do not read I. The onset decision reads I only if it can produce no row before a, and otherwise reads nothing. By induction over k, every row before a comes from decisions that read nothing of the request, given histories that contain nothing of it. Condition 2: every decision with t_k ≥ b has t_k ∈ R_k outside I and reads nothing of the request; EOS reads I only when everything it can produce lies in S. The exit decision's law given its history is free of the request, so is its conditional law of the part at or after b given the part before b. Later decisions read nothing of the request given their history.

**What rule L costs** [to measure in stage 0 by `draw_sim.py`, §8.3]: (1) at an onset with an open hold and a candidate before a, the onset row's heads are counted by ν and not governed; (2) closes in [a, b) decided by the exit decision are not governed; (3) S-born holds closed after b are not governed, as decided. LN share depends only on heads, so (2) and (3) do not touch what an LN target can govern; (1) removes at most one head row per scope [inferred: at most 1 of ≈ 61 rows on a 16-beat piece at 3.8 rows per beat, ≈ 1.6 %]. Difficulty depends on all three (§7.5). **Stated rule:** if more than 5 % of in-scope head decisions are outside V on 16-beat LN scopes, a split boundary decision (§4.3) is designed and costed before stage 1; otherwise rule L stands.

### 4.5 Ownership equals visibility

Ω_I is the set of scored factors (the action factor and every release factor) of decisions D_k with I ∈ V_k. A term on I never scores a factor that cannot read I, and every factor that can read I is in Ω_I. v3's caveat "inputs are not ownership" no longer applies. Ω_κ is the union over intervals of kind κ. A factor may be in Ω_LN and Ω_star at once; attributing a behaviour change to one kind is a matter for the counterfactual diagnostics (§9.2), not for the loss.

<a id="v4-locality-tests"></a>
### 4.6 Locality tests (engineering, binding on stage 0)

Fixture: a chart with (i) an LN scope whose start falls on a head row entered with an open hold (B1, outside V); (ii) a scope whose start falls inside a gap with no hold open (B1, in V); (iii) an exit decision with an open S-born hold and candidates on both sides of b (B2); (iv) an S-born hold closed three decisions after b (B3); (v) a hold born before a and closed inside S; (vi) a whole-song scope with a hold open at EOS (B4); (vii) two adjacent scopes of one kind; (viii) an explicit LN 0 request; (ix) a star cell overlapping an LN scope. Tiny models with non-zero FiLM weights as in `tests/r2/helpers.py:87-94`; bit-identical unless stated.

- **T-V visibility, input reach and ownership (replaces v3's T-O).** (1) V_k equals the hand-listed fixture cases (i)-(vi). (2) For each fixture interval I, change its value, its bounds within its fixture case, or remove it: every factor log-probability of a decision with I ∉ V_k is bit-identical, and at least one factor of every decision with I ∈ V_k changes. (3) The loss code's Ω_I equals {factors of D_k : I ∈ V_k}. (4) Power: with rule L switched off (a test-only switch), (2) fails on cases (i) and (iii); with rule L off and the birth role restored, it also fails on (iv).
- **T-P1 targets.** Change the target of any scored factor not in Ω_κ that lies at or after the last decision carrying an Ω_κ factor in its window: L_κ unchanged. For L_LN additionally: change the release targets at the last Ω_LN decision; L_LN is unchanged, and L_star changes when that decision is also in Ω_star.
- **T-P2 gradient.** Backpropagate L_κ alone with hooks on the per-decision action log-probabilities and on the pointer scores: exact zeros at every factor outside Ω_κ, non-zero inside; for L_LN, zeros on every pointer score.
- **T-P3a normalisation.** Add a natural-only window to the batch: every L_κ unchanged (fixed divisors).
- **T-P3b input locality.** (i) Edit the value, bounds or existence of an interval I of kind κ that lies in no V_k of a decision with a scored factor: L_κ unchanged. Binding with no exception under presence `none`, rule L and FiLM; expected to fail under `anywhere`, the token conditioner or rule L off, and the test says so. (ii) Edit an interval J of another kind with J ∉ V_k for every decision carrying an Ω_κ factor: L_κ unchanged.
- **T-L1 condition 1 (free run).** On fixture charts × 3 random seeds × 2 chart seeds: generate with 𝒰 and with 𝒰 \ {u} from the same chart seed and random seed. The exported rows at times before a are identical, and the per-decision log-probabilities of the decisions before u's first decision in V are bit-identical.
- **T-L2 condition 2 (free run).** Generate with 𝒰 through the decision before u's exit decision; from that committed prefix continue with 𝒰 (u keeping its original addition boundary) and with 𝒰 \ {u}, same random seed. The rows from the exit decision on are identical, including the closes of S-born holds, and teacher-forced log-probabilities of the same continuation are bit-identical.
- **T-C cancellation.** Cancel u at g' = t_j: decisions after j are bit-identical to those under 𝒰 \ {u} given the same history; the record marks u cancelled with the readout of its committed part; a replacement starting at or before g' is rejected.
- **T-A validity.** Adding u at g ≥ a raises; at the start of generation a = 0 is accepted; after a chart seed with g0 = t_{s−1}, a ≤ g0 raises; a request already in the set is kept by a continuation call whose frontier has passed a.
- **T-E whole-song scope.** With b = T the EOS factors read the interval through the row role as well as the candidate role; a scope given with b = T contains T.

None of these is evidence about learning. Section 6.4 lists the loss tests that complete this set.

---

## 5. Requests, properties, scopes

### 5.1 The `properties` module and ν

One module, `properties.py`, declares ν = (version, LN-share semantics, difficulty semantics) and computes every property from an object list (taps and holds with lane, start and end) and a scope [a, b) in song milliseconds, b = T meaning [a, T]. ν is hashed; the hash enters the label file, the manifests, every record and request validation. A chart may be complete, a committed prefix, or a prefix with a provisional continuation: the function takes objects, not a run state, and has no request argument (`notation.md` lines 246-263).

- **LNShare_ν(H̄, S).** Heads are objects whose start time t satisfies a ≤ t < b (t ≤ T when b = T), one count per object. LN share = LN heads / heads. A hold whose head is in S belongs to S whatever its end, including a hold not yet closed; a hold whose head precedes a is not counted. Undefined (None) when S owns no head, never 0. Deviation: realised − target; resolution 1/n for n heads; invariant under the mirror. The row-level form from per-row counts (as `labels.ln_share`, `labels.py:42-45`, and the frame counters, `features.py:264-272`) and the object-level form (as `report.py:46-48`) are both implemented, and a test asserts they agree on 200 source charts and 2,000 random scopes.
- **Difficulty_ν(H̄, S).** The tiled star as built (`labels.py:78-110`): the objects whose heads lie in S, untrimmed tails, translated by −a, repeated with period b − a to 240 s, the 1 ms seam rule, scored by `compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)`. Context: the section's own objects only; the surrounding chart is not scored. Undefined when b − a < 30 s, when S owns no object (today 0.0, `labels.py:108-109`), when a seam cut leaves a nonpositive hold (`labels.py:91-96`), or while an S-headed hold is not closed. Deviation: realised − target in star; resolution that of the calculator's float output. Mirror behaviour [to measure]: T-M asserts equality within 1e-9 on 200 charts or records the observed difference, and ν declares the result. `TILING_VERSION` and the calculator name are part of ν.
- **Users of ν.** The LN-share draw value (`conditions.py:42-52` becomes a call); the star relabel (§8.2); the frame counters (§6.8; a test that the committed counters at decision k equal LNShare_ν on the committed prefix restricted to S); every readout in §9; request validation (§5.2). The probes that computed their own share (`probe.py:177-184`) are re-pointed to the module before they are frozen again (rule 5: the frozen evaluator is the hashed module, not a copy of its arithmetic).
- **Tests.** Labels equal readouts: for 200 fit_train charts, Difficulty_ν of the cache representation equals the stored label within 0.001 star except where the cache representation differs from the `.osu` (count reported, L13); LNShare_ν equals the draw value to 1e-9. Readout never echoes: generate with request v, overwrite the request record, the readout is unchanged; the readout of a source chart over any scope equals its label. **T-U**: LN share over a headless scope and tiled star over an empty scope are undefined, not 0. **T-M**: mirror behaviour as above.

<a id="v4-schema"></a>
### 5.2 Request schema and validation

```
Request:
  id        = str                                   # unique within the request set
  scope     = (a, b) in song ms                     # [a, b); b = T means [a, T]; 0 <= a < b <= T
  property  = {kind: Target}                        # kind in {ln_share, difficulty}; at most one per kind
  style     = None                                  # {'named': {attribute: level}} from stage 4;
                                                    #   'reference', 'edit' reserved tags
  eta       = Eta(priority=None, transitions=())    # the default η; any other value is rejected
  demand    = None                                  # reserved (N7); any value is rejected
Target:
  nu        = the module's ν id for the kind         # any other ν is rejected as unsupported
  value     = float                                 # LN share in [0, 1]; difficulty in absolute star
  strength  = 'default'                             # the only level built; any other is rejected
RequestSet entry: (request, added_at g_u, cancelled_at g' or None)
```

A request carries at least one directive. The scope is song time (answer 3): beats and bars are a caller convenience converted with the grid before the request is built.

**Validation** `validate(request_set, frontier, skeleton, ν) → (effective track per decision boundary, record entries)`, deterministic:

1. **Schema.** At least one directive; one target per kind; every reserved field at its default; ν equal to the module's ν for the kind; style directives rejected before stage 4 (V5).
2. **Validity.** g_u < a for each request at its addition: g_u = 0^- at the start of generation, otherwise the time of the last committed decision. A request already in the set keeps its g_u across continuation calls. A request added at or after its start raises `ContractError('request added at or after its start')`. A cancellation boundary satisfies g_u ≤ g' and is a committed boundary; a replacement starting at or before g' raises.
3. **Same-quantity overlap.** Two requests whose scopes intersect and that target the same property, under any ν, make the set invalid: `ContractError('same-quantity overlap')` naming both. From stage 4, two style directives specifying the same attribute likewise (q9 written default). Adjacent scopes do not intersect.
4. **Different-quantity overlap.** All apply. The record lists each co-active pair; no trade-off is claimed or trained, and each target's deviation is its declared shortfall.
5. **Feasibility flags** (recorded, never silently applied): LN value outside [0, 1] → rejected; an LN scope with no head row → accepted, flagged "unattainable: the readout will be undefined"; difficulty with b − a < 30 s → rejected (ν undefined there); difficulty with ∣v_res∣ > 1.5 star → flagged "outside trained residual range" and clipped in the frame; a scope length outside the trained set (§8.2) → flagged "extrapolation"; a scope no decision can read under rule L (no head row in S, or only a masked onset) → flagged "ungovernable".
6. **Output.** Per kind, the non-overlapping effective track in today's `Interval` form, one interval per target, with no merging (V4); `validate_track` keeps its non-overlap check as the invariant. V_k follows from it by rule L.

`replace_interval` is retired (H4). There are no mid-generation edits in v3's sense: adding is validation with the current frontier; changing is cancellation followed by a new request with a > g'.

**Training.** Source values never conflict, so the draw emits effective tracks directly, as if every request had been added at 0^-; a test asserts that every drawn track passes validation and maps to itself (`test_conditions.py:68-75` already checks non-overlap). The model never sees a request object, only the effective track filtered by V_k.

### 5.3 Transition policy η and priority

Only the default η is built: no priority and no transition interval. A request with the default η is read by decision D_k exactly when the scope is in V_k (rule L, §4.4). There is no compensation and no hold: a decision outside every scope of a kind sees the all-zero frame for that kind (§5.4).

Reserved, rejected until built: a **lead-in** interval before a (v3's `announce`; the token conditioner is one instance) and a **release** interval after b (v3's `ramp`). When built, each is a named input with its own lesion test, a draw that produces it in training, and a locality test restricted to its declared interval; the validity rule then also requires g < the lead-in start (`style-conditions-and-control.md` lines 176-177). A longer hold of a value is a longer scope, so there is no `hold` policy. **Priority** is rejected (H3). The token conditioner raises `ContractError` unless a lead-in policy is declared; its code and tests stay for that use.

### 5.4 Presence `none`

`features.py:291` goes behind a switch `presence ∈ {none (default), anywhere (v1 behaviour; test power checks and v1 reproduction only), lead-in (reserved, raises)}`. Under `none` channel 15 is zero; the channel stays as the reserved lead-in channel so FRAME_DIM is stable. A decision with no interval of a kind in V_k sees exactly what it sees with no request of that kind. `DEVIATIONS.md` items 2, 5 and 6 are rewritten (rule L, no birth role, presence `none`, the token conditioner's status). Tests: T-P3b(i), T-V, T-Z.

### 5.5 Strength: the default only

`Target.strength` admits only `'default'`; any other value raises `ContractError('strength levels above the default are not built')`. The default is the operating point of §9.6: what the selected checkpoint does under the stated decoding, with its adherence reported per property. It does not promise that a target is met exactly. Higher levels, their calibration and the strength comparison of `style-conditions-and-control.md` lines 321-329 are reserved. No level below the default exists (q2 written default); strength applies to property targets only (q3 written default).

### 5.6 Baseline style ρ: a reserved slot

`continue_chart(..., baseline=None)` and the request CLI accept a `baseline` argument whose only admissible value in this repair is `None`; any other value raises. The record states `baseline: none (not implemented; identity carried by committed history only)`. This is a declared deviation from the formulation, in which a baseline is always in effect (`style-conditions-and-control.md` lines 81-88). What ρ must satisfy and how it would be tested is stage R (§13); no representation is chosen here. Today an override enters the history and can shift what follows; G3(b) and the persistence report (§9.7 (a2)) measure how much.

### 5.7 Readouts and the generation record

Every generation (in-run panel, CLI, probe) writes one record with:

- code identity and the checkpoint hash (as `receipts.py`); the ν hash; the evaluator hashes; the **operating-point** hash (§9.6); decoding (sampler, temperature 1);
- the random seed; the chart seed (hash of the prefix decisions, number of decisions, g0 = t_{s−1}, open holds at g0);
- the baseline slot (§5.6);
- every request as given: scope, directives, ν, target values, strengths, η, g_u, g' if cancelled;
- the effective track per kind with provenance (which request), flags, and the decisions inside each scope that rule L masks (onset with an open hold; closes in [a, b) decided by the exit decision);
- per target: the readout under ν over S from the exported chart, the deviation (readout − target), undefined readouts as undefined, cancelled targets marked "cancelled at g'" with the readout of their committed part; for difficulty also b(S), v_res and the realised residual;
- co-active different-quantity targets with each one's deviation (the declared shortfall);
- objects crossing a scope end: S-headed holds closed at or after b (lane, start, end), so a readout can separate persistence from enforcement;
- the defect counts of §9.4.

The training receipt adds the conditioner type, the presence switch, rule L's version, λ and μ per kind, the IPW switch, the draw parameters and N̄ values, the manifest hashes. Rule 2: a component absent from the record is not in the system. Realised values are computed from exported objects, never copied from the request (test in §5.1).

---

## 6. The loss

### 6.1 Objects and ownership

- A chart is a head-row skeleton with K rows and song length T. Decision D_k at row k (k = K is EOS) factorises as one action factor (the 625-way masked joint action; at EOS it has one legal value and contributes 0) and zero or more release factors (one per gap-released lane, scored in two orientations, `model.py:4-9, 236-251` [code]).
- A condition kind κ has an effective track of intervals (a, b, value), half-open (closed at T when b = T), non-overlapping within a kind. The scope of an interval is its time support.
- **Ownership.** Ω_I is the set of scored factors of decisions D_k with I ∈ V_k (rule L, §4.4-4.5). A release factor belongs to the scope its deciding decision reads, not to the scope of its LN's head: a hold born in S and closed after b belongs to no term of S; a hold born before a and closed inside S belongs to S. ν still assigns a hold to the scope of its head (§5.1), so a readout can depend on factors its term does not own; the record lists those objects (§5.7). Ω_κ is the set of factors owned by some interval of kind κ.
- **Inputs are ownership.** A factor reads I if and only if it is in Ω_I (T-V). The emphasis term of I therefore scores exactly what I can govern.

### 6.2 Terms

Per scored decision j, ℓ_j = −(log P(A_j) + log Q(U_j)) as now (action plus release mixture; the decision is the unit). Per factor f, ℓ_f is its own −log-probability; for a kind κ with governed factor g_κ, ℓ_f^κ is the governed part (§6.3).

- **Base**: L_base = (1/N̄) Σ_j u_j ℓ_j over all scored decisions, u_j the inverse-probability weight of §8.1 for decisions with V_k empty and 1 otherwise, N̄ the fixed expected weighted count of scored decisions per batch (§6.6). This alone is the L20 fix: no window mean, no EOS or song-end over-weighting.
- **Emphasis per kind**: L_κ = (1/N̄_κ) Σ_{f ∈ Ω_κ} ℓ_f^κ, N̄_κ the fixed expected count of Ω_κ factors per batch.
- **Own-sample terms** (stage 3, §13): L_κ^own, which score Ω_κ factors only (V6).
- **Total**: L = L_base + Σ_κ λ_κ (L_κ + μ_κ L_κ^own). Defaults λ_κ = 0, μ_κ = 0: plain uniform CE. λ_κ = 1 is a labelled arm ("emphasis"), never the silent default: any λ > 0 changes the balance between natural and conditioned training, and so the default operating point. Every L_κ is computed and logged at λ = 0, so the per-kind numbers of §9.1 exist in every run.
- The DPO anchor uses L_base (L22).

### 6.3 Governed factors per kind

| Kind | Statistic the value is made of | Governed factor g_κ and its exact likelihood | Excluded from L_κ |
| --- | --- | --- | --- |
| LN share (property) | LNShare_ν over the scope | Per head row in V: the tap-versus-LN choice of each head given the head mask and release types. Group the 625 codes by r(a): per lane, codes 1 and 2 (free) merge, codes 3 and 4 (held) merge, all else distinct. log P(ℓ∣r, s, C) = log P(a) − log Σ_{a' : r(a') = r(a)} P(a'), a group log-sum-exp over the masked log-softmax; a precomputed 625 → group index and a scatter. The split is exact because `common.py:7-18` [code] gives codes 1/2 (free) and 3/4 (held) as the only tap-versus-LN pairs. Cost: none measurable. | Head mask, release types, release positions, EOS; every decision outside V (the masked onset, the exit decision and later). |
| Difficulty (property, residual internally, §7) | Difficulty_ν(scope) − b(scope) | The whole decision of every decision in V: its action factor and its release factors, which close holds inside S. Difficulty is a function of everything the decision sets, so no narrower governed factor exists. | Every decision outside V, in particular the closes of S-headed holds at or after b (H5) and the exit decision's closes in [a, b); ν still counts them (§7.5). |
| Named-attribute style directive (Lens attribute, ordinal level; stage 4) | the attribute's section judgment | The factor the attribute describes: head-mask sequence for jack, stream, trill (group by which lanes carry a head: per lane, free {1,2} and held {3,4} are "head", free {0} and held {0,1,2} are "no head"; the term scores log P(mask) = log Σ over the group); LN-ness plus owned releases for LN coordination; whole decision for tech. A style directive governs organisation, a property target governs the result; both may be active on one decision and their terms do not double count because their governed factors differ. | Per attribute; decisions outside V. |

Under reading (a) of direction 1 ("rows outside the span") L_LN could be the whole-decision NLL on decisions in V; under reading (b) ("aspects the condition does not govern") it must be the governed part. The governed form satisfies both, is cheaper to reason about, and removes double counting when a style attribute overlaps an LN scope. Both numbers are logged (Q-M, a default; stage 3 only).

### 6.4 Loss tests (engineering, binding on stage 0)

The locality tests of §4.6 (T-V, T-P1, T-P2, T-P3a, T-P3b, T-L1, T-L2, T-C, T-A, T-E) plus:

- **T-G governed split.** Σ over the group of exp(log P(ℓ∣r) + log P(r)) equals P(a) for every legal code; the LN share expected per row from the governed probabilities equals the value from the full action distribution.
- **T-Z explicit zero (N4).** For each property kind, the likelihood of a fixture decision in V under an explicit request equal to the kind's zero (LN share 0; difficulty equal to b(S)) differs from the likelihood under no request.
- **T-R validator.** Every drawn track validates to itself; overlapping targets for one property are rejected, also when one names another ν (rejected as unsupported at step 1, and the overlap check keys on the property, not on ν); adjacent requests are accepted and kept separate; an LN target overlapping a difficulty target is accepted and recorded as co-active; a non-default strength, any priority, any transition interval, any demand value, a style directive before stage 4, a request with no directive and a request added at or after its start each raise; a headless LN scope is flagged, a difficulty scope under 30 s rejected.

These are engineering tests; none is evidence about learning.

### 6.5 What the two readings of direction 1 change

Nothing in stage 1 (λ = 0). In stage 3 they decide whether the emphasis and own-sample terms for a statistic-defined kind score the governed factor (default) or the whole decision on decisions in V. Difficulty is the same under both.

### 6.6 Normalisation

N̄ and N̄_κ are expected counts per batch under the draw, measured once by the stage-0 simulation (2,000 draws), stored in the run config and receipt, recomputed whenever a draw parameter or rule L changes; a test asserts the stored values are within 5 % of a fresh estimate, and that Σ_j u_j over 200 batches averages N̄ within 5 %. Gradient accumulation divides by the number of windows as now; the divisor is constant across batches.

### 6.7 Decisions outside every scope (settled)

Membership is V_k. A decision with no interval of a kind in V_k is natural for that kind: it sees the all-zero frame for that kind, enters L_base with weight u_j when V_k is empty (§8.1), and is in no Ω_κ of that kind. Nothing is held, nothing compensates, no corpus-typical value is substituted: the source's own decisions are the only target on such decisions, which is consistent with the defaults because the source is one free continuation under its own baseline (C4). This includes in-scope decisions that rule L masks. A caller who wants a value held writes a longer scope. If a lead-in or release interval is built later, the decisions it touches get a named channel and a lesion test; whether they then join Ω_κ is part of that design. The loss code is the same in every case.

### 6.8 The frame per kind

Per kind: value (1) · offsets to the scope bounds (8) · progress (1) · committed statistics of the scope so far by ν (LN: log1p heads, log1p LNs, ratio; difficulty: the four proxies below), padded to a common width with constant zeros that are not named inputs · log1p remaining head rows (1) · active (1) · the reserved lead-in channel (1, zero under the default η). FRAME_DIM becomes 17. FiLM reads two roles (row, candidate) × two kinds, 4 × FRAME_DIM, after the birth role's removal (H5). All lane-free; the hand-swap identity test stays binding.

| Kind | Value encoding | Committed statistics | Lesion tests (stage 0) |
| --- | --- | --- | --- |
| LN share | 2v − 1 | log1p heads, log1p LNs, ratio (as now, by ν) | the value; each counter; remaining; active |
| Difficulty | v_res / 0.5, clipped to ±3 and flagged beyond ±1.5 star | mean chord size, same-lane repeat rate, held-lane occupancy, LN share over the committed part of the scope (mirror-invariant) | the value; each proxy; b depends on the skeleton only |
| Style attribute (stage 4) | one-hot {absent, supporting, prominent} | counts of the deterministic query evidence of the attribute (jack: same-column repeats; stream: directional four-note groups; trill: alternation of fixed groups; LN coordination: LN-occupied columns; tech: none) | per attribute; fixture kind in stage 4 |

Roles: the row role and the candidate role each get a lesion test; under rule L every candidate of a decision in V lies in the same scopes as its row, so the candidate role carries per-candidate offsets and progress, not other scopes. A style label is per section (half-open source milliseconds), multi-label, ordinal, with unresolved and unreviewed statuses (`gameplay-state.md` "Style observations", lines 191-299 at `bdbbbde`). Requested levels are present-supporting, present-prominent and absent; unresolved and unreviewed are not request values (`style-conditions-and-control.md` lines 45-53). The interface takes such a label when one exists; the labeller is not designed here; the attribute set is open by schema (G4).

---

## 7. Difficulty relative to the skeleton

### 7.1 Baseline b(S)

A quadratic ridge from head-time features of the span to Difficulty_ν of the span, as A fitted (23 features, R² 0.746, RMSE 0.528 star on fit_dev [data]), **refitted on fit_train cells of every length the draw and the interface use** (30, 60, 120 s and the whole song, §8.2), frozen and hashed, kNN-32 as the check that the linear form loses nothing (on A's fit, R² 0.753 against 0.746: none [data]). b is a function of heads in S only: tests that changing every decision of a chart leaves b(S) unchanged, that b is mirror-invariant, and that b(S) depends only on heads in S. b(S) is computed from the skeleton before generation for every difficulty target, by the validator (§5.2), and is not fed to the model (an optional arm feeds it; not adopted).

### 7.2 The residual as the internal condition value; absolute requests

A request carries an absolute target in star under ν. Internally v_res = target − b(S), clipped to ±1.5 in the frame and flagged beyond (§5.2). Corpus residual SD ≈ 0.53 star [data, A's RMSE]; the stage-0 refit prints the quantiles and the per-band width. Stage 0 changes `conditions.py:100` (negative values) and `features.py:276` (scale) (L49). The whole-song interval stays an interval with a residual value, not a chart-level scalar (rejected by the human on 2026-10-03). Readouts report absolute Difficulty_ν of the generated section, b(S), the realised residual and the deviation from the target. Long-term: b(S) needs the given skeleton, which R2 has and a timing-generating system will not; the residual is an R2 parametrisation and the interface does not depend on it (C3).

### 7.3 "The model's response is trained towards the loss": what it can mean, with costs

Costs relative to one CE step of four 256-row windows (≈ 0.37 s at 2,800 decisions/s [inferred from `run.json` and the pilot]); sampling at ≈ 338 rows/s [data, v3, `evals.jsonl` free-run median]; tiled star ≈ 0.02 s per span [data, label build].

| Term | What it trains | Sampling | Cost | Gaming risk and guard |
| --- | --- | --- | --- | --- |
| F1-res: CE on Ω_star factors with v_res in the frame (plus the committed proxies) | the likelihood of the source's own decisions under the residual it has | none | none | the proxies reveal part of the realised residual late in S; the history before S reveals the chart's offset if residuals correlate within a chart (§7.4) |
| TF-mono: teacher-forced monotone hinge on real onset states: g(E_θ[proxies ∣ s, r_hi]) − g(E_θ[proxies ∣ s, r_lo]) ≥ margin, g a frozen ridge from the committed proxies to the residual fitted on real cells | the direction of the value channel at onsets | none (two extra conditioner passes per row) | ≈ +20 % [inferred] | bounded and directional; surrogate gap tracked |
| Relaxed-proxy: sample S once from the real prefix with v_res, gradient through the per-row probabilities of decisions in V of (g(E_θ[proxies over S ∣ own history]) − v_res)² | the realised response on the model's own states | one sample per span, one window in four | ≈ +45 % on the step average [inferred] | hold-length features kept out of g; holds ≤ 60 ms guard; surrogate gap tracked |
| True F3: score function on −(Difficulty_ν(generated S) − b(S) − v_res)², summed over Ω_star factors only, with a leave-one-out baseline over k = 4 samples | the realised tiled star itself | 4 samples per span, run on past b until the last S-headed LN is released, the decisions after b natural and without gradient | ≈ 2.7× wall on one update in four [inferred] | fake holds raise star cheaply: guard (iv) and the CE anchor; k ≥ 4; the reward includes tails the request does not govern (§7.5) |

**What this plan proposes.** Stage 2 trains F1-res (B1 against B0 absolute) and offers TF-mono as a third arm B2. The relaxed-proxy term and true F3 are own-sample terms: they use no preference labels, so the DPO decision does not cover them, but whether a self-supervised term on the model's own samples may run before real pairs exist is the human's standing question (Q-I). The narrow question for the human is Q-G (§15).

**Realised response, how measured.** For a requested (S, target) on a panel chart: chart seed = the real decisions through the last decision before a; generate with the request until the last S-headed LN is released (decisions after b do not read the request), export, score Difficulty_ν(S) with the hashed module, subtract the same b(S), compare with v_res over requests {−0.5, −0.25, 0, +0.25, +0.5} star of residual (absolute targets b(S) + v_res in the request): slope, MAE and the deviation distribution in star units, with |Δ_tail| (§7.5) alongside. S is 30 s or 60 s.

### 7.4 Leak of the residual through the prefix

Stage 0 measures, from the refit's residuals, the within-chart correlation of adjacent cells. Rule stated now: if the correlation exceeds 0.5, star draws get an informativeness criterion like LN's (requested residual against the preceding cells' residual, weight 3 when ∣Δ∣ > 0.3 star) and the onset-row diagnostics for star are read against that stratum; otherwise star cells keep weight 1.

<a id="v4-tails"></a>
### 7.5 Tails after the scope

ν keeps untrimmed tails (`labels.py:39`), and the formulation allows it (`style-conditions-and-control.md` lines 103-105, 207-210). Under rule L the decisions that close S-headed holds at or after b, and the exit decision's closes inside [a, b), do not read the request. So part of Difficulty_ν(S) is set by decisions the target cannot govern. Consequences:

- CE training is unaffected: the label is the source's own value, and the scored factors are those in V.
- The realised-response measurement must still run past b until the last S-headed LN is released (§7.3); those decisions are natural, and the record lists the crossing holds.
- An own-sample term sees a reward partly set by natural decisions; it sums only over Ω_star (V6).

**Stage-0 measurement** on fit_train source charts, 2,000 cells per length (30, 60, 120 s): (i) the share of S-headed LNs closed at or after b; (ii) Δ_tail = Difficulty_ν(S) − Difficulty_trim(S), where the diagnostic variant cuts each S-headed hold at max(head + 1 ms, b − 1 ms), as the median and 90th percentile of ∣Δ_tail∣ per length; (iii) the same for closes in [a, b) decided by the exit decision. The trimmed variant is a diagnostic, never a target. Cost ≈ 5 min on one worker [inferred: 6,000 cells × 2 evaluations × 23 ms].

**Stated rule.** If the median ∣Δ_tail∣ on 30 s cells exceeds 0.10 star (about a fifth of the residual SD of 0.53 [data]), a trimmed ν is put to the human before stage 2; it would change the relabel and the label-equals-readout tests. Otherwise ν stays untrimmed and ∣Δ_tail∣ is reported with every stage-2 difficulty readout.

---

## 8. The draw

### 8.1 Steps (parameters are config; `draw_sim.py` measures them in stage 0)

1. Group uniform, chart uniform (unchanged).
2. **Candidates.** LN: a fresh random partition of the song into 8/16/32/64-beat pieces with at least one head (`conditions.py:34-52` returning all pieces); with p_long = 0.25 consecutive pieces are merged into runs of U{2..8} pieces; with p_whole = 0.10 the single candidate is [0, T]. Values are LNShare_ν of the source over the candidate (prefix sums, online). The value is the section's share even when the scored window covers part of the section (N9). Star: one cell length drawn from {30, 60, 120 s} and one phase from {0, 10, 20 s}; the cells of that (length, phase) partition the song (no overlap, so `validate_track` holds); with p 0.10 the whole song instead. Values are the cached Difficulty_ν of the cell (§8.2 relabel).
3. **Dropout first**, on the candidate lists: drop all with 0.20; drop a kind with 0.25; drop each candidate with 0.20. Nothing after this step removes an interval. A dropped candidate's decisions are unspecified for that kind: the all-zero frame, natural (C4).
4. **Alignment.** With p_align = 0.6, if any candidate survives: choose a kind uniformly among kinds with survivors; draw the lead u ∈ U{0..32}; for each surviving candidate of that kind take its onset row as its first decision in V and compute its informativeness against the 64 rows ending at j = max(0, onset row − u), rows the window never scores: z = ∣v − share_before∣ / SE_binomial(v, n_heads) with n_heads ≥ 20 required; importance weight 3 if z ≥ 2 (LN), weight 1 otherwise; star cells weight 1 (or the §7.4 rule); the whole song weight 1. Choose one candidate by weight; set j as above, stop = min(j + 256, K+1). Otherwise (p 0.4, or no survivor): the current start rule (`data.py:74-81`).
5. **Intervals per window.** For each kind: the aligned candidate (if that kind) plus others among the surviving candidates intersecting [t_j, t_stop), U{1..3} in total, consecutive with p 0.5. The track is the union over kinds and is an effective track by construction (T-R).
6. **Inverse-probability weight.** For every decision in the window with V_k empty, u_j = p_old(j ∣ chart) / p_new(j ∣ chart), with p_old the current start rule and p_new = 0.4 p_old + 0.6 P_align(j). Decisions with V_k non-empty get u_j = 1. The draw is a known mixture, so the ratio is exact per window [inferred range: ≈ 0.25-0.65 in aligned windows, 2.5 elsewhere]. Effect: the natural conditional and its state distribution are those of the current recipe (the design's rule, `design.md:303`), while conditioned decisions get the aligned exposure. A switch turns the weight off; off is a labelled arm.

Values that are not the source's own are **not CE targets**. Two values on one skeleton come only from the model's own samples (stage 3) or accepted alternative arrangements (Q-J, a default: not now).

### 8.2 Span lengths: training must cover what the interface promises

The interface is any scope in song time. Training today covers LN spans of 8-64 beats and star spans of 30 s, 60 s and the whole song [code]. Hence:

- **LN**: pieces, runs of pieces (up to 512 beats) and the whole song, at the probabilities in §8.1 step 2. The probabilities are Q-B; both stage-1 arms get the same lengths.
- **Star**: cells of 30, 60 and 120 s at 10 s offsets and the whole song (Q-C, a default). Relabel: ≈ 30 labels per chart, ≈ 375k labels, ≈ 36 min on 4 workers [inferred from 23 ms per label]; computed from the cache representation by the `properties` module (L13, C2). The label file carries the ν hash.
- **Evaluation spans** are drawn from the same lengths. A request outside them is flagged as extrapolation (§5.2).
- Difficulty_ν needs ≥ 30 s: star scopes are ≥ 30 s; the residual baseline is defined on the same spans.

### 8.3 Properties and the thresholds stated before the simulation

Measured by `draw_sim.py` on 2,000 draws against the cache (stage 0); each is also a test that skips when the cache is absent.

| Property | Threshold | Lever if missed (stated now) |
| --- | --- | --- |
| D1 onset coverage: among active intervals in a window, those with their onset row (first decision in V) scored | ≥ 70 % | p_align up to 0.75 |
| D2 song-length independence: active intervals per window regressed on K | 2-SE interval of the slope contains zero | none needed by construction |
| D3 informativeness: scored LN decisions in V within 16 rows of an onset with z ≥ 2 | ≥ 2 % of scored heads [inferred expectation 2-4 %] | importance weight 5, then keep every intersecting piece; natural decisions stay protected by u_j |
| D4 active share | LN ≥ 25 %, star ≥ 25 % of scored heads in V [inferred 25-45 %] | reported; set by the dropout rates and the per-window count |
| D6 natural weight: effective sample size of u_j over natural decisions | ≥ 50 % of the natural count | p_align down to 0.5 |
| D7 long-span coverage: scored decisions in V under LN spans longer than 64 beats | ≥ 20 % of LN-active decisions at p_long 0.25, p_whole 0.10 | Q-B |
| D8 rule-L cost: in-scope head decisions outside V, by kind and scope length; share of onsets entered with an open hold | ≤ 5 % on 16-beat LN scopes | split boundary decision designed before stage 1 (§4.4) |

**Two manifests**: a **natural manifest** (64 fit_dev windows by the current start rule, empty tracks, random seed 954 for the draw; draw-independent, shared by every arm; it carries the selection primary) and a **condition manifest** (64 windows by the §8.1 draw, stratified so each kind has ≥ 24 windows with a scored onset, ≥ 8 with a span longer than 64 beats, ≥ 8 with a value switch at a boundary, ≥ 8 with a masked onset; diagnostics only). Both are versioned by a hash of the draw parameters, the label file, ν and rule L's version (L50) and frozen before stage 1.

---

## 9. Diagnostics and the default operating point

All conditioned measurements use fixed panels, random seeds 954-956 unless stated, A's frozen `probe.py` re-pointed to the `properties` module and re-hashed, and a record per checkpoint (§5.7).

### 9.1 Teacher-forced, per kind and per factor (every log point; both manifests at every checkpoint)

NLL on Ω_κ factors, split: governed part and whole decision; strata onset (first 16 decisions in V of a scope), middle, end (last 16 decisions in V); informative onsets (z ≥ 2). The **null contrast** log p(D_j ∣ v) − log p(D_j ∣ ∅) on Ω_κ decisions by stratum. Natural-decision NLL on the natural manifest. Train-versus-dev NLL on onset decisions (memorisation of up-weighted onsets). The count of in-scope decisions outside V, by stratum.

### 9.2 Counterfactual response on fixed real states (every checkpoint)

A's P1 probe on two strata, onset decisions and late decisions in V: same state, value swapped (LN 0 / 0.9; residual −0.5 / +0.5): expected statistic change (governed probabilities for LN) and action KL. A late-decision response without an onset response means the model follows counters, not the value.

### 9.3 Representation probe (per checkpoint, diagnostic only)

Linear probe from the hand vector at onset decisions to v.

### 9.4 Free-run panels (every second checkpoint; records)

Charts × random seeds. The **prefix panel** is 16 fit_dev charts (A's 8 plus 8 across star bands and lengths to 1,500 rows) with the chart seed = the real decisions through the decision before t_{1/3}, the head-row time nearest one third of the song (so g0 < t_{1/3}); it serves the natural guard and §9.7.

| Panel | Charts × random seeds | Requests | Metrics | Informs |
| --- | --- | --- | --- | --- |
| Natural from BOS | 16 × 3 | none | LN share against the source panel (reported, not binding: no prefix to pair on), B's `drift_ln_share`, holds ≤ 60 ms, releases 1-40 ms before another head, legality, chord-histogram JS | defect guard; descriptive |
| **Natural continuation (prefix panel)** | 16 × 3 | none | LNShare_ν of the generated continuation over [t_{1/3}, T] against LNShare_ν of the real continuation of the same prefix over the same span; chart-paired mean difference and SE (2,000 chart-bootstrap resamples); SD ratio generated/real across prefixes | **guard (i), binding** |
| **Span following at onsets (stage-1 primary)** | A's 8 × 3 | 16- and 32-beat scopes at the beat-partition boundaries nearest 1/3 and 2/3 of the song; values {0.05, 0.3, 0.6, 0.9}; chart seed = the real decisions through the last decision before a; generation through the scope until the last scope-headed LN is released | realised LNShare_ν over the scope; slope, MAE and the deviation distribution; per length; the share of masked onsets | the stage-1 claim |
| Whole-song LN | 8 × 3 | [0, T] at {0, 0.1, 0.3, 0.6, 0.9} | slope, MAE, deviation distribution | generalisation to the longest scope; reported |
| Switch | 8 × 3 | two adjacent requests 0.1 then 0.6, and 0.6 then 0.1, at half song, both added at 0^- | DiD/2, per-half MAE; whether the boundary decision was masked | boundary response |
| Residual star (stage 2 on) | 8 × 3 | absolute targets b(S) + {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s onset-aligned scopes | slope, MAE and deviation in star; ∣Δ_tail∣ (§7.5) | the stage-2 claim |

### 9.5 Teacher-forced against own-history calibration (every second checkpoint)

B's M1H: expected LN forecast on real histories against the same on own-sampled histories from the same start states (16 charts × 96 states). The exposure-bias number.

<a id="v4-operating-point"></a>
### 9.6 The default operating point (fixed before any run)

The formulation's default strength is the operating point a generator is trained and validated for (H2). For R2 v2 that point is defined by four things, frozen and hashed together with the evaluators (rule 5); the hash enters every record:

1. **The recipe**: the arm's training config (hash), with λ and μ as given (λ > 0 is a different default).
2. **The selection rule** below, with its guards.
3. **Decoding**: the Gumbel-maximum sampler at temperature 1 on the masked log-softmax, a fair orientation, the pointer one lane at a time (`sampling.py:21-24, 64-79`), no truncation.
4. **Conditioning**: rule L, presence `none`, the default η, FiLM.

Changing any of the four defines a different generator whose default adherence is measured again.

**Selection rule.**

1. Candidates: checkpoints after warm-up at regular cadence (safe checkpoints excluded, L47) whose natural and conditioned free runs are all legal with every head present, and whose code passes the stage-0 suite, the locality tests included.
2. Primary: natural-manifest per-decision NLL, read as the mean over the checkpoint and its two predecessors (checkpoint-level noise 0.02-0.03 [data, v3], L48), with a paired song-group bootstrap SE (2,000 resamples).
3. Guards, each a part of what "default" means for this generator: (i) **binding**: on the prefix panel the chart-paired mean difference of continuation LN share (generated − real) is within ±0.05, and the SD ratio is reported (a ratio outside [0.5, 2] is a flag, not a failure, until a threshold is justified); never per chart against the chart seed; (ii) span-following slope ≥ 0.7 and MAE ≤ 0.15 on the onset panel, whole-song slope reported; (iii) own-history gap ≤ 0.05; (iv) holds ≤ 60 ms ≤ 0.5 % and releases 1-40 ms before another head within the source band's rate. v3's guard (v) (no compensation) is gone: under rule L it is a property of the code, checked by T-L2 and by §9.7 (a1).
4. Select the earliest candidate passing all binding guards whose primary is within 2 SE of the minimum among passing candidates. None passing: "no selection", with the failing guard; such a checkpoint serves mechanism tests only.

**Reading of the balance.** The formulation's default balances playability, style and control. In R2 v2 the balance is lexicographic: legality, the defect ceilings (iv), the natural-continuation guard (i) and the control floor (ii) first, then the natural NLL. Style enters only through the natural NLL and is otherwise unmeasured (G3 statistics are reported, not guarded). The guards are this plan's choice of operating point, not a pass threshold on adherence.

**Default adherence report** (one file per selected checkpoint, hashed, referenced by every record): per property and scope length, the deviation distribution (mean, SD, 10/50/90th percentiles, mean absolute deviation) over the onset, whole-song, switch and (from stage 2) residual-star panels, with undefined readouts counted separately and masked-onset scopes marked. This is the generator's reported adherence at the default; no further threshold applies to it.

Read against the overnight run [data]: the BOS natural share was 0.312 against 0.187 at 61.4M, so guard (i) in its old form failed there; the prefix-panel form has not been measured on any checkpoint and is a stage-0 measurement on ckpt-0061432779 before stage 1 (a baseline row, rule 4).

<a id="v4-g3"></a>
### 9.7 Diagnostics for the formulation's targets (G3; every second checkpoint unless stated; records)

All on the prefix panel of §9.4 (16 prefixes), random seeds 954-956 paired across conditions (the same random seed with and without the request), 2,000 chart-bootstrap resamples for SE. Organisation statistics, all by the `properties` module or the deterministic Lens query evidence: chord-size histogram (JS divergence against the comparison run), same-lane repeat rate (jack evidence), directional four-note run rate (stream), fixed-pair alternation rate (trill), hand-role balance (share of heads on outer against inner lanes, left against right), LN share.

- **(a1) Release is free: the formulation's comparison (engineering, at every selected checkpoint).** The formulation compares readouts after the scope with natural continuations from the same post-scope history (`style-conditions-and-control.md` evaluation table). Override: an LN request on [t_{1/3}, t_{1/2}) with value 0.9 if the prefix's LNShare_ν < 0.5 else 0.05. On 4 prefixes × 3 random seeds, from the override run's history through the decision before the exit decision, continue 64 decisions with the request kept in 𝒰 (it expires on the way) and with it removed: rows and per-decision log-probabilities must be bit-identical. Failure means rule L is broken in the deployed code, and the checkpoint is not selectable.
- **(a2) Persistence through history (reported, no claim).** v3's Δ_post: LNShare_ν(override run, [t_{1/2}, T]) − LNShare_ν(natural run, [t_{1/2}, T]), chart-paired mean and SE, with the same override, prefix and random seed; holds born under the override and closed after t_{1/2} belong to the override scope by ν and do not enter Δ_post. Under rule L any difference is carried by the realised history, which the formulation permits; with no ρ in R2 it says how strongly an override's history steers what follows, an input to stage R. Also reported: the override scope's realised share, and the defect rates (holds ≤ 60 ms, releases 1-40 ms before another head, legality) in the 32 decisions after t_{1/2} against the natural run's rates on the same rows (continuity at the boundary).
- **(b) Identity kept under a property change (reported in stages 1-2; binding in stage R).** Same prefix and random seed with and without an LN request of 0.3 on [t_{1/3}, t_{2/3}). Over that scope, the paired change in each organisation statistic the property does not govern (all but LN share). Reference spread: the same statistic's difference between two natural runs of the same prefix with different random seeds (954 against 955, 955 against 956). Identity is "kept" for a statistic when the mean paired change under the request is < 2 SE of the reference spread's mean absolute difference. Reported per statistic, with the LN following in the scope so a "kept identity with no following" is visible.
- **(c) Within- against between-identity variation (at 25M, 50M and the selected checkpoint; reported, no claim).** 16 prefixes × 8 random seeds (954-961), natural continuations. Per statistic: variance across random seeds within a prefix (mean over prefixes) against variance of per-prefix means across prefixes; the ratio with a bootstrap SE. The 39.49M number (across-random-seed variance twice across-skeleton, one checkpoint [data, v3]) is the baseline row. The formulation's target (different identities differ, each supports several realisations) has no R2 mechanism yet; the number says where the model is. Recognisability is not claimed (`style-conditions-and-control.md` lines 457-460).
- **(d) Records.** §5.7; the test that realised values are computed, not echoed (§5.1).
- **(e) Properties kept under a style directive (stage 4 only).** With a fixture style attribute active and an LN request of 0.3 on the same scope: LN following within guard (ii); the mirror of (b).
- **(f) Unspecified against explicit zero (reported in stages 1-2).** Same prefix and random seed with an LN request of 0 on [t_{1/3}, t_{2/3}) and with no request: paired difference of LNShare_ν over the scope; distinct when the difference is beyond 2 SE. Baseline row: at 61.4M the whole-song panel gave 0.134 ± 0.020 under a request of 0 against 0.312 ± 0.038 natural [data, `control-table.md`]. "Absent" for a named concept is stage 4.

Cost: (a2) 96 continuations of about two thirds of a song and (b) 48 more, ≈ 1.6 s each [inferred from 338 rows/s and a median of 500 rows], ≈ 4 min; (f) 48 runs over a third of a song ≈ 1 min; (a1) 12 short continuations ≈ 0.2 min; (c) 80 further runs ≈ 2 min at its three checkpoints only. The natural continuations are shared between guard (i), (a2), (b) and (f).

### 9.8 Cost of in-run evaluation [inferred from `execution.json` and `evals.jsonl`]

Per full evaluation: manifests ≈ 10 s; natural from BOS 48 runs ≈ 2 min; prefix panel natural 48 runs ≈ 1.3 min; onset panel 384 short runs ≈ 3 min; whole-song 120 runs ≈ 5 min; switch 48 runs ≈ 2 min; calibration ≈ 5 min; G3 (a1), (a2), (b), (f) ≈ 5.2 min; residual star (stage 2) 240 spans ≈ 4 min. About 24 min per full evaluation in stage 1 and 28 min in stage 2. A 50M run checkpoints every 4.39M, so 12 checkpoints, a full evaluation at every second one (6), the teacher-forced manifests at the other 6 (≈ 2 min in all) and G3(c) at three (≈ 6 min): ≈ 2.5 h of evaluation in stage 1 (≈ 2.9 h in stage 2) on top of 5.6 h of training at 2,500 decisions/s, ≈ 8.1 h per run (≈ 8.5 h in stage 2). v3's 7.6 h counted about five full evaluations; the recount and the new diagnostics add ≈ 0.5 h.

---

## 10. Exposure bias

Teacher-forced forecasts are calibrated and own-history forecasts sit 0.129 ± 0.044 above them at 61.4M [data]; natural LN share from BOS is +0.10 to +0.13 above the source on three panels [data]. The plan keeps **check first, train second**: stage 1 measures the gap at every second checkpoint in both arms; stage 3 trains the relaxed own-history LN term (gradient through the per-row governed probabilities of decisions in V on own-sampled spans; the sampling path carries no gradient) only with the human's permission (Q-I). This is not DPO and uses no preference labels; it is not scheduled sampling either.

---

## 11. Infrastructure

1. **Guard**: keep the RSS limit (12 GiB) and `min_available_bytes`; the system swap-growth trip (`runtime.py:129-133` [code]) measures other processes: default (Q-D) remove it and trip on the trainer's own RSS growth > 2 GiB between checkpoints; record `available_bytes`, `pressure_level` and RSS in every `resource_limit` event. **For the main thread**: the mac's `resources.jsonl` says whether the trainer's RSS grew across the 13 hours before the first trip.
2. **Restart budget as a rate**: at most 5 restarts in any 6 hours; a resume that reaches the next checkpoint clears the counter (`launch.py:27, 106-112` [code]).
3. **Log segmentation**: `train.jsonl` and `resources.jsonl` per checkpoint segment, each well under 4 MB; `resources.jsonl` compacted. **For the main thread** (L36): the existing run's two oversize files are re-staged by mutagen every cycle (`mutagen.yml:161, 179-182` [code]); a pattern exclusion for `r2-runs/**/train.jsonl` and `resources.jsonl` is a sync patch outside R2's code.
4. **Plateau stop**: available in the trainer, **off in every arm comparison** (L51).
5. **Generation entry point**: `continue_chart` and a request CLI take a request set (§5.2: each request with g_u and an optional g'), a chart seed, a random seed and a `baseline` slot; each call validates, applies rule L per decision, and writes the record of §5.7 next to the `.osu`. A continuation call passes the request set on, so a request whose start lies before the new frontier stays in force (V9). §9 runs in `evaluate()`.
6. **Memory**: a 200-window pilot recording peak RSS per window against factor × candidate counts (L37); float32 per-candidate arrays and per-factor chunking if confirmed.
7. **Records**: `receipts.py` gains the generation fields of §5.7; `train_ce.py:331-339` gains the training fields; a test that every field named in §5.7 is present and non-null where required.

---

<a id="v4-formulation-map"></a>
## 12. The formulation mapped to the plan

Sources are at `bdbbbde`: `style-conditions-and-control.md` unless another file is named; line numbers in parentheses. "Test" is what fails if R2 v2 violates the requirement.

### 12.1 Implemented by R2 v2

| Requirement | Where in the plan | Test or evaluator that fails |
| --- | --- | --- |
| Style and properties are different kinds; a target fixes what the result measures, not its organisation (15-41) | §5.2 schema, §6.3 governed factors | T-R (style rejected before stage 4); T-G |
| Readouts under ν use the targets' definitions, depend only on chart content, never echo the request (95-123; `notation.md` 246-263) | §5.1 | labels-equal-readouts; readout-never-echoes; row/object LN-share agreement |
| ν declares undefined readouts, which are not zero (110-111, 259-261) | §5.1 | T-U |
| ν declares the deviation measure, the resolution and the mirror behaviour (112-116) | §5.1, §5.7 | T-M; record completeness |
| A readout over a partly materialised scope describes what it receives; ν assigns an open hold (121-123) | §5.1 | properties test: an open S-headed LN counts for LN share; tiled star undefined until closed |
| A request carries at least one directive, at most one target per property, one value per target (127-155) | §5.2 | T-R |
| S = [a, b), with T included when b = T; a row at a is in scope, a row at b < T is not (135-140) | §5.1, §4.4 | T-E; T-V |
| Scopes are independent of generation windows and continuation calls (141-143; 90-93) | §5.2 request set; §8.1 windows around spans | T-A (request kept across calls); T-L2 |
| A target concerns its section as a whole; steering later rows of an active scope is not compensation (161-164, 179-182) | committed counters over S (§6.8); N9 | counters-equal-ν test |
| Validity g < a; adding at or before g is rejected; a = 0 allowed at the start; a > g0 after a chart seed (168-182) | §5.2 step 2 | T-A |
| Cancellation at g' governs through g'; the record marks it; a change is a new request after g' (184-193) | §5.2, §4.4, §5.7 | T-C |
| Expiry and cancellation leave history, holds and the baseline unchanged (192-193) | rule L removes inputs only; nothing committed is rewritten | T-C; T-L2 |
| Locality condition 1, including no-row decisions (200-201, 205-206) | rule L, presence `none`, token conditioner disabled | T-L1; T-V; T-P3b |
| Locality condition 2, including the close of a hold started in S (202-210) | rule L, birth role removed | T-L2; T-V case (iv) |
| Objects crossing a scope end are listed (209-210, 439-441) | §5.7 | record completeness |
| Same-quantity overlap is invalid: same property under any ν; same style attribute (223-229) | §5.2 step 3 | T-R |
| Different-quantity overlaps all apply; a shortfall is declared (231-241) | §5.2 step 4, §5.7 | T-R; record completeness |
| Unspecified, explicit zero and absent differ; an explicit zero is a target (247-264) | T-Z; §9.7 (f) | T-Z (mechanism); (f) reported |
| Default strength is the trained and validated operating point; adherence reported per property (273-280) | §5.5, §9.6 | adherence-report completeness; operating-point hash in every record |
| Strength is not temperature; a strength comparison holds temperature fixed (308-310) | §9.6 decoding; §5.7 | record completeness |
| Chart seed: a committed, legally replayable prefix whose open holds are inherited (335-344; `notation.md` 125-133) | `continue_chart` prefix (`sampling.py:35-42`) | existing replay check; T-A |
| The seed's statistics are not continuing targets (351-352) | guard (i) at panel level, never per chart against the seed (§9.6) | guard (i) definition |
| The random seed is distinct from the chart seed and from identity (362-367) | Terms; §5.7 | record completeness |
| A free property is not held, compensated or pulled toward a population or seed value (371-378) | presence `none`, rule L, base CE with IPW on natural decisions | T-L2; §9.7 (a1); guard (i) |
| "Is a released property free?" against the same post-scope history (450) | §9.7 (a1) | (a1) |
| "Is a request invisible outside its scope?" (449) | rule L | T-L1; T-L2 |
| "Does a target hold?" as a deviation distribution across random seeds, undefined kept apart (447) | §9.4; §9.6 adherence report | adherence report |
| The generation record (427-441) | §5.7 | record completeness |
| Probability one on legal continuations; no request fixes a row or overrides committed decisions (`notation.md` 217, 237-238) | support mask independent of conditions (`state.py:49-57`); conditions enter through FiLM only | existing legality and replay suites |
| 𝒰 is retained across calls and is not chart state (`notation.md` 160-163, 223-227) | request set object (§5.2, §11 item 5) | T-A; T-L2 |
| Demand requests keep their own semantics; their interface is open (`gameplay-state.md` 364-372, 384-392) | `demand` reserved | T-R (any value raises) |
| Tendency, adherence, value, strength, demand and temperature are separate (`gameplay-state.md` 374-382) | strength default only; temperature recorded | record completeness |

### 12.2 Not implemented

| Requirement | Status | Where |
| --- | --- | --- |
| Strength levels above the default; their calibration and comparison (281-329) | reserved: `strength` slot, any non-default value rejected | §5.5; T-R |
| Priority among different-quantity directives (233-241) | reserved: rejected, R2 has no mechanism to honour it | §5.3; H3 |
| Transition intervals in η, lead-in and release (176-177, 212-216) | reserved: rejected until built with a channel, a draw and tests | §5.3 |
| The baseline ρ, always in effect, established and retained, not chart state (81-88, 345-360) | reserved slot (`baseline=None` only); declared deviation in every record | §5.6; stage R |
| Style directives on named concepts (45-53) | deferred to stage 4 (named attributes, engineering only); `reference` and `edit` tags reserved | §5.2; stage 4 |
| Identities differ and each varies (403-423; evaluation 452) | deferred: G3(c) reported now, the mechanism in stage R | §9.7 (c); stage R |
| Return to natural reaches the baseline (380-390; evaluation 451) | deferred to stage R, with the formulation's comparison (V13) | stage R |
| Local overrides do not rewrite the baseline; a baseline update is explicit (355-360, 400-401) | deferred to stage R | stage R |
| Targets kept under a style change (evaluation 454) | deferred to stage 4 | §9.7 (e) |
| Recognisability of identity or style by humans or a validated recogniser (457-460) | out of this repair; G3 statistics claim nothing about it | §9.7 |
| Audio as a generation input (72-77; `notation.md` 13-15) | out of scope: R2 is heads-only | PR #16 |
| Provisional branches and prefix commit (`notation.md` 287-327) | out of scope: R2 commits decision by decision (`sampling.py:56-87`); readouts accept provisional object lists | §5.1 (N3) |
| A chart seed whose boundary lies inside a gap (`notation.md` 113-115) | out of scope: a support limitation (V10) | §5.7 chart-seed field |
| Style references, their extraction and reuse (362-364, 405-406) | out of scope | — |
| A continuous-time probability measure (`notation.md` 281-285) | out of scope: R2's finite candidate support `cand-v1` is a support limitation (`notation.md` 329-340) | — |

---

<a id="v4-stages"></a>
## 13. Stages, re-costed

Throughput 2,000-4,000 decisions/s on 4 threads; budgets at 2,500/s. One training seed is labelled as such until the second runs. Code estimates are working days [inferred].

### Stage 0: code, measurements, tests (≈ 3.3 working days of code; ≈ 1.5 h of mac)

| # | Item | Sections | Days | Against v3 |
| --- | --- | --- | --- | --- |
| 0.1 | Ownership by visibility and per-factor terms with the governed split | §4.5, §6.1-6.3 | 0.25 | changed: visibility replaces birth |
| 0.2 | Uniform base CE with fixed divisors and the inverse-probability weights | §6.2, §8.1 | 0.10 | as v3 |
| 0.3 | The §8.1 draw with span lengths behind config; `draw_sim.py` with D1-D8 | §8 | 0.20 | D8 added |
| 0.4 | Relabel at the new lengths from the cache representation through ν | §8.2 | 0.05 | as v3 |
| 0.5 | Baseline refit, frozen and hashed; residual quantiles; within-chart correlation | §7.1, §7.4 | 0.05 | as v3 |
| 0.6 | Frame generalisation: one value channel per kind, the star proxies, the residual encoding; `conditions.py:100`, `features.py:276`; FRAME_DIM 17 | §6.8, §7.2 | 0.10 | (lo, hi) dropped |
| 0.7 | Two manifests with version hashes | §8.3 | 0.05 | masked-onset stratum added |
| 0.8 | Teacher-forced and free-run evaluation of §9.1-9.5 | §9 | 0.10 | as v3 |
| 0.9 | Guard, restart budget, log segmentation | §11 items 1-3 | 0.05 | as v3 |
| 0.10 | DPO anchor on L_base; DEVIATIONS entry for the panel (L46) | §6.2 | 0.05 | as v3 |
| 0.11 | `properties` module with ν: hash; undefined readouts (fixing `labels.py:108-109`); b = T; open holds; deviation, resolution, mirror; labels equal readouts; T-U, T-M | §5.1 | 0.30 | v3 0.25, + 0.05 |
| 0.12 | Request schema and validator: reserved fields rejected; same-quantity overlap rejected; g < a; no merge; flags; request set with addition and cancellation boundaries across calls; `replace_interval` retired; T-R, T-A, T-C | §5.2-5.3 | 0.35 | v3's resolver 0.30 |
| 0.13 | Absolute difficulty targets through the validator | §7.2 | 0.10 | as v3 |
| 0.14 | Presence switch with `none` default; T-Z | §5.4, §6.4 | 0.10 | as v3 |
| 0.15 | Rule L in `window()` and `continue_chart`; birth role removed (FiLM 4 × FRAME_DIM); activity closed at T; token conditioner raises under the default η; `DEVIATIONS.md` items 2, 5, 6 | §4.4, §5.3-5.4 | 0.30 | new |
| 0.16 | Locality tests: T-V with power checks, T-P1 to T-P3 restated, T-L1, T-L2, T-E | §4.6 | 0.25 | new |
| 0.17 | Generation record with every §5.7 field; completeness test | §5.7, §11 item 7 | 0.20 | v3 0.15, + 0.05 |
| 0.18 | Prefix panel, guard (i) in its binding form, G3 (a1), (a2), (b), (c), (f) in `evaluate()` and the CLI | §9.4, §9.7, §11 item 5 | 0.45 | v3 0.40, + 0.05 |
| 0.19 | Operating-point definition and default adherence report, hashed | §9.6 | 0.10 | new |
| 0.20 | `baseline` slot | §5.6 | 0.05 | as v3 |
| 0.21 | Style-attribute naming in the frame code (directives still rejected) | §6.8 | 0.05 | as v3 |
| 0.22 | Tail-share measurement script | §7.5 | 0.05 | new |
| | **Total** | | **3.30** | v3 2.5; removed: (lo, hi) encoding 0.10 |

**Mac measurements** [inferred durations]:

| Measurement | Produces | Time |
| --- | --- | --- |
| `draw_sim.py`, 2,000 draws | N̄, N̄_κ, D1-D8, the u_j distribution | ≈ 10 min |
| Relabel at 30/60/120 s × 3 phases plus the whole song | the label file with the ν hash | ≈ 36 min on 4 workers |
| Baseline refit | b(S), residual quantiles, within-chart correlation | seconds |
| Tail share (§7.5) | ∣Δ_tail∣ per length; share of S-headed LNs closed after b | ≈ 5 min |
| 200-window pilot | throughput, peak RSS, rule L overhead, cost of the TF-mono and relaxed-proxy forward paths on one window in four | ≈ 15 min |
| Baseline rows on ckpt-0061432779 | guard (i) and G3(c) natural rows (any code); G3 (a2), (b), (f) by the run's frozen v1 code (`r2-runs/r2-ce-overnight-20261004/code`, which has the presence bit and the birth role, `features.py:291`, `model.py:276` there [code]), read by the v4 `properties` module and labelled "v1 conditioning" | ≈ 16 min on 2 threads |
| Stage-0 suite on the mac | all tests below | ≈ 10 min |

**Tests that fail if the stage is wrong** (engineering): input lesions per named input on a tiny model (`tests/r2/helpers.py` pattern), one forward each: the LN value; each LN counter; remaining; active; the residual value; each star proxy; the row role; the candidate role; presence under the `anywhere` switch; a documented "unused" input is not accepted. T-V (with its power checks), T-P1, T-P2, T-P3a, T-P3b (binding), T-L1, T-L2, T-C, T-A, T-E, T-G, T-Z, T-R, T-U, T-M. D1-D8 on 2,000 draws. Weight-sum and stored-N̄ tests. b skeleton-only, mirror-invariant, span-local. Labels equal readouts; readout never echoes; record completeness; `baseline` other than none raises; `demand`, priority, transition intervals and non-default strength raise. `test_conditions.py:55-65` replaced by T-A and T-C; `test_conditions.py:25-43` keeps the token conditioner's tests behind its lead-in flag. Existing suites unchanged (normalisation, mirror, leakage, causality, free-run smoke).

### Stage 1: span-aligned, window-scoped draws at matched exposure (training)

- **Claim**: placing scope onsets in the scored window, with dropout before alignment and natural decisions weighted back to the current distribution, raises in-distribution LN following at matched exposure without worsening natural prediction or natural continuation.
- **Arms**: both on stage-0 code: uniform base CE, λ = 0, the same span lengths, presence `none`, rule L, the default η, exact targets, absolute tiled star as the star value (stage 2 is the only star change). **A0**: the current start rule and per-song interval selection (p_align = 0, u_j ≡ 1). **A1**: the §8.1 draw. The one difference is the draw. Optional **A2**: A1 with λ_LN = 1 (Q-E).
- **Training seeds**: 171/471 and 172/472 (weights/draws) per arm; the second pair runs after the first pair's 50M read-out exists. Random seeds 954-956 in every panel.
- **Budget and stop**: 50M exposures (≈ 3.9 average passes), cosine horizon 50M, lr 1e-3 unless Q-F re-tunes it, checkpoints every 4.39M, full evaluation every second checkpoint, **no plateau stop**; both arms run to 50M; nothing is read before 50M.
- **Primary**: onset-panel slope (§9.4), per training seed, read on the mean of the last three checkpoints, chart-paired between arms.
- **Threshold**: A1 − A0 ≥ +0.15 and > 2 paired SE **in each training seed** (panel SE on a paired slope difference ≈ 0.06 [inferred]). Natural-manifest NLL: A1 − A0 ≤ +0.02 (mean of the last three checkpoints, paired bootstrap). Guard (i) within ±0.05 in both arms at 50M. Holds ≤ 60 ms ≤ 0.5 % in both arms. The onset-decision NLL paired difference A1 − A0 under the true value must be negative and > 2 SE. §9.7 (a1) passes at every read-out checkpoint (engineering, not evidence).
- **Secondary (reported)**: whole-song slope and MAE, switch DiD/2, own-history gap, BOS natural LN share and drift, G3 (a2), (b), (c), (f), the D-properties realised in the run's draws, and each arm's default adherence report.
- **Outcome rules**: threshold met → A1's draw is the base; stage 2. Not met with the onset NLL difference resolved → the value is read in teacher forcing but not realised in free run → stage 3 moves ahead of stage 2 (Q-I). Neither → the sparse-signal hypothesis is dead for this interface; F2 and Q-J are the next candidates; no architectural conclusion. Guard (i) failed in both arms with the primary met → the draw works and the natural bias is a separate problem; stage 3's own-history term is the candidate (Q-I).
- **Cost**: 4 runs × 8.1 h ≈ 32.5 h of mac, sequential (A2: +16 h at two training seeds, +8 h at one).
- **Engineering tests**: a config-diff test that the arms differ only in `p_align` and the weight switch; an exposure-accounting test that both arms' 50M checkpoints are within 1 % in head decisions; the stage-0 suite, the locality tests included, green on each run's frozen copy; every record complete.
- **Human**: Q-B, Q-E, Q-F before launch.

### Stage 2: difficulty as a residual (training)

- **Claim**: conditioning on Difficulty_ν − b(S), with the committed proxies, gives a residual response where the absolute value gave none.
- **Arms**: B0 absolute tiled star (the stage-1 winner's recipe); B1 residual; B2 per Q-G (TF-mono, or the relaxed-proxy term). One training seed first; the second if B1 − B0 passes at one.
- **Metric**: residual slope, MAE and deviation distribution over absolute targets b(S) + {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s onset-aligned scopes, 8 charts × 3 random seeds, mean of the last three checkpoints; ∣Δ_tail∣ reported alongside (§7.5).
- **Threshold**: B1 − B0 slope ≥ +0.3 and > 2 paired SE; A's star gate (slope ≥ 0.5, MAE ≤ 0.5 star) reported; LN metrics unchanged within 2 SE; guards (i), (iv); (a1).
- **Budget, stop**: as stage 1; 2-4 runs × 8.5 h ≈ 17-34 h (B2: +1-2 runs).
- **Engineering tests**: b and proxy tests of stage 0; relabel consistency; config-diff (arms differ in the value encoding, and B2's term only); exposure accounting.
- **Human**: Q-G before launch; Q-H is a default; a trimmed ν only if the §7.5 rule fires.

### Stage R: baseline style ρ (reserved; after stage 2; no design commitment)

Entry: stage 2 read out and the human's go. The first deliverable is a design brief, not code; what follows is the specification it must meet, written now so the slot, records and diagnostics of this repair fit it.

- **What ρ must satisfy** (`style-conditions-and-control.md` "Generation inputs", "Chart seed, baseline and random seed", "Natural continuation and return to natural", "Identity and control requirements"): a baseline is always in effect; it can be supplied, extracted from style-only references, established from a chart seed's evidence, or sampled when nothing is given (the random seed then covers that sampling); a supplied baseline governs over a chart seed, which stays history (q7 written default); it is not chart state and persists across continuation calls and local overrides until explicitly changed; adopting an override's organisation is a separate, explicit baseline update at a committed boundary; returning to natural restores baseline-conditioned generation while active property targets still apply; on the same music, history and targets, different baselines give recognisably different charts and one baseline gives several coherent realisations under different random seeds; a property change keeps the identity where compatible; a style reference shapes organisation without entering the committed prefix.
- **How it would be tested** (thresholds set in that brief before its runs): G3(c) with ρ as the manipulated variable: between-identity variance across two ρ on the same prefixes exceeds within-identity variance across random seeds by > 2 SE on every organisation statistic; G3(b) binding: identity kept under an LN change of 0.3; **return to natural**: continuation after release compared with natural continuations under the same ρ from the same post-override history, not with pre-override or seed statistics (`style-conditions-and-control.md` evaluation table; V13); a round trip: extract ρ from chart A, generate on skeleton B, extract again, agreement above the extraction's test-retest agreement on real charts; persistence across two continuation calls with the same ρ equals persistence within one call; the explicit update changes the identity and the record shows it. Lesion test per ρ input. Recognisability needs human judgments or a validated recogniser (`style-conditions-and-control.md` lines 457-460).
- **Not decided**: whether ρ is a vector, tokens, a reference prefix, or a sampled latent; whether it is trained with CE alone or with an objective on free-run continuations; how references are encoded. The Fable judgment's ordering (a persistent vector only after CE alone holds one condition over a song) is the entry condition.

### Stage 3: own-sample terms (training; behind Q-I)

- **Claim**: the relaxed own-history LN term closes the own-history gap and lifts following without fake holds; the relaxed-proxy star term (if not already in stage 2) does the same for the residual.
- **Arms**: C0 the stage-2 winner; C1 with μ_LN = 1 (and μ_star = 1 if stage 2 passed), one window in four, continuation of 20M exposures from C0's selected checkpoint, both arms continued equally. Every own-sample term scores Ω factors only: decisions outside V, those after b included, contribute neither to the score-function sum nor to the relaxed gradient (V6).
- **Metric, threshold**: own-history gap ≤ 0.05; onset-panel slope ≥ 0.9 and MAE ≤ 0.10, > 2 SE over C0; holds ≤ 60 ms ≤ 0.5 %; natural NLL within 0.02; guard (i); (a1). True F3 (k = 4) on one update in sixteen as calibration of the surrogate gap, reported.
- **Budget**: 2 × 20M at ≈ 1.45× step cost ≈ 3.2 h each plus ≈ 1.0 h of evaluation each.
- **Engineering tests**: T-P2 on the own-sample path, including zero gradient into factors outside Ω; no gradient through sampling; a fixture test that 200 steps move the realised statistic toward the request by > 2 SE over three fixture random seeds, threshold fixed here and not loosened after a failure.

### Stage 4: named-attribute style path (engineering, no claim)

A fixture attribute with a deterministic section statistic (a three-level band of chord-size ≥ 2 rate) exercises the one-hot encoding, ownership by visibility, rule L, locality and lesion tests on a tiny model; the validator accepts style directives for the fixture attribute only and rejects two directives on the same attribute with overlapping scopes (q9 written default); a toy run shows a fixture response at 2 SE as a mechanism check; G3(e) on the fixture. No style claim; it is what lets a Lens label plug in as a named-attribute style directive without interface work. The attribute set is open by schema (G4).

### Deferred

A split boundary decision (pre-boundary closes decided first), only if D8 exceeds its threshold (§4.4). Token versus FiLM: the token form is a lead-in instrument (R9); the comparison waits for a lead-in policy. Higher strength levels; priority; transition intervals (§5.3, §5.5). Landmark readout ablation (Q-L, default after stage 2; the test: NLL with and without the readout on rows with k > 511, a 6-level TCN arm). DPO with real pairs (horizon ≥ the longest conditioned span). Capacity and lr/decay re-tune on fit_dev evidence after stage 1. Reference and edit style directives, learned dimensions (G4). Gameplay-demand requests (N7). Provisional branches and prefix commit (N3).

---

<a id="v4-ledger"></a>
## 14. Ledger

Status **C** checked in code or data (cited), **I** inferred. Action: **S0..S4, SR** the fixing stage, **D** deferred, **H** a human question, **MT** for the main thread, **none**, **settled**.

| # | Wrong piece | Evidence | St. | Action |
| --- | --- | --- | --- | --- |
| L1 | Condition values are always the source's own statistic; the counterfactual response is never supervised | `conditions.py:52, 62-75` | C | S1 informative draws; S3 own samples; Q-J (default: not now) |
| L2 | Inside a span the committed counters reveal the value; CE on middle rows needs no value read (the end is informative again) | `features.py:264-272, 296-302`; audit §2 | C | S0 onset / middle / end strata; S1 onset-aligned draws |
| L3 | Absolute tiled star is ¾ determined by the skeleton; no committed-difficulty quantity in the frame; value KL 1e-5 at 61.4M | A `star-prediction-summary.json`; recheck table; `features.py:296-302` | C | S0 proxies; S2 residual |
| L4 | "Natural behaviour" had no stated definition; the code implements "imitate the source on those rows" | `conditions.py:83-90`, `features.py:291` | C | **settled** (C4): free; source rows are the target in training; §6.7 |
| L5 | Rows between two spans had no stated semantics | `features.py:291, 308-330` | C | **settled** (C4, C5): natural; presence `none`; §6.7 |
| L6 | Spans drawn without knowing the window; LN active on 13.5 % of scored heads, 32 % of active rows with an unscored onset | `data.py:82`; audit #1 | C | S0 §8.1 (dropout first, alignment, IPW) |
| L7 | Interval count per song, not per window (LN-active 21 % for K < 600, 8 % for K ≥ 1500) | `conditions.py:55, 74`; audit #4 | C | S0 per-window selection; D2 |
| L8 | The fit_dev manifest inherits L6/L7: 14 of 64 windows with an active LN row | `data.py:94-119`; `fit_dev_manifest.json` | C | S0 two manifests (§8.3) |
| L9 | Window start rule; 25 % short windows; harmful only through 1/L | `data.py:74-81` | C | S0 keep the rule, fix the weighting |
| L10 | Star cells fixed per chart by a hash; an onset can never be placed relative to a window | `labels.py:56-75` | C | S0 relabel at 10 s offsets, three lengths and the whole song |
| L11 | Cells mostly longer than a window; the whole-cell value on every row | `labels.py:37`; audit #2 | C | S0 aligned draws score the onset side; lengths kept (§8.2); N9 |
| L12 | The whole-song star (p 0.10) fed per row though set by the hardest section | `conditions.py:67-69` | C | S2 residual on the same span; chart-level scalar rejected by the human |
| L13 | Labels from the original `.osu`, not the cache representation (≤ 0.00016 star measured) | `labels.py:116-118` | C | S0 relabel from the cache through ν |
| L14 | Tiled star needs ≥ 30 s: a constraint | `labels.py:35, 103-104` | C | kept; part of ν |
| L15 | Presence bit set on every row when any interval exists; T-P3b(i) fails through it | `features.py:291`; DEVIATIONS 5 | C | **settled** (C5): S0 switch, default `none`; necessary, not sufficient (rule L, L66) |
| L16 | Token form reads the whole track at every row, before and after each interval | `features.py:308-330`; DEVIATIONS 6 | C | S0 raises under the default η; kept for a lead-in policy (R9, L71) |
| L17 | Star frame carries LN counters and no difficulty statistic | `features.py:296-302` | C | S0 proxies (§6.8) |
| L18 | `Interval.value` scalar; no categorical kind | `features.py:256-261, 275-276` | C | S0 per-kind encoding with one value channel; S4 |
| L19 | FiLM zero-initialised: natural mode is learned FiLM(0); not a defect | `model.py:62-64` | C | none |
| L20 | Loss is the window mean: EOS 4.2×, last 64 rows 1.8× | `train_ce.py:266, 269`; audit #5 | C | S0 uniform base CE, fixed divisor |
| L21 | No condition-scoped term; nothing weighted, tested or reported per kind | `train_ce.py:263-273` | C | S0 per-factor terms (§6) |
| L22 | DPO anchor reuses the window mean | `train_dpo.py:190-191` | C | S0 shared L_base |
| L23 | No term sees own histories; own-history LN forecast shift 0.129 ± 0.044 at 61.4M | recheck `average-table.md`; `train_ce.py:263-269` | C | S1 measure; S3 term (Q-I) |
| L24 | "Conditioned" NLL keyed on track presence (47 % of such decisions active) | `train_ce.py:145` | C | S0 per-factor, per-stratum NLL |
| L25 | In-run free run natural only, from BOS | `train_ce.py:317` | C | S0 §9.4 |
| L26 | Selection on fit_dev action CE alone, no SE, no guards | `checkpoint-selection.json` | C | S0 §9.6 |
| L27 | Free-run panel 4 charts × 3 random seeds; noise floor ≈ 0.1 in LN share | `data.py:111-116`; `evals.jsonl` | C | S0 panels of §9.4 |
| L28 | Natural LN share on the 4-chart panel moves by up to 0.17 between adjacent checkpoints to the end, but safe pairs 52k-170k exposures apart move by 0.08-0.10, so late deltas are within panel noise | `evals.jsonl` [data, v3] | C | none; "CE recipe first" rests on L23 and the natural LN bias |
| L29 | fit_dev NLL minimum 1.975 at 61.4M, 2.061 at 121.7M; ≈ 9.6 average passes, more for charts in small groups | `evals.jsonl`; `summary.json`; `data.py:69-71` | C; cause I | S1 ≤ 4 passes, cosine at the budget; Q-F |
| L30 | No conditioned generation entry point with records | `train_ce.py:552-564`; `sampling.py:28-29` | C | S0 request CLI and request set (§11 item 5) |
| L31 | No baseline b for a generated span's star | — | C | S0 §7.1 |
| L32 | Boundaries inside a window only by chance; `replace_interval` unexercised | `conditions.py:106-134` | C | S0 draws keep adjacent pieces; switches in the condition manifest; `replace_interval` retired (H4); T-A, T-C |
| L33 | Free-run summary lacks following, calibration and the short-hold rate under requests | `report.py` | C | S0 §9.4 fields; records |
| L34 | Guard trips on system-wide swap growth since the trainer's start; six trips 1.19-4.15 GB; the trainer's RSS limit never tripped | `runtime.py:31, 95, 114, 119, 129-133`; `events.jsonl` | C; cause I | S0 §11 item 1; MT reads `resources.jsonl` |
| L35 | Every restart resets the swap baseline; lifetime restart budget 5 | `launch.py:27, 106-112` | C | S0 rate budget |
| L36 | `train.jsonl` and `resources.jsonl` exceed the 4 MB staging limit and are re-staged every cycle; not pattern-excluded | `mutagen.yml:161, 179-182` | C | S0 segmented logs; MT pattern exclusion for the existing run |
| L37 | RSS spikes to 2.5-4.6 GiB at irregular points; inferred cause per-candidate float64 arrays | mirrored `resources.jsonl`; `model.py:265-278` | C; cause I | S0 pilot; float32 and chunking if confirmed |
| L38 | Throughput 2,177/s wall with contention; 3,700-3,957 in the pilot | `run.json`; DEVIATIONS 9 | C | none (planning number 2,500) |
| L39 | Landmark readout: lesion none at 61.4M; redundancy with the 511-token field the inferred cause | recheck table; audit §3 | C; cause I | D; Q-L (default after stage 2) |
| L40 | DPO: synthetic labellers off the path; 64-decision horizon shorter than a 30 s span | `train_dpo.py` | C | D (waits for real pairs) |
| L41 | Release factors read frames at the LN's birth time and at candidate times across scope boundaries | `model.py:274-276` | C | S0 rule L; birth role removed (H5); T-V |
| L42 | No LN span above 64 beats; whole-song and half-song requests are extrapolation | `conditions.py:20, 26-31, 34-56` | C | S0 span lengths; Q-B; primary changed |
| L43 | v1's draw applied dropout after alignment; the design rule requires inverse-probability weights for any balancing | plan v1 §4; `design.md:303` | C | S0 §8.1 steps 3-6 |
| L44 | v1 refitted b on 30 s cells with `duration` constant | A's feature list | C | S0 §7.1 |
| L45 | The star statistic counts tails decided after b, which no longer read the request | `labels.py:39, 78-98` | C | the term does not own them (H5); S0 measures their share (§7.5); realised response sampled until the last S-headed LN closes (§7.3) |
| L46 | Manifest and free-run panel depart from `design.md:535` without a DEVIATIONS entry; free-run charts K ≤ 600 | `design.md:535`; `data.py:95, 111` | C | S0 DEVIATIONS entry with the power cost; panels of §9.4 |
| L47 | Safe checkpoints at irregular exposures enter `evals.jsonl` | `train_ce.py:368-369, 411`; last `evals.jsonl` entry `ckpt-0121700759-safe.pt` | C | S0 excluded from selection; tagged in logs |
| L48 | Checkpoint-level fit_dev noise ≈ 0.02-0.03 not in the bootstrap SE | `evals.jsonl` 109.7-111.1M [data, v3] | C | S0 mean of the last three checkpoints |
| L49 | `validate_track` rejects star < 0; star normalised by /4 | `conditions.py:100`; `features.py:276` | C | S0 |
| L50 | Manifest reuse keyed on `star_conditions` only | `data.py:121-129` | C | S0 version hash (draw parameters, labels, ν, rule L) |
| L51 | A plateau stop can break a matched-exposure comparison | plan v1 §8.4 | C | off in comparisons |
| L52 | Stage costs omitted in-run evaluation | recheck `execution.json` | C | §9.8 |
| L53 | ∣Δ∣ > 0.1 is within binomial noise for 8-beat pieces (≈ 40 heads, SE ≈ 0.06) | audit rows per beat | I | S0 z-criterion with n ≥ 20 |
| L54 | At 61.4M the whole-song gate is met, switches are not, natural LN is biased +0.10 to +0.13, star has no effect | recheck tables | C | folded into L3, L23, L28, L42 and the stage-1 primary |
| L55 | Four separate definitions of LN share; tiled star is a version string; no ν object | `conditions.py:42-52`, `labels.py:42-45`, `report.py:46-48`, `probe.py:177-184`, `labels.py:39` | C | S0 `properties` module (C2) |
| L56 | No request object: callers build `Interval` tuples; no validation of overlap or validity; no record of intended against realised | `sampling.py:28-29`, `conditions.py:94-103`, `receipts.py` | C | S0 schema, validator, request set, record (H3, H4) |
| L57 | v2 asked for difficulty requests in residual units | v2 §4.1 | C | S0 absolute targets (C3) |
| L58 | The token conditioner shows every interval of the track, started or not | `features.py:308-330`; DEVIATIONS 6 | C | merged into L16 and L71 |
| L59 | Point values only; no range | `features.py:275-276` | C | **settled**: targets only (H1) |
| L60 | "Seeds" conflated in v2 | v2 §6.4, §9 | C | terminology; record fields (N2) |
| L61 | No baseline style slot; identity only in the committed history | `sampling.py:28`; judgment | C | S0 slot; SR (G1) |
| L62 | Generation records lack the chart seed (g0, prefix hash), the request list and readouts | `train_ce.py:314-325, 331-339` | C | S0 §5.7 |
| L63 | No test that an explicit zero request differs from no request in the likelihood | `tests/r2/test_inputs_matter.py` | C | S0 T-Z (N4); readout §9.7 (f) |
| L64 | No transition policy; activation and expiry only through `replace_interval`'s frontier | `conditions.py:106-134` | C | S0 default η only; transition intervals and priority reserved and rejected (§5.3) |
| L65 | `docs/formulation/` on `r2/train` predates the formulation | `r2/train` `notation.md`, `gameplay-state.md` | C | The formulation is on `docs/style-formulation` at `bdbbbde`; `r2/train`'s copy is stale until that branch is merged (MT) |
| L66 | Decision k decides the gap before its row: the onset decision's joint action moves the law of closes before a when a hold is open; the exit decision's candidate frames read S | `common.py:12-15`; `features.py:77`; `state.py:76-100`; `model.py:216-219, 275, 309-319` | C | S0 rule L (§4); D8 |
| L67 | The birth-role frame reads an expired scope | `model.py:276` | C | S0 removed (H5) |
| L68 | The activity test is half-open at T: EOS does not read a whole-song scope through the row role | `features.py:49, 293` | C | S0 closed at T when b = T; T-E |
| L69 | Tiled star of a scope that owns no object is 0.0 | `labels.py:108-109` | C | S0 undefined in `properties`; T-U |
| L70 | `replace_interval` clips a start before the frontier and merges equal neighbours; its test asserts the clip | `conditions.py:106-131`; `test_conditions.py:60-63` | C | S0 retired; the validator rejects (H4); no merge (V4) |
| L71 | The token conditioner shows expired intervals (`started` = 1, the value) as well as future ones | `features.py:316-329` | C | S0 raises under the default η (R9) |
| L72 | `continue_chart` takes one fixed track; a request's addition boundary is not kept across calls | `sampling.py:28-29` | C | S0 request set with g_u and g' (V9) |
| L73 | R2 chart seeds end at a head row (g0 = t_{s−1}) | `sampling.py:35-42, 56`; `state.py:76-77` | C | none: support limitation, recorded (V10) |
| L74 | v3's G3(a) compared with the run without the override, not with the same post-scope history | v3 §7.7; `style-conditions-and-control.md` line 450 | C | §9.7 (a1), (a2) (V3) |
| L75 | ckpt-0061432779 was trained with the presence bit and the birth role | the run's frozen `features.py:291`, `model.py:276`; config `conditioner: film` | C | S0 baseline rows with requests by the v1 code (V11) |
| L76 | An own-sample term over a whole generated section would train decisions that cannot read the request | §7.3 | I | S3 restricted to Ω (V6) |
| L77 | PR #16's prose describes priority for overlapping requests and a "release without compensation" diagnostic | PR #16 description | C | MT (V15) |

---

<a id="v4-questions"></a>
## 15. Decisions for the human

Only questions whose answer changes what is built or run, in plain words, each with a chart example, the options and what each changes, and the stage it blocks. Settled since v3: ranges (targets only), what default strength means, same-property overlap, adding a request after its start, hold ends after a scope.

### Questions that block a stage (ordered by what they block)

**Q-B. How long may a training LN scope be?** (blocks stage 1: the draw is frozen before the runs)

- *What is decided.* Whether the model trains on LN-share scopes longer than 64 beats (16 bars at 4/4), so that long requests are something it has seen.
- *Example.* Today every training request looks like "LN share 0.4 for bars 9-16". A request "LN share 0.4 for the whole song" or "for bars 1-64" has never appeared in training; the model answers it by stretching what it learned on short scopes. Last session's whole-song test (slope 0.715) was such a stretch.
- *Options.* (a) Default: pieces of 8-64 beats as now, plus runs of 2-8 pieces one draw in four, plus the whole song one draw in ten, in both arms. Code: the candidate list of the draw. Runs: nothing extra; D7 measures the coverage. Effect: whole-song and half-song requests become in-distribution; onsets per window fall a little. (b) Pieces only, as now: the whole-song panel stays an extrapolation measure and the half-song switch test stays out of distribution. (c) Other probabilities: say which.

**Q-E. Does stage 1 also test the LN emphasis term?** (blocks the stage-1 budget)

- *What is decided.* Whether stage 1 runs only the two draw arms, or also a third arm with the LN emphasis term switched on.
- *Example.* A0 and A1 both train plain cross-entropy ("predict the source's decisions"); they differ only in which 256-row windows are scored (A1 chooses windows so that scope starts fall inside them). A2 would be A1 plus a second score, on every head that can see an LN request, for the choice "tap or LN" (λ_LN = 1): the model is pushed harder to get the LN share right exactly where a request is active, and nowhere else. Because training changes, A2's selected checkpoint is a different default operating point, with its own adherence report.
- *Options.* (a) A0/A1 at two training seeds: ≈ 32.5 h of mac; stage 1 says nothing about the term itself. (b) Add A2 at two training seeds: +16 h; stage 1 also says whether the term adds following beyond the draw. (c) A2 at one training seed: +8 h; a one-seed reading, labelled as such.

**Q-F. Keep last run's learning rate for stage 1, or re-tune first?** (blocks the stage-1 launch)

- *What is decided.* Whether to run a short learning-rate check before stage 1.
- *Example.* The overnight run used lr 1e-3 with a cosine schedule over 158M exposures and began to overfit after 61M (fit_dev NLL 1.975 → 2.061). Stage 1 runs 50M exposures with the cosine ending at 50M, so the model sees each chart about four times instead of ten.
- *Options.* (a) Default: keep lr 1e-3, cosine to 3e-5 over 50M, weight decay 0.01. No cost. (b) Two 30-minute pilots on the stage-0 code (lr 6e-4 against 1e-3, weight decay 0.01 against 0.05) judged on fit_dev NLL at equal exposure, then the better setting: ≈ 1 h of mac before stage 1. The tuning hour of 2026-10-03 showed run-to-run variation as large as the lr effect, so (b) is weak evidence either way.

**Q-G. For difficulty, what does "train the response towards the loss" mean?** (blocks stage 2's arms)

- *What is decided.* Whether difficulty is learned only from real charts, or whether the model's own generated sections are also scored for difficulty during training and pushed toward the request.
- *Example.* Request "4.2 stars for 0:30-1:00" on a skeleton whose head rows alone predict 3.9 stars, so the request asks for +0.3 star above the skeleton. Under (A) the model has learned from real charts how mappers make a section harder than its skeleton suggests (more LNs, bigger chords, same-lane repeats) and applies that; afterwards we measure what difficulty the generated section has. Under (B), in addition, sections are generated during training, scored with the difficulty evaluator, and a loss pushes the generated difficulty toward the request. In both, long notes that start inside 0:30-1:00 and end after 1:00 are ended without the request (the decision of 2026-10-06), yet the difficulty score of the section still counts their full length; under (B) the training signal reaches only the decisions inside the section.
- *Options.* (A) Default: cross-entropy on real rows with the residual as the value; stage 2 = B0 (absolute star) / B1 (residual) / optional B2 (TF-mono, +20 % step cost, no sampling). (B) In addition a term on generated sections: B2 = the relaxed-proxy term, +45 % step cost, with true F3 as a calibration measurement one update in sixteen. (B) is a self-supervised term on the model's own samples (no preference labels, not DPO), so the permission question Q-I below applies to it.

### A formulation question that changes what is built but blocks nothing

**q10. May a running request be cancelled before its scope ends?** (blocks nothing: stage-1 and stage-2 runs never cancel; stage 0 builds option (a) unless told otherwise)

- *What is decided.* Whether a request that has already started can be stopped, or switched to another value, before its scope ends.
- *Example.* "LN share 0.5 for 1:00-2:00" is running; at 1:30 you want to stop it, or to switch to 0.3 for the rest.
- *Options.* (a) Default, as the formulation is written: cancel at 1:30; the target stops being a goal, and the 1:00-1:30 readout is recorded as "cancelled", not as adherence. To switch, cancel and add "LN 0.3 from just after 1:30"; it must start after the cancellation point, and the moment between is governed by neither. Code: a cancellation boundary per request, its record entry and test T-C (≈ 0.05 d). (b) No: a running request always runs to its end; to change course mid-section the caller splits the section into several requests in advance. Code: cancellation inside a scope raises. (c) Cancel yes, switch no: as (a), but no new request for the same property may start inside the cancelled scope. Code: one more validator rule.

### Defaults the plan takes (no answer needed; say so to change one)

- **Q-C star cell lengths** (stage 0): 30, 60, 120 s at 10 s offsets plus the whole song; ≈ 36 min relabel. Alternatives: add 45 and 90 s (≈ 55 min); 30 s and the whole song only.
- **Q-D guard** (stage 0): remove the system swap trip; trip on the trainer's own RSS growth > 2 GiB between checkpoints or RSS > 12 GiB; restart budget 5 per 6 h.
- **Q-H whole-song star interval and request range** (stage 2): keep the whole-song interval at p 0.10 as a residual; requests within ±0.5 star of the skeleton baseline (one SD), wider ones flagged.
- **Q-I own-sample terms before real preference pairs** (stage 3; stage 2 under Q-G(B)): not run without a yes. Options when asked: yes / no / only if stage 1 fails the free-run gate while passing the teacher-forced one.
- **Q-M reading of direction 1 for the emphasis terms** (stage 3): governed factor; both numbers logged from stage 0.
- **Q-J constructed alternatives** (accepted re-arrangements of one skeleton, which need a judge): not now.
- **Q-K style vocabulary** (stage 4): the five Lens names with three levels as named attributes; the set is open by schema.
- **Q-L landmark run**: after stage 2.
- **Natural guard threshold** (stage 1): ±0.05 on the prefix panel's chart-paired mean difference.
- **Token conditioner**: kept as the instrument for a future lead-in policy; raises under the default η; not an arm.
- **Rule L's cost** (stage 0): a scope's onset row is not governed when a hold is open across it; a split boundary decision is designed only if more than 5 % of in-scope head decisions on 16-beat LN scopes are outside V (D8).
- **Difficulty ν** (stage 2): untrimmed tails as labelled today; a trimmed ν is put to you before stage 2 only if the median ∣Δ_tail∣ on 30 s cells exceeds 0.10 star (§7.5).
- **Unattainable requests**: a headless LN scope is accepted and flagged; a difficulty scope under 30 s is rejected (ν is undefined there).
- **Formulation questions at their written defaults** ([style-formulation-rethink](style-formulation-rethink.md#open-questions)): q2, no strength level below the default (nothing is built either way); q3, strength for property targets only (nothing is built either way); q7, a supplied baseline governs over a chart seed (ρ is not built; stage R's brief starts from it); q8, different properties may overlap without priority and the shortfall is reported (built as stated; under option (b) the validator would reject every overlapping LN and difficulty pair until priority exists, and dropping priority altogether changes nothing built, since R2 rejects priority now); q9, the same attribute and the same property under another ν are also rejected (R2 accepts one ν per property and rejects any other at the schema step, so options (a) and (c) build the same thing now; (b) would matter only for stage 4's style validator).

Withdrawn since v3: Q-R (settled by [d-target-strength](r2-style-formulation-check.md#d-target-strength)); the priority-rule default (replaced by H3). Withdrawn earlier: Q1, Q2, Q-A (settled by answer 1); v1's Q3; Q10/Q11 (Q-D, Q-H); Q6 (Q-A); Q4 (Q-I); Q5 (Q-J); Q7 (Q-F); Q8 (Q-K); Q9 (Q-L).

---

## 16. What could not be checked

- Mac-only files: the checkpoints, the cache `index.parquet`, A's `star-predictions.parquet` (the within-chart residual correlation), and the parts of `train.jsonl` and `resources.jsonl` beyond the mirror. No `ens` command was run.
- Numbers tagged [data, v3], carried from v3 without re-opening their files: the teacher-forced Δ 0.060 ± 0.010 and the 8-chart real share 0.187; checkpoint-level NLL noise and the adjacent-checkpoint LN-share deltas; 4,167 fit_train groups; the recheck `execution.json` timings; the free-run 338 rows/s; the Fable judgment's variances. Nothing in v4 turns on them beyond cost estimates.
- Every [inferred] number in §4.4, §7.3, §7.5, §8.3, §9.7, §9.8 and §13, until `draw_sim.py`, the tail-share script and the stage-0 pilot run. In particular D8 (the share of in-scope head decisions rule L masks) and the size of ∣Δ_tail∣.
- The exactness argument for rule L (§4.4) is reasoning about the code at `7d9640a`; T-V, T-L1 and T-L2 test it once written. No test or code was run for this plan.
- Whether `compute_mania_star_rating_20241007` is invariant under the lane mirror (T-M measures it).
- The run's frozen code was checked only for the presence bit and the birth role (L75).
- The formulation at `bdbbbde` has not been reviewed by anyone but its author ([style-formulation-rethink](style-formulation-rethink.md#not-checked)); its paraphrases here (§2.3, §12, stage R) are this plan's and give way to the documents.
- Draft PR #16's description was read with `gh pr view 16` (read-only); its outdated points are listed for the main thread (V15), not edited.
- Acceptability of any generated chart; no renders were viewed.
- Whether the five Lens attributes are the named attributes R2 should take (Q-K, a default).
