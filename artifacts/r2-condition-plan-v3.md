# R2 condition-scoped losses and recipe repair: plan v3 (proposed, awaiting the human)

Shareable. Written 2026-10-05 by a fresh Fable subagent after [r2-condition-plan-v2](r2-condition-plan-v2.md) and the main thread's [formulation check](r2-style-formulation-check.md). Supersedes v2 as the working plan; a reader needs neither v1, v2 nor the review. Serves [d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss), [d-dpo-synthetic-off](r2-average-and-control.md#d-dpo-synthetic-off) and the defaults the human adopted on 2026-10-05 ([d-formulation-answers](r2-style-formulation-check.md#d-formulation-answers)). The human's style and controllable-generation formulation takes precedence over this plan and states the long-term problem ([private, local](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-1)); its answers to the check's questions are [answer-1](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-1). The formulation is paraphrased here, never quoted. Nothing is adopted until the human decides the questions in section 12.

Code read at `r2/train` `7d9640a` (paths relative to `src/ensomi_model/r2/` unless stated). Nothing in the repository was edited; no job was launched.

Evidence tags: **[code]** checked at the cited line by me; **[data]** read by me from the named file; **[data, v2]** a number v2 read from a file I did not re-open (listed in section 14); **[inferred]** reasoning; **[to measure]** a number a stage-0 job must produce.

**Standing rules (bind every stage):**

1. Before any run, state the claim, metric, threshold, songs, random seeds, training seeds and stop rule. Any other number is post-hoc.
2. Every output carries an execution receipt.
3. A component is "in the system" only if the receipt lists it.
4. Every claim names the test or evaluator that fails if it is false. Mechanism tests are engineering, not evidence. Every model forward has an input-lesion test per named input.
5. No conclusion from a run below its stop rule.
6. A difference under two standard errors is "none". Tables carry baseline rows and the spread across random seeds and songs.
7. Evaluators are frozen and hashed, and never scored on what they were tuned on.
8. Keep a ledger of recognised questions (section 11).

**Terms.** A **random seed** selects sampling randomness (the `seed` of `sampling.py:29` [code]; 954-956 in the panels). A **training seed** selects weights and draws (`seed_weights`, `seed_draws`, `train_ce.py:75-76` [code]; 171/471). A **chart seed** is a committed prefix with its fixed-through time (`prefix_actions`, `prefix_gap` of `sampling.py:28` [code]). The three are never written as "seed" alone. A **property** is a scoped measurement of the result (LN share, section difficulty). A **style directive** asks for organisation (named attributes now; references and edits later). A **request** carries a scope, optional directives, a transition policy η and a priority. The **effective track** is what the model reads after requests are resolved. A **readout** measures the actual chart with the same definitions ν as the targets. A **receipt** lists intended against realised. The **baseline style ρ** is reserved (section 3.5).

---

## 0. The plan in one paragraph

Requests become objects: scope in song time, a property directive (target or range) or a named-attribute style directive, a transition policy η (only "immediate" is built) and a priority; a deterministic resolver turns them into the non-overlapping effective track the model reads, rejects unprioritised overlap and records feasibility flags. One `properties` module declares the measurement semantics ν shared by labels, frame counters, readouts and request validation. Difficulty is requested in absolute star; the residual against a frozen ridge baseline of the skeleton stays the internal parametrisation. Outside a scope a row sees nothing (presence `none`); a released or unspecified property is free, with no hold, no compensation and no population target. The loss is a uniform per-decision cross-entropy (fixed divisor, no window mean) plus per-condition emphasis terms that score only the factor a condition governs in the spans it owns, with weight zero in stage 1, so that stage tests one thing: a draw that places scope onsets inside the scored window (dropout before alignment, natural rows weighted back to the current distribution, spans of every length the interface promises). Stage 1's primary metric is in-distribution span following at onsets (whole-song slope is already 0.715 ± 0.045 at 61.4M [data]), two arms differing only in the draw, two training seeds, 50M exposures, thresholds relative to the control arm. Receipts list every request with its ν, target or range, realised readout and the effective track; the natural guard binds at panel-distribution level; new free-run diagnostics measure release without compensation, identity under a property change and within-against-between-identity variation. The baseline style ρ is an optional slot now and its own stage after stage 2.

---

## 1. Facts the plan rests on

### 1.1 Code [code]

| Fact | Where |
| --- | --- |
| A track is a tuple of `Interval(kind, a, b, value)`, half-open [a, b) in song milliseconds; kinds 0 LN share and 1 tiled star; intervals of one kind never overlap; an empty track is natural mode | `conditions.py:3-5`, `features.py:256-261` |
| LN spans are single partition pieces of 8/16/32/64 beats; 1-4 per song whatever K, consecutive with p 0.5; the value is the source's own share over the piece, computed inline | `conditions.py:20-23, 26-31, 34-56` |
| Star spans: cached cells (30 s, 60 s, start by a hash of the heads-only inputs, up to four consecutive per length) or the whole song with p 0.10; 1-3 per song; the same cells every draw | `labels.py:37, 56-75`, `conditions.py:59-75` |
| Dropout 0.20 all, 0.25 per kind, 0.20 per interval, on the per-song track | `conditions.py:83-90` |
| `validate_track` rejects nonpositive length, LN share outside [0, 1], star < 0 and overlap within a kind | `conditions.py:94-103` |
| `replace_interval(track, kind, a, b, value, frontier)` edits from the next decision on; the part of an interval before the frontier keeps its old value; equal adjacent values merge | `conditions.py:106-134` |
| The window start never reaches the track draw | `data.py:68-83` (`draw_track` at `:82` sees no `start`) |
| Start rule: BOS p 0.125, last 256 rows p 0.125, else uniform in [0, K]; stop = min(start+256, K+1) | `data.py:74-81` |
| Loss = window mean of decision NLL, averaged over the batch; the DPO anchor reuses it | `train_ce.py:266, 269`; `train_dpo.py:190` (`window_ce`) |
| "Conditioned" statistics keyed on a non-empty track | `train_ce.py:145` |
| In-run free-run is natural only from BOS; `continue_chart` takes a prefix, a track, a random seed and a stop | `train_ce.py:317`; `sampling.py:28-29` |
| Presence bit set for a kind on every row when any interval of that kind exists in the track | `features.py:291`; `DEVIATIONS.md` item 5 |
| Row frame per kind (16 channels): value, offsets to the bounds (8), progress, log1p heads, log1p LNs, ratio, log1p remaining, active, presence; star value normalised by /4; LN value 2v − 1 | `features.py:275-276, 296-304` |
| Frame counters count head objects of decisions before k whose head time lies in [a, b) (the same ownership as the labels) | `features.py:264-272` |
| The token conditioner reads the whole track at every query time, with offsets to intervals not yet active and a `started` flag | `features.py:308-330`; `DEVIATIONS.md` item 6 |
| Release factors read frames in three roles: at the row time, at every candidate time, at the held LN's start (birth); FiLM applied to the pointer query | `model.py:265-281, 283-291` |
| History tokens carry no condition features; FiLM enters only at the queried row and the pointer | `features.py:182-199`; `model.py:171-175` |
| FiLM output layers zero-initialised: natural mode is learned FiLM(0) | `model.py:62-64` |
| Code table: per lane, free-before 0 nothing / 1 tap / 2 LN head; held-before 0 keep / 1 release on row / 2 gap release / 3 gap release + tap / 4 gap release + LN head; 625-way joint action under a support mask | `common.py:7-18`; `state.py:49-57` |
| Decision log-probability = action log-probability + the two-orientation release mixture | `model.py:4-9, 236-251` |
| Tiled star: objects whose heads lie in [a, b), tiled to 240 s with a 1 ms seam rule, scored by `compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)`; ≥ 30 s; described by a version string; untrimmed tails | `labels.py:35-39, 78-110` |
| Four separate definitions of LN share: the draw (`conditions.py:42-52`), `labels.ln_share` (`labels.py:42-45`), the free-run summary by object start time (`report.py:46-48`), and A's probe (`artifacts/r2-analysis-20261004/control/scripts/probe.py:177-184`) | as cited |
| Labels read the original `.osu`, not the cache representation | `labels.py:116-118` |
| Manifest reused when only `star_conditions` matches; free-run charts K ≤ 600 | `data.py:121-129, 95, 111` |
| Safe checkpoints at resource trips enter `evals.jsonl` at irregular exposures | `train_ce.py:411, 433` |
| Resource guard trips on system-wide swap growth > 1 GiB since the trainer's own start; restart budget 5 for the run's life | `research/oracle_time_continuation/runtime.py:31, 95, 114, 119, 129-133`; `launch.py:27, 106-112` |
| Training receipt: entry point, config, code identity, cache hashes, training seeds, device; free-run entries carry the random seed only; no request, chart-seed or readout fields | `train_ce.py:331-339`, `receipts.py:14-39`, `train_ce.py:314-325` |
| Mirror: `maxStagingFileSize: "4MB"`; pattern exclusions cover `/reports/**/*.jsonl` only; the comment says the limit does not stop a transfer and retries every cycle | `~/ensomi/mutagen.yml:161, 179-182` |
| Design rules: no LN-share balancing of the draw without inverse-probability weights; the evaluation manifest is 128 groups, 24 per star band | `artifacts/r2-ml-design-20261003/design.md:303, 535` |
| `docs/formulation/notation.md` (generation formula with optional `c^style`, `c^demand`) and `gameplay-state.md` ("Controls") predate the formulation: no ρ, no request set; an absent style request "permits the learned natural style distribution" | `docs/formulation/notation.md:189-217`, `gameplay-state.md:332-366` |
| Input-lesion test pattern: tiny model, one forward, each named input changes the likelihood | `tests/r2/test_inputs_matter.py:31-60`, `tests/r2/helpers.py:87-94` |

### 1.2 Data

- Recheck, ckpt-0061432779 (61.4M exposures, one training seed), `artifacts/r2-recheck-20261005/control-table.md` [data]: whole-song LN slope 0.715 ± 0.045, MAE 0.115 ± 0.011 (A's gate slope ≥ 0.7, MAE ≤ 0.15: met); half-song switch DiD/2 0.272 ± 0.034 (gate 0.30: not met); natural LN share 0.312 ± 0.038 against 0.187 real (8 charts); teacher-forced Δ expected LN fraction for request 0→0.9 0.060 ± 0.010; star slope 0.014 ± 0.015, star value KL 1.0e-5 nats/row; 0 legality violations in 272 rows.
- `average-table.md` [data]: generated-minus-real-history LN forecast shift 0.129 ± 0.044 (0.217 at 30.7M, 0.012 at 39.5M); natural LN share 0.315 against 0.190 (24 charts) and 0.298 (48 charts × 3 random seeds); landmark-by-prefix interaction −0.41 ± 0.32 (none).
- `evals.jsonl` [data, v2]: fit_dev action NLL minimum 1.9753 at 61.4M, 2.0612 at 121.7M; natural-row NLL 1.979 at 61.4M; between checkpoints a few hundred updates apart it moves 0.02-0.03. Adjacent-checkpoint natural LN share on the 4-chart panel: safe pairs 52k and 170k exposures apart differ by +0.081 and +0.097; after 35.1M, 1 of 24 adjacent deltas exceeds 2 SE under an unpaired seed SE, 0 of 24 under a chart-paired SE.
- `artifacts/r2-analysis-20261004/control/star-prediction-summary.json` [data]: ridge on 23 head-time features (`duration`, `rows`, `density`, gap quantiles, chord-count statistics), fit_dev R² 0.746 ± 0.013, RMSE 0.528 ± 0.014 [data, v2 for the metrics]; real dev star mean 3.59, SD 1.05.
- `artifacts/r2-cache/v1/labels/star_summary.json` [data, v2]: 82,773 labels in 472 s on 4 workers (≈ 23 ms per label per worker). `summary.json`: 12,531 charts, 13.97M head rows, fit_train 11,368 charts in 4,167 groups.
- `run.json` [data, v2]: 2,177 decisions/s wall. Recheck `execution.json` [data, v2]: 264 continuations in 9 min 50 s on 2 threads.
- Range audit (`r2-range-audit.md`, 3,000 draws) [data]: LN active on 13.5 % of scored heads, star 23.9 %; informative scored onset rows 0.18 %; median LN piece 77 rows; star interval median 278 rows; a window holds a median 38 s; 3.8 rows per beat; 15 % of active rows within 16 rows of their onset.
- Fable judgment [data, from the note]: at 39.49M the variance of generated charts across random seeds on one skeleton (7.14) was twice the variance across skeletons (3.51); seed persistence fades by 256-512 rows ([a-r2-fable-judgment](r2-average-and-control.md#a-r2-fable-judgment)).

---

## 2. Rulings

<a id="v3-rulings"></a>
### 2.1 On the formulation check and the human's answers

Ruling: **adopt** (plan changed as the check proposed), **amend** (adopted with a change, argument given), **reject** (with evidence), **closed** (settled by the human).

| # | Point | Ruling | Argument and where it lands |
| --- | --- | --- | --- |
| C1 | LN share, difficulty and Lens dimensions are three kinds of one track; property and style directives differ in kind | adopt | Request schema splits them (§3.2); Lens dimensions are named-attribute style directives (§4.3); stage 4 is the named-attribute style path (§10). The frame per kind is unchanged in form (§4.8). |
| C2 | Several definitions of each quantity; no ν object | adopt | Four LN-share definitions exist [code, §1.1]; tiled star is a version string (`labels.py:39`). One `properties` module with ν (§3.1), used by labels, the draw value, frame counters, readouts and request validation; a test that labels equal readouts on source charts. |
| C3 | Difficulty requested as a residual | adopt | Requests carry absolute star in ν; v_res = target − b(S) is internal; readouts give absolute star and the residual; feasibility flags (§3.2, §5.1-5.2). The residual stays an R2 parametrisation; the interface does not depend on it. |
| C4 | Natural and between-span behaviour deferred with no default | closed (answer 1) | A released or unspecified property is free: no hold, no compensation, no population target, the chart seed's statistics are not a target. Training: base CE on unconditioned source rows is consistent (the source is one free continuation). v2's Q1 and Q2 leave the list; §4.7 replaces v2 §3.7; guard (i) binds at panel-distribution level (§7.6). |
| C5 | Presence bit on every row once any interval exists; v2 kept it | closed (answer 1) | Default `none` (§3.4); T-P3b(i) binds with no exception (§4.4); `anywhere` is a labelled arm only; `announce` is a reserved η, not built. |
| C6 | `validate_track` rejects overlap; requests may overlap by priority | amend | Adopted: a deterministic resolver by explicit integer priority, unprioritised overlap rejected, receipt logs the resolution; training unchanged. Amended: overlap is defined within a kind only (an LN scope and a difficulty scope on the same rows are not an overlap: both apply, `features.py:287-304` already reads both [code]); equal adjacent values merge as `replace_interval` does; the resolver also attaches the feasibility flags the formulation's "declared trade-off" needs (§3.2). |
| C7 | Scopes in musical time versus seconds-based star cells | closed (answer 3) | Song time independent of chunking: intervals are already stored in song milliseconds (`conditions.py:3` [code]) and the draw places windows around spans (§6.1). Seconds-based star cells stay; Q-C stays a stage-0 default. |
| C8 | Point values only | adopt | The frame reserves (lo, hi) per property kind (§3.7, §4.8); stage 1 trains lo = hi as the provisional default; whether ranges are trained is Q-R (§12). |
| G1 | No baseline style ρ | closed (answer 2) | An optional slot that accepts only "none" now (§3.5), the G3 diagnostics (§7.7), and a reserved stage after stage 2 with what ρ must satisfy and how it would be tested, no design commitment (§10, stage R). |
| G2 | No η policy | amend | Adopted: `immediate` is the only built policy (activation at the scope start, release at its end, consequences persist, no compensation); `announce` and `ramp` are reserved names, each a named input with a lesion test and a draw when built. Amended: `hold` is not a policy; holding a value longer is a longer scope, i.e. a request (§3.3). |
| G3 | Diagnostics for what the formulation targets | adopt | §7.7 (a)-(e) with metrics, panels, random seeds and thresholds fixed here; receipts in §3.6. (b) is reported in stages 1-2 and binding in stage R; (a) is binding from stage 1. |
| G4 | Style directives beyond named attributes | adopt | The schema's style field is a tagged union with `named` as the only built tag; `reference` and `edit` are reserved tags (§3.2). Nothing built rules them out. |
| A1 | Answer 1: defaults adopted ([answer-1](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#answer-1)) | adopt | As C4, C5. Q1, Q2 and Q-A are dropped from §12. |
| A2 | Answer 2: ρ reserved (same source) | adopt | As G1. |
| A3 | Answer 3: song time is enough (same source) | adopt | As C7. |
| A4 | Answer 4: ranges not understood (same source) | open | Provisional default kept; the question is rewritten in plain language with a chart example (Q-R, §12). |

Rows I add after reading the formulation myself:

| # | Point | Ruling | Argument and where it lands |
| --- | --- | --- | --- |
| N1 | The token conditioner is an announce form | amend v2's deferral | `features.py:308-330` [code]: every query sees every interval of the track, with offsets to intervals that have not started and a `started` flag; `DEVIATIONS.md` item 6 says so. Under the adopted default (no announce signal outside a scope) it is not admissible as the default conditioner. It stays built as the instrument for a future `announce` η; v2's "token versus FiLM after Q2" becomes "after an announce η is wanted". A masked token form (active intervals only) would be the only default-compatible variant and is not built now (§3.3, §10 deferred). |
| N2 | "Seeds" conflated | adopt | v2 wrote "seeds" for random seeds (954-956), training seeds (171/471) and the chart seed. The formulation separates them. Renamed throughout; receipts carry the three as separate fields (§3.6). |
| N3 | Readouts on provisional sections | adopt the signature, scope the rest | A property query must read a provisional section as well as a committed one. `Properties_ν` takes any object list (§3.1). R2's `continue_chart` commits decision by decision with no branch structure (`sampling.py:56-87` [code]); provisional branches and prefix commit are the contract of `notation.md` and out of this repair. Listed, not built. |
| N4 | Explicit zero versus unspecified | adopt (test) | An LN request of 0 is active = 1, value −1; no request is an all-zero frame; a difficulty request equal to the baseline is active = 1, residual 0. Under presence `none` these must differ in the model's likelihood: a stage-0 lesion test asserts it for each property kind (§4.4). |
| N5 | Expiry preserves ongoing holds; ownership by birth | keep, with a receipt field | A release factor of an LN born in span S and decided after S's end is owned by S and reads S's value through the birth frame (`model.py:276` [code]). The formulation allows a hold to end past its scope and says persistence is not enforcement. Consistent. The receipt lists such holds as obligations carried past the scope end so a readout separates them (§3.6); ν assigns them to S by head ownership (§3.1). |
| N6 | Mid-generation cancellation | keep | `replace_interval` keeps the part of an interval before the frontier with its old value (`conditions.py:117-123` [code]), so an LN born under a cancelled directive still reads the old value at its birth role. This is "cancellation preserves committed history and ongoing holds". |
| N7 | Gameplay-demand requests | out of scope, field reserved | The formulation carries demand requests with their own semantics; `gameplay-state.md:344-349` [code] says their interface is open until the response specification exists. The request schema reserves the field name `demand` and rejects any value (§3.2). |
| N8 | Baseline when no chart seed is given | folded into G1 | From BOS, R2 establishes no explicit baseline; what emerges is measured by G3(c). This is the "sampled baseline" case of stage R. |
| N9 | Section-level semantics of a target | checked, consistent | A target concerns the declared section, not every window in it. v2's draw already uses the source's share over the whole candidate section as the value even when the scored window covers part of it (§6.1 step 2), and the committed counters are section-level. The emphasis term scores per-decision likelihoods, not a per-window constraint. Stated in §6.1 so the next reader need not re-derive it. |
| N10 | A style change should leave an active property target met where both can hold (the mirror of G3(b)) | adopt (diagnostic) | With a style directive and an active property target on one scope, the property's readout stays within the following threshold. Stage 4 diagnostic G3(e) (§7.7). |

### 2.2 On the adversarial review (carried from v2, unchanged)

v2 ruled on the Opus review ([r2-condition-plan-review](r2-condition-plan-review.md)); v3 keeps every ruling. Summary so this plan stands alone:

| Point | Ruling | Where it lands |
| --- | --- | --- |
| P1 whole-song LN requests are out of distribution (no LN span above 64 beats, `conditions.py:20, 26-31` [code]) and near ceiling (0.715 at 61.4M [data]) | adopt | Primary is in-distribution span following at onsets (§7.4, §10 stage 1); long spans in both arms (§6.2, Q-B); thresholds relative to A0. |
| P2 dropout after alignment biases natural rows; the design's IPW rule (`design.md:303` [data]) | adopt | Dropout first, alignment among survivors, informativeness against unscored rows, per-window inverse-probability weights, a draw-independent natural manifest (§6.1, §6.3). |
| P3 L_κ is a reweighted CE; governed-factor split; uniform base | adopt | The split is exact: `common.py:7-18` [code] gives codes 1/2 (free) and 3/4 (held) as the only tap-versus-LN pairs, so grouping the 625 codes by "head or not and release type" is a bijection; log P(ℓ∣r) is a group log-sum-exp. Base CE uniform; emphasis additive with λ = 0 in stage 1 (§4.2-4.3). |
| P4 direction 2 asks for a realised-response loss; F3 cost; baseline on 30 s only; residual leak; negative-star validation | amend | Q-G put narrowly with costs (§5.3); baseline refitted on every interface length (§5.1); within-chart residual correlation measured with a stated rule (§5.4); `conditions.py:100` and `features.py:276` change in stage 0. |
| P5 attribution, stop rule, checkpoint noise, cost, budget | adopt | Arms differ in the draw only; plateau stop off; read at 50M on the mean of the last three checkpoints; onset-row NLL paired; effect per training seed; costs include evaluation (§7.8). |
| P6 release factors read conditions across span boundaries (`model.py:274-276` [code]); ownership by birth | adopt | Ownership: action factor by row, release factor by its LN's head time, EOS included (§4.1); locality tests per factor (§4.4). |
| P7 D3 missed; ∣Δ∣ > 0.1 within binomial noise; late rows informative; skeleton predicts share | adopt | D3 thresholded before the simulation (§6.3); z-criterion with n ≥ 20 heads; strata onset / middle / end (§7.1). |
| P8 decisions taken that belong to the human | adopt | Presence is now settled (C5); weights uniform by default; direction 2 is Q-G; cell lengths Q-C (default). |
| P9 L28 is not evidence of oscillation after 35M | amend | Conclusion adopted; v2's recount stands (§1.2); "CE recipe first" rests on the own-history shift 0.129 ± 0.044 and the natural LN bias +0.10 to +0.13 on three panels [data]. |
| P10 sync mechanism; manifest reuse; overlapping cells; D5; undocumented panel deviation | adopt | L36 corrected (staging limit); manifest version hash; one (length, phase) star partition per draw; two manifests; DEVIATIONS entry for the panel (§6.3, §9). |
| W1-W16 | adopted as ledger rows L41-L54 (§11). | |

Where the review and v1 agree (fixed expected denominators, no relabelling of source decisions with values they lack, F2 rejected, F4 a probe, the ridge baseline, the whole-song star as an interval and not a chart scalar, the infrastructure fixes, per-kind diagnostics with receipts, the landmark run deferred), v3 keeps v1.

---

## 3. Requests, properties, scopes

### 3.1 The `properties` module and ν

One module, `properties.py`, declares ν = (version, LN-share semantics, difficulty semantics) and computes every property from an object list (taps and holds with lane, start and end) and a scope [a, b) in song milliseconds. ν is hashed; the hash enters the label file, the manifests, every receipt and request validation. A chart may be committed or provisional: the function takes objects, not a run state (N3).

- **LNShare_ν(H̄, S).** Heads are objects whose start time t satisfies a ≤ t < b (half-open, song ms, one count per object). LN share = LN heads / heads. A hold whose head is in S belongs to S whatever its end; a hold whose head precedes a and ends inside S is not counted. Undefined (None, "unreadable") when S owns no head; never 0 by default. The row-level form from per-row counts (as `labels.ln_share`, `labels.py:42-45` [code], and the frame counters, `features.py:264-272` [code]) and the object-level form (as `report.py:46-48` [code]) are both implemented and a test asserts they agree on 200 source charts and 2,000 random scopes.
- **Difficulty_ν(H̄, S).** The tiled star as built (`labels.py:78-110` [code]): the objects whose heads lie in S, translated by −a, repeated with period b − a to 240 s, the 1 ms seam rule, scored by `compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)`. Requires b − a ≥ 30 s. Context is the section's own objects only (a declared choice; the surrounding chart is not scored). A seam cut that leaves a nonpositive hold makes the readout "unreadable", never 0. `TILING_VERSION` and the calculator name are part of ν.
- **Users of ν.** The LN-share draw value (`conditions.py:42-52` [code] becomes a call); the star relabel (§6.2); the frame counters (§4.8; a test that the committed counters at row k equal LNShare_ν on the committed prefix restricted to S); every readout in §7; request validation (§3.2). The probes that computed their own share (`probe.py:177-184` [data]) are re-pointed to the module before they are frozen again (rule 7: the frozen evaluator is the hashed module, not a copy of its arithmetic).
- **Tests.** Labels equal readouts: for 200 fit_train charts, Difficulty_ν of the cache representation equals the stored label within 0.001 star except where the cache representation differs from the `.osu` (count reported, L13); LNShare_ν equals the draw value to 1e-9. Readout never echoes: generate with request v, overwrite the request record, the readout is unchanged; the readout of a source chart over any scope equals its label.

### 3.2 Request schema and resolution

```
Request:
  scope      = [a, b) in song ms (the whole song is [0, T))
  style      = None | {'named': {attribute: level}}       # 'reference', 'edit' reserved tags
  property   = None | {kind: target | (lo, hi)}           # kind in {ln_share, difficulty}
  eta        = 'immediate'                                # 'announce', 'ramp' reserved names
  priority   = None | int                                 # higher wins on overlap
  demand     = None                                       # reserved (N7); any value rejected
```

Either directive may be absent, both may be present. The scope is song time (answer 3): beats and bars are a caller convenience converted with the grid before the request is built.

**Resolution** `resolve(requests, skeleton, ν) → (effective track, receipt entries)`, deterministic:

1. Split each request into per-kind intervals (LN share, difficulty, each named style attribute).
2. Within a kind, where two intervals overlap, the interval with the higher explicit priority keeps the overlap and the other is clipped; the clip is a receipt entry naming both requests. Equal or missing priority on an overlap raises `ContractError('unprioritised overlap')` listing the pair: no hidden rule resolves it.
3. Adjacent intervals of one kind with equal value and equal η merge (as `replace_interval` merges, `conditions.py:126-131` [code]).
4. Across kinds nothing is resolved: a row under an LN scope and a difficulty scope reads both. If they cannot both be met the readout reports both; no trade-off is claimed or trained (C6).
5. The output is the per-kind non-overlapping track in today's `Interval` form, extended by (lo, hi) and η; `validate_track` keeps its non-overlap check as the invariant of the effective track.

**Feasibility flags** (the formulation's declared trade-offs; recorded, never silently applied): LN share outside [0, 1] or a scope owning no head → rejected as unreadable; difficulty with b − a < 30 s → rejected; difficulty target in absolute star: v_res = target − b(S) is computed from the skeleton (§5.1); ∣v_res∣ > 1.5 star → flagged "outside trained residual range", generation proceeds with the clipped value and the flag; a scope length outside the trained set (§6.2) → flagged "extrapolation"; a range request while ranges are untrained (Q-R(a)) → flagged "range not trained; midpoint used".

**Mid-generation edits.** The resolver re-runs on the edited request list with the frontier (the time of the last committed decision); `replace_interval` stays the primitive and the pre-frontier part of every interval is immutable (N6).

**Training.** Source values never conflict, so the draw emits effective tracks directly; a test asserts every drawn track passes `resolve` unchanged. The model never sees a request object, only the effective track.

### 3.3 Transition policy η

Only `immediate` is built: a directive applies from the first decision whose row time is ≥ a (or from the next decision after the frontier when injected mid-generation) and stops at the first decision whose row time is ≥ b. Release factors of LNs born inside the scope and decided after b still read the scope's value at their birth role (N5): a consequence that persists, listed in the receipt. There is no compensation and no hold: a row after b with no other scope sees the all-zero frame (§3.4).

Reserved, not built: `announce` (a channel that shows the next scope before it starts: the token conditioner in its current form is one instance, N1), `ramp` (a value path across the boundary). Each, when built, is a named input with its own lesion test and a draw that produces it in training; both stage-1 arms use `immediate`. A longer hold of a value is a longer scope, so there is no `hold` policy (G2).

### 3.4 Presence `none`

`features.py:291` is behind a switch `presence ∈ {none (default), anywhere (the current behaviour; a labelled arm only), announce (reserved, raises)}`. Under `none` channel 15 is zero; the channel is kept as the reserved announce channel so FRAME_DIM is stable. A row outside every scope of a kind sees exactly what it sees with no request of that kind. `DEVIATIONS.md` item 5 is rewritten. Tests: T-P3b(i) (§4.4) and the explicit-zero lesion (N4).

### 3.5 Baseline style ρ: a reserved slot

`continue_chart(..., baseline=None)` and the request CLI accept a `baseline` argument whose only admissible value in this repair is `None`; any other value raises, and the receipt records `baseline: none (identity from history)`. What ρ must satisfy and how it would be tested is stage R (§10); no representation is chosen here. Today identity lives only in the committed history, so an override enters the history and can shift what follows; G3(b) measures how much (§7.7).

### 3.6 Readouts and receipts

Every generation (in-run panel, CLI, probe) writes one receipt with: code identity and the checkpoint hash (as `receipts.py` [code]); the ν hash and the evaluator hashes; the random seed; the chart seed (hash of the prefix decisions, number of decisions, g0 = the time of the last committed decision, open holds at g0); the request list as given; the effective track with provenance per interval (which request, clipped by which, flags); η per interval; the baseline slot; per request and kind: intended (target or range), realised readout by ν on the exported chart over the scope, and for difficulty b(S), v_res and the realised residual; holds carried past a scope end; the defect counts of §7.4. The training receipt adds the conditioner type, the presence switch, λ and μ per kind, the IPW switch, the draw parameters and N̄ values, the manifest hashes. Rule 3: a component absent from the receipt is not in the system. Realised values are computed from exported objects, never copied from the request (test in §3.1).

### 3.7 Ranges: reserved encoding, point targets first (provisional)

Each property kind's value is encoded as a pair (lo, hi): LN share 2v − 1 each; difficulty v_res / 0.5 each, clipped to ±3 and flagged beyond ±1.5. A point target has lo = hi. Under the provisional default every training draw has lo = hi, so the named input "target pair" is lesioned jointly in stage 0; the separate half-width lesion test is added with Q-R(b). What a range changes if trained (Q-R(b)): the draw widens the source's value to (v − w, v + w) with w ~ U(0.05, 0.2) clipped to the kind's domain in half the LN candidates (difficulty likewise in star units, w ~ U(0.1, 0.4)); the loss is unchanged (CE on the source decisions, which lie inside the range by construction); the readout adds the hit rate (realised inside [lo, hi]) and the distance to the nearest bound. A range request at inference under Q-R(a) is served as its midpoint with the flag of §3.2.

---

## 4. The loss

### 4.1 Objects and ownership

- A chart is a head-row skeleton with K rows and song length T. Decision D_k at row k (k = K is EOS) factorises as one action factor (the 625-way masked joint action) and zero or more release factors (one per gap-released lane, scored in two orientations, `model.py:4-9, 236-251` [code]).
- A condition kind κ has an effective track of intervals (a, b, lo, hi), half-open, non-overlapping within a kind. The span of an interval is its time support; a scope is a span after resolution.
- **Ownership.** The action factor of D_k belongs to the span (per kind) containing t_k; EOS belongs to no span. A release factor belongs to the span containing its LN's head time, whatever row decides it, EOS included. This matches the pointer's birth role (`model.py:276`) and the label's head ownership (`labels.py:80`) and is the ν ownership rule (§3.1). Ω_κ is the set of scored factors owned by a span of kind κ. A factor may be in Ω_LN and Ω_star at once; attribution of a behaviour change to one kind is a matter of the counterfactual diagnostics (§7.2), not of the loss.
- **Inputs are not ownership [code].** The value of interval I enters every factor whose row time, candidate time or birth time lies in I (`model.py:274-276`). An LN born before I and released inside I reads I's value through the row and candidate frames; an LN born in I and released after b reads it through the birth frame. v3 keeps these inputs and attributes by ownership; the locality tests state what is tested, not assumed.

### 4.2 Terms

Per scored decision j, ℓ_j = −(log P(A_j) + log Q(U_j)) as now (action plus release mixture; the decision is the unit). Per factor f, ℓ_f is its own −log-probability; for a kind κ with governed factor g_κ, ℓ_f^κ is the governed part (§4.3).

- **Base**: L_base = (1/N̄) Σ_j u_j ℓ_j over all scored decisions, u_j the inverse-probability weight of §6.1 for decisions under no span and 1 otherwise, N̄ the fixed expected weighted count of scored decisions per batch (§4.6). This alone is the L20 fix: no window mean, no EOS or song-end over-weighting.
- **Emphasis per kind**: L_κ = (1/N̄_κ) Σ_{f ∈ Ω_κ} ℓ_f^κ, N̄_κ the fixed expected count of Ω_κ factors per batch.
- **Own-sample terms** (stage 3, §10): L_κ^own.
- **Total**: L = L_base + Σ_κ λ_κ (L_κ + μ_κ L_κ^own). Defaults λ_κ = 0, μ_κ = 0: plain uniform CE. λ_κ = 1 is a labelled arm ("emphasis"), never the silent default: any λ > 0 changes the balance between natural and conditioned training. Every L_κ is computed and logged at λ = 0, so the per-kind numbers of §7.1 exist in every run.
- The DPO anchor uses L_base (L22).

### 4.3 Governed factors per kind

| Kind | Statistic the value is made of | Governed factor g_κ and its exact likelihood | Excluded from L_κ |
| --- | --- | --- | --- |
| LN share (property) | LNShare_ν over the scope | Per head row: the tap-versus-LN choice of each head given the head mask and release types. Group the 625 codes by r(a): per lane, codes 1 and 2 (free) merge, codes 3 and 4 (held) merge, all else distinct. log P(ℓ∣r, s, C) = log P(a) − log Σ_{a' : r(a') = r(a)} P(a'), a group log-sum-exp over the masked log-softmax; a precomputed 625 → group index and a scatter. Cost: none measurable. | Head mask, release types, release positions, EOS. |
| Difficulty (property, residual internally, §5) | Difficulty_ν(scope) − b(scope) | The whole decision: action factor of rows in the span and every release factor owned by the span (tails after b included). Difficulty is a function of everything the decision sets, so no narrower governed factor exists. | Nothing owned by other spans. |
| Named-attribute style directive (Lens attribute, ordinal level) | the attribute's section judgment | The factor the attribute describes: head-mask sequence for jack, stream, trill (group by which lanes carry a head: per lane, free {1,2} and held {3,4} are "head", free {0} and held {0,1,2} are "no head"; the term scores log P(mask) = log Σ over the group); LN-ness plus owned releases for LN coordination; whole decision for tech. A style directive governs organisation, a property directive governs the result; both may be active on one row and their terms do not double count because their governed factors differ. | Per attribute. |

Under reading (a) of direction 1 ("rows outside the span") L_LN could be the whole-decision NLL on span rows; under reading (b) ("aspects the condition does not govern") it must be the governed part. The governed form satisfies both, is cheaper to reason about, and removes double counting when a style attribute overlaps an LN scope. Both numbers are logged (Q-M, a default; stage 3 only).

### 4.4 Locality, as tests that are true of the code to be written

For each kind κ and each term L_κ (bit-identical unless stated; a fixture chart with LN pieces, star cells, LNs born in one span and released in another, EOS releases, and one explicit-zero request):

- **T-O ownership.** A release factor of an LN born in span S and decided at a row after S's end (including EOS) is in Ω_S; an LN born before S and released inside S is not.
- **T-P1 targets.** Change the target of any scored factor not in Ω_κ that lies at or after the last row carrying an Ω_κ factor in its window: L_κ unchanged. For L_LN additionally: change the release targets at the last Ω_LN row: L_LN unchanged, L_star changes.
- **T-P2 gradient.** Backpropagate L_κ alone with hooks on the per-row action log-probabilities and on the pointer scores: exact zeros at every factor outside Ω_κ, non-zero inside; for L_LN, zeros on every pointer score.
- **T-P3a normalisation.** Add a natural-only window to the batch: every L_κ unchanged (fixed divisors).
- **T-P3b input locality.** (i) Edit the value, bounds or existence of an interval of kind κ that owns no scored factor and covers no candidate time of a scored factor: L_κ unchanged. **Binding with no exception under presence `none`** (C5); under the `anywhere` arm the test is expected to fail and says so. (ii) Edit an interval of another kind whose span contains no row time, candidate time or birth time of any factor in Ω_κ: L_κ unchanged.
- **T-G governed split.** Σ over the group of exp(log P(ℓ∣r) + log P(r)) equals P(a) for every legal code; the LN-share expected per row from the governed probabilities equals the value from the full action distribution.
- **T-Z explicit zero (N4).** For each property kind, the likelihood of a fixture decision under an explicit request equal to the kind's zero (LN share 0; difficulty equal to b(S)) differs from the likelihood under no request.
- **T-R resolver.** Every drawn track resolves to itself; two overlapping requests with priorities resolve to the clipped track; without priorities they raise; equal adjacent values merge; a mid-generation edit leaves the pre-frontier part unchanged.

These are engineering tests, binding on stage 0; none is evidence about learning.

### 4.5 What the two readings of direction 1 change

Nothing in stage 1 (λ = 0). In stage 3 they decide whether the emphasis and own-sample terms for a statistic-defined kind score the governed factor (default) or the whole decision on span rows. Difficulty is the same under both.

### 4.6 Normalisation

N̄ and N̄_κ are expected counts per batch under the draw, measured once by the stage-0 simulation (2,000 draws), stored in the run config and receipt, recomputed whenever a draw parameter changes; a test asserts the stored values are within 5 % of a fresh estimate, and that Σ_j u_j over 200 batches averages N̄ within 5 %. Gradient accumulation divides by the number of windows as now; the divisor is constant across batches.

### 4.7 Rows between scopes and after a scope (settled)

Membership is a property of the effective track. A row under no scope of a kind is natural for that kind: it sees the all-zero frame for that kind and enters L_base with weight u_j (§6.1); it is in no Ω_κ. Nothing is held, nothing compensates, no corpus-typical value is substituted: the source's own decisions are the only target on such rows, which is consistent with the defaults because the source is one free continuation under its own baseline (C4). A caller who wants the value held writes a longer scope. If an `announce` or `ramp` η is built later, the rows it touches get a named channel and a lesion test; whether they then join Ω_κ is part of that design. The loss code is the same in every case.

### 4.8 The frame per kind

Per kind: value encoding (property kinds: lo, hi, 2 channels; style attribute: one-hot {absent, supporting, prominent}, 3 channels; "unreviewed" = no interval, i.e. no frame) · offsets to the scope bounds (8) · progress (1) · kind-specific committed statistics over the owned part of the scope so far (≤ 4, by ν) · remaining rows (1) · active (1) · the reserved announce channel (1, zero under `immediate`). All lane-free; the hand-swap identity test stays binding.

| Kind | Value encoding | Committed statistics | Lesion tests (stage 0) |
| --- | --- | --- | --- |
| LN share | (2lo − 1, 2hi − 1) | log1p heads, log1p LNs, ratio, remaining (as now, by ν) | the target pair (jointly until Q-R(b)); each counter |
| Difficulty | (v_res,lo / 0.5, v_res,hi / 0.5), clipped ±3 | mean chord size, same-lane repeat rate, held-lane occupancy, LN share over the committed owned part (mirror-invariant) | the target pair; each proxy; b depends on the skeleton only |
| Style attribute | one-hot of the level | counts of the deterministic query evidence of the attribute (jack: same-column repeats; stream: directional four-note groups; trill: alternation of fixed groups; LN coordination: LN-occupied columns; tech: none) | per attribute; fixture kind in stage 4 |

A style label is per section (half-open source milliseconds), multi-label, ordinal, with unresolved and unreviewed statuses (`gameplay-state.md:211-239` [code]). The interface takes such a label when one exists; the labeller is not designed here; the attribute set is open-ended by schema (G4).

---

## 5. Difficulty relative to the skeleton

### 5.1 Baseline b(S)

A quadratic ridge from head-time features of the span to Difficulty_ν of the span, as A fitted (23 features, R² 0.746, RMSE 0.528 star on fit_dev [data]), **refitted on fit_train cells of every length the draw and the interface use** (30, 60, 120 s and the whole song, §6.2), frozen and hashed, kNN-32 as the check that the linear form loses nothing. b is a function of heads in S only: tests that changing every decision of a chart leaves b(S) unchanged, that b is mirror-invariant, and that b(S) depends only on heads in S. b(S) is computed from the skeleton before generation for every difficulty request, by the resolver (§3.2), and is not fed to the model (an optional arm feeds it; not adopted).

### 5.2 The residual as the internal condition value; absolute requests

A request carries an absolute target (or range) in star under ν. Internally v_res = target − b(S), clipped to ±1.5 in the frame and flagged beyond (§3.2). Corpus residual SD ≈ 0.53 star [data, A's RMSE]; the stage-0 refit prints the quantiles and the per-band width. Stage 0 changes `conditions.py:100` (negative values) and `features.py:276` (scale) (L49). The whole-song interval stays an interval with a residual value, not a chart-level scalar (rejected by the human on 2026-10-03). Readouts report absolute Difficulty_ν of the generated section, b(S) and the realised residual. Long-term: b(S) needs the given skeleton, which R2 has and a timing-generating system will not; the residual is an R2 parametrisation and the interface does not depend on it (C3).

### 5.3 "The model's response is trained towards the loss": what it can mean, with costs

Costs relative to one CE step of four 256-row windows (≈ 0.37 s at 2,800 decisions/s [inferred from `run.json` and the pilot]); sampling at ≈ 338 rows/s [data, v2, `evals.jsonl` free-run median]; tiled star ≈ 0.02 s per span [data, label build].

| Term | What it trains | Sampling | Cost | Gaming risk and guard |
| --- | --- | --- | --- | --- |
| F1-res: CE on owned factors of S with v_res in the frame (plus the committed proxies) | the likelihood of the source's own decisions under the residual it has | none | none | the proxies reveal part of the realised residual late in S; the history before S reveals the chart's offset if residuals correlate within a chart (§5.4) |
| TF-mono: teacher-forced monotone hinge on real onset states: g(E_θ[proxies ∣ s, r_hi]) − g(E_θ[proxies ∣ s, r_lo]) ≥ margin, g a frozen ridge from the committed proxies to the residual fitted on real cells | the direction of the value channel at onsets | none (two extra conditioner passes per row) | ≈ +20 % [inferred] | bounded and directional; surrogate gap tracked |
| Relaxed-proxy: sample S once from the real prefix with v_res, gradient through the per-row probabilities of (g(E_θ[proxies over S ∣ own history]) − v_res)² | the realised response on the model's own states | one sample per span, one window in four | ≈ +45 % on the step average [inferred] | hold-length features kept out of g; holds ≤ 60 ms guard; surrogate gap tracked |
| True F3: score-function on −(Difficulty_ν(generated S) − b(S) − v_res)² with a leave-one-out baseline over k = 4 samples | the realised tiled star itself | 4 samples per span until the last S-headed LN is released | ≈ 2.7× wall on one update in four [inferred] | fake holds raise star cheaply: guard (iv) and the CE anchor; k ≥ 4 |

**What v3 proposes.** Stage 2 trains F1-res (B1 against B0 absolute) and offers TF-mono as a third arm B2. The relaxed-proxy term and true F3 are own-sample terms: they use no preference labels, so the DPO decision does not cover them, but whether a self-supervised term on the model's own samples may run before real pairs exist is the human's standing question (Q-I). The narrow question for the human is Q-G (§12).

**Realised response, how measured.** For a requested (S, target) on a panel chart: teacher-force the real prefix to the onset, generate with the request from there until the last S-headed LN is released, export, score Difficulty_ν(S) with the hashed module, subtract the same b(S), compare with v_res over requests {−0.5, −0.25, 0, +0.25, +0.5} star of residual (absolute targets b(S) + v_res in the request): slope and MAE in star units. S is 30 s or 60 s.

### 5.4 Leak of the residual through the prefix

Stage 0 measures, from the refit's residuals, the within-chart correlation of adjacent cells. Rule stated now: if the correlation exceeds 0.5, star draws get an informativeness criterion like LN's (requested residual against the preceding cells' residual, weight 3 when ∣Δ∣ > 0.3 star) and the onset-row diagnostics for star are read against that stratum; otherwise star cells keep weight 1.

---

## 6. The draw

### 6.1 Steps (parameters are config; `draw_sim.py` measures them in stage 0)

1. Group uniform, chart uniform (unchanged).
2. **Candidates.** LN: a fresh random partition of the song into 8/16/32/64-beat pieces with at least one head (`conditions.py:34-52` returning all pieces); with p_long = 0.25 consecutive pieces are merged into runs of U{2..8} pieces; with p_whole = 0.10 the single candidate is [0, T). Values are LNShare_ν of the source over the candidate (prefix sums, online). The value is the section's share even when the scored window covers part of the section (N9). Star: one cell length drawn from {30, 60, 120 s} and one phase from {0, 10, 20 s}; the cells of that (length, phase) partition the song (no overlap, so `validate_track` holds); with p 0.10 the whole song instead. Values are the cached Difficulty_ν of the cell (§6.2 relabel).
3. **Dropout first**, on the candidate lists: drop all with 0.20; drop a kind with 0.25; drop each candidate with 0.20. Nothing after this step removes an interval. A dropped candidate's rows are unspecified for that kind: the all-zero frame, natural (C4).
4. **Alignment.** With p_align = 0.6, if any candidate survives: choose a kind uniformly among kinds with survivors; draw the lead u ∈ U{0..32}; for each surviving candidate of that kind compute its informativeness against the 64 rows ending at j = max(0, onset row − u), i.e. rows the window never scores: z = ∣v − share_before∣ / SE_binomial(v, n_heads) with n_heads ≥ 20 required; importance weight 3 if z ≥ 2 (LN), weight 1 otherwise; star cells weight 1 (or the §5.4 rule); the whole song weight 1. Choose one candidate by weight; set j as above, stop = min(j + 256, K+1). Otherwise (p 0.4, or no survivor): the current start rule (`data.py:74-81`).
5. **Intervals per window.** For each kind: the aligned candidate (if that kind) plus others among the surviving candidates intersecting [t_j, t_stop), U{1..3} in total, consecutive with p 0.5. The track is the union over kinds and is an effective track by construction (T-R).
6. **Inverse-probability weight.** For every decision in the window under no span, u_j = p_old(j ∣ chart) / p_new(j ∣ chart), with p_old the current start rule and p_new = 0.4 p_old + 0.6 P_align(j). Decisions under a span get u_j = 1. The draw is a known mixture, so the ratio is exact per window [inferred range: ≈ 0.25-0.65 in aligned windows, 2.5 elsewhere]. Effect: the natural conditional and its state distribution are those of the current recipe (the design's rule), while conditioned decisions get the aligned exposure. A switch turns the weight off; off is a labelled arm.

Values that are not the source's own are **not CE targets**. Two values on one skeleton come only from the model's own samples (stage 3) or accepted alternative arrangements (Q-J, a default: not now). Ranges, if trained (Q-R(b)), widen the source's value around itself and keep the source decisions as the target (§3.7).

### 6.2 Span lengths: training must cover what the interface promises

The interface is any scope in song time. Training today covers LN spans of 8-64 beats and star spans of 30 s, 60 s and the whole song [code]. Hence:

- **LN**: pieces, runs of pieces (up to 512 beats) and the whole song, at the probabilities in §6.1 step 2. The probabilities are Q-B; both stage-1 arms get the same lengths.
- **Star**: cells of 30, 60 and 120 s at 10 s offsets and the whole song (Q-C, a default). Relabel: ≈ 30 labels per chart, ≈ 375k labels, ≈ 36 min on 4 workers [inferred from 23 ms per label]; computed from the cache representation by the `properties` module (L13, C2). The label file carries the ν hash.
- **Evaluation spans** are drawn from the same lengths. A request outside them is flagged as extrapolation (§3.2).
- Difficulty_ν needs ≥ 30 s: star scopes are ≥ 30 s; the residual baseline is defined on the same spans.

### 6.3 Properties and the thresholds stated before the simulation

Measured by `draw_sim.py` on 2,000 draws against the cache (stage 0); each is also a test that skips when the cache is absent.

| Property | Threshold | Lever if missed (stated now) |
| --- | --- | --- |
| D1 onset coverage: among active intervals in a window, those with their onset row scored | ≥ 70 % | p_align up to 0.75 |
| D2 song-length independence: active intervals per window regressed on K | 2-SE interval of the slope contains zero | none needed by construction |
| D3 informativeness: scored LN rows within 16 rows of an onset with z ≥ 2 | ≥ 2 % of scored heads [inferred expectation 2-4 %] | importance weight 5, then keep every intersecting piece; natural rows stay protected by u_j |
| D4 active share | LN ≥ 25 %, star ≥ 25 % of scored heads [inferred 25-45 %] | reported; set by the dropout rates and the per-window count (plan defaults, no longer behind a human question) |
| D6 natural weight: effective sample size of u_j over natural decisions | ≥ 50 % of the natural count | p_align down to 0.5 |
| D7 long-span coverage: scored rows under LN spans longer than 64 beats | ≥ 20 % of LN-active rows at p_long 0.25, p_whole 0.10 | Q-B |

**Two manifests**: a **natural manifest** (64 fit_dev windows by the current start rule, empty tracks, random seed 954 for the draw; draw-independent, shared by every arm; it carries the selection primary) and a **condition manifest** (64 windows by the §6.1 draw, stratified so each kind has ≥ 24 windows with a scored onset, ≥ 8 with a span longer than 64 beats, ≥ 8 with a value switch at a boundary; diagnostics only). Both are versioned by a hash of the draw parameters, the label file and ν (L50) and frozen before stage 1.

---

## 7. Diagnostics and selection

All conditioned measurements use fixed panels, random seeds 954-956 unless stated, A's frozen `probe.py` re-pointed to the `properties` module and re-hashed, and a receipt per checkpoint (§3.6).

### 7.1 Teacher-forced, per kind and per factor (every log point; both manifests at every checkpoint)

NLL on Ω_κ factors, split: governed part and whole decision; strata onset (first 16 rows of a span), middle, end (last 16 rows); informative onsets (z ≥ 2). The **null contrast** log p(D_j ∣ v) − log p(D_j ∣ ∅) on Ω_κ rows by stratum. Natural-decision NLL on the natural manifest. Train-versus-dev NLL on onset rows (memorisation of up-weighted onsets).

### 7.2 Counterfactual response on fixed real states (every checkpoint)

A's P1 probe on two strata, onset rows and late rows: same state, value swapped (LN 0 / 0.9; residual −0.5 / +0.5): expected statistic change (governed probabilities for LN) and action KL. A late-row response without an onset response means the model follows counters, not the value.

### 7.3 Representation probe (per checkpoint, diagnostic only)

Linear probe from the hand vector at onset rows to v.

### 7.4 Free-run panels (every second checkpoint; receipts)

Charts × random seeds. The **prefix panel** is 16 fit_dev charts (A's 8 plus 8 across star bands and lengths to 1,500 rows) with the chart seed = the real decisions through the row nearest one third of the song (g0 at that row); it serves the natural guard and §7.7.

| Panel | Charts × random seeds | Requests | Metrics | Informs |
| --- | --- | --- | --- | --- |
| Natural from BOS | 16 × 3 | none | LN share against the source panel (reported, not binding: no prefix to pair on), B's `drift_ln_share`, holds ≤ 60 ms, releases 1-40 ms before another head, legality, chord-histogram JS | defect guard; descriptive |
| **Natural continuation (prefix panel)** | 16 × 3 | none | LNShare_ν of the generated continuation over [g0, T) against LNShare_ν of the real continuation of the same prefix over the same span; chart-paired mean difference and SE (2,000 chart-bootstrap resamples); SD ratio generated/real across prefixes | **guard (i), binding** |
| **Span following at onsets (stage-1 primary)** | A's 8 × 3 | 16- and 32-beat scopes at the beat-partition boundaries nearest 1/3 and 2/3 of the song; values {0.05, 0.3, 0.6, 0.9}; real prefix teacher-forced to the onset, generation through the scope until the last scope-headed LN is released | realised LNShare_ν over the scope; slope and MAE; per length | the stage-1 claim |
| Whole-song LN | 8 × 3 | {0, 0.1, 0.3, 0.6, 0.9} | slope, MAE | generalisation to the longest scope; reported |
| Switch | 8 × 3 | two adjacent requests 0.1 then 0.6, and 0.6 then 0.1, at half song (`immediate`) | DiD/2, per-half MAE | boundary response |
| Residual star (stage 2 on) | 8 × 3 | absolute targets b(S) + {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s onset-aligned scopes | slope, MAE in star | the stage-2 claim |

### 7.5 Teacher-forced against own-history calibration (every second checkpoint)

B's M1H: expected LN forecast on real histories against the same on own-sampled histories from the same start states (16 charts × 96 states). The exposure-bias number.

### 7.6 Checkpoint selection (fixed before any run)

1. Candidates: checkpoints after warm-up at regular cadence (safe checkpoints excluded, L47) whose natural and conditioned free-runs are all legal with every head present.
2. Primary: natural-manifest per-decision NLL, read as the mean over the checkpoint and its two predecessors (checkpoint-level noise 0.02-0.03 [data, v2], L48), with a paired song-group bootstrap SE (2,000 resamples).
3. Guards: (i) **binding**: on the prefix panel the chart-paired mean difference of continuation LN share (generated − real) is within ±0.05, and the SD ratio is reported (a ratio outside [0.5, 2] is a flag, not a failure, until a threshold is justified); never per chart against the chart seed; (ii) span-following slope ≥ 0.7 and MAE ≤ 0.15 on the onset panel, whole-song slope reported; (iii) own-history gap ≤ 0.05; (iv) holds ≤ 60 ms ≤ 0.5 % and releases 1-40 ms before another head within the source band's rate; (v) G3(a) no compensation (§7.7).
4. Select the earliest candidate passing all binding guards whose primary is within 2 SE of the minimum among passing candidates. None passing: "no selection", with the failing guard; such a checkpoint serves mechanism tests only.

Read against the overnight run [data]: the BOS natural share was 0.312 against 0.187 at 61.4M, so guard (i) in its old form failed there; the prefix-panel form has not been measured on any checkpoint and is a stage-0 measurement on ckpt-0061432779 before stage 1 (a baseline row, rule 6).

<a id="v3-g3"></a>
### 7.7 Diagnostics for the formulation's targets (G3; every second checkpoint unless stated; receipts)

All on the prefix panel of §7.4 (16 prefixes), random seeds 954-956 paired across conditions (the same random seed with and without the request), 2,000 chart-bootstrap resamples for SE. Organisation statistics, all by the `properties` module or the deterministic Lens query evidence: chord-size histogram (JS divergence against the comparison run), same-lane repeat rate (jack evidence), directional four-note run rate (stream), fixed-pair alternation rate (trill), hand-role balance (share of heads on outer against inner lanes, left against right), LN share.

- **(a) Release without compensation (binding from stage 1).** Override: an LN request on [t_{1/3}, t_{1/2}) with value 0.9 if the prefix's LNShare_ν < 0.5 else 0.05; nothing after t_{1/2}. Comparison: the natural continuation of the same prefix and random seed. Metric: Δ_post = LNShare_ν(override run, [t_{1/2}, T)) − LNShare_ν(natural run, [t_{1/2}, T)), chart-paired mean and SE. Holds born under the override and released after t_{1/2} belong to the override scope by ν and do not enter Δ_post. **Compensation** = Δ_post has the sign opposite to (override value − prefix share), ∣Δ_post∣ > 0.05 and > 2 SE. Pass: no compensation. Also reported: the override scope's realised share (following), and the defect rates (holds ≤ 60 ms, releases 1-40 ms before another head, legality) in the 32 rows after t_{1/2} against the natural run's rates on the same rows (continuity at the boundary). A positive Δ_post of the override's sign is "persistence", reported, no claim.
- **(b) Identity kept under a property change (reported in stages 1-2; binding in stage R).** Same prefix and random seed with and without an LN request of 0.3 on [t_{1/3}, t_{2/3}). Over that scope, the paired change in each organisation statistic the property does not govern (all but LN share). Reference spread: the same statistic's difference between two natural runs of the same prefix with different random seeds (954 against 955, 955 against 956). Identity is "kept" for a statistic when the mean paired change under the request is < 2 SE of the reference spread's mean absolute difference. Reported per statistic; also the LN following in the scope so a "kept identity with no following" is visible.
- **(c) Within- against between-identity variation (at 25M, 50M and the selected checkpoint; reported, no claim).** 16 prefixes × 8 random seeds (954-961), natural continuations. Per statistic: variance across random seeds within a prefix (mean over prefixes) against variance of per-prefix means across prefixes; the ratio with a bootstrap SE. The 39.49M number (across-random-seed variance twice across-skeleton, one checkpoint [data, judgment]) is the baseline row. The formulation's target (different identities differ, each supports several realisations) has no R2 mechanism yet; the number says where the model is.
- **(d) Receipts.** §3.6; the test that realised values are computed, not echoed (§3.1).
- **(e) Properties kept under a style directive (stage 4 only).** With a fixture style attribute active and an LN request of 0.3 on the same scope: LN following within guard (ii); the mirror of (b).

Cost: (a) 96 continuations of about two thirds of a song and (b) 48 more, ≈ 1.6 s each [inferred from 338 rows/s and a median of 500 rows], ≈ 4 min; (c) 80 further runs ≈ 2 min at its three checkpoints only. The natural continuations are shared between guard (i), (a) and (b).

### 7.8 Cost of in-run evaluation [inferred from `execution.json` and `evals.jsonl`, v2's figures plus §7.7]

Per full evaluation: manifests ≈ 10 s; natural from BOS 48 runs ≈ 2 min; prefix panel natural 48 runs ≈ 1.3 min; onset panel 384 short runs ≈ 3 min; whole-song 120 runs ≈ 5 min; switch 48 runs ≈ 2 min; calibration ≈ 5 min; G3 (a)+(b) ≈ 4 min; residual star (stage 2) 240 spans ≈ 4 min. About 20-26 min per full evaluation, every second checkpoint, plus ≈ 2 min for G3(c) at three checkpoints; the teacher-forced manifests at every checkpoint. A 50M run: 5.6 h of training at 2,500 decisions/s plus ≈ 2.0 h of evaluation ≈ 7.6 h.

---

## 8. Exposure bias

Teacher-forced forecasts are calibrated and own-history forecasts sit 0.129 ± 0.044 above them at 61.4M [data]; natural LN share from BOS is +0.10 to +0.13 above the source on three panels [data]. v3 keeps **check first, train second**: stage 1 measures the gap at every second checkpoint in both arms; stage 3 trains the relaxed own-history LN term (gradient through per-row governed probabilities on own-sampled spans; the sampling path carries no gradient) only with the human's permission (Q-I). This is not DPO and uses no preference labels; it is not scheduled sampling either.

---

## 9. Infrastructure

1. **Guard**: keep the RSS limit (12 GiB) and `min_available_bytes`; the system swap-growth trip (`runtime.py:129-133` [code]) measures other processes: default (Q-D) remove it and trip on the trainer's own RSS growth > 2 GiB between checkpoints; record `available_bytes`, `pressure_level` and RSS in every `resource_limit` event. **For the main thread**: the mac's `resources.jsonl` says whether the trainer's RSS grew across the 13 hours before the first trip.
2. **Restart budget as a rate**: at most 5 restarts in any 6 hours; a resume that reaches the next checkpoint clears the counter (`launch.py:27, 106-112` [code]).
3. **Log segmentation**: `train.jsonl` and `resources.jsonl` per checkpoint segment, each well under 4 MB; `resources.jsonl` compacted. **For the main thread** (L36): the existing run's two oversize files are re-staged by mutagen every cycle (`mutagen.yml:161, 179-182` [code]); a pattern exclusion for `r2-runs/**/train.jsonl` and `resources.jsonl` is a sync patch outside R2's code.
4. **Plateau stop**: available in the trainer, **off in every arm comparison** (L51).
5. **Evaluation in-run**: §7 in `evaluate()`; a request CLI (`--requests <file>` with the schema of §3.2, a chart seed, a random seed, a `baseline` slot) that writes the receipt of §3.6 next to the `.osu`.
6. **Memory**: a 200-window pilot recording peak RSS per window against factor × candidate counts (L37); float32 per-candidate arrays and per-factor chunking if confirmed.
7. **Receipts**: `receipts.py` gains the generation fields of §3.6; `train_ce.py:331-339` gains the training fields; a test that every receipt field named in §3.6 is present and non-null where required.

---

## 10. Stages

Throughput 2,000-4,000 decisions/s on 4 threads; budgets at 2,500/s. One training seed is labelled as such until the second runs.

### Stage 0: code, measurements, tests (≈ 2.5 working days of code; < 2 h of mac)

v2's stage 0 (≈ one working day): ownership map and per-factor terms with the governed split (§4); uniform base CE with fixed divisors and the inverse-probability weights (§4.2, §6.1); the §6.1 draw with span lengths (§6.2) behind config; relabel at the new lengths from the cache representation; baseline refit, frozen and hashed, with residual quantiles and the within-chart correlation (§5); frame generalisation with the star proxies and the residual encoding (§4.8, §5.2; `conditions.py:100`, `features.py:276`); the two manifests with version hashes; §7.1-7.6 evaluation; §9 guard, restart, log changes; DPO anchor on L_base; a DEVIATIONS entry for the panel (L46).

Added by v3 (≈ 1.5 working days): the `properties` module with ν, its hash and the label-equals-readout tests (§3.1; ≈ 0.25 d); the request schema, resolver, priority rule, feasibility flags and T-R (§3.2; ≈ 0.3 d); absolute difficulty requests through the resolver (§5.2; ≈ 0.1 d); presence switch with `none` default, `anywhere` arm, DEVIATIONS 5 rewritten, T-P3b(i) binding, T-Z (§3.4, §4.4; ≈ 0.1 d); (lo, hi) encoding with lo = hi draws (§3.7; ≈ 0.1 d); receipts with intended against realised and the three seed fields (§3.6, §9.7; ≈ 0.15 d); the prefix panel, guard (i) in its binding form and G3 (a)-(c) in `evaluate()` and the CLI (§7.4, §7.7; ≈ 0.4 d); the `baseline` slot (§3.5; ≈ 0.05 d); style-attribute naming in the frame code (≈ 0.05 d).

Mac measurements: `draw_sim.py` (N̄, N̄_κ, D1-D7, the u_j distribution); the relabel (≈ 36 min); the refit (seconds); the 200-window pilot (throughput, peak RSS, cost of the TF-mono and relaxed-proxy forward paths on one window in four); **the prefix panel, guard (i) and G3 (a)-(c) on ckpt-0061432779** as the baseline rows for stage 1 (≈ 12 min, 2 threads).

Tests that fail if the stage is wrong (engineering): input lesions per named input on a tiny model (`tests/r2/helpers.py` pattern), one forward each: the LN target pair; each LN counter; the residual target pair; each star proxy; active bit; presence bit under the `anywhere` arm; candidate-role and birth-role frames; a documented "unused" is not accepted (the half-width channel is part of the pair until Q-R(b)). T-O, T-P1, T-P2, T-P3a, T-P3b (binding), T-G, T-Z, T-R. D1-D7 on 2,000 draws. Weight-sum and stored-N̄ tests. b skeleton-only, mirror-invariant, span-local. Labels equal readouts; readout never echoes; receipt completeness; `baseline` other than none raises; `demand` other than none raises. Existing suites unchanged (normalisation, mirror, leakage, causality, free-run smoke).

### Stage 1: span-aligned, window-scoped draws at matched exposure (training)

- **Claim**: placing scope onsets in the scored window, with dropout before alignment and natural decisions weighted back to the current distribution, raises in-distribution LN following at matched exposure without worsening natural prediction or natural continuation.
- **Arms**: both on stage-0 code, uniform base CE, λ = 0, same span lengths, presence `none`, η `immediate`, point targets (lo = hi), absolute tiled star as the star value (stage 2 is the only star change). **A0**: the current start rule and per-song interval selection (p_align = 0, u_j ≡ 1). **A1**: the §6.1 draw. The one difference is the draw. Optional **A2**: A1 with λ_LN = 1 (Q-E).
- **Training seeds**: 171/471 and 172/472 (weights/draws) per arm; the second pair runs after the first pair's 50M read-out exists. Random seeds 954-956 in every panel.
- **Budget and stop**: 50M exposures (≈ 3.9 average passes), cosine horizon 50M, lr 1e-3 unless Q-F re-tunes it, checkpoints every 4.39M, full evaluation every second checkpoint, **no plateau stop**; both arms run to 50M; nothing is read before 50M.
- **Primary**: onset-panel slope (§7.4), per training seed, read on the mean of the last three checkpoints, chart-paired between arms.
- **Threshold**: A1 − A0 ≥ +0.15 and > 2 paired SE **in each training seed** (panel SE on a paired slope difference ≈ 0.06 [inferred]). Natural-manifest NLL: A1 − A0 ≤ +0.02 (mean of the last three checkpoints, paired bootstrap). Guard (i) within ±0.05 in both arms at 50M. G3(a): no compensation in either arm. Holds ≤ 60 ms ≤ 0.5 % in both arms. The onset-row NLL paired difference A1 − A0 under the true value must be negative and > 2 SE.
- **Secondary (reported)**: whole-song slope and MAE, switch DiD/2, own-history gap, BOS natural LN share and drift, G3(b), G3(c), D-properties realised in the run's draws.
- **Outcome rules**: threshold met → A1's draw is the base; stage 2. Not met with the onset-row NLL difference resolved → the value is read in teacher forcing but not realised in free-run → stage 3 moves ahead of stage 2 (Q-I). Neither → the sparse-signal hypothesis is dead for this interface; F2 and Q-J are the next candidates; no architectural conclusion. Guard (i) failed in both arms with the primary met → the draw works and the natural bias is a separate problem; stage 3's own-history term is the candidate (Q-I).
- **Cost**: 4 runs × 7.6 h ≈ 30.5 h of mac, sequential (A2: +15 h at two training seeds, +7.6 h at one).
- **Engineering tests**: a config-diff test that the arms differ only in `p_align` and the weight switch; an exposure-accounting test that both arms' 50M checkpoints are within 1 % in head decisions; the stage-0 suite green on each run's frozen copy; every receipt complete.
- **Human**: Q-B, Q-E, Q-F, Q-R before launch.

### Stage 2: difficulty as a residual (training)

- **Claim**: conditioning on Difficulty_ν − b(S), with the committed proxies, gives a residual response where the absolute value gave none.
- **Arms**: B0 absolute tiled star (the stage-1 winner's recipe); B1 residual; B2 per Q-G (TF-mono, or the relaxed-proxy term). One training seed first; the second if B1 − B0 passes at one.
- **Metric**: residual slope and MAE over absolute targets b(S) + {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s onset-aligned scopes, 8 charts × 3 random seeds, mean of the last three checkpoints.
- **Threshold**: B1 − B0 slope ≥ +0.3 and > 2 paired SE; A's star gate (slope ≥ 0.5, MAE ≤ 0.5 star) reported; LN metrics unchanged within 2 SE; guards (i), (iv), (v).
- **Budget, stop**: as stage 1; 2-4 runs × 7.6 h (B2: +1-2 runs).
- **Engineering tests**: b and proxy tests of stage 0; relabel consistency; config-diff (arms differ in the value encoding, and B2's term only); exposure accounting.
- **Human**: Q-G before launch; Q-H is a default.

### Stage R: baseline style ρ (reserved; after stage 2; no design commitment)

Entry: stage 2 read out and the human's go. The first deliverable is a design brief, not code; what follows is the specification it must meet, written now so the slot, receipts and diagnostics of this repair fit it.

- **What ρ must satisfy** (my paraphrase of [prompt-1](private/human-inputs/aa6818ba-9ce4-4d72-b1de-e7d78a7ffbee.md#prompt-1)): it can come from a chart seed, from stated preferences, from references used for style only, or be sampled when nothing is given; it stays in force across continuation calls and across local overrides until the caller changes it; making an override's organisation the new identity is a separate, explicit operation; returning to natural brings back ρ-conditioned generation while active property targets still apply; on the same music, history and property targets, two different ρ yield charts a reader can tell apart, and one ρ yields several different charts under different random seeds that each hang together; a property change leaves the identity in place where the two can coexist; a reference used for style shapes the organisation but never becomes part of the committed prefix.
- **How it would be tested** (the G3 panels with ρ as the manipulated variable; thresholds set in that brief before its runs): G3(c) between-identity variance across two ρ on the same prefixes exceeds within-identity variance across random seeds by > 2 SE on every organisation statistic; G3(b) binding: identity kept under an LN change of 0.3; G3(a) under ρ: return to natural after an override restores the pre-override organisation statistics within the within-identity spread; a round trip: extract ρ from chart A, generate on skeleton B, extract again, agreement above the extraction's test-retest agreement on real charts; persistence across two continuation calls with the same ρ equals persistence within one call; the "explicit update" operation changes the identity and the receipt shows it. Lesion test per ρ input.
- **Not decided**: whether ρ is a vector, tokens, a reference prefix, or a sampled latent; whether it is trained with CE alone or with an objective on free-run continuations; how references are encoded. The Fable judgment's ordering (a persistent vector only after CE alone holds one condition over a song) is the entry condition.

### Stage 3: own-sample terms (training; behind Q-I)

- **Claim**: the relaxed own-history LN term closes the own-history gap and lifts following without fake holds; the relaxed-proxy star term (if not already in stage 2) does the same for the residual.
- **Arms**: C0 the stage-2 winner; C1 with μ_LN = 1 (and μ_star = 1 if stage 2 passed), one window in four, continuation of 20M exposures from C0's selected checkpoint, both arms continued equally.
- **Metric, threshold**: own-history gap ≤ 0.05; onset-panel slope ≥ 0.9 and MAE ≤ 0.10, > 2 SE over C0; holds ≤ 60 ms ≤ 0.5 %; natural NLL within 0.02; guard (i); G3(a). True F3 (k = 4) on one update in sixteen as calibration of the surrogate gap, reported.
- **Budget**: 2 × 20M at ≈ 1.45× step cost ≈ 3.3 h each plus evaluation ≈ 0.8 h each.
- **Engineering tests**: T-P2 on the own-sample path; no gradient through sampling; a fixture test that 200 steps move the realised statistic toward the request by > 2 SE over three fixture random seeds, threshold fixed here and not loosened after a failure.

### Stage 4: named-attribute style path (engineering, no claim)

A fixture attribute with a deterministic section statistic (a three-level band of chord-size ≥ 2 rate) exercises the one-hot encoding, ownership, locality and lesion tests on a tiny model; a toy run shows a fixture response at 2 SE as a mechanism check; G3(e) on the fixture. No style claim; it is what lets a Lens label plug in as a named-attribute style directive without interface work. The attribute set is open by schema (G4).

### Deferred

Token versus FiLM: the token form is an announce instrument (N1); the comparison waits for an announce η, and a masked token form is not built. Landmark readout ablation (Q-L, default after stage 2; the test: NLL with and without the readout on rows with k > 511, a 6-level TCN arm). DPO with real pairs (horizon ≥ the longest conditioned span). Capacity and lr/decay re-tune on fit_dev evidence after stage 1. Reference and edit style directives, learned dimensions (G4). Gameplay-demand requests (N7). Provisional branches and prefix commit (N3).

---

## 11. Ledger

Status **C** checked in code or data (cited), **I** inferred. Action: **S0..S4, SR** the fixing stage, **D** deferred, **H** a human question, **MT** for the main thread, **none**, **settled**.

| # | Wrong piece | Evidence | St. | Action |
| --- | --- | --- | --- | --- |
| L1 | Condition values are always the source's own statistic; the counterfactual response is never supervised | `conditions.py:52, 62-75` | C | S1 informative draws; S3 own samples; Q-J (default: not now) |
| L2 | Inside a span the committed counters reveal the value; CE on middle rows needs no value read (the end is informative again) | `features.py:264-272, 296-302`; audit §2 | C | S0 onset / middle / end strata; S1 onset-aligned draws |
| L3 | Absolute tiled star is ¾ determined by the skeleton; no committed-difficulty quantity in the frame; value KL 1e-5 at 61.4M | A `star-prediction-summary.json`; recheck table; `features.py:296-302` | C | S0 proxies; S2 residual |
| L4 | "Natural behaviour" had no stated definition; the code implements "imitate the source on those rows" | `conditions.py:83-90`, `features.py:291` | C | **settled** (C4): free; source rows are the target in training; §4.7 |
| L5 | Rows between two spans had no stated semantics | `features.py:291, 308-330` | C | **settled** (C4, C5): natural; presence `none`; §4.7 |
| L6 | Spans drawn without knowing the window; LN active on 13.5 % of scored heads, 32 % of active rows with an unscored onset | `data.py:82`; audit #1 | C | S0 §6.1 (dropout first, alignment, IPW) |
| L7 | Interval count per song, not per window (LN-active 21 % for K < 600, 8 % for K ≥ 1500) | `conditions.py:55, 74`; audit #4 | C | S0 per-window selection; D2 |
| L8 | The fit_dev manifest inherits L6/L7: 14 of 64 windows with an active LN row | `data.py:94-119`; `fit_dev_manifest.json` | C | S0 two manifests (§6.3) |
| L9 | Window start rule; 25 % short windows; harmful only through 1/L | `data.py:74-81` | C | S0 keep the rule, fix the weighting |
| L10 | Star cells fixed per chart by a hash; an onset can never be placed relative to a window | `labels.py:56-75` | C | S0 relabel at 10 s offsets, three lengths and the whole song |
| L11 | Cells mostly longer than a window; the whole-cell value on every row | `labels.py:37`; audit #2 | C | S0 aligned draws score the onset side; lengths kept (§6.2); N9 |
| L12 | The whole-song star (p 0.10) fed per row though set by the hardest section | `conditions.py:67-69` | C | S2 residual on the same span; chart-level scalar rejected by the human |
| L13 | Labels from the original `.osu`, not the cache representation (≤ 0.00016 star measured) | `labels.py:116-118` | C | S0 relabel from the cache through ν |
| L14 | Tiled star needs ≥ 30 s: a constraint | `labels.py:35, 103-104` | C | kept; part of ν |
| L15 | Presence bit set on every row when any interval exists; T-P3b(i) fails through it | `features.py:291`; DEVIATIONS 5 | C | **settled** (C5): S0 switch, default `none` |
| L16 | Token form reads the whole track at every row | `features.py:308-330`; DEVIATIONS 6 | C | D: an announce instrument (N1), not default-compatible |
| L17 | Star frame carries LN counters and no difficulty statistic | `features.py:296-302` | C | S0 proxies (§4.8) |
| L18 | `Interval.value` scalar; no categorical kind; no range | `features.py:256-261, 275-276` | C | S0 per-kind encoding with (lo, hi); S4 |
| L19 | FiLM zero-initialised: natural mode is learned FiLM(0); not a defect | `model.py:62-64` | C | none |
| L20 | Loss is the window mean: EOS 4.2×, last 64 rows 1.8× | `train_ce.py:266, 269`; audit #5 | C | S0 uniform base CE, fixed divisor |
| L21 | No condition-scoped term; nothing weighted, tested or reported per kind | `train_ce.py:263-273` | C | S0 per-factor terms (§4) |
| L22 | DPO anchor reuses the window mean | `train_dpo.py:190` | C | S0 shared L_base |
| L23 | No term sees own histories; own-history LN forecast shift 0.129 ± 0.044 at 61.4M | recheck `average-table.md`; `train_ce.py:263-269` | C | S1 measure; S3 term (Q-I) |
| L24 | "Conditioned" NLL keyed on track presence (47 % of such decisions active) | `train_ce.py:145` | C | S0 per-factor, per-stratum NLL |
| L25 | In-run free-run natural only, from BOS | `train_ce.py:317` | C | S0 §7.4 |
| L26 | Selection on fit_dev action CE alone, no SE, no guards | `checkpoint-selection.json` | C | S0 §7.6 |
| L27 | Free-run panel 4 charts × 3 random seeds; noise floor ≈ 0.1 in LN share | `data.py:111-116`; `evals.jsonl` | C | S0 panels of §7.4 |
| L28 | Natural LN share on the 4-chart panel moves by up to 0.17 between adjacent checkpoints to the end, but safe pairs 52k-170k exposures apart move by 0.08-0.10, so late deltas are within panel noise | `evals.jsonl` [data, v2] | C | none; "CE recipe first" rests on L23 and the natural LN bias |
| L29 | fit_dev NLL minimum 1.975 at 61.4M, 2.061 at 121.7M; ≈ 9.6 average passes, more for charts in small groups | `evals.jsonl`; `summary.json`; `data.py:69-71` | C; cause I | S1 ≤ 4 passes, cosine at the budget; Q-F |
| L30 | No conditioned generation entry point with receipts | `train_ce.py:552-564`; `sampling.py:28-29` | C | S0 request CLI (§9.5) |
| L31 | No baseline b for a generated span's star | — | C | S0 §5.1 |
| L32 | Boundaries inside a window only by chance; `replace_interval` path unexercised except at natural onsets | `conditions.py:106-134` | C | S0 draws keep adjacent pieces; switches in the condition manifest; T-R |
| L33 | Free-run summary lacks following, calibration and the short-hold rate under requests | `report.py` | C | S0 §7.4 fields; receipts |
| L34 | Guard trips on system-wide swap growth since the trainer's start; six trips 1.19-4.15 GB; the trainer's RSS limit never tripped | `runtime.py:31, 95, 114, 119, 129-133`; `events.jsonl` | C; cause I | S0 §9.1; MT reads `resources.jsonl` |
| L35 | Every restart resets the swap baseline; lifetime restart budget 5 | `launch.py:27, 106-112` | C | S0 rate budget |
| L36 | `train.jsonl` and `resources.jsonl` exceed the 4 MB staging limit and are re-staged every cycle; not pattern-excluded | `mutagen.yml:161, 179-182` | C | S0 segmented logs; MT pattern exclusion for the existing run |
| L37 | RSS spikes to 2.5-4.6 GiB at irregular points; inferred cause per-candidate float64 arrays | mirrored `resources.jsonl`; `model.py:265-278` | C; cause I | S0 pilot; float32 and chunking if confirmed |
| L38 | Throughput 2,177/s wall with contention; 3,700-3,957 in the pilot | `run.json` | C | none (planning number 2,500) |
| L39 | Landmark readout: lesion none at 61.4M; redundancy with the 511-token field the inferred cause | recheck table; audit §3 | C; cause I | D; Q-L (default after stage 2) |
| L40 | DPO: synthetic labellers off the path; 64-decision horizon shorter than a 30 s span | `train_dpo.py` | C | D (waits for real pairs) |
| L41 | Release factors read frames at the LN's birth time and at candidate times across span boundaries | `model.py:274-276` | C | S0 ownership by birth; tests §4.4; N5 |
| L42 | No LN span above 64 beats; whole-song and half-song requests are extrapolation | `conditions.py:20, 26-31, 34-56` | C | S0 span lengths; Q-B; primary changed |
| L43 | v1's draw applied dropout after alignment; design rule requires inverse-probability weights for any balancing | plan v1 §4; `design.md:303` | C | S0 §6.1 steps 3-6 |
| L44 | v1 refitted b on 30 s cells with `duration` constant | A's feature list | C | S0 §5.1 |
| L45 | The star statistic owns tails decided after b; F3 must sample until the last owned LN is released | `labels.py:39, 78-98` | C | §5.3, §7.4 |
| L46 | Manifest and free-run panel depart from `design.md:535` without a DEVIATIONS entry; free-run charts K ≤ 600 | `design.md:535`; `data.py:95, 111` | C | S0 DEVIATIONS entry with the power cost; panels of §7.4 |
| L47 | Safe checkpoints at irregular exposures enter `evals.jsonl` | `train_ce.py:411, 433` | C | S0 excluded from selection; tagged in logs |
| L48 | Checkpoint-level fit_dev noise ≈ 0.02-0.03 not in the bootstrap SE | `evals.jsonl` 109.7-111.1M [data, v2] | C | S0 mean of the last three checkpoints |
| L49 | `validate_track` rejects star < 0; star normalised by /4 | `conditions.py:100`; `features.py:276` | C | S0 |
| L50 | Manifest reuse keyed on `star_conditions` only | `data.py:121-129` | C | S0 version hash (draw parameters, labels, ν) |
| L51 | A plateau stop can break a matched-exposure comparison | plan v1 §8.4 | C | off in comparisons |
| L52 | Stage costs omitted in-run evaluation | recheck `execution.json` | C | §7.8 |
| L53 | ∣Δ∣ > 0.1 is within binomial noise for 8-beat pieces (≈ 40 heads, SE ≈ 0.06) | audit rows per beat | I | S0 z-criterion with n ≥ 20 |
| L54 | At 61.4M the whole-song gate is met, switches are not, natural LN is biased +0.10 to +0.13, star has no effect | recheck tables | C | folded into L3, L23, L28, L42 and the stage-1 primary |
| L55 | Four separate definitions of LN share; tiled star is a version string; no ν object | `conditions.py:42-52`, `labels.py:42-45`, `report.py:46-48`, `probe.py:177-184`, `labels.py:39` | C | S0 `properties` module (C2) |
| L56 | No request object: callers build `Interval` tuples; no priority, no feasibility flag, no receipt of intended against realised | `sampling.py:28-29`, `conditions.py:94-103`, `receipts.py` | C | S0 schema, resolver, receipts (C1, C6, G3d) |
| L57 | v2 asked for difficulty requests in residual units | v2 §4.1 | C | S0 absolute requests (C3) |
| L58 | The token conditioner shows every interval of the track, started or not | `features.py:308-330`; DEVIATIONS 6 | C | D: announce instrument (N1) |
| L59 | Point values only; no range | `features.py:275-276` | C | S0 (lo, hi) (C8); Q-R |
| L60 | "Seeds" conflated in v2 | v2 §6.4, §9 | C | terminology; receipt fields (N2) |
| L61 | No baseline style slot; identity only in the committed history | `sampling.py:28`; judgment | C | S0 slot; SR (G1) |
| L62 | Generation receipts lack the chart seed (g0, prefix hash), the request list and readouts | `train_ce.py:314-325, 331-339` | C | S0 §3.6 |
| L63 | No test that an explicit zero request differs from no request | `tests/r2/test_inputs_matter.py` | C | S0 T-Z (N4) |
| L64 | No transition policy; activation and expiry exist only through `replace_interval`'s frontier | `conditions.py:106-134` | C | S0 η `immediate` named; `announce`, `ramp` reserved (G2) |
| L65 | `docs/formulation/` predates the formulation (no ρ, no request set; "learned natural style distribution") | `notation.md:189-217`, `gameplay-state.md:332-342` | C | H: whether the formulation enters `docs/formulation/` (the check's side observation; not a plan question) |

---

<a id="v3-questions"></a>
## 12. Decisions for the human

Only questions whose answer changes what is built or run, in plain words, each with a chart example, the options and what each changes, and the stage it blocks. The settled ones (natural and between-scope behaviour, presence, ρ now or later, musical time) are gone. Questions that v2 listed and that have a safe default are moved to the second list: no answer is needed; say so to change a default.

### Questions that block a stage (ordered by what they block)

**Q-B. How long may a training LN scope be?** (blocks stage 1: the draw is frozen before the runs)

- *What is decided.* Whether the model trains on LN-share scopes longer than 64 beats (16 bars at 4/4), so that long requests are something it has seen.
- *Example.* Today every training request looks like "LN share 0.4 for bars 9-16". A request "LN share 0.4 for the whole song" or "for bars 1-64" has never appeared in training; the model answers it by stretching what it learned on short scopes. Last session's whole-song test (slope 0.715) was such a stretch.
- *Options.* (a) Default: pieces of 8-64 beats as now, plus runs of 2-8 pieces one draw in four, plus the whole song one draw in ten, in both arms. Code: the candidate list of the draw. Runs: nothing extra; D7 measures the coverage. Effect: whole-song and half-song requests become in-distribution; onsets per window fall a little. (b) Pieces only, as now: the whole-song panel stays an extrapolation measure and the half-song switch test stays out of distribution. (c) Other probabilities: say which.

**Q-E. Does stage 1 also test the LN emphasis term?** (blocks the stage-1 budget)

- *What is decided.* Whether stage 1 runs only the two draw arms, or also a third arm with the LN emphasis term switched on.
- *Example.* A0 and A1 both train plain cross-entropy ("predict the source's decisions"); they differ only in which 256-row windows are scored (A1 chooses windows so that scope starts fall inside them). A2 would be A1 plus a second score on every head inside an LN scope for the choice "tap or LN" (λ_LN = 1): the model is pushed harder to get the LN share right exactly where a request is active, and nowhere else.
- *Options.* (a) A0/A1 at two training seeds: ≈ 30.5 h of mac; stage 1 says nothing about the term itself. (b) Add A2 at two training seeds: +15 h; stage 1 also says whether the term adds following beyond the draw. (c) A2 at one training seed: +7.6 h; a one-seed reading, labelled as such.

**Q-F. Keep last run's learning rate for stage 1, or re-tune first?** (blocks the stage-1 launch)

- *What is decided.* Whether to run a short learning-rate check before stage 1.
- *Example.* The overnight run used lr 1e-3 with a cosine schedule over 158M exposures and began to overfit after 61M (fit_dev NLL 1.975 → 2.061). Stage 1 runs 50M exposures with the cosine ending at 50M, so the model sees each chart about four times instead of ten.
- *Options.* (a) Default: keep lr 1e-3, cosine to 3e-5 over 50M, weight decay 0.01. No cost. (b) Two 30-minute pilots on the stage-0 code (lr 6e-4 against 1e-3, weight decay 0.01 against 0.05) judged on fit_dev NLL at equal exposure, then the better setting: ≈ 1 h of mac before stage 1. The tuning hour of 2026-10-03 showed run-to-run variation as large as the lr effect, so (b) is weak evidence either way.

**Q-R. May a request be a range, and should stage 1 train ranges?** (blocks stage 1's draw under option (b); otherwise nothing)

- *What is decided.* Whether a property request can say "between X and Y" instead of one number, and whether the model trains on such requests now.
- *Example.* "LN share 0.4 for bars 33-48" is a point target: the closer the realised share to 0.4, the better. "LN share between 0.3 and 0.5 for bars 33-48" is a range: any share inside counts as met and nothing inside is better than anything else. For difficulty: "4.0 stars for the chorus" against "between 3.5 and 4.0 stars for the chorus".
- *Options.* (a) Default: the interface accepts ranges from stage 0 and the receipt reports whether the realised value fell inside, the frame carries two value channels (low, high), but every training draw uses low = high; a range request at inference is served as its midpoint and flagged "range not trained". Code: two channels instead of one, one flag. Runs: none. (b) Train ranges from stage 1: half the draws widen the source's value into a range around it (±0.05 to ±0.2 for LN share, ±0.1 to ±0.4 star); the model sees ranges in training; the panels add the hit rate. Code: the draw and one lesion test. Runs: no extra mac time, but stage 1 then tests two things at once (the draw and ranges), so a failure is harder to attribute. If ranges are wanted, a stage-2 arm is the cleaner place.

**Q-G. For difficulty, what does "train the response towards the loss" mean?** (blocks stage 2's arms)

- *What is decided.* Whether difficulty is learned only from real charts, or whether the model's own generated sections are also scored for difficulty during training and pushed toward the request.
- *Example.* Request "4.2 stars for 0:30-1:00" on a skeleton whose head rows alone predict 3.9 stars, so the request asks for +0.3 star above the skeleton. Under (A) the model has learned from real charts how mappers make a section harder than its skeleton suggests (more LNs, bigger chords, same-lane repeats) and applies that; afterwards we measure what difficulty the generated section has. Under (B), in addition, sections are generated during training, scored with the difficulty evaluator, and a loss pushes the generated difficulty toward the request.
- *Options.* (A) Default: cross-entropy on real rows with the residual as the value; stage 2 = B0 (absolute star) / B1 (residual) / optional B2 (TF-mono, +20 % step cost, no sampling). (B) In addition a term on generated sections: B2 = the relaxed-proxy term, +45 % step cost, with true F3 as a calibration measurement one update in sixteen. (B) is a self-supervised term on the model's own samples (no preference labels, not DPO), so the permission question Q-I below applies to it.

### Defaults the plan takes (no answer needed; say so to change one)

- **Q-C star cell lengths** (stage 0): 30, 60, 120 s at 10 s offsets plus the whole song; ≈ 36 min relabel. Alternatives: add 45 and 90 s (≈ 55 min); 30 s and the whole song only.
- **Q-D guard** (stage 0): remove the system swap trip; trip on the trainer's own RSS growth > 2 GiB between checkpoints or RSS > 12 GiB; restart budget 5 per 6 h.
- **Q-H whole-song star interval and request range** (stage 2): keep the whole-song interval at p 0.10 as a residual; requests within ±0.5 star of the skeleton baseline (one SD), wider ones flagged.
- **Q-I own-sample terms before real preference pairs** (stage 3; stage 2 under Q-G(B)): not run without a yes. Options when asked: yes / no / only if stage 1 fails the free-run gate while passing the teacher-forced one.
- **Q-M reading of direction 1 for the emphasis terms** (stage 3): governed factor; both numbers logged from stage 0.
- **Q-J constructed alternatives** (accepted re-arrangements of one skeleton, which need a judge): not now.
- **Q-K style vocabulary** (stage 4): the five Lens names with three levels as named attributes; the set is open by schema.
- **Q-L landmark run**: after stage 2.
- **Priority rule** (stage 0): explicit integer priorities; an overlap of two requests of one kind without priorities is rejected, not resolved.
- **Natural guard threshold** (stage 1): ±0.05 on the prefix panel's chart-paired mean difference.
- **Token conditioner**: kept as the instrument for a future announce policy; not an arm.

Withdrawn since v2: Q1, Q2 (settled by answer 1); Q-A (presence; settled by answer 1). Withdrawn earlier: v1's Q3; Q10/Q11 (Q-D, Q-H); Q6 (Q-A); Q4 (Q-I); Q5 (Q-J); Q7 (Q-F); Q8 (Q-K); Q9 (Q-L).

---

<a id="v3-changes"></a>
## 13. Change log from v2

| Point | What changed |
| --- | --- |
| C1 | Request schema splits property directives from style directives (§3.2); Lens dimensions re-filed as named-attribute style directives (§4.3, §4.8); stage 4 renamed the named-attribute style path. |
| C2 | New `properties` module with ν, hashed; labels, the draw value, frame counters, readouts and request validation all use it; label-equals-readout and readout-never-echoes tests (§3.1); probes re-pointed before freezing. |
| C3 | Difficulty requested in absolute star; v_res internal through the resolver; readouts report absolute star, b(S) and the residual; range flags (§3.2, §5.2). |
| C4 | v2 §3.7 replaced by §4.7: between-scope and post-scope rows are natural, no hold, no compensation; Q1, Q2 dropped; D4 no longer behind a human question. |
| C5 | Presence default `none`; `anywhere` a labelled arm; T-P3b(i) binding with no exception; DEVIATIONS 5 rewritten (§3.4, §4.4). |
| C6 | Resolver with explicit priority, rejection of unprioritised overlap, merge of equal adjacent values, feasibility flags; overlap defined within a kind; T-R (§3.2, §4.4). |
| C7 | Closed by answer 3; nothing changed; Q-C stays a default. |
| C8 | (lo, hi) encoding reserved; stage 1 trains lo = hi; the target pair is one named input until ranges are trained; Q-R (§3.7, §4.8, §12). |
| G1 | `baseline` slot accepting only none; stage R specified by what it must satisfy and how it would be tested (§3.5, §10). |
| G2 | η `immediate` named and built; `announce`, `ramp` reserved; no `hold` policy (§3.3). |
| G3 | §7.7 (a)-(e) with metrics, panels, random seeds and thresholds; (a) binding from stage 1 (guard (v)); (b) reported until stage R; (c) at three checkpoints; receipts in §3.6; prefix panel in §7.4. |
| G4 | Tagged style field with reserved tags (§3.2); listed under deferred. |
| A1-A3 | Adopted as above. A4 open as Q-R in plain language. |
| N1 | Token conditioner re-filed as an announce instrument; the token-versus-FiLM comparison deferred until an announce η is wanted (§3.3, §10 deferred, L16, L58). |
| N2 | "Seeds" split into random, training and chart seeds throughout; receipt fields (§Terms, §3.6). |
| N3-N8 | Readouts take any object list; T-Z explicit-zero test; obligations carried past a scope end listed in receipts; `demand` field reserved and rejected; no-chart-seed case folded into stage R. |
| N9, N10 | Section-level semantics stated in the draw; G3(e) for stage 4. |
| Guard (i) | Binding at panel-distribution level on the prefix panel (generated against real continuations of the same prefixes), never per chart against the chart seed; the BOS comparison reported only (§7.4, §7.6). |
| Costs | Stage 0 ≈ 2.5 working days of code (+1.5), < 2 h of mac (+ the baseline rows on ckpt-0061432779); full evaluation 20-26 min (+4) plus G3(c) at three checkpoints; a run ≈ 7.6 h (+0.4); stage 1 ≈ 30.5 h (+1.5), A2 +15 h; stage 2 2-4 × 7.6 h; stage 3 +0.8 h evaluation per arm. |
| §12 | Rewritten in plain language with chart examples; five questions block stages (Q-B, Q-E, Q-F, Q-R, Q-G); the rest are defaults. |
| Ledger | L4, L5, L15 settled; L16 re-filed; L55-L65 added. |

---

## 14. What I could not check

- Mac-only files: the full `train.jsonl` and `resources.jsonl`; A's `star-predictions.parquet` (within-chart residual correlation); the cache `index.parquet`; the checkpoints. No `ens` call was made.
- Numbers tagged [data, v2]: `evals.jsonl` (NLL minimum and rise, checkpoint noise, adjacent-checkpoint deltas, free-run rows/s), `star_summary.json`, `summary.json`, `run.json`, the recheck `execution.json` (its JSON did not parse with the keys I tried), and the ridge metrics of `star-prediction-summary.json` (I re-read its feature list and the dev star mean and SD). v2's reading is carried; nothing in v3 turns on them beyond the cost estimates.
- Every [inferred] number of §5.3, §6.3, §7.7 and §7.8 until `draw_sim.py` and the stage-0 pilot run.
- The cost of G3 continuations (1.6 s each) is scaled from v2's 2.4 s whole-chart figure by row count; the first stage-0 evaluation on ckpt-0061432779 replaces it.
- Whether the prefix-panel natural guard passes or fails on any existing checkpoint: not measured by anyone; stage 0 measures it before stage 1.
- The private formulation was read in full; the paraphrases in §2.1 rows N1-N10, §3.3, §3.5, §4.7 and stage R are mine and are not the human's wording. Any reading of it that the human disputes replaces mine.
- Acceptability of any generated chart; no renders were viewed.
- Whether the five Lens attributes are the named attributes R2 should take (Q-K, a default).
