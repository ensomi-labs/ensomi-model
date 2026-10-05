# R2 condition-scoped losses and recipe repair: plan v2 (proposed, awaiting the human)

Shareable. Written 2026-10-05 by a fresh Fable subagent after [r2-condition-plan](r2-condition-plan.md) (v1) and its [adversarial review](r2-condition-plan-review.md); rules on every review point and supersedes v1 as the working plan. Serves [d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss) and [d-natural-gap-deferred](r2-average-and-control.md#d-natural-gap-deferred). Kept as written; nothing adopted until the human decides the questions in section 11.


Third voice (Fable subagent, control plane, read-only) after plan v1 and its adversarial review.
Code read at `r2/train` `7d9640a` (paths relative to `src/ensomi_model/r2/` unless stated). Nothing
in the repository was edited; no job was launched. Written as a plan: a reader needs neither v1 nor
the review. Section 2 rules on every review point; section 12 lists what changed from v1 and why.

Evidence tags: **[code]** checked at the cited line by me; **[data]** read by me from the named
file; **[inferred]** reasoning; **[to measure]** a number a stage-0 job must produce. Rules that
bind: a difference under two standard errors is "none"; no conclusion from a run below its stop rule;
mechanism tests are engineering, not evidence; every model forward has an input-lesion test per
named input; before any run the claim, metric, threshold, panel, seeds and stop rule are fixed here.

---

## 0. The plan in one paragraph

The loss becomes a uniform per-decision cross-entropy (every scored decision weighs one, fixed
divisor, no window mean) plus per-condition emphasis terms that score only the factors a condition
governs, in the spans it owns: for LN share the LN-versus-tap choice of each head given the head
mask (an exact split of the row likelihood), for difficulty the whole decision, with every release
owned by the span of its LN's head. The emphasis weights are zero in stage 1, so that stage tests one
thing: a draw that places condition spans in the scored window (onset-aligned, dropout applied before
alignment, natural rows weighted back to the old window distribution, spans of every length the
interface promises). Difficulty becomes a residual against a frozen ridge baseline of the skeleton,
fitted on the span lengths the interface uses; whether direction 2 also asks for a term on generated
spans is put to the human with costs. Stage 1's primary metric is in-distribution span following at
onsets, not whole-song slope (already 0.715 ± 0.045 at 61.4M [data]), with two arms, two training
seeds, matched 50M exposure, no plateau stop, thresholds relative to the control arm, and costs that
include in-run evaluation. Natural behaviour and between-span behaviour stay undecided: the dropout
rates, presence bit and natural weighting stay as now until the human changes them.

---

## 1. Facts the plan rests on

### 1.1 Code [code]

| Fact | Where |
| --- | --- |
| LN spans are single partition pieces of 8/16/32/64 beats; adjacent pieces stay separate intervals; 1-4 per song whatever K; the value is the source's own share over the piece | `conditions.py:20, 26-31, 34-56` |
| Star spans: cached cells (30 s, 60 s, start by a hash of the inputs) or the whole song with p 0.10; 1-3 per song; same cells every draw | `labels.py:56-75`, `conditions.py:59-75` |
| Dropout 0.20 all, 0.25 per kind, 0.20 per interval, applied to the per-song track | `conditions.py:83-90` |
| The window start never reaches the track draw | `data.py:68-83` (`draw_track` at `:82` sees no `start`) |
| Start rule: BOS p 0.125, last 256 rows p 0.125, else uniform in [0, K]; stop = min(start+256, K+1) | `data.py:74-81` |
| Loss = window mean of decision NLL, averaged over the batch; DPO anchor the same | `train_ce.py:266, 269`; `train_dpo.py` `window_ce` |
| "Conditioned" statistics keyed on a non-empty track | `train_ce.py:145` |
| In-run free-run is natural only; `continue_chart` accepts a track | `train_ce.py:317`; `sampling.py:28-29` |
| Presence bit set for a kind on every row when any interval of that kind exists in the track | `features.py:291`; `DEVIATIONS.md` item 5 |
| Row frame: value, offsets, progress, LN counters (heads, LNs, ratio, remaining), active; star value normalised by /4; `validate_track` rejects star < 0 | `features.py:296-304, 275-276`; `conditions.py:100` |
| Release factors read frames in three roles: at the row time, at every candidate time, and at the held LN's start time (birth); FiLM applied to the pointer query | `model.py:265-281, 283-291` |
| History tokens carry no condition features; FiLM enters only at the queried row and the pointer | `features.py:182-199`; `model.py:171-175` |
| Code table: per lane, free-before 0 nothing / 1 tap / 2 LN head; held-before 0 keep / 1 release on row / 2 gap release / 3 gap release + tap / 4 gap release + LN head; 625-way joint action under a support mask | `common.py:7-18`; `state.py:49-57` |
| Tiled star owns the heads in [a, b) with untrimmed tails | `labels.py:39, 78-98` |
| Manifest reused when only `star_conditions` matches; free-run charts K ≤ 600 | `data.py:121-129, 95, 111` |
| Safe checkpoints at resource trips enter `evals.jsonl` at irregular exposures | `train_ce.py:411, 433` |
| Resource guard trips on system-wide swap growth > 1 GiB since the trainer's own start; restart budget 5 for the run's life | `runtime.py:31, 95, 114, 119`; `launch.py:27, 106-112` |
| Mirror: `maxStagingFileSize: "4MB"`; pattern exclusions cover `/reports/**/*.jsonl` only; the comment says the limit does not stop the transfer and retries every cycle | `~/ensomi/mutagen.yml:161, 179-182` |
| Design rules: no LN-share balancing of the draw without inverse-probability weights; the evaluation manifest is 128 groups, 24 per star band | `artifacts/r2-ml-design-20261003/design.md:303, 535` |

### 1.2 Data [data]

- Recheck, ckpt-0061432779 (61.4M exposures, one training seed), `artifacts/r2-recheck-20261005/control-table.md`: whole-song LN slope 0.715 ± 0.045, MAE 0.115 ± 0.011 (A's gate slope ≥ 0.7, MAE ≤ 0.15: met); half-song switch DiD/2 0.272 ± 0.034 (gate 0.30: not met); natural LN share 0.312 ± 0.038 against 0.187 real (8 charts); teacher-forced Δ expected LN fraction for request 0→0.9 0.060 ± 0.010; star slope 0.014 ± 0.015, star value KL 1.0e-5 nats/row.
- `average-table.md`: generated-minus-real-history LN forecast shift 0.129 ± 0.044; natural LN share 0.315 vs 0.190 (24 charts) and 0.298 vs 0.202 (48 charts × 3 seeds); landmark lesion ΔNLL −9e-5 ± 3e-4 (none at 61.4M, against −0.0015 ± 0.0003 at 39.5M).
- `evals.jsonl`: fit_dev action NLL minimum 1.9753 at 61.4M, 2.0612 at 121.7M; natural-row NLL 1.979 at 61.4M; between checkpoints a few hundred updates apart it moves 0.02-0.03 (2.0286 → 2.0608 → 2.0391 at 109.7M, 111.0M, 111.1M). Adjacent-checkpoint natural LN share on the 4-chart panel: safe pairs 52k and 170k exposures apart differ by +0.081 and +0.097, as much as pairs 4.39M apart; after 35.1M, 1 of 24 adjacent deltas exceeds 2 SE under an unpaired seed SE (that 52k pair, z 2.4), 0 of 24 under a chart-paired SE at the 3-dof threshold. Source panel mean 0.281.
- `checkpoint-selection.json`: 32 candidates, tie tolerance 0.001, one checkpoint selected.
- `artifacts/r2-analysis-20261004/control/star-prediction-summary.json`: ridge on 23 head-time features (including `duration`), fitted on 74,907 intervals of all three lengths; fit_dev R² 0.746 ± 0.013, RMSE 0.528 ± 0.014; kNN-32 R² 0.753, RMSE 0.522; real dev star mean 3.59, SD 1.05.
- `artifacts/r2-cache/v1/labels/star_summary.json`: 82,773 labels in 472 s on 4 workers (≈ 23 ms per label per worker). `summary.json`: 12,531 charts, 13.97M head rows, fit_train 11,368 charts in 4,167 groups.
- `run.json`: 2,177 decisions/s wall over the run. `execution.json` of the recheck: 264 continuations in 9 min 50 s on 2 threads; B's package 32 min.
- Range audit (`r2-range-audit.md`, 3,000 draws): LN active on 13.5 % of scored heads, star 23.9 %; informative scored onset rows 0.18 %; median LN piece 77 rows; a window holds a median 38 s; 3.8 rows per beat.

---

## 2. Rulings on the review

Ruling: **adopt** (plan changed), **amend** (adopted with a change), **reject** (with evidence).

| Point | Ruling | Argument and where it lands |
| --- | --- | --- |
| P1 whole-song LN requests are out of distribution and near ceiling | adopt | `conditions.py:20, 26-31` [code]: no LN span above 64 beats; whole-song offsets and counters leave the trained range. Slope 0.715 at 61.4M [data] meets the gate. Primary becomes in-distribution span following (§6.4, §9.1); long spans enter training in both arms (§5.2, human call Q-B); star cells at several lengths (§5.2); thresholds relative to A0. |
| P2 dropout after alignment biases natural rows; design's IPW rule | adopt | v1 §4 applied dropout after alignment; `design.md:303` [data] is the rule. Draw: dropout first, alignment among survivors, informativeness against rows before the window start, per-window inverse-probability weights on natural decisions, a draw-independent natural manifest (§5.1, §5.3). |
| P3 L_κ is a reweighted CE; governed-factor split; uniform base | adopt | The split is exact: `common.py:7-18` [code] gives codes 1/2 (free) and 3/4 (held) as the only tap-versus-LN pairs, so grouping the 625 codes by "head or not and release type" is a bijection; log P(ℓ∣r) is a group log-sum-exp. Base CE uniform per decision; emphasis terms additive with λ = 0 in stage 1 (§3.2-3.3). Null contrast and the monotone hinge taken as diagnostic and stage-3 candidate (§6.1, §9.3). |
| P4 direction 2 asks for a realised-response loss; F3 cost; baseline on 30 s only; residual leak; negative-star validation | amend | Reading of direction 2 put to the human narrowly (Q-G) with the cheaper surrogates and honest costs (§4.3); the baseline is refitted on the span lengths the draw and interface use (§4.1); within-chart residual correlation is a stage-0 measurement with a stated rule (§4.4); `conditions.py:100` and `features.py:276` [code] change in stage 0 (L49). Not adopted: ordering (b) before (a) as fixed; the order depends on Q-G. |
| P5 attribution, stop rule, checkpoint noise, cost, budget | adopt | Arms differ in the draw only, both on stage-0 code (§9.1); plateau stop off in comparisons; read at 50M on the mean of the last three checkpoints; onset-row NLL as the A1−A0 paired difference; effect required in each training seed; conditioned panel every second checkpoint; costs recomputed (§6.7); train-versus-dev NLL on onset rows logged; group-uniform pass skew noted (L29). |
| P6 release factors read conditions across span boundaries; ownership by birth | adopt | `model.py:274-276` [code]: `fr[:, 2]` is the frame at `d.start[f.k][f.lane]`, `fr[:, 1]` at candidate times. v1 §2.1 was wrong. Ownership: action factor by row, release factor (EOS included) by its LN's head time, matching `labels.py:39` head ownership [code]. Locality tests restated per factor so that they are true of the code to be written (§3.4); F3 sampling runs until the last span-headed LN is released (§4.3). |
| P7 D3 missed; ∣Δ∣ > 0.1 within binomial noise; late rows informative; skeleton predicts share | adopt | D3 recomputed under dropout-first with the threshold stated before the simulation (§5.3); informativeness by a z-criterion with a minimum head count (§5.1); strata onset / middle / end (§6.1); a skeleton-only share predictor measured in stage 0 as a diagnostic, not a draw criterion (cost). |
| P8 decisions taken that belong to the human | adopt | Presence bit stays as now behind a switch (Q-A, documented T-P3b(i) exception); weights uniform by default; membership under Q2(c) listed as a consequence of Q2; direction 2 is Q-G; cell lengths are Q-C; v1's Q3 is withdrawn (P9). |
| P9 L28 is not evidence of oscillation after 35M | amend | Conclusion adopted; the count is not reproduced. My recount [data]: 1 of 24 adjacent deltas after 35.1M above 2 SE (unpaired seed SE; the 52k-exposure safe pair), 0 of 24 chart-paired at the 3-dof threshold; the review's "0 of 23, max 1.75" depends on an SE method I could not match. What settles it: safe pairs 52k-170k exposures apart move by 0.08-0.10, so the 4-chart panel's noise floor is about 0.1 and the late deltas sit inside it. L28 reads "unresolved on this panel"; the basis for "CE recipe first" is the own-history shift 0.129 ± 0.044 and the natural LN bias of +0.10 to +0.13 on three panels, both > 2 SE. |
| P10 sync mechanism; manifest reuse; overlapping cells; D5 conflict; undocumented panel deviation | adopt | `mutagen.yml:161, 179-182` [code]: a staging-size limit, not a pattern exclusion; r2-runs JSONL are re-staged every cycle (L36 corrected; for the main thread). `data.py:121-129` [code] (L50). Cells at 10 s offsets overlap, so one (length, phase) partition per draw (§5.1). Two manifests replace D5's equality test (§5.3). The panel's departure from `design.md:535` and the K ≤ 600 restriction are recorded as a deviation with its power cost (L46). |
| W1-W16 | adopted as ledger rows L41-L54 and corrections to L23, L28, L29, L36 (§10) | W15 is evidence, folded into the rows it changes. |
| Citation corrections | adopt | `launch.py:27` and `:106-112`; `mutagen.yml:161` is the staging limit; natural NLL at 61.4M is 1.979. |

Where the review and v1 agree (fixed expected denominators, no relabelling of source decisions with values they lack, F2 rejected as a maximised full-row contrast, F4 a probe, the ridge baseline, the whole-song star as an interval and not a chart scalar, the infrastructure fixes, per-kind diagnostics with receipts, token-versus-FiLM and the landmark run deferred), v2 keeps v1.

---

## 3. The loss

### 3.1 Objects and ownership

- A chart is a head-row skeleton with K rows and song length T. Decision D_k at row k (k = K is EOS)
  factorises as one action factor (the 625-way masked joint action) and zero or more release factors
  (one per gap-released lane, scored in two orientations, `model.py:4-9, 236-251` [code]).
- A condition kind κ has a track of intervals (a, b, v), half-open, non-overlapping within a kind.
  The span of an interval is its time support.
- **Ownership.** The action factor of D_k belongs to the span (per kind) containing t_k; EOS belongs
  to no span. A release factor belongs to the span containing its LN's head time, whatever row
  decides it, EOS included. This matches the pointer's birth role (`model.py:276`) and the label's
  head ownership (`labels.py:39, 80`). Ω_κ is the set of scored factors owned by a span of kind κ. A
  factor may be in Ω_LN and Ω_star at once (conditioned on both); attribution of a behaviour change
  to one kind is then a matter of the counterfactual diagnostics (§6.2), not of the loss.
- **Inputs are not ownership [code].** The value of interval I enters every factor whose row time,
  candidate time or birth time lies in I (`model.py:274-276`). So an LN born before I and released
  inside I reads I's value through the row and candidate frames; an LN born in I and released after
  b reads it through the birth frame. v1's claim that the value at row j influences p(D_j) and
  nothing else is false for the pointer. v2 keeps these inputs (the design's choice, `design.md`
  "candidate-dependent frames are scored inside the pointer") and attributes by ownership; the
  locality tests below are stated so that this is tested, not assumed.

### 3.2 Terms

Per scored decision j, ℓ_j = −(log P(A_j) + log Q(U_j)) as now (action plus release mixture; the
decision is the unit, `design.md` §4 "do not divide a row by its note count"). Per factor f,
ℓ_f is its own −log-probability; for a kind κ with governed factor g_κ, ℓ_f^κ is the governed part
(§3.3).

- **Base**: L_base = (1/N̄) Σ_j u_j ℓ_j over all scored decisions, u_j the inverse-probability
  weight of §5.1 for decisions under no span and 1 otherwise, N̄ the fixed expected weighted count of
  scored decisions per batch (§3.6). This alone is the L20 fix: no window mean, no EOS or song-end
  over-weighting, no dependence of a row's weight on how many rows lie outside spans.
- **Emphasis per kind**: L_κ = (1/N̄_κ) Σ_{f ∈ Ω_κ} ℓ_f^κ, N̄_κ the fixed expected count of
  Ω_κ factors per batch.
- **Own-sample terms** (stage 3, §9.3): L_κ^own.
- **Total**: L = L_base + Σ_κ λ_κ (L_κ + μ_κ L_κ^own). Defaults λ_κ = 0, μ_κ = 0: plain uniform
  CE. λ_κ = 1 is a labelled arm ("emphasis"), never the silent default, because any λ > 0 changes
  the balance between natural and conditioned training, which is inside the deferred natural
  question. Every L_κ is computed and logged at λ = 0 (it costs nothing), so the per-kind numbers of
  §6.1 exist in every run.
- The DPO anchor uses L_base (L22).

### 3.3 Governed factors per kind

| Kind | Statistic the value is made of | Governed factor g_κ and its exact likelihood | Excluded from L_κ |
| --- | --- | --- | --- |
| LN share | LN heads / heads over the span | Per head row: the tap-versus-LN choice of each head given the head mask and release types. Group the 625 codes by r(a): per lane, codes 1 and 2 (free) merge, codes 3 and 4 (held) merge, all else distinct. log P(ℓ∣r, s, C) = log P(a) − log Σ_{a' : r(a') = r(a)} P(a'), a group log-sum-exp over the masked log-softmax; a precomputed 625 → group index and a scatter. Cost: none measurable. | Head mask, release types, release positions, EOS. |
| Difficulty (residual, §4) | tiled star of the span − b(span) | The whole decision: action factor of rows in the span and every release factor owned by the span (tails after b included; the label owns them, `labels.py:78-98`). | Nothing owned by other spans. |
| Style dimension (Lens, §3.8) | ordinal section label | The factor the dimension describes: head-mask sequence for Jack, Stream, Trill (group by which lanes carry a head: per lane, free {1,2} and held {3,4} are "head", free {0} and held {0,1,2} are "no head"; the term scores log P(mask) = log Σ over the group); LN-ness plus owned releases for LN coordination; whole decision for Tech. | Per dimension. |

Under reading (a) of direction 1 ("rows outside the span"), L_LN could be the whole-decision NLL on
span rows; under reading (b) ("aspects the condition does not govern") it must be the governed part.
The governed form satisfies both readings, is cheaper to reason about, and removes the double
counting a whole-row term would create when a Lens dimension overlaps an LN span. v2 adopts it.
Both numbers (governed and whole-row NLL on span rows) are logged so the choice can be revisited; it
changes nothing in stage 1 (λ = 0) and is listed for the human as Q-M (can wait).

### 3.4 Locality, as tests that are true of the code to be written

For each kind κ and each term L_κ (all tests bit-identical unless stated; a fixture chart with LN
pieces, star cells, LNs born in one span and released in another, and EOS releases):

- **T-O ownership.** A release factor of an LN born in span S and decided at a row after S's end
  (including EOS) is in Ω_S; an LN born before S and released inside S is not. Fails if the owner
  map is computed from the deciding row.
- **T-P1 targets.** Change the target of any scored factor not in Ω_κ that lies at or after the last
  row carrying an Ω_κ factor in its window (so no Ω_κ state changes by causality): L_κ unchanged.
  For L_LN additionally: change the release targets at the last Ω_LN row: L_LN unchanged, L_star
  changes.
- **T-P2 gradient.** Backpropagate L_κ alone with hooks on the per-row action log-probabilities and
  on the pointer scores: exact zeros at every action factor and release factor outside Ω_κ, non-zero
  inside; for L_LN, zeros on every pointer score.
- **T-P3a normalisation.** Add a natural-only window to the batch: every L_κ unchanged (fixed
  divisors).
- **T-P3b input locality.** (i) Edit the value, bounds or existence of an interval of kind κ that
  owns no scored factor and covers no candidate time of a scored factor: L_κ unchanged. **Documented
  exception while the presence bit is kept (Q-A): this test fails through `features.py:291`; it
  binds under the switch value `none`.** (ii) Edit an interval of another kind whose span contains no
  row time, candidate time or birth time of any factor in Ω_κ: L_κ unchanged. Both are true of the
  three-role pointer frames by construction; the edge they exclude (an interval covering only
  candidate times before the window's first row) is why the condition is stated with candidate times.
- **T-G governed split.** Σ over the group of exp(log P(ℓ∣r) + log P(r)) equals P(a) for every legal
  code; the LN-share expected per row from the governed probabilities equals the value computed from
  the full action distribution.

These are engineering tests, binding on stage 0; none is evidence about learning.

### 3.5 What the two readings of direction 1 change

Nothing in stage 1 (λ = 0). In stage 3 they decide whether the emphasis and own-sample terms for a
statistic-defined kind (LN share, Jack, Stream, Trill) score the governed factor (adopted default)
or the whole decision on span rows. Difficulty is the same under both. The only human question is
Q-M, which can wait until stage 3 is designed in detail.

### 3.6 Normalisation

N̄ and N̄_κ are expected counts per batch under the draw, measured once by the stage-0 simulation
(2,000 draws), stored in the run config, and recomputed whenever a draw parameter changes; a test
asserts the stored values are within 5 % of a fresh estimate, and that Σ_j u_j over 200 batches
averages N̄ within 5 %. Gradient accumulation divides by the number of windows as now; the divisor is
constant across batches. A batch with few Ω_κ factors contributes proportionally less to L_κ (not
inflated), and each factor carries the same weight in every batch.

### 3.7 Span membership under the deferred questions

Membership is a property of the track, not of the loss code. If the human settles "between two
conditioned parts" as hold-the-last-value, the track is rewritten by an explicit, logged transform
before the loss sees it; if as natural, nothing changes; if as an announced transition, an
announcement channel is added to the frame as a named input with its own lesion test, and whether
pre-span rows then belong to Ω_κ is part of that answer (the review's point: membership follows
visibility). The loss code is the same in every case; the track and the frame are what change.

### 3.8 Three kinds of value in one frame

The frame per kind becomes: kind-specific value encoding (≤ 4 channels) · offsets to the span
bounds (8) · progress (1) · kind-specific committed statistics over the owned part of the span so
far (≤ 4) · remaining rows (1) · active (1) · presence (1, while kept). All lane-free; the hand-swap
identity test stays binding.

| Kind | Value encoding | Committed statistics | Lesion tests (stage 0) |
| --- | --- | --- | --- |
| LN share | 2v − 1 | log1p heads, log1p LNs, ratio, remaining (as now) | value; each counter separately |
| Difficulty | v_res / 0.5, clipped to ±3 | mean chord size, same-lane repeat rate, held-lane occupancy, LN share over the committed owned part (mirror-invariant) | v_res; each proxy; b depends on the skeleton only |
| Style dimension | one-hot {absent, supporting, prominent}; "unreviewed" = no interval of that kind (absent frame) | counts of the deterministic query evidence of the dimension where one exists (Jack: same-column repeats; Stream/roll: directional four-note groups; Trill: alternation of fixed groups; LN coordination: LN-occupied columns; Tech: none) | per dimension; fixture kind in stage 4 |

Read from Beatmap Lens decisions 0004-0006: a style label is per episode or section (half-open
source milliseconds, adapted by the consumer), multi-label (dimensions judged independently),
ordinal (supporting / prominent, not calibrated numbers), with masks for unresolved and unreviewed.
The interface above takes such a label when one exists; the labeller is not designed here.

---

## 4. Difficulty relative to the skeleton

### 4.1 Baseline b(S)

A quadratic ridge from head-time features of the span to the tiled star of the span, as A fitted
(23 features, R² 0.746, RMSE 0.528 star on fit_dev [data]), **refitted on fit_train cells of every
length the draw and the interface use** (30, 60, 120 s and the whole song, §5.2), frozen and
hashed, kNN-32 as the check that the linear form loses nothing. A's fit used `duration` across mixed
lengths [data]; a refit on 30 s cells only (v1) would have made every whole-song b an extrapolation.
b is a function of heads in S only: tests that changing every decision of a chart leaves b(S)
unchanged, that b is mirror-invariant, and that b(S) depends only on heads in S. At inference the
caller gives a span and a requested residual; b is computed from the skeleton before generation and
is not fed to the model (an optional arm feeds it; not adopted).

### 4.2 The residual as the condition value

v_res = tiled_star(S) − b(S), in star units, clipped to ±1.5 in the frame and flagged beyond
(extrapolation). Corpus residual SD ≈ 0.53 star [data, A's RMSE]; the stage-0 refit prints the
quantiles and the per-band width. Stage 0 changes `conditions.py:100` (negative values) and
`features.py:276` (scale) (L49). The whole-song interval stays an interval with a residual value, not
a chart-level scalar (rejected by the human on 2026-10-03).

### 4.3 "The model's response is trained towards the loss": what it can mean, with costs

Costs relative to one CE step of four 256-row windows (≈ 0.37 s at 2,800 decisions/s [inferred from
`run.json` and the pilot]); sampling at ≈ 338 rows/s [data, `evals.jsonl` free-run median]; tiled
star ≈ 0.02 s per span [data, label build].

| Term | What it trains | Sampling | Cost | Gaming risk and guard |
| --- | --- | --- | --- | --- |
| F1-res: CE on owned factors of S with v_res in the frame (plus the committed proxies) | the likelihood of the source's own decisions under the residual it has | none | none | the proxies reveal part of the realised residual late in S (as LN counters do); the history before S reveals the chart's offset if residuals correlate within a chart (§4.4) |
| TF-mono: teacher-forced monotone hinge on real onset states: g(E_θ[proxies ∣ s, r_hi]) − g(E_θ[proxies ∣ s, r_lo]) ≥ margin, g a frozen ridge from the committed proxies to the residual fitted on real cells | the direction of the value channel at onsets, where history cannot substitute for the value | none (two extra conditioner passes per row; TCN shared) | ≈ +20 % [inferred] | bounded and directional: cannot be met by moving mass to arbitrary decisions; surrogate gap (g against true tiled star on generated spans) tracked |
| Relaxed-proxy: sample S once from the real prefix with v_res, gradient through the per-row probabilities of (g(E_θ[proxies over S ∣ own history]) − v_res)² | the realised response on the model's own states | one sample per span, one window in four | ≈ +45 % on the step average [inferred: 0.56 s sample + 0.08 s backward per 4 steps] | hold-length features kept out of g; holds ≤ 60 ms guard; surrogate gap tracked |
| True F3: score-function on −(tiled_star(generated S) − b(S) − v_res)² with a leave-one-out baseline over k = 4 samples | the realised tiled star itself | 4 samples per span until the last S-headed LN is released | ≈ 2.7× wall on one update in four [inferred, the review's arithmetic checked: 4 × 0.56 + 4 × 0.02 + 0.08 ≈ 2.6 s per span against 0.37 s] | fake holds raise star cheaply: guard (iv) and the CE anchor; variance of k = 1 with a global baseline is unbounded, hence k ≥ 4 |

**What v2 proposes.** Stage 2 trains F1-res (B1 against B0 absolute) and offers TF-mono as a third
arm B2 (cheap, no sampling, no own-sample term, so it needs no permission beyond the budget). The
relaxed-proxy term and true F3 are own-sample terms: they use no preference labels, so the DPO
decision does not cover them, but whether a self-supervised term on the model's own samples may run
before real pairs exist is the human's standing question (Q-I). **The narrow question for the human
(Q-G):** does direction 2 mean (A) the condition value is the residual and the model learns it by CE
on real rows, with the realised response of generated spans measured, or (B) in addition a term on
generated spans trains the realised difficulty towards the request? Under (A), stage 2 is B0 / B1 /
optional B2. Under (B), the relaxed-proxy term moves from stage 3 into stage 2 as B2 (replacing
TF-mono) at +45 % step cost, with true F3 (k = 4) as a calibration measurement on one update in
sixteen, and Q-I then applies to difficulty as well.

**Realised response, how measured.** For a requested (S, v_res) on a panel chart: teacher-force the
real prefix to the onset, generate with the request from there until the last S-headed LN is
released, export on the cache representation, score tiled_star(S) with the unchanged calculator,
subtract the same b(S), compare with v_res over requests {−0.5, −0.25, 0, +0.25, +0.5}: slope and
MAE in star units. S is 30 s or 60 s (in-distribution lengths).

### 4.4 Leak of the residual through the prefix

Stage 0 measures, from the refit's residuals, the within-chart correlation of adjacent cells. Rule
stated now: if the correlation exceeds 0.5, star draws get an informativeness criterion like LN's
(requested residual against the preceding cells' residual, weight 3 when ∣Δ∣ > 0.3 star) and the
onset-row diagnostics for star are read against that stratum; otherwise star cells keep weight 1.
A's per-chart RMSE 0.457 against global 0.528 [data] cannot separate the two cases.

---

## 5. The draw

### 5.1 Steps (parameters are config; `draw_sim.py` measures them in stage 0)

1. Group uniform, chart uniform (unchanged).
2. **Candidates.** LN: a fresh random partition of the song into 8/16/32/64-beat pieces with at
   least one head (`conditions.py:34-52` returning all pieces); with p_long = 0.25 consecutive
   pieces are merged into runs of U{2..8} pieces (the candidates are then the runs); with
   p_whole = 0.10 the single candidate is [0, T). Values are the source's own share over the
   candidate (prefix sums, online). Star: one cell length drawn from {30, 60, 120 s} and one phase
   from {0, 10, 20 s}; the cells of that (length, phase) partition the song (no overlap, so
   `validate_track` holds); with p 0.10 the whole song instead. Values are the cached tiled star of
   the cell (§5.2 relabel).
3. **Dropout first**, on the candidate lists: drop all with 0.20; drop a kind with 0.25; drop each
   candidate with 0.20. Rates as now. Nothing after this step removes an interval.
4. **Alignment.** With p_align = 0.6, if any candidate survives: choose a kind uniformly among kinds
   with survivors; draw the lead u ∈ U{0..32}; for each surviving candidate of that kind compute its
   informativeness against the 64 rows ending at j = max(0, onset row − u), i.e. rows the window
   never scores: z = ∣v − share_before∣ / SE_binomial(v, n_heads) with n_heads ≥ 20 required;
   importance weight 3 if z ≥ 2 (LN), weight 1 otherwise; star cells weight 1 (or the §4.4 rule);
   the whole song weight 1. Choose one candidate by weight; set j as above, stop = min(j + 256, K+1).
   Otherwise (p 0.4, or no survivor): the current start rule (`data.py:74-81`).
5. **Intervals per window.** For each kind: the aligned candidate (if that kind) plus others among
   the surviving candidates intersecting [t_j, t_stop), U{1..3} in total, consecutive with p 0.5.
   The track is the union over kinds. Presence (while kept, Q-A) is computed from the song-level
   surviving candidate list, not from the window's selection, so the bit means the same under
   either arm.
6. **Inverse-probability weight.** For every decision in the window under no span,
   u_j = p_old(j ∣ chart) / p_new(j ∣ chart), with p_old the current start rule and
   p_new = 0.4 p_old + 0.6 P_align(j), P_align computed from the surviving candidates, their weights
   and the uniform lead. Decisions under a span get u_j = 1. The draw is a known mixture, so the
   ratio is exact per window [inferred range: ≈ 0.25-0.65 in aligned windows, 2.5 elsewhere].
   Effect: the natural conditional and its state distribution are those of the current recipe (the
   design's rule, and "natural stays as now"), while conditioned decisions get the aligned exposure.
   A switch turns the weight off; off is a labelled arm, not a default.

Values that are not the source's own are **not CE targets** (label noise that teaches the value is
not to be followed; v1 and the review agree). Two values on one skeleton come only from the model's
own samples (stage 3) or accepted alternative arrangements (Q-J).

### 5.2 Span lengths: training must cover what the interface promises

The interface is "whole song or an interval". Training today covers LN spans of 8-64 beats and star
spans of 30 s, 60 s and the whole song [code]. Hence:

- **LN**: pieces, runs of pieces (up to 512 beats) and the whole song, at the probabilities in
  §5.1 step 2. Half-song switches (A's panel) become in-distribution through runs. The
  probabilities p_long and p_whole are a human call (Q-B): they trade onset count against long-span
  coverage. v2's default is stated; both stage-1 arms get the same lengths.
- **Star**: cells of 30, 60 and 120 s at 10 s offsets and the whole song. The 60 s cells are not
  dropped (v1 dropped them; that narrowed the interface, P1.3). Relabel: ≈ 30 labels per chart,
  ≈ 375k labels, ≈ 36 min on 4 workers [inferred from 23 ms per label]; computed from the cache
  representation (L13). The set of lengths is Q-C.
- **Evaluation spans** are drawn from the same lengths: whole song (LN and star), 16 and 32 beats
  at onsets (LN primary), half-song switches, 30 s and 60 s star spans. A request outside these
  lengths is reported as extrapolation.
- Tiled star needs ≥ 30 s (`labels.py:35`): star spans are ≥ 30 s; the residual baseline is defined
  on the same spans.

### 5.3 Properties and the thresholds stated before the simulation

Measured by `draw_sim.py` on 2,000 draws against the cache (stage 0); each is also a test that skips
when the cache is absent.

| Property | Threshold | Lever if missed (stated now) |
| --- | --- | --- |
| D1 onset coverage: among active intervals in a window, those with their onset row scored | ≥ 70 % | p_align up to 0.75 |
| D2 song-length independence: active intervals per window regressed on K | 2-SE interval of the slope contains zero | none needed by construction (per-window selection) |
| D3 informativeness: scored LN rows within 16 rows of an onset with z ≥ 2 | ≥ 2 % of scored heads [inferred expectation 2-4 %: alignment ≈ 0.14 per window × 16 rows, plus onsets of the 1-3 intersecting pieces ≈ 5 rows per window; the review's 0.6-1 % was for v1's draw] | importance weight 5, then keep every intersecting piece; the natural rows stay protected by u_j whatever the lever |
| D4 active share | LN ≥ 25 %, star ≥ 25 % of scored heads [inferred 25-45 % at the parameters above] | reported; set by the dropout rates and the per-window count, both the human's (Q1) |
| D6 natural weight: effective sample size of u_j over natural decisions | ≥ 50 % of the natural count | p_align down to 0.5 |
| D7 long-span coverage: scored rows under LN spans longer than 64 beats | ≥ 20 % of LN-active rows at p_long 0.25, p_whole 0.10 | Q-B |

**Two manifests** replace v1's D5: a **natural manifest** (64 fit_dev windows by the current start
rule, empty tracks, seed 954; draw-independent, shared by every arm; it carries the natural guard and
the selection primary) and a **condition manifest** (64 windows by the §5.1 draw, stratified so each
kind has ≥ 24 windows with a scored onset, ≥ 8 with a span longer than 64 beats, ≥ 8 with a value
switch at a boundary; diagnostics only). Both are versioned by a hash of the draw parameters and
the label file (L50) and frozen before stage 1.

---

## 6. Diagnostics and selection

All conditioned measurements use fixed panels, seeds 954-956, A's frozen `probe.py` where it exists,
and a receipt per checkpoint.

### 6.1 Teacher-forced, per kind and per factor (every log point; both manifests at every checkpoint)

NLL on Ω_κ factors, split: governed part and whole decision; strata onset (first 16 rows of a span),
middle, end (last 16 rows: there v and the counters fix the remaining LN count, so the end is
informative, P7); informative onsets (z ≥ 2). The **null contrast** log p(D_j ∣ v) − log p(D_j ∣ ∅)
on Ω_κ rows by stratum (the pointwise information the condition carries; dropout trains both sides).
Natural-decision NLL on the natural manifest. Train-versus-dev NLL on onset rows (memorisation of
up-weighted onsets). *Informs*: whether the value is read where history cannot substitute; whether
onsets are memorised; the natural guard.

### 6.2 Counterfactual response on fixed real states (every checkpoint)

A's P1 probe on two strata, onset rows and late rows: same state, value swapped (LN 0 / 0.9;
residual −0.5 / +0.5): expected statistic change (governed probabilities for LN) and action KL. A
late-row response without an onset response means the model follows counters, not the value.

### 6.3 Representation probe (per checkpoint, diagnostic only)

Linear probe from the hand vector at onset rows to v. Representation failure versus decoder failure.

### 6.4 Free-run panels (every second checkpoint; receipts)

| Panel | Charts × seeds | Requests | Metrics | Informs |
| --- | --- | --- | --- | --- |
| Natural | 16 fit_dev (A's 8 plus 8 across star bands and lengths to 1,500 rows) × 3 | none | LN share against the source panel, B's `drift_ln_share`, holds ≤ 60 ms, releases 1-40 ms before another head, legality, chord-histogram JS | natural guard (under Q1(a)), defect guard |
| **Span following at onsets (stage-1 primary)** | A's 8 × 3 | 16- and 32-beat spans at the beat-partition boundaries nearest 1/3 and 2/3 of the song; values {0.05, 0.3, 0.6, 0.9}; real prefix teacher-forced to the onset, generation through the span until the last span-headed LN is released | realised share over the span's heads; slope and MAE; per length | the stage-1 claim |
| Whole-song LN | 8 × 3 | {0, 0.1, 0.3, 0.6, 0.9} | slope, MAE (A's gate) | generalisation to the longest span; reported, not decisive |
| Switch | 8 × 3 | 0.1→0.6, 0.6→0.1 at half song | DiD/2, per-half MAE | boundary response (in-distribution through runs) |
| Residual star (stage 2 on) | 8 × 3 | {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s onset-aligned spans | slope, MAE in star | the stage-2 claim |

### 6.5 Teacher-forced against own-history calibration (every second checkpoint)

B's M1H: expected LN forecast on real histories against the same on own-sampled histories from the
same start states (16 charts × 96 states). The exposure-bias number. It does not depend on the
natural question: an own-history gap is exposure bias under any natural semantics.

### 6.6 Checkpoint selection (fixed before any run)

1. Candidates: checkpoints after warm-up at regular cadence (safe checkpoints excluded from
   selection, L47) whose natural and conditioned free-runs are all legal with every head present.
2. Primary: natural-manifest per-decision NLL, read as the mean over the checkpoint and its two
   predecessors (checkpoint-level noise 0.02-0.03 [data], L48), with a paired song-group bootstrap SE
   (2,000 resamples).
3. Guards: (i) natural LN share within 0.05 of the source panel mean **(presupposes natural =
   source; reported only until Q1 is answered)**; (ii) span-following slope ≥ 0.7 and MAE ≤ 0.15 on
   the onset panel, and whole-song slope reported; (iii) own-history gap ≤ 0.05; (iv) holds ≤ 60 ms
   ≤ 0.5 % and releases 1-40 ms before another head within the source band's rate.
4. Select the earliest candidate passing all binding guards whose primary is within 2 SE of the
   minimum among passing candidates. None passing: "no selection", with the failing guard; such a
   checkpoint serves mechanism tests only.

Read against the overnight run [data]: guard (i) fails at 61.4M (0.312 vs 0.187) and passes at
39.5-43.9M (0.230, 0.198), where the whole-song slope was 0.45-0.50; the fit_dev minimum and the
following gate pull apart on this run, which is why the guards are reported per arm and the
pre-stated primary decides arm comparisons.

### 6.7 Cost of in-run evaluation [inferred from `execution.json` and `evals.jsonl`]

Per full evaluation: manifests ≈ 10 s; natural 48 runs × 2.4 s ≈ 2 min; onset panel 8 × 3 × 2
onsets × 2 lengths × 4 values = 384 short runs × ≈ 0.5 s ≈ 3 min; whole-song 120 runs ≈ 5 min; switch 48
runs ≈ 2 min; calibration ≈ 5 min; residual star (stage 2) 240 spans × ≈ 1 s ≈ 4 min. About 15-20
min per full evaluation, run every second checkpoint; the teacher-forced manifests at every
checkpoint. A 50M run: 5.6 h of training at 2,500 decisions/s plus ≈ 1.6 h of evaluation ≈ 7.2 h.

---

## 7. Exposure bias

Teacher-forced forecasts are calibrated and own-history forecasts sit 0.129 ± 0.044 above them at
61.4M [data]; natural LN share is +0.10 to +0.13 above the source on three panels [data]. By v1's
rule (gap ≤ 0.05 and slope ≥ 0.7 make the own-sample term unnecessary) the gap already triggers
stage 3 for LN. v2 keeps **check first, train second**: stage 1 measures the gap at every second
checkpoint in both arms; stage 3 trains the relaxed own-history LN term (gradient through per-row
governed probabilities on own-sampled spans; the sampling path carries no gradient) only with the
human's permission (Q-I). This is not DPO and uses no preference labels; it is not scheduled
sampling either (no real label is scored on a substituted state).

---

## 8. Infrastructure

1. **Guard**: keep the RSS limit (12 GiB) and `min_available_bytes`; the system swap-growth trip
   (`runtime.py:114, 119`) measures other processes: remove it or key it on the trainer's own RSS
   growth between checkpoints (Q-D); record `available_bytes`, `pressure_level` and RSS in every
   `resource_limit` event. **For the main thread**: the mac's `resources.jsonl` says whether the
   trainer's RSS grew across the 13 hours before the first trip (one `ens cat`).
2. **Restart budget as a rate**: at most 5 restarts in any 6 hours; a resume that reaches the next
   checkpoint clears the counter (`launch.py:27, 106-112`).
3. **Log segmentation**: `train.jsonl` and `resources.jsonl` per checkpoint segment, each well
   under 4 MB; `resources.jsonl` compacted to one line per log point. **For the main thread
   (L36, corrected)**: the existing run's two oversize files are re-staged by mutagen every cycle
   (`mutagen.yml:161, 179-182`); a pattern exclusion for `r2-runs/**/train.jsonl` and
   `resources.jsonl` is a sync patch, outside R2's code.
4. **Plateau stop**: available in the trainer (no primary improvement > 1 SE over 3 checkpoints
   after 2 passes), **off in every arm comparison** (L51).
5. **Evaluation in-run**: §6 in `evaluate()`; the conditioned-generation CLI (`--freerun-only`
   with a track file and a receipt).
6. **Memory**: a 200-window pilot recording peak RSS per window against factor × candidate counts
   (L37); float32 per-candidate arrays and per-factor chunking if confirmed.

---

## 9. Stages

Throughput 2,000-4,000 decisions/s on 4 threads; budgets at 2,500/s. One training seed is labelled
as such until the second runs.

### Stage 0: code, measurements, tests (≈ one working day of code; < 1.5 h of mac)

Changes: ownership map and per-factor terms with the governed split (§3); uniform base CE with
fixed divisors and the inverse-probability weights (§3.2, §5.1); the §5.1 draw with span lengths
(§5.2) behind config; relabel at the new lengths from the cache representation; baseline refit,
frozen and hashed, with residual quantiles and the within-chart correlation (§4); frame
generalisation with the star proxies and the residual encoding (§3.8, §4.2; `conditions.py:100`,
`features.py:276`); presence switch {`anywhere` (now, default), `none`, `announce`}; the two
manifests with version hashes; §6 evaluation and the CLI; §8 guard, restart, log changes; DPO anchor
on L_base; a DEVIATIONS entry for the panel (L46).

Mac measurements: `draw_sim.py` (N̄, N̄_κ, D1-D7, the u_j distribution); the relabel (≈ 36 min);
the refit (seconds); the 200-window pilot (throughput, peak RSS, cost of the TF-mono and
relaxed-proxy forward paths exercised on one window in four).

Tests that fail if the stage is wrong (engineering): input lesions per named input on a tiny model
(`tests/r2/helpers.py` pattern), one forward each: LN value; each LN counter; residual value; each
star proxy; active bit; presence bit while kept; candidate-role and birth-role frames; announcement
channel if added; a documented "unused" is not accepted. T-O, T-P1, T-P2, T-P3a, T-P3b (with the
Q-A exception recorded), T-G. D1-D7 on 2,000 draws. Weight-sum and stored-N̄ tests. b skeleton-only,
mirror-invariant, span-local. Relabel consistency on the existing 30 s cells within 0.001 star
except where the cache representation differs (count reported). Existing suites unchanged
(normalisation, mirror, leakage, causality, free-run smoke).

### Stage 1: span-aligned, window-scoped draws at matched exposure (training)

- **Claim**: placing condition spans in the scored window, with dropout before alignment and natural
  decisions weighted back to the current distribution, raises in-distribution LN following at
  matched exposure without worsening natural prediction.
- **Arms**: both on stage-0 code, uniform base CE, λ = 0, same span lengths, presence as the human
  sets it, absolute tiled star as the star value (stage 2 is the only star change). **A0**: the
  current start rule and per-song interval selection (p_align = 0, u_j ≡ 1). **A1**: the §5.1 draw.
  The one difference is the draw. Optional **A2**: A1 with λ_LN = 1 (emphasis), if the human wants
  the emphasis tested in this round (+2 runs).
- **Seeds**: 171/471 and 172/472 (weights/draws) per arm; the second seed pair runs after the first
  pair's 50M read-out exists.
- **Budget and stop**: 50M exposures (≈ 3.9 average passes; group-uniform sampling gives charts in
  small groups several times more, L29), cosine horizon 50M, lr 1e-3 unless Q-F re-tunes it,
  checkpoints every 4.39M, full evaluation every second checkpoint, **no plateau stop**; both arms
  run to 50M; nothing is read before 50M (30M is descriptive).
- **Primary**: onset-panel slope (§6.4), per training seed, read on the mean of the last three
  checkpoints, chart-paired between arms.
- **Threshold**: A1 − A0 ≥ +0.15 and > 2 paired SE **in each training seed** (panel SE on a paired
  slope difference ≈ 0.06 [inferred from A's 0.042-0.045 per arm], so +0.15 is near the smallest
  resolvable effect; anything smaller is "none"). Natural-manifest NLL: A1 − A0 ≤ +0.02 (mean of the
  last three checkpoints, paired bootstrap). Holds ≤ 60 ms ≤ 0.5 % in both arms. The onset-row NLL
  paired difference A1 − A0 under the true value must be negative and > 2 SE.
- **Secondary (reported)**: whole-song slope and MAE, switch DiD/2, own-history gap, natural LN
  share and drift, D-properties realised in the run's draws.
- **Outcome rules**: threshold met → A1's draw is the base; stage 2. Not met with the onset-row NLL
  difference resolved → the value is read in teacher forcing but not realised in free-run → stage 3
  moves ahead of stage 2 (Q-I). Neither → the sparse-signal hypothesis is dead for this interface;
  F2 and Q-J are the next candidates; no architectural conclusion.
- **Cost**: 4 runs × 7.2 h ≈ 29 h of mac, sequential (A2: +14 h).
- **Engineering tests**: a config-diff test that the arms differ only in `p_align` and the weight
  switch; an exposure-accounting test that both arms' 50M checkpoints are within 1 % in head
  decisions; the stage-0 suite green on each run's frozen copy.
- **Human**: Q-B, Q-F before launch; budget.

### Stage 2: difficulty as a residual (training)

- **Claim**: conditioning on tiled_star − b(S), with the committed proxies, gives a residual
  response where the absolute value gave none.
- **Arms**: B0 absolute tiled star (the stage-1 winner's recipe); B1 residual; B2 per Q-G (TF-mono,
  or the relaxed-proxy term). One seed first; the second if B1 − B0 passes at one seed.
- **Metric**: residual slope and MAE over {−0.5, −0.25, 0, +0.25, +0.5} on 30 s and 60 s
  onset-aligned spans, 8 charts × 3 seeds, mean of the last three checkpoints.
- **Threshold**: B1 − B0 slope ≥ +0.3 and > 2 paired SE; A's star gate (slope ≥ 0.5, MAE ≤ 0.5 star)
  reported; LN metrics unchanged within 2 SE; guard (iv).
- **Budget, stop**: as stage 1; 2-4 runs (B2: +1-2).
- **Engineering tests**: b and proxy tests of stage 0; relabel consistency; config-diff (arms differ
  in the value encoding, and B2's term only); exposure accounting.
- **Human**: Q-G, Q-H before launch.

### Stage 3: own-sample terms (training; behind Q-I)

- **Claim**: the relaxed own-history LN term closes the own-history gap and lifts following without
  fake holds; the relaxed-proxy star term (if not already in stage 2) does the same for the residual.
- **Arms**: C0 the stage-2 winner; C1 with μ_LN = 1 (and μ_star = 1 if stage 2 passed), one window in
  four, continuation of 20M exposures from C0's selected checkpoint, both arms continued equally.
- **Metric, threshold**: own-history gap ≤ 0.05; onset-panel slope ≥ 0.9 and MAE ≤ 0.10, > 2 SE over
  C0; holds ≤ 60 ms ≤ 0.5 %; natural NLL within 0.02. True F3 (k = 4) on one update in sixteen as
  calibration of the surrogate gap, reported.
- **Budget**: 2 × 20M at ≈ 1.45× step cost ≈ 3.3 h each.
- **Engineering tests**: T-P2 on the own-sample path (gradient only through owned factors of the
  sampled span); no gradient through sampling; a fixture test that 200 steps move the realised
  statistic toward the request by > 2 SE over three fixture seeds, threshold fixed here and not
  loosened after a failure.

### Stage 4: categorical kind path (engineering, no claim)

A fixture kind with a deterministic span statistic (a three-level band of chord-size ≥ 2 rate)
exercises the categorical encoding, ownership, locality and lesion tests on a tiny model; a toy run
shows a fixture response at 2 SE as a mechanism check. No style claim; it is what lets a Lens label
plug in without interface work.

### Deferred

Token versus FiLM (after Q2); landmark readout ablation (Q-L; the test: NLL with and without the
readout on rows with k > 511, a 6-level TCN arm); DPO with real pairs (horizon ≥ the longest
conditioned span); capacity and lr/decay re-tune (Q-F) on fit_dev evidence after stage 1.

---

## 10. Ledger

Status **C** checked in code or data (cited), **I** inferred. Action: **S0..S4** the fixing stage,
**D** deferred, **H** a human question, **MT** for the main thread (outside R2's code), **none**.

| # | Wrong piece | Evidence | St. | Action |
| --- | --- | --- | --- | --- |
| L1 | Condition values are always the source's own statistic; the counterfactual response is never supervised | `conditions.py:52, 62-75` | C | S1 informative draws; S3 own samples; Q-J |
| L2 | Inside a span the committed counters reveal the value; CE on middle rows needs no value read (the end is informative again, P7) | `features.py:264-272, 296-302`; audit §2 | C | S0 onset / middle / end strata; S1 onset-aligned draws |
| L3 | Absolute tiled star is ¾ determined by the skeleton; no committed-difficulty quantity in the frame; value KL 1e-5 at 61.4M | A `star-prediction-summary.json`; recheck table; `features.py:296-302` | C | S0 proxies; S2 residual |
| L4 | "Natural behaviour" has no stated definition; the code implements "imitate the source on those rows" | `conditions.py:83-90`, `features.py:291` | C | Q1 (deferred; dependencies in §11) |
| L5 | Rows between two spans have no stated semantics | `features.py:291, 308-330` | C | Q2 (deferred); §3.7 |
| L6 | Spans drawn without knowing the window; LN active on 13.5 % of scored heads, 32 % of active rows with an unscored onset | `data.py:82`; audit #1 | C | S0 §5.1 (dropout first, alignment, IPW) |
| L7 | Interval count per song, not per window (LN-active 21 % for K < 600, 8 % for K ≥ 1500) | `conditions.py:55, 74`; audit #4 | C | S0 per-window selection; D2 |
| L8 | The fit_dev manifest inherits L6/L7: 14 of 64 windows with an active LN row | `data.py:94-119`; `fit_dev_manifest.json` | C | S0 two manifests (§5.3) |
| L9 | Window start rule; 25 % short windows; harmful only through 1/L | `data.py:74-81` | C | S0 keep the rule, fix the weighting |
| L10 | Star cells fixed per chart by a hash; an onset can never be placed relative to a window | `labels.py:56-75` | C | S0 relabel at 10 s offsets, three lengths and the whole song |
| L11 | Cells mostly longer than a window; the whole-cell value on every row | `labels.py:37`; audit #2 | C | S0 aligned draws score the onset side; lengths kept (§5.2) |
| L12 | The whole-song star (p 0.10) fed per row though set by the hardest section | `conditions.py:67-69` | C | S2 residual on the same span; chart-level scalar rejected by the human |
| L13 | Labels from the original `.osu`, not the cache representation (≤ 0.00016 star measured) | `labels.py:116-118` | C | S0 relabel from the cache |
| L14 | Tiled star needs ≥ 30 s: a constraint | `labels.py:35, 103-104` | C | kept |
| L15 | Presence bit set on every row when any interval exists; T-P3b(i) fails through it | `features.py:291`; DEVIATIONS 5 | C | S0 switch; default as now; Q-A |
| L16 | Token form reads the whole track at every row (by design; the Q2 instrument) | `features.py:308-330` | C | D until Q2 |
| L17 | Star frame carries LN counters and no difficulty statistic | `features.py:296-302` | C | S0 proxies (§3.8) |
| L18 | `Interval.value` scalar; no categorical kind | `features.py:256-261, 275-276` | C | S0 per-kind encoding; S4 |
| L19 | FiLM zero-initialised: natural mode is learned FiLM(0); not a defect | `model.py:62-64` | C | none |
| L20 | Loss is the window mean: EOS 4.2×, last 64 rows 1.8× | `train_ce.py:266, 269`; audit #5 | C | S0 uniform base CE, fixed divisor |
| L21 | No condition-scoped term; nothing weighted, tested or reported per kind | `train_ce.py:263-273` | C | S0 per-factor terms (§3) |
| L22 | DPO anchor reuses the window mean | `train_dpo.py` `window_ce` | C | S0 shared L_base |
| L23 | No term sees own histories; own-history LN forecast shift 0.129 ± 0.044 at 61.4M (was 0.217 at 30.7M, 0.012 at 39.5M) | recheck `average-table.md`; `train_ce.py:263-269` | C | S1 measure; S3 term (Q-I) |
| L24 | "Conditioned" NLL keyed on track presence (47 % of such decisions active) | `train_ce.py:145` | C | S0 per-factor, per-stratum NLL |
| L25 | In-run free-run natural only | `train_ce.py:317` | C | S0 §6.4 |
| L26 | Selection on fit_dev action CE alone, no SE, no guards | `checkpoint-selection.json` | C | S0 §6.6 |
| L27 | Free-run panel 4 charts × 3 seeds; noise floor ≈ 0.1 in LN share | `data.py:111-116`; `evals.jsonl` | C | S0 panels of §6.4 |
| L28 | **Corrected**: natural LN share on the 4-chart panel moves by up to 0.17 between adjacent checkpoints to the end, but safe pairs 52k-170k exposures apart move by 0.08-0.10, so late deltas are within panel noise; early swings (4-35M) resolved | `evals.jsonl` [data, my recount] | C | none; v1's Q3 withdrawn; the exposure-bias shift (L23) and the natural LN bias (+0.10 to +0.13, three panels) carry the "CE recipe first" ordering |
| L29 | fit_dev NLL minimum 1.975 at 61.4M, 2.061 at 121.7M; ≈ 9.6 average passes, more for charts in small groups (group-uniform draw) | `evals.jsonl`; `summary.json`; `data.py:69-71` | C; cause I | S1 ≤ 4 passes, cosine at the budget; Q-F |
| L30 | No conditioned generation entry point with receipts | `train_ce.py:552-564`; `sampling.py:28-29` | C | S0 CLI |
| L31 | No baseline b for a generated span's star | — | C | S0 §4.1 |
| L32 | Boundaries inside a window only by chance; `replace_interval` path unexercised except at natural onsets | `conditions.py:106-134` | C | S0 draws keep adjacent pieces; switches in the condition manifest |
| L33 | Free-run summary lacks following, calibration and the short-hold rate under requests | `report.py` | C | S0 §6.4 fields; guard (iv) |
| L34 | Guard trips on system-wide swap growth since the trainer's start; six trips 1.19-4.15 GB; the trainer's RSS limit never tripped | `runtime.py:31, 95, 114, 119`; `events.jsonl` | C; cause I | S0 §8.1; MT reads `resources.jsonl` |
| L35 | Every restart resets the swap baseline; lifetime restart budget 5 | `launch.py:27, 106-112` | C | S0 rate budget |
| L36 | **Corrected**: `train.jsonl` and `resources.jsonl` exceed the 4 MB staging limit and are re-staged every cycle; they are not pattern-excluded | `mutagen.yml:161, 179-182` | C | S0 segmented logs; MT pattern exclusion for the existing run |
| L37 | RSS spikes to 2.5-4.6 GiB at irregular points; inferred cause per-candidate float64 arrays | mirrored `resources.jsonl`; `model.py:265-278` | C; cause I | S0 pilot; float32 and chunking if confirmed |
| L38 | Throughput 2,177/s wall with contention; 3,700-3,957 in the pilot | `run.json` | C | none (planning number 2,500) |
| L39 | Landmark readout: lesion −0.0015 ± 0.0003 at 39.5M, none (−9e-5 ± 3e-4) at 61.4M; redundancy with the 511-token field is the inferred cause | recheck table; audit §3 | C; cause I | D; Q-L |
| L40 | DPO: synthetic labellers off the path; 64-decision horizon shorter than a 30 s span | `train_dpo.py` | C | D (waits for real pairs) |
| L41 | Release factors read frames at the LN's birth time and at candidate times across span boundaries; v1 §2.1 wrong | `model.py:274-276` | C | S0 ownership by birth; tests §3.4 |
| L42 | No LN span above 64 beats; whole-song and half-song requests are extrapolation | `conditions.py:20, 26-31, 34-56` | C | S0 span lengths; Q-B; primary changed |
| L43 | v1's draw applied dropout after alignment: natural windows selected on their own targets; design rule requires inverse-probability weights for any balancing | plan v1 §4; `design.md:303` | C | S0 §5.1 steps 3-6 |
| L44 | v1 refitted b on 30 s cells with `duration` constant, making whole-song b an extrapolation | A's feature list | C | S0 §4.1 |
| L45 | The star statistic owns tails decided after b; F3 must sample until the last owned LN is released | `labels.py:39, 78-98` | C | §4.3, §6.4 |
| L46 | Manifest and free-run panel depart from `design.md:535` (128 groups, 24 per band) without a DEVIATIONS entry; free-run charts K ≤ 600 | `design.md:535`; `data.py:95, 111` | C | S0 DEVIATIONS entry with the power cost; panels of §6.4 |
| L47 | Safe checkpoints at irregular exposures enter `evals.jsonl` | `train_ce.py:411, 433`; `events.jsonl` | C | S0 excluded from selection; tagged in logs |
| L48 | Checkpoint-level fit_dev noise ≈ 0.02-0.03 not in the bootstrap SE | `evals.jsonl` 109.7-111.1M | C | S0 mean of the last three checkpoints |
| L49 | `validate_track` rejects star < 0; star normalised by /4 | `conditions.py:100`; `features.py:276` | C | S0 |
| L50 | Manifest reuse keyed on `star_conditions` only | `data.py:121-129` | C | S0 version hash |
| L51 | A plateau stop can break a matched-exposure comparison | plan v1 §8.4 | C | off in comparisons |
| L52 | Stage costs omitted in-run evaluation | recheck `execution.json` | C | §6.7 |
| L53 | ∣Δ∣ > 0.1 is within binomial noise for 8-beat pieces (≈ 40 heads, SE ≈ 0.06) | audit rows per beat | I | S0 z-criterion with n ≥ 20 |
| L54 | Evidence update: at 61.4M the whole-song gate is met (0.715 ± 0.045), switches are not (0.272 ± 0.034), natural LN is biased +0.10 to +0.13, star has no effect | recheck tables | C | folded into L3, L23, L28, L42 and the stage-1 primary |

---

## 11. Decisions for the human

Only questions whose answer changes what is built or run. Options as they would be presented; what
each changes. Order: by what they block.

**Blocking stage 0 (code and relabel)**

- **Q-C Star cell lengths.** (a) 30, 60, 120 s at 10 s offsets plus the whole song (default; ≈ 36
  min relabel); (b) add 45 and 90 s (≈ 55 min); (c) 30 s and the whole song only (v1). Changes the
  relabel, the baseline refit, the star frame's trained range and the evaluation spans.
- **Q-D Guard policy.** Swap trip: remove / key on the trainer's own RSS growth / keep at 4 GiB; the
  restart rate (default 5 per 6 h). Changes `runtime` and `launch` in stage 0.

**Blocking stage 1 (runs)**

- **Q-B LN span lengths in training.** (a) pieces plus runs of 2-8 pieces at p 0.25 plus the whole
  song at p 0.10, in both arms (default); (b) pieces only (as now); (c) other probabilities. Changes
  whether whole-song and half-song requests are in-distribution, and D7; (b) keeps the whole-song
  panel an extrapolation measure.
- **Q-F Learning rate before stage 1.** Keep lr 1e-3 cosine over 50M / a 2 × 30-minute re-tune pilot
  (lr and weight decay) first. Changes the stage-1 schedule and adds ≈ 1 h.
- **Q-E Budget and arms.** A0/A1 at two seeds (≈ 29 h) / add the emphasis arm A2 with λ_LN = 1
  (+14 h). Changes what stage 1 can say about the per-kind term.

**Blocking stage 2**

- **Q-G Direction 2.** (A) the residual is the condition value, learned by CE on real rows; the
  realised response is measured (stage 2 = B0 / B1 / optional TF-mono B2); (B) in addition a term on
  generated spans trains the realised difficulty towards the request (B2 = relaxed-proxy term,
  +45 % step cost; true F3 as calibration; Q-I then applies to difficulty too).
- **Q-H Whole-song star interval and request range.** Keep at p 0.10 as a residual / lower / drop;
  residual requests ±0.5 star (default, one SD) or wider. Changes the stage-2 arms and panel.

**Can wait**

- **Q-A Presence bit.** Keep as now (default while Q2 is open; T-P3b(i) carries a documented
  exception) / remove (`none`) / announce the next span (an explicit tested channel). Changes the
  input-locality test and overlaps Q2; both stage-1 arms use the same setting.
- **Q-I Own-sample terms before real preference pairs.** Yes / no / only if stage 1 fails the
  free-run gate while passing the teacher-forced one. Changes whether stage 3 exists and, under
  Q-G(B), stage 2's B2.
- **Q-M Reading of direction 1 for the emphasis terms.** Governed factor (default) / whole decision
  on span rows. Changes the stage-3 terms only; both numbers are logged from stage 0.
- **Q-J Constructed alternatives** (accepted re-arrangements of one skeleton, which need a judge):
  now / after the evaluator is calibrated / never. An extra data route for L1.
- **Q-K Style vocabulary** (the three ordinal levels, "unreviewed" = absent frame): confirm / amend.
  Stage 4's fixture and the frame width.
- **Q-L Landmark run**: now / after stage 2 / drop the readout.

**Deferred by the human (listed, no default chosen)**

- **Q1 Natural behaviour.** (a) imitate the source on unconditioned rows, as now; (b) an implicit
  corpus-typical value the model should realise; (c) a value drawn from a corpus prior at inference,
  with unconditioned training rows scored as (a) or not at all. Depends on it: the dropout rates and
  the per-window interval count (D4), the inverse-probability weight's target distribution, guard
  (i), what the natural panel is compared with, whether windows without any condition are scored.
- **Q2 Between two conditioned parts.** (a) natural per Q1; (b) hold the last value (track
  transform); (c) a transition the model shapes with an announcement channel or the token form;
  (d) a rule given by the interface (ramp). Depends on it: Q-A, the §3.7 transform, membership of
  pre-span rows in Ω_κ under (c), whether token-versus-FiLM is meaningful, which between-span rows
  enter the base CE as natural.

Withdrawn from v1: Q3 (oscillation branch; the panel cannot resolve it and the exposure-bias shift
already orders "CE recipe first"); Q10 and Q11 are Q-D and Q-H; Q6 is Q-A; Q4 is Q-I; Q5 is Q-J;
Q7 is Q-F; Q8 is Q-K; Q9 is Q-L.

---

## 12. Change log from v1, by review point

| Point | What changed |
| --- | --- |
| P1 | Stage-1 primary: whole-song slope → in-distribution onset-span slope; whole-song slope reported as generalisation. Long LN spans (runs, whole song) and 60/120 s star cells enter training in both arms (Q-B, Q-C). Threshold restated relative to A0 and per seed. |
| P2 | Draw reordered: dropout on candidates first, alignment among survivors, informativeness against rows the window never scores, inverse-probability weights on natural decisions, a draw-independent natural manifest for the guard and the selection primary. |
| P3 | Loss recomposed: uniform per-decision base CE (the L20 fix alone) plus additive governed-factor emphasis terms with λ = 0 in stage 1; the LN term is the exact LN-versus-tap split; releases leave L_LN; no double counting across kinds. Null contrast as a diagnostic; the monotone hinge as a stage-3 candidate. |
| P4 | Direction 2 put to the human as Q-G with four terms and honest costs (true F3 ≈ 2.7×, not +30-50 %); baseline refit on all interface lengths; within-chart residual correlation measured with a stated rule; negative-star validation and scale fixed in stage 0. |
| P5 | Arms differ only in the draw, both on stage-0 code; plateau stop off; read at 50M on the mean of the last three checkpoints; onset-row NLL as a paired arm difference; effect required in each seed; evaluation every second checkpoint; costs ≈ 7.2 h per run; onset memorisation logged. |
| P6 | Ownership by birth for every release factor; v1 §2.1 withdrawn; locality tests per factor with the precise exception; F3 sampling until the last owned LN is released. |
| P7 | D3 recomputed and thresholded before the simulation; z-criterion with a minimum head count; onset / middle / end strata; skeleton share predictor as a stage-0 diagnostic. |
| P8 | Presence bit kept behind a switch (Q-A); uniform weights by default; membership under Q2(c) listed; cell lengths Q-C; v1's Q3 withdrawn. |
| P9 | L28 corrected to "unresolved on this panel" with my recount; "CE recipe first" rests on the own-history shift and the natural LN bias. |
| P10 | L36 corrected (staging limit, main-thread sync patch); manifest version hash; one (length, phase) star partition per draw; two manifests instead of D5; DEVIATIONS entry for the panel. |
| W1-W16 | Ledger rows L41-L54 and corrections to L23, L28, L29, L36. |

---

## 13. What I could not check

- Mac-only files: the full `train.jsonl` and `resources.jsonl` (training-loss side of L29; RSS at
  the swap trips); A's `star-predictions.parquet` (within-chart residual correlation); the cache
  `index.parquet` (per-chart pass counts under group-uniform sampling).
- Every [inferred] number of §5.3, §6.7 and §4.3 until `draw_sim.py` and the stage-0 pilot run; no
  mac job was launched.
- The review's P9 SE method; my recount is in §1.2 and the conclusion does not depend on it.
- Acceptability of any generated chart; no renders were viewed.
- Whether the Lens five-dimension vocabulary (0006) is the one R2 should take (Q-K).
