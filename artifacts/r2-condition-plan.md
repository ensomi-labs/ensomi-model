# R2 condition-scoped losses and recipe repair: plan (proposed, under review)

Shareable. Written 2026-10-05 by a fresh Fable subagent for [r2-condition-plan](r2-average-and-control.md#r2-condition-plan), serving the human decisions [d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss) and [d-natural-gap-deferred](r2-average-and-control.md#d-natural-gap-deferred). Kept as written; nothing in it is adopted until the adversarial review and the human have seen it. Bracketed anchors such as `[d-r2-conditions]` are in [r2-implementation](r2-implementation.md), [r2-average-and-control](r2-average-and-control.md) and [r2-ln-design](r2-ln-design.md). Main thread checked: `[d-r2-conditions]` does record the rejection of a chart-level scalar fed to every row.


Planner: Fable subagent, control plane, read-only. Code read at `r2/train` `7d9640a` (identical to the
overnight run's frozen copy for every file cited; the audit checked this at `78aa22c` and `7d9640a`
touches only `train_dpo.py` and tests). Nothing in the repository was edited; no job was launched.

Evidence tags used throughout: **[code]** checked in the source at the cited line, **[measured]** a
number read from a run log, an analysis data file or the audit's measurements (job id or file
named), **[inferred]** reasoning from those. The brief's rules bind: nothing below draws an
architectural conclusion from the 77 % run; every claim in a stage names the test or run that
fails if it is false; a difference under two standard errors is "none".

**Mac measurements.** None were run. The mac was unreachable (sshd dropping sessions) when this
plan started; when it came back, the one simulation I prepared,
`~/ensomi/.sync/cp/scratch/r2-cond-plan/draw_sim.py` (current draw against the proposed
span-aligned draw, 3,000 draws each, NumPy only, about five minutes), was refused by the control
plane's auto-mode classifier because the recheck Astra job `20261005-044020-r2-recheck` is
running on the mac. The script is syntax-checked and mirrored; it is stage 0's first job (section
9). Every number this plan needs that the script would have produced is marked **[to measure]**
with the expectation stated as inferred. Numbers that were available from mirrored files are used
instead where they answer the question (A's star-prediction summary for the residual width, the
label build receipt for the relabelling cost, `evals.jsonl` for the in-run free-run trajectory).

Sources read: the seven notes files named in the brief; `src/ensomi_model/r2/*` and `tests/r2/*`;
`artifacts/r2-ml-design-20261003/design.md` sections 3, 4, 8; the three analysis reports and
A's `star-prediction-summary.json`, `extensions-summary.json`; the overnight run's `config.json`,
`events.jsonl`, `evals.jsonl`, `run.json`, `launch.json`, `trainer.log`, `fit_dev_manifest.json`,
and the mirrored prefixes of `train.jsonl` (to 11.0M exposures) and `resources.jsonl` (to 33.2M);
the recheck job's `checkpoint-selection.json`; `research/oracle_time_continuation/runtime.py`
(resource guard); Beatmap Lens decisions 0001, 0004, 0005, 0006; `~/ensomi/mutagen.yml`.

---

## 1. Ledger of wrong pieces

Status: **C** confirmed in code or data (cited), **I** inferred. Action: **S0..S4** the stage that
fixes it (section 9), **D** deferred with reason, **H** for the human (section 10).

### 1.1 Problem definition

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L1 | Condition values are always the source chart's own statistic over the span, so every skeleton carries exactly one (span, value, target) triple; the counterfactual response (same skeleton, other value) is never supervised. | `conditions.py:52` (LN share of the source heads in [a,b)), `:62-75` (cached star of the source); A §4.1 | C | Control is learned only by generalisation across charts; a request that contradicts the preceding content is never seen in training (audit: 0.18 % of scored heads carry such information). | S1 (informative onset draws), S3 (own-sample statistic term), H-Q5 (constructed alternatives) |
| L2 | Inside a span, the teacher-forced history plus the committed counters reveal the value; CE on later rows of the span is satisfiable without reading the value. | `features.py:264-272` (counts of committed heads and LNs in the interval before k), `:296-302` (ratio and remaining in the frame); audit §2: median 63 rows after onset, 15 % within 16 rows | C | The gradient that teaches the value sits on the first rows after an onset, which are seldom scored (L6). | S0 per-kind onset metrics, S1 onset-aligned draws, S3 |
| L3 | Absolute tiled star is about three quarters determined by the given head rows (R² 0.746 ridge, 0.753 32-NN on fit_dev intervals, RMSE 0.52 star), and the frame has no committed-difficulty quantity. | A `star-prediction-summary.json`; judgment §2, §5(2); `features.py:296-302` | C (measured by A) | A star request carries little information the skeleton does not; the model shows value KL 2e-6 nats/row at every checkpoint (A). | S2 residual-to-baseline (human decision 2) plus kind-specific committed statistics (L17) |
| L4 | "Natural behaviour" (no condition of a kind) has no stated definition; the code implements "imitate the source on those rows", split by the presence bit (L15). | `conditions.py:83-90` (dropout), `features.py:291` | C | Guards on natural free-runs, the natural-row loss weight and the inference default all presuppose an answer. | H-Q1 (deferred by the human); dependencies listed in section 10 |
| L5 | Rows between two conditioned spans have no stated semantics; FiLM gives them the presence bit and nothing else, the token form shows them both neighbours. | `features.py:291, 308-330` | C | Whether the model should anticipate an upcoming span, hold the last value or revert is undecided; it fixes the visibility rule and the token-vs-FiLM comparison. | H-Q2; the plan keeps span membership a property of the track so any answer is a track transform (section 2.5) |

### 1.2 Data and draws

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L6 | Condition intervals are drawn over the whole song without knowing the scored window. | `data.py:82`, `:108` (`draw_track(chart, labels, rng)` never sees `start`/`stop`); audit #1: LN active on 13.5 % of scored heads, star 23.9 %; 49 % of LN-track windows have no active row; 32 % of active LN rows belong to an interval whose onset is unscored history | C | The condition is supervised on one scored row in seven, and almost never at an onset. | S0 span-aligned draw (section 4) |
| L7 | Interval count per song is 1-4 (LN) and 1-3 (star) whatever K is. | `conditions.py:55`, `:74`; audit #4: LN-active 21 % for K < 600, 8 % for K ≥ 1500 | C | Conditioning is learned mostly on short songs. | S0 (per-window draw makes the count independent of K; test D2) |
| L8 | The fit_dev manifest draws the same way, so it inherits L6/L7: 42 of 64 windows conditioned, 34 with an LN track but only 14 with an active LN row (1,691 rows, 11.1 % of 15,188 heads), 22 with active star. | `data.py:94-119`; the run's `fit_dev_manifest.json`; audit §3 | C | The per-checkpoint "conditioned NLL" rests on few, clustered rows and cannot show a conditioning effect. | S0 rebuild with the shared draw function, stratified so each kind has ≥ 24 windows with a scored onset |
| L9 | Window start: BOS with p 0.125, last 256 rows with p 0.125, else uniform in [0, K]; stop = min(start+256, K+1); 25 % of windows are shorter than 256 and 0.17 % are EOS-only. | `data.py:74-81`; audit §1 | C | Harmless in itself; it becomes a skew only through the 1/L weighting (L20). An EOS-only window encodes the whole prefix for one decision (cost only). | S0 keep the rule, fix the weighting |

### 1.3 Condition labels

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L10 | Star cells are fixed per chart by a hash of the inputs: the same chart always offers the same cells. | `labels.py:56-75` (`j0` from SHA-256 at `:65-66`) | C | A cell onset can never be placed relative to a window; the label set is 82,773 labels for 12,531 charts. | S0 relabel: 30 s cells at 10 s offsets (cost: 472 s for 82,773 labels with 4 workers [measured, `labels/star_summary.json`] → ~23 ms per label per worker → about 15 min for ~150k labels [inferred from median song length ~150 s]) |
| L11 | Star cells (30 s, 60 s, whole song) are mostly longer than a 256-row window (median 278 rows) and the whole-cell value is applied to every row; half of the overlapping cells have less than half their rows scored. | `labels.py:37`, `conditions.py:59-75`; audit #2 | C | The value on a scored row is a statistic of rows the model mostly cannot see. | S0: interval draws use 30 s cells only (≈190 rows at the median 6.4 rows/s; dense charts overflow and score what fits with the onset first); 60 s cells leave the training draw; the whole-song interval stays (L12) |
| L12 | The whole-song star value (p 0.10) is fed per row although it is set by the hardest section (local cells sit a median 0.09, mean 0.21 below it). | `conditions.py:67-69`; audit §2 | C | A systematic over-statement of local difficulty on 19 % of star-active rows. | S2: the whole-song value becomes a residual against the baseline of the same span, which is coherent per row (it says "this song sits above/below what its skeleton implies"); the audit's "own global channel" is **rejected** because a chart-level scalar fed to every row was rejected by the human on 2026-10-03 ([d-r2-conditions]) |
| L13 | Labels are computed from the original `.osu` objects, not the cache representation the model emits (2 ms row merge, snapped releases). | `labels.py:116-118`; A measured ≤ 0.00016 star difference on 16 charts | C (effect measured small) | Negligible label noise; a consistency matter since generated spans are scored on the cache representation. | S0 relabel from the cache decisions (`cache.py` replay to objects), since relabelling anyway |
| L14 | Tiled star needs ≥ 30 s; shorter requests are extrapolation. | `labels.py:35`, `:103-104`; [d-star-tiled] | C | A constraint, not a defect: star spans are ≥ 30 s; the residual baseline is defined on the same spans. | kept |

### 1.4 Condition visibility in the model

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L15 | The per-kind presence bit is set on every row when any interval of that kind exists anywhere in the track, active or not (FiLM form). | `features.py:291`; DEVIATIONS.md item 5 | C | Rows outside S_k receive information about k; "no track" and "track elsewhere" are different inputs. A measured presence/bounds/counter effects larger than value effects (natural-to-source-star KL 1.2e-4 vs value KL 2e-6, A §2). It also makes the value-locality test of section 2.2 fail on the current code. | S0: presence := active (the bit is redundant with the active channel and is removed), unless H-Q2/Q6 asks for announced upcoming spans, in which case announcement becomes an explicit, tested channel |
| L16 | The token form reads the whole announced track (past and future) at every row. | `features.py:308-330`; `tests/r2/test_conditions.py:25-37` | C | By design (look-ahead to upcoming spans); it is exactly the H-Q2 instrument, so the FiLM-vs-token comparison cannot be judged before Q2 is settled. | D until Q2 (advice ledger: token vs FiLM) |
| L17 | The star frame carries the LN counters (heads, LNs, ratio, remaining) and no committed-difficulty quantity. | `features.py:296-302` (same five channels for both kinds) | C | Star has no quota-controller path, unlike LN; the judgment's reading of why LN follows and star does not. | S2: kind-specific committed statistics (section 2.6) |
| L18 | `Interval.value` is a scalar and the frame has one value channel; a categorical or multi-label kind (style concepts) has no representation. | `features.py:256-261`, `:275-276` | C | The interface cannot take the Lens section targets. | S0 generalise the value encoding per kind (section 2.6); engineering, no run |
| L19 | FiLM's last layer is zero-initialised, so natural mode is the learned FiLM(0). | `model.py:62-64`, `:285-288` | C | Not a defect; noted because "natural" has a learned bias. | none |

### 1.5 Loss

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L20 | The loss is the window mean (1/L per decision) averaged over windows. | `train_ce.py:266`, `:269`; audit #5: EOS 4.2× and the last 64 rows 1.8× their share | C | The song end and EOS are over-weighted; a row's weight depends on how many rows lie outside any span in its window, which is itself a locality violation (section 2.2, P3). | S0 per-decision weighting with a fixed divisor (section 5) |
| L21 | There is no condition-scoped term: one CE over all scored rows, condition rows and natural rows indistinguishable in the objective and in the logs. | `train_ce.py:263-273`, `Stats.add` | C | Nothing can be weighted, tested or reported per kind. | S0 masked per-kind terms (section 2.4) |
| L22 | The DPO anchor CE reuses the window mean. | `train_dpo.py:190-191` | C | Inherits L20 when DPO runs. | S0 share one loss function |
| L23 | No term sees the model's own histories; teacher-forced LN forecasts are calibrated (0.191 vs 0.190) while forecasts on own histories sit at 0.409 at 30.72M (B), i.e. exposure bias with positive feedback. | `train_ce.py:263-269`; B §"ranked evidence" 1; judgment §3 | C (one checkpoint) | Free-run condition following is never trained; natural LN runs away and oscillates (L28). | S0 diagnostic; S3 own-history statistic term (human gate, H-Q4) |

### 1.6 Diagnostics and selection

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L24 | "Conditioned" vs "natural" NLL is keyed on track presence, not on active rows (47 % of "conditioned" decisions are active). | `train_ce.py:145`; audit §3 | C | The logged split cannot show a conditioning effect. | S0 per-kind, per-active-row NLL with onset and late sub-splits |
| L25 | The in-run free-run is natural mode only; no conditioned free-run, no counterfactual probe, no own-history calibration. | `train_ce.py:317` (`continue_chart` without a track) | C | Condition following was first measured by A on 2026-10-04, a day after the run started. | S0 (section 6) |
| L26 | Checkpoint selection is on fit_dev action CE alone (the recheck job picked ckpt-0061432779 as the minimum, tie tolerance 0.001); the notes' "earliest within two SE" rule had no SE and no guards. | `artifacts/r2-recheck-20261005/checkpoint-selection.json`; [r2-overnight-run] | C | Selection ignores release CE, following, calibration and legality. | S0 rule in section 6.5 |
| L27 | The free-run panel is 4 short fit_dev charts × 3 seeds; the chart-mean SE of natural LN share per checkpoint is 0.015-0.106. | `data.py:111-116`; `evals.jsonl` [measured] | C | The in-run report cannot resolve a 0.1 change. | S0 panel of 16 charts × 3 seeds × {natural, conditioned}; cost section 6.6 |
| L28 | Natural LN share oscillates between adjacent checkpoints to the end of the run: chart-mean deltas larger than 0.1 in 8 of the 31 steps after 4.4M, including 0.216→0.381→0.288 (96.5M→99.8M→100.9M), 0.361→0.188→0.285 (109.7M→111.0M→111.1M) and 0.343→0.221→0.141→0.222 (114.1M→121.7M); source panel mean 0.281. | `evals.jsonl` [measured, 4 charts × 3 seeds; several deltas are within two chart SEs and are "none" individually] | C (weak) | Under item 3 of [q-r2-after-judgment], this is the "CE recipe is the problem" branch, on weak evidence; the one-checkpoint recheck cannot measure oscillation. | S0 diagnostic with a larger panel; H-Q3 |
| L29 | fit_dev action NLL reaches its minimum 1.9753 at 61.4M and rises to 2.061 at 121.7M (natural rows 1.977 → 2.047; release NLL keeps falling 0.073 → 0.072). The run made ≈ 9.6 passes over fit_train (121.7M / 12.7M head rows). The training loss after 11M exposures is not visible from here (L36). | `evals.jsonl`; `artifacts/r2-cache/v1/summary.json` (13.97M rows, 11,368 of 12,531 charts in fit_train) [measured] | C; cause I | A generalisation gap opens at about five passes; the last checkpoints are worse on fit_dev than the 61M one. The LN oscillation coincides with this regime. The cause (memorisation at lr 1e-3, data size, or both) is not identified. | S1 budgets of ≤ 4 passes with fit_dev-based stop; H-Q7 (re-tune lr/decay or add regularisation) |

### 1.7 Generation and evaluation

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L30 | No conditioned generation entry point with receipts; A wrote its own harness. `continue_chart` accepts a track. | `train_ce.py:552-564` (`--freerun-only`, natural), `sampling.py:28-29` | C | Condition following cannot be reproduced from the trainer. | S0 CLI (section 6) |
| L31 | Generated-span star is evaluated with `labels.tiled_star` on exported objects (A did this), but there is no baseline b to compare a response against. | A §1; this plan §3 | C | Needed for the residual condition. | S2 |
| L32 | Interval boundaries inside a window occur only by chance; a mid-chart request change (`replace_interval`) is a runtime path the training distribution does not exercise except at natural onsets. | `conditions.py:106-134`; design review deferral | C | With span-aligned draws every aligned window scores an onset and, when the window runs past b, an end; the "switch" case (value change at a boundary) occurs when two intervals of one kind are adjacent. | S0 draws keep adjacent pieces (consecutive with p 0.5) so switches are scored |
| L33 | The free-run summary has LN share and drift but no following metrics, no calibration, and the short-hold rate is not tracked under requests (holds ≤ 60 ms: 0.0 % in the source panel; 1-3 % under guidance or count feedback in A). | `report.py`; A §5 | C | Following gains could be bought with fake holds unnoticed. | S0 report fields and guard (iv) in section 6.5 |

### 1.8 Infrastructure

| # | Defect | Evidence | St. | Consequence | Action |
| --- | --- | --- | --- | --- | --- |
| L34 | The resource guard stops the trainer when **system-wide** swap grows by more than 1 GiB relative to the swap at the trainer's own start; the trainer's RSS limit (12 GiB) never tripped. | `runtime.py` `snapshot()` (`psutil.swap_memory().used`), `ResourceGuard.__init__` (`initial_swap`), `ResourceConfig.max_swap_growth_bytes = GiB`; `train_ce.py:378-380` passes no override; `events.jsonl`: six `resource_limit` events with growth 1.41, 2.69, 1.19, 1.43, 1.98, 4.15 GB between 09:01 and 11:35 UTC on 2026-10-04 [measured] | C | The run was stopped by memory pressure the trainer did not cause (I: the last trip is a 4.1 GB swap growth within 95 s of a resume, while the trainer's typical RSS was 1.1-1.4 GiB with 12-14 GiB available in the mirrored first 33M exposures; the 24 GiB machine (`hw-probe` job) had 4.2-4.5 GiB of swap in use before the run started). What only the mac's `resources.jsonl` can answer: the trainer's RSS, `available_bytes` and `pressure_level` at each trip, hence whether anything of the trainer's grew. | S0 guard on process RSS and `available_bytes` (section 8) |
| L35 | Every restart resets the swap baseline, and the supervisor's restart budget is a lifetime count of five. | `launch.py:24`, `:95-105` | C | Trips cluster (two 167 s apart at 110.97M-111.14M, three within five minutes at 121.65M-121.73M) and the run dies at 77 % with lr not at its floor. | S0 restart budget as a rate (section 8) |
| L36 | Text files above 4 MB are excluded from the mirror by pattern; `train.jsonl` (to 11.0M exposures, 7 %) and `resources.jsonl` (to 33.2M, 27 %) are truncated on the control plane. | `mutagen.yml:161,179-181`; file sizes 3,999,844 and 3,986,471 bytes [measured] | C | The control plane cannot read most of the training curve; L29's training-loss side is unverifiable here. | S0 per-checkpoint log segments (section 8) |
| L37 | RSS spikes to 2.5-4.6 GiB at irregular log points, not at checkpoint boundaries (23 of 16,609 mirrored snapshots; e.g. 4.57 GiB at 9.33M exposures). | mirrored `resources.jsonl` [measured] | C; cause I | Inferred cause: per-candidate float64 NumPy arrays built per release factor (frames `[C,3,2,16]`, relations `[C,28]`, candidate features `[C,34]`, `model.py:265-278`, `features.py:350-366`) on windows with many gap releases over long gaps. Not the cause of L34 by itself, but it raises the footprint the guard must tolerate. | S0 measure (peak RSS per window against factor × candidate counts); float32 and per-factor chunking if confirmed |
| L38 | Throughput over the run was 2,177 decisions/s wall against 3,700-3,957 in the pilot, with DPO tests and three Astra jobs sharing the CPU. | `run.json`; [s-r2-tune] [measured] | C | Planning number for this plan: 2,000-4,000 decisions/s, as the brief says. | none |
| L39 | Landmark memory: zeroing the readout improves action NLL by 0.0015 ± 0.0003 at 39.49M and adds nothing to prefix persistence (B); the audit refuted range misplacement and reads the cause as redundancy with the 511-token TCN field (54 % of scored rows have k ≤ 511). | B §"frozen real-history probes"; audit §3 | C (one checkpoint); cause I | Not a condition defect; a capacity path with no pressure to be used. | D: the test is specified (NLL with and without the readout on rows with k > 511; a 6-level TCN arm so that the landmarks carry the long range); H-Q9 whether to spend a run |
| L40 | DPO: synthetic labellers are off the real path (`7d9640a`); the default 64-decision pair horizon is shorter than the 30 s a tiled-star label needs (A §"Q5 and Q6"). | README; `train_dpo.py` | C | Nothing to do until real pairs exist; the horizon must be ≥ the longest conditioned span when star pairs are built. | D (DPO waits, human decision 2026-10-05) |

---

## 2. Condition-scoped loss, general form

### 2.1 Objects

- A chart is a head-row skeleton with K rows and song length T; decision D_k at row k (k = K is
  EOS) has state s_k = (committed history, held-lane state, look-ahead over the skeleton,
  conditions at row k).
- A condition kind κ has a track of intervals (a, b, v) with half-open time support [a, b);
  intervals of one kind do not overlap; kinds may overlap each other. The **span** S_κ,i of an
  interval is the set of rows with t_k ∈ [a, b) (EOS is never in a span, `features.py:293`).
  Row membership m_κ[k] = 1 if k is in some span of kind κ.
- The **value** v is one of: a statistic of the chart over the span (LN share: LN heads / heads of
  the rows in S; the human's framing "a statistic of the chart over S"); a residual against a
  skeleton baseline (difficulty, section 3); or a label of the span (a style concept's salience,
  section 2.6).
- Structural fact that the design relies on **[code]**: in the FiLM form, conditions enter the
  forward pass only at the queried row (`model.py:171-175` applies `condition()` after the fused
  history and landmark read; `model.py:283-291` applies it to the release query) and history
  tokens carry no condition features (`features.py:182-199`). Under teacher forcing the value at
  row j therefore influences p(D_j | s_j) and nothing else; the only cross-row condition
  information is the presence bit (L15) and, in the token form, the whole track (L16). In
  free-run, conditions at earlier rows reach later rows through the committed history, which is
  the legitimate causal path.

### 2.2 The property "term κ does not measure what is not conditioned on κ", as tests

Let L_κ(θ; D, C) be the loss term attributed to kind κ on a batch, and let Ω_κ be the set of
scored rows under κ in that batch. Three testable parts:

- **P1 (target locality).** L_κ depends on the targets D_j only for j ∈ Ω_κ. Test T-P1: replace
  every target at rows j ∉ Ω_κ **that lie after the last row of Ω_κ in their window** by another
  legal decision (so no state inside Ω_κ changes, by causality) and assert L_κ is bit-identical.
  For rows before a span the targets are history and legitimately shape s_j; they are covered by
  P2.
- **P2 (gradient locality).** ∂L_κ/∂(logits_j) = 0 for every row j ∉ Ω_κ, for both the action
  logits and every release-factor score owned by j. Test T-P2: register hooks on the per-row
  action log-probabilities and on the pointer scores, backpropagate L_κ alone, assert exact zeros
  outside Ω_κ and non-zeros inside.
- **P3 (normalisation and input locality).** The weight of a row in L_κ does not depend on rows
  outside Ω_κ (so 1/L with L the window length is excluded), and L_κ is invariant to changes of
  the track outside the span: changing the value, bounds or existence of (i) an interval of kind
  κ that does not intersect the window and (ii) any interval of another kind at rows outside Ω_κ
  leaves L_κ bit-identical. Test T-P3a: add a natural-only window to the batch, assert L_κ
  unchanged. Test T-P3b: the two track edits above, assert L_κ unchanged. **On the current code
  T-P3b(i) fails through the presence bit (L15) and T-P3a fails through 1/L (L20); in the token
  form T-P3b(ii) fails by design (L16).** These are binding tests, not restatements.

Overlapping kinds: a row under κ and κ' belongs to Ω_κ and Ω_κ'; both terms may use it (it is
conditioned on both). Attribution of a *change* in behaviour to one kind is then a matter of the
counterfactual diagnostics (section 6), not of the loss.

### 2.3 Candidate forms

For a span S with value v, with the TCN history encoding shared across every variant of the
conditioning (section 2.1), costs are relative to one CE step of four 256-row windows (about 0.4 s
at 2,500 decisions/s).

| Form | What it measures | What it can leak or reward wrongly | Composition over kinds | Cost per update |
| --- | --- | --- | --- | --- |
| **F1** CE on rows of S, span-aligned draws: L_κ = mean over j ∈ Ω_κ of −log p(D_j \| s_j, v) | The likelihood of the source's own decisions under the requested value; at onset rows the value is the only information about the share, so the per-row Bernoulli-like gradient pushes p(LN head) toward v; on later rows the counters make v redundant and the term trains the quota controller (tracking given counts). | Late rows: v redundant (L2). The informative rows are the first ≈ 16 after an onset whose value differs from the preceding content; their share of scored rows is the quantity to maximise (section 4). | Clean: one masked average per kind; a row under two kinds enters both. | none beyond CE |
| **F2** Counterfactual contrast on the same rows: −[log p(D_j \| s_j, v) − log p(D_j \| s_j, v')] with v' a swapped value or null, bounded as an InfoNCE over a candidate set V ∋ v | Whether the observed decisions are more likely under the true value than under others: a direct pressure to read v. | The model can satisfy it by learning that v' is inconsistent with the history and pushing mass anywhere away from the observed decisions under v', which is not "follow v'". The same history leak as F1. Pathological behaviour at inference under requests that contradict history is possible. | Per kind, other kinds fixed; clean. | +1 conditioner, joint head and pointer pass per counterfactual (TCN shared): I 20-40 % per value [to measure in the S0 pilot] |
| **F3** Span-level statistic on the model's own samples: sample the span from the real prefix with the requested value, then (relaxed) score the per-row expected statistic on the **own** history against v, or (score-function) reward −(ŝ(S) − v)² with a baseline | The free-run response: realised statistic against the request, on the state distribution the model produces. The relaxed LN form is a differentiable quota-controller objective on own histories; it addresses B's 0.191-vs-0.409 calibration gap directly. For star, ŝ is non-differentiable (tiled star) so the score-function form is needed, or a differentiable proxy (chord density, repeat rate) trained to track it. | Rewards the statistic only: it can be satisfied with defective material (short holds, fake LN) unless the CE anchor and the hold-length guard hold. Score-function gradients are noisy; spans are short, so variance is manageable [I]. No preference labels are used, so this is not DPO; whether it may be used before real pairs is **H-Q4**. | Statistic of kind κ computed over S_κ on the same sample; gradient flows through rows in S_κ only (P2 holds by construction). | one free-run of the span (median 77 rows LN, ≈190 rows star) at I 250-350 rows/s on 4 threads plus one scoring pass: about doubles a step if done on every window; on one window in four, +30-50 % [to measure] |
| **F4** Auxiliary head predicting v from the hand vector on rows of S | Whether the representation carries v. | Trivially solved on late rows by the counters; a feature the decoder never uses (C's caveat). | Per kind. | negligible |

**Decision in this plan.** F1 with span-aligned, informative draws is the base (stage 1). F3
relaxed for LN and score-function for the star residual is stage 3, behind H-Q4. F2 is not
adopted: its failure mode is the one that produces strange output under contradicting requests,
which is the case steering tests make; it may be revisited only if stage 1 resolves nothing on
onset rows. F4 is a probe, not a loss (section 6.3).

### 2.4 The chosen composition

Per scored decision j in a batch, the per-row loss ℓ_j = −log p(D_j | s_j, C) (action plus
release, as now). The terms:

- L_0 = (1/N̄_0) Σ_j [j under no kind] ℓ_j (the natural-row term; its meaning depends on H-Q1),
- L_κ = (1/N̄_κ) Σ_j m_κ[j] ℓ_j for each kind κ,
- L_κ^own (stage 3) = the F3 term for kind κ,

with N̄ the **fixed expected** row counts per batch (section 5), and the total
L = w_0 L_0 + Σ_κ w_κ (L_κ + λ_κ L_κ^own). With w = 1 and λ = 0 every scored row counts once in
each term that contains it and nowhere else; the per-row weights are given in section 5. Each
L_κ satisfies P1-P3 by construction once L15 and L20 are fixed; T-P1..T-P3 are the tests that fail otherwise.

Reported per kind at every log point and checkpoint (section 6): L_κ, and its sub-splits on
onset rows (first 16 rows of a span), late rows, and informative rows (|v − preceding share| > 0.1).

### 2.5 Span membership under the deferred questions

Span membership is a property of the track, not of the loss code. If the human settles H-Q2 as
"hold the last value until the next span", the track is rewritten (an explicit, logged transform
that extends b to the next a) before the loss sees it; if as "natural", nothing changes; if as
"announce the upcoming span", an announcement channel is added to the frame as a named input with
its own lesion test. The loss code is the same under every answer, which is the sense in which the
design is workable without choosing.

### 2.6 Extension to three kinds of value

The frame per kind becomes: kind-specific value encoding (≤ 4 channels) · offsets to the span
bounds (8, as now) · progress (1) · kind-specific committed statistics over the span so far (≤ 4)
· remaining rows (1) · active (1). Lane-free throughout (mirror equivariance is untouched: every
new channel is invariant under the lane mirror; the hand-swap identity test in
`tests/r2/test_freerun.py` stays binding).

| Kind | Value encoding | Statistic over S | Committed statistics in the frame | F3 statistic on own samples | Leak | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| LN share (continuous, [0,1]) | 2v − 1 (as now) | LN heads / heads over S (`labels.ln_share`) | log1p heads, log1p LNs, ratio, remaining (as now) | realised share over S (relaxed: expected LN heads / expected heads per row on own history) | counters reveal v late in S | lesion: value channel; counters; each separately. T-P1..P3. Following: section 6 |
| Difficulty (residual to baseline, star units) | v_res / 0.5, clipped to ±3 | tiled_star(S) − b(S) (section 3) | mean chord size, same-lane repeat rate, held-lane occupancy, LN share over the committed part of S (all mirror-invariant) | tiled_star of the generated span − b(S) against v_res (score-function) | the skeleton explains b; the residual is new information; committed proxies reveal part of the realised residual late in S | lesion: v_res; each proxy; b must not change when decisions change (skeleton-only test). T-P1..P3 |
| Style concept c (categorical per section, independent salience per dimension) | one-hot over {absent, supporting, prominent} (3 channels) plus "unreviewed" = no interval of this kind (absent frame), never "absent" | the label of the section as annotated | counts of the deterministic query evidence available for the dimension (Jack: same-column repeats; Stream/roll: four-note directional groups; Trill: alternation of fixed disjoint groups; LN coordination: LN-occupied columns; Tech: none) | the same query evidence over the generated section where a deterministic rule exists; otherwise no F3 (Tech) | the first rows of a jack section reveal "jack" to the history; onset alignment applies as for LN | lesion per dimension; locality; a fixture kind with a deterministic span statistic exercises the categorical path (stage 4) |

What Beatmap Lens defines, read for this generalisation (decisions 0004-0006): the unit a style
concept labels is an **episode or section**, a half-open source-millisecond interval that the
consumer must adapt to its own coordinates and endpoint membership; labels are **multi-label**
(each dimension judged independently: Jack, Stream, Trill, Tech, LN coordination) and **ordinal**
("supporting" / "prominent", not to be silently converted to calibrated numbers), with explicit
masks for positive, negative, unresolved and unreviewed; they are obtained by expert-calibrated
annotation (human and agent) with deterministic, zero-false-positive query rules for the simple
patterns. The labeller is not designed here; the interface above is what R2 needs to take such a
label when it exists: kind = dimension, span = section, value = ordinal level, "unreviewed" =
absent frame.

---

## 3. Difficulty relative to the skeleton

**Baseline b(S).** Candidates:

1. A frozen regressor from head-time features of the span to the tiled star of the span, fitted
   on fit_train cells only. A already fitted and reported this: 23 features (duration, head-row
   density, gap statistics, local count bursts; no lane, multiplicity, LN or band feature);
   quadratic ridge R² 0.746 ± 0.013, RMSE 0.528 ± 0.014 star; 32-NN R² 0.753 ± 0.012, RMSE
   0.522 ± 0.010 on 7,862 fit_dev intervals from 1,163 charts; real interval star mean 3.59, SD
   1.05 **[measured, A `star-prediction-summary.json`]**.
2. The tiled star of a canonical arrangement of the same heads (for example one single tap per
   row, round-robin): deterministic, needs no fitting, but its offset from the source varies with
   the arrangement (on skeleton `0f6ab03b` single taps give 1.87 against a 2.66 source; A §1), so
   the residual would carry the arrangement's whole difficulty rather than its departure.

**Chosen: candidate 1, the quadratic ridge** (closed form, a few hundred bytes of weights, no
neighbour search at inference), refitted on the stage-0 relabelled cells (30 s at 10 s offsets)
on fit_train only, frozen and hashed; the kNN is the check that a linear-in-features form loses
nothing (A found them within one SE of each other). Consistent with tiling: it predicts the tiled
star of the span from the heads of the span, so the residual is in tiled-star units and the
whole-song interval uses the same b on [0, T).

**At inference.** The caller gives a span and a requested residual; b(S) is computed from the
skeleton's heads in S (the whole skeleton is an input, so b is available before generation,
including for spans not yet reached). The frame carries v_res only; b itself is a deterministic
function of inputs the model already sees in part (look-ahead over 16 rows and densities to
32 beats) and is not fed, to keep the residual the only new channel (optional arm: feed b too;
not adopted).

**Error of the baseline.** By definition the residual's spread *is* the baseline's error:
RMSE 0.53 star on fit_dev intervals; per-chart RMSE mean 0.46 ± 0.26 **[measured, A]**. So the
corpus residual target has SD ≈ 0.53 star; if roughly Gaussian its 10th-90th percentiles are about
±0.68 star **[inferred; the quantiles are a stage-0 output of the refit]**. That is the natural
width of "the response": requests of ±0.5 star sit at one SD, inside the ±0.8 the judgment called
the sane range on a fixed skeleton. A's earlier finding that head features explain ~75 % of the
label's variance is the same number seen from the other side: the residual is the remaining 25 %.

**What the model is trained towards.** CE on rows of S with v_res in the frame (F1), plus the
committed difficulty proxies (L17) so a tracking path exists; stage 3 adds the score-function F3
with reward −(tiled_star(generated S) − b(S) − v_res)². Residual targets outside ±1.5 star are
clipped in the frame and flagged in the receipt (extrapolation).

**Realised response on generated spans.** For a requested (S, v_res): generate with the real
prefix before S, export the span's objects on the cache representation, score tiled_star(S) with
the unchanged calculator, subtract the same b(S), and compare with v_res: dose-response slope and
MAE in star units over requests {−0.5, −0.25, 0, +0.25, +0.5} (section 6.2). The tiled-star
minimum of 30 s applies to S.

**Residual width in the corpus [to measure if cheap].** A's number suffices for the design; the
stage-0 refit prints the residual quantiles and the per-band width (the judgment's ±0.8 was from
one skeleton).

---

## 4. Span-aligned sampling

The draw (implemented in `draw_sim.py` for measurement; parameters are config):

1. Group uniform, chart uniform (unchanged).
2. Candidate spans: every nonempty LN piece of a fresh random partition (8/16/32/64 beats, as
   `conditions.py:34-52` but returning all pieces), and every 30 s star cell of the chart plus
   the whole song.
3. With probability **p_align = 0.6**: choose a kind (uniform among kinds with candidates), then
   one candidate of that kind with importance weight **3** for an LN piece whose share differs
   from the preceding 64 rows' share by more than 0.1 (weight 1 otherwise; star cells weight 1,
   the whole song 0.10); set **j = onset row − U{0..32}**, stop = min(j + 256, K + 1). Then draw
   1-3 intervals of the other kind among those intersecting [t_j, t_stop) (consecutive with
   p 0.5, as now). Otherwise (p 0.4): the current start rule, and 1-3 intervals per kind among
   those intersecting the window.
4. Dropout as now (0.20 all, 0.25 per kind, 0.20 per interval) **pending H-Q1**: the dropout
   rates are what define how much natural training the model gets.

Properties and the tests that bind them:

- **D1 onset coverage**: among active intervals in a window, ≥ 80 % have their onset row scored
  (current, per the audit: 32 % of active LN rows belong to intervals whose onset lies before j;
  the new target counts intervals, not rows). Expected under the rule
  [inferred]: every aligned interval (onset at or after j by construction) plus most
  window-intersecting ones.
- **D2 song-length independence**: the number of active intervals per window regressed on K over
  2,000 draws has a slope whose 2-SE interval contains zero (current code fails: `conditions.py:55,74`).
- **D3 informativeness**: scored rows that are within 16 rows of an onset whose value differs
  from the preceding 64 rows by more than 0.1 are ≥ 3 % of scored heads [inferred expectation:
  about 5-8× the current ≈ 0.7 %; current with > 0.25 is 0.18 %]. If the simulation shows less,
  raise the importance weight or p_align; the thresholds are stated before the measurement.
- **D4 active share**: LN-active ≥ 25 % and star-active ≥ 25 % of scored heads
  [inferred expectation: 15-25 % at the parameters above with the current dropout; the dropout
  rates, not the alignment, cap it — part of H-Q1].
- **D5 manifest equality**: `build_manifest` calls the same draw function; a test draws 1,000
  training windows and 64 manifest windows with the same rule and checks that the manifest's
  active-row shares per kind are within two SE of the training shares; the manifest is
  stratified to carry ≥ 24 windows with a scored LN onset and ≥ 24 with a scored star onset.
- **Switches scored**: adjacent LN pieces with different values (consecutive draws) give a value
  change inside the window; the diagnostics measure response at those rows (section 6.2).

Values that are not the source's own: **rejected for CE targets** (relabelling source decisions
with a value they do not have is label noise that teaches the model the value is not to be
followed; A proposal 3 says the same). The two valid routes to "two values on one skeleton" are
the model's own samples with their realised statistics (stage 3) and accepted alternative
arrangements with recomputed labels, which need an acceptability judge that does not exist
(H-Q5).

Windows with no condition: kept at the rate the dropout gives (so the comparison with the
overnight run is matched and empty tracks stay supported); their rows go to L_0 with weight w_0.
What "natural" means for them, and hence w_0 and the dropout rates, is H-Q1; nothing in this plan
picks it.

Star spans and the window: a 30 s cell is ≈ 190 rows at the median rate and fits; on dense charts
it overflows and the scored part is the onset side, which is where the information is. The
unscored remainder is then under the condition without a term, which is permitted (it is simply
not scored), not a violation (no term uses it). The 60 s cells and the whole song exceed every
window; the whole song stays at p 0.10 as a residual (L12), the 60 s cells leave the draw.

---

## 5. Loss normalisation

Per-row weight of a scored decision j in a batch of B windows:

u_j = w_0 [j ∉ any span] / N̄_0 + Σ_κ w_κ m_κ[j] / N̄_κ,   L = Σ_j u_j ℓ_j,

with N̄_0, N̄_κ the **expected** number of such rows per batch under the draw (measured once by the
stage-0 simulation and stored in the run config; recomputed whenever the draw parameters change,
with a test that the stored values match a fresh 2,000-draw estimate within 5 %). Fixed
denominators, not per-batch counts, so that:

- a batch with few active rows of kind κ contributes proportionally less to L_κ (not inflated:
  per-batch normalisation would give five rows the weight of five hundred);
- a window with few active rows is not drowned: each active row carries the same weight in every
  batch (per-window 1/L is gone, which also removes the EOS 4.2× and last-64-rows 1.8× skew);
- the expected contribution of each term per batch is w_0 : w_κ, so a rare kind is neither
  drowned nor inflated in expectation.

Defaults w_0 = w_κ = 1, λ_κ = 0 until stage 3. Rows under two kinds get both weights; EOS belongs
to L_0 with weight 1/N̄_0. Gradient accumulation divides by the number of windows as now; the
divisor is constant across batches. Tests: T-P3a (adding natural windows leaves L_κ unchanged);
a unit test that Σ_j u_j over a batch drawn with the stored N̄ equals w_0 + Σ_κ w_κ within 5 %
over 200 batches; and the DPO anchor uses the same function (L22).

---

## 6. Diagnostics and selection

Each metric names the decision it informs. All conditioned measurements use the fixed panel and
seeds of section 9 and A's frozen `probe.py` where it already exists, so numbers are comparable
with 2026-10-04.

### 6.1 Per-kind teacher-forced metrics (every log point; fit_dev at every checkpoint)

- NLL on active rows of kind κ; on onset rows (first 16 of a span); on late rows; on informative
  onset rows (|Δ| > 0.1). *Informs*: whether the value is being read at all (onset-row NLL under
  the true value against the same rows with the value swapped: the paired difference must be
  negative and > 2 SE before any free-run claim is made); whether a kind term is drowned.
- Natural-row NLL (rows under no kind), per-decision, replacing "natural vs conditioned".
  *Informs*: the natural guard and the checkpoint selection.

### 6.2 Counterfactual response on fixed real states (every checkpoint, cheap)

A's P1 probe, restricted to two strata: onset rows (≤ 16 rows after a span start) and late rows.
Same state, value swapped (LN 0 / 0.9; residual −0.5 / +0.5; style absent / prominent later):
expected statistic change and action KL. *Informs*: whether the response exists at the onset,
where history cannot substitute for the value (the stratum the audit identified as 0.18 % of
scored heads today); a late-row response without an onset response means the model follows
counters, not the value.

### 6.3 Representation probe (per checkpoint, diagnostic only)

Linear probe from the hand vector at onset rows to v (F4 as a probe). *Informs*: whether a
missing response is a representation failure or a decoder failure.

### 6.4 Conditioned free-runs (every checkpoint; in-run)

Panel: 16 fit_dev charts (A's 8 plus 8 covering star bands and lengths to 1,500 rows), seeds 954-956
(generation), modes: natural; whole-song LN requests {0, 0.1, 0.3, 0.6, 0.9}; half-song switch
0.1→0.6 and 0.6→0.1; residual-star requests {−0.5, −0.25, 0, +0.25, +0.5} on 30 s onset-aligned
spans (stage 2 on). Metrics (A's definitions): LN slope, MAE, switch DiD/2 and per-half MAE;
residual slope and MAE in star; natural LN share against the source panel; holds ≤ 60 ms;
releases 1-40 ms before another head; legality; chord histogram JS to the source. *Informs*:
following gates, the natural guard, the defect guard.

### 6.5 Teacher-forced against own-history calibration (every checkpoint)

B's M1H: expected LN forecast per row on real histories against the same on the model's own
sampled histories from the same start states (16 charts × 96 states). The gap is the exposure-bias
number. *Informs*: H-Q4 (whether stage 3 is needed) and guard (iii).

### 6.6 Checkpoint selection rule (fixed before any run)

1. Candidates: checkpoints after warm-up whose 48 natural and 240 conditioned free-runs are all
   legal with every head present.
2. Primary: fit_dev per-decision NLL over all scored rows of the manifest, with a paired
   song-group bootstrap SE (2,000 resamples).
3. Guards: (i) natural free-run LN share: panel mean within 0.05 of the source panel mean and
   change from the previous checkpoint ≤ 0.1 **(this guard presupposes "natural = source
   behaviour"; it changes under other answers to H-Q1)**; (ii) LN slope ≥ 0.7 and MAE ≤ 0.15
   (A's registered gate); (iii) own-history calibration gap ≤ 0.05; (iv) holds ≤ 60 ms ≤ 0.5 %
   and releases 1-40 ms before another head within the source band's rate in conditioned runs.
4. Select the earliest checkpoint passing all guards whose NLL is within 2 SE of the minimum
   among guard-passing checkpoints. If none passes, report "no selection" with the failing
   guard; such a checkpoint may serve mechanism tests only.

For arm comparisons (stage 1, 2) the guards are reported per arm and the pre-stated primary
metric decides; the rule above selects within an arm.

Cost of the in-run evaluation: the current 12 free-runs take 3-9 s per checkpoint (`evals.jsonl`
`eval_s`); 48 natural + 240 conditioned runs on charts up to 1,500 rows at about 250-350 rows/s
is 10-15 min per checkpoint [inferred], against 25-30 min of training between checkpoints at
4.39M exposures; acceptable, or run the conditioned panel every second checkpoint.

---

## 7. Exposure bias and free-run

Must the condition response be trained on own histories, or only checked? Evidence: teacher-forced
forecasts calibrated, own-history forecasts +0.22 at 30.72M and "none" at 39.49M (B); natural LN
oscillating to the end of the run (L28, weak); count feedback at decode time tracks exactly (A)
but adds 1-3 % holds ≤ 60 ms. Reading [inferred]: CE on real histories cannot stabilise the
free-run first moment because nothing in it sees the states the model produces; the DPO objective
would, but waits for real pairs (human decision).

The plan: **check first, train second.** Stage 1 adds the calibration gap and the conditioned
free-runs to every checkpoint (section 6.4, 6.5). If, at the stage-1 stop rule, the calibration gap
is ≤ 0.05 and the LN slope ≥ 0.7, stage 3 is unnecessary for LN. Otherwise stage 3 trains F3
relaxed on own histories for LN (the per-row expected LN share on the own-sampled span against
the request, through the per-row probabilities; the sampling itself carries no gradient), with
the CE terms as anchor and guard (iv) as the stop; and the score-function form for the star
residual. This is not DPO and uses no preference labels; whether it may run before real pairs is
H-Q4. "Teacher-forced history reveals the statistic" is not a problem for F3: on own histories the
counters reveal the model's own realised share, and matching the request from there is exactly the
quota control the decode-time feedback did by hand.

---

## 8. Infrastructure

What the mirrored files say (L34-L37) and what the plan needs from the supervisor and trainer:

1. **Guard semantics.** Keep the RSS limit (12 GiB) and `min_available_bytes`; raise or remove
   the system swap-growth trip (it measures other processes), or key it on the trainer's own RSS
   growth between checkpoints. Record `available_bytes`, `pressure_level` and the trainer's RSS
   in every `resource_limit` event so the next stop is attributable from the control plane.
2. **Restart budget as a rate**: at most 5 restarts in any 6-hour window, unlimited over the
   run; a resume that reaches the next checkpoint clears the counter; the supervisor logs the
   trainer's RSS and the system numbers at every exit.
3. **Log segmentation**: `train.jsonl` and `resources.jsonl` written per checkpoint segment
   (`train-<exposures>.jsonl`), each well under 4 MB, so the mirror carries the whole curve;
   `resources.jsonl` compacted to one line per log point (it is 16,609 lines for 33M exposures).
4. **Stop on the data, not only on the clock**: a fit_dev plateau rule in the trainer (no
   per-decision NLL improvement > 1 SE over 3 consecutive checkpoints after 2 passes) as an
   optional stop, so a schedule cannot run nine passes past its minimum (L29); the schedule's
   cosine horizon then follows the budget in section 9.
5. **Evaluation in-run**: the per-kind metrics, the counterfactual probe, the calibration gap and
   the conditioned free-run panel (section 6) in `evaluate()`, with the frozen panel manifest and
   a receipt per checkpoint; the conditioned-generation CLI (`--freerun-only` with a track file).
6. **Memory**: a pilot that records peak RSS per window against factor × candidate counts (L37);
   float32 for the per-candidate arrays and per-factor chunking if the spikes are confirmed.
7. **Only the mac's `resources.jsonl` can answer**: the trainer's RSS and `available_bytes` at
   each of the six trips; whether RSS was growing across the 13 hours before the first trip
   (a leak) or flat (external pressure). Reading it is a 30-second `ens cat` once the recheck job
   has ended; it changes item 1's choice between "remove" and "re-key" the swap trip.

---

## 9. Stages

Throughput assumption: 2,000-4,000 head decisions/s on 4 threads (brief); budgets quoted at
2,500/s. One training seed is labelled as such until the second seed runs (fm-margin-below-noise).

### Stage 0: measurements, code, tests (no training; about one working day of code plus < 1 h mac)

Changes: span-aligned draw with the stored N̄ (section 4, 5); per-decision loss with per-kind
masked terms and logging; presence bit → active (L15); frame generalisation per kind (L18) with
the star committed proxies (L17); relabelling of 30 s cells at 10 s offsets from the cache
representation (L10, L11, L13; about 15 min on the mac); baseline b refit on the new cells, frozen
with a hash, and the residual quantiles printed (section 3); manifest rebuilt with the shared draw
and stratified (L8); conditioned free-run CLI and the section-6 evaluation; the guard, restart
and log changes of section 8; the DPO anchor on the shared loss (L22).

Mac measurements: `draw_sim.py` (current vs proposed draw; gives N̄, D1-D4); the relabel; the
baseline refit; a 200-window pilot for throughput and peak RSS with the new frames (cost of F2 and
F3 forward passes measured in the same pilot, with the F3 sampling path exercised on 1 window in 4).

Tests that fail if the stage is wrong (all engineering, none evidence):

- input-lesion per named input, one forward each, on a model with randomised zero-initialised
  paths (`tests/r2/helpers.py` pattern): LN value; LN counters (heads, LNs, ratio, remaining)
  individually; star residual value; each star committed proxy; the active bit; the candidate-role
  and birth-role frames in the pointer; the announcement channel if H-Q2 adds one. Each asserts
  the decision log-probability changes; a documented "unused" is not accepted for a mandated input.
- T-P1, T-P2, T-P3a, T-P3b for L_LN and L_star (section 2.2); T-P2 also for the F3 code path on
  a fixture.
- D1-D5 on 2,000 draws against the cache (skipped where the cache is absent, as the existing
  train-step tests do).
- The weight-sum test of section 5; EOS weight equals one decision's.
- b is a function of the skeleton only: changing every decision of a chart leaves b(S) unchanged;
  b is mirror-invariant; b(S) depends only on heads in S.
- Relabel consistency: for the existing 30 s cells, new labels equal old ones within 0.001 star
  except where the cache representation differs (count reported).
- Stored N̄ within 5 % of a fresh estimate.
- Existing suites unchanged (normalisation, mirror, leakage, causality, free-run smoke).

Human decisions needed before stage 1 starts: H-Q1 only insofar as the dropout rates and w_0
are kept at the current values for the matched comparison (the plan keeps them; the human may
change them); H-Q6 (presence bit) if the human wants announcement of upcoming spans now.

### Stage 1: span-aligned draws and per-decision loss, matched exposure (training)

- **Claim**: aligning condition spans with the scored window and weighting decisions equally
  raises the learned LN response at matched exposure.
- **Arms**: A0 the overnight recipe (current draw, 1/L) and A1 the stage-0 recipe; same width,
  lr 1e-3 cosine, seeds 171 (weights) and 471 (draws); then the second training seed 172/472 for
  both arms. Both arms with star conditions on (absolute tiled star in both, so that stage 2 is
  the only star change).
- **Budget and stop rule**: 50M exposures per run (≈ 4 passes; the overnight minimum was at
  61M with the schedule's cosine stretched to 158M) with the cosine horizon set to 50M, checkpoints
  every 4.39M; a run stops at 50M or on the plateau rule; no comparison is read before 30M.
- **Metric, panel, seeds**: primary: whole-song LN slope on A's 8-chart panel, seeds 954-956,
  seed-averaged then chart-paired between arms at matched exposure (30M and 50M). Secondary:
  onset-row NLL under the true value minus swapped value (section 6.1); switch DiD/2; natural
  fit_dev NLL; calibration gap; guard (iv).
- **Threshold**: A1 − A0 slope ≥ +0.20 and > 2 paired SE at 50M, with natural-row NLL not worse
  by more than 0.02 nats and holds ≤ 60 ms ≤ 0.5 %. The onset-row NLL difference must be
  negative and > 2 SE (otherwise the draw change did not reach the value channel and the slope
  result, if any, is attributed elsewhere).
- **Outcome rules**: threshold met → the stage-0 recipe is the base; stage 2 starts. Not met with
  the onset-row NLL resolved → the response exists in teacher forcing but not in free-run → stage 3
  moves ahead of stage 2 (H-Q4). Neither resolved → the sparse-signal hypothesis is dead for this
  interface; F2 and H-Q5 are the next candidates; no architectural conclusion.
- **Cost**: 4 runs × 50M / 2,500/s ≈ 5.6 h each, 22 h of mac in total, sequential (the mac is
  shared); the second seed pair runs only after the first pair's 30M read-out exists (one seed is
  labelled "one training seed" until then).
- **Engineering tests that fail if the stage is wrong**: a config-diff test that the two arms differ only in the draw and the loss weighting (same code hash otherwise); an exposure-accounting test that both arms' checkpoints at "30M" and "50M" are within 1 % of each other in head decisions; the stage-0 suite green on the frozen copy of each run.
- **Human**: none beyond approving the budget.

### Stage 2: difficulty as a residual to the skeleton baseline (training)

- **Claim**: conditioning on tiled_star − b(S) with committed difficulty proxies gives a measurable
  residual response where the absolute value gave none.
- **Arms**: B0 absolute tiled star (the stage-1 winner's recipe) and B1 residual + proxies; one
  seed first, second seed if B1 − B0 passes at one seed.
- **Metric**: realised residual (section 3) against requested {−0.5, −0.25, 0, +0.25, +0.5} on
  30 s onset-aligned spans, 8-chart panel, seeds 954-956: slope and MAE in star; A's star gate
  (slope ≥ 0.5, MAE ≤ 0.5 star) on the residual scale; LN metrics unchanged within 2 SE (no
  coupling regression); holds ≤ 60 ms guard.
- **Threshold**: B1 slope − B0 slope ≥ 0.3 and > 2 paired SE; B1 passes A's gate.
- **Budget**: 50M per run, 5.6 h each; 2-4 runs.
- **Stop rule**: as stage 1.
- **Engineering tests**: the stage-0 skeleton-only, mirror and lesion tests for b and the proxies; a relabel-consistency test; the same config-diff and exposure-accounting tests as stage 1.
- **Human**: confirm the residual request range (±0.5 star) and whether the whole-song interval
  stays at p 0.10 (H-Q11).

### Stage 3: condition response on own histories (training; behind H-Q4)

- **Claim**: the relaxed F3 term for LN (and score-function F3 for the residual) closes the
  own-history calibration gap and lifts free-run following without fake holds.
- **Arms**: C0 the stage-2 winner; C1 with λ_LN = 1 (and λ_star = 1 if stage 2 passed), F3 on one
  window in four, started from C0's weights at its selected checkpoint (continuation, 20M
  exposures, both arms continued equally).
- **Metric**: calibration gap (section 6.5); LN slope and MAE; switch DiD/2; holds ≤ 60 ms;
  natural-row NLL.
- **Threshold**: gap ≤ 0.05; slope ≥ 0.9 and MAE ≤ 0.10, > 2 SE over C0; holds ≤ 60 ms ≤ 0.5 %;
  natural NLL within 0.02.
- **Budget**: 2 runs × 20M at roughly 1.5× the step cost ≈ 3.3 h each.
- **Engineering tests**: T-P2 on the F3 term (gradient zero outside the span, on own-sampled rows); no gradient through the sampling path; a fixture test on a tiny model and synthetic charts that 200 F3 steps move the realised statistic toward the request in the stated direction by more than 2 SE over three fixture seeds, with the threshold fixed here and not loosened after a failure (the DPO integration test of [s-r2-dpo] is the precedent to avoid).
- **Human**: H-Q4 before it starts.

### Stage 4: categorical kind path (engineering, no claim)

A fixture kind with a deterministic span statistic computed from the source (for example a
three-level band of chord-size ≥ 2 rate over the span) exercises the categorical value encoding,
locality tests and lesion tests on a tiny model; a toy run must show a positive fixture response
at 2 SE as a mechanism check. This is not a style labeller and makes no style claim; it is what
lets a Lens label plug in later with no interface work.

### Deferred

- Token vs FiLM at matched exposure: after H-Q2.
- Landmark readout ablation (L39): after H-Q9; the test is section 1.8.
- DPO with real pairs: when they exist; horizon ≥ 30 s for star pairs (L40).
- Capacity (width) and the lr/decay re-tune (L29, H-Q7): after stage 1, on fit_dev evidence.

---

## 10. Open questions for the human

| # | Question | Options (stated, none recommended) | What it changes in the plan |
| --- | --- | --- | --- |
| Q1 | **Natural behaviour**: what the model does when a kind is not specified (deferred). | (a) imitate the source on unconditioned rows, as now; (b) an implicit default value of the kind (the corpus-typical statistic) that the model should realise; (c) a value drawn once per song or section from a corpus prior at inference (C's suggestion), with training rows under no kind scored as (a) or not scored. | w_0 and the dropout rates (section 5); guard (i) of section 6.6; what the free-run natural panel is compared with; whether windows with no condition are scored (L_0) at all. |
| Q2 | **Between two conditioned parts** (deferred). | (a) natural behaviour per Q1; (b) hold the last value until the next span (track transform); (c) a transition the model shapes itself because it can see the upcoming span (announcement channel / token form); (d) a transition rule given by the interface (for example a ramp). | The track transform of section 2.5; the presence/announcement channel (L15, Q6); whether the token-vs-FiLM comparison is meaningful; which rows between spans enter L_0. |
| Q3 | Which branch of [q-r2-after-judgment] item 3 applies: the in-run free-run shows natural LN still moving by > 0.1 between adjacent checkpoints to the end (L28, 4 charts, weak), and the recheck measures one checkpoint only. | (a) accept the weak evidence as "CE recipe first" (this plan's ordering); (b) wait for a larger-panel measurement on three late checkpoints (about 1.5 h on the mac with B's `rerun.py`). | Whether stage 1 starts now or after the measurement. |
| Q4 | May a self-supervised term on the model's own samples (F3: statistic against the request, no preference labels) be used before real preference pairs exist? | yes / no / only if stage 1 fails the free-run gate while passing the teacher-forced one. | Stage 3's existence and ordering. |
| Q5 | Should counterfactual values be manufactured from accepted alternative arrangements of the same skeleton (A proposal 3), which needs an acceptability judge? | now / after the evaluator is calibrated / never. | An extra data route for L1; otherwise L1 is answered by stages 1 and 3 only. |
| Q6 | Presence bit: remove it (rows outside a span learn nothing about the kind) or replace it by an explicit announcement of the next span (start offset and value)? | remove / announce (FiLM) / whole-track (tokens). | L15 fix; the input-locality test; overlaps Q2. |
| Q7 | The overnight run's fit_dev minimum at 39 % of its schedule (L29): re-tune lr and weight decay before stage 1, or keep lr 1e-3 with the 4-pass budget? | keep / re-tune in a 2 × 30-minute pilot. | Stage 1's budget and schedule. |
| Q8 | Style concept interface: the three ordinal levels and the "unreviewed = absent frame" rule (section 2.6) read from Lens 0005/0006; confirm, or name the vocabulary version R2 should target. | confirm / amend. | Stage 4's fixture and the frame width. |
| Q9 | Spend a run on the landmark question (L39)? | now / after stage 2 / drop the readout. | Deferred list. |
| Q10 | Guard policy on the shared mac: remove the system swap trip or re-key it (section 8.1), and the restart-rate numbers. | remove / re-key / keep with 4 GiB. | Stage 0 infrastructure. |
| Q11 | The whole-song star interval: keep at p 0.10 as a residual, lower it, or drop it from training draws. | keep / lower / drop. | Stage 2's arms. |

---

## 11. Question ledger and advice ledger

### 11.1 Questions

| Question | Status |
| --- | --- |
| [q-r2-after-judgment] 1: star as a control: drop or redefine | **Settled by the human (direction 2)**: keep difficulty, as a residual to a skeleton baseline (stage 2). "Decided quantities" instead of star: rejected by that decision. |
| 2: window-aligned draws in the next CE run at matched exposure | **Scheduled**: stage 1, with the thresholds above. |
| 3: end-of-run re-probe and the decision rule | **In progress** (recheck job, one checkpoint, numbers only per the human); the oscillation branch is Q3 for the human; this plan proceeds on the "CE recipe first" ordering pending Q3. |
| 4: fix the DPO labeller before DPO | **Settled further by the human (2026-10-05)**: no synthetic labeller at all; DPO waits for real pairs (`7d9640a`). |
| 5: a style vector only after a 1-D condition is held over a song by CE | **Adopted as ordering**: the categorical interface is built (stage 4, engineering) and no style training happens before stages 1-3 pass their gates. |
| Audit minimal fix 1.1-1.5 (pass j/stop; align with p 0.5; keep intersecting intervals, count per window; oversample informative; same for the manifest) | **Adopted with changes**: p_align 0.6, importance weight 3 at |Δ| > 0.1 (not only > 0.25), the onset lead U{0..32}, the manifest from the shared function and stratified. |
| Audit 2.1 (window-scale star labels at random offsets) | **Adopted as precomputed 30 s cells at 10 s offsets** (online random offsets would cost 23 ms per draw; precomputation is 15 min once). |
| Audit 2.2 (whole-song star as its own global channel) | **Rejected**: contradicts the human's 2026-10-03 decision against a chart-level scalar; the whole-song interval stays an interval, with a residual value (Q11). |
| Audit 2.3 (measure R² of star from head features before more training) | **Done by A** (0.75); adopted as the baseline. |
| Audit 3.1-3.3 (per-kind NLL; onset-aligned counterfactual evaluation; conditioned free-run) | **Adopted** (section 6). |
| Audit 4 (count by K) | **Subsumed** by the per-window draw; test D2. |
| Audit 5 (normalise by total decisions, or move short windows back) | **Adopted the first** (fixed divisor); moving windows back changes the position distribution and is not needed. |
| Audit 6 (star from the cache representation) | **Adopted** with the relabel. |
| Audit's memory test (readout on k > 511 rows; 6-level TCN) | **Deferred**, Q9. |
| Does the model need to see values other than the source's own? | **Partly settled**: not as CE targets (label noise); own samples (stage 3, Q4) and accepted alternatives (Q5). |
| Is the condition range still misplaced after stage 0? | **Scheduled**: D1-D5 on the simulation and in the test suite. |
| Is the swap stop the trainer's fault? | **For the mac's `resources.jsonl`** (section 8.7); the plan's guard change does not depend on the answer. |

### 11.2 Advice

| Recommendation (paraphrased) | Source | Condition | Status in this plan |
| --- | --- | --- | --- |
| Repeat A's and B's frozen panels at the end of the run and two intermediate checkpoints; decide on oscillation, calibration shift, slope | judgment §7.1 | run ended | Partly: the recheck does one checkpoint by the human's instruction; Q3 asks whether to do the three-checkpoint version. |
| Window-aligned condition draws, matched exposure, two training seeds, same panel and gates | judgment §7.2, A proposal 2 | after the end-of-run probe | Adopted (stage 1), two seeds, A's panel and gates. |
| Drop tiled star or redefine as residual star | judgment §7.3, A proposal 4 | decision, no compute | Residual adopted (human direction 2); "drop" rejected by the human. |
| Fix the labeller; calibration pairs; typicality judge only as a defect filter | judgment §7.4 | before DPO | Superseded: no synthetic labellers; DPO waits for real pairs. Pair-construction rules kept for later. |
| Style z only after a 1-D condition is held | judgment §7.5, C E3 | stages 1-2 pass | Adopted as ordering. |
| Compare guided sampling and count feedback under an acceptability gate | A proposal 1 | one pinned checkpoint | Not adopted now: decode-time controls hide the training defect (judgment §5.3); the hold-length guard (iv) carries the quality concern into training. |
| Build valid contrasts from accepted same-head arrangements; never relabel with a value the actions do not have | A proposal 3 | acceptability judge | The "never relabel" half adopted (section 4); the contrasts are Q5. |
| Candidate selection by proximity as a fallback with explicit cost | A proposal 5, C E1 | after conditional sampling works | Deferred; not a training matter. |
| Judge a larger model by rollout calibration and following, not teacher-forced loss alone; repeat B's package before changing size | B | before capacity changes | Adopted: capacity deferred until after stage 1; selection rule includes calibration and following. |
| Align the target interval with the scored loss window on conditioned draws | C §3 | — | Adopted (stage 0). |
| Every condition needs its own absence state; dropping style must not drop LN or star | C §3 | — | Adopted in the frame design (per-kind absent frame). |
| No scheduled sampling before DPO (substituted states make real labels illegal) | design §4 | — | Respected: F3 uses the request, not real labels, on own histories; it is not scheduled sampling. |
| No guidance in the first build; a guidance proposal must define a normalised distribution and pass the mirror tests | design §3 | — | Respected: no guidance in training or in the gates. |
| Selection: earliest checkpoint within two paired SE of the lowest fit_dev CE among checkpoints passing the generation and condition guards | design §4 | — | Adopted and made concrete (section 6.6); the overnight selection did not implement the SE or the guards (L26). |
| Expert advice from the lineage: do not read non-convergence at equal updates as "does not help"; keep training if the curve is still improving | agent-failure-modes §7 | curve still improving | Stage budgets carry a plateau rule and no claim below 30M. |
| Two training seeds for any recipe claim | agent-failure-modes §8 | — | Adopted in stages 1-3. |

---

## 12. What I could not read or check

- The mac's full `train.jsonl`, `resources.jsonl` and checkpoints (mirror limit; L36); the
  trainer's RSS at the swap trips.
- The simulation of the proposed draw (`draw_sim.py`, ready, not run: launch refused while the
  recheck job runs); every D1-D4 expectation is inferred until it runs.
- The recheck job's results (running at the time of writing).
- Acceptability of any generated chart; no renders were viewed.
- Whether the Lens five-target vocabulary (0006) is the one R2 should take; Q8.
