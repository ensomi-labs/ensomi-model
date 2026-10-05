# R2 condition plan: adversarial review

Shareable. Written 2026-10-05 by a fresh Opus subagent on [r2-condition-plan](r2-condition-plan.md), for [r2-condition-plan](r2-average-and-control.md#r2-condition-plan). Kept as written. Main thread checked against code and data: the recheck slope 0.715 ± 0.045, the design rule against draw balancing without inverse-probability weights, release frames at the LN start and candidate times (`model.py:265-281`), the 64-beat cap on LN pieces (`conditions.py:20`), the rejection of negative star values (`conditions.py:100`).


Reviewer: fresh Opus subagent, control plane, read-only. Plan reviewed:
`scratchpad/r2-condition-plan.md` (701 lines; the notes copy `artifacts/r2-condition-plan.md`
differs only in its header). Code: `r2/train` at `7d9640a`. Nothing in the repository was edited,
nothing committed, no mac job launched.

Tags: **[code]** read at the cited `file:line` (paths relative to `src/ensomi_model/r2/` unless
stated); **[data]** read from the named file; **[inferred]** my reasoning. "SE" follows the project
rule: a difference under two SE is "none". Human direction is cited by the brief's paraphrase only.

---

## 0. Summary

The ledger is accurate where I checked it. Of 26 citation groups spot-checked (about 50 line
references), 23 hold exactly, one holds with a 0.002 slip in a number, and two have line or
mechanism drift (section 4). Sections 5 (fixed denominators), 6 (per-kind diagnostics) and 8
(infrastructure) are sound and I would keep them. The plan has five problems that would change
its outcome:

1. **Stage 1's primary metric cannot decide stage 1.** No LN interval in training is longer than
   64 beats, so whole-song and half-song LN requests are out of distribution for both arms. The
   recheck job, which finished after the plan was written, puts the overnight recipe at LN slope
   0.715 ± 0.045 at 61.4M. That passes A's gate, so the +0.20 threshold sits against a ceiling.
2. **The draw changes the natural-row distribution through target-dependent selection.** Dropout
   is applied after alignment, so about half of the aligned, informative windows become natural
   windows chosen for an LN-share change. The design's own rule (design.md:303) requires
   inverse-probability weights for any such balancing, and the plan applies none. Natural LN is
   the one failure that persists at 61.4M, and this draw is likely to make it worse.
3. **L_κ is a reweighted CE.** It meets the "rows" reading of the human's constraint but not the
   "aspects" reading. It also double-counts rows under several kinds and is not compositional. An
   exact governed-factor decomposition (LN-ness given the head mask) costs nothing extra and fits
   both readings.
4. **Difficulty.** Direction 2 most naturally asks for a loss on the realised response (F3). The
   plan instead gates F3 behind stage 3 and a question framed by the DPO decision. F3's cost and
   variance are underestimated. The baseline refit on 30 s cells only turns every whole-song b
   into an extrapolation.
5. **Release factors read conditions across span boundaries** (birth-role and candidate-role
   frames). The structural claim in §2.1 is wrong, T-P3b(i) fails even after the L15 fix, and
   F3's realised statistic depends on decisions outside the span.

---

## 1. New evidence the plan could not use

The recheck job (`20261005-044020-r2-recheck`) wrote its probe tables while the plan was being
finished (control table 05:13 UTC, B package 05:34 UTC). The main thread has not checked them yet,
and they cover one checkpoint of one training seed.

- [data] `artifacts/r2-recheck-20261005/control-table.md`, ckpt-0061432779:
  - LN whole-song slope 0.715 ± 0.045, MAE 0.115 ± 0.011. A's gate is slope ≥ 0.7 and MAE ≤ 0.15,
    so it is met.
  - Half-song switch: DiD/2 0.272 ± 0.034 (gate 0.30, not met), switch MAE 0.142.
  - Natural LN share 0.312 ± 0.038 against 0.187 real.
  - Teacher-forced Δ expected LN fraction (request 0→0.9) 0.060 ± 0.010.
  - Star: slope 0.014 ± 0.015, value KL 1.0e-5 nats/row.
- [data] `artifacts/r2-recheck-20261005/average/tables.md`: LN forecast on own histories minus real
  histories, 0.129 ± 0.044. Teacher forecast minus source is −0.003 ± 0.003.
- [inferred] Consequences for the plan:
  - CE with the current draw keeps improving whole-song LN following (0.21, 0.45, 0.50, 0.72 at
    30.7M, 39.5M, 43.9M, 61.4M).
  - What still fails: switches, natural LN (bias +0.125 and an own-history shift above 2 SE), and
    star.
  - By the plan's own §7 rule ("gap ≤ 0.05 and slope ≥ 0.7 makes stage 3 unnecessary"), the gap
    of 0.13 already triggers stage 3 for LN.

---

## 2. Ranked points

Ranked by how much each would change the plan's outcome.

### P1. Stage 1's primary metric is out of distribution and near ceiling: disagree

- **Evidence.**
  - [code] `conditions.py:20` gives `LN_BEATS = (8, 16, 32, 64)`. In `:34-56` an LN interval is
    one partition piece, and `_choose` (`:26-31`) keeps adjacent pieces as separate intervals.
    No training LN interval exceeds 64 beats (audit: median 77 rows, p90 254).
  - For a whole-song request, the frame at row k carries offsets −t_k and T − t_k, `remaining` up
    to K rows, and counters over the whole prefix (`features.py:296-302`). `psi` clips only the
    linear part at ±32 (`common.py`, `psi`); the log part keeps growing. On a 500-row chart a
    whole-song span is about twice the longest training span; on long charts it is much more.
  - [plan §4 step 2] The proposed draw keeps the same 8-64-beat pieces.
  - [inferred] So A's dose-response and switch panels, the plan's §6.4 panel and the stage-1
    primary all measure extrapolation in span length, in both arms.
  - [data] The overnight recipe reaches slope 0.715 at 61.4M (section 1), while lr was still
    about 6.8e-4 [inferred from `config.json` cosine].
  - [inferred] An A0 annealed to 50M may well reach about 0.7. The threshold A1 − A0 ≥ +0.20 would
    then need A1 ≥ 0.9 against a ceiling near 1.
- **Consequence if left.** The outcome rules fire on a saturated, out-of-distribution metric:
  "not met" sends stage 3 ahead of stage 2, or declares the sparse-signal hypothesis dead, for
  reasons unrelated to the hypothesis.
- **Change (§4, §6.4, §9 stage 1).**
  1. Primary: in-distribution span following at onsets. Use 16- and 32-beat requests placed at
     real onsets, with values chosen to contradict the preceding 64 rows (for example 0.05 and 0.9
     against a source near 0.2). Measure the realised share over the span by head ownership.
     Secondary: the switch at in-distribution lengths. Whole-song slope becomes a generalisation
     metric, reported but not decisive.
  2. The interface is "whole song or interval", so long LN spans belong in training. One way: an
     interval that is the union of 1..N consecutive pieces, with whole-song at a stated
     probability as star has. Put it in both arms, or as a separate, declared change; it must not
     ride along with the draw comparison. Whether to add it, and when, is a human call.
  3. Star has the same issue. Dropping the 60 s cells (L11) leaves only 30 s and whole-song spans,
     so every request between those lengths is an unseen span geometry. Draw several lengths
     instead (30, 45, 60, 90, 120 s, whole song).
  4. Re-derive the threshold relative to A0's own 50M value on the new primary. For example: A1
     passes the span-level gate where A0 fails, by more than 2 paired SE, in each training seed
     separately.

### P2. Draw-induced shift with target-dependent selection on natural rows: disagree with §4 as written

- **Evidence.**
  - [plan §4 steps 3-4] Alignment and the importance weight come first; "Dropout as now" comes
    after.
  - [code] `conditions.py:83-90`: an interval survives with p 0.8 × 0.75 × 0.8 = 0.48, so it is
    removed with p 0.52 (design.md:283 gives the same arithmetic).
  - [inferred] About half of aligned windows, roughly 0.3 of all windows, reach the loss with the
    aligned interval removed. Their natural rows were selected because the source's LN share
    changes there, with weight 3 when the change exceeds 0.1. The selection is a function of
    those rows' own targets: they are the piece's decisions, or the U{0..32} lead rows that
    belong to "the preceding 64 rows". A target-dependent weight biases the learned natural
    conditional. Natural mode would see LN-share changes several times more often than the corpus
    has them, which is the direction of the drift A, B and the recheck report.
  - [data] `artifacts/r2-ml-design-20261003/design.md:303`: no LN-share balancing, because it
    "would change the natural conditional distribution and can recreate an LN controller through
    the sampling distribution". A future balance "requires inverse-probability weights". The plan
    neither cites nor meets this.
  - [inferred from D4's expectations] Per-kind fixed denominators with w = 1 (§5) also move
    gradient toward span rows, which the draw concentrates near informative onsets. With LN rows
    at about 25 % and natural rows at about 50 % of scored heads, an LN-span row weighs about 2×
    a natural row, and a row under both kinds about 4×.
  - [plan §4 D5, §6.6] The rebuilt manifest uses the same draw. Stage 1's natural-NLL guard
    (≤ 0.02 worse) and the selection primary are therefore measured on A1's distribution.
- **Consequence if left.** The draw comparison is confounded with a change of the natural
  distribution. The natural guard is biased toward A1. A1 can pass while natural free-run drift
  gets worse.
- **Change (§4, §5, §6, stage 1).**
  1. Draw dropout first. Align only on intervals that survive. A track emptied by dropout uses the
     unaligned start rule.
  2. Measure informativeness against rows before the window start j, or give the aligned lead
     rows zero weight in L_0.
  3. Weight base-CE rows by p_old(window) / p_new(window). The draw is a known mixture and the
     ratio can be computed per window. Equivalently, the alignment then serves only the condition
     terms, as the brief suggests and the design requires.
  4. Use a fixed, draw-independent natural manifest (current start rule, empty tracks) for the
     natural guard and the selection primary, separate from the condition manifest. Add B's
     `drift_ln_share` and the own-history shift to stage 1's guards.

### P3. L_κ is a reweighted CE, double-counts and is not compositional: amend

- **Evidence.**
  - [plan §2.4, §5] u_j = w_0[j ∉ spans]/N̄_0 + Σ_κ w_κ m_κ[j]/N̄_κ, and L = Σ_j u_j ℓ_j. These are
    the same per-row log-likelihoods as now, with a region-dependent weight.
  - [code] `model.py:4-9, 221-241`: ℓ_j is the 625-way joint action plus the release factors.
  - [inferred] With w = 1, no term carries information the current CE lacks. Stage 1's
    learning-signal change is the draw plus an up-weighting of span rows. The weights depend on
    the inputs (the track), so the conditional optimum does not move, but a natural row's weight
    now depends on N̄, i.e. on the dropout rates the human deferred.
  - Two readings of "must not measure what is not conditioned on κ": (a) rows outside κ's spans,
    which is the plan's; (b) aspects of the decision κ does not govern. LN share is a function of
    the tap-vs-LN choice of each head only. A full-row CE on an LN span is mostly lane-pattern and
    release-position likelihood ([data] `evals.jsonl`: action NLL about 2.0 nats per row). Under
    reading (b), L_LN mostly measures what LN share does not condition.
  - Double counting: a row under k kinds gets k weights. Adding the five Lens dimensions would
    re-weight the LN rows they overlap, so LN training would change whenever style conditions
    are added.
- **Alternatives, weighed.**
  - **Governed factor** (feasibility checked against [code] `common.py:7-18` code table). Per lane,
    a code splits into r = head/no-head plus the held lane's release type, and ℓ = LN-vs-tap of
    each head. The split is bijective, so log P(a) = log P(r) + log P(ℓ | r), with P(r) from a
    logsumexp over the 625 codes grouped by r.
    - LN share: L_LN = −Σ_{j∈Ω_LN} log P(ℓ_j | r_j, s_j, C). It measures exactly the choice the
      statistic is made of, needs no extra forward pass, and keeps releases out.
    - Difficulty: the whole decision, since difficulty governs everything. Release factors are
      owned by the head's span (P6).
    - Style dimensions: the factor each describes (head-mask sequence for Jack, Stream and Trill;
      LN-ness plus releases for LN coordination). Lens labels are section-level and sparse within
      a section (Lens 0004 episodes), so span-level statistic terms fit them better than per-row
      CE.
  - **Composition.** Base = uniform per-decision CE over all scored rows, each row once; this is
    the L20 fix alone and keeps today's per-row weights. Condition terms are additive emphases
    λ_κ L_κ^factor with expected-count denominators; λ = 0 recovers plain CE. The plan's
    equal-contribution weighting becomes one labelled arm rather than the default.
  - **Log-ratio forms.** log p(D|v) − log p(D|v') cancels every aspect that does not depend on v,
    so it fits reading (b) best of all forms. I agree with the plan that maximizing it, or an
    InfoNCE over values, on full rows is unsafe: it can be satisfied by moving mass under v'
    anywhere. Two bounded uses are safe:
    - The null contrast log p(D_j|v) − log p(D_j|∅) on rows under κ, as a diagnostic in §6.1. It
      is the pointwise information the condition carries, needs no choice of v', and dropout
      already trains both sides. Report it by onset, middle and end strata.
    - For the binary LN-ness factor, a monotonicity hinge p(LN | s, v_hi) ≥ p(LN | s, v_lo). It is
      directional and cannot be met by pushing mass to arbitrary decisions. It is a stage-3
      candidate, not a stage-1 change.
  - Condition dropout, the standard practice of training conditional and null from the same data,
    is already kept. It trains both sides by CE and does not maximize the contrast, so it neither
    supports nor refutes the plan's F2 argument.
- **Consequence if left.** Stage 1 tests "draw + up-weighting of span rows" without separating
  them. Each new kind shifts every overlapping kind's weight. The human's constraint is met only
  in reading (a).
- **Change.** §2.3, §2.4 and §5 as above. State T-P1 to T-P3 per factor. Run the stage-1 arms at
  λ = 0 (base CE only), so the stage tests the draw.

### P4. Difficulty: direction 2, F3 and the baseline: amend

- **Reading.** In this project "the response" means the realised dose response of generated
  output (A's usage). Direction 2 ("the model's response is trained towards the loss") therefore
  most naturally asks for a loss on the realised response, which is F3.
  - The plan reads it as F1 with a residual label and puts F3 in stage 3 behind Q4. Q4 is framed by
    the DPO decision, which concerns synthetic preference labels. F3 uses no preference labels;
    its target is the condition's own definition.
  - Q4 is therefore at least partly answered for difficulty. Deferring it is a decision for the
    human, not the planner. Ask the narrower question: "does direction 2 mean a loss on the
    realised difficulty of generated spans, built in stage 2?"
- **F3 cost for star** [inferred from data: audit 6.4 rows/s; [data] `evals.jsonl` free-run median
  338 rows/s; [data] `labels/star_summary.json` 472 s for 82,773 labels on 4 workers ≈ 23 ms per
  label].
  - One sample of a 30 s span (about 190 rows) is about 0.56 s; tiled star about 0.02 s; the
    gradient pass about 0.08 s.
  - A usable score-function estimate needs a per-span baseline: k ≥ 4 samples with leave-one-out
    is about 2.6 s per span, against a 0.37 s CE step. On one update in four that is about 2.7× the
    wall time, not the plan's +30-50 % (§2.3) or "1.5× step cost" (stage 3).
  - One sample with a global baseline has variance the plan does not bound.
- **Cheaper options that keep the realised-response meaning.**
  - (a) Relaxed proxy. Fit g(proxy features) → tiled-star residual on real cells, using the
    mirror-invariant proxies §2.6 already puts in the frame (chord size, same-lane repeat at short
    gaps, held occupancy, LN rate). Train on (g(E_θ[features over S | own history]) − v_res)²:
    one sample per span, gradient through per-row probabilities, like the plan's relaxed LN term.
    Leave hold-length features out of g so short fake holds earn nothing. Track the surrogate gap
    (g against true tiled star on generated spans) as the anti-gaming check.
  - (b) Teacher-forced monotone term on real onset states: E_θ[g | s, r_hi] − E_θ[g | s, r_lo] ≥ a
    margin. No sampling at all.
  - (c) True F3 with k ≥ 4, as calibration on a fraction of updates.
  - Order: (b), then (a), then (c), behind the human's answer.
- **Baseline.**
  - [data] `control/star-prediction-summary.json`: A's ridge uses 23 features including `duration`
    and was fitted on all label intervals (30 s, 60 s, whole song; 74,907 training intervals).
  - §3 refits on 30 s cells only and applies the same b on [0, T). `duration` is then constant in
    the fit, so every whole-song b is an extrapolation.
  - Fit b on the span-length distribution the draw and the interface use (P1.3).
- **Leak of the residual.**
  - Besides the committed proxies inside S (§2.6), the history before S reveals the mapper's
    offset if residuals are correlated within a chart.
  - [data] Per-chart RMSE 0.457 against global 0.528 cannot separate a chart-level offset from
    independent residuals.
  - Stage 0 should measure the within-chart correlation of adjacent-cell residuals. If it is high,
    star needs an informativeness criterion like LN's (requested residual against the preceding
    cells' residual). §4 currently gives star cells weight 1.
- **Engineering, not listed.** [code] `conditions.py:100` rejects star values < 0, and
  `features.py:276` normalises star by /4. Residual requests fail validation until stage 0 changes
  both.

### P5. Stage-1 attribution, stop rule, noise and cost: amend

- **Attribution.**
  - The arms differ in draw, loss weighting and star label set (§9 stage 1).
  - The plan calls A0 "the overnight recipe", yet its config-diff test implies A0 runs the stage-0
    code (presence fix, new frames). State which.
  - Make the arms differ only in the draw: put per-decision weighting, the presence handling
    decided by the human, the frames and the relabel in both. Test per-kind emphasis, if wanted,
    as a third arm or a later one.
- **Stop rule.**
  - §8.4's plateau rule (no improvement > 1 SE over 3 checkpoints after 2 passes, about 25M) can
    stop the arms at different exposures before 50M, which breaks the matched read-out.
  - Turn it off in comparisons; both arms run to 50M.
  - Read 50M only. The 30M point is descriptive, since lr is still mid-cosine there.
  - "The onset-row NLL difference must be negative and > 2 SE" should be the A1 − A0 paired
    difference. A0 already has a teacher-forced value response (section 1), so a within-arm
    reading would pass trivially.
- **Checkpoint noise.**
  - [data] `evals.jsonl`: fit_dev action NLL is 2.0286 at 109.70M, 2.0608 at 110.97M (safe
    checkpoint) and 2.0391 at 111.14M (safe, 170k exposures later). That is about 0.02-0.03 nats
    between checkpoints a few hundred updates apart.
  - This equals the stage-1 natural guard (0.02) and is not captured by a song-group bootstrap SE
    at one checkpoint (§6.6).
  - Read the primary and the guards on the mean of the last three checkpoints. An EMA of weights
    for evaluation would also work, but it is a recipe change for the human.
- **Panel.**
  - 8 charts × 3 seeds gives about 0.06 SE on a paired slope difference [inferred from A's
    0.042-0.045 per arm], so +0.2 is resolvable within one training seed.
  - Training-seed variance is not in that SE. Require the effect in each seed, not in their mean.
- **Cost** [inferred from data].
  - Stage 1's §6.4 suite per checkpoint is 48 natural + 240 whole-song + 96 switch runs at about
    2.4 s each (about 800 rows at 338 rows/s), plus B's own-history calibration. In the recheck,
    A's 264 continuations took 10 min on 2 threads and B's package 32 min ([data]
    `artifacts/r2-recheck-20261005/execution.json`).
  - That is about 20-25 min per checkpoint against about 29 min of training per 4.39M. A 50M run
    takes about 9-10 h, and four take about 38 h, not 22 h. Stage 2 adds 240 star runs per
    checkpoint.
  - Running the conditioned panel every second checkpoint should become the default, not the
    fallback.
- **Budget.** The 61M minimum came under a 158M cosine and does not transfer to a 50M cosine.
  - Support for at most about 4 passes: the fit_dev rise is real over many checkpoints.
  - Two risks run the other way:
    - LN following was still improving at 61M, so an annealed 50M run may leave both arms below
      the absolute gates. This is acceptable, since gates are reported per arm.
    - Up-weighting informative onsets repeats them more often, so those rows are memorised before
      4 average passes. Log train-vs-dev NLL on onset rows.
  - [inferred] Group-uniform sampling makes the "9.6 passes" an average; charts in small groups
    get many more.

### P6. Attribution across span boundaries: disagree with §2.1 and with T-P3b as stated

- **Evidence.**
  - [code] `model.py:265-281` `pair_condition`: every release factor gets frames at the row time,
    at each candidate time, and at the held LN's start time (`fr[:, 2]` from
    `d.start[f.k][f.lane]`). All of them use counters at k. The audit noted the roles
    (r2-range-audit §2).
  - So interval I's value enters p(D_k) for rows k outside I in two ways: (a) releases, including
    the EOS release, of LNs born in I and decided after b; (b) the first row after b, through
    candidates in (t_{k−1}, b).
- **Plan claims that fail.**
  - §2.1 says the value at row j "influences p(D_j | s_j) and nothing else". That does not hold
    for the pointer.
  - T-P3b(i) fails in the FiLM form even after L15: an LN born in an earlier κ interval that does
    not intersect the window, released at a row in Ω_κ, reads that interval's frame.
  - L_0 (natural rows and EOS) also depends on conditions.
- **Statistic.**
  - [code] `labels.py:39` (`TILING_VERSION`: "head ownership, untrimmed tails") and `:78-98`
    (`tile`): the label owns the full bodies of S-headed LNs.
  - The realised star of S therefore depends on releases decided after b, under the next span's
    row condition.
  - §2.3 has F3's gradient flow "through rows in S only". F3 then cannot train those releases and
    credits S's reward to decisions made under another condition.
- **Change.**
  - Own decisions per factor: the action factor of D_j goes to the span containing t_j; each
    release factor, EOS included, goes to the span containing its LN's head time. This matches
    the pointer's birth role and the label's head ownership.
  - State T-P1 to T-P3 per factor. Under the governed-factor LN term (P3), releases drop out of
    L_LN altogether.
  - F3 sampling continues until the last S-headed LN is released, and the owned release factors
    carry the gradient.

### P7. Onsets, informativeness and the leak inside a span: amend

- **D3 is likely missed.** [inferred]
  - The D3 expectation leaves out dropout. P(LN-aligned) is 0.6 × 0.5. P(informative | chosen) is
    about 0.6, given weight 3 and an informative share q of about 0.33 (the audit's row share
    with |Δ| > 0.1, used as a piece share). P(the aligned interval survives) is 0.48.
  - That gives about 1.4 informative onset rows per window, about 0.6 % of scored heads, and
    about 1 % with unaligned windows added. That is far from 3-5 %.
  - The plan's fallback (raise the weight or p_align) amplifies P2.
  - Better levers: dropout-first alignment (P2), and keeping every LN piece that intersects the
    window as its own interval. At the 77-row median a 256-row window holds about 3 pieces, and
    each boundary is an onset or a switch.
- **The criterion is noisy for short pieces.** |Δ| > 0.1 ignores binomial noise.
  - An 8-beat piece is about 30 rows (3.8 rows per beat, audit), about 40-45 heads. At p 0.2 the
    share's SE is about 0.06.
  - Many "informative" short pieces are chance, and the importance weight piles onto them.
  - Use |Δ| / SE or a minimum head count, and report up-weighted pieces by length.
- **Late rows are not uninformative.**
  - Near the span end, v and the counters fix the remaining LN count exactly (v·N − l).
  - [data] Recheck: the teacher-forced per-row response is 0.060 ± 0.010, beside a free-run slope
    of 0.715. The free-run response is produced closed-loop through the counters, by the quota
    controller CE learns on middle and late rows.
  - L2's "satisfiable without reading the value" is true of the middle of a span, not of its end.
  - Strata for §2.4 and §6.1: onset (first 16 rows), middle, end (last 16 rows).
- **The skeleton.** It also predicts part of the LN share (heads thin out where mappers hold).
  "At onset rows the value is the only information about the share" (§2.3, F1) is too strong;
  the criterion should compare v with a skeleton-plus-history prediction where that is cheap.

### P8. Decisions the plan takes that belong to the human

1. **Presence bit.** Removing it (L15, a stage-0 default "unless H-Q2/Q6...") makes between-span
   rows identical to natural rows, which is option (a) of the deferred Q2. Either keep the bit
   until Q2/Q6 is answered, accepting a documented T-P3b(i) exception, or make Q6 a precondition
   of stage 1.
2. **Weights.** Equal expected contribution per term (w = 1) changes natural-row emphasis against
   today's CE, which is a choice inside the deferred natural question. Default to uniform
   per-decision weights (P3).
3. **Membership.** §2.5's "the loss code is the same under every answer" holds only if membership
   follows visibility. Under Q2 (c), announcement, pre-span rows see κ, and whether they belong to
   Ω_κ is part of the answer. List it as a consequence of Q2.
4. **Direction 2.** Reading it as F1 and ordering F3 for star into stage 3 (P4).
5. **60 s cells.** Dropping them narrows the interval lengths the interface promises (P1.3). Ask
   alongside Q11.
6. **Q3.** It presents L28 as weak evidence of oscillation; after 35M it is none (P9).

I agree the plan does not choose natural semantics explicitly, and that it flags guard (i). Guard
(iii) does not depend on Q1: an own-history gap is exposure bias under any natural semantics.

### P9. L28 is no evidence of oscillation after 35M: amend the ledger and Q3

- [data] `evals.jsonl`, using seed-only SE of the 4-chart panel mean (the charts are fixed across
  checkpoints, so the comparison is paired):
  - After 35.1M, 0 of 23 adjacent deltas exceed 2 SE. The largest is |z| 1.75 (109.70M →
    110.97M).
  - Before that, three do: 4.4M → 8.8M (z −2.6), 17.6M → 21.9M (z +5.4), 30.7M → 35.1M (z −2.3).
- Several late "adjacent checkpoints" are safe checkpoints taken at resource trips, 52k-170k
  exposures apart ([code] `train_ce.py:368-369, 411`; [data] `events.jsonl`). Their deltas (+0.081
  and +0.097) are as large as those between pairs 4.39M apart, so the panel delta is sampling
  noise.
- L28 should read "none after 35M; early swings resolved". Q3 option (a) then rests on no evidence
  for the oscillation branch. The recheck's own-history shift (0.129 ± 0.044) is evidence of
  exposure bias, and that is the honest basis for "CE recipe first, own-history term next".

### P10. Smaller corrections

- **Sync (L36).** The mechanism is wrong.
  - [code] `~/ensomi/mutagen.yml:161` sets `maxStagingFileSize: "4MB"`; the pattern exclusions at
    `:182` cover only `/reports/**/*.jsonl`. The r2-runs JSONL files are not excluded by pattern.
  - Per the file's own comment (`:179-181`), the size limit "does not stop the transfer... and
    retries every cycle". The growing `train.jsonl` and `resources.jsonl` are probably re-staged
    and rejected every cycle.
  - Segmenting the logs (§8.3) fixes it. The existing run directory may need a pattern exclusion
    now; that is a sync patch for the main thread, not this plan.
- **Manifest reuse.** [code] `data.py:121-129`: a manifest is reused when only `star_conditions`
  matches. A frozen panel shared across arms needs a version hash covering the draw and the
  labels.
- **Overlapping cells.** 30 s cells at 10 s offsets overlap, and `validate_track`
  (`conditions.py:94-103`) forbids same-kind overlap, so "consecutive with p 0.5" is undefined.
  Pick a phase (0, 10 or 20 s) per draw; that phase's cells then partition the song.
- **D5 conflict.** "Manifest shares within 2 SE of training" contradicts "stratified to ≥ 24 onset
  windows per kind". Keep stratification and drop the equality test, or weight it.
- **Undocumented deviation.** The evaluation panel departs from design.md:535 (128 groups, 24 per
  star band, for both natural and condition suites) without a DEVIATIONS entry, and the free-run
  charts are restricted to K ≤ 600 ([code] `data.py:95, 111`). The plan's 16 charts are still
  well below the design; record it as a deviation with its power cost.

---

## 3. Wrong pieces the ledger lacks

| # | Defect | Evidence | Plan section affected |
| --- | --- | --- | --- |
| W1 | Release factors read conditions at the LN's birth time and at candidate times, across span boundaries; L_0 and EOS depend on conditions. | [code] `model.py:265-281` | §2.1, §2.2 (T-P3b(i) fails after L15), §2.3 F3 |
| W2 | No LN interval longer than 64 beats in training; whole-song and half-song LN requests (A's panel, §6.4, stage-1 primary) are extrapolation. | [code] `conditions.py:20, 26-31, 34-56` | §4, §6.4, stage 1 |
| W3 | Dropout is drawn after alignment: about half of aligned informative windows become natural windows selected on their own targets. | [plan §4 steps 3-4]; [code] `conditions.py:83-90` | §4, natural guard |
| W4 | Design rule that draw balancing needs inverse-probability weights, not met. | [data] `design.md:303` | §4, §5 |
| W5 | Baseline refit on 30 s cells only; A's fit used a `duration` feature over mixed lengths, so whole-song b would be an extrapolation. | [data] `star-prediction-summary.json` | §3, L12 |
| W6 | Star realised statistic owns LN tails decided after b (untrimmed tails). | [code] `labels.py:39, 78-98` | §2.3 F3, §6.2 |
| W7 | Evaluation manifest and free-run panel depart from design.md:535, undocumented; free-run panel limited to K ≤ 600. | [data] `design.md:535`; [code] `data.py:95, 111` | L27, §6.4 |
| W8 | Safe checkpoints at irregular exposures enter `evals.jsonl` and the oscillation evidence. | [code] `train_ce.py:368-369, 411`; [data] `events.jsonl` | L28, Q3 |
| W9 | Checkpoint-level SGD noise in fit_dev NLL (about 0.02-0.03) is not in the selection SE. | [data] `evals.jsonl` 109.70M-111.14M | §6.6, stage-1 guard |
| W10 | `validate_track` rejects negative star values; star normalised by /4. | [code] `conditions.py:100`, `features.py:276` | stage 0/2 |
| W11 | Manifest reuse checks only `star_conditions`. | [code] `data.py:121-129` | §6, stage 1 |
| W12 | The plateau stop can break the matched-exposure comparison. | [plan §8.4, stage 1] | stage 1 |
| W13 | Stage costs omit the in-run evaluation (about 20-25 min per checkpoint). | [data] `evals.jsonl`, recheck `execution.json` | §6.6, stage costs |
| W14 | `|Δ| > 0.1` is within binomial noise for 8-beat pieces. | [inferred] from the audit's rows/beat | §4 D3, §6.1 |
| W15 | Recheck at 61.4M: LN whole-song gate met by the current recipe; own-history shift 0.129 ± 0.044. | [data] recheck tables | stage 1 claim, §7, Q3, Q4 |
| W16 | Oversize r2-runs JSONL hit the staging limit, not a pattern exclusion (re-staged each cycle). | [code] `mutagen.yml:161, 179-182` | L36, §8.3 |

---

## 4. Citation spot-checks

| Plan citation | Claim | Result |
| --- | --- | --- |
| `conditions.py:52` | LN share of the source's heads in [a, b) | held |
| `conditions.py:55, :74` | 1-4 LN and 1-3 star intervals regardless of K | held |
| `conditions.py:62-75` | cached star value | held |
| `conditions.py:83-90` | dropout 0.20 / 0.25 / 0.20 | held |
| `conditions.py:106-134` | `replace_interval` | held |
| `data.py:74-81, :82, :94-119, :108, :111-116` | start rule, independent track draw, manifest, short free-run charts | held |
| `features.py:264-272, :291, :293, :296-302, :308-330` | counters, presence bit, active test, frame channels, tokens | held |
| `features.py:182-199` | history tokens carry no condition features | held |
| `features.py:256-261, :275-276` | scalar `Interval.value`, value normalisation | held |
| `model.py:62-64, :171-175, :283-291` | FiLM zero init, condition after fuse and landmarks, pointer FiLM | held, but §2.1's conclusion omits `:265-281` (W1) |
| `model.py:265-278`, `features.py:350-366` | per-candidate arrays, `make_factor` | held |
| `labels.py:35, :37, :56-75 (:65-66), :103-104, :116-118` | 30 s minimum, cell lengths, hashed j0, raise, `.osu` source | held |
| `train_ce.py:145, :263-273, :266, :269, :317, :378-380, :552-564` | presence-keyed split, window mean, natural free-run, guard without override, CLI | held |
| `train_dpo.py:190-191` | anchor CE uses the window mean | held |
| `runtime.py` snapshot / `initial_swap` / `max_swap_growth_bytes = GiB` | system swap growth | held (`runtime.py:31, 95, 114, 119`) |
| `events.jsonl` six trips (1.41 to 4.15 GB), 167 s and 266 s clusters | | held |
| `evals.jsonl` L28 deltas and L29 minimum (1.9753 at 61.4M, 2.061 at 121.7M) | | numbers held; reading disputed (P9); natural NLL at 61.4M is 1.979, not 1.977 (minor) |
| `resources.jsonl` L37 spikes (23 snapshots > 2.5 GiB, 4.57 GiB at 9.33M) | | held |
| A `star-prediction-summary.json` (R² 0.746/0.753, RMSE 0.528/0.522, per-chart 0.46 ± 0.26) | | held |
| `labels/star_summary.json` (82,773 labels, 472 s) | | held |
| `run.json` 2,177 decisions/s | | held |
| `tests/r2/test_conditions.py:25-37` | FiLM hides future values, tokens show them | held |
| `[d-r2-conditions]` rejected a chart-level scalar | | held (`r2-implementation.md` §d-r2-conditions) |
| `launch.py:24` | `MAX_RESTARTS` | drift: it is at `launch.py:27`; restart logic at `:99-112` |
| `mutagen.yml:161, 179-181` "excluded by pattern" | | drift: line 161 is the staging size limit and 179-181 a comment; r2-runs files are not pattern-excluded (P10) |
| Audit numbers in L6-L9, L11-L12, L20, L24 | | held against `r2-range-audit.md` |

---

## 5. Where the plan is right and I would not change it

- Fixed expected denominators instead of per-batch counts (§5). The argument is correct.
- No relabelling of source decisions with values they do not have (§4). This agrees with A.
- F2 as a maximized full-row contrast is rejected; I add only bounded, factor-level uses (P3).
- F4 is a probe, not a loss.
- A ridge baseline rather than a canonical arrangement (§3). The arrangement's offset varies by
  skeleton.
- The whole-song star stays an interval, not a chart-level scalar, consistent with the human's
  2026-10-03 rejection.
- The infrastructure fixes (§8): a guard on process RSS and available memory, restart budget as a
  rate, segmented logs, data-based stop as an option.
- Per-kind, per-active-row diagnostics, the conditioned free-run CLI, and a fixed panel with
  receipts (§6).
- Deferring token vs FiLM until Q2, and the landmark run until Q9.

## 6. What I could not check

- **Mac-only files:**
  - full `train.jsonl` and `resources.jsonl` (training-loss side of L29; RSS at the swap trips);
  - A's `star-predictions.parquet` (within-chart residual correlation, P4);
  - the cache `index.parquet` (per-chart exposure skew from group-uniform sampling).
- **The recheck job.** Its numbers are read from its output files. Its final message and the main
  thread's check of them did not exist when I read them.
- **Inferred estimates.** The D3 and cost estimates (P5, P7) are inferred. `draw_sim.py` and a
  stage-0 pilot would settle them. No mac job was run.
- **Acceptability** of any generated chart. No renders were viewed.
- **Beatmap Lens 0004-0006**: read only for the episode and section and the ordinal-level facts
  used in P3.
