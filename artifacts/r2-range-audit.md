# R2 range audit: condition, memory and loss ranges

Fresh Opus worker, 2026-10-05, answering [q-r2-ranges](r2-average-and-control.md#q-r2-ranges). Report kept as written; the main thread checked `data.py:82`, `model.py:194-211`, `train_ce.py:266` and `conditions.py:55` against the claims. Line numbers refer to `78aa22c`.

Read-only audit of `src/ensomi_model/r2` at `78aa22c`. The overnight run's frozen copy
(`artifacts/r2-runs/r2-ce-overnight-20261004/code/`) is byte-identical for `data.py`,
`conditions.py`, `features.py`, `model.py`, `labels.py`, `state.py`, `sampling.py`, `train_ce.py`,
`candidates.py`, `common.py`, `cache.py` and the TCN (`research/bounded_typed_continuation/temporal.py`).
Run config: FiLM, landmarks, levels 8, window 256, `star_conditions` auto, which resolved to on
(receipt: labels complete).

Tags: **[code]** checked in code, **[measured]** measured on the cache, **[inferred]** reasoning
only.

Measurements (no model run; mac jobs, scripts in `~/ensomi/.sync/cp/scratch/r2-range-audit/`):

- `20261005-043513-r2-range-audit`: `measure.py`, 3000 training draws through `Corpus.draw`
  (fit_train, star on, seed 471 as in the run), plus the run's `fit_dev_manifest.json`.
- `20261005-043836-r2-range-audit-2`: `measure2.py`, the same 3000 draws. It adds informative
  onset rows, coverage by song length and star cell coverage.

## Verdict

- **The condition range is misplaced relative to the loss range: confirmed.** Conditions are
  drawn over the whole song, independent of the scored window. The rows where a condition says
  something history does not (a scored onset whose LN share differs from the preceding content)
  are 0.18 % of scored heads.
- **The memory range is not misplaced: the hypothesis is refuted.** In training the TCN and the
  landmarks read the full prefix from song start, exactly as in free-run. Landmarks exist for
  almost every scored row (median 8 visible). The memory is probably unused because the
  511-token TCN receptive field already covers the whole prefix on 54 % of scored rows.
- **Loss range: one separate skew.** The window-mean loss weights decisions by 1/L, so the end
  of the song and EOS are overweighted. This is not linked to the three observations.

## 1. Loss range (which decisions are scored)

- **Draw [code].** Group uniform, then a chart uniform within the group (`data.py:69-71`).
  Start j is 0 with p 0.125, `max(0, K-255)` with p 0.125, else uniform in [0, K]
  (`data.py:74-80`). Stop is `min(j+256, K+1)` (`data.py:81`).
- **Scored decisions [code].** Head decisions k in [j, stop). EOS (decision K) is scored exactly
  when stop = K+1 (`model.py:221-241`). EOS is never a history token (`features.py:182-185`).
  The history is `N = min(stop-1, K)` tokens and the state before k is `enc[k-1]`
  (`model.py:197-207`). No off-by-one found.
- **In time [measured].** Median K is 944 rows (p10 419, p90 2533), at 6.4 rows/s and
  3.8 rows/beat. A full 256-row window spans a median of 38 s (p10 22, p90 56), or 65 beats
  (p10 38, p90 115).
- **Weighting [code + measured].** The loss is the mean over a window's decisions
  (`train_ce.py:266`), and windows are averaged over the batch (`train_ce.py:269`), so each
  decision weighs 1/L. 25 % of windows are shorter than 256.

  | Decisions | Share of decisions | Share of loss weight | Ratio |
  | --- | --- | --- | --- |
  | EOS | 0.17 % | 0.70 % | 4.2x |
  | Last 64 rows | 10.6 % | 18.8 % | 1.8x |

- **fit_dev manifest [code].** It uses the same start rule and `draw_track` (`data.py:107-108`),
  with one window per group drawn without replacement (`data.py:99`). It draws the same way as
  training.

## 2. Condition range

**LN intervals [code].**

- Units: the song [0, T) is partitioned from `beat(0)` into pieces of 8, 16, 32 or 64 beats,
  each converted to ms bounds (`conditions.py:34-41`).
- Only pieces with at least one head are kept. 1 to 4 are chosen (`conditions.py:55`),
  consecutive with p 0.5.
- The value is the source chart's own LN share over the heads in [a, b) (`conditions.py:52`).
- A row is active when its head time t_k is in [a, b) (`features.py:293`). EOS (t = T) is never
  active.

**Star intervals [code].**

- Spans (`labels.py:56-75`): up to 4 consecutive 30 s cells, up to 4 consecutive 60 s cells, and
  [0, T). Units are absolute ms. The cell start `j0` comes from a hash of the chart's inputs
  (`labels.py:66`), so a chart gets **the same cells in every draw**.
- Value: the star rating of the objects whose heads lie in the cell, tiled to 240 s
  (`labels.py:78-110`). It is computed from the original `.osu` (`labels.py:118`), not from the
  cache representation the model emits.
- Draw: the whole song with p 0.10, else one cell length and 1 to 3 cells of it
  (`conditions.py:59-75`).
- The value matches the span it is attached to: the same `(a, b, v)` tuple is used
  (`conditions.py:62-75`). It is the tiled steady-state difficulty of the whole cell, not the
  difficulty of the scored part.

**Dropout [code].** The whole track is dropped with p 0.2, a kind with p 0.25 and an interval
with p 0.2 (`conditions.py:83-90`).

**Independence [code].** `draw_track(chart, labels, rng)` never sees j or stop (`data.py:82`,
and the same at `data.py:108`).

**Sizes [measured].**

- LN interval: median 77 rows (p10 21, p90 254), or 25 beats. That is about a third of a window.
- Star interval (cells and whole song): median 278 rows (p10 140, p90 591). Most are longer than
  a window.

**Overlap with the scored window [measured].**

- Active share of scored heads: 13.5 % for LN and 23.9 % for star. This reproduces the earlier
  14.4 % and 24.7 % within sampling noise.
- 55 % of windows carry an LN track, but 49 % of those have no active scored row. For star the
  figures are 52 % and 39 %.
- Only 28 % of windows have any active LN row, and 31 % any active star row.

**Onsets and informativeness [measured].**

- 32 % of LN-active scored rows belong to an interval that began before j. Its onset rows are
  teacher-forced history, not scored.
- Only 15 % of active rows are within 16 rows of their onset. The median is 63 rows after the
  onset.
- The value differs from the LN share of the 64 rows before the onset by a median of 0.058. 33 %
  of active rows differ by more than 0.1 and 8.7 % by more than 0.25. The median difference from
  the whole chart's share is 0.035.
- Rows that are both within 16 rows of the onset and differ by more than 0.25: **1.3 % of
  LN-active rows, 0.18 % of scored heads.**

**Coverage by song length [measured].** The interval count does not scale with K
(`conditions.py:55,74`).

| Song length | LN-active share | Star-active share |
| --- | --- | --- |
| K < 600 | 21.2 % | 32.4 % |
| 600 ≤ K < 1500 | 13.1 % | 24.5 % |
| K ≥ 1500 | 7.8 % | 16.0 % |

**Star [measured].**

- Local cells relative to the same chart's whole-song label: median -0.09, mean -0.21,
  p10 -0.60.
- 19 % of star-active rows come from the whole-song label. A peak-weighted song-level value is
  applied to every row.
- Of star intervals that overlap a window (whole song included), a median of 49 % of their head
  rows are scored, and 51 % have less than half scored.

**Visibility and causality [code].**

- FiLM (the overnight run):
  - The row frame is taken at t_k only (`model.py:177-184`).
  - It is filled only when t_k is inside an interval (`features.py:293-304`).
  - Elsewhere there is only a per-kind presence bit (`features.py:291`). It says a track of that
    kind exists somewhere, not where.
  - The model has no look-ahead to an upcoming interval and no memory of a finished one. History
    tokens carry no condition features (`features.py:182-199`).
  - The release pointer also sees frames at candidate times and at the hold start
    (`model.py:265-278`).
- Token form: every interval is visible at every row, with offsets and started/active flags
  (`features.py:308-330`). It sees future and past intervals.
- Counts use committed decisions before k only (`features.py:264-272`). `remaining` uses the
  given head skeleton.
- This is the specified design (`tests/r2/test_conditions.py:25,46`). No leakage found.

## 3. Memory range

- **TCN [code].** Kernel 3 with dilations 1 to 128 gives a receptive field of 1 + 2·255 =
  **511 tokens** (`temporal.py:31-37`). That is about 80 s at the median 6.4 rows/s, or two
  windows.
- **Training history [code].** Training encodes the **full prefix from row 0** for every window
  (`model.py:196-198`). There is no crop, and every token is a real source row.
- **Landmarks [code].** The landmarks are TCN outputs at token positions 0, 64, 128, ... < N
  (`model.py:208`), visible when p < k (`model.py:210`). Free-run uses the same cadence and
  visibility: prefix marks at `sampling.py:48-54`, online marks at `sampling.py:82-87`, read
  before decision k at `sampling.py:59-63`.
- **Counts [measured].**
  - Visible landmarks per scored head: median 8 (p10 2, p90 25), mean 12.
  - Landmarks older than the current TCN window (p < k - 511): median 0, mean 6, p90 17.
  - Heads with k > 511: 46 % of scored heads in training, 42 % of decisions in free-run
    (uniform over the chart). 80 % of drawn charts have K > 511.
- **Redundancy [code + measured].** For k ≤ 511, `enc[k-1]` already spans the whole prefix, so
  landmarks add no information on 54 % of scored rows. On the other 46 % they add content more
  than about 80 s old.
- **Look-ahead [code].** The query sees 16 future head gaps, head-count densities up to 32 beats
  ahead, and the remaining time (`features.py:24-25,233-247`). Training and generation are
  identical here because the skeleton is given.

## Ranked mismatches

| # | Severity | Mismatch | Tags |
| --- | --- | --- | --- |
| 1 | High | Condition intervals drawn independently of the scored window; informative onsets are almost never scored | code, measured |
| 2 | High (star) | Star spans and semantics do not match the window: fixed hashed cells, cells longer than a window, song-level value applied per row | code, measured |
| 3 | Medium | Conditioning diagnostics measure the wrong range: conditioned/natural is split by track presence, the manifest has few active rows, free-run is natural only | code, measured |
| 4 | Medium-low | Interval count independent of song length, so coverage falls from 21 % to 8 % (LN) | code, measured |
| 5 | Medium-low | 1/L weighting overweights EOS (4.2x) and the last 64 rows (1.8x) | code, measured |
| 6 | Low | Star label computed from the original `.osu` and tiled to 240 s, not from the emitted representation in place | code |
| — | None | Memory range: training reads from song start like free-run (hypothesis refuted) | code, measured |
| — | None | fit_dev manifest draws, EOS and off-by-one, landmark cadence, look-ahead | code |

### 1. Condition range independent of loss range (High)

- **Code does.** `draw_track` places 1-4 LN pieces and 1-3 star cells anywhere in the song,
  ignoring j (`data.py:82`).
- **Measured.**
  - LN is active on 13.5 % of scored heads and star on 23.9 %. Half of the windows that carry an
    LN track have no active scored row.
  - 32 % of active LN rows come from intervals whose onset lies in the unscored teacher-forced
    history.
  - Only 15 % of active rows are within 16 rows of the onset.
  - The value is the source's own share and is close to the preceding content: median
    |Δ| 0.058, and only 8.7 % of rows have |Δ| > 0.25.
  - Rows that carry information history lacks: 0.18 % of scored heads.
- **Interaction with FiLM [code].** The frame appears only from the onset row on. On most scored
  active rows, the interval's own committed rows, given as history and as the count channels,
  already reveal the share.
- **Effect on results [inferred].** CE rewards ignoring the condition: almost every scored active
  row is predictable from history. In training the model never sees a request that contradicts
  the preceding content, and those are exactly the requests a steering test makes. This fits
  "LN responds only partly". It also weakens star (see 2).
- **Plausible behaviour.** The scored window should contain the intervals' onsets and most of
  their rows, and enough requests should differ from the preceding content.
- **Minimal fix.**
  1. Pass j and stop to `draw_track`.
  2. With some probability (for example 0.5), choose a piece or cell first and set
     j = onset_row - U[0, 32]. Otherwise keep the current rule.
  3. Keep only intervals that intersect [t_j, t_stop), and draw their count per window, not per
     song.
  4. Optionally oversample intervals whose share differs from the preceding 64 rows by more than
     0.25.
  5. Apply the same change to `build_manifest`.

  Expected [inferred]: a large share of scored rows active, and most active intervals with
  their onset scored.

### 2. Star spans and semantics do not match the window (High for star)

- **Code does.**
  - Labels exist only for fixed hashed 30 s and 60 s cells per chart and for the whole song
    (`labels.py:56-75`). The same chart always gets the same cells.
  - The value is the tiled steady-state star of the whole cell (`labels.py:78-110`).
  - The whole-song star (p 0.10) is fed to every row, although it is set by the hardest section.
- **Measured.**
  - Intervals span a median of 278 rows, longer than a window.
  - Half of the overlapping star intervals have less than half of their rows scored. The
    unscored rest includes rows after the window that the model never sees.
  - Local cells sit a median 0.09 (mean 0.21, p10 0.60) below the whole-song label, so the
    whole-song value on a typical window overstates local difficulty.
- **Effect [inferred].** On a given scored row the star value is a noisy statistic of a span the
  model mostly cannot see. Also, the star rating is largely determined by the given head
  skeleton (density and timing) plus the teacher-forced history, which leaves little residual
  information for the condition. This was not measured. Together with 1, this fits "star has no
  effect".
- **Minimal fix.**
  1. Label spans at window scale, placed relative to the window (for example a cell starting at
     the window start and lasting 15-30 s), with a random offset per draw rather than a hash.
  2. Give the whole-song star its own global channel, not a per-row interval value.
  3. Before more training, measure how much of the cell star the skeleton alone explains (R² of
     star from head-time features). If it is high, star conditioning cannot be learned from
     these labels.

### 3. Conditioning diagnostics measure the wrong range (Medium, diagnostic)

- **Code does.**
  - `Stats.add` labels a whole window "conditioned" when its track is non-empty
    (`train_ce.py:145`).
  - The in-run free-run report is natural mode only: `continue_chart` is called without a track
    (`train_ce.py:317`).
- **Measured.**
  - In training, 70 % of windows count as conditioned, but only 47 % of their decisions are
    inside an active interval.
  - In the overnight run's fit_dev manifest:

    | Manifest item | Value |
    | --- | --- |
    | Conditioned windows | 42 of 64 |
    | Windows with active LN | 14 (1691 rows, 11.1 % of 15188 heads) |
    | Windows with active star | 22 (3444 rows, 22.7 %) |
    | Active share of conditioned decisions | 43 % |

- **Effect [inferred].** The logged natural-vs-conditioned NLL cannot show a conditioning effect.
  The manifest's active-LN sample is small and clustered in 14 windows.
- **Minimal fix.**
  1. Split NLL per row by active kind: `Stats.add` has `out.ks` and the track.
  2. Add a conditioning evaluation on onset-aligned windows with counterfactual values
     (`replace_interval`).
  3. Free-run a fixed chart with a requested track.

### 4. Interval count independent of song length (Medium-low)

- **Code does.** It keeps 1-4 LN pieces and 1-3 star cells whatever K is (`conditions.py:55,74`).
- **Measured.** LN-active share is 21.2 %, 13.1 % and 7.8 % for K < 600, 600-1500 and ≥ 1500.
  Star is 32.4 %, 24.5 % and 16.0 %.
- **Effect [inferred].** Conditioning is learned mostly on short songs. A full-song steering test
  on a long chart is outside the training coverage.
- **Minimal fix.** Subsumed by 1: draw per window. Otherwise scale the count with K.

### 5. Window-mean loss overweights the song end (Medium-low)

- **Code does.** Per-decision weight is 1/L (`train_ce.py:266,269`), with 25 % short windows.
- **Measured.** EOS takes 0.70 % of the loss weight against 0.17 % of decisions (4.2x). The last
  64 rows take 18.8 % against 10.6 % (1.8x).
- **Effect [inferred].** The song end and EOS are overfit relative to free-run, where each
  decision counts once. This is not tied to star, LN or memory. The DPO anchor CE
  (`train_dpo.window_ce`) inherits it.
- **Minimal fix.** Either normalise by the total decisions in the batch, or move short windows
  back (j = min(j, K+1-256)) so every window has 256 decisions. The second also removes the
  EOS-only windows (0.17 % of draws).

### 6. Star label source and tiling (Low)

- **Code does.** Labels are computed on the original `.osu` objects (`labels.py:118`). The model
  emits the cache representation (2 ms row merge, snapped releases). Tiling to 240 s gives a
  steady-state value, not the cell's difficulty in place.
- **Effect [inferred].** A small label-noise term.
- **Minimal fix.** Compute on `objects_from_decisions` of the cache. Low priority.

### Not mismatched: memory range (hypothesis refuted)

- **Code does.** The training forward pass encodes all rows 0..stop-2 (`model.py:196-198`).
  Landmarks are read from song start with the same cadence and visibility as free-run
  (`model.py:208-210`, `sampling.py:48-63,82-87`).
- **Measured.** Median 8 visible landmarks per scored head. 46 % of scored heads in training have
  k beyond the TCN field, against 42 % of decisions in free-run.
- **Likely cause of "unused" [inferred].** On 54 % of scored rows the TCN read already covers the
  whole prefix, so landmarks add nothing. On the rest they add content more than about 80 s old,
  which is probably worth little for the next row once 511 rows are seen. A zero-initialised
  readout that adds noise would then cost NLL, which fits the 39.49M ablation.
- **Test (not a range fix).**
  - Compare NLL with and without the readout on k > 511 rows only.
  - Or cut the TCN to 6 levels (field 127) so that the landmarks carry the long range.

### Not mismatched: other checks

- The fit_dev manifest draws the same way as training (`data.py:107-108` vs `data.py:74-82`).
- EOS inclusion and history length have no off-by-one.
- Look-ahead is the same in training and generation.
- Condition counts are causal.

## Observations and likely causes

| Observation | Range-related cause | Other cause [inferred] |
| --- | --- | --- |
| Star conditioning has no effect | 2 and 1: spans longer than and offset from the window, value of unseen rows, song-level value per row, few informative scored rows | Star largely fixed by the given skeleton plus history; diagnostics (3) could not see a small effect |
| LN responds only partly | 1: 0.18 % of scored heads carry information history lacks; FiLM is blind before onset; coverage falls on long songs (4) | Requests that contradict the history never occur in training |
| Landmark memory unused | None (refuted) | Redundant with the 511-token TCN field |
