# Head-time decomposition: what supplied head times carry

The [decomposed problem](../formulation/notation.md#decomposed-problem-given-head-times)
takes the head times as given and generates the arrangement on them. It rests
on [three assumptions](../formulation/notation.md#assumptions). This document
records the measurements bearing on them and a verdict on each. The
formulation defines the problem and does not depend on these results.

Head times fix head-row density and its time profile exactly. Note density
also depends on chord multiplicity, which the arrangement stage controls.
Head-time features strongly predict difficulty in this corpus. For other
chart-level properties, the tested predictors explained up to 41 % of dev-set
variance; the remaining variation is unresolved by these probes. Human
agreement on chord placement exceeds the tested local timing predictors, and
LN agreement is higher within one mapper than across mappers. The sources of
these gaps remain unresolved. In a four-song check, head times from the legacy
audio timing model compressed the differences in predicted difficulty between
songs; the small sample gives direction only.

## Evidence and its location

The measurements were made on the R2 line, branch `r2/train`, against the
corpus cache of that line, on 2026-10-08. The notes, with the scripts' outputs
condensed, are on the `relay-notes` branch under `artifacts/`:

- `r2-represent-skeleton-20261008.md`: the tables and sections cited below as
  the skeleton measurements;
- `r2-represent-step1-20261008.md`: the split of human pairs by mapper;
- `r2-representation-20261008.md`: the reading of that split and the direction
  that the assumptions behind supplied head rows be examined.

Statements are marked **measured** when a script computed them on the stated
sample and **inferred** when they are a reading of measured numbers. The sample
is the corpus of 12,272 charts of the cache with at least 256 rows (11,118
train, 1,154 dev) unless a section states another. A row means a head row
here, and "rows-only" means a predictor that sees only the head times, not the
lanes. Difficulty band and star rating are the corpus's labels for a chart.

## What the head times fix and the probes predict

### Density, profile and difficulty

Head times fix the head-row count in every time interval, hence head-row
density and its profile exactly. Note density is head-row density multiplied
by mean chord multiplicity (heads per row), which the arrangement stage
controls.

- Measured: on the 99 panel charts, the median ratio of between-part variance
  in log head-row density to that in log notes per second is 1.00 in the source
  charts and 0.87-1.06 in the generated arms. This is an empirical variance
  ratio in the tested sample, not a fraction of note density fixed by head times.
- Measured: a rows-only gradient-boosting predictor reaches difficulty-band
  accuracy 0.70 (majority class 0.31, within one band 0.99) and star R² 0.83.
  The tested head-time features strongly predict difficulty in this corpus.
- Measured: 69 % of near-identical-row pairs (1,558 of 2,243) lie in different
  bands. Near-identical head times coexist with different difficulty labels;
  this comparison does not isolate the contribution of chord multiplicity.

### Chart-level properties

R² on the dev charts of a tested rows-only predictor of chart-level
statistics, using richer row features with gradient boosting:

| statistic | R² | statistic | R² |
| --- | ---: | --- | ---: |
| heads per row | 0.41 | LN share | 0.21 |
| rows with 3 or more heads | 0.31 | LN length | 0.23 |
| jack rate | 0.27 | pattern entropy | 0.41 |
| LN level (held) | 0.22 | loop | 0.27 |
| rows with 4 heads | -0.02 | hand share | 0.00 |

- Measured: the tested rows-only predictors explained up to 41 % of dev-set
  variance in these properties; hand-share R² was 0.00. Adding the true
  difficulty band raises heads per row to 0.66 and jack to 0.53. The remaining
  variation is unresolved by these probes.
- Measured: given the chart's measured property levels, the tested local timing
  predictor explains about 40 % of the within-chart variation of chord size
  and jack rate, and 11-14 % of LN variation. It explains little hand and trill
  variation (1 % and 3 %).
- Measured: two humans on near-identical head times agree on the exact lane
  pattern of a row at 0.21 (chance 0.10) within a band, and 0.39 (chance 0.19)
  up to left-right mirror. These pairs show variation in lane choice at
  near-identical head times; they do not measure its attainable range at fixed
  head times.

### Chord and LN placement between humans

Within-chart correlation over matched rows, median per chart. Pairs are
different charts of one song whose head times align at F1 of at least 0.95 at
20 ms, copies excluded.

| comparison | chord size r | LN-head r |
| --- | ---: | ---: |
| Two humans, same band (615 pairs) | 0.79 | 0.65 |
| Two humans, different band (1,556 pairs) | 0.77 | 0.66 |
| Rows-only predictor against the human, 300 charts | 0.59 | 0.32 |
| Rows-only predictor against the source, 99 panel charts | 0.56 | 0.30 |
| R2 (56M) against the source, same head times | 0.30 | 0.08 |

- Measured: human agreement on chord placement exceeds the tested local timing
  predictors (0.77 against 0.56-0.59). The source of that gap remains unresolved.
  Inferred: audio, song structure, mapper conventions or timing information
  missed by these predictors could contribute. The probes do not separate them.
- Measured, by creator: 568 of the 615 same-band pairs (92 %) are two
  difficulties by the same creator. Splitting by creator, 47 pairs have
  different creators:

  | | same creator | different creator | rows-only | R2 |
  | --- | ---: | ---: | ---: | ---: |
  | LN-head r | 0.671 | 0.451 | 0.32 | 0.08 |
  | LN-level ICC | 0.68 | 0.20 | | |
  | chord size r | 0.792 | 0.735 | 0.59 | 0.30 |

- Measured: LN-head and LN-level agreement are lower in the 47
  different-creator pairs than in the same-creator pairs; chord agreement
  remains higher than the tested rows-only predictor in both groups.
  Inferred: mapper conventions may contribute to the LN agreement difference.
  Preferences for which lanes, how much and how long to hold could be part of
  chart identity. Neither that explanation nor a contribution from audio or
  song structure to chord agreement is isolated by these comparisons.

### Timing of the head rows

- Measured, earlier audit (2026-09-23, not repeated here): two humans mapping
  the same audio agree on head times at a median F1 of 0.785 at 20 ms. Head
  times agree well but not exactly.
- Measured: head F1 of the legacy timing model against the source is 0.70 at
  20 ms and 0.82 at 40 ms, on the runs described below.

## Head times from a timing model

Measured on 4 songs of the cache, each with 6 timing configurations from the
2026-09-23 audio-skeleton integration runs, with the source's beat grid. The
sample is small, so the numbers give direction only.

- The row count against the source has median ratio 1.12 (range 0.50-2.18).
- The difficulty band predicted from the timing model's head times agrees with
  the band from the source's head times in 25 % of runs. The predicted star
  rating shifts by a median 0.86 SD.
- The spread across songs of the rows-only predictions shrinks to 0.17 of its
  size for star rating, 0.43 for heads per row, 0.48 for LN level and 0.65 for
  LN share. The timing model's head times all resemble band-3 head times. The
  band-5 song has a predicted star of 5.3 from its own head times and 3.2-4.4
  from the timing model's.

R2 (56M, two seeds each) on source head times and on timing-model head times:

| song (band) | rows with 3 or more heads: source / R2 on source / R2 on timing | jack rate: same | LN share: same |
| --- | --- | --- | --- |
| a9dd (5) | 0.15 / 0.14 / 0.03-0.05 | 0.23 / 0.24 / 0.06-0.14 | 0.13 / 0.17 / 0.13-0.36 |
| b23b (2) | 0.04 / 0.08 / 0.06-0.15 | 0.11 / 0.14 / 0.17-0.21 | 0.06 / 0.11 / 0.14-0.33 |
| c56c (4, LN-heavy) | 0.07 / 0.01 / 0.00-0.03 | 0.14 / 0.11 / 0.05-0.09 | 0.40 / 0.28 / 0.005-0.25 |
| a01b (2) | 0.00 / 0.04 / 0.01-0.04 | 0.15 / 0.15 / 0.04-0.08 | 0.09 / 0.54 / 0.01-0.21 |

Inferred: the chord and jack changes on timing-model head times suggest a
compression of difficulty toward the middle in this tested R2 setup. The beat
grid is the source's in this check, so the effect of a generated grid is not
measured.

## Verdicts on the assumptions

These verdicts concern the tested no-audio R2, timing model and predictors.
The broader decomposed formulation already includes audio and chart identity;
these results do not contradict that formulation.

1. **Audio provides timing cues for the rows.** Human head-time agreement is
   consistent with shared cues, but does not establish that audio uniquely
   determines the rows. The legacy timing model changed timing-based difficulty
   predictions and R2's arrangements in the 4-song check (direction only).
   This limits conclusions about that setup, not the decomposed formulation.
2. **Given rows support choreography and control.** The tested no-audio R2 and
   local timing predictors fall below human placement agreement. The predictors
   leave much chart-level variation unresolved; their residuals do not identify
   chart identity or audio as the missing information. The mapper-convention
   explanation for LN agreement is inferred from the creator split, not a
   measured cause.
3. **Separating the concerns lets the arrangement stage achieve some things
   without the others.** Supplied head times give the arrangement stage exact
   head-row rhythm, density and density profile without generating those times
   or reading audio. Note density still depends on chord multiplicity. The
   observed difficulty predictability and placement agreement describe the
   tested corpus and models, not guarantees for other arrangements or inputs.

For control, predictability across existing charts does not establish the
range attainable by changing an arrangement at fixed head times. These probes
do not measure that range for note density, difficulty or the other properties.

## The implementation measured

For orientation only, the implementation measured here, the R2 line, works as
follows; these are properties of that implementation, not of the problem. It
takes the head times and the beat grid from an existing human chart. At each
head row it decides which lanes receive taps and long-note heads, together with
the releases of open holds in the gap before that row at grid positions, so
releases can produce close-only rows at times that are not head times. After
the last head row, a terminal decision releases the remaining holds. It does
not use audio.

Measured: R2's chord and LN-head agreement with the source (0.30 and 0.08)
approximately matches the squared correlation of the rows-only predictor
(0.56² is about 0.31, 0.29² is about 0.08). On the 99 panel charts its
chart-level statistics (heads per row, jack, LN level, LN share, loop) correlate
with a rows-only prediction at 0.48-0.64 and with the source at 0.12-0.41.
Inferred: these correlations are consistent with R2 sampling around patterns
captured by the tested rows-only predictor. They establish neither equivalence
with an ideal rows-only sampler nor that the tested models use all available
timing information. The remaining variation is unresolved by this comparison.

## Limits

- Whether the agreement between two humans comes from the audio, the song's
  structure, a mapper's conventions or timing information missed by the tested
  predictors was not separated. The creator split measures agreement
  differences, not their causes.
- The timing check covers 4 songs and the source's beat grid; a timing model's
  own grid was not tested.
- The measurements use one corpus cache and contain no melody-to-lane test.
