# Joint audio-memory fitting: native generation results

The [audio/history memory prototype](audio_history_memory.md) completes a matched
384-update joint fit and 84 native/fixed-timing evaluation exports, but does not
qualify as a playable improvement. It reduces pressure relative to a continued
baseline fit while regressing against their common initialization. Loaded-model
publication speed remains within the measured budget. The main unresolved
problem is the generated distribution, including excessive H occurrence and
poorly supported control requests, rather than generation throughput alone.

## Matched comparison

Both fits start from the same 4,583,985-parameter core2500 model. The memory arm
has 7,616,517 parameters, adding complete-audio attention and separate H/R/R1
history readers. Their zero-output initialization preserves the common model's
law. All audio, timing and row modules remain trainable. R1 owns counts, columns,
TAP/LN choices and release subsets; H owns head-event times.

The fixed input ledger contains 768 examples from 417 charts and 383 song groups:
one ranked population and one human-annotation-centered example per update.
Eligible ranked TRAIN charts have native difficulty 2–6 and complete audio lasting
60–480 seconds. Both arms see identical accepted examples, controls and ordering.
Each scores a 32-second interval while recomputing differentiable full-song Mel
features and current-weight prefix histories. The row loss reconstructs the
native recovery and LN-feedback policy; H/R use complete waiting-time losses.
No training NLL endpoint selection is performed.

Inherited parameters use learning rate 3e-5; new parameters use 3e-4. AdamW weight
decay is 1e-4 and gradient clipping is 1. Each update weights its two microbatches
equally. Human concept/assessment cells are balanced, with original annotation
scopes retained. Numeric controls use whole-chart targets with independently
attempted 8/16-second local overrides and 15% family dropout. Styles are dropped
as a whole family, a limitation examined below.

All three evaluation arms use the same recovery profile: 60 ms between same-column
attacks, 50 ms release-to-head and 40 ms head-to-release. The final value admits
real 42–43-ms LN tails in a supporting-Tech training source. This common
comparison differs from the selected runtime's 50-ms head-to-release profile;
the unfitted arm is also not the previously reported actor-128 model.

On an Apple M5 with 24 GiB, Python 3.10 and PyTorch 2.11/MPS, baseline fitting takes
1,227.90 seconds. Total memory-arm accounting is 3,058.49 seconds, including
discarded resource attempts. Standard attention and checkpointed groups of 128
training queries retain the full calculation. Following a process-footprint stop
after 115 valid updates, fresh processes restore model, AdamW and CPU/MPS RNG state
every at most 32 updates. The final ledger contains every update 1–384 exactly
once. Resumed segments stay below the 18-GiB process guard. This establishes a
recoverable bounded fit, not a complete MPS allocator explanation or bitwise
cross-process reproducibility claim.

## Native rollout and control qualification

Each arm exports 28 cases: nine prominent-Stream requests on three songs and three
seeds, four other style requests, one live control change, six developmental
conditions, four reserved validation sources, and four genuine-style examples
with fixed source H. Fixed-H diagnostics remain separate from native H results.
The reserved songs are excluded from this fit; they are not certified unseen by
the inherited checkpoint.

Pressure $J$ is the exact time integral of squared positive per-column attack-rate
excess relative to a difficulty-conditioned ranked-corpus envelope, averaged over
.5/1/2/4/8/16-second windows and summed over columns. It includes incoming history.
The [response definition and ranked exceptions](sustained_response_planning.md)
explain why positive excess is a review signal, not an automatic BAD label.
Whole-star error is a proxy against the requested four stars. Both measurements
are retained separately from semantic and musical judgments.

| Measurement | Unfitted | Continued baseline | Memory |
| --- | ---: | ---: | ---: |
| Mean pressure $J$, s, nine Stream cases | .04447 | .78553 | .38230 |
| Worst pressure $J$, s, nine Stream cases | .28577 | 2.31211 | 1.72508 |
| Mean absolute whole-star error, nine Stream cases | 1.15214 | 1.51711 | 1.33452 |
| Slowest first thirty rows, s, all 28 cases | .90983 | .86192 | 1.59063 |
| Slowest two-second publication service, s, all 28 cases | .38319 | .49556 | 1.14145 |

The candidate fails the frozen pressure guard, at most
`max(.001, 1.1 * unfitted mean J)`, and the star-error guard, at most the unfitted
error plus .15. Improvement over the continued baseline is therefore insufficient.
These descriptive results come from three songs; they do not estimate a
population-wide effect or isolate every architecture component.

All 84 exports complete and match their reparsed osu rows. No below-20-ms
same-column attacks appear in evaluated scopes. Actual publication traces meet
the two-second-lookahead replay check and the declared two-second startup/service
bounds. Timing begins with loaded weights and cached Mel and includes model audio
encoding; waveform decoding, Mel creation and model loading are excluded.

The live request changes from difficulty 3 / LN fraction .2 to 4.5 / .6 on
[64000,96000) ms, announced after publication through 63999 ms. Earlier controls
resume afterward without resetting gameplay state. Difficulty proxies retain
their own scopes:

| Scope / requested difficulty | Unfitted | Continued baseline | Memory |
| --- | ---: | ---: | ---: |
| Before / 3 | 2.49169 | 4.76831 | 3.17298 |
| Override / 4.5 | 3.65430 | 5.65558 | 5.20984 |
| Restored / 3 | 4.16956 | 6.21590 | 5.09766 |

All three memory LN fractions are within .10 of their requests, but restored
difficulty misses its ±1 guard. A successful amount control cannot cancel this
failure. No whole-song score pools the three requested ranges.

## What the regressions reveal

On the same Zenithfall seed, H counts rise from 3,582 in the unfitted model to
6,390 in the continued baseline and 5,514 in memory. Total heads are respectively
5,003, 6,892 and 5,968. The baseline therefore nearly doubles timing events while
reducing mean chord size from 1.40 to 1.08 heads per H. H occurrence contributes
substantially to the dense result; R1 is already reducing chord size. This is not
an impossibility proof, nor a reason to transfer R1's count ownership to H.

The largest memory Hysteric pressure episode lasts 47.662 seconds, from 252419
to 300081 ms, on the sixteen-second response scale. At its peak, the contributing
window (248806,264806] contains column attack counts [93,97,96,104] with no held
columns. Its episode H rate is 20.06/s. The chart's four-second maximum is only
7 Hz and occurs elsewhere. Thus long-scale distributed overload can dominate
even when the most visible short-scale single-column peak is modest. The
[review selector](gameplay_regression_evaluation.md#localizing-sustained-pressure-episodes)
now includes contributing contexts from every response scale. This particular
long-scale context has measured evidence but no completed Lens judgment.

## Breathing and held-texture inspection

Eleven whole-song audio/gameplay overviews and 45 Lens pages are inspected. The
Lens pages cover the source, baseline and memory versions of four fixed
developmental passages. A ranked source is one valid arrangement, not the unique
target. These are agent visual judgments without audio listening or player tests;
uninspected reserved/style and supplemental contexts have no semantic pass.

- **Max Burning:** during a source held voice on [70879,72142) ms, the source
  adds no heads. Baseline adds 11, memory 6 and unfitted 6. Memory has a visible
  short pause relative to baseline but does not recover the sustained voice.
  This is a limited local pacing improvement, without a clear unfitted advantage.
- **Classic Pursuit:** on [88589,93589), source repeats mixed TAP/LN groups.
  Memory instead changes from a TAP-dominant run to predominantly short LNs. Its
  aggregate LN fraction is closer to source, but 76 heads exceed source's 47 and
  baseline's 59. Similar amounts do not establish similar temporal organization.
- **STYX HELIX:** on [1800,11800), memory reduces heads from baseline's 91 to 54,
  but any-held time rises from .6703 to .8478, versus source's .2434. Fewer
  attacks do not mean more unoccupied recovery. An alternate LN arrangement may
  be valid; improved musical response is not established here.
- **Blizzard Heights:** on [40342,56342), baseline and memory each have 122 LN
  heads. Memory's lower LN fraction comes from increasing total heads from 140
  to 155. Any-held time drops from .9695 to .9523 but remains above source's
  .6931. Both fitted outputs largely fill a passage where the source has more
  separated attacks and held voices. No overall improvement is established.

Together these checks do not establish the required two convincing developmental
gains. They also demonstrate why amount, occupied duration, recovery and local
organization need separate evaluation channels.

## Verified control-support gaps

An audit of the actual scored training exposure, resolving partial overrides
with the native `ControlSchedule`, gives 21,722.180 seconds with no known styles
and 1,272.035 seconds with all five styles known. Singleton masks receive 157.880
seconds for Tech, 66.565 for LN coordination, 32.114 for Jack, 8.919 for Stream and
9.889 for Trill. No two-, three- or four-style masks occur. Repeated draws count
repeatedly; prominent Stream as the only known style consists of two copies of
one 3.184-second source interval.

Known style scopes last .260–10 seconds, with median 4.900 seconds. Evaluation
Stream requests last 144–358 seconds. The per-field encoding sends both scope
clocks to each factor, including an unbounded `asinh(seconds)` coordinate.
Balancing concept/assessment cells therefore did not balance request masks or
duration features. Missing styles must remain unknown; assigning them an absent
label would alter the requested condition. Human annotation boxes must not be
extended as if they supplied long-range gold labels.

This covariate gap is confirmed; its causal contribution to native density remains
unresolved. Generated-H feedback, teacher-forced training and oversampling of
annotation-centered passages remain competing explanations. A bounded next-H
survival probe with fixed audio and prefix can separate scope-feature sensitivity
from value/mask effects and H-history sensitivity before another large fit.
The memory checkpoint is not promoted. No conclusion that attention cannot help
follows from this joint comparison.

## Fixed-prefix timing diagnosis

A follow-up holds complete audio, observation clocks and H prefixes fixed across
the three checkpoints. Twelve contexts come from four predetermined relative
clocks on each of three failed memory generations. Three additional contexts
come from genuine prominent-Stream annotations with all five style fields known.
Each query evaluates the exact first-event distribution over the next two seconds,
including censoring, rather than treating unchanged-history hazards as expected
rollout counts. The [waiting-law evaluator](gameplay_regression_evaluation.md#diagnosing-next-event-timing)
also reports paired distribution distances. Dense-prefix queries agree with
native append/cache scoring to a maximum absolute logit difference of 9.54e-7.

The control-support gap does not explain most immediate sensitivity on this slice.
For memory, replacing the long Stream scope with a ten-second scope increases
restricted mean waiting time by a median 1.0%; capping only the unbounded style
clock coordinates at `asinh(5 seconds)` changes it by 1.2%. No context reaches
the predeclared 20% material-change threshold. The other checkpoints have similarly
small changes. Removing all style requests produces a larger median 10.2% change,
but changes the requested condition. On the three human prefixes, retaining only
the already-known Stream label changes the median by .2%. These are conditional
diagnostics, not proof of correct long-scope or singleton control behavior.

Removing alternating older H events while preserving the latest four, last H and
physical support increases memory's median wait by 27.0%; six of twelve contexts
reach 20%. This is a substantial but heterogeneous history effect, below the
predeclared eight-of-twelve consistency rule. Nulling only its attention read
increases the median by 8.2%, with one material context. Neither observation
supports deleting history or attention as a complete repair.

The native H logit has two additive terms: an audio/control base and a bounded,
elapsed-time-gated residual that reads H history. Source likelihood constrains
their sum; it does not uniquely identify the two terms. A second probe exchanges
their outputs between fitted and unfitted checkpoints at the same inputs. Each
term retains its own checkpoint's complete-audio encoder; learned feature vectors
are not transferred between encoders.

| Memory model, twelve fixed generated prefixes | Median relative change in restricted mean wait |
| --- | ---: |
| Actual memory law versus unfitted law | -17.2% |
| Unfitted base + memory residual, versus actual memory | +25.0% |
| Unfitted base + memory residual, versus unfitted law | +5.5% |
| Memory base + unfitted residual, versus unfitted law | -19.6% |

Restoring only the base reaches a 20% wait increase in eight of twelve contexts.
Restoring only the residual does not recover the unfitted law. With common
unfitted-survival weighting over query clocks, the median fitted-minus-unfitted
base change is +.521 logit units, while the residual change is -.072. The separate
three-human-prefix cohort has changes +.572 and -1.077. Different songs, controls
and prefixes confound a direct comparison between those cohorts; the observations
show that compensating components can behave differently across contexts.

This identifies a consistent local base contribution to faster next events.
It does not establish its contribution to complete-chart pressure: every chosen
event changes the next history and R1's downstream choices. Bounding a logit
residual also does not bound difficulty or require rhythmic relief. A complete
native substitution comparison is needed before changing the training objective
or treating a two-model combination as useful generation.

The input probes score 339 laws in 22.31 seconds; component exchanges score 150
in 10.80 seconds on one CPU thread, each below 1 GiB sampled process peak. These
are instrument runtimes, not generation latency. One earlier zero-record attempt
failed because parameters created inside inference mode lacked native-cache
version counters; using a no-gradient context preserves those counters without
changing query math. Owner: `20260927-head-control-hazard-probe-v1`, `run-v2` and
`components-v1`, source `fc641aa740a7528decbb0b5b522aa91039d06ff1`.

## Reproduction identities

Model source: `7fdad44331c75da035fdb9720d0875aa02bdd732`; evaluation-only descendant:
`ef49d63a8c328eac7fa5e41dd8273cfbc435e204`. Study owner:
`20260927-audio-memory-joint-fit-v1`, final preparation `preparation-final-v2`.
Both final checkpoints contain 384 updates.

| Input or result | SHA-256 |
| --- | --- |
| Common core2500 | `0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8` |
| Continued baseline | `a6916403daf6c658f0952d1181703732156f4ac37a8e5b773007c5dc4e2c5c24` |
| Memory endpoint | `7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81` |
| Frozen training draws | `6ed9e3609d31d19e459eb4703766e55f1c49d95466ffdbc008f22fc205ce77fd` |
| Qualification plan | `cf19396e4977f6d3bd4857359ae516391838d15ae967f720779bed59408780bd` |
| Numerical comparison | `de54acb1677de9ac95051432a11ddd42145e71f3f4ce088a1fb6ab3c1b28484d` |
| Visual review record | `cea2cad049653fc920fe0f0a5fc6ab803eedbbbc3e7a09cc1e0fea4da2b07018` |
| Control exposure audit | `f5ba7eb47f7a139d6bce096e12541e19e95a75a837cfadbc2ad0df0893502beb` |
