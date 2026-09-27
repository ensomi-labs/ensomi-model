# Temporal gameplay regression evaluation

The evaluation package supplements scoped Lens inspection with repeatable
whole-trajectory and cross-scale observations. Its immediate targets are known
failure families: sustained single-column overload, lost relief under similar
head counts, excessive occupation under a plausible LN fraction, flattened
phrasing, and control-range drift. It returns evidence and witness times rather
than a single score that could hide one failure behind another improvement.

## Exact prefix and scope semantics

[`ChartTrace`](../../src/ensomi_model/research/gameplay_evaluation/temporal.py)
reads complete rows from BOS through an explicit exclusive coverage boundary.
Replay validates their physical execution. A local boundary does not close an
LN: observed holding is integrated through coverage while release count stays
unchanged. Every named `Scope` is half-open. Pre-scope attacks, holding and
recovery remain available; a control change cannot reset them.

Each scope reports H/head/LN/release counts, LN-head fraction, exact mean held
columns and any-held fraction. It also records per-column and all-column time
spent freely recovered after .25/.5/1 seconds without an action. Recovery credit
begins at the actual previous action plus that duration, not at the scope start.
Adjacent scope credits therefore add without a fictitious restart.

`LN_timing_relations` supplements these amounts with complete hold durations,
duration relationships between distinct LN onset groups, simultaneous-group
spread and a second coordinate based on the actual H sequence. H coordinates
linearly interpolate observed head timestamps; they are not beats or a timing
grid. Comparing milliseconds and H spans helps separate irregular H intervals
from release-span choices. Open holds remain censored, real tails may extend
beyond the selected scope, and a predecessor group before that scope remains
available. Largest-change contexts are inspection witnesses, not automatic BAD
labels. Short or unequal holds are never rejected by this analysis, and smaller
variation is not a quality objective. The
[native LN and pressure analysis](native_pattern_failure_analysis_zh.md)
defines the measurements, causal comparison and interpretation limits.

`LN_interactions` describes what happens while a hold continues, using the
[typed interaction observer](../../src/ensomi_model/research/gameplay_evaluation/hold_interactions.py).
At a row, a continuing LN started earlier and does not release on that row.
Count its relations to new TAP heads, new LN heads and other LN releases,
separating same-hand and opposite-hand pairs under the canonical mapping.
Release/head coincidences are separate relations: a hold ending at this row
does not also count as continuing through its new heads. Co-start hold/release
pairs identify one member of a simultaneously started group releasing while
another continues; this is a subset of the hold/release channel.

These are pair counts, not independent actions or calibrated strain. One TAP
under two continuing holds contributes two pairs but one TAP to the event
denominator. Reports retain raw counts, rates per second, per-hold observed
companion-row quantiles and bounded origin/time witnesses. A missing TAP
denominator is `null`, not zero. Larger or smaller values are not inherently
better: anchors, LN streams and independent releases can all be appropriate.

Only actions inside the half-open scope contribute. Incoming holds retain their
true origins; holds without an observed release stay open in this observation.
The observer does not read later tails, even when the trace contains them, so a
private-prefix report matches the same scope of a longer trace. Event/pair counts
add across adjacent scopes; involved-hold counts need not, because the same hold
can cross both. Existing duration descriptors retain their separate retrospective
endpoint semantics.

A focused counterexample swaps two LN tails while preserving all head times,
head types, total LN fraction, duration/H-span distributions, total occupied
time and any-held fraction. The typed relations still distinguish a long LN
accompanying TAPs from one accompanying new LNs. This catches a lost arrangement
relationship that the matching marginals cannot identify; it is not a rule
declaring either synthetic chart BAD.

Trailing per-column attack peaks use .5/1/2/4/8/16-second windows, retain prefix
attacks and return witness clocks. Successive same-column attacks below 20 ms
have explicit endpoint witnesses. These attack checks count TAP and LN press;
releases remain distinct. The existing
[sustained-response envelope](sustained_response_planning.md) owns calibrated
corpus-reference excess and can be attached separately. This package does not
replace it with another invented strain scale.

These distinctions catch failures that an average cannot:

- Identical H times and head count can concentrate demand on one finger.
- Identical LN-head fraction can conceal much longer simultaneous holding.
- Fewer heads can coexist with much less actual recovery.
- A new control range can inherit already accumulated attack pressure.

## Consecutive head membership

The scope field head_recurrence records whether a finger keeps participating in
consecutive head-bearing rows. TAP and LN press both count; release-only rows
do not reset this head sequence. Each attack receives its current prefix age:
one at a new run, two at its second consecutive H group, and so on. A missing
column in an H group resets that column's age.

The [observer](../../src/ensomi_model/research/gameplay_evaluation/head_recurrence.py)
returns exact age histograms, quantiles, repeated-head counts per column and
timed witnesses. The repeated fraction uses heads with a preceding H as its
denominator; heads at the first H have no such opportunity. Scope boundaries
retain earlier membership. Events at or beyond the exclusive end are not read,
so prefix ages and witnesses agree between a private prefix and the same scope
of a longer trace. Event counts add across adjacent scopes; quantiles do not.

Witnesses also report companion heads by column, rows with companions and the
recurrent column's share of all heads in the run. These contextual counts include
the pre-scope portion of an incoming run and are not additive scope workload.
Continuing holds on other columns are reported separately as hold/head pairs,
per-column counts, affected head rows, maximum simultaneous other holds and
their original starts at run entry. Simultaneous companion releases are also
separate; a hold closed at that H is not counted as continuing. These facts use
all preceding rows, including pure releases, but never future endpoints.

Zero companion heads therefore does not imply that the other fingers are free.
A regression fixture has identical recurring heads with either three fingers
free or three continuing holds, yielding zero versus twelve hold/head pairs.
The same repeated-finger timing can carry different LN obligations; neither
case is assigned a quality label by the observer.

There is no gap-based reset. A slow repeated note and a fast repeated note can
have the same age, so witnesses include real span, median/max HH gap, incoming
run status and whether a later H was observed to end membership. This is not
a complete Jack/Stream classifier, physiological state or sampling constraint.
Intervening H groups can split a musically related repeat.

A regression fixture keeps H times and all four column totals identical between
finger rotation and four eight-head column blocks. Both traces have zero
sustained attack excess under the stated D4 references, yet their maximum ages
are one and eight. This makes an organizational difference visible when the
threshold response is silent. The
[full-R1 and control study](full_row_learning_and_difficulty_response.md)
shows actual cases where lower stars coexist with more recurrence; corpus
quantiles are descriptive, not automatic rejection thresholds.

## Multi-scale phrasing observations

For each half-window length .5/1/2/4/8/16 seconds, compare adjacent left/right
intervals at a 250-ms analysis stride. Each vector contains H rate, four column
attack rates, four held fractions, LN-head rate and release rate. Signed
right-minus-left differences preserve the direction and time of change.
Column-share transfer is also reported separately from total activity.

Both halves must lie in the same named control range. A short range is explicitly
insufficient for larger scales; other ranges cannot be pooled to manufacture
coverage. Exact event counts and occupied-time integrals supply the interval
features. The stride samples contrast locations, so it is not an exact maximum
detector; rare attack peaks have their separate event-time calculation.

The measurements locate changed pacing and texture. Higher contrast or greater
finger redistribution is not inherently better: steady streams, anchors and
chordjacks are valid. Ranked examples and labeled organization determine which
changes matter, and each control condition keeps its own comparison.

## Audio correspondence is a diagnostic

[`audio_correspondence`](../../src/ensomi_model/research/gameplay_evaluation/alignment.py)
uses the confirmed canonical full-song Mel clock: 128 bins, 10-ms hop, 40-ms
window, centers at `20 + 10*i` ms. Eight log-power bands and their positive
frame-to-frame changes describe spectral development. Their means over 1/4/16
seconds are paired with the chart's interval-feature vectors on the same clock.

Within each scope and scale, center and standardize feature dimensions, then
measure linear centered kernel alignment. This adapts the representation-
correspondence statistic discussed by
[Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html); their neural
representation results do not establish musical quality. The implementation uses
feature-space products rather than allocating a time-by-time kernel matrix.

Report zero-lag alignment beside five within-scope circular shifts of the chart
features. Shifts preserve marginal values while disrupting temporal pairing.
They are descriptive comparators, not exchangeable samples or p-values.
Constant inputs or fewer than eight windows are explicitly unevaluable.

A loudness-following generator can score well while producing poor arrangements.
Deliberate dumps or steady passages can score poorly while being valid. Low
alignment must not become an automatic BAD label, and this statistic must not
be optimized as a musical-quality reward. Its role is to help distinguish a
changed temporal relationship from a changed global mean, with source comparisons
and actual listening/Lens judgments supplying the missing interpretation.

## Publication deadlines

[`publication_report`](../../src/ensomi_model/research/gameplay_evaluation/publication.py)
reads the actual wall time and settled-through audio clock of each publication.
Private forecasts contribute no playable buffer. Given a required lookahead, it
computes the minimum startup delay that would avoid starvation in that observed
trace, identifies the limiting publication, and tests a supplied startup delay.
Inclusive native coverage makes `previous_coverage + 1` the next unknown time.
An incomplete trace cannot certify full playback, even with a large startup.

For publication $i$ at wall time $w_i$, prior settled time $g_{i-1}$ in
milliseconds, and lookahead $L$, the observed startup lower bound is

$$
s_{\min}=\max_i\left[w_i-\max\left(0,
\frac{g_{i-1}+1-L}{1000}\right)\right]_+.
$$

The zero clamp covers initial buffer preparation before playback. The initial
coverage is -1; a trace's wall-clock origin and excluded startup work must be
declared. This catches a late dense passage even when total generation is faster
than playback. It is an observation-based bound, not a guarantee under another
machine load or decoding policy.

## Use and qualification status

```python
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.gameplay_evaluation.report import evaluate_scopes

trace = ChartTrace(complete_rows, coverage_ms=audio_duration_ms + 1)
report = evaluate_scopes(trace, [Scope("override", 64000, 96000)], mel=full_mel,
                         identity=verified_run_identity)
```

The report format is `gameplay-temporal-evaluation/v1`. Caller-owned identity
should record audio/source/model byte hashes, controls, seed and generation
completion. Reports retain separate scopes and contain no pooled pass verdict.
`feature_order` names every contrast coordinate. Artifact arrays are diagnostic
evidence, not new human annotations or source labels.

Eleven focused tests cover concentrated versus distributed attacks at equal
timing/counts, smeared LN tails at equal head fractions, carried recovery and
open holds, flattened temporal activity at equal totals, cross-boundary short
attacks, constant-input handling, canonical-Mel/chart alignment, startup buffer
requirements, local stalls hidden by good average speed, and incomplete traces.
These verify algorithmic distinctions. Real ranked exceptions and historical
generated failures must still calibrate regression rules before the framework
can qualify a model. Semantic style and player experience remain separate
evaluation dimensions; neither is certified by these tests.

## Historical replay evidence

An initial replay reparses actual osu bytes for 41 reports in 17.88 seconds:
four ranked sources, ten source-matched generated conditions and nine outputs
each from the initial actor, bounded planner and failed response-trained actor.
Model/audio identities, seeds and named ranges remain separate.

The existing sustained-response mean is reproduced exactly: .0275164 for the
initial actor, .0004245 for the planner and .1071601 for the failed response fit.
Its new Zenithfall failure reaches 9.5 attacks/s on one column over four seconds.
This catches a native regression despite that fit's improved training-bank cost.
The cost was optimized, so it cannot independently establish musical improvement.

Occupation measurements reproduce STYX's phrase changing from 24.34% any-held
time in the source to 95.37% in the actor. Multi-scale correspondence supplies a
different diagnostic: on Classic Pursuit, aligned-minus-shift-median values are
.183/.212/.171 for the source and .042/.082/.044 for the actor at 1/4/16 seconds.
These are small developmental comparisons, not significance or quality thresholds.

Pressure and correspondence improvements need not agree. On Zenithfall, the
planner's sixteen-second correspondence difference averages .1238 versus the
initial actor's .1654 despite a large pressure improvement; on Hysteric it rises
from .2966 to .4271. Keeping those dimensions separate prevents one successful
objective from being presented as a complete repair.

The memory prototype's actual publication trace also passes the two-second
lookahead check with startup .9583 seconds; its observed minimum is .6521 seconds.
The inherited core trace needs .3814 seconds and uses startup .5379 seconds.
These clocks begin with loaded weights and cached Mel, including model audio
encoding. They are not complete client cold-start measurements.

The evaluation also locates a previously unreviewed Max Burning Stream failure:
the actor's whole-chart proxy is 3.8617, but column 2 receives 33 attacks in
(19174,23174] ms, or 8.25 Hz over four seconds. All six source/generated Lens
pages on [18174,25175) are subsequently read. They confirm an extended same-column
sequence, initially almost alone and then with occasional chord accompaniment;
other columns are free. The ranked source distributes changing groups over the
columns. This agent-confirmed regression witness is retained separately from
human annotations, along with its controls, seed 273100 and exact output bytes.

Three previously inspected ranked counterexamples are also replayed. Happy Love
Expert reaches 7.5 Hz over four seconds with positive excess about .00594;
Extra Mode reaches 6.5 Hz with excess .000805; the human-prominent Stream
reference has zero excess. None has a below-20-ms attack in the inspected scope.
The valid positive-excess examples prevent a universal percentile-to-BAD rule.
Reference context and sustained organization remain necessary for interpretation.

## Localizing sustained-pressure episodes

[`sustained_attack_witnesses`](../../src/ensomi_model/research/gameplay_evaluation/witnesses.py)
localizes the existing corpus-envelope excess at exact attack and expiry clocks.
For each column and window, consecutive positive-excess intervals form an
episode. Actual zero-excess time separates episodes; no arbitrary gap merging
or event-count horizon is used. The report retains each window's full integral
and the largest episodes, with explicit omitted counts.

Each episode records its duration, peak time, contributing history start and
simultaneous per-column attack rates and held fractions. The peak's contributing
trailing window is also reported separately: pressure can persist after the
sequence that caused it, so episode-wide peer averages alone can hide idle peers
during that sequence. Thus a concentrated
sequence with idle peers can be distinguished from repeated chords or attacks
beside occupied fingers before Lens review. A control boundary clips the
observation and changes the requested reference when appropriate; it never
forgets incoming attacks. Boundary-clipped witnesses are marked explicitly.

```python
from ensomi_model.research.gameplay_evaluation.witnesses import sustained_attack_witnesses

witnesses = sustained_attack_witnesses(trace, scope, corpus_envelope, stars=4)
```

Five focused tests verify concentration at unchanged timing/counts, scope
partition additivity against the existing exact excess integral, relief-separated
episodes, retained totals when witnesses are truncated, and chord/held-peer
context. An unknown difficulty has no invented reference capacity. The witness
integral is the existing diagnostic, not an independent validation score; its
added value is localization and inspectable context. It does not automatically
classify real ranked repetitions as bad patterns.

[`pressure_review_contexts`](../../src/ensomi_model/research/gameplay_evaluation/review.py)
ranks retained episodes across every response scale by integrated excess and
includes the peak's contributing history in each proposed inspection range.
Reviewing only the largest four-second peak can miss sustained eight- or
sixteen-second load, including passages where all columns remain busy rather
than one isolated column repeating. The original scored scope remains attached
when an inspection context extends into its incoming history. Overlapping
contexts are allowed; selection does not merge control scores or label quality.
A regression fixture has no four-second excess but sustained sixteen-second
excess, ensuring this failure family still receives an inspection context.

Owners: `20260927-gameplay-evaluation-v1` and
`20260927-audio-history-memory-v1/preflight-v3`.
Historical replay source: `720457b8651d40095c2247b5992c3318b2cb5ced`.

## Diagnosing next-event timing

[`first_event_law`](../../src/ensomi_model/research/gameplay_evaluation/waiting.py)
evaluates native-ms Bernoulli hazards along a fixed prefix's no-event branch.
It retains both per-ms first-event probability and the right-censored no-event
atom. Reports include survival at declared elapsed horizons, a median when
reached, and the restricted mean `E[min(wait, horizon)]`. This is not an expected
event count in a generated continuation: after the first event, history changes.

Paired comparisons hold audio, observation clock and prefix fixed unless a named
intervention changes one of them. Total variation compares timing mass and the
censoring atom; restricted Wasserstein distance measures displacement in ms.
Thus equal probability of an event somewhere in two seconds cannot hide a shift
from long waits to immediate attacks. Four focused tests cover native-sampler
agreement, censoring, invalid/forced clocks and equal-amount timing differences.

This instrument localizes timing sensitivity to controls or history. It does not
assign BAD labels, certify learned control semantics, or replace complete native
rollouts. A hypothetical input lesion must be labeled as such. In particular,
removing another known style changes the request and need not preserve the law.

## Rejecting gains that move the failure elsewhere

The [matched memory fit and H-base diagnostic](audio_memory_joint_fit.md) provide
an actual regression example. Replacing only the fitted H base reduces nine-case
pressure excess from .38230 to zero and whole-star MAE from 1.33452 to 1.16212.
It still fails: H counts have median ratio .318, three startups exceed two
seconds, and the before/override control ranges undershoot difficulty despite an
improved restored range. Lens and occupation measurements find continued LN
texture where fewer attacks could otherwise be mistaken for recovery.

Future comparisons should retain these as independent evidence channels. A
passing pressure objective or average proxy cannot override a failed scoped
control, publication or inspected-organization guard. Report partial gains with
their tradeoffs, leave unreviewed dimensions unreviewed, and retain ranked positive
exceptions to automatic BAD classification. The framework supplies repeatable
facts and witnesses; each experiment must declare its own promotion criteria.

## Executable native qualification

[`run_qualification`](../../src/ensomi_model/research/gameplay_evaluation/qualification.py)
executes generation, verified osu export/reparse, independent scoped measurements
and publication checks in one call. It supports `controlled-audio/v1` and
`controlled-audio-memory/v1` checkpoints. Its packaged entrypoint is:

```bash
uv run --extra mps python -m ensomi_model.research.gameplay_evaluation.qualification_hydra \
  checkpoint_file=/path/to/model.pt checkpoint_sha256=CHECKPOINT_SHA256 \
  plan_file=/path/to/plan.json plan_sha256=PLAN_SHA256 \
  output_dir=/path/to/fresh-run
```

The canonical settings are
[`native_gameplay_qualification.yaml`](../../src/ensomi_model/configs/hydra/native_gameplay_qualification.yaml).
Hydra is confined to startup; training code can call `run_qualification` directly
with a `QualificationConfig`. The runtime uses the requested number of CPU
threads. Weights and cached Mel are ready at each measured generation start;
complete model audio encoding and actual row publication are included.

A plan has format `native-gameplay-qualification/v1` and a nonempty `cases` list.
Each case supplies a unique `key`, `seed`, complete-audio `asset` containing
`audio_file`, `audio_sha256`, `mel_file`, `mel_sha256`, `duration_ms`, a list of
`ControlSpan` dictionaries in `controls`, and named half-open `scopes`. Each scope
may declare requested `stars`, `ln_fraction` and `maximum_excess_seconds`.
Unknown targets are not assigned invented values. A diagnostic phrase inside a
whole-song amount request should leave its local amount target unknown unless
that phrase is itself a requested control scope.

LN amount gates additionally require a matching, completed declared request
extent with no conflicting override. An original whole-song request interrupted
by a later announcement does not retroactively impose its total ratio on the
already committed prefix. A restored fragment likewise is not a new total-amount
scope unless explicitly requested as one. Such fragments still receive separate
occupation, recovery and realized-ratio reports, but their `ln_fraction` target
must be null. An explicit override's complete requested duration can be assessed.
This prevents evaluation from silently reintroducing the prefix-balance preference
being investigated. Difficulty response proxies remain separately declared criteria.

Optional `switch` supplies `announce_after_ms` and the new `ControlSpan` as `span`.
The runner publishes through the announcement before updating controls and checks
that prior rows stay unchanged. Before, override and restored scopes must be
declared separately. Optional `head_times_ms` enables a fixed-H diagnostic, which
is explicitly identified and cannot combine with a live switch. Native candidate
assessment must not substitute that diagnostic for generated timing.

An optional plan `recovery` changes the row/release profile, while HH must match
the checkpoint's H-capacity law. `ln_feedback=false` disables only the projected
amount controller; direct model controls and its analytic requested-ratio tilt
remain active. The checkpoint, policy, plan, audio and Mel identities, source
revision and executable working-tree status are recorded. Paths and assets are
caller supplied; a fresh clone does not contain the private evaluation corpus.

When `envelope_file` and `envelope_sha256` are supplied, pressure witnesses and
all-scale review contexts are generated. A declared pressure bound without an
envelope or explicit scope difficulty is rejected instead of silently ignored.
Other default gates check complete mechanical export, below-20-ms attacks over
all coverage, startup after thirty rows and required lookahead, service time,
playback deadlines, difficulty error and LN amount error in their declared scopes.
Their limits are explicit startup settings. No pressure quantile is silently
converted into a universal BAD threshold.

The output retains `settings.json`, `plan.json`, identities, per-case osu files,
actual `publications.json`, scoped `evaluation.json`, `cases.json` and the final
`result.json`. Numeric failure does not remove a completed case or skip later
cases. Runtime failure retains its reason, stops further generation and identifies
unrun cases. The resource guard observes task footprint on macOS and RSS elsewhere;
these are different ledgers. An incomplete trace cannot pass publication checks.

`run_status=complete` means execution completed. `candidate_status=failed` blocks
numerically failed or incomplete candidates; the CLI exits 2. A numerically clear
candidate instead returns `review_required`, with `promoted=false` and pending
semantic review; the CLI exits 3, so an ordinary success exit cannot accidentally
stand for completed qualification. Declared `review_contexts` and automatically
located pressure contexts remain unreviewed until actually inspected. No module
test, successful command or missing review record creates a playable qualification.
