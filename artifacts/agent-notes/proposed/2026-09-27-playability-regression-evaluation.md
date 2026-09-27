# Agent Note: Reusable temporal playability and musical regression evaluation

Note ID: 2026-09-27-playability-regression-evaluation
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: b69d360bd044ff11327c98b50a4d514f11728470
Scope: Reusable exact-time chart observations, scoped regression comparisons, multi-scale pressure/texture and audio-linked diagnostics
Related: 2026-09-27-audio-history-memory, 2026-09-27-four-star-musical-phrasing, 2026-09-27-player-response-frontier

## User priority and ownership

The user explicitly prioritizes evaluation algorithms that distinguish actual
iteration gains and prevent previous faults, especially failures Lens alone
does not spot. This is authorized implementation work during the architecture
experiment. Do not reduce evaluation to NLL, stars, a single variation score or
new machine style labels. Lens remains a semantic inspection tool; this work
adds long-horizon, cross-scope, paired-realization and whole-system diagnostics.

The audio-memory integration remains active, not replaced: its first nine new
checks and affected owners pass36 tests. Its learning/profiling is not yet run.
Evaluation is developed before any architecture quality promotion.

## Selected first implementation

Create a model-independent research/gameplay_evaluation package consuming
complete timed rows, explicit observed coverage, complete canonical Mel when
available, and named control ranges. Carry pre-range attacks and open LNs into
observations. Do not close holds at local boundaries or reset recovery there.
Retain separate range reports and local witnesses; do not pool distinct requests.

First algorithms:

- Exact source-clock interval counts, occupied-key integration and all/finger
  recovery opportunities after actual attacks/releases. A lower head count or
  matching LN-head fraction cannot hide continuous holding.
- Multi-scale attack, release, held-duty and finger-distribution contrasts on
  paired adjacent time windows. These locate flattening or relocated overload;
  greater contrast is not automatically better.
- Music/chart temporal correspondence with auditable Mel band/flux features,
  per-scope multi-scale kernel alignment and within-scope shift comparisons.
  Constant/insufficient inputs are explicitly unevaluable. This is a diagnostic
  of correspondence, not a sufficiency test for an encoder or a universal
  musical-quality reward. Deliberate dump/steady-pattern positives prohibit
  using low alignment alone as BAD.
- Reuse the independently fitted sustained-attack response/envelope for overload
  rather than invent another incompatible strain score. Keep absolute short-
  attack violations, corpus outliers and musical-comparison evidence distinct.

Synthetic fixtures test discriminating invariants, not human quality: identical
H/counts but one-finger accumulation; unchanged LN fraction with lengthened tails;
unchanged total activity but flattened phrases; exact recovery carry across a
scope boundary; known audio/response relation versus shifted alignment.

Then run on the named four-star source/generated study and historical reproduced
long-jack failures. Preserve real ranked exceptions, mixed results and cases where
metrics cannot decide. A reusable regression manifest should identify audio,
source/model bytes, conditions, seeds, ranges and expected fault category. It
must not silently turn agent observations into human labels. Multi-seed/song-
group evaluation and dense/startup service traces remain required for promotion.

No new trained checkpoint may be selected using only its optimized training cost.
This first library is not a complete playability oracle. It is intended to make
specific prior failures repeatably visible, with auditable coverage and actual
witness times, before an expanded quality comparison is designed.


## Result Log: initial reusable implementation and real historical replay

Implementation138a1f5c777e8d968eb4d54e080fbce3c5de5c07 adds exact-clock ChartTrace,
Scope, multiscale adjacent-window contrasts, Mel/chart correspondence and a
non-pooled report. Eight focused tests pass, including complete Mel/chart-clock
alignment and explicit unevaluable constant inputs. Product guide:
 docs/research/gameplay_regression_evaluation.md.

Exploratory run at clean720457b8651d40095c2247b5992c3318b2cb5ced:
uv run --extra mps python artifacts/joint-audio/20260927-gameplay-evaluation-v1/run.py.
Handle21230 is terminal, exit0. In17.877s, it reparses actual osu bytes for41
reports: four ranked sources, ten matched generated conditions and27 historical
Stream cases (initial actor, bounded planner and failed response-trained endpoint,
nine each). Identified output/input hashes, independent named scopes and complete
numeric evidence are retained under this artifact owner's results directory.

Recomputed mean J reproduces earlier evidence exactly: initial .0275164272,
planned .0004245142, failed response .1071601211. Worst costs .1061111,.0022319,
.7035150. Response Zenithfall0's four-second peak is9.5Hz. This detects the known
sustained-load regression, but J was itself optimized and is not an independent
semantic metric. The newly replayed four-star Max Burning Stream output reaches
8.25Hz on one column despite whole proxy3.86; it warrants a targeted witness
inspection. No generated label is inferred automatically.

Independent occupation/texture facts reproduce STYX's phrase .2434 source vs
.9537 actor any-held time, and Blizzard .6931 vs.9451. Multi-scale audio/chart
aligned-minus-shift-median diagnostics are lower in the actor than the source
at all three measured scales for the four matched unknown-style comparisons.
For Classic Pursuit, source .183/.212/.171 versus actor .042/.082/.044 at1/4/16s;
for STYX, source .301/.417/.358 versus actor .051/.142/.145. These support further
investigation of temporal musical correspondence, not a causal encoder diagnosis.

The bounded planner's pressure improvement is not a uniform correspondence gain:
Zenithfall's16s mean diagnostic .1654 -> .1238, while Hysteric .2966 -> .4271.
Keep these dimensions distinct. A learned relation may favor superficial loudness
tracking or reject deliberate dump/steady positives; this diagnostic has no BAD
threshold and is not a training reward. Closest statistic source is Kornblith et
al., https://proceedings.mlr.press/v97/kornblith19a.html; its neural-representation
results do not validate chart quality. More real positive exceptions and independent
fault-injection/cross-song calibration are needed before fixed model-selection
rules. The framework is useful executable evidence, not a complete quality oracle.


## Publication and positive-counterexample checks

Publication evaluator is committed in251ba0f47970b18f739a3441893e1e0ab9a858cd.
It computes startup required by actual settled-coverage timestamps, carries an
explicit lookahead, finds the limiting publication and refuses to certify an
incomplete trace. A synthetic case generates10s in7.2s overall but needs3s startup
because of a late dense publication; average faster-than-realtime alone would
miss it. Eleven evaluation tests pass in.31s. Actual core/memory native traces
with2s lookahead require.3814/.6521s startup, below their first30 times.5379/.9583s.
Neither has a deadline miss in that trace. Full model loading/waveform/Mel work
is excluded from the recorded clock origin. No general load guarantee follows.

Positive source replay34485 is terminal, exit0. Happy Love Expert source
22e16fc2... has scoped4s peak7.5Hz and J.00593949; Extra Mode4194d810... has6.5Hz
and J.00080477; human Streamd4ec7882... has4.25Hz and J0. No<20ms attack appears in
these scopes. These are prior ranked context positives, not new generated human
labels. The first cost's tiny difference from the earlier rounded report follows
this probe's explicit half-open source scope and endpoint b-1. Preserve the
positive excess counterexamples; a percentile alone is not a BAD classifier.

The newly flagged Max Burning Stream output is now inspected through Lens.
Rendering32346 is terminal; all six source/generated pages on[18174,25175) were
read. It has33 column2 attacks in(19174,23174], with free other lanes. Generated
motion becomes an almost isolated same-column sequence after19.2s, then repeats
that column with sparse chord accompaniment through23.3s. The ranked source uses
changing groups spread across columns. This is an agent-confirmed instance of
the user's regression family despite whole proxy3.8617. Exact viewed coverage,
source/output hashes, controls and seed273100 are in new-fault-lens-review.json.
No human annotation or literal listening is claimed. Treat it as a fixed failure
witness for future candidates, not an automatically generalized style label.

All evaluation and rendering handles are terminal. The framework has useful
independent channels and detected a new failure location, but no universal
quality score or fully calibrated promotion contract. Next freeze a regression
suite with known failures AND ranked positive exceptions, coverage requirements,
per-control-range reporting and independent native trajectories before judging
the main memory fit. New candidate gains cannot be inferred solely from NLL,
optimized J, CKA, or a lack of schema/test errors.

## Result Log: exact pressure episode and peak-context localization

Source58321912c22dd2f6bd0dabf50e52a116e1203bae adds reusable
sustained_attack_witnesses. Sourceb69d360bd044ff11327c98b50a4d514f11728470
adds the contributing peak-window peer counts/holding separately from episode
averages. This distinction matters because pressure can persist after the
sequence that caused it; peers may become active later and obscure the original
concentration in an episode-wide average. No model/sampler/trainer source changes.
Five new tests pass; all16 package tests pass before the final peak-context
extension, and the changed five pass afterward. Existing eleven owners unchanged.
Commands use uv run --extra mps --group dev pytest -q with the package and then
its test_witnesses.py owner. No new quality labels, thresholds or strain law.

Historical audit first handle96809 failed after four generated cases because its
source-map coverage retained a NumPy integer instead of the required Python int.
Correct the audit adapter, preserving source times. Handle45835 completes all
seven cases; the enhanced final audit76365 also completes all seven. Preserve
witness-audit.json and witness-audit-v2.json in20260927-gameplay-evaluation-v1.
Whole-trajectory J matches the existing exact integral to1e-10 on all three
historical initial/planned/failed-response comparisons.

The failed-response Zenithfall0 has a4s-reference positive episode
[120610,128143), duration7533ms. Its peak window(118841,122841] has attacks
[6,38,4,1] and no held columns: pressure is concentrated while peers are free.
The initial Zenithfall271201's largest4s episode instead lies[184543,188141),
with peak-window[3,0,30,1]; this is an additional inspection location, not a new
human semantic label. The paired planner has no4s-positive episode, while its
other-scale whole J remains.0022318945. Max Burning Stream's peak window is
[6,3,33,4], reproducing the inspected8.25Hz regression and whole J.0800069107.

Ranked counterexamples remain explicit: Happy Love has[30,5,17,6] in its7.5Hz
peak window and a2358ms positive episode clipped at the reference scope end;
Extra Mode has[26,7,8,5] and923ms; the human Stream example has no excess.
Happy Love J.0059418449 uses exact[a,b) observation here; the earlier.00593949
probe used[a-1,b-1), explaining its small endpoint difference. A positive
corpus-envelope episode is not automatically BAD. The added facts localize
pressure and peer context; they do not supply independent validation of an
objective already used for selection. No fresh listening or playtest occurs.

### Lens checks of episode context

Render73558 completes nine pages for the initial Zenithfall's newly located
[181674,188674), failed-response Zenithfall[117841,124841), and ranked Happy Love
[136712,143712). All nine pages are viewed; preserve pressure-episodes-lens-review.json
and lens-pressure-episodes/plan.json in the evaluation owner. No audio listening.
The two generated cases show a dominant, largely isolated repeated column lasting
through several seconds, with occasional peer taps/chords rather than shared
burden. The initial model's late-song occurrence is an additional agent-confirmed
Stream-to-jack regression location beyond its previously inspected41s failure.

The ranked counterexample also sustains one column for a long passage. Its short
repeated groups interleave with a recurrent counterline of singles and alternating
chords, producing articulated multi-column cells. Do not confuse its shorter or
boundary-clipped positive-excess episode with a short actual repeating pattern.
Duration or peak rate alone would lose this distinction. Preserve the source's
counterexample role and do not invent a new human assessment. The actual peak
contexts, not episode-wide averages after pressure lingers, expose the relevant
idle-peer versus organized-counterline difference. These observations calibrate
inspection, not a universal acceptance threshold.


### Additional reusable timing diagnostics and native false-success evidence

Productfc641aa740a7528decbb0b5b522aa91039d06ff1 adds first_event_law and paired
waiting-law comparisons, with4 focused passing tests: native sampler agreement,
right censoring, invalid/forced clocks and equal-amount timing differences.
This is a mechanistic instrument, not expected rollout counts or quality score.
Owner2026-09-27-head-control-hazard-probe records339 input laws and150 component
laws, all terminal. No inferred human labels or model promotion.

The matched384 joint fit completes84 exports and fails native pressure/star/
restored-range gates. Its16s Hysteric witness has[93,97,96,104] attacks with all
columns unheld and a47.662s positive episode, while a4s-only selector points
elsewhere. Productef49d63a8c328eac7fa5e41dd8273cfbc435e204 adds all-scale review
context selection and a regression test. This episode's detailed Lens review
remains unperformed; do not label it agent-confirmed BAD from metrics alone.

A14-export H-base substitution provides a false-success reference: Stream mean
J becomes0 and starMAE improves1.334522 to1.162118, but median H ratio.318318,
three first30 times>2s and before/override difficulty failures block promotion.
Restored-range difficulty improves while other ranges worsen. All9 Lens pages
on two fixed development contexts are read; Classic changes local type mix,
Blizzard continues chained LNs with anyheld.922 and no250ms all-column recovery.
No single improved channel substitutes for scope, occupation, organization and
publication checks. The diagnostics and real counterexample are documented in
 gameplay_regression_evaluation.md and audio_memory_joint_fit.md at14ee1534fbffcd6e809482f9ce748383358cb5a1.
No broad suite rerun is claimed: the new waiting-law owner has4 passing checks;
previous17 existing gameplay checks retain their previously recorded evidence.
