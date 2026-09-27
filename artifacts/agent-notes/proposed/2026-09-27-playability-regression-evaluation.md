# Agent Note: Reusable temporal playability and musical regression evaluation

Note ID: 2026-09-27-playability-regression-evaluation
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 720457b8651d40095c2247b5992c3318b2cb5ced
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
