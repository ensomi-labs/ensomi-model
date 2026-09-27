# Agent Note: Reusable temporal playability and musical regression evaluation

Note ID: 2026-09-27-playability-regression-evaluation
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: f65de370416255477f81993bfd594680ba40cbd6; audio-memory implementation currently dirty
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
