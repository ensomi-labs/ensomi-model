# Agent Note: Time-horizon player responses for sustained load and breathing

Note ID: 2026-09-27-player-response-frontier
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: ba406ef827807e3a7045623dff0afa4e6084873b
Scope: Canonical continuation-response semantics; sustained per-column demand, temporal variation, and publication-time playability for the audio/H/R/R1 system
Related: 2026-09-23-audio-skeleton-r1-integration

## User evidence and required response distinctions

The user played the a99519ccee60925dce10a4200f88a6c049d37964-associated candidate
(common-prefix actor-128, SHA-256
364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3) and reported a
severe 4-star long-jack failure with Stream control: burden concentrated on one
finger instead of the expected flowing distribution. No generated file, audio
or seed is available. The reproduction condition is 4 stars with Stream prominent;
other style fields and LN amount unspecified. This is not an exact reproduction
of the user's unseen chart. The same user reports consistently dense generation,
lack of breathing/intervals and larger-scale rhythm variation.

The user requires the frontier to reflect accumulating strain and to constrain
responses outside the requested difficulty, including excessive lane concentration.
Responses must forecast a future duration in actual time, not a fixed row count.
These requirements take priority over candidate selection by NLL or the existing
scalar difficulty proxy. Numeric gains from the paired-response study cannot
cancel these playability failures.

## Formulation and existing mechanism

The formulation's C0(H through t,t;Y,e) and frontier already include an explicit
horizon e. Empty continuations and silence after the last row matter; occupied
LNs are not automatic rest. Canonical pi0 fixes successful exact execution and
left/right symmetry. This study does not infer personal capacity, physiological
fatigue, pain, or a player-specific tolerance. A response representation must be
assessed against independently declared distinctions; named state coordinates
and low training NLL are not sufficient evidence.

RecoveryPreference currently checks immediate HH/RH/HR intervals. At 4 stars,
the empirical HH reference is107ms; larger gaps have zero immediate HH cost.
The separate head-pressure term counts heads across all columns within that
short lookback and does not identify a persistently overloaded finger. Thus a
sequence of individually unpenalized repetitions can continue for seconds without
this recovery preference accumulating a distinct penalty. Learned frontier2
energy can read longer history, but its NLL-trained value is not a calibrated
canonical player response or a direct sustained-load constraint.

A synthetic read-only diagnostic keeps a16-second TAP passage and H rate fixed,
changing only columns. At12 H/s, one-column repetition scores4.1407 under the
scoped strain proxy, while four-finger cycling scores2.5996. At10 H/s the values
are3.4958/2.2182. The scalar orders the patterns in the expected direction but
can still classify an extreme sustained jack near the requested4-star value.
This is not corpus evidence or an approved physical threshold; it explains why
optimizing the scalar alone need not prevent the reported failure.

## Investigation and design direction

Inspect the native ranked corpus at actual1x times, with computed difficulty
bands and source provenance. Measure duration/rate/concentration jointly, then
read the high-tail cases and ordinary comparisons through Lens. Use complete
attack groups and occupancy; same-column statistics locate possible episodes
but do not by themselves assign a Jack label. Distinguish genuinely free columns
from other fingers pinned by LNs. Repeat representative audio generation under
the reported control combination and locate the strongest sustained-load regions.

Define response questions before choosing a compressed state: how burden changes
when the same future follows a loaded versus recovered history; how concentrated
versus distributed legal futures compare at the same timestamps/cardinality; how
a pause changes recovery; and how open holds alter that comparison. Forecast over
explicit real-time windows, preserving state across control switches. The same
canonical response must not change merely because the requested difficulty changes;
the request changes the acceptable response region.

R1 retains cardinality, column, TAP/LN and release-subset ownership. H must create
musically justified timing density and gaps; changing R1 columns cannot create
breathing while every dense H must be materialized. R1 reads direct audio, timing
preview, controls and exact/player state. R may receive execution-feasibility
release bounds; it does not inherit the R1 content encoder. A scheduler evaluates
private joint continuations under the frontier before publication, retaining all
committed rows and real open-hold state. This direction must preserve plausible
Jack expression at compatible intensity rather than force uniform four-column
usage in every window.

## Limits and next evidence

No numerical frontier law, time constants, corpus cutoff, new decoder or training
intervention is selected yet. The corpus's normal arrangements calibrate the
reported distinction; rarity alone is not proof of poor quality. Low activity with
open holds is not full recovery, and low H-rate variance can be appropriate for
some musical passages. Compare actual contexts, not only global aggregate counts.

The ongoing paired-response fits are complete, with their original qualifications
retained as comparative evidence. They are not adopted. Diagnostic owner:
artifacts/joint-audio/20260927-player-response-frontier-v1. After corpus and native
inspection, define one response specification and a bounded implementation/test
that addresses sustained concentration and real-time-horizon behavior. Keep this
Note proposed; no acceptance or remote publication is implied.
