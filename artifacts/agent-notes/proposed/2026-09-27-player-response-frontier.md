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

## Implementation decision after source inspection

The existing frontier2 cannot serve as the required response interface. Keep its
useful exact candidate-action coordinates and row preference path, but introduce
an explicit committed-history player state and time-horizon continuation-response
operator. This is not a claim that its history TCN cannot encode any coordination
or repeated attacks. Its learned context can distinguish histories; the missing
part is declared response semantics, training/calibration and the required API.

RowConsequence.score currently returns one learned scalar per complete candidate
row. Its inputs include the action's post-occupancy, last-attack/release clocks,
LN age, next H and second H. Planned consequences passively advance that immediate
post-action state; they do not simulate intervening future actions. There is no
explicit future-duration argument, continuation Y, or dedicated demand-state
advance through no-row time. Its context also contains audio and controls, and
its score is learned as part of normalized row likelihood. It is therefore a
conditional arrangement preference rather than an independently calibrated C0.

The replacement response interface must consume the same committed rows for
training/replay/inference, advance on actual milliseconds, preserve active LN
origins/occupancy, and evaluate legal candidate continuations through an explicit
end time, including empty endings. Accumulation/decay laws are candidate
representations to fit/compare against corpus and mapper-defined response
contrasts, not human capacity curves supplied by the repository. Keep the
canonical response independent of the requested difficulty; apply difficulty
when selecting an admissible response region. R1 retains all spatial/count and
TAP/LN decisions; H owns timing variation and silence.

## Reproduction evidence

Nine native outputs under 4 stars + Stream prominent completed using the reported
checkpoint, across Zenithfall, Hysteric and Take with seeds271200+10*case+i. Other
style fields and LN amount are unknown. Zenithfall draw1 has31 of34 heads on one
column over4 seconds (7.75 attacks/s); other columns are not pinned by holds at
those attacks. Hysteric draw2 has28 of37 heads on one column over4 seconds
(7 attacks/s), again with no other held columns. This reproduces the reported
failure family, not the user's unavailable exact chart. Their whole proxies are
4.9844 and3.9434. A130ms chain cutoff misses the longer episodes: time-based
pressure and complete-row Lens inspection are necessary.

The reproduction process56717 is terminal. A pre-event occupancy indexing issue
at the very first row was corrected only after it ended; original results remain
in result.json, recomputed facts in result-v2.json with metric-revision.json.
No generated rows changed. Native ranked TRAIN corpus profiling now runs under
an explicit four-worker process, using full source rows and recomputed1x stars.
It measures fixed real-time windows, sustained-column chains, activity variation
and occupancy-aware no-action recovery. These remain descriptive source facts;
no automatic Jack label or admissibility cutoff has been selected.
