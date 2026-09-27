# Full R1 learning and difficulty response

The ordinary full-R1 source fit restores some LN/TAP variety and reduces one
attack-pressure statistic, but has not produced a qualified playable model.
Reducing its neural difficulty code lowers achieved stars on Zenithfall while
increasing repeated-column organization across the four tested audio/seed pairs.
That correction is not adopted.

The experiments separate three questions: whether the actual row decision
owners benefit from broader factual learning, whether the difficulty code is a
usable control, and whether evaluation can detect an apparently easier output
that still repeats the same finger. The
[formal diagnosis](native_pattern_failure_analysis_zh.md) defines the objects,
module ownership and causal limits. H owns head-bearing times; R owns pure-release
times; R1 owns complete rows, including counts, TAP/LN types and columns. Full
audio is available throughout. No shape filter is added.

## What the first full-R1 fit visibly changed

The parent actor128 has checkpoint SHA
364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3.
Both ordinary and missing-history variants train 2,753,715 of 4,600,369 parameters
on the same 256 factual eight-second windows from 224 charts and 214 song groups.
Audio encoding and H/R networks remain frozen; changed LN states can still change
actual R outcomes. Each fit uses 128 updates, batch two, deployed source-row NLL
per second, feedback off and 60/50/50-ms recovery. The missing variant hides
learned history in 64 windows without changing the factual chart world.

Ordinary source validation improves from 1.730817 to 1.618333 nats/row.
The missing variant scores 1.623110 on ordinary observations and 2.335572 on its
missing observations, compared with initial 2.957741 for the latter. Its native
pressure is worse than the ordinary variant in both Stream seeds, so the auxiliary
likelihood improvement is not a reason to scale that branch.

Twenty-four Lens pages were inspected for source and ordinary-fit outputs, with
both generated seeds: Classic [88589,93589), STYX [1800,7800), and Blizzard
[40342,46342) ms. Source action tables retain complete endpoints and entering
holds. These are agent observations, not new human annotations or listening/
playtest results.

| Passage | Source organization | Ordinary full-R1 output |
| --- | --- | --- |
| Classic | Short LN accents and longer holds interleave with TAP/chord responses | Both seeds regain mixed LN/TAP organization. Seed 0 has more paired endings; seed 1 carries several holds under other-column TAPs |
| STYX | This local episode is TAP/chord flow despite a nonzero whole-song LN fraction | Both seeds remain predominantly LN; whole-song quantity improvement does not establish a better local interpretation |
| Blizzard | Sustained anchors coexist with TAP accents and paired short holds, followed by reduced activity | Both seeds retain near-continuous changing holds. A longer anchor occurs in seed 1, but predominantly under other LN entries |

In Classic, the initial two outputs have 2/0 LN heads within the inspected scope;
the ordinary fit has 16/7. Blizzard's source has 12 continuing-hold/TAP pairs
versus 0/1 in the fitted seeds. A TAP under two holds counts as two pairs. These
are relational observations, not targets requiring an alternative chart to
copy source counts. They show why longer duration or correct global LN amount
cannot certify the arrangement.

The ordinary fit checkpoint is
32afd48e5dbad6714401735e01200963c9d447e9f8ca92057bf6c7b67ab36406.
Its unchanged fourteen-case native panel has exact H parity with the initial
actor. Both Blizzard amount checks, the live override and style difficulty
cases remain unresolved; no model is promoted.

## Broader source learning improves recurrence but loses LN control

An additional 512-update phase starts from that ordinary fit, using 1,024 new
eight-second factual windows from 748 charts and 651 song groups. It keeps the
same R1 ownership, frozen audio/H/R tensors, feedback-off law, recovery and
learning rates. There are 763 population and 261 human-annotation draws;
recomputed star bands 2/3/4/5 have 238/258/259/269 windows. All five native-panel
audio identities are excluded from this phase. The exact 22 previous validation
windows and controls are reused.

Population proposals balance nonempty metadata-star/LN-fraction strata and then
song groups/charts; 25% of proposals use human-annotation windows. Actual
acceptance includes 70 rejected draws before the 1,024 accepted windows.
Controls are whole-song in 505 draws, otherwise factual 16/32/64-second scopes;
835 draws retain an LN request. True hold/TAP interactions occur in 498 windows
and hold/LN-entry interactions in 408. These are exposure observations, not
new style labels or response targets.

Fresh AdamW uses $10^{-4}$ for composition, row-control, preview and layout
modulation and $3\times10^{-5}$ for other R1 parameters, weight decay $10^{-4}$,
gradient cap one and batch two. Serial sixteen-update workers restore both
weights and optimizer, recomputing current-weight factual history. The phase
completes in 893.52s on MPS with maximum observed footprint 5,633,691,776 bytes.
Validation improves from **1.618333 to 1.572470 nats/row**.

Twenty native cases comprise the original fourteen plus two seeds each for
unspecified-style Zenithfall, unspecified-style Classic and prominent-Stream
Classic. All H timestamps exactly match the frozen parent outputs. No under-20ms
or publication failure appears. The nineteen whole-chart D4 cases' mean absolute
star error changes only .630285 to .622187, below the declared .1 improvement.
The live override retains separate scope checks.

| Whole-song LN request | Parent fractions, s0 / s1 | Broader fit, s0 / s1 |
| --- | --- | --- |
| Classic .217 | .217 / .154 | .360 / .438 |
| STYX .485 | .519 / .487 | .564 / .696 |
| Blizzard .838 | .972 / .953 | .969 / .976 |

The fit adds three LN-amount failures in the original control/LN panel. STYX s0
also falls to 2.991 stars for D4. The D4.5/LN .6 override produces LN fraction
.941 and newly fails its difficulty check; its preceding/restored scopes are not
relabelled as independent LN quotas. These failures prevent qualification.

There are nevertheless concrete routing/trajectory gains. In the four Stream
cases, whole-song maximum consecutive-H column age changes from
21/16/8/17 to **8/7/7/6**:

| Stream case | Parent repeated-head fraction | Broader fit | Parent / new attack excess, seconds |
| --- | ---: | ---: | --- |
| Zenithfall 271200 | .211 | .105 | .011005 / .000028 |
| Zenithfall 271201 | .153 | .070 | .023828 / .001567 |
| Classic 273110 | .302 | .120 | 0 / 0 |
| Classic 273111 | .391 | .172 | .013307 / .000039 |

Twelve new Lens pages cover the parent recurrence witnesses and two LN contexts.
Zenithfall [41606,45072) at seed 271201 now has distributed TAP flow with short
repeats and occasional chords; it contains no LN. This local improvement is
therefore not solely a consequence of occupied columns forbidding repetition.
Other inspected Stream witnesses become mixed TAP/LN arrangements. Classic's
LN passage remains organized while exceeding its requested whole amount;
Blizzard still does not restore the inspected source anchor/TAP relationships.
This is a mixed result, not proof of a full musical or control repair.

On the same four *parent-native* prefixes used below, switching only model
weights reduces one-TAP-family repeat probability from .470/.700/.415/.742 to
.406/.685/.387/.739. These fixed-history changes are smaller than the native
recurrence differences and do not establish recovery from every old bad prefix.
Changed reached histories and LN occupation remain part of the mechanism.
The checkpoint is retained as an arrangement candidate, not promoted, and the
same recipe is not extended merely because source likelihood improved.

Endpoint SHA:
5206b1e4dcbff9820a1e84ecf1ba02aee22c60105cae93d8f648ab6458780ea5.
Owner 20260928-broader-full-row-learning-v1, frozen source-plan SHA
fb1d2634f8061c3fb55da2bcd369b9ea51872358dcafe50d8a85b082582c553f,
fit-plan SHA 04a07294455d9560fd12cae9648c916dfa4580c2e9ab6f111bfaa5086f593e7e,
native cases SHA c657aee0c275fd46b06401945b387b4a75f431978332c368ee134fb2af138db9,
Lens review SHA 20b96468127bad0782a1b3c2627568a0c78503684175fd29e20a92b89aaed8d3.
The twenty generations take 319.06s; maximum qualifier startup/service are
.944/.372s. This remains the two-second qualifier, not the separate realtime
benchmark or a client guarantee.

## Difficulty changes must retain the player's actual request

A comparison on the earlier ordinary 128-update fit uses two audios and two seeds
each; it does not evaluate the broader endpoint above. Every user request
is D4 with unknown LN fraction. The three conditions are unspecified style,
prominent Stream, and prominent Stream with a neural-only star-code offset of
-1. The last condition changes the control tensor at H, R and R1 neural calls,
from normalized star coordinate 0 to -.5. Recovery preferences, scope requests
and the sustained-pressure reference remain D4.

The experiment verifies the actual call-site input/output ranges. It does not
rewrite checkpoint weights or treat a system-wide D3 request as meeting D4.
Ten new outputs complete; two unshifted Zenithfall Stream outputs are reused by
hash. Achieved values below are whole-chart stars from
compute_mania_star_rating_20241007, not a local difficulty proxy.

| Audio / seed | Style unspecified | Stream | Stream, neural code -1 |
| --- | ---: | ---: | ---: |
| Zenithfall / 271200 | 4.918 | 5.071 | 4.388 |
| Zenithfall / 271201 | 5.206 | 5.134 | 4.586 |
| Classic / 273110 | 4.450 | 4.366 | 4.375 |
| Classic / 273111 | 4.452 | 4.360 | 4.005 |

Stream itself does not produce a repeated .5-star increase on either audio.
Zenithfall is high even without a style request. The neural offset does produce
two decreases exceeding .5 stars on Zenithfall, but not on Classic. The evidence
therefore establishes a usable but audio/seed-dependent response, not a uniform
calibration or the full 2–6-star response curve.

Under the unchanged D4 reference, Zenithfall attack excess changes from
.011005/.023828 seconds to .012677/.002178. The first increases; the second falls.
Classic changes from 0/.013307 to .000821/.003689. Neither pooling seeds nor
reporting stars alone resolves these different consequences.

Twelve further Lens pages compare unshifted/shifted Zenithfall seed 271201 at
[41094,49094) and Classic seed 273110 at [88589,93589). Zenithfall's unshifted
output contains a long left-column repeat. The shifted version distributes
repetition into several column-local jack blocks, including right and right-inner
runs. It does not establish the requested continuous multi-finger flow. Classic
remains a TAP/chord mixture, with some repeated left-inner attacks after the
offset; no general musical/style benefit is assigned.

## Combining the broader row model with the same neural offset

A subsequent four-case interaction probe applies the same neural-only -1
projection to the broader endpoint, retaining the actual D4 request and recovery.
The H sequences exactly match the earlier shifted ordinary-fit outputs for
each audio/seed. This makes the changed rows comparable under those generated
times, without claiming an isolated effect on reached history.

| Stream case | Achieved stars | Repeated-head fraction | Maximum prefix H age | Attack excess |
| --- | ---: | ---: | ---: | ---: |
| Zenithfall 271200 | 4.234 | .102 | 12 | 0 |
| Zenithfall 271201 | 4.633 | .108 | 16 | 0 |
| Classic 273110 | 4.112 | .104 | 5 | 0 |
| Classic 273111 | 3.937 | .123 | 6 | 0 |

Three cases are within .5 stars of D4; the second Zenithfall case misses the
prespecified two-seed condition. Recurrence averages retain much of the broader
model's improvement, but two new Zenithfall witnesses contain mostly isolated
12/16-head runs over 1.702/2.170 seconds. The older dense witness in the first
seed becomes sparse under the changed H plan; the other becomes LN flow.
Thus neither disappearance of an old witness nor zero attack excess establishes
the complete repair. Seventeen pages inspect old/new witnesses, Classic
transfer cases and the ranked comparisons below.

Actual ranked data also prevents turning these lengths into a ban. Two
independently recomputed source examples are:

| Ranked source | Whole stars | Inspected recurrent run | Other-column heads during those 16 H |
| --- | ---: | --- | ---: |
| LiSA — Brave Freak Out (TV Size), Limit Breaker | 4.260980 | 23282–25532ms, 150ms HH gaps | 12 |
| HOYO-MiX — Termination of Desires, Eternity | 3.769631 | 74916–77208ms, median/max HH gap 105/209ms | 12 |

Both have 16 TAPs on the recurring column and changing accompaniment on the
others. Source SHA identities are
43f20aa3129a415b5bca7d2a5c1923b6a28267cb561d24e69ccecab8e9198eee
and fb0e61e4ec6277177aeef3a38d8b34abd44df860f22f78a020f94c6075d64987.
The two generated witnesses have zero/two companion heads respectively.
That difference concerns arrangement roles; adding accompaniment would not
automatically relieve the recurring finger or prove better music. Cadence,
history, requested style and audio correspondence still matter.

The reusable observer now reports companion counts and columns alongside each
run, retaining legitimate anchor/jack examples. The composition is retained as
a diagnostic, not a universal calibration or qualified release. It takes 62.95s
on CPU one thread with observed footprint 585,991,488 bytes. Source is
6fa3a544e52624c9922c050743bd351aa2dc4f9f; owner is
20260928-broader-control-composition-v1. Native cases SHA:
d6537a316c0fd53ae88343f2049511b9280d8a0e90a37e5d18a2d27d3e7c8857;
Lens reading SHA:
569814040c1cfa957d48b12238c75c17d8dc4ac7c73601b85ff5de93619eb010.

## An evaluation distinction that stars and threshold excess can miss

Let $H_i$ be the set of columns attacked by the $i$-th head-bearing row. Define
the causal consecutive-membership age

$$
n_{k,i}=
\begin{cases}
n_{k,i-1}+1,& k\in H_i,\\
0,&k\notin H_i,
\end{cases}
\qquad n_{k,-1}=0.
$$

This counts consecutive H groups containing a column, including both TAP and
LN press. Pure-release rows do not advance this index. It is not a complete
definition of Jack style: intervening head groups can split a musically related
repeat, and slow repetitions can have large ages. The observer therefore reports
real spans and HH gaps, keeps pre-scope history, and never reads events beyond
the observation end. Repeated-head fractions exclude the first H's heads from
their denominator because no preceding H is available.

Two synthetic four-second traces have identical 32 H times, one head per H and
eight heads per column. One rotates fingers; the other places eight consecutive
heads on each column before switching. Both have zero attack excess under the
existing D4 rate references, but their maximum prefix ages are 1 and 8 and their
repeated-head counts are 0 and 28. This demonstrates a missing observational
distinction, without declaring either synthetic chart a universal BAD pattern.

The whole-song fraction of heads repeating a column present in the previous H
increases under the neural offset in all four cases:

| Audio / seed | Unshifted Stream | Neural code -1 |
| --- | ---: | ---: |
| Zenithfall / 271200 | .211 | .310 |
| Zenithfall / 271201 | .153 | .235 |
| Classic / 273110 | .302 | .358 |
| Classic / 273111 | .391 | .442 |

For descriptive context, a read-only census covers 1,972 ranked TRAIN charts
from 1,613 song groups with metadata stars in [3.5,4.5). Unweighted per-chart
q50/q90/q95/q99 repeated-head fractions are .1554/.2863/.3284/.4236.
The corresponding fractions of heads whose *current prefix age* is at least
eight are 0/.00358/.00695/.02318. This is not a count of every head belonging to
an eventually long run.

These metadata-band values are not recomputed-star, style-conditioned or
physiological thresholds. Ranked Jack/dump exceptions remain legitimate. The
census supplies distributions and inspectable witnesses; it does not install
a sampling mask, automatic style label or acceptance gate.

## Is the direct R1 difficulty code responsible for the repetition change?

A secondary read-only probe fixes the actual unshifted native histories, H
preview, exact state, support, audio and D4 recovery. Only the current R1 neural
star code changes by -1. It scores [40000,48000) on Zenithfall and [88000,96000)
on Classic, for 58/63 and 64/70 H queries respectively.

The mean expected head count falls by about .039–.045 per query. Within the
same one-TAP/no-release count family, the mean absolute change in repeat
probability is only .01352/.01001 and .01380/.00842; maximum query changes are
.03242/.02999 and .02989/.02343. Signed averages alone could hide cancellation,
so these absolute changes are reported explicitly.

The fixed-history comparison does not reproduce a large uniform direct routing
shift. Actual rollouts also change H, composition and the histories subsequently
reached; small local changes may propagate. This is not a numerical attribution
of responsibility to H. It explains why turning one difficulty input down
cannot be assumed to redistribute finger pressure or repair the closed loop.

## Provenance and limits

Training implementation source is 0882315ef23097e44e031707abd382d971b8c82c;
the later evaluation/source inspection is at
0f5ec10b34c43fe21a6143d371a2f2bc36070e47, whose changes after 13eefbc are
documentation only. The native difficulty probe runs on CPU one thread on
Apple M5 / 24 GiB / Torch 2.11, takes 139.87s and observes 656,065,952 bytes
peak task footprint. Call-site auditing overhead is included. This does not
replace the separate 30-row/eight-second realtime benchmark.

| Evidence | SHA-256 |
| --- | --- |
| Difficulty-response frozen plan | 625114fdfe3a8f43d1fe4606a1b21963de89f2d742c4af3a2afa5146340fb3b7 |
| Twelve generated/reused case records | 88ae0e1bacb38745c96a65f3edd830b4d9b951b35fdf873c32c934af60773709 |
| Full-R1 LN Lens reading | 1c8324d43f3ab49c3928475ce4befbf3c18a9301c357ff5c7e09f099bda0f19d |
| Difficulty-response Lens reading | 134384050320e1e6ed8ff1a74081546d9cb6a3d1c49c9ab43dc2049a663656ec |
| Fixed-history R1-code cases | 165e72c6b706b63c77ca4de66e3502f19630ccd8a8001669e5c0fececc702ea5 |
| Ranked recurrence census | 5cebefa1dccdd7aff6d7c3c19672ee3ba91e124dd2e8eb8188498988dbfa2f08 |

Owners are 20260927-full-row-history-views-v1,
20260928-style-difficulty-response-v1 and
20260928-recurrence-observations-v1 under the ignored local joint-audio assets.
Lens harness revision is 22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60. Corpus
census and fixed-history probes are descriptive; they do not create human
response labels. Musical interpretation, broader style preservation, dynamic
controls and the full 2–6-star task remain unqualified.
