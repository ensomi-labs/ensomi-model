# LN feedback: scope semantics and complete native ablation

The projected LN-amount controller favors prefix balance beyond what a whole-scope
ratio request specifies. That mismatch is real, but disabling it is not a repair
for the tested models. In 36 complete generations with fixed weights and matched
seeds, all 18 on/off H streams are exactly equal. Removing feedback often makes
LN usage more extreme, while the inspected passages do not establish replicated
musical-organization gains in two songs. The controller is both a potential
source of local allocation bias and a stabilizing part of these fitted policies.

## What the controller changes

After a complete row with $h$ heads and $l$ new LN starts, the controller updates

$$
b_{n+1}=\operatorname{clip}\left(b_n+\frac{\rho h-l}{8},-2,2\right).
$$

It adds $b_n l$ to candidate log probabilities and renormalizes within each
head/release-count family. This preserves the immediate head/release-count
marginal, but changes LN choices, subsequent occupancy, releases and later R1
decisions. Complete-trajectory head counts and difficulty need not be preserved.
Only the timing-only H stream is unchanged in this ablation.

The distinction from a scope-total target is concrete. A .5 request can be met by
64 TAP heads followed by 64 LN heads. Feedback reaches +2 after the first 32 TAP
heads and, because clipped debt is discarded, ends the valid complete sequence
at -2. The request itself did not require every prefix to approach .5.

A genuine ranked STYX HELIX source supplies a corresponding witness. Its whole
LN-head fraction is .485281. Before its TAP at 4252 ms, it has 34 heads and no LN
starts, so replayed feedback is already +2. Both the source TAP on zero-based
column 2 and an LN start on that same column are feasible under 60/50/50 with
the actual H preview. They share head/release counts. Feedback therefore multiplies the LN
alternative's odds relative to the source TAP by $e^2\approx7.389$, although the
complete source exactly meets the requested ratio. This is a preference change,
not proof of its effect on neural raw odds or generated musical quality.

## Fixed-weight comparison

Two checkpoints are evaluated separately:

- The actor128 row-outcome endpoint, with its saved HH/RH/HR profile 60/50/50.
- The failed joint-memory384 endpoint, with its saved profile 60/50/40.

Each generates Max Burning, Classic Pursuit, STYX HELIX and Blizzard Heights
at two seeds, requesting four stars, the source's whole-chart LN-head fraction
and unspecified styles. A ninth case changes difficulty/LN controls during
Zenithfall. Each checkpoint is run with feedback enabled and disabled, for 36
outputs. Disabling feedback leaves the neural controls, analytic requested-ratio
tilt, recovery preference, R/R1 weights and all audio/timing inputs unchanged.
No model parameters are updated and no source suffix is attached to a generated
prefix.

The [packaged native qualifier](gameplay_regression_evaluation.md#executable-native-qualification)
executes all cases from BOS, verifies exported osu rows, measures separate scopes
and records actual publication deadlines. All outputs complete; every on/off H
pair is identical, including the live switch. Thus differences below are
downstream consequences of the feedback intervention, not different skeletons.
Seeds match, but after actions diverge the row/release histories and random-draw
positions can also diverge; the comparison is of complete policies.

## Amount correction is substantial

Entries show the two seeds in the same order; LN fractions count note heads,
not occupied time. Different songs and requested amounts are not pooled.

| Song / requested fraction | Actor on | Actor off | Memory on | Memory off |
| --- | --- | --- | --- | --- |
| Max Burning / .043 | .033 / .023 | .011 / .027 | .021 / .021 | .001 / .001 |
| Classic Pursuit / .217 | .225 / .215 | .145 / .191 | .206 / .212 | .071 / .075 |
| STYX HELIX / .485 | .504 / .496 | .939 / .933 | .481 / .486 | .605 / .362 |
| Blizzard Heights / .838 | .941 / .946 | .993 / .993 | .852 / .849 | .961 / .941 |

Removing feedback does not merely permit a little more local variation. Actor
STYX becomes nearly all LN at both seeds. Memory Classic moves toward very few
LNs, and memory STYX's two seeds move to opposite sides of the request. Feedback
substantially corrects these outcomes, though actor Blizzard still misses the
.10 amount-error bound with feedback enabled.

The actual feedback direction also matters. In the first STYX review passage,
actor's head-weighted mean offset is -1.293; in Blizzard's it is -1.984, with
69.3% of heads encountering the negative bound. In these observed states it is
already reducing LN preference. Conversely, memory Max Burning encounters +2
before every one of its 61 heads in the selected passage and still starts no
LN there. A finite correction does not determine the complete learned/support-
conditioned response. Replayed offsets on feedback-off charts are retained only
as hypothetical diagnostics and are not described as applied controller states.

## Local organization and actual recovery

Eighteen new time-proportional Lens pages cover Classic [88589,93589) ms and
STYX [1800,11800) ms at the first seed: actor on/off and memory off. Memory-on
exports are byte-identical to the previously inspected counterparts, whose six
pages and source comparisons remain valid. Seed-two pages, other new passages
and semantic style labels remain unreviewed. No audio listening or player test
is claimed, and these readings do not create human annotations.

In actor Classic, disabling feedback breaks the early concentrated column-1 TAP
run into more distributed TAP/chord motion. This is a limited local redistribution
gain. Heads increase from 51 to 68 and almost all held texture disappears: LN
fraction falls from .373 to .029. The later held block is removed rather than
replaced by a demonstrated source-like mixed organization. A TAP-forward alternate
arrangement can be valid; these observations do not establish an overall musical
gain or replicated improvement across songs.

Memory Classic becomes TAP-only through both pages. Its previous TAP-to-LN block
disappears, but all-column recovery after 250 ms remains zero: H pacing continues
throughout. Eliminating holds is different from creating a rest. Whole-song LN
amount also misses its request substantially at both seeds.

Actor STYX remains dominated by serial and overlapping held voices in both
variants. Feedback-off removes most remaining TAP punctuation and does not
recover the source's TAP-flow-to-held contrast. Memory-off STYX likewise retains
held texture through the intro, including a long inner-column anchor around
6.7–8.35 seconds. Its local LN fraction rises from .648 to .708, heads from 54 to
65, and any-held time from .848 to .855. No convincing recovery or texture repair
is established. Matching a ranked source is not mandatory, but changing an
aggregate cannot substitute for an inspected organization gain.

## Correct quantity scopes after live changes

The switch starts with a whole-song request of difficulty 3 / LN fraction .2.
After publication through 63999 ms, an explicit request sets 4.5 / .6 on
[64000,96000), after which earlier control values resume. The explicit override
is a complete declared amount scope. Its realized LN fractions are .584 versus
.965 for actor on/off, and .623 versus .868 for memory on/off. Removing feedback
worsens this requested amount in both models.

The preceding prefix and restored fragment are not new completed LN-total
requests under the current original-extent restoration contract. Requiring their
ratios to match .2 would silently impose the prefix-balancing behavior under
investigation. The first frozen qualifier did apply those two amount gates.
Those original reports remain unchanged. A separate assessment correction marks
these fragment ratios diagnostic, while retaining separate occupation, recovery,
difficulty-proxy and realized-ratio observations. No differently controlled
ranges are pooled. All four arms still fail other declared gates; this correction
does not promote an endpoint or erase the ablation result.

The runner now rejects an LN-total gate without a matching declared extent or
when a conflicting override invalidates that total. This distinguishes model
conditioning values from the unit on which an outcome can legitimately be judged.
An explicitly requested new restored scope could be assessed as its own total;
merely resuming an earlier extent does not create one retroactively.

## Runtime and decision

The full study takes 615.45 seconds on an Apple M5 with 24 GiB, CPU one thread.
Maximum first-thirty-row times are .811/.814 seconds for actor on/off and
1.278/1.229 for memory on/off. Maximum two-second publication services are
.237/.257 and .771/.802 seconds. No observed publication deadline is missed and
no below-20-ms same-column attack appears. Clocks begin with loaded weights and
cached Mel, including complete model audio encoding. These are measured cases,
not complete client cold-start or concurrent-load guarantees.

Disabling this controller is not selected as a general repair. Its scope-semantic
bias and its stabilizing effect must both be accounted for in a replacement.
The stronger remaining question concerns the learned LN conditional distribution,
the analytic ratio conditioning, history dependence and source/native training
policy together. Fixed-weight removal cannot determine whether training without
the controller would work better.

Further outcome learning must use the current policy's actual BOS trajectories
and complete declared response horizons, with genuine source-prefix imitation
kept separate. A fixed old-prefix bank remains useful for local regression and
mechanism diagnosis, but does not measure which states an updated policy visits.
Additional history capacity alone is not selected as the explanation for these
results. The [recovery support census](joint_action_spacing.md#ranked-support-excluded-by-605050-and-605040)
also remains a separate expressive-support issue, not an intervention made here.

## Identities

Native source: `226725c5838dab2ef7f381741079ea7c35bb4dc7`.
Study owner: `20260927-controller-semantics-v1`.

| Item | SHA-256 |
| --- | --- |
| Actor128 | `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3` |
| Memory384 | `7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81` |
| Original generation/assessment plan | `4ab74f18ece7b747a9410ca259f7e2a882c379744795228365f96ce99b621e02` |
| Four-arm execution ledger | `bc19e058d7233e318f8222543248d87b17b0deb5d3ad94924ce8395a3ffc1644` |
| Paired numerical analysis | `e6ac3cf2cd04334fef5b05ef297c8f315380179b7b73dc62b8e0e37180905291` |
| Feedback ledgers | `2238b8bf0dc572d10b397c4da9f5865ed11b2c1d8041eb4ce1a3b31b9f1bc605` |
| Visual review | `03081901630e579e29298ba31d15ecbc4b4bb9be10450224cf2929ee64775cb7` |
| Corrected scope assessment plan | `df760658ced078aa55867bb6eb1710745f434fa584b11d91a7f6966ed74089b2` |
| Corrected scope assessment | `b977ce43dd833edf6a5c8e3c797618d83d17bcff335b3f940c6d540a925d38cf` |

The source controller witness uses STYX chart
`0cc766925a6c98343ffe812927bcec6f62ba923d99d4c7cd9e9a16d23da43412`.
Lens harness revision: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`.
