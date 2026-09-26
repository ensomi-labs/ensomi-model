# Bounded planning with sustained-attack responses

An explicit continuation response substantially reduces the reproduced sustained
single-column overload without changing H timing or model weights. It does not
establish reliable Stream control, solve difficulty calibration, or add breathing
to the skeleton. The [player-state investigation](time_horizon_player_responses.md)
describes the failure and the distinction between learned row preferences and
canonical gameplay responses.

## Response and selection rule

The reference uses 6,923 ranked TRAIN charts at native speed. For each chart,
measure its maximum per-column attack rate in .5/1/2/4/8/16-second windows.
Within each one-star-wide difficulty band, each song group receives equal total
weight, shared among its charts. The fitted envelope is the 99th percentile of
those chart maxima, with difficulty knots 2,2.5,...,6. Linear interpolation gives
the requested-difficulty reference. A monotonicity operation is available, but
the observed raw curves already increase and require no adjustment.

At four stars the six references are 10/8/7/6/5.375/5 attacks per second. The
six-star endpoint has only 125 groups, compared with 1,613 at four stars. These
are empirical arrangement ranges, not physiological capacity estimates.

Let $r_{c,w}(t)$ count TAP/LN presses on column $c$ in $(t-w,t]$, divided by the
window duration in seconds. The committed state and these observations do not
depend on the request. For selection, a difficulty request $D(t)$ supplies
reference $B_w(D(t))$. The private continuation cost is

$$
J(Y;e)=\int_g^e \frac{1}{|W|}
\sum_{w\in W}\sum_c
\left[\frac{r_{c,w}(t)}{B_w(D(t))}-1\right]_+^2\,dt.
$$

The integral uses seconds and includes residual attacks from committed history.
Actual attack, window-expiry and control-boundary times make it exact for these
box-window observations. Each control range has its own reported contribution.
Unknown difficulty applies no cost; changing difficulty does not clear history.
Releases are not attacks. LN occupancy still governs legal candidate generation,
but held demand and coordination are not evaluated by this first cost channel.

[`AttackEnvelope` and `sustained_response`](../../src/ensomi_model/research/player_response/envelope.py)
keep canonical observations separate from the requested reference. Positive
excess does not imply BAD organization: inspected ranked Happy Love Expert and
Extra Mode contexts receive costs .005942 and .000805, while the human-prominent
Stream comparison receives zero. These real exceptions remain evidence against
treating a quantile as an absolute quality boundary.

## Private forecast and immutable publication

[`ResponsePlanner`](../../src/ensomi_model/research/controlled_audio_continuation/frontier.py)
forecasts four seconds from the current publication boundary and commits the
first two. If the original proposal has zero cost, it is retained. Otherwise,
at most three additional R/R1 continuations are sampled from that same boundary;
the minimum-cost proposal wins, with earlier proposals winning ties. Search
stops at a zero-cost candidate or the fixed budget. A nonzero-cost result is
allowed when that budget is exhausted.

This transfers the finite-horizon evaluation and partial-application mechanism
of [receding-horizon control](https://doi.org/10.1016/S0005-1098(99)00214-9).
Discrete stochastic chart proposals, empirical response references and a fixed
candidate budget supply no corresponding optimality or stability theorem.

All candidates share model weights, complete audio, controls and the H process.
Retries change only R/R1 randomness. Only the selected prefix's rows, exact replay,
history caches, random-generator state and player state become current. A later
hypothetical LN endpoint is not published early or stored as an observed fact.
Control updates apply after published coverage. Empty time and accumulated demand
survive those updates.

The selected trajectory has a different distribution from the native policy.
`replay_row_scores` still reconstructs native proposal probabilities; it must not
be used as if it were the selected policy's probability. Learning from selected
prefixes requires an explicitly declared imitation or policy objective.

## Nine matched Stream generations

The reported actor-128 checkpoint generates three seeds each on Zenithfall,
Hysteric and Take, requesting four stars and prominent Stream; other style and
LN fields are unknown. Original outputs are reused as the baseline. Every
planned H stream is exactly equal to its counterpart. All nine exports complete.

| Case | Original maximum column Hz over 4 s | Planned | Original longest chain, ms | Planned |
| --- | ---: | ---: | ---: | ---: |
| Zenithfall 0 | 6.75 | 6.25 | 1173 | 1252 |
| Zenithfall 1 | 7.75 | 6.00 | 3907 | 1058 |
| Zenithfall 2 | 7.50 | 6.00 | 2004 | 997 |
| Hysteric 0 | 6.50 | 5.25 | 1016 | 866 |
| Hysteric 1 | 6.25 | 6.00 | 944 | 825 |
| Hysteric 2 | 7.00 | 6.00 | 1083 | 854 |
| Take 0 | 5.50 | 5.50 | 693 | 693 |
| Take 1 | 5.50 | 5.50 | 1098 | 1098 |
| Take 2 | 6.25 | 5.75 | 1270 | 958 |

Chains use successive same-column gaps at most 170 ms; they are diagnostic
sequences, not semantic Jack labels. The whole-chart maximum is reported so an
improvement in one crop cannot hide relocation. Eight/sixteen-second rates are
also retained in the comparison. Integrated excess falls from mean .027516 to
.000425 seconds, a 98.46% reduction. This is the selected objective, not an
independent quality score. Mean whole-star error barely changes, .47893 to .48392.

In the original Zenithfall failure window, column heads change from [1,31,1,1]
to [5,9,18,8]; Hysteric changes [4,4,28,1] to [7,12,15,10]. Both contain more
heads after selection. Reduced concentration is not simply note removal.
Two already-zero-cost Take outputs remain exactly identical in all rows.

Lens inspection confirms that the prolonged single-column sequences in the two
original failure regions are broken. Short jack groups, chord repetitions and
anchors remain. The new strongest Zenithfall region moves across columns with
chord accents; the strongest Hysteric region still contains repeated anchors.
These observations support a sustained-load improvement, not a semantic Stream
pass. Tightening a load threshold to force Stream organization would conflate
demand and style.

On an Apple M5 with 24 GiB unified memory, one CPU thread and loaded model,
the first thirty published rows take .327-.615 seconds; the slowest two-second
publication takes .968 seconds. Complete-song generation takes 6.29-33.92 seconds
for these 144-358-second audios. Timing starts with cached canonical Mel and
includes model audio encoding and private forecasts, excluding waveform decoding,
Mel construction and model loading. Brief local rendering/analysis activity is
not an isolated system-load benchmark. These measurements demonstrate headroom
on the tested panel, not a universal latency guarantee.

## Other controls and a live scope change

Five additional Zenithfall conditions receive matched native/planned generations.
All ten complete, with matching H streams. Requests are four stars with the named
style, except the switch case. The LN condition additionally requests fraction .6.

| Condition | Native excess | Planned excess | Native whole stars | Planned whole stars | Replaced publication decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jack | .000069 | 0 | 5.093 | 5.102 | 2/179 |
| Tech | 1.644136 | .162401 | 5.581 | 5.449 | 87/179 |
| Trill | 1.461985 | .530589 | 5.688 | 5.594 | 49/179 |
| LN coordination | 0 | 0 | 4.803 | 4.803 | 0/179 |
| Difficulty/LN switch | .004002 | .000063 | 4.543 | 4.365 | 3/179 |

Tech and Trill exhaust the candidate budget with positive cost at 79 and 63
publication decisions. Their four-star requests remain poorly calibrated.
The candidate budget stays fixed; more sampling is not treated as a substitute
for improving the proposal model. First-thirty-row latency across these five
planned cases is .476-.641 seconds, and the slowest publication is 1.097 seconds.
The heaviest complete generation takes 82.90 seconds for 357.80 seconds of audio.

The LN output is exactly identical in all rows, including endpoints. Its realized
LN fraction is .7634 versus the requested .6, an existing amount-control error
that the attack-only response does not address. The Trill output's LN fraction
changes from .0104 to .0529 with LN amount unspecified. An attack-only cost cannot
establish that any resulting held demand is appropriate.

The switch is submitted after publication through 63,999 ms. It requests stars
4.5 and LN fraction .6 on [64000,96000), then restores stars 3 and fraction .2.
Both implementations preserve the entire published prefix. Evaluate the ranges
separately:

| Range | Requested stars / LN fraction | Native scoped proxy / fraction | Planned scoped proxy / fraction |
| --- | --- | --- | --- |
| Before | 3 / .2 | 2.2268 / .2992 | 2.2268 / .2992 |
| Override | 4.5 / .6 | 3.9010 / .5836 | 3.9010 / .5836 |
| Restored | 3 / .2 | 4.4218 / .2064 | 4.2057 / .1992 |

Restored-range concentration improves, but its difficulty still overshoots.
Neither unchanged earlier ranges nor a good LN fraction cancels that error.

Matched Lens views inspect the largest changed Jack publication and the original
maximum-load Tech/Trill contexts. Jack still contains changing chord/single groups;
the selected preference does not impose uniform columns. Both Tech versions are
very dense in the inspected body. The original Trill context includes inner-lane
exchange, then a long repeated column; the planned version replaces that with
broader movement. Removing the overload does not prove that the requested Trill
organization survives. Semantic control is consequently not declared passed.

## Local use

The optional planner consumes a fitted envelope, a native session and its seed:

```python
from ensomi_model.research.controlled_audio_continuation.frontier import ResponsePlanner
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.player_response.envelope import AttackEnvelope

envelope = AttackEnvelope(**calibration["envelope"])
session = ControlledSession(model, mel, duration_ms, controls, seed=seed)
planner = ResponsePlanner(session, envelope, seed=seed)
planner.publish_to(min(8000, duration_ms))
published_rows = tuple(planner.session.rows)
```

Here `calibration` is the parsed study `calibration.json`; weights, canonical Mel
and controls use the native session's existing contracts. Continue through the
planner and apply control updates through `planner.update_controls`. The original
session is a publication snapshot, not the evolving selected session. Generated
calibration/checkpoint assets are local research inputs and are not included in a
fresh clone.

## Interpretation and remaining work

The existing R1 proposal distribution contains less concentrated alternatives
that a bounded future-response comparison can select. Explicit time-based state
and forecasting therefore provide practical value before increasing model size.
The remaining repeated-group bias still belongs to learned R1 organization and
style control. Model training should incorporate the response information while
retaining genuine annotated organization; a selected Stream request is not a
human-observed Stream label.

H is unchanged, so this intervention cannot create musical rests or larger-scale
timing variation. Sustained holding and coordination also need their own response
comparisons. The four-candidate budget leaves some positive-cost futures, and
real ranked exceptions prevent turning this reference into a universal hard mask.
The planner remains an optional research component; the selected runtime model
and default sampler are unchanged.

## Reproduction identity

Baseline source: `7ea3b956ebccdc4d4238bc52e5cb89762402f054`.
Intervention: `24e786b4b8ef8c4752835371bd9fd415c5ed0891`.
Checkpoint SHA-256: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Corpus chart-facts SHA-256: `e529cfd868b4ec4bc337803d1712bbb319d2cba730a47dcfc13cc6a9e8f00f3b`.
Owner: `20260927-sustained-response-planning-v1`.
Harness revision: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`.
Native Stream seeds: `271200 + 10 * audio_index + draw_index`.
The local owner's calibration, run configurations, candidate decisions, complete
row exports and Lens review identify the exact inputs and inspected contexts.
Six focused tests verify the response integral, history/control distinction,
group weighting, zero-cost equivalence and selected-prefix state ownership.
