# Continuation acceptance from committed action responses

The row actor's `frontier2` energy is not an independently trained gameplay
response. `RowConsequence` contributes a residual to row logits and is trained
through their NLL. The deployed demo on the stream-generation-benchmark branch
and the native qualification runner both instantiate `ControlledSession`
directly; neither invokes `ResponsePlanner`. Their generated charts therefore
do not receive that planner's continuation comparison.

The earlier planner also has a narrower objective than its name suggests:
four-second futures are compared by per-finger attack-rate excess, with at
most four R/R1 retries on the same H skeleton. Release recovery and coordination
do not enter this score. A zero attack excess ends search, and the original
implementation published the least excess even when all candidates failed.
These mechanisms explain missing protections, not the fraction of every
generation defect attributable to one training stage.

## Separate the actor from its response reference

For an actor score $s_\theta(a)$ and a fixed preference $c(a)$, imitation of
$p(a)\propto\exp(s_\theta(a)-c(a))$ can learn
$s_\theta(a)=\log p_{\rm data}(a)+c(a)$ where its function class permits it.
Likewise, NLL on a sum of actor and consequence scores cannot independently
identify either summand as player demand. A soft cost inside the imitation law
is not a separately enforced continuation constraint.

The new [action response](../../src/ensomi_model/research/player_response/action_response.py)
keeps a chart-only, request-independent committed state. A future is evaluated
against a separate ranked-data reference at the requested difficulty. Response
observations, reference calibration and actor likelihood have separate owners.
This remains a partial empirical response model under the
[V3 canonical profile](../formulation/gameplay-state.md); it does not claim
identified individual physiology or complete musical quality.

## Event state and response hypothesis

The exact replay retains occupied fingers and actual LN origins. Positive
event responses retain continuous-time memory:

$$
z_{\tau}^{-}=e^{-\Delta/\tau}z_{\tau}^{+},\qquad
z_{\tau}^{+}=z_{\tau}^{-}+\frac{b(x^{-},a)}{\tau}.
$$

The numerical memory scales are 250, 1000, 4000 and 16000 ms, not asserted
human recovery constants. No-row advances compose exactly. A control change
does not clear memory, and one intervening action does not erase a repeated
finger's accumulated response.

The event vector contains:

| Coordinate | Observation |
| --- | --- |
| Attack and release counts | Per finger; TAP and LN press are both attacks |
| HH speed | $100\,\mathrm{ms}/\Delta_{HH}$ at an actual attack |
| RH speed | The same reciprocal gap for the first attack after an actual LN release |
| HR speed | Reciprocal duration when an actual LN ends |
| Hand events | One joint action row for each participating hand |
| Hand turnover | Changed-finger count times reciprocal gap since that hand's previous action row |
| Attack with held partner | An attack while the other finger on the same hand continues holding |

Missing predecessors contribute no invented gap. TAP release times are unknown
and are never fabricated. A simultaneous chord is one hand event, not a
column-order sequence. The reciprocal-gap response is a convex speed hypothesis:
it lets equal attack counts with different recovery geometry have different
responses. Its scale and permissive reference must be checked on real charts.
It is not a measured force law.

Transient coordinates decay during holding, while the exact occupied fingers
and their origins remain available. This distinction keeps a long held role
different from four free fingers without declaring sustained occupancy alone
to be an overload.

## Corpus calibration and future evaluation

`source_peaks` computes the same response maxima from a legal source chart.
Its eight-second numerical blocks carry the previous state; they are not resets
or musical sections. A reference fits equal-song then equal-chart quantiles
inside declared difficulty bands and retains raw quantiles and counts before
monotone interpolation.

The reference compares each response kind at each memory scale. Over a private
future, excess is integrated in actual time, including inherited load and silent
intervals. For an exponentially decaying ratio $r$ to its reference, its positive
squared-excess integral over duration $d$ is

$$
a=\min(d,\tau\log\max(r,1)),\qquad
I=\tau\left[\frac{r^2}{2}(1-e^{-2a/\tau})
-2r(1-e^{-a/\tau})\right]+a.
$$

Each control range retains its own observations and cost. Per-kind contributions
remain visible. A missing difficulty request is unscored, rather than certified
safe because its selection cost happens to be zero.
Calibration is not an automatic human judgment: source false rejections,
known generated failures, expressive LN/Tech controls and ordinary held-role
examples must be checked before this reference can be used for a release.

The first actual calibration used 5,556 charts and held out 1,368 charts by
song group from a 6,924-chart ranked 2–6★ cohort. Independently requiring every
coordinate to stay below its marginal 99th percentile rejected 113 held-out
charts (8.3%), including two of nine previously inspected source controls.
Although it detected several fixed generated failures, those source errors
prevent adopting this pointwise acceptance rule.

### Added recovery work

A brief high response and sustained repeated loading should not be treated as
the same event. Define the recovery potential $\Psi_D(z)$ as the integrated
reference excess that remains if no further actions occur:

$$
\Psi_D(z)=\sum_{\tau,c}
\frac{\tau}{|\mathcal T|}
\left[\frac{a_{\tau,c}^2}{2}-a_{\tau,c}+\log(1+a_{\tau,c})\right],
\qquad
a_{\tau,c}=\max(z_{\tau,c}/r_{\tau,c}(D)-1,0).
$$

An action contributes
$w_k=\Psi_{c(t_k)}(z_k^+)-\Psi_{c(t_k)}(z_k^-)$.
The sum over a future measures newly added recovery work, rather than charging
the candidate for overload that was already committed. It includes consequences
after the forecast horizon without inventing later actions or LN endpoints.
For a constant request it satisfies

$$
\sum_k w_k =
\int_{t}^{e}\operatorname{excess}_D(z(s))\,ds
+\Psi_D(z(e))-\Psi_D(z(t)).
$$

This identity ties immediate decisions, accumulated state and a terminal value
to one explicit response hypothesis. It remains an empirical model, not a
physiological measurement. Its admissible work over real-time horizons is
calibrated from ranked continuations separately from the coordinate references.
Per-kind work remains inspectable. Short request ranges use a declared enclosing
calibration horizon, not an unverified linear rescaling of an impulse budget.

Since work is already normalized by the difficulty-specific response reference,
its tolerance is pooled across the fitting source groups. A first attempt to
force these normalized tolerances upward with difficulty incorrectly transferred
the 2★ normalization tail to every higher request; that result was retained
and rejected. The shared 99th-percentile tolerance uses fitting groups only
and preserves monotone admission as the physical reference becomes more
permissive with difficulty.

The resulting 4 s work tolerance is .049906. It rejects 21/1368 held-out source
charts (1.54%); all nine previously inspected source controls pass, including
the short-LN specialist examples. Joint-release80 STYX/Blizzard/Stream have
maximum 4 s added work .11388/.12119/1.56594, so all three fail this pressure
coordinate. This is still partial response calibration, not complete
playability or a held-out estimate of musical quality.

## Candidate commitment

[ResponsePlanner](../../src/ensomi_model/research/controlled_audio_continuation/frontier.py)
can use either its historical attack reference or the action reference.
It now raises `NoAcceptableContinuation` if every private candidate fails.
The published session and committed response state stay unchanged; the failed
decision retains the candidate reports. It no longer silently publishes the
least-bad rejected proposal.

This behavior exposes infeasible or poorly proposed futures; it does not by
itself repair them. The caller still needs a bounded recovery strategy and
enough proposal support to meet realtime deadlines. H is currently unchanged
across retries, so an excessive skeleton can defeat every R1 candidate.
Nothing in this implementation establishes that more retries alone will solve
that case.

The action reference is not yet wired into the demo's default path or declared
qualified. It also does not teach an ordinary one-LN-plus-TAP role arrangement:
that requires the proposal to learn persistent role relationships and musical
timing. Candidate acceptance and proposal learning must both succeed.

## Native search and independent proposal guidance

An actual native run with the joint-release80 weights, original full audio,
controls and seeds completed STYX after selecting one alternative at72s.
Blizzard exhausted four candidates at100s; Stream exhausted them at224s.
The rejected futures were not committed, and Blizzard's two incoming open holds
were preserved. These failures demonstrate insufficient proposal coverage for
that bounded search; four rejected samples do not prove that H is infeasible.
Stream also had a2.77s service window, exceeding the2s target.

[ResponseGuidedSession](../../src/ensomi_model/research/controlled_audio_continuation/response_guidance.py)
uses the same committed response state to add a candidate energy

$$
g(P,t,a)=\alpha\,
\frac{\Psi_D(z^{+}(P,t,a))-\Psi_D(z^{-}(P,t))}
     {B_D(4000\,\mathrm{ms})}.
$$

It compares all complete R1 rows while preserving the actor's exact support.
The empty action adds zero work. For joint R, this energy enters before
wait/release marginalization and again in the corresponding conditional mark
law. A single eligible release can therefore be delayed rather than having
its cost cancel after R has already committed.

The response reference is independently calibrated and frozen; this guidance
is not an actor parameter optimized by row imitation. Its strength is an
explicit sampling-policy choice. It does not certify a future, so the same
four-second continuation acceptance remains in place. H, cardinality ownership,
audio inputs and the publication protocol do not change. Forks own their
response state; rejected branches cannot load the committed player's state.

Training from these improved or rejected rollouts needs the correct trajectory
law. The existing row-only trace likelihood assumes other timing factors can
be held fixed or cancel. In `r1_joint` mode R's timing also depends on R1
parameters, so a row-only policy-gradient update would omit part of its credit.
Any such learning must include R survival/event terms on actual partial futures,
without fabricating crop-end LN closures.

The [joint trace scorer](../../src/ensomi_model/research/controlled_audio_continuation/joint_trace.py)
now reconstructs the R survival/event factors and chosen R1 row factors on an
actual, possibly open continuation, conditional on a fixed complete H proposal.
Observed empty time still contributes release survival. A forced deadline has
unit event mass but retains its mark probability. This scores the ordinary
proposal with explicitly declared preferences; independent guidance, latent
plans and response-selected trajectories require their corresponding laws.

## Remaining native failures

Adding immediate guidance completed Blizzard, but did not restore requested
composition. STYX/Blizzard had stars 3.901/4.506 and LN fractions .689/.600,
against requested fractions .485/.838. Stream exhausted four candidates at146s.
The fixed Lens scopes retained short/medium-LN bodies and changing held roles.
No checkpoint was promoted.

Substituting only actual source H timestamps completed STYX/Blizzard at
4.116/4.638 stars, with LN fractions .720/.633. Stream still stopped at58s;
in its fixed [18335,23335) ms scope, 51 of52 heads became LN, with median130ms
duration and one-H span. Correcting timing alone is therefore insufficient.
No source actions, LN types, endpoints or startup seed entered that diagnostic.
The Zenithfall source is 5.873 stars; its H is not a proven 4-star-feasible timing
reference. That case isolates a timing input change, not sole responsibility
for pressure failure. STYX/Blizzard references are 4.004/3.928 stars.

Immediate work cannot distinguish TAP from LN birth on the same free columns:
their current attack impulse is identical. Their occupancy and later coordination
differ. Waiting also adds zero immediate work even when an earlier release could
improve future recovery. The explicit rollout can observe those differences,
but four samples need not contain a useful alternative. This reactive energy is
not a prospective action value or a learned persistent role plan.

The aligned four-second acceptance also does not bound every sliding window.
Source-H Stream's committed prefix reached maximum 4 s added work .051639 and
maximum 8 s work .087604, above respective references .049906/.061507. A rolling
budget across publication boundaries remains necessary; lowering pressure on
selected horizons alone is not a complete runtime guarantee.
