# Style conditions and control

Ensomi generates musically coherent, playable charts with distinctive
organization, diverse realizations, and composable controls over both style and
chart properties. This page defines the chart identity and its relation to a
prefix, style and chart properties, natural continuation, the measurement
semantics of chart properties, scoped requests with their validity, locality,
overlap, and strength, release and return to natural, and the overall
target.

[notation.md](notation.md) owns chart legality, committed history and supplied
prefixes, the decomposed problem on given head times, and the generation and
property-query interfaces. [gameplay-state.md](gameplay-state.md) owns
gameplay-demand requests and style observations with their vocabulary. Nothing
on this page changes the legal continuation set or a committed decision.

## Style and chart properties

**Style** describes how timing, actions, hand roles, motifs, and transitions
are organized and developed. Its dimensions are open-ended: named, inferred
from references, or learned. Named concepts of a versioned vocabulary, such as
the [style observation](gameplay-state.md#style-observations) concepts of
@ensomi-labs/beatmap-lens, provide partial semantic anchors rather than a
complete style space.

**Chart properties** are scoped measurements or assessments of a chart,
including **LN share** and **section difficulty**. A property target specifies
what the desired result measures, not how its organization must be realized.
Charts with the same property values can express different styles; one style
can support different property values. LN share and the LN coordination
concept show the boundary: LN share counts long-note heads among the objects of
a scope, while LN coordination judges how holds are organized with other
actions.

The [chart identity](#chart-identity) holds both kinds of choice for a whole
chart: its organization, and its preferred levels of chart properties where no
target is in effect.

**Requests** specify desired behavior; **readouts** describe realized results;
**representations** carry the information used to generate and interpret them.
A request is not evidence of its own result. Gameplay-demand requests keep the
separate response semantics of [Controls](gameplay-state.md#controls) and are
interpreted jointly with style directives and property targets.

### Style observations and style directives

A [style observation](gameplay-state.md#presence-and-ordinal-strength) is a
readout: a scoped judgment of a realized chart under a pinned vocabulary
version. A style directive is a request. A directive on a named concept states
a requested assessment under a pinned vocabulary version: present-supporting,
present-prominent, or absent. The assessment of the generated chart is its
readout. Unresolved and unreviewed are observation statuses, not request
values. A concept that no directive names is unspecified, the request-side
counterpart of an unreviewed concept: it is neither requested present nor
requested absent, and it follows the identity.

Named concepts do not exhaust style. Organization outside the named vocabulary
remains part of an identity and can be specified through references or
relative edits. Naming a learned dimension does not by itself give it
mapper-facing meaning.

### Section difficulty and gameplay demand

Section difficulty is a chart property: the value that the evaluator declared
by $\nu$ assigns to the chart over a scope. It is not a demand coordinate, a
summary of a demand state, or a target response. A difficulty target concerns
that evaluator's value only; it requests nothing of the mapper-defined
responses that a demand request refers to in the
[target-response specification](gameplay-state.md#target-response-and-frontier).
Both kinds of request can apply to the same scope; they can be correlated and
may be jointly unattainable. In the
[decomposed problem](notation.md#decomposed-problem-given-head-times), given
head times fix most of a chart's difficulty, so a difficulty target is
attainable only within what they leave open.

## Chart identity

The **chart identity** $\rho$ is the whole-chart choice that the music, any
given head times, and the requests leave open, held across the song. It
comprises:

- the chart's organization: its organizing preferences for timing, actions,
  hand roles, motifs, and transitions, named or not;
- chart-level preferences for property levels where no target is in effect,
  such as how much and how long to hold (LN amount and length) and how densely
  to chord.

The level preferences are preferences, not targets. A property target
overrides them within its scope, and a style directive overrides the
organization attributes it specifies. Elsewhere generation follows the
identity, and its expression adapts to the music, the head times, the history,
and active directives while retaining recognizable character. Given head times
fix part of what an identity would otherwise choose, such as part of the chord
density; the identity governs only the part they leave open
([What given head times fix](notation.md#what-given-head-times-fix)).

An identity is always in effect. It can be supplied, extracted or calculated
from some chart (any chart, such as a style reference, possibly the one a
prefix was taken from), or established by the system when none is supplied at
the first generation call. An identity the system establishes is retained
across continuation calls and can be supplied again.

The identity is a generation input, not chart state: it is neither stored in
nor recomputed from $(H,g)$, so a growing history cannot rewrite it. Local
overrides do not rewrite it either. Changing it is an explicit identity update,
which applies to decisions after the committed boundary at which it is made.
Adopting the organization of an overridden section as the new identity is such
an update; a request never performs one.

A **style reference** informs organization, as a source of $\rho$ or of
reference characteristics in a style directive, without becoming committed
history.

Hand balance belongs to the identity and is directional: the
[mirror](notation.md#canonical-hand-role-coordinates) $\mu$ exchanges the
hands. This formulation fixes neither how an identity transforms under $\mu$
nor a representation of the identity.

### Identity and prefix

A [supplied prefix](notation.md#committed-history-and-exact-replay) is
committed history and nothing else. The identity is a separate input, and a
prefix is not by definition evidence for it. A prefix and a supplied identity
need not agree: the prefix fixes its rows, its no-row decisions, and its open
long-note obligations, and the identity governs the choice they leave open. The
prefix's LN share, difficulty, and other statistics become neither targets nor
identity preferences by appearing in the prefix.

When a prefix is supplied without an identity, the system establishes one as
it does without a prefix. This formulation does not fix whether it takes the
prefix into account.

Sampling randomness is outside the formulation: variation within an identity,
part of the [overall target](#overall-target), is a property of the generation
law.

## Natural continuation

**Natural continuation** is generation that follows the identity where no
directive is in effect, adapted to the music, any given head times, and the
actual history. A style attribute is natural when no style directive on it is
in effect. A chart property is **free** when no target for it is in effect.
Directives on other quantities still apply. With $\mathscr U$ empty and no
demand request, the whole continuation is natural.

A free property follows natural continuation under the identity, including the
identity's preference for its level. Nothing holds it at an earlier value,
compensates for an earlier deviation, or pulls it toward a population average
or the prefix's statistics.

Natural continuation is defined relative to the identity. Without one, nothing
holds the open chart-level choice across the song: a chart that continues only
from its own history can drift away from any such choice, and natural
continuation names no particular law. In the definitions, the identity
therefore comes before every condition stated in terms of natural
continuation: a free property, the
[locality conditions](#activation-expiry-and-locality) on requests, release,
and [return to natural](#release-and-return-to-natural).

## Generation inputs

[Generation](notation.md#generation-and-optional-controls) takes the audio
representation $X$, the committed pair $(H,g)$, the window $W=(g,e]$, the chart
identity $\rho$, a request set $\mathscr U$, and optional gameplay-demand
requests $c_W^{\mathrm{demand}}$. In the
[decomposed problem](notation.md#decomposed-problem-given-head-times) it also
takes the head times $\mathcal T_{\mathrm h}$ and the beat grid $\Gamma$.
Selected output is committed through the
[prefix commit](notation.md#provisional-branches-and-prefix-commit);
uncommitted plans remain revisable.

The **request set** $\mathscr U$ is the finite, possibly empty, set of scoped
requests in effect for a call. Request scopes are in song time, so one request
can govern decisions in several windows. A continuation call that does not
change $\mathscr U$ continues under the same requests.

## Chart properties and measurement semantics

The [property query](notation.md#generation-and-optional-controls)
$\mathsf{Properties}_\nu(\bar H,S)$ reads a chart $\bar H$ over a scope $S$.
$\bar H$ may be a complete chart, a committed prefix, or a committed prefix
extended by a provisional continuation. The measurement semantics $\nu$ state,
for each property:

- which objects or rows are counted, and how an object crossing a scope
  boundary is assigned. For LN share: what counts as a head, whether a hold
  belongs to the scope by its head time, and that each object counts once.
- for an assessed property, the evaluator, the chart context it receives
  beyond $S$, and the gameplay profile it assumes. For difficulty: the
  calculator or panel, whether the surrounding chart is scored, the key count,
  and the playback rate.
- when the readout is undefined, such as LN share over a scope with no head.
  An undefined readout is not zero.
- the deviation measure that compares a readout with a target, and the
  resolution at which the readout can vary. LN share over $n$ heads takes
  multiples of $1/n$.
- how the readout transforms under the [mirror](notation.md#canonical-hand-role-coordinates)
  $\mu$. LN share and a symmetric evaluator are invariant.

Queries and targets use the same property definitions under the same $\nu$, so
$\nu$ fixes what a target means. A readout depends only on chart content and
$\nu$. It does not depend on the requests that produced the chart, and
replacing a request record does not change it. A readout over a partly
materialized scope describes the chart it receives; it does not predict the
value after the scope is complete. $\nu$ assigns a hold whose head lies in $S$
but whose close is not yet materialized.

## Scoped requests

A request is

$$
u=(S,q^{\mathrm{style}},q^{\mathrm{property}},\eta).
$$

Either directive may be omitted, but a request carries at least one.

- $S$ is a scope in song time: one interval $[a,b)$ with $0\leq a<b\leq T$. An
  interval with $b=T$ also contains $T$, so a whole-song scope contains every
  row. Rows and no-row decisions at times in $S$ are in scope: a row at $a$ is
  in scope, a row at $b<T$ is not. This matches the half-open scopes of
  [style observations](gameplay-state.md#scope-context-and-evidence), so an
  observed scope can serve as a request scope and a readout scope. Beats and
  bars may state a scope, which then resolves to song time. Scopes are
  independent of generation windows $W=(g,e]$: a scope may start or end inside
  a window or span several.
- $q^{\mathrm{style}}$ expresses organizational preferences, reference
  characteristics, or relative edits: requested assessments of named concepts,
  characteristics taken from a style reference, or a change relative to what
  generation would produce in $S$ without that directive. A directive declares
  the attributes or characteristics it specifies; the others stay with the
  identity. A directive whose specified characteristics cannot be separated,
  such as a reference that characterizes the whole organization, specifies
  every style attribute.
- $q^{\mathrm{property}}$ is a set of property targets, at most one for each
  property. Each target names a property under a declared $\nu$, a target value
  $v$, and an optional [strength](#target-strength). A target is one value, not
  a range.
- $\eta$ is the request's policy for transitions and composition: its
  [priority](#overlap-and-priority) and any
  [transition intervals](#activation-expiry-and-locality). The default $\eta$
  has no priority and no transition intervals.

For example, a scope can request an LN share and a section difficulty while
keeping the identity's organization, or combine those targets with a directive
for prominent stream organization. A target concerns its declared section as a
whole, not necessarily every smaller window within it.

### Activation, expiry and locality

Activation applies a request to future decisions only. A request is valid only
if the committed boundary $g$ at which it enters $\mathscr U$ precedes its
start:

$$
g<a.
$$

If $\eta$ declares a transition interval before $a$, $g$ must also precede
that interval. At the start of generation $g=0^-$, so a scope may start at $0$;
after a supplied prefix, a scope starts after $g_0$. Adding a request whose
start is at or before $g$ is rejected. The whole scope of a valid request
therefore lies after the committed prefix, and its targets concern $S$,
including rows committed under the same request in earlier windows. Steering
later rows of an active scope toward its section target is not compensation.

A request expires when the committed boundary reaches the end of its scope.
Before its scope starts, while $g<a$, a request may be withdrawn or replaced;
a withdrawn request governs nothing, and a replacement is a new request under
the validity rule. Once the committed boundary reaches $a$, the request is
fixed: it can be neither cancelled nor changed, and it governs its scope until
it expires. Expiry removes the directive and leaves committed history, ongoing
holds, and the identity unchanged.

Outside its scope, generation sees nothing of a request unless its $\eta$
declares a transition. For $u\in\mathscr U$ with scope $[a,b)$ and the default
$\eta$, let $\mathscr U'=\mathscr U\setminus\{u\}$. For every window $W$, with
$X$, $(H,g)$, the head times and grid when given, $\rho$, and the demand
requests fixed:

1. The law of the rows of $Y_W$ before $a$ is the same under $\mathscr U$ and
   $\mathscr U'$.
2. Given the rows of $Y_W$ before $b$, the conditional law of the rows at or
   after $b$ is the same under $\mathscr U$ and $\mathscr U'$.

The laws include no-row decisions, and both compare generation under the same
identity. Together the conditions exclude any presence or announce signal
outside the scope. Condition 1 excludes announcing a request before its scope.
Condition 2 lets a request act after its scope only through the realized
history. It applies to every decision at or after $b$, including the close of
a hold that started inside $S$: that close is decided without the request,
even when $\nu$ counts the hold in $S$. A readout can then depend on decisions
its request does not govern.

$\eta$ may declare transition intervals next to $S$, such as a lead-in before
$a$ or a release interval after $b$, together with how the request influences
generation there. Transitions occur only in intervals that $\eta$ declares. A
transition interval does not change $S$; targets and readouts still concern
$S$.

### Overlap and priority

Requests overlap when their scopes intersect. Adjacent scopes such as
$[0,60)$ and $[60,90)$ do not overlap.

Overlapping directives on the same quantity are invalid: two targets for the
same property, under any $\nu$, or two style directives that specify the same
attribute. A request set that contains such a pair is rejected. There is no
composition and no priority between them. A directive that specifies every
style attribute therefore cannot overlap any other style directive. A caller who
wants different values for one property in different parts of the song states
scopes that do not overlap.

Overlapping directives on different quantities all apply: an LN-share target
and a difficulty target, a style directive and a property target, or style
directives on different attributes. They can be correlated and may be jointly
unattainable. When they cannot all be met, the shortfall of each directive
concerned is declared with the generated continuation, not silently applied.

Priority is optional and exists only for that case. The priority in $\eta$
orders overlapping requests whose directives on different quantities cannot
all be met: the higher-priority request's directives are followed first, and
adherence to a lower-priority request's directives is never obtained at their
expense. Directives within one request have no order among them, and when
overlapping directives without an order cannot all be met, no precedence
applies; the shortfall is still declared.

Within its scope, a style directive takes precedence over the identity for the
attributes it specifies, and a property target over the identity's preference
for that property. This is not a priority relation: the identity is not a
request.

### Unspecified, zero and absent

| Input | Unspecified | Explicit |
| --- | --- | --- |
| A property | Free: follows natural continuation under the identity | A target value, including $0$ |
| A named style concept | Follows the identity | A requested assessment, including absent |
| Target strength | The default level | A stated level, above or below the default |
| $\eta$ | No priority and no transition intervals | A declared priority or transition |
| Chart identity $\rho$ | Established by the system and retained | Supplied, or extracted or calculated from a chart |
| Demand request | None | A request under the response specification |
| Prefix | The empty prefix at $0^-$ | A supplied committed pair $(H_0,g_0)$ |
| Head times | Generated (the full problem) | Given (the decomposed problem) |

An explicit zero is a target. LN share $0$ asks that every head in $S$ be a
tap; since an undefined readout is not zero, a scope with no head does not
meet it. Explicit absence of a named concept asks that the realized section be
assessed absent for that concept; an identity in which the concept is rare
does not make that request. Neither is a release. Releasing a directive makes
its attribute or property unspecified.

## Target strength

A property target's strength is an ordered request level for how strongly the
target governs generation in its scope, weighed against everything else
generation balances. Strength applies to property targets only; a style
directive carries no strength. Strength does not change the target value,
$\nu$, the scope, $\eta$, or the priority.

- **Default.** An unspecified strength is the default level. The default is the
  generator's primary operating point: the one it is trained for, which
  balances overall playability, style, and control over every active
  directive. It is neither the target met exactly nor maximal adherence: at
  the default, a readout may deviate from the target where meeting it would
  cost more of that balance. This formulation does not fix how close the
  default comes to a target.
- **Higher levels.** A higher level weighs the target more against that
  balance. It asks for a smaller deviation than the default leaves and accepts
  a greater cost in playability, fidelity to the identity, musical
  correspondence, the organization of unspecified attributes, and variation
  within the identity. It never trades legality or committed obligations.
  Where the default already reaches the resolution of $\nu$, a higher level
  has nothing left to reduce.
- **Lower levels.** A level below the default weighs the target less than the
  default balance does, so it may leave a larger deviation in favor of
  everything else generation balances. It is still a target, not a release.
- **Monotone.** With every other input fixed, raising the strength must not
  increase the expected deviation under $\nu$.
- **Order, not scale.** Levels are ordered. They are not equally spaced,
  probabilities, or numerical weights. A numerical strength scale needs its
  own declared calibration for each property.
- **No zero strength.** A strength belongs to a target and cannot be stated
  alone. No level disables a target; removing a target's effect is a release.

Strength is distinct from the other request parameters:

- **Target value.** The value states the desired readout and the strength how
  strongly it is pursued. LN share $0.2$ at a higher strength requests no more
  long notes than LN share $0.2$ at the default.
- **Priority.** Priority orders overlapping requests on different quantities
  that cannot all be met. Strength trades one target against the default
  balance, not against other directives: a higher-strength target of lower
  priority still yields to a higher-priority directive.
- **Transition policy.** $\eta$ decides where and when the request applies.
  The stated strength applies throughout $S$; behavior in a transition
  interval is whatever $\eta$ declares there.
- **Sampling temperature.** Temperature changes the randomness of every
  decision; strength concerns the adherence of one target.
- **Ordinal strength of an observation.** The two are analogous as ordered
  degrees. Present-prominent and present-supporting say how strongly a concept
  characterizes a realized section; target strength says how strongly a target
  should govern generation. A higher strength does not ask the property to
  become a prominent characteristic of the section, a prominent realized
  concept does not reveal a strength, and the ordinal assessment scale does
  not define a strength scale.
- **Section scope.** Strength concerns the target over $S$. It does not make
  the target apply to every smaller window.

## Release and return to natural

**Returning to natural releases selected directives; it does not reset the
chart or its identity.** Ending a style directive returns the attributes it
specified to natural continuation, while active LN-share, difficulty, and
gameplay-demand targets remain. Releasing a property target frees that
property without changing the identity or any style directive. Unspecified is
not zero or absent.

Continuation uses the actual post-control history, including ongoing holds and
developed motifs. It returns to the identity's organizing and level
preferences, not necessarily to the prefix's local statistics, a
population-average style, or the trajectory that would have occurred without
the request. Returning to the identity's level preference is not compensation:
compensation would offset the released value, for example by holding less than
the identity prefers after a scope that held more.

An expired directive ceases to be an objective. Its historical consequences
can persist: a hold may end after the scope, and a motif may be completed
naturally. A persisting consequence is not continued enforcement, and expiry
does not require an immediate opposite behavior. An expired target creates no
obligation to compensate in later sections. The second locality condition in
[Activation, expiry and locality](#activation-expiry-and-locality) states
these requirements on the generation law.

## Overall target

The target is **diversity between identities, variation within an identity,
composable control over organization and properties, and interpretable
outcomes**:

- **Diversity between identities.** For the same music, head times, history,
  and property targets, different identities produce recognizably different
  charts.
- **Variation within an identity.** Each identity supports multiple coherent
  realizations.
- **Composable control.** Changing a property target preserves the rest of the
  identity where compatible; changing style preserves active property targets
  where compatible. Local control and its withdrawal preserve continuity,
  playability, and committed obligations. A higher target strength may lower
  playability as [Target strength](#target-strength) describes; continuity and
  committed obligations hold at every level. Playability is a quality
  judgment; it does not change the legal continuation set.
- **Interpretable outcomes.** The identity in effect, the requests that
  governed, and every declared shortfall accompany each generation, and the
  property query reads the realized result under the definitions its targets
  use.

The system supports extracting and reusing identities and style references,
editing scoped directives, and reading both realized organization and chart
properties. Representations retain long-range organizing choices, adapt their
local expression, and mediate requested changes. Named attributes coexist with
additional learned organization.
