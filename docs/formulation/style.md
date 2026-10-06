# Style and controllable generation

Ensomi generates musically coherent, playable charts with distinctive
organization, diverse realizations, and composable controls over both style and
chart properties. This page defines style identity, the baseline style, chart
properties and their readouts, scoped requests, their composition and release,
and the roles of the chart seed and the random seed.

[notation.md](notation.md) owns chart legality, committed history, the chart
seed's commit semantics, and the generation and property-query interfaces.
[gameplay-state.md](gameplay-state.md) owns gameplay-demand requests and style
observations with their vocabulary. Nothing on this page changes the legal
continuation set or a committed decision.

## Style and chart properties

**Style** describes how timing, actions, hand roles, motifs, and transitions
are organized and developed. Its dimensions are open-ended: named, inferred
from references, or learned. Named concepts of a versioned vocabulary, such as
the [style observation](gameplay-state.md#style-observations) concepts of
@ensomi-labs/beatmap-lens, provide partial semantic anchors rather than a
complete style space.

A **style identity** is an organizing preference sustained across a declared
scope. Its expression adapts to the music, property targets, and history while
retaining recognizable character.

**Chart properties** are scoped measurements or assessments of a chart,
including **LN share** and **section difficulty**. A property target specifies
a property of the desired result, not how its organization must be realized.
Charts with the same property values can express different styles; one style
can support different property values. LN share and the LN coordination
concept show the boundary: LN share counts long-note heads among the objects of
a scope, while LN coordination judges how holds are organized with other
actions.

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
requested absent.

Named concepts do not exhaust style. Organization outside the named vocabulary
remains part of an identity and can be specified through references or
relative edits. Naming a learned dimension does not by itself give it
mapper-facing meaning.

### Section difficulty and gameplay demand

Section difficulty is a chart property: the value that the evaluator declared
by $\nu$ assigns to the chart over a scope. It is not a demand coordinate, a
summary of a demand state, or a target response. A difficulty target constrains
that evaluator's readout, while a demand request refers to quantities in the
[target-response specification](gameplay-state.md#target-response-and-frontier).
Both can apply to the same scope; they can be correlated and may be jointly
unattainable. Meeting a difficulty target shows that the declared evaluator
reports the requested value. It does not show that mapper-defined responses
changed as intended.

## Generation inputs

[Generation](notation.md#generation-and-optional-controls) takes the audio
representation $X$, the committed pair $(H,g)$, the window $W=(g,e]$, a baseline
style $\rho$, a request set $\mathscr U$, and optional gameplay-demand requests
$c_W^{\mathrm{demand}}$. Selected output is committed through the
[prefix commit](notation.md#provisional-branches-and-prefix-commit);
uncommitted plans remain revisable.

The **baseline style** $\rho$ is a reusable specification of one style
identity, obtained from style references, explicit preferences, evidence from a
chart seed, or sampling (see
[Chart seed, baseline and random seed](#chart-seed-baseline-and-random-seed)).
A baseline is always in effect. When none is supplied at the first generation
call, the system establishes one and retains it across continuation calls.
$\rho$ is a generation input, not chart state: it is neither stored in nor
recomputed from $(H,g)$, so a growing history cannot rewrite it.

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

Queries and targets use the same property definitions under the same $\nu$. A
readout depends only on chart content and $\nu$. It does not depend on the
requests that produced the chart, and replacing a request record does not
change it. A readout over a partly materialized scope describes the chart it
receives; it does not predict the value after the scope is complete. $\nu$
assigns a hold whose head lies in $S$ but whose close is not yet materialized.

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
  baseline. A directive whose specified characteristics cannot be separated,
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
retaining the baseline style, or combine those targets with a directive for
prominent stream organization. A target concerns its declared section as a
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
after a chart seed, a scope starts after $g_0$. Adding a request whose start
is at or before $g$ is rejected. The whole scope of a valid request therefore lies after the committed prefix, and its targets
concern $S$, including rows committed under the same request in earlier
windows. Steering later rows of an active scope toward its section target is
not compensation.

A request expires when the committed boundary reaches the end of its scope.
Cancellation removes it at a committed boundary $g'$ before then. A request
cancelled before its start governs nothing. A request cancelled inside its
scope governs only the decisions through $g'$; its targets cease to be
objectives, and the record keeps the readout of its committed part, marked as
cancelled rather than as a measure of adherence. A changed directive is a new
request and must satisfy the validity rule: a replacement for a request
cancelled at $g'$ starts after $g'$, and the time between $g'$ and that start
is governed by neither. Expiry and cancellation remove the directive and leave
committed history, ongoing holds, and the baseline unchanged.

Outside its scope, generation sees nothing of a request unless its $\eta$
declares a transition. For $u\in\mathscr U$ with scope $[a,b)$ and the default
$\eta$, let $\mathscr U'=\mathscr U\setminus\{u\}$. For every window $W$, with
$X$, $(H,g)$, $\rho$, and the demand requests fixed:

1. The law of the rows of $Y_W$ before $a$ is the same under $\mathscr U$ and
   $\mathscr U'$.
2. Given the rows of $Y_W$ before $b$, the conditional law of the rows at or
   after $b$ is the same under $\mathscr U$ and $\mathscr U'$.

The laws include no-row decisions. Condition 1 excludes announcing a request
before its scope. Condition 2 lets a request act after its scope only through
the realized history. It applies to every decision at or after $b$, including
the close of a hold that started inside $S$, even when $\nu$ assigns that hold
to $S$. A readout can then depend on decisions its request does not govern;
the [generation record](#generation-record) lists such objects.

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
unattainable. Priority exists only for that case. The priority in $\eta$ orders
overlapping requests whose directives on different quantities cannot all be
met: the higher-priority request's directives are followed first, and adherence
to a lower-priority request's directives is never obtained at their expense.
Priority is optional. Directives within one request have no order among them,
and when overlapping directives without an order cannot all be met, no
precedence applies. In every case the shortfall is declared in the
[generation record](#generation-record), not silently applied.

Within its scope, a style directive takes precedence over the baseline for the
attributes it specifies. This is not a priority relation: the baseline is not a
request.

### Unspecified, zero and absent

| Input | Unspecified | Explicit |
| --- | --- | --- |
| A property | Free: no target is in effect | A target value, including $0$ |
| A named style concept | Follows the baseline | A requested assessment, including absent |
| Target strength | The default level | A stated level |
| $\eta$ | No priority and no transition intervals | A declared priority or transition |
| Baseline $\rho$ | Established by the system and retained | Supplied |
| Demand request | None | A request under the response specification |
| Chart seed | The empty prefix at $0^-$ | A supplied committed pair $(H_0,g_0)$ |

An explicit zero is a target. LN share $0$ asks that every head in $S$ be a
tap; since an undefined readout is not zero, a scope
with no head does not meet it. Explicit absence of a named concept asks that
the realized section be assessed absent for that concept; a baseline in which
the concept is rare does not make that request. Neither is a release.
Releasing a directive makes its attribute unspecified.

## Target strength

A target's strength is an ordered request level for how strongly the target
governs generation in its scope, weighed against everything else generation
balances. Strength does not change the target value, $\nu$, the scope, $\eta$,
or the priority.

- **Default.** An unspecified strength is the default level. The default is the
  generator's primary operating point: the one it is trained and validated
  for, which balances overall playability, style, and control over every
  active directive. It is neither the target met exactly nor maximal
  adherence: at the default, a readout may deviate from the target where
  meeting it would cost more of that balance. Default adherence is therefore a
  measured property of a generator, reported for each property; this
  formulation does not fix its value.
- **Higher levels.** A higher level weights the target more against that
  balance. It asks for a smaller deviation than the default leaves and accepts
  a greater cost in playability, fidelity to the baseline, musical
  correspondence, the organization of unspecified attributes, and variation
  across random seeds. It never trades legality or committed obligations. With
  every other input fixed, raising the strength must not increase the expected
  deviation. Where the default already reaches the
  resolution of $\nu$, a higher level has nothing left to reduce.
- **Order, not scale.** Levels are ordered. They are not equally spaced,
  probabilities, or numerical weights. A numerical strength scale needs its
  own declared calibration for each property. No level below the default is
  defined.
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
  decision; strength concerns the adherence of one target. A strength
  comparison holds temperature fixed.
- **Lens prominence.** The two are analogous as ordered degrees.
  Present-prominent and present-supporting say how strongly a concept
  characterizes a realized section; strength says how strongly a target should
  govern generation. A higher strength does not ask the property to become a
  prominent characteristic of the section, a prominent realized concept does
  not reveal a strength, and the ordinal assessment scale does not define a
  strength scale.
- **Section scope.** Strength concerns the target over $S$. It does not make
  the target apply to every smaller window.

A single readout cannot show strength. An effect is measured against the
default: generations that differ only in one target's strength, under the
same random seeds. The effect is the reduction of the deviation distribution
beyond the variation between seeds. The cost is the change in what strength
may trade: playability judgments, style readouts against the baseline, musical
correspondence, other targets' deviations, and variation within the identity.
A reduction obtained by collapsing variation across seeds shows in the last of
these. When the default deviation is already at the resolution of $\nu$, no
effect is expected, and its absence is not a failure.

[Controls](gameplay-state.md#controls) separates style adherence from requested
style tendency. Adherence of a style directive is a quantity of the same kind
as target strength; this page defines strength for property targets only.

## Chart seed, baseline and random seed

A [chart seed](notation.md#committed-history-and-exact-replay)
$\sigma=(H_0,g_0)$ is an optionally supplied, legally replayable initial
prefix with a fixed-through time. Its rows and no-row decisions through $g_0$
are committed; continuation preserves them and inherits any open long-note
obligations. A seed conditions generation as committed history and as evidence
for the baseline; it is not a request. Generation without a seed starts from
the empty prefix at $0^-$, and its baseline is established without seed
evidence.

The seed initializes action history and provides evidence for the baseline
style $\rho$. A baseline may instead be specified explicitly or extracted from
style-only references. Without a supplied baseline, the system establishes one
from the seed's evidence when a seed is given, and by sampling otherwise. A
seed need not uniquely determine an identity; sampling then chooses among the
identities it supports. The seed's LN share, difficulty, and other statistics
do not become continuing targets unless requested. When both a baseline and a
seed are supplied, the supplied baseline governs and the seed remains history.

The baseline persists across continuation calls and local overrides until it
is explicitly changed; its expression continues to adapt. An established
baseline is recorded and can be supplied again. A baseline update is an
explicit change of $\rho$ that applies to decisions after the committed
boundary at which it is made. Adopting the organization of an overridden
section as a new identity is such an update; a request never performs it.

A **style reference** informs organization, as a source of $\rho$ or of
reference characteristics in a style directive, without becoming committed
history. A **random seed** selects sampling randomness, including any sampling
that establishes a baseline. It is distinct from the chart seed and from style
identity: generation with a recorded $\rho$ and a different random seed gives
another realization of the same identity.

## Natural continuation and return to natural

**Natural style continuation** means that no style directive is in effect:
generation is conditioned on the baseline, and active property targets and
demand requests still apply. A **free** property has no target in effect.
Nothing holds it at an earlier value, compensates for an earlier deviation, or
pulls it toward a population average or the seed's statistics; its value is
whatever generation conditioned on the baseline, music, history, and active
directives produces. With $\mathscr U$ empty and no demand request, coherent
baseline-conditioned variation continues.

**Returning to natural releases selected directives; it does not reset the
chart or its identity.** Ending a style directive restores baseline-conditioned
generation for the attributes it specified, while active LN-share, difficulty,
and gameplay-demand targets remain. Releasing a property target frees that
property without changing the baseline or any style directive. Unspecified is
not zero or absent.

Continuation uses the actual post-control history, including ongoing holds and
developed motifs. It returns to the baseline's organizing preference, not
necessarily to the seed's local statistics, a population-average style, or the
trajectory that would have occurred without the request.

An expired or cancelled directive ceases to be an objective. Its historical
consequences can persist: a hold may end after the scope, and a motif may be
completed naturally. Persistence alone does not prove continued enforcement,
and expiry does not require an immediate opposite behavior. An expired target
creates no obligation to compensate in later sections. The second locality
condition in [Activation, expiry and locality](#activation-expiry-and-locality)
states these requirements on the generation law.

Local overrides do not rewrite the baseline. Adopting their result as the
identity is a separate baseline update.

## Identity and control requirements

The system supports extracting and reusing style references, editing scoped
directives, and reading both realized organization and chart properties.
Representations retain long-range organizing choices, adapt their local
expression, and mediate requested changes. Named attributes coexist with
additional learned organization.

For the same music, history, and property targets, different identities
produce recognizably different charts, and each identity supports multiple
coherent realizations. Changing a property target preserves identity where
compatible; changing style preserves active property targets where compatible.
Local control and its withdrawal preserve continuity, playability, and
committed obligations. A higher target strength may lower playability as
[Target strength](#target-strength) describes; continuity and committed
obligations hold at every level. Playability is a quality judgment; it does
not change the legal continuation set.

The target is **diversity between identities, variation within identities,
composable control over organization and properties, and interpretable
generation outcomes**.

### Generation record

Readouts expose the relationship between intended and realized results. Each
generation records:

- the baseline in effect and its source (supplied, extracted from references,
  established from a chart seed, or sampled), and any baseline update with its
  boundary;
- the chart seed and the random seed;
- every request with its scope, directives, $\nu$, target values, strengths,
  and $\eta$, and the boundary of any cancellation;
- for each target, its readout under $\nu$ over $S$ and the deviation, with
  undefined readouts recorded as undefined and cancelled targets marked;
- overlaps that could not all be met, and the shortfall of each directive;
- objects that cross the end of a scope, such as a hold started inside it and
  closed after it, so that a readout can separate persistence from
  enforcement.

### Evaluation

| Question | Required comparison |
| --- | --- |
| Does a target hold? | Readouts under $\nu$ over $S$ against the target across random seeds, with the deviation distribution and undefined readouts kept separate; at the default level this distribution is the generator's reported adherence, not a pass threshold |
| Does a higher strength have an effect? | The same inputs at two strengths under the same random seeds: the deviation and every cost readout of [Target strength](#target-strength) |
| Is a request invisible outside its scope? | Generation with and without the request: the same law before $a$, and the same conditional law after $b$ given the same history |
| Is a released property free? | Readouts after the scope against natural continuations from the same post-scope history: no hold of the released value and no shift opposite to it |
| Does return to natural reach the baseline? | Continuation after release against natural continuations under the same $\rho$ from the same post-override history, not against pre-override or seed statistics |
| Do identities differ, and does each vary? | Variation between baselines against variation across random seeds for fixed music, history, and targets, judged by humans or a validated style recognizer |
| Is identity kept under a property change? | Style readouts with and without the target, under the same $\rho$, history, and random seeds |
| Are targets kept under a style change? | Property readouts of an active target with and without the style directive |
| Are unspecified, zero, and absent distinct? | The three conditions on the same inputs, with readouts that distinguish them |

Recognizability of an identity or a style change needs human judgments or an
independently validated recognizer, as for
[style directives](gameplay-state.md#evaluation-questions). Changes in note
count, global difficulty, or decoding entropy alone do not establish it.
