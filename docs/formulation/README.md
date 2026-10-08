# Ensomi V3 formulation

Ensomi's target is to generate musically coherent, legal and playable 4K
choreography from complete audio and committed chart history, with distinctive
organization, diverse realizations, and composable controls over both style
and chart properties. Generation always follows a chart identity; scoped style
directives, chart-property targets, and gameplay-demand requests are optional.
The materialized output is a sequence of complete timed rows; implementations
may use different representations and generation methods.

In the full problem the system generates every row time. Ensomi's generation
is posed on a [decomposed problem](notation.md#decomposed-problem-given-head-times):
the head times are given, and the arrangement on them is generated.

This directory separates the generation contract, the gameplay semantics that
guide preference among legal continuations, and the semantics of style
conditions and control:

| Document | Ownership |
| --- | --- |
| [Generation notation](notation.md) | Chart language, absolute time, exact legality, committed history and supplied prefixes, legal continuations, the generation and property-query interfaces, prefix commits, and the decomposed problem on given head times with its assumptions |
| [Gameplay demand and style](gameplay-state.md) | Target continuation responses, the gameplay frontier, demand representations, section-style observations, demand requests, and the gameplay semantics every control respects |
| [Style conditions and control](style-conditions-and-control.md) | The chart identity and its relation to a prefix, style and chart properties with their measurement semantics, natural continuation, scoped requests with validity, locality, overlap, priority and target strength, release and return to natural, and the overall control target |

## Open questions

Target responses take semantic priority over the state used to represent them.
The response specification is open: it is to define, from a mapper's
perspective, what gameplay demand should and should not describe. Source-backed
style judgments and concrete chart contrasts help identify the responses worth
preserving.

The gameplay frontier names the response function over possible legal futures.
A finite demand state and its dynamics are candidate representations of that
function, and their adequacy is defined against the independently defined
target responses.

Style dimensions are open-ended. Named concepts anchor part of the style
space, and references and learned organization supply the rest. The
[style formulation](style-conditions-and-control.md) states what the chart
identity and control must satisfy without fixing a representation of the
identity or a calibrated strength scale.

The decomposed problem rests on the [assumptions](notation.md#assumptions)
stated with it; the
[research document](../research/head_time_decomposition.md) records the
evidence on them. The full problem, in which the head times are generated as
well, remains the target.

This formulation does not provide a completed response specification, a
calibrated demand scale, or an executable V3 model.

## Authority

The chart language and commit rules are formal constraints. The canonical
gameplay profile and annotation meanings are declared conventions. The
decomposed problem is a declared problem scope; its assumptions are stated
with it. Identity and request semantics are interface
requirements: the chart identity as a generation input outside chart state,
scope membership, request validity, locality outside a scope, overlap,
priority, release, the order of strength levels, and the shared definition of
targets and readouts under declared measurement semantics. Response
definitions, representation adequacy, style and identity representations, a
generator's default operating point, the calibration of strength levels, and
realized control behavior have the research status stated in their owning
sections.

The formulation defines the problem, not how results are checked. Evaluation
protocols, concrete candidate generators, state encoders, style encoders,
dynamics, training objectives, pooling models, decoding methods, and
experimental results belong in [research documentation](../research/). A
particular implementation's reachable charts do not redefine the legal chart
space.

The [repository README](../../README.md) defines the V3 status and the boundary
around retained pre-V3 systems. Legacy code and local generated assets are not
sources of V3 specification.
