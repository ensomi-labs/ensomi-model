# Ensomi V3 formulation

Ensomi's target is to generate musically coherent, legal and playable 4K
choreography from complete audio and committed chart history, with distinctive
organization, diverse realizations, and composable controls over both style
and chart properties. Generation always follows a baseline style; scoped style
directives, chart-property targets, and gameplay-demand requests are optional.
The materialized output is a sequence of complete timed rows; implementations
may use different representations and generation methods.

This directory separates the generation contract, the gameplay semantics that
guide preference among legal continuations, and the semantics of style
conditions and control:

| Document | Ownership |
| --- | --- |
| [Generation notation](notation.md) | Chart language, absolute time, exact legality, committed history and chart seeds, legal continuations, the generation and property-query interfaces, and prefix commits |
| [Gameplay demand and style](gameplay-state.md) | Target continuation responses, the gameplay frontier, demand representations, section-style observations, demand requests, and semantic evaluation |
| [Style conditions and control](style-conditions-and-control.md) | Style identity, the baseline style, chart properties and their measurement semantics, scoped requests with target strength, priority and transition policy, return to natural, and the roles of chart and random seeds |

## Research direction

Target responses take semantic priority over the state used to represent them.
After the initial chart dataset is complete, the response specification will
define, from a mapper's perspective, what gameplay demand should and should not
describe. The connection with style is part of developing that specification:
source-backed style judgments and concrete chart contrasts help identify the
responses worth preserving.

The gameplay frontier names the response function over possible legal futures.
A finite demand state and its dynamics are candidate representations of that
function. Their adequacy must be assessed against the independently defined
target responses. This formulation does not provide a completed response
specification, a calibrated demand scale, or an executable V3 model.

The initial style dataset uses the scoped, ordinal judgments supplied by
@ensomi-labs/beatmap-lens. Presence, ordinal strength, unresolved judgments,
and unreviewed dimensions remain distinct. Those observations provide style
supervision; they do not directly label numerical demand.

Style dimensions are open-ended. Named concepts anchor part of the style
space, and references and learned organization supply the rest. The
[style formulation](style-conditions-and-control.md) states what identity, control, and readouts must
satisfy without fixing a representation of the baseline style or a calibrated
strength scale.

## Authority

The chart language and commit rules are formal constraints. The canonical
gameplay profile and annotation meanings are declared conventions. Request
semantics are interface requirements: scope membership, request validity,
locality outside a scope, priority, release, and the shared definition of
targets and readouts under declared measurement semantics. Response
definitions, representation adequacy, style representations, a generator's
default operating point, the calibration of strength levels, and realized
control behavior have the research status stated in their owning sections.

Concrete candidate generators, state encoders, style encoders, dynamics,
training objectives, pooling models, decoding methods, and experimental
results belong in [research documentation](../research/). A particular
implementation's reachable charts do not redefine the legal chart space.

The [repository README](../../README.md) defines the V3 status and the boundary
around retained pre-V3 systems. Legacy code and local generated assets are not
sources of V3 specification.
