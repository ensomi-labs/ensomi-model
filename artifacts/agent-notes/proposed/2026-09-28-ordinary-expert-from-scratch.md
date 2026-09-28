# Agent Note: Learn an ordinary four-star expert from fresh initialization

Note ID: 2026-09-28-ordinary-expert-from-scratch
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 4c8463a2028cc451a06aeabc302e20673522ed77
Scope: Fresh ordinary-chart recipe, positive corpus, independent failure evaluation and continuation-level fusion
Related: 2026-09-28-continuous-segment-context; 2026-09-28-response-blindspot

## User direction and remaining goal

The user explicitly asks for a new recipe, fresh relevant modules, learning
ordinary approximately4star arrangements with simple LN/rhythmic organization
and strict long-jack control, then trying fusion. This authorizes local data
preparation, implementation, bounded fresh training and evaluation. It does not
approve an exact Card revision, a lifecycle transition or remote publication.
The ultimate goal remains playable2–6star generation, expressive optional styles,
scoped controls and the existing realtime/CompleteRow interface. The ordinary
expert is one component, not a narrower replacement success criterion.

The previous turn made progress: implemented a direct hand-relative context
path, found36/60expert-selected pilot views outside their actual style labels,
ran matched learning to112durable updates and actual64-step generation, and
found weak fixed-prefix LN-control response. The paired run later aborted inside
MPS at update122; it is terminal and preserved. No qualified model exists.

## Main design decisions

Start learned audio, H, R/R1 and control pathways from new initialization;
retain deterministic Mel, exact replay, row syntax, likelihood machinery and
publication interfaces. Hidden layers use normal random initialization; zero
initialization applies only to suitable output/residual projections. Do not
zero every neuron or load actor tensors from a previous checkpoint. Input
normalization comes from the new training split. Full audio is available during
both training and inference; learned audio encodings cannot be reused across
optimizer updates.

Use one interpretable full-audio encoder with local Mel context and coarse
bidirectional audio context, a timing-only H owner, and a complete-row R1 whose
energies own both release survival/marks and H-row choices. R1 owns cardinality,
columns and LN entry/exit; H does not prescribe chord counts. Keep direct audio,
future H, actual hold origins and persistent factual history visible. Independent
player-response evaluation is outside actor NLL. A simple specialist distribution
should be learned before adding expert routing.

Difficulty semantics are fixed: source whole-chart approximately4star labels
remain the reference while learning their quiet and dense passages. The older
recipe mixed whole labels with local strain-derived labels at16/32/64s scopes;
that can teach a different conditional distribution and is a live contributor
to uniformly pressured output, not a proven exclusive cause. Do not relabel
every ordinary quiet crop as a request for a different difficulty. Source
annotations belong only to their actual ranges. Default samples come from the
natural selected population, and visible-condition balancing must not contaminate
missing-condition training.

Ordinary includes clear TAP groups, breath/transition structure, a sustained LN
with other-finger TAPs, and regular LN streams/high coverage where sources support
them. It does not mean no LN, no chords or one fixed pattern template. Offline
selection of a teaching population does not declare other ranked styles bad or
introduce runtime duration bans.

## Evaluation and fusion

Use failure evals in three separate roles: characterize/select the positive
teaching population; retain a disjoint native regression set; later supply
training-side negatives for a separate continuation evaluator and actual
on-policy learning. The old scalar work reference still misses25ms LN and some
sustained repetitions. Its green flag is not a reward oracle. Preserve source
positives and compare short-LN tail coordinates, RH recovery, time-based finger
load/recurrence, rhythm organization, amount, scope controls, music and runtime.
A good mean must not compensate a severe failure. Train-side signals and sealed
regression cases require separate identities.

Teacher forcing supplies one valid source trajectory. It does not label an
arbitrarily changed generated prefix with the old source suffix. The DAgger
analogue motivates visiting policy-induced states, but this project lacks an
expert action oracle there (Ross et al., PMLR15,2011). Actual candidate futures
can instead receive independent response/control judgments, with the correct
joint H/R/R1 likelihood when those paths are optimized. NLL remains a warm-start
objective and diagnostic, not the final quality criterion.

Fusion is over coherent private continuations from the same committed state.
A router selects an expert for a real-time interval; each candidate includes
its compatible H and R/R1 decisions. The frontier evaluates each actual future,
and only a selected prefix publishes. Open LNs, player state and controls persist
across expert changes; private future endpoints are not public facts. The options
analogue is Sutton/Precup/Singh1999. Once this system qualifies, policy distillation
(Rusu et al., arXiv1511.06295) may reduce runtime cost. Keep the ordinary expert's
regressions protected during fusion; output selection is not parameter averaging
and proposal log probabilities are not selected-policy probabilities.

## Experiment Card: ordinary-fresh-bootstrap-v1

Revision: 1
Accepted revision: none

First establish a reproducible positive population from the existing1973ranked
3.5–4.5star charts. Split connected song-group/audio components before selection;
reserve all previous native-case audio. Record exact source clocks, LF strata,
short-LN/RH tails and real-time per-column attack peaks. These are descriptors
and candidate-selection evidence, not adopted physiological thresholds. Follow
with actual Lens/rhythm inspection for selected teaching anchors. Include ordinary
LN organization as well as TAP and quiet passages.

Frozen scouting script/plan live in artifacts/joint-audio/20260928-ordinary-scratch-v1,
plan SHAcdbe5f2d3d9d939f51a6daec238abc6cc8a8389ab567cb0b4be05ee10a12d6d5.
Inputs are the pinned corpus census and full Mel/row manifest. One CPU process,
600s, exclusive scout-v1, no overwrite or automatic restart. Verify source and
cached-row hashes/counts. Fit any selection/reference cutoffs on training groups
only; held-out groups retain their independent meaning. Do not call the candidate
population the completed ordinary subset before those decisions are frozen.

The first actual fresh fit must use a clean source implementing fresh initialization
and a joint H/R/R1 learning score, then a fixed small set of verified ordinary
sources as a capacity/backend probe. Freeze its data, architecture, initialization,
controls, optimizer, native seeds and resources in a revised Card before execution.
Start with a short CPU/MPS stability comparison on the actual work units after
the prior MPS assertion. Do not inherit cached learned audio or a silent pretrained
H. A source-H diagnosis can isolate materialization but does not replace native
end-to-end audio evaluation. Scaling and fusion require real generated evidence;
no checkpoint is qualified by the bootstrap fit alone.
