# Ordinary arrangements and an operational player-response frontier

This work lives on `codex/release-calibration`, now at
`96f84fd32218e39ff809c651d73e6edbd39a3500`, in
`/Users/l/.codex/worktrees/release-calibration/ensomi-model`. It was kept separate
while the main [three-arm fit](clean-joint-and-current-state.md) ran. Its code is
not merged into the primary `codex/audio-skeleton` checkout. Existing Agent Notes
at `4afc915c5bb15e4ce371a534b012c3332cbae939` supply the experimental chronology.

Human feedback called for ordinary approximately-four-star TAP organization,
trackable LN roles and difficulty-sensitive main rhythmic subdivision while
retaining Tech and other expressive exceptions. A short-LN-dominated specialist
style should not silently become the default. The user questioned H/R decision
timing and later authorized a fresh ordinary expert with possible fusion.
Sources: [ordinary references, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0983),
[H/R concern](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0992),
[subdivision clarification](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0999),
and [default-style correction](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1022).

## Calibration and release decision timing

The [coordination study](/Users/l/projects/ensomi-model/docs/research/coordination_frontier_hypotheses_zh.md)
at `0a74a19` compared 6,924 ranked charts and 125,590 eight-second windows,
then read 68 Lens pages in matched/contrasting contexts. Similar star, LN amount,
occupation and attack measures could conceal very different release groupings
and recurring finger roles. An exact constructed example leaves both whole and
scoped star measures unchanged while altering release organization. This is a
blind direction of those metrics, not proof that every irregular release is bad.

The [release calibration](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/ln_release_calibration.md)
and `row_likelihood_parts` (`dc50ce2`) separated head signature, release
cardinality and release identity. Across 288 saved laws/23,616 queries the
reusable decomposition matched prior factors to at most 3.25e-7 nats.
Removing learned consequences improved some individual choices but worsened
some integrated cardinality likelihoods; it was not uniformly harmful.
This prevented attributing every short tail to one inherited module.

`965d670` then added joint R1 wait/release scoring. An independent R event
previously committed to release before R1 could prefer waiting; with one legal
release mark, a finite row penalty could cancel. The joint law supplies a
no-row survival action before selecting time and preserves full subset choice
and forced terminal/deadline atoms. [Joint release account](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/joint_r1_release_decisions.md).

A 16-update pilot reduced <=80-ms-LN/all-head burden from .5525 to .0689 on
the known Stream witness, but its difficulty remained 5.79 for D4 and it became
longer overlapping LN organization. Blizzard amount and Stream startup failed.
At 80 updates all three complete cases still failed overall checks. Most
Stream short tails had occurred at H with a supported keep-holding row; STYX
had a larger pure-R contribution. Different contexts required both paths.

## Rhythm hierarchy remained a proposal

The [audio rhythm hierarchy design](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/audio_rhythm_hierarchy_zh.md)
separates an internal musical clock, a main subdivision and optional finer
events, all learned within audio-to-chart training. It does not require oracle
BPM/redlines or a fixed quantized output grid. It was **not implemented or
trained**. The implemented lattice observer found weak shared timing in fixed
generated scopes (roughly 31–46% at strict tolerance versus 92–100% source),
but low lattice coverage is not a universal Tech/tempo-change failure gate.
More tolerant inspection later found approximate source periods plus jitter/
doublets in the fresh expert, refining the question beyond absence of rhythm.

## The actual planner had weaker semantics than its name suggested

The operational audit found that qualification and the benchmark demo called
`ControlledSession` directly; the optional `ResponsePlanner` was not an
automatic part of those generation paths. `frontier2` was an actor energy
trained by row NLL, not an independently calibrated future-response objective.
The optional planner also formerly published the least-excess candidate even
when all failed. [Action-response account](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/action_response_frontier.md).

`02e522d` introduced request-independent action-response state with four
exponential scales and HH/HR/RH, releases, hand turnover and partner-held
coordinates. The planner now retains the published prefix and raises
`NoAcceptableContinuation` when none pass. The response is a corpus-calibrated
engineering hypothesis, not a measured physiology model.

An actual three-song gate probe completed STYX, stopped Blizzard at 100 s and
Stream at 224 s, preserving open holds. Response-guided proposals (`b3f0184`)
completed Blizzard but still missed amount; Stream stopped at 146 s. Source-H
substitution still failed, with Stream stopping at 58 s. These stopping clocks
are different reached trajectories, not a same-state ranking. Observed startup/
publication deficits also remained. Zenithfall's source is 5.87349★, so its
source H cannot be called a proven D4-feasible reference.

`822f34f` fixed a further execution gap: acceptance of isolated four-second
forecasts did not enforce longer sliding windows across publications. Rolling
work accounting now retains prior committed charges. The focused test checks
that an eight-second breach rejects without changing published state. This
fixes accounting, not the inadequacy of the response itself.

An actual planner execution published a 25-ms LN with zero work. Isolated
23/25/40/80/150/300-ms releases likewise had zero under the tested reference;
32 same-finger TAPs 125 ms apart were sensitive relative to rotation but still
accepted. A quadratic work alternative did not repair discrimination and was
not adopted. The [short-hold diagnosis](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/r1_short_hold_acceptance_zh.md)
adds a preserved red regression: all six segment outputs fail the 40-ms burden
coordinate while passing 80 ms; three source positives pass both. STYX's broad
80-ms burden improved even as its extreme 40-ms tail worsened. One threshold
had been hiding another failure.

## Conditional coverage, latent plans and the incomplete context fit

The clean ledger contained 221 effective prominent-Stream draws but zero
style-known/LN-unknown Stream pieces. `1658123` supplies a factual paired
visibility view. Yet hiding LN changed expected LN fraction by median only
.26 percentage points on matched source states; changing style extent had
median .051-point effect. The coverage defect was real but not a sufficient
account of native LN collapse. [Optional controls](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/optional_control_training.md).

A persistent scalar LN-composition prior reduced but did not eliminate the
collapse; even a sampled zero-LN condition left 27/49 LN heads in the fixed
source-H scope. This motivated a genuine action-segment distribution, not
simply more scalar guidance. `33645c9` first made joint R/R1 partial-future
likelihood include survival/event factors, necessary before optimizing those
new parameter-dependent release paths.

`7d31b1e` implemented exact one-/four-state action-segment mixtures, persistent
birth context and a nonlinear row decoder. Thirty-two updates on 158 factual
segments yielded six complete source-H outputs with short holds and poor
amounts; only one training segment supervised the actual first source row.
All 85 four-state prior selections chose the same code. Longer history/future
audio reaching only that code did not prove effective plan conditioning.
[Segment model](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/action_segment_r1.md).

`e4c4453` added continuous relative-hand plan context; `4c8463a` repaired
human-branch supervision after 36/60 selected pilot views fell outside actual
annotation extents. At 64 updates both code-only and continuous arms' three
source-H outputs still failed every 40-ms gate, despite slightly better factual
NLL for continuous context. Only four of twenty rendered generated pages were
actually read. The planned 256-update fit aborted in MPS during the 112–128
worker, last logged update 121 and last durable checkpoint 112. Its channel
mismatch assertion was not diagnosed as memory exhaustion and was not restarted.

The subsequent [fresh ordinary expert and current state](clean-joint-and-current-state.md)
must be read as a new small capacity experiment. It is neither completion of
that interrupted comparison nor evidence that the full playable system was
finally solved.
