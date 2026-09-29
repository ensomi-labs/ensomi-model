# Why an onset detector was not the whole problem

The first experiment asked how changing release opportunities affects frozen
R1 while retaining head times and an observed seed. It was a useful dependency
probe, not a candidate audio-only system. Product `ff6471c` implements the probe;
`135e181`–`068988e` add the audio pilot, slot handling, calibration and native-ms
materialization. The consolidated evidence is in
[audio-conditioned choreography](/Users/l/projects/ensomi-model/docs/research/audio_conditioned_choreography.md).

## Observations and their consequences

| Experiment | Reported evidence | What it changed |
| --- | --- | --- |
| Added non-head opportunities, 12 songs × 2 seeds per intervention | Median paired suffix-LN-duration ratio .686 with additions in every fourth eligible gap; .375 with additions in every gap. LN-head share rose by median 25.6 and 68.6 percentage points. | The schedule changes R1's action distribution. Extra candidate releases are not neutral padding. Timing features and RNG consumption also change, so this is not an isolated release-hazard estimate. |
| 48-TRAIN-song frame pilot; six calibration and six assessment songs | Frozen BeatThis features improved head F1 at 20 ms from .677 to .732; release-only F1 remained .035 to .051. | Audio transfer was plausible, but release-only prediction was weak. This used supplied coarse densities, one chart per audio and a noncanonical frontend; it did not establish calibrated difficulty or coherent alternatives. |
| Alternative arrangements on identical audio, 416 charts/584 pairs | Head F1 median .785, release-only .092; 78.7% of release rows also contained heads. | A reference onset list is not the unique correct arrangement. Releases depend on the chosen arrangement and cannot be represented exhaustively by an independent release-only detector. |

The BeatThis experiment used frozen features, not a demonstrated successful
BeatThis fine-tune. Early fine-offset F1 scores precede the native-ms correction;
they should not be silently combined with later timing materialization results.
Same-audio alternatives were preserved as separate targets, never union labels.

The human redirected the work toward a simple canonical Mel encoder developed
jointly with event timing and complete rows, and deferred new long-range memory.
This rejected treating encoder sufficiency as a prerequisite. The available
supervision was audio plus charts, not a MERT-like pretraining corpus. Source:
[private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0038).

## First joint model and the first false success

Products `09b919c`, `608f092` and `78222bc` implement a 2.95M-parameter model
with the confirmed 24-kHz/128-bin/10-ms-hop Mel frontend, native integer-ms
hazards, shared audio conditioning and complete legal rows. The 10-ms feature
clock does not quantize generated event times. Exact LN state and survival
across empty queries remain causal; BOS requires no source-chart seed.

The 48-song corpus contained 121 separate TRAIN arrangements. Memorizing 32
queries drove TRAIN NLL from 9.619 to .00124 while validation worsened from
10.494 to 45.349. Broader random queries reached 6.071 on a 48-query development
panel; complete-wait supervision reached 6.091 while improving two full-gap
probes from 17.127 to 14.628. These establish fitting and a narrow coverage
repair, not general playability.

One inspected YOASOBI output had 896 events, 967 heads and 301 LNs, enough
structural promise to package for playtesting. No recovered human playtest
verdict follows from that delivery. Other outputs retained duplicate-like
rapid TAPs. The optional 27-ms head-age prior reduced <=10-ms TAP pairs from
42 to zero and <=20-ms pairs from 157 to ten in an 18-output probe, with
median head ratio .9866. It remained a disabled-by-default research prior;
27 ms was a corpus observation, not physiology.

The next paired 600-update comparison illustrates why local repair cannot be
the success criterion:

| Readout, 18 outputs | Start | Source-only continuation | Native-window correction |
| --- | ---: | ---: | ---: |
| <=10-ms same-lane TAP pairs | 42 | 11 | 2 |
| Median head-count ratio | 1.000 | .642 | .688 |
| Total LN heads | 1,839 | 3,202 | 4,091 |
| Held-out native-window KL | .0530 | .2012 | .1029 |

Both continuations failed preservation/transfer guards. One SCREW output had
only ten heads early in a 121-s song. On its fixed prefix, integrated future
hazard 5.0165 was below the sampled threshold 6.3416; independent integration
and two query sizes agreed. The silent tail was a learned continuation risk,
not evidence of a query partition bug. Mature-prefix-only recovery probes had
excluded this early failure by construction.

Products `3a5cea9` and `16209ea` preserve this failed correction rather than
promoting it. The original complete next-event KL includes survival/censor mass
and never attaches an old source suffix to an altered generated prefix.

## What was verified, and where to return

Historical mechanical evidence includes exact replay/export/reparse, crop/full
audio parity, query-partition equality and source-free generation. On one
4,630-row sample, 500-ms versus 4,000-ms queries produced identical rows while
generation time fell from 38.22 to 9.39 seconds. That is a concrete optimization
on one workload, not a deadline guarantee. Source code lives in
[joint_audio_continuation](/Users/l/projects/ensomi-model/src/ensomi_model/research/joint_audio_continuation/)
with nearby `tests/research/joint_audio_continuation/` tests. These historical
experiments were not rerun during recovery.

Evidence owners named by the documents include `artifacts/audio-skeleton/20260923-v1`
and `artifacts/joint-audio/20260923-v1` in the product checkout. Large local
artifacts remain external to these notes. Product `10a16fe` made the expert
question and key counterexamples independently readable. The resulting
[full-audio and transfer investigation](full-audio-and-r1-transfer.md) matters
because it separates valid probability factorization, missing information and
the policy actually inherited from R1.
