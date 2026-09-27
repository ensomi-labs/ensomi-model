# Audio and committed-history memory prototype

The [four-star phrasing study](four_star_phrasing_and_audio_memory.md) identifies
loss of rhythmic and TAP/LN texture contrasts that total density and star ratings
miss. This prototype adds a nonlinear multiscale audio path and separate
context-queried H/R/R1 histories. It implements a research hypothesis, not a
qualified replacement for the selected model.

The [matched 384-update joint fit](audio_memory_joint_fit.md) completes but fails
native pressure, difficulty and restored-range guards. Its measured runtime
passes; the failed quality comparison identifies timing-density growth and
control-support gaps that require separate diagnosis.

## Audio and history paths

[`AudioPyramid`](../../src/ensomi_model/research/audio_memory_continuation/audio.py)
reads learned 100 Hz fine features. Each 500-ms cell retains their mean and
maximum; pooled .5/2/8-second views feed three bidirectional attention layers
of width 256 with four heads. A projection adds to the existing 128 global
audio coordinates. The original fine and global encoders remain trainable.
Partial cells use only real frames, and padding cannot become attention keys.
Complete audio is required at training and inference; crop-only scoring fails.

[`HistoryAttention`](../../src/ensomi_model/research/audio_memory_continuation/memory.py)
reads the latest known event state in each absolute 500-ms cell within the
preceding 64 seconds, at most 129 cells including partial boundary cells. Each
item binds its post-event TCN state to audio at that event. This is a lossy
history sampling scheme; the existing local TCN retains immediate order. Empty
cells add no fake events, while relative elapsed time and current clocks retain
the distinction between a recent pattern and a long silence.

H, R and R1 have distinct four-head, width-128 readers. H contains only H history
and aligned audio. R contains H/R timing roles and uses its permitted LN clocks
at the current query. R1 contains complete committed rows and reads current
audio, exact state, timing-only H preview and controls. Hand-specific queries
share weights in mirrored coordinates. Their outputs modify the corresponding
local histories, allowing relative layout choices to depend on retrieved content.
Counts, columns, TAP/LN and release subsets remain R1-owned.

The readers use PyTorch scaled dot-product attention with a learned elapsed-time
bias, an explicit invalid-memory mask and a zero null key/value. Dropout is zero.
This avoids explicitly expanding the query/key and weighted-value products over
every query, cell and channel. CPU and MPS tests compare nonzero outputs and all
input/parameter gradients with the original expanded contraction; backend memory
and throughput still require workload measurements.
During gradient-enabled scoring, more than 128 queries use groups of 128 with
activation checkpointing. The shared projected key/value tables remain available;
each group's expanded gathers are recomputed for backward. Every query and all
audio/history gradients are retained. Inference without gradients uses the direct
path. This bounds saved query-specific activations without shortening history.

Output projections start at zero, reproducing the baseline probability law.
The full-audio encoding can be cached for generation. Learned history/audio
values cannot be reused across training parameter updates. The first teacher
implementation recomputes full prefix states under current weights; it does
not trade correctness for a stale cache.

## Causality and native execution

Memory selection requires both a query time and the last-known event index.
A hazard-bin query at 109 ms might score a target at 103 ms; selecting every
event before 109 would expose that target. The index cap excludes it even
inside the same 10-ms bin or 500-ms memory cell. Full future audio is permitted;
future chart materialization is not.

[`MemorySession`](../../src/ensomi_model/research/audio_memory_continuation/generation.py)
writes H memory when the H planner chooses a timestamp, and row/release memory
after an actual complete row is committed. Forks share immutable prior records;
each new write belongs to its branch. A control-suffix rollback truncates H
memory to the restored planner prefix. It neither rewrites published rows nor
resets exact gameplay or historical row memory. Open LN origins remain in exact
state even when their head leaves the attention window.

[`collate_interval` and `score_interval`](../../src/ensomi_model/research/audio_memory_continuation/intervals.py)
reuse the native H/R/row laws, complete waiting-time likelihood and conditioned
release waits. Query-specific memory uses the same current-weight histories
and full audio. Interval ends remain distinct from true audio termination.

## Local use and verification

Initialize from a loaded controlled model:

```python
from ensomi_model.research.audio_memory_continuation.model import initialize
from ensomi_model.research.audio_memory_continuation.generation import MemorySession

model = initialize(controlled_model).eval()
session = MemorySession(model, mel, duration_ms, controls, seed=273200)
session.publish_to(min(8000, duration_ms))
```

Save `model.checkpoint()` with `torch.save`; the matching `load_model` in the
memory package reads `controlled-audio-memory/v1`. This research family does not
silently replace the existing packaged runtime or its default checkpoint.

Seventeen memory-specific test cases cover index-capped causality, empty memory, hand equivariance,
query sensitivity, exact zero-initialization compatibility, joint CPU/MPS
gradients, checkpoint roundtrip, teacher/native row and H/R query agreement,
skeleton ownership, fork/control rollback, and padding-independent audio and memory
values/gradients, and direct/checkpointed attention equivalence to the expanded contraction.
The original integration also checked affected distribution, ownership and sampling
owners, with 40 distinct checks before the four contraction-equivalence cases.
These are implementation checks.

## Bounded Mac integration

The full prototype has 7,616,517 parameters versus core2500's 4,583,985. On one
118.334-second Max Burning audio, both zero-initialized models produce identical
complete rows at seed 273200. They request the source's 4.0005 stars and .0429 LN
fraction, with styles unknown. One CPU thread, loaded weights and cached Mel give:

| Measurement | Core2500 | Memory prototype |
| --- | ---: | ---: |
| Complete-audio encoding, s | .2090 | .2356 |
| First thirty rows, s | .5379 | .9583 |
| Complete generation, s | 6.5025 | 13.4471 |
| Slowest two-second publication service, s | .1677 | .4098 |

These timings exclude waveform decoding, Mel construction and model loading.
They demonstrate headroom on one chart, not a general dense-passage guarantee.
The [publication evaluator](gameplay_regression_evaluation.md#publication-deadlines)
finds no missed deadlines in either actual trace when playback begins at its
first-thirty-row time and requires two seconds of settled future coverage.

Eight MPS updates score 32-second source intervals with differentiable full-song
audio. Every audio/H/R/R1 and added-memory parameter group changes, all gradients
remain finite, and fitting takes 53.29 seconds. These smoke weights are not a
musical-quality candidate and do not initialize the main fit.

Sampled footprint reaches 15.19 GiB and MPS driver memory 12.63 GiB, while active
MPS storage after updates is about .37 GiB. Variable-shape retention therefore
needs attention before a long fit. The teacher scorer now keeps its existing
128-event padding through memory key/value projection rather than slicing every
table back to a distinct real event count. Only real indices are selectable;
CPU/MPS values and gradients retain parity.

A fresh 24-update probe repeats the eight intervals twice, then visits four BOS
and four interval-index-three scopes. It completes in 110.00 seconds. Peak
footprint is 15.00 GiB in the first cycle, 15.03 GiB in the repeated cycle, and
15.98 GiB with the new scopes. Driver peaks are 12.57/12.58/12.86 GiB respectively.
The repeated shape set plateaus; arbitrary corpus shapes remain untested. New
forward-phase sampling finds active-storage peaks of 4.51 GiB, a phase omitted
from the earlier audio/update-only sampling.

After gradients are released, a one-time diagnostic cache release lowers driver
memory from 12.86 to 1.45 GiB, while process footprint only falls from 15.98 to
14.66 GiB. Those overlapping ledgers must not be added or interpreted as the same
allocation. The bucket change does not materially lower the cold peak, and the
probe does not establish a complete memory root cause. It supports bounded fitting
with the declared 18-GiB guard on the tested shapes. No quality improvement follows
from the repair or these discarded integration updates.

Integration source: `720457b8651d40095c2247b5992c3318b2cb5ced`.
Owner: `20260927-audio-history-memory-v1`, successful `preflight-v3`.
Integration checkpoint SHA-256:
`bd5cee5e5d532272e12784a78a8fb152d9f1842874e5b48b611832e36de58fb2`.
Two earlier driver attempts ended on missing callback/duplicate report-key
errors; their partial outputs are retained and are not completed comparisons.
Stability source: `251ba0f47970b18f739a3441893e1e0ab9a858cd`, `stability-v1`.
