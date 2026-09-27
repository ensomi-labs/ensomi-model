# Audio and committed-history memory prototype

The [four-star phrasing study](four_star_phrasing_and_audio_memory.md) identifies
loss of rhythmic and TAP/LN texture contrasts that total density and star ratings
miss. This prototype adds a nonlinear multiscale audio path and separate
context-queried H/R/R1 histories. It implements a research hypothesis, not a
qualified replacement for the selected model.

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

Nine new test cases cover index-capped causality, empty memory, hand equivariance,
query sensitivity, exact zero-initialization compatibility, joint CPU/MPS
gradients, checkpoint roundtrip, teacher/native row and H/R query agreement,
skeleton ownership, and fork/control rollback. Together with affected distribution,
ownership and sampling owners, 36 tests pass. These are implementation checks.
Native startup/dense-service profiling and a trained musical-quality comparison
remain required before drawing a model-quality conclusion.
