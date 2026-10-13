# ONNX runtime export and replay

`ensomi_model.export` exports the released R2 system's networks as float32
ONNX, with opset 17 by default. Audio processing, grid fitting, feature
construction, both decode loops, controls, the corpus prior and chart export
remain Python. The export supports the released FiLM R2 checkpoint. R2's
optional arrangement copy model is outside this export; the release uses copy
off. The rows model's own copy heads are included.

Install the model dependencies and the export extra on Apple Silicon:

```sh
uv sync --python 3.10 --extra mps --extra export --group dev
PYTHONPATH=src .venv/bin/python -m ensomi_model.export.onnx \
  --model-dir artifacts/release-r2/bulk/hf/ensomi-r2 \
  --out artifacts/release-r2/bulk/hf/ensomi-r2/onnx
```

Use the `cuda` extra instead of `mps` on Linux with NVIDIA. Export and parity
run on CPU with one thread. The exporter runs the ONNX checker on every graph;
the parity command below exercises ONNX Runtime's CPU provider.

## Graph calls and state

| Graph | Input | Output |
| --- | --- | --- |
| `beatthis` | One log-Mel chunk `[1, time, 128]` | Beat and downbeat logits `[1, time]` |
| `rows_encoder` | Whole-song Mel, position and beat frame indices, extra features and validity | Position vectors, beat context, copy queries and keys |
| `rows_step` | One beat's encoded inputs, previous figure ID, GRU hidden state, copy history and temperature | New hidden state, raw figure and copy scores, gate, generator log probabilities and copy attention |
| `r2_predict` | Last temporal vector, exact row query, row condition, and landmark state when enabled | Hand vectors and 625 raw joint-action logits |
| `r2_release` | Hand vectors and one oriented release factor's features and candidates | One score per candidate release time |
| `r2_commit` | The completed row's history token, row count and temporal buffers | Updated temporal vector, count and buffers; updated landmarks when enabled |

Run `rows_encoder` once per song. For each beat, call `rows_step`, apply the
Python copy mixture, available-position mask and section tilt, then sample a
figure. Carry `hidden_out` into the next call. Keep the most recent figure IDs
and copy keys in newest-first order. Initial hidden state is zero and the
initial previous-figure ID is the vocabulary size. A figure is a **24-bit mask
over `POSITIONS`**; those positions are measured in forty-eighths of a beat.

For each R2 decision, call `r2_predict`, apply the legal support and decode
controls, sample the action, and draw the orientation. Call `r2_release` for
each gap-release lane in that orientation's order, updating the placed-lane
features after each selection. Commit the action and releases to the chart,
build its history token, and call `r2_commit`. Carry every returned state
tensor into the following decision. EOS closes the remaining holds and has no
history commit. Prediction and commit are separate because the history token
depends on the sampled action and release times. No graph re-encodes history.

## `runtime.json`

Each graph entry lists its file, byte size, opset, input and output names,
dtypes, shapes and dynamic axes. Batch size is one. Frame counts, beat counts
and release candidate counts are dynamic; landmark capacity is dynamic for
models that use landmarks. The released checkpoint has `memory="none"` and
carries no landmark tensors.

The manifest also contains initial R2 state files, network configuration,
figure vocabulary and membership, the star map, action table, prior,
calibration, frontend parameters, and decode constants. The scope table in
`calibration.scope_table` is exported as named raw columns with row counts,
dtypes and shapes. Its original row order is preserved for stable neighbour
ties. Exact Mel filters and Hann windows are separate raw arrays; their axes
are declared because the two frontends store their Mel filters in opposite
orders. BeatThis's MIT notice is copied beside its graph.

All referenced binary files are contiguous, row-major and little-endian.
Paths are relative to the manifest directory. The prior and calibration
remain data interpreted by the decode code, not learned network graphs.

## `ensomi-package.json`

The exporter also writes the package manifest `ensomi-package.json` beside
`runtime.json`. The Ensomi runtime owns its format, `ensomi-package/1`, and
reads it from runtime 0.3.0. The package manifest names the model, its licence
and notices, the decode pipeline, the oldest runtime that runs the package and
the runtime config. It lists every file the package needs except itself, with
byte size and SHA-256. The exporter lists the files it wrote. For a package
exported earlier, this command derives the files from `runtime.json` by the
runtime's rule and writes the package manifest; it needs neither torch nor
onnx:

```sh
PYTHONPATH=src .venv/bin/python -m ensomi_model.export.package \
  artifacts/release-r2/bulk/hf/ensomi-r2/onnx
```

## Reference recordings

The reference command runs `System.make` on the four release examples. It
reads each example's star, seed, title and artist from its receipt, uses the
default controls with copy off, and checks every final hitobject against the
published `.osu`. It needs the four local audio files listed in
`export/reference.py`; the model distribution does not include those songs.

```sh
PYTHONPATH=src .venv/bin/python -m ensomi_model.export.reference \
  --model-dir artifacts/release-r2/bulk/hf/ensomi-r2 \
  --out artifacts/runtime-ref
PYTHONPATH=src .venv/bin/python -m ensomi_model.export.parity \
  --model-dir artifacts/release-r2/bulk/hf/ensomi-r2 \
  --reference artifacts/runtime-ref
```

Both commands accept `--songs` followed by example names for a smaller run.
Recordings and charts live in `bulk/<song>/`; `<song>/summary.json` and
`parity.json` hold the small results. The index names every array with its
dtype, shape, file, byte offset and byte length. Graph records and decode
events refer to these entries by index. Unchanged tensors may share a byte
range. Scalar shapes are `[]`; missing release times are IEEE NaN and masked
log probabilities can be negative infinity. No pickle is used.
Array ranges can be unaligned; JavaScript readers can slice the addressed
bytes into a new `ArrayBuffer` before constructing a typed view.

Recordings include both PCM paths and sample rates, spectrograms, activations,
Mel features, grids, every graph input and output, decode rule outputs,
random variates, prior targets and final chart objects. Each difficulty pass
has its own `r2_pass_N` scope. Every pass restarts the R2 generator at the
configured seed, as the normal system does.

Rows replay uses the recorded exponential race and selects
`argmax(probability / exponential_draw)`. Recording still calls the original
`torch.multinomial`; a cloned generator exposes its variates without consuming
the live generator. R2 replay uses the recorded float64 uniforms for Gumbel
selection and the recorded orientation draws. Chosen indices are comparison
targets only. A runtime's own seeded generator need not reproduce PyTorch's
seed stream.

The parity report separates two checks: numerical replay on every recorded
graph input, with absolute tolerance `1e-4`, and a fresh Python decode through
ORT with carried network state and the recorded draws. The latter recomputes
audio features and the grid and requires identical hitobjects, every draw
consumed, and no sampled-choice differences, including discarded difficulty
passes. Any choice difference reports the original and replayed competing
scores and margins. CPU timings are measured per graph and per song. These
checks establish runtime equivalence on the examples; they do not evaluate
chart quality or establish browser-provider parity.
