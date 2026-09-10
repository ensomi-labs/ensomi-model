# Gold-only fixed-placement diffusion

This experiment generates alternative 4K arrangements at known action timestamps,
conditioned on a scoped human style label, committed source history and music mel.
It is an experimental implementation, not the V3 reference architecture.

## Run

From the repository root on Apple Silicon:

```sh
uv run --extra mps python -m pulsefield_model.experiments.gold_diffusion.cli
```

The packaged `gold_diffusion.yaml` owns runtime settings. The defaults use MPS,
3,000 optimizer steps, batch size four, validation every 100 steps, and a
two-hour optimization limit. Eight consecutive validations without improvement
stop training. Preparation and final evaluation are outside that time limit.
An explicit output directory must not already exist:

```sh
uv run --extra mps python -m pulsefield_model.experiments.gold_diffusion.cli \
  output=artifacts/gold_diffusion/example steps=1000
```

A short integration run uses the same data and model, without examining the test
split:

```sh
uv run --extra mps python -m pulsefield_model.experiments.gold_diffusion.cli \
  steps=2 eval_every=2 log_every=1 sample_scopes=1 sample_seeds=1 \
  sampling_steps=4 final_test=false
```

## Data and supervision

`lens_root` defaults to `../beatmap-lens`. Source facts come from
`.local/corpus-500`; human decisions come from the canonical documents in
`.local/corpus-500-v2/workspace/workflow`. For an agent claim, only its latest
accepted or modified human decision supplies gold. Direct human observations
also qualify. Machine reviews and unresolved decisions never supply labels.
The exporter rejects conflicting judgments on the same exact source/scope/tag.
Unknown dimensions remain unknown; no labels are inferred from prose.

Targets contain the entire gold scope, using Lens half-open `[startMs,endMs)`
intervals. A scope with zero rows or more than `max_rows` is excluded and recorded,
not cropped or relabeled. All earlier source actions supply history; losses are
computed only on gold targets. There is no unannotated pretraining. The label is
one of absent/supporting/prominent for Jack, Stream, Trill, Tech or LN coordination.

Splits group matching normalized artist/title metadata and identical audio bytes.
This prevents known same-song groups from crossing splits; alternate song names
with differently encoded audio may still require curated grouping. At least one
training group remains for every supervised tag/assessment cell. Validation and
test coverage need not include every cell. The manifest records exact identities,
scopes, splits, exclusions, source/audio/mel hashes and workflow provenance.

## Model and sampling

Each fixed placement is one complete nonempty four-lane action row. The 255-row
alphabet includes explicit LN starts and closes. `MASK` is unknown content, not
an empty row. The experiment does not choose placement times or generate
`NO_ROW`. Source-derived LN close times are part of the supplied skeleton.

The small denoiser reads static row-content embeddings of the full preceding
history. History does not self-attend and has no positional embeddings. External
real-time differences control history reads and candidate interactions. A
time-weighted history-mass feature retains information that normalized averages
of identical repeated rows would erase. These reads are an experimental history
interface, not a calibrated demand representation.

Music uses `MUSIC_MEL_CACHE_CONFIG`: 24 kHz, 128 bins, 10 ms hop, 40 ms window.
Missing caches are generated only for selected gold sources. The adapter reads
interpolated mel patches extending 200 ms on either side of each placement;
frame midpoints are 20 ms after their start because extraction uses
`center=False`. Chart timestamps are never rounded to the mel grid. The local
audio adapter does not claim to represent whole-song musical structure.

Training independently masks rows at a uniformly sampled noise level and uses
inverse-noise-weighted masked cross-entropy, averaged per scope. Condition dropout
trains the unspecified-style comparison. Sampling starts fully masked and
progressively reveals rows. A 16-state occupancy dynamic program samples legal
proposals from the current unary logits while preserving already revealed rows.
This constraint modifies the reverse sampler distribution.

For inspectable whole-chart exports, the scope's original entry and exit LN
occupancies are fixed constraints. Exit occupancy is used only by the sampler,
not exposed to the denoiser or derived from masked teacher rows during training.
The source outside the gold scope is retained; crossing holds keep their legal
connections. Exported `.osu` files preserve source timing points and link to the
local audio file. Hit sounds and non-gameplay hit-object fields are normalized.

## Evaluation and artifacts

Validation selects the best checkpoint by mean per-scope full-mask cross-entropy.
Metrics include row/lane reconstruction accuracy, unconstrained argmax boundary
legality, and the NLL difference between the observed style request and an
unspecified request. These are likelihood and reconstruction diagnostics, not
independent measurements of semantic style adherence.

After optimization, the best checkpoint automatically evaluates the held-out
test split. Both splits receive samples for each positively supervised tag, at
its strongest training-supported salience, plus the reference cell when available.
Matching seeds across conditions support paired review. Generated sample records
leave `human_style_judgment` unset until a human reviews them.

Every run contains `resolved.yaml`, `runner.json`, `manifest.json`, a serialized
data snapshot, exact experiment-source copies, `metrics.jsonl`, `status.json`,
`best.pt`, `last.pt`, and final `evaluation.json`. Validation/test sample manifests
link to generated charts under `samples/`. Source snapshots make local uncommitted
experiment code recoverable; they do not imply a clean published revision.

Run the relevant checks with:

```sh
uv run --extra mps --group dev pytest -q tests/experiments/test_gold_diffusion.py
```
