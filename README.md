# Ensomi Model

Research code for Ensomi V3, a system that generates 4-key rhythm-game charts
(osu!mania 4K) from music. This branch holds the problem definition and the two
released models: the end-to-end system [Ensomi R2](#released-model-ensomi-r2)
and, below it, the earlier continuation model [R1](#earlier-released-model-r1).
Earlier and unfinished lines of work live on other refs, listed under
[History](#history).

## Released model: Ensomi R2

Ensomi R2 turns a song into an osu!mania 4K chart. Give it an audio file and a
star rating, and it writes a complete `.osz` with the chart, the audio and a
receipt of every setting used. A timing system fits a beat grid to the song, a
rows model decides where notes start, and R2 arranges the lanes, taps and long
notes, holding a chart-level character taken from similar human charts. It
generates a whole song at a time, in 8 to 12 s for a 2 to 3.3 minute song on
one CPU thread of an Apple M5. The [report](docs/research/r2_system.md) tells
what the project is aiming for, the decisions that shaped R2 and what it can
and cannot do yet.

![SaiGetsu: Ensomi R2 generated chart](https://huggingface.co/sed-i/ensomi-r2/resolve/main/previews/saigetsu.png)

| Adventure | Catch the Moment |
| --- | --- |
| ![Adventure: Ensomi R2 generated chart](https://huggingface.co/sed-i/ensomi-r2/resolve/main/previews/adventure.png) | ![Catch the Moment: Ensomi R2 generated chart](https://huggingface.co/sed-i/ensomi-r2/resolve/main/previews/catch-the-moment.png) |

Yellow notes are taps and cyan notes are long notes. Each panel reads bottom to
top, then continues in the panel to its right. These are seed-0 outputs of the
released system; they illustrate behavior and are not a quality benchmark. The
[gallery](https://huggingface.co/sed-i/ensomi-r2/blob/main/PREVIEWS.md) has four
songs with their generated `.osu` files.

| | |
| --- | --- |
| Weights, prior, calibration | [`sed-i/ensomi-r2`](https://huggingface.co/sed-i/ensomi-r2) on Hugging Face, licensed CC BY-NC-SA 4.0 (no commercial use) |
| Release tags | `rows-1.0`, `timing-1.0`, `r2-1.0`, all on one commit |
| Report | [`docs/research/r2_system.md`](docs/research/r2_system.md) |
| Inputs | complete audio, a star request in [2, 6), optional controls |
| Outputs | `.osz` with whole-millisecond times, and `receipt.json` |

What has been checked, and what has not:

- Every chart is legal and exported exactly. Playability and musical quality
  rest on one listener's blind screens, one seed per side: the new rows were
  preferred over the earlier generator 10 to 3, and the human chart was chosen
  over R2's in all three reference pairs.
- Star rating and jumpstream and chordstream requests work in those screens.
  The long-note amount control does not work yet; section difficulty and
  chordjack are not offered as working, and copy is off.
- It generates a whole song at a time; it does not play along in real time.

The [report](docs/research/r2_system.md#what-has-been-checked) has the evidence
and its limits.

### Generate a chart

Install [uv](https://docs.astral.sh/uv/) and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Use `--extra cuda` instead of `--extra mps` on Linux with NVIDIA; only macOS
arm64 with the CPU has been checked.

```sh
git clone https://github.com/ensomi-labs/ensomi-model.git
cd ensomi-model
git checkout --detach r2-1.0
uv sync --python 3.10 --extra mps
hf download sed-i/ensomi-r2 --local-dir artifacts/hf-r2

uv run --python 3.10 --extra mps python -m ensomi_model.system.make \
  --model-dir artifacts/hf-r2 --audio /path/to/song.mp3 --star 3.5 \
  --title "Song title" --artist "Artist" --seed 0 --out artifacts/my-chart.osz
```

`--model-dir` takes `artifacts/hf-r2/system.json` as the config and resolves its
paths against that directory, so the command runs from any working directory.
BeatThis `final0` is downloaded on the first run from its own source. The
`--config` form with paths relative to the working directory remains for
development. `python -m ensomi_model.system.make --help` lists the controls.

## Earlier released model: R1

R1 continues a chart. Given a short playable seed and the times at which events
occur, it chooses the lanes, taps, long-note starts and releases for the rest of
the song. It has 3,084,432 parameters.

![Alone: R1 generated continuation](https://huggingface.co/sed-i/pulsefield-r1-restored/resolve/main/previews/alone.png)

| Descent | Slash Dot Slash |
| --- | --- |
| ![Descent: R1 generated continuation](https://huggingface.co/sed-i/pulsefield-r1-restored/resolve/main/previews/descent.png) | ![Slash Dot Slash: R1 generated continuation](https://huggingface.co/sed-i/pulsefield-r1-restored/resolve/main/previews/slash-dot-slash.png) |

Yellow notes are taps and cyan notes are long notes. Each panel reads bottom to
top, then continues in the panel to its right. These are outputs of the released
checkpoint at sampling seed 17, taken after the supplied seed. They illustrate
behavior; they are not a quality benchmark. The
[gallery](https://huggingface.co/sed-i/pulsefield-r1-restored/blob/main/PREVIEWS.md)
has six examples with their generated `.osu` files.

| | |
| --- | --- |
| Weights | [`sed-i/pulsefield-r1-restored`](https://huggingface.co/sed-i/pulsefield-r1-restored) on Hugging Face, published under the project's earlier name |
| Checkpoint SHA-256 | `4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70` |
| Release tag | [`r1-restored-6.75m`](https://github.com/ensomi-labs/ensomi-model/tree/r1-restored-6.75m) |
| Training | six stages, 6.75M source-onset exposures, CPU, one thread |
| Inputs | seed prefix, event times, and which times require a new note head |
| Outputs | complete four-lane rows: taps, long-note starts and releases |

What R1 does not do:

- It does not listen to audio. Event times come from an existing chart.
- It is checked for legal rows and exact osu! export. Playability, difficulty
  and musical quality have not been established; that review is still open.

### Generate a chart

Install [uv](https://docs.astral.sh/uv/) and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Use `--extra cuda` instead of `--extra mps` on Linux with NVIDIA.

```sh
git clone https://github.com/ensomi-labs/ensomi-model.git
cd ensomi-model
uv sync --python 3.10 --extra mps
hf download sed-i/pulsefield-r1-restored --local-dir artifacts/hf-r1

uv run --python 3.10 --extra mps python -m \
  ensomi_model.research.bounded_typed_continuation.generate_hydra \
  checkpoint_file=artifacts/hf-r1/checkpoint.pt \
  checkpoint_sha256=4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70 \
  condition_file=artifacts/hf-r1/examples/alone/condition.json \
  condition_sha256=bfce659d94bbb13a9206d762b04962abab15737ff8ec0d490ff156c5b1244348 \
  output_dir=artifacts/r1-alone-generated \
  device=cpu cpu_threads=1 seed=17
```

The run writes `generated.osu`, `rows.jsonl`, `decisions.jsonl` and
`result.json` into a fresh output directory. On macOS arm64 this reproduces every
note of the published Alone chart. Only the `Artist` and `Creator` header lines
differ, because the published file predates the project's rename. To continue one of your own charts,
prepare a condition from a native 4K `.osu` file first; the
[generation guide](docs/research/bounded_typed_continuation.md#portable-condition-and-generation-commands)
has the command and the condition format.

### How it was trained

| Stage | Exposures at completion | Parameters | Trained |
| --- | ---: | ---: | --- |
| Plain R1 | 4,500,000 | 2,281,104 | entire model |
| Persistent observed seed | 5,000,000 | 2,330,384 | entire model |
| Landmark memory | 6,000,000 | 2,777,232 | entire model |
| Head routing | 6,250,000 | 2,917,008 | added residual only |
| Release routing | 6,500,000 | 3,056,784 | added residual only |
| Two-onset row response | 6,750,000 | 3,084,432 | added residual only |

The [staged restoration](docs/research/r1_staged_restoration.md) describes each
stage and `scripts/r1-restore.sh` runs them.
[`experiments/vacation_rebuild`](experiments/vacation_rebuild/README.md) prepares
the inputs and keeps the recorded recipe. The
[vacation training queue](docs/research/vacation_training.md) is the separate
multi-day queue for a 35M-parameter teacher profile. Training needs the chart
corpus, which is not distributed.

## Problem definition

The [V3 formulation](docs/formulation/README.md) defines the target: legal,
musically coherent 4K choreography generated from complete audio and committed
chart history. Generation always follows a chart identity. Optional scoped
requests (style directives and chart-property targets, with gameplay-demand
requests) apply only to the ungenerated future. Generation is posed on a
decomposed problem with given head times; the full problem also generates
them. The formulation owns the chart language, legality, commit
rules, style and control semantics and the open questions. R1 implements the
chart language and exact replay; it does not yet meet the audio part of that
target.

## Where the research stands

After the R1 release, work from 2026-09-23 to 2026-09-29 tried to replace the
supplied event times with timing learned from audio and to train timing and rows
jointly. It produced working prototypes and evaluation tools, and no model that
qualified as playable. That code is not on this branch. The line that followed
split the problem: the timing system, the rows model and R2 are trained
separately and released together as [Ensomi R2](#released-model-ensomi-r2).

- Research questions, human decisions and outcomes:
  [`RESEARCH.md` on the `relay-notes` branch](https://github.com/ensomi-labs/ensomi-model/blob/relay-notes/RESEARCH.md).
- The code and documents of that period:
  tag [`audio-joint-2026-09`](https://github.com/ensomi-labs/ensomi-model/tree/audio-joint-2026-09).

## Repository map

| Path | Contents |
| --- | --- |
| [`docs/formulation/`](docs/formulation/README.md) | V3 problem definition, notation, gameplay semantics, and style conditions and control |
| [`docs/research/`](docs/research/README.md) | the Ensomi R2 report; R1 task, training recipe and generation guide; index of earlier work |
| `src/ensomi_model/system/` | the end-to-end entry point (`System`, `make`), the prior, the controls and the arrangement loop |
| `src/ensomi_model/audio_rows/` | the timing system (Listener, grid fitter) and the rows model |
| `src/ensomi_model/r2/` | R2: model, sampling, strain, export |
| `src/ensomi_model/research/bounded_typed_continuation/` | R1 model, training and generation |
| `src/ensomi_model/research/r1_restore/`, `vacation_training/` | staged training workers |
| `src/ensomi_model/research/oracle_time_continuation/` | row replay, corpus cache and the causal backbone R1 builds on |
| `src/ensomi_model/research/chart/` | source chart parsing and the lane action schema |
| `src/ensomi_model/osu_core/`, `features/` | osu! parsing and difficulty, audio loading and log-Mel features |
| `experiments/vacation_rebuild/` | input preparation and the recorded six-stage recipe |
| `scripts/` | macOS launchers for the two training workers |

Tests run with `uv run --python 3.10 --extra mps --group dev pytest`.

## History

| Period | Line of work | Where |
| --- | --- | --- |
| 2026-05 to 2026-08 | mapper v2 and v2.1, timing grid fitting, Control V3, the websocket inference service | branch [`legacy/v2`](https://github.com/ensomi-labs/ensomi-model/tree/legacy/v2) |
| 2026-09-13 to 2026-09-15 | scoped style probes, source-action modeling | branch `legacy/v2` |
| 2026-09-15 to 2026-09-21 | oracle-time continuation, R1 stages and release | this branch, tag `r1-restored-6.75m` |
| 2026-09-23 to 2026-09-29 | audio-conditioned joint model | tag `audio-joint-2026-09` |
| 2026-10 | R2, the timing system and the rows model; the end-to-end system | this branch, tags `r2-1.0`, `timing-1.0`, `rows-1.0` |

`legacy/v2` is this branch as it stood before the cleanup. The Ensomi client's
websocket endpoint is served from there. [`docs/research/README.md`](docs/research/README.md)
lists the archived branches.

## License

Ensomi Model is licensed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`). See [`LICENSE`](LICENSE). The released weights, prior,
calibration and previews on Hugging Face are licensed separately: Ensomi R2's
under CC BY-NC-SA 4.0 (no commercial use) and R1's under `AGPL-3.0-only`.
BeatThis, which the timing system downloads, is MIT licensed.
