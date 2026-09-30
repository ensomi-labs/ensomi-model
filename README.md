# Ensomi Model

Research code for Ensomi V3, a system that generates 4-key rhythm-game charts
(osu!mania 4K) from music. This branch holds the problem definition and the one
released model. Earlier and unfinished lines of work live on other refs, listed
under [History](#history).

## Released model: R1

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
`result.json` into a fresh output directory. To continue one of your own charts,
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
chart history, with optional style and gameplay-demand controls. It owns the
chart language, legality, commit rules and the open questions. R1 implements the
chart language and exact replay; it does not yet meet the audio part of that
target.

## Where the research stands

After the R1 release, work from 2026-09-23 to 2026-09-29 tried to replace the
supplied event times with timing learned from audio and to train timing and rows
jointly. It produced working prototypes and evaluation tools, and no model that
qualified as playable. That code is not on this branch.

- Research questions, human decisions and outcomes:
  [`RESEARCH.md` on the `relay-notes` branch](https://github.com/ensomi-labs/ensomi-model/blob/relay-notes/RESEARCH.md).
- The code and documents of that period:
  tag [`audio-joint-2026-09`](https://github.com/ensomi-labs/ensomi-model/tree/audio-joint-2026-09).

## Repository map

| Path | Contents |
| --- | --- |
| [`docs/formulation/`](docs/formulation/README.md) | V3 problem definition, notation and gameplay semantics |
| [`docs/research/`](docs/research/README.md) | R1 task, training recipe and generation guide; index of earlier work |
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

`legacy/v2` is this branch as it stood before the cleanup. The Ensomi client's
websocket endpoint is served from there. [`docs/research/README.md`](docs/research/README.md)
lists the archived branches.

## License

Ensomi Model is licensed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`). See [`LICENSE`](LICENSE).
