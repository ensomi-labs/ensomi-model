# Ensomi R2: from a song to a playable chart

Ensomi R2 turns a song into an osu!mania 4K chart. Give it an audio file and a
star rating, and a few seconds later it writes a complete `.osz` that opens in
osu!. It is the first Ensomi release that starts from audio alone. The previous
release, [R1](bounded_typed_continuation.md), could only continue a chart that
a person had already mapped.

This report explains what the project wants a chart generator to be, the
decisions that shaped R2 and why they were made, how the system works, and what
it can and cannot do yet.

| | |
| --- | --- |
| Weights | [`sed-i/ensomi-r2`](https://huggingface.co/sed-i/ensomi-r2) on Hugging Face, MIT |
| Code | tags `r2-1.0.1` and `rows-1.0.1`, with `timing-1.0`; `r2-1.0` and `rows-1.0` mark the first release commit; AGPL-3.0-only |
| Components | a timing system (no trained weights of its own), a rows model (2.1M parameters) and R2 (2.4M parameters) |
| Input | an audio file, a star rating from 2 up to 6, optional requests |
| Output | a legal osu!mania 4K `.osz` with the audio, and a receipt of every setting used |
| Speed | 8-12 s for a 2-3.3 minute song on one CPU thread of an Apple M5 |

## What we are building toward

Ensomi wants to change how music and choreography are joined, so that
listening to music, playing it and creating for it become one activity. Three
scenarios carry that aim:

- **Ambient play.** Music is already playing around you. Ensomi recognises it,
  finds the current position and plays a chart, generated in real time, that
  follows the music without interrupting it.
- **Mapping.** A mapper selects one section and asks for candidate
  choreography while the rest of the chart stays in place. Candidates are
  compared, kept, retried or undone. Choice and expression stay with the
  mapper.
- **Training.** A player picks music they like and a movement to practise, and
  the chart is shaped around it.

From these follow the properties a generated chart needs:

- **Legal and playable** at the requested difficulty, from 2 to 6 stars. The
  reference is the whole population of osu!'s ranked and loved 4K charts in
  that range: mostly ordinary charts, together with the expressive styles that
  ranked mapping contains. A generator should be able to produce what that
  population produces, not one average chart.
- **Musical.** Notes fall on the music's events; the chart rests where the
  music rests and repeats where the music repeats.
- **Coherent over a whole song.** A human chart has a character (how many long
  notes, how dense the chords, how the hands share the work) that the mapper
  chooses once and holds to the end. A generated chart needs one too.
- **Steerable.** Difficulty, long-note amount and section styles such as stream
  or jack are requests that change the output in the requested way while it
  stays playable.
- **Causal in its decisions.** The system may listen to the whole song, but it
  places notes in order, so that one day a chart can be produced ahead of
  playback.

R2 serves the simplest form of the mapping and training scenarios: a whole
song, generated while you wait. It does not yet play along with music in real
time, and it does not regenerate one section inside an existing chart.

## How we got here

**Earlier lines of work, May to September 2026.** A mapper, a timing-grid
fitter, a control system and later an audio-conditioned joint model all
produced charts. A review at the end of September found that their own
evaluators could not tell their failures from their successes, and that some
claims described components the played system did not run. That work is on
branch `legacy/v2` and tag `audio-joint-2026-09`.

**R1, September.** The project restarted from a small model with an exact
chart contract. R1 continues a chart from a seed, at event times taken from an
existing chart. Its legality, replay and export are exact, and that core lives
on in R2. As a chart maker it failed, for two reasons. The event times it was
given included the source chart's long-note releases, so its long notes were
partly another mapper's choreography. And it was trained only to predict the
next row of human charts, which did not keep it good over a whole song of its
own output.

**R2, October.** R2 arranges notes on given head rows. Trained first on
natural structure alone, it produced legal charts that drifted over long songs.
Once the cause was found (below), a decode-time fix and corpus-measured
playability limits turned it into a demo that does not drift and responds to
four controls, without new training. Head-row generation was then redesigned
from scratch to hear the music's rhythm, and with it the system runs from audio
alone. The project's lead screened the result blind on 2026-10-11 and judged it
playable. That system is this release.

## The decisions that shaped R2

**The ear decides; measures describe.** No automatic measure has yet been
shown to recognise what an experienced player rejects in a chart. Ranked charts
contain the very patterns a player objects to in generated ones, so a defect is
a matter of where and how often, not whether. Every quality decision in R2 was
therefore made by blind listening: pairs of charts in Beatmap Lens Compare,
sides hidden, criteria written before the pairs were built. Measures are used
to describe and diagnose. A capability is claimed only after such a check, and
human charts serve as a reference, not as a ceiling the system may not pass.

**An exact contract under everything.** Every chart is a sequence of complete
rows whose legality and long-note state are derived by replay. The model can
only choose among legal rows, and every output is parsed back and replayed
after export. Whatever else is wrong with a chart, it is a valid chart.

**Split when notes fall from how they are arranged.** The
[formulation](../formulation/README.md) poses generation in two steps: first
the head rows, the moments where at least one note starts; then the
arrangement on them, which lanes, taps or long notes, and where each long note
ends. [Measurements](head_time_decomposition.md) showed that head times fix
the density, its shape through the song and most of the difficulty, but only
part of a chart's character. So each half has its own job: the rows must carry
the music's rhythm, and the arrangement must make the choices the rows leave
open.

**No release times are given.** R1's given releases leaked another mapper's
long notes into the output. R2 instead makes one decision per head row, and
that decision also places the releases of the long notes the row closes. Notes
born together can end together or apart.

**Natural structure first, control at decode time.** R2 learned the corpus's
own structure with no difficulty or style inputs. Controls act at decode time
on the frozen model by tilting probability inside each class of actions, so a
style request can change which lanes a row uses but not how many notes it has
or whether it uses long notes.

**Every chart chooses a character and holds it.** Free-running over a whole
song, R2's long-note level and density wandered until the chart settled into
one texture. The cause was precise: R2 copies the level of its recent past
almost one to one (a slope of .85 to .97 per 64 rows), as real charts do. In a
real chart that persistence holds a level the mapper chose for the whole chart;
R2 had no such choice, so its level performed a random walk. A search at decode
time toward a target level held it, which showed that the right level was among
R2's proposals. What was missing was a choice and something to hold it. The
system now takes the choice from the corpus, as the median of the 64 human
charts whose head rows look most like these, and a restoring force holds it to
the end.

**The beat grid comes from the song's own sound.** A pretrained beat tracker
gets the tempo right but places beats several milliseconds from where mappers
put them. The song's own note onsets, read from its spectrogram, sit much
closer, so each tempo segment is refitted to them. This needs no training.

**The rows hear the music, beat by beat, and remember it.** The rows model
chooses a rhythmic figure for each beat from what it hears at each position in
the beat, and four copy heads let it repeat an earlier figure where the music
sounds the same. Charts repeat where songs repeat; a model that looks back by
musical similarity can do that, where one that only remembers the last few
rows cannot.

**Playable first, then more.** When the project chose between adding audio to
R2, a copy mechanism and fixing the drift, it chose a playable demo first: no
mistakes, no drift, responding to control. R2 therefore does not hear the
music. Measurements showed that audio would help it place chords and jacks,
and that work is deferred. A part that did not earn its place in blind checks
ships switched off, as R2's own copy path does.

## How R2 makes a chart

```text
audio ─> timing system ─> rows model ─> corpus prior ─> R2 with controls ─> .osz
         beat grid        head rows     chart-level      lanes, taps,
                                        targets          long notes
```

### Timing system

BeatThis, a published beat tracker, finds beats and downbeats, which are
grouped into constant-tempo segments. Each segment's phase and tempo are then
refitted to the song's own onsets from the log-Mel spectrogram. Within a chart
those onsets sit about 2.7 ms (robust SD) from the mapper's beats, against 6.5
ms for the beat tracker's peaks.

On a sealed holdout of 519 human charts with 289 recorded audio-file hashes,
held out from the timing system's tuning but among R2's training charts, 85 %
of charts have 90 % of their heads within 20 ms of the fitted grid (78 % before
the refit), and 93 % of all heads fall within 20 ms. The grid costs about 1.5 s
per minute of audio. Without a chart to read, some things stay out of reach: an
offset of about 10 ms that all charts of one mapset share, short tempo sections
and phase resets. No one has yet screened the grids by ear.

### Rows model

At each beat the rows model picks one *figure*: the set of positions inside
the beat, on a lattice of sixteenths and twelfths, that carry a note head. Its
vocabulary of 8,853 figures covers 99.6 % of the beats in the training charts.
A figure's score combines the spectrogram at each of its positions, a
transformer that hears 64 beats either side, and a recurrent prior over the
figures already chosen. Four copy heads look 1 to 64 beats back, keyed by how
alike the audio sounds. The star rating maps to a workload the model is asked
to produce.

Beats are counted from the start of each tempo segment of the grid, so a
figure's positions lie on its own segment's beat lines. Where a segment ends
partway through a beat, a remainder under half a beat gets no positions; a
longer one is a beat cut short, whose positions count only up to a sixteenth of
a beat before the next segment starts, the space a whole beat leaves after its
last position.

The model uses every input: shifting the spectrogram, removing the
per-position audio or the copy heads, or giving it another chart's request all
cost likelihood in teacher-forced tests. On 96 held-out development audio
files (by recorded hash, with 231 human charts), with two generation seeds, it
beats the earlier head generator on every rhythm measure by margins two to ten
times the seed and checkpoint spread, in every star band.
These measures describe the output; none has been validated against a player.

| Measure | Rows model | Earlier generator | Human charts |
| --- | ---: | ---: | ---: |
| heads on onsets, over a half-beat shift | .050 | .024-.028 | .052 |
| rows repeating where the audio repeats | .19 | .05 | .234 |
| head F1 against human charts at 20 ms | .74 | .66 | .868 (two humans) |
| same figure on comparable beats | .71 | .61 | .978 (two humans) |

These measures, and the blind pairs in
[What has been checked](#what-has-been-checked), were taken with beats
numbered continuously across segments, as `r2-1.0` numbers them: after a
segment boundary at a fractional beat, positions sat off that segment's beat
lines by the fraction
([#34](https://github.com/ensomi-labs/ensomi-model/issues/34)). A grid of one
segment gives the same chart either way.

Two decode-time rules keep long songs alive. Left free, the copy heads copied
rests, rests reinforced themselves, and long songs thinned out toward the end.
The copy heads may therefore attend to an empty beat only where the music is
quiet, on the song's quietest 40 % of beats. That keeps the last fifth of long
songs close to the requested density and raises the share of rests in quiet
passages from .053 to .076, against .085 in human charts. This holds the
problem at decode time; its training-time cause is not yet separated. A
section request tilts the density of figures inside its span.

### The chart's character

The rows come with no arrangement and so with no character. The corpus prior
supplies one: for the requested star band it finds the 64 training charts whose
head rows look most like these (rows per second, median gap, workload, tempo)
and takes their median long-note level, heads per row and share of rows where
all four lanes are busy, together with a base difficulty. These targets play
the part of the mapper's choice for the whole chart.

### R2

R2 has 2.4M parameters. A causal temporal convolution network reads the head
rows and everything already placed; R2 does not read the audio. At each head
row it chooses a code for every lane at once, 625 joint actions in all. On a
free lane the code means nothing, a tap or a long-note head. On a lane holding
a long note it means keep holding, release on this row, or release earlier in
the gap before it, possibly starting a new note on this row. A pointer then
places each gap release. Mirror symmetry between the hands is built into the
model.

It was trained on osu!'s ranked and loved 4K charts from 2 to 6 stars: 11,368
eligible training charts in 4,167 groups used to keep related sources within
the same data partition, associated with 5,026 recorded audio-file hashes.
Another 1,163 charts, in 424 other groups, were held out for development, and
the corpus's reference parts were never opened. Neither groups nor hashes
count unique songs; the [data provenance report](r2_data_provenance.md) gives
each stage's charts, groups, audio hashes and metadata labels. Training was
teacher-forced cross-entropy for 64 million head-row decisions, about 4.4
passes. The released checkpoint is the one at 56 million. Development loss
was flat from 40 million (2.075 nats per decision at 56 million), and among the
checkpoints on that plateau only 56 million passed the project's checks on very
short holds, releases just before another note and level drift. Those checks
were set by the project and have not been validated against a player.

At generation time three things act on R2's choices:

- **The restoring force** pulls the running long-note level, heads per row and
  all-busy share over the last 64 rows back toward the targets. On human head
  rows it keeps the long-note level about as close to its target as a human
  chart stays to its own mean (RMSE .075 against .071; .139 without the
  force).
- **Playability limits** measured per star band on human charts: a minimum
  time before re-pressing the same lane (145 ms at 2 stars, 75 ms at 5), and
  caps on four-note runs and jack runs. When no action survives, a single legal
  note is placed and counted.
- **The requests** described below.

## What you can ask for

| Request | What it does | Status |
| --- | --- | --- |
| Star rating | sets the rows' workload and the arrangement's difficulty for the whole song | **works**: both blind pairs called the right way |
| Jumpstream, chordstream sections | an 8-32 s span in that style, at a strength from 0 to 1 | **works** in blind pairs |
| Stream, jack sections | the same, plain styles | reach their target band in measurement (48 of 48 and 35 of 48 runs); not screened on their own |
| Chordjack sections | the same | out of band on generated rows (18 of 28 runs); not offered as working |
| Long-note amount | tap only, half, or natural | **does not work yet**: "half" removes about a third, and a blind pair could not tell |
| Section difficulty | an absolute star target for one span | not yet screened; spans asked to be easier land about .15 log units short |
| Copy in R2 | the arrangement repeats its own earlier patterns | off: on generated rows it did not make the arrangement follow the music's repetition |

## What has been checked

**Mechanically, every time.** Every chart passes the replay and legality
checks, every time in the `.osu` is a whole millisecond, and the export is
parsed back and replayed. At each release, every head and release of the four
example charts lies on a 1/1 to 1/16, 1/3, 1/6 or 1/12 line of its own red line,
and no two head rows are nearer than a sixteenth of a beat
(`python -m ensomi_model.evaluation.self_snap`); the runtime's release checks
run the same check on its own charts. `r2-1.0` failed it on Catch the Moment
([#34](https://github.com/ensomi-labs/ensomi-model/issues/34)).

**By ear.** The project's lead judged blind pairs in Beatmap Lens Compare.
Each pair is one listener and one generation seed per side. In the two pairs
that secretly compared the system with itself under different seeds, the
listener still picked a side, so one pair's preference is weak evidence.

- New rows against the earlier generator: **10 to 3, one undecided.** The new
  side was preferred for catching the rhythm and the details in quiet
  passages. The old side won three times: once for more breath in a dense
  passage, once where the new rows broke a late section into abrupt short
  phrases, and once with the listener unsure.
- Whole-song difficulty: both star pairs called the right way.
- Jumpstream and chordstream: work. Long-note amount: fails.
- Copy on against off: 1 to 2, one no difference.
- R2 against the human chart of the same song: **the human chart was chosen in
  all three pairs.** On this evidence R2 has not reached human charts.

**Reproduction.** Generated from the published files, from outside the
repository and with BeatThis freshly downloaded, the four example charts come
out with every note identical. For `r2-1.0.1`, Catch the Moment's example was
regenerated with its code; the other three come out identical with it.

## What it does not do yet

- **Long notes.** The long-note control does not work on generated rows, and
  earlier screens placed the main problems in where long notes fall against the
  sound.
- **Section difficulty** is unscreened, and easier sections land short of
  their target.
- **Breath.** In one dense passage the rows streamed in quarter beats where the
  mapper used half beats; in one quiet passage they left too much silence.
- **Long-song stability** rests on the decode-time rest rule, not on training.
- **R2 does not hear the music.** Its arrangement follows the music only
  through the head rows.
- **Real time and local edits.** It generates a whole song at a time and cannot
  regenerate one section inside a fixed chart.
- **One judge.** Every verdict above is one person's.

Since the release, the Ensomi macOS app generates a song and plays it end to
end, and one runtime, shared by that app and the web page at
[ensomi.app](https://ensomi.app), generates charts for people's own songs on
their own computers from the downloaded model. [Serve the app](#serve-the-app)
says how.

## Generate a chart

Install [uv](https://docs.astral.sh/uv/) and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
The release was checked on macOS arm64 (Apple M5, Python 3.10.20, torch
2.11.0, CPU, one thread). Linux and CUDA (`--extra cuda` instead of
`--extra mps`) were not run.

```sh
git clone https://github.com/ensomi-labs/ensomi-model.git
cd ensomi-model
git checkout --detach r2-1.0.1
uv sync --python 3.10 --extra mps
hf download sed-i/ensomi-r2 --revision r2-1.0.1 --local-dir artifacts/hf-r2

uv run --python 3.10 --extra mps python -m ensomi_model.system.make \
  --model-dir artifacts/hf-r2 --audio /path/to/song.mp3 --star 3.5 \
  --title "Song title" --artist "Artist" --seed 0 --out artifacts/my-chart.osz
```

`--model-dir` reads `system.json` from the directory and resolves every path
in it there, so the command runs from any working directory. The first run
downloads BeatThis's `final0` checkpoint (81 MB) from its authors. Decoding
audio other than WAV needs ffmpeg. Requests, for example:

```sh
  --stream 60000,90000,0.7,jumpstream    # a jumpstream section from 60 s to 90 s at strength .7
  --ln tap                               # no long notes
```

`python -m ensomi_model.system.make --help` lists every option. Without
`--title` and `--artist` the chart is named after the audio file.

On the macOS 27.0 build used for the check (26A428), the scipy 1.15.3 wheel
that `uv sync` installs failed to load (`dlopen` rejects its `__thread_bss`
section); the check used a repaired copy of the same version. Other macOS
versions were not tried.

## Serve the app

Once the model is installed, the Ensomi macOS app makes charts by default on
the device that plays them. The iOS target shares that path and has so far been
built only for the simulator. Since 2026-10-12 the app generates through the
shared [Ensomi runtime](https://github.com/ensomi-labs/runtime), a Rust core
whose generator runs the released networks with ONNX Runtime as the ONNX step
graphs that `python -m ensomi_model.export.onnx` writes
(`src/ensomi_model/export/`). They are published as the `onnx/` directory of
the Hugging Face repository. The app downloads the files that directory's
`ensomi-package.json` lists and checks each one's SHA-256, or reads the package
from a folder; nothing is sent to a server. The same core, built for
WebAssembly, runs the in-browser generator at [ensomi.app](https://ensomi.app).
The app pins the runtime's exact version in its `project.yml`, the site in its
`package.json`.

The runtime arranges a song once. `r2-1.0.1` arranges it a second time when the
first arrangement misses the star's difficulty target by more than the
difficulty fit's seed spread, and for such a song the runtime's chart differs
from the release's. Two of the four example charts, SaiGetsu and Catch the
Moment, took that second pass. The receipt's `controls.difficulty` records
every pass.

The app keeps this repository's model service as its other engine, the chart
server, selectable on the same screen. The service loads the model directory
once and answers the complete-chart messages of
[EnsomiProtocol](https://github.com/ensomi-labs/protocol) (`chart.proto`, from
version 0.2.0) as binary protobuf `Envelope` frames. Like the runtime, it
arranges each song once (`max_difficulty_passes=1`), while the command in
[Generate a chart](#generate-a-chart) keeps the release's second pass.

```sh
uv sync --python 3.10 --extra mps --extra service
uv run --python 3.10 --extra mps --extra service python -m ensomi_model.service \
  --model-dir artifacts/hf-r2 [--host 127.0.0.1] [--port 8765] [--cache-dir ensomi-service-cache]
```

The app connects to `ws://localhost:8765` unless `ENSOMI_SERVICE_URL` or its
`EnsomiServiceURL` default names another address, and sends

- `CapabilitiesRequest`, answered with `ModelCapabilities`: the model name and
  version, the checksums in the model directory's `SHA256SUMS`, the star range
  [2, 6) and each control with its status from "What you can ask for";
- `GenerateChartRequest` with a non-empty `session_id`, the audio as
  `local_path` (the service reads that path, so app and service run on one
  machine), a star, an optional seed (default 0), the long-note amount and
  section requests. It is answered with `GenerationProgress` as each phase
  starts (loading, timing, rows, arrangement, export), then one `ChartResult`:
  the chart as `{lane, start_ms, end_ms?}` objects in whole milliseconds with
  no latency applied, the song length, a receipt, and the path of the `.osz`
  written to the cache directory;
- `StopSessionRequest` for that session to give up: the result is discarded and
  the service answers with a stopped `StatusEvent`.

Failures arrive as `ErrorEvent` with the code `protocol_error` (a frame the
service cannot parse or handle, or one that breaks the session flow, such as a
request without a `session_id`), `star_out_of_range`, `audio_unreadable`,
`unsupported_control` (a request the model refuses, for example a section
shorter than 8 or longer than 32 seconds) or `generation_failed`. One
chart is generated at a time and further requests wait. A stopped generation
still runs to its end, so the next one starts after it.

## Files and licences

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `rows/model.pt` | 9,622,005 | `46b6ca37870120fda4d5de632949921fc26e020cf80abe44b4ba7f6f9557dda4` |
| `r2/model.pt` | 9,655,631 | `8dedefe4b11d238ad0b3186e6dfefe31f52cb333091aafe2e8efda8ad2ee3c6f` |
| `prior.json` | 1,958,310 | `be5741d520120aaeb7ecf9126c2c12401015095b107b025584802f8e715590af` |
| `calibration.json` | 443,193 | `d39cd16fda451163d00d70cd6a913eb4bf64326ecd577feab08c25d8c22455ab` |
| `scopes.parquet` | 2,447,837 | `61f8648523cddb86e7316812b46896410f12b1a6ae179dcbf082c8b8615e3134` |
| `system.json` | 209 | `333d1c59a5ca199e07369b980992c2bb9683851246807426f1152260502e9718` |

`SHA256SUMS` in the Hugging Face repository lists every file, including the
four example charts with their receipts and the previews. `r2/model.pt` holds
only R2's weights and model configuration; the training checkpoint also held
optimizer state. The example receipts of Hesitation Snow, SaiGetsu and
Adventure record the training checkpoint's hash; Catch the Moment's,
regenerated for `r2-1.0.1` from the published files, records `r2/model.pt`'s.
The `.pt` files are PyTorch pickles, loaded by this repository's code.

Everything in the Hugging Face repository (weights, prior, calibration, scope
table, ONNX graphs, examples, previews and model card) is licensed MIT. BeatThis's
parts, `onnx/beatthis.onnx`, an ONNX export of its `final0` checkpoint, and its
front-end arrays `onnx/beatthis-mel-filter.bin` and `onnx/beatthis-hann-window.bin`,
are MIT under BeatThis's own copyright notice (`onnx/BeatThis-LICENSE.txt`, which
`runtime.json` names). The code is AGPL-3.0-only. The Python system reads nothing in `onnx/`:
the `beat_this` package downloads BeatThis (code and weights, MIT) from its
authors.

The training charts are osu! community beatmaps. Neither they nor their audio
are distributed. The prior, the calibration and the scope table hold numbers
derived from them, with no titles, artists, mappers or file paths. The example
charts carry their songs' titles and artists, without audio.

## Code and training lineage

| Component | Trained from | Released at |
| --- | --- | --- |
| Timing system | no training | `timing-1.0` |
| Rows model | commit `cf753aa`, 5,000 steps; the run's recorded settings, not the defaults of a later checkout, are authoritative | `rows-1.0`; decoder `rows-1.0.1` |
| R2 | commit `3954031`, 64M decisions without conditions; checkpoint at 56,000,574 | `r2-1.0`; system `r2-1.0.1` |

`timing-1.0`, `rows-1.0` and `r2-1.0` mark the first release commit.
`rows-1.0.1` and `r2-1.0.1` mark the commit whose rows decoder places each
segment's rows on its own beats, with the same files; it holds the code that
runs them. The training commits are earlier commits on `main`.
