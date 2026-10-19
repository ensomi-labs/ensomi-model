# ensomi-n

The N line of Ensomi: open music understanding for mappers, starting with
timing. This distribution, `ensomi-n` (package `ensomi_n`), holds the
evaluation harness every N1 candidate is scored by:

- `ensomi_n.labels`: N's own `.osu` timing reader;
- `ensomi_n.mapformat`: the music map, the file an N model writes;
- `ensomi_n.timing`: notated beats, segments and canonical beat lines;
- `ensomi_n.eval`: the measures, nulls, aggregation and the command line;
- `ensomi_n.synth`: synthetic music with exact timing ground truth (suite
  v0, for evaluation E1), rendered in numpy from (setting, seed).

It is MIT licensed ([`LICENSE`](LICENSE)), apart from the rest of
ensomi-model, which is `AGPL-3.0-only`. It never imports the R line
(`ensomi_model`); `tests/test_boundary.py` fails on any such import. R2's
timing enters as files: R2 writes `.osu`, which the harness reads as a
prediction. The root distribution leaves `n/` out (`where = ["src"]`).

The design behind the measures is in members' research notes
(`n1-eval-20261018`, with the decisions `d-n1-eval-answers` and
`d-bl20-no-offset`). This README states every rule the code applies.

## Install and test

Needs Python 3.10 or later and numpy; the tests need pytest.

```sh
cd n
pytest                  # runs from the source tree (pytest's pythonpath is src)
pip install .           # or install it into an environment
```

The tests need no corpus: every fixture is synthetic and asserts absolute
values (a perfect prediction scores 1.0, a half-beat shift 0).

## Command line

```sh
python -m ensomi_n.eval score --refs dataset/0 --preds runs/arch-a/maps \
    [--strata strata.csv] --out report.json
python -m ensomi_n.eval nulls [--out nulls.json]
```

`score` scores every reference set under `--refs` (`.osu` files or
directories) against its prediction in `--preds`.

- **Sets and keys.** A set is the `.osu` files of one directory that name one
  audio file (compared without case). Its key is the directory's path
  relative to the `--refs` root, with `/<audio stem>` appended when the
  directory holds several audio files (`find_sets`).
- **Predictions.** Set K takes the first of `PREDS/K.json` (a music map),
  `PREDS/K.osu`, or the directory `PREDS/K/` holding one music map or `.osu`
  files (the first in sorted order). A set without a prediction is scored as
  an empty prediction and listed under `inputs.missing_preds`, so every arm
  is scored on the same recordings; `--skip-missing` leaves it out.
  Predictions that match no set are listed under `inputs.unused_preds`.
- **Strata table** (CSV with a header): `key` (required), `recording`,
  `stratum` (names separated by `;`), `source` (for example `ranked`,
  `loved`), `start_ms` and `end_ms` (a span; empty for the whole recording),
  `reference` (`1` or `0`). A key may take several rows. Without a table,
  every set is its own recording in stratum `all`.
- **Options.** `--bootstrap N` resamples (default 2000, 0 for none),
  `--seed`, `--flag-below` (confidence under which a window is flagged
  "listen here", default 0.5), `--confident-at` (default 0.9),
  `--decoded-to-osu-ms` (see the clock rule below).
- **Exit status.** 0 on success, 1 when `nulls` finds a null the scorer
  misses, 2 on bad arguments or unreadable input.

The report holds `harness` (version and settings), `roles` (which measure
decides), `nulls`, `inputs`, `strata` (per stratum and per
`stratum|source`: each measure as `{"value", "lo", "hi"}` plus counts), and
`sets` (every set's measures).

## Python API

```python
import numpy as np
from ensomi_n.labels import read_set
from ensomi_n.mapformat import MusicMap
from ensomi_n.eval import score, score_beats, aggregate

ref = read_set('dataset/0/123456')                  # a set's directory, or a list of .osu
pred = MusicMap.from_beats(beats_ms, downbeats=bar_idx, confidence=p, model='arch-a')
s = score(ref, pred)                                # reference first
print(s.bl20, s.offset_ms, s.level, s.bars_same)
quick = score_beats(label_beats_ms, pred_beats_ms, heads=head_ms)   # arrays only
report = aggregate([s, quick], ['rec-1', 'rec-2'])  # recording means, 95 % intervals
```

`score` raises `TypeError` when the prediction comes first.
`MusicMap.from_osu(path)` and `ensomi_n.mapformat.load(path)` read any
`.osu` as a prediction.

## The reader

`read_osu` reads a chart's timing as the mapper wrote it, without judging it:

- every uninherited (red) timing point is a segment: offset, beat length,
  meter. Per-bar metronome resets, odd and changing meters, fractional
  segment boundaries and non-integer tempos are kept;
- inherited (green) points are ignored. A point is red when its
  `uninherited` field is 1, or, in files without that field, when its beat
  length is positive;
- of several red lines at one offset, the last applies, as in osu!; a meter
  below 1 reads as 4;
- bars restart at every red line and repeat every `meter` notated beats; a
  line with omit-first-barline is no downbeat itself, and its first bar line
  falls `meter` beats later, as osu! draws it;
- a red line outside [20, 1000] BPM places no beats, and heads inside it
  weigh nothing;
- the first red line also governs times before it, as osu! applies it.

`read_set` makes a set's reference: the timing of the difficulty with the
most distinct heads, and the union of all difficulties' head times. It
records in `meta['timing_consistent']` whether the difficulties' red lines
agree. R's reader (`ensomi_model.evaluation.redlines`) drops resets, short
bars and detours, so it reads odd-meter downbeats wrongly; it stays
unchanged, and the two readers are separate on purpose.

## Canonical beat lines

Beat lines are compared at the canonical level: each notated beat's tempo is
moved into [80, 160) BPM by a power of two, its fold, so a timing notated at
half or double the tempo gives the same lines.

- A beat slower than 80 BPM keeps its line and gains lines one canonical
  beat apart after it, up to the next beat.
- A beat at 160 BPM or faster is a line only when its count since the last
  anchor is a multiple of `2**-fold`. Anchors are the first beat, every
  segment start, every change of fold, and every downbeat whose bar holds at
  least one canonical beat. So a downbeat is always a line: 7/8 notated as
  240 BPM with meter 7, as 4/4 at 120 BPM with a reset on every bar, or as
  alternating meters 4 and 3 at 240 BPM all give the same lines. A meter of
  1 at 240 BPM (a meter set for visual effect) does not restart the count.
- Where a prediction gives neither segment starts nor downbeats, its first
  beat fixes the phase: a double-time beat list that starts on an off-beat
  has its canonical lines on the off-beats and scores 0.
- With segments, a beat's notated length is its segment's, exactly. A bare
  beat list takes the median of the five intervals around each beat, so
  tracker jitter does not flip the fold.

## The reference and its weights

The scored range runs from the last canonical line at or before the first
head to the first at or after the last head, so every head lies between two
lines. Nothing after the last head is scored: not beats, and not the end of
the bar holding it. Each head splits its weight between the two lines around
it in proportion to its closeness, as its position is interpolated from
them; a head on a line gives that line all of it. Lines without heads
(breaks) weigh nothing in head-weighted measures.

## The music map

One JSON file per song, version 1. The full contract is in the
`ensomi_n.mapformat` docstring.

```json
{
  "format": "ensomi-music-map", "version": 1,
  "model": {"name": "n1-a", "version": "0.1.0"}, "clock": "osu", "audio": "audio.mp3",
  "beats": [512.0, 1012.0, 1512.0],
  "downbeats": [0],
  "confidence": [0.97, 0.95, 0.4],
  "subdivision": [[0.1, 0.8], [0.0, 0.1], null],
  "segments": [{"start_ms": 512.0, "beat_ms": 500.0, "meter": 4}],
  "heads": []
}
```

- `beats`: notated beat times in ms, strictly increasing, at the model's own
  level; required unless `segments` is given. `"beats": []` is an empty
  prediction.
- `downbeats`: indices of bar starts. Absent, bars come from `segments`;
  absent both, the map states no bars and every bar term scores 0.
- `confidence`: per beat, P(a mapper's beat line lies within 20 ms).
- `subdivision`: per beat, `[p_triple, p_duple]` (duple finer than the
  half); both at 0.5 or more mean mixed, neither plain.
- `segments`: the model's red lines. With `beats` they give exact notated
  beat lengths and segment starts; alone they are the timing, applied as
  osu! applies red lines.
- `heads`: notes the model placed; without `subdivision`, each beat's class
  comes from them.
- `clock`: `osu` (osu!'s convention, the charts' clock) or `decoded`. A
  decoded map is scored only with `decoded_to_osu_ms`, one constant measured
  once; the harness never fits it per song.

## Measures

e_b is the signed distance from reference canonical line b to the nearest
predicted canonical line (prediction minus reference); an exact tie goes to
the earlier line.

No measure removes the set offset o unless its name ends in `_o`. Bars and
resets are matched within 20 ms of the reference, as BL-20 is, so a
whole-grid shift that fails BL-20 fails them too. Under a uniform whole-grid
shift, a bar start or a reset passes with chance 40 ms over the bar's length.
With o removed, any shift under half a beat would pass, and the bar-start
chance would be 1 / meter (0.25 in 4/4), above the 0.15 bound for a
deciding statistic. The `_o` variants therefore only characterise.

| Measure | Definition | Role |
| --- | --- | --- |
| BL-τ (`bl20`, `bl10`, `bl5`) | Head-weighted share of reference lines with \|e_b\| ≤ τ. No offset is fitted or subtracted | BL-20 decides, by the human's rule; no measure is validated against listening yet |
| BL per beat (`bl20_beat`, ...) | The same, every line in range weighing 1 | Characterises |
| With o removed (`bl10_o`, `bl5_o`, `bars_start_o`, `bars_same_o`, `bars_precision_o`, `reset_recall_o`) | BL-10 and BL-5 on e_b − o; bars and reset recall with the predicted bar starts and segment starts moved by −o | Characterise how the timing is shaped apart from its phase; never decide |
| Chance (`chance20`, ...) | Head-weighted 2τ / B over the reference lines: BL-τ of a random-phase prediction at the reference tempo | Characterises |
| S-BL20 (`s_bl20`) | Share of recordings with BL-20 ≥ 0.9 | Characterises |
| Set offset o (`offset_ms`) | Head-weighted median of e_b over the whole range (the midpoint where weight splits exactly). Across recordings: median, robust SD (1.4826 × MAD), shares with \|o\| ≤ 5, 10, 20 ms | Its own term; never removed from BL, bars, resets or any other measure without the `_o` suffix |
| Level (`level`, `level_<class>`) | Per reference segment, r = predicted over reference notated beat length (the predicted step at the segment's middle, or for a bare beat list the median interval of its beats inside the segment): correct \|r − 1\| ≤ 0.5 %, octave (2^k), family (3/2, 4/3, 5/4 and inverses), other, missing. Duration-weighted share correct | Characterises |
| Bars (`bars_start`, `bars_same`, `bars_precision`) | Share of reference bar starts with a predicted one within 20 ms, no offset removed; share of reference bars whose start and end both match with no predicted bar start inside (scored only when both ends lie in range); share of predicted bar starts matching a reference one | Characterises |
| Boundaries (`boundary_recall`, `boundary_precision`) | Tempo runs end where a step leaves the run's first step by more than 0.5 % (1 % for beat lists). A long change sits between two runs of 8 s or more whose tempi differ by more than 1 %, at the earlier run's end. Matched within one reference canonical beat | Characterises |
| Reset recall (`reset_recall`) | Same-tempo red lines off the previous segment's grid by more than 2 ms, matched by a predicted segment start or downbeat within 20 ms, no offset removed; against the union of a recording's sets | Characterises |
| Subdivision (`subdivision_kappa`, `subdivision_accuracy`) | Per reference notated beat with two or more heads: each head snaps to the smallest d in 1, 2, 3, 4, 6, 8, 12, 16 within 2 ms; the beat's divisor is their lcm. Plain (1, 2), duple (power of two ≥ 4), triple (3 divides, 4 not), mixed (3 and 4 divide), finer (a head off every d). Compared with the predicted class at the beat's middle | Characterises |
| Calibration (`ece`, `auc`, `confident_fail`) | Over scored lines with a confidence (the nearest predicted line's, the lower of its two beats for an inserted line): ECE over 10 bins; AUC of low confidence for failing lines (p to 0.001); failure rate among lines with p ≥ 0.9 beside the rate p predicts | Characterises |
| Mapper work (`edits_*`, `edits_per_min`, `check_load_s_per_min`, `hidden_per_min`, `hidden_share`, `redo_s_per_min`) | Windows are reference bars (4 s without bars); a window fails when a scored line in it fails BL-20 and is flagged when one has p below `--flag-below`. Edits per audio minute: one offset edit when \|o\| > 10 ms; one re-time per reference segment with a failing line; one deletion per predicted red line more than one beat from every reference red line; one level edit per run of segments at the wrong level; one meter edit per scored bar not matched. Redo: duration of segments where under half the lines pass | Characterise until validated against mapper case studies |

With `--strata` spans, a stratum's measures use only heads, lines, bars and
segments inside its spans; o still comes from the whole recording, as one
offset serves the song.

## Aggregation

The source recording is the decision unit: a recording's value is the mean
over its sets (only the sets marked `reference=1` when any is), and pooled
counts (calibration bins, kappa, boundaries, resets, windows) are averaged
the same way, so every recording weighs 1. Intervals are 95 % percentile
bootstraps over recordings. Means skip recordings where a measure is
undefined, and `n` says how many remain.

## Nulls

A prediction at the right tempo with a random phase passes a line exactly
when its own line falls within τ, so the chance of BL-τ is 2τ / B at
canonical beat length B: under one uniform shift of the whole grid and under
an independent phase per beat alike.

| Canonical BPM | 80 | 100 | 120 | 140 | 160 (limit) |
| --- | --- | --- | --- | --- | --- |
| BL-20 | 0.0533 | 0.0667 | 0.0800 | 0.0933 | 0.1067 |
| BL-10 | 0.0267 | 0.0333 | 0.0400 | 0.0467 | 0.0533 |
| BL-5 | 0.0133 | 0.0167 | 0.0200 | 0.0233 | 0.0267 |

`python -m ensomi_n.eval nulls` runs the scorer on each null: the reference
against itself scores 1, shifts of half and a third of a beat score 0, the
mean over 1,000 equally spaced whole-grid shifts lies within 0.001 of
2τ / B, and one draw of per-beat phases over 20,000 beats within 0.01.

## Not built yet

- The legacy S1-20 and H-20 columns: their exact definitions are in scripts
  outside this repository.
- Anchored BL needs a model's re-detection runs; score each anchor count as
  its own prediction directory.
- No strata census: strata and spans come from the table.
- The decoded-to-osu clock constant is not measured here.
- Without play counts, a recording's reference set is the one the table
  marks; otherwise its sets are averaged.

## Synthetic suite

```sh
python -m ensomi_n.synth list
python -m ensomi_n.synth render --audio DIR --truth DIR [--settings a,b] [--seeds 0,1] [--commit SHA]
```

Each setting is a symbolic program for one hard-timing stratum (steady 4/4
controls, mixed subdivision, odd and changing meters, tempo steps, ramps,
rubato, edits, polyrhythm, beatless and sparse spans, syncopation, swing,
80/160 BPM). A clip is 60 s of 24 kHz mono 16-bit WAV; its voices are
computed in numpy, so nothing is downloaded. The ground truth is on the
decoded-audio clock: `<id>/map.json` (a music map, clock `decoded`, with the
rest of the truth under `truth`), `<id>/onsets.json` (every layer's onsets)
and `<id>/truth.osu`. `manifest.jsonl` records each clip's seed, commit and
WAV sha256. Charts are on osu!'s clock, about 20 ms earlier: scoring
converts with one constant (`decoded_to_osu_ms`), never per clip.

`ensomi_n.synth.e1` scores baseline predictions on the suite. Synthetic
results say where a timing system fails on audio of known structure, not
how it does on real music.
