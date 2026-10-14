# Contributing to ensomi-model

ensomi grew out of the osu!mania community: its models learn from charts that
mappers made, and it gives back open weights, open code, open datasets and
open research. Results, decisions and working notes are public on the
[`relay-notes` branch](https://github.com/ensomi-labs/ensomi-model/blob/relay-notes/RESEARCH.md).
What mappers and players tell us decides two things: what our models are
optimized to do better, and what we choose to build. The most useful
contribution today needs no code.

English, Japanese and Chinese are all welcome, in reports and in discussions.

## Feedback on a generated chart

1. Generate a chart at [ensomi.app](https://ensomi.app), in the ensomi app, or
   with the [command in the README](README.md#generate-a-chart).
2. Play it, or read it in the osu! editor.
3. Open a [chart feedback report](https://github.com/ensomi-labs/ensomi-model/issues/new?template=chart-feedback.yml).

Any report helps, even one sentence or a general impression. These make it
more useful, and all of them are optional:

- a screenshot of the moment, or its time range (an osu! editor timestamp
  works too);
- the `.osu` file renamed to `.txt`, because GitHub does not accept `.osu`
  attachments;
- the song: artist and title, or a link;
- the model, for example "Ensomi R2 1.0 on ensomi.app", with the star and seed
  listed under the result (or `receipt.json` from the command), so that we can
  regenerate the chart;
- what a mapper would do instead, and what worked.

Never attach audio or the `.osz` file: songs are copyrighted, and the `.osz`
contains the song. A song name or link is enough.

What happens to it:

- We read every report and label it by what it is about: timing, rhythm,
  patterns, long notes, difficulty, section requests, flow.
- Reported charts become test cases for our next models: R3, the successor to
  R2, and the bigger model built for mappers. When a model has been checked
  against a report, we reply on it with the result.
- Reports are published as an open dataset under CC BY 4.0, credited by the
  name each reporter gives.

## Feedback on tools and the dataset

Mapping and developer tools and the published pattern dataset belong to
[beatmap-lens](https://github.com/ensomi-labs/beatmap-lens), and their feedback
goes there:

- the Inspector, the chart renderer, the npm package or the annotation tools:
  a bug, something that gets in the way, or a tool you wish existed, in a
  [tooling feedback report](https://github.com/ensomi-labs/beatmap-lens/issues/new?template=tooling-feedback.yml);
- a label in the [dataset](https://huggingface.co/datasets/sed-i/mania-pattern-annotations)
  that reads wrong to you, a Foundation concept, or another problem in the
  dataset, in a
  [dataset feedback report](https://github.com/ensomi-labs/beatmap-lens/issues/new?template=dataset-feedback.yml).

## Ideas: what ensomi should build

Controls you wish you had, ways to play, anything ensomi could become: post
them in [Discussions](https://github.com/orgs/ensomi-labs/discussions), where
others can add to them and upvote them. Questions and charts you made with
ensomi go there too. Problems with ensomi.app go to
[ensomi-web](https://github.com/ensomi-labs/ensomi-web/issues), and with the app
to [ensomi](https://github.com/ensomi-labs/ensomi/issues).

## Code

Pull requests for these are welcome without asking first:

- a bug fix with a test that fails before it;
- clearer documentation, error messages or setup steps;
- the generate command, the model service or the exporter, when the released
  system's charts stay the same.

Open an issue or a discussion before changing a model, its training, its
evaluation or a generation rule. Research here follows recorded decisions (the
[`relay-notes` branch](https://github.com/ensomi-labs/ensomi-model/blob/relay-notes/RESEARCH.md))
and the guardrails in
[`ensomi-research-guardrails`](.agents/skills/ensomi-research-guardrails/SKILL.md):
a measure decides only after it is shown to agree with human judgments, and a
change is judged against the variation between seeds and checkpoints. A short
discussion first avoids building on a direction that has already been ruled
out. A change to the released model's files, a version, a protocol message or a
generation rule of a released model reaches other repositories; see
[`docs/ecosystem.md`](docs/ecosystem.md).

### Setup and tests

Install [uv](https://docs.astral.sh/uv/), then use the `mps` extra on Apple
Silicon or the `cuda` extra on Linux:

```sh
uv sync --python 3.10 --extra mps --group dev
uv run --python 3.10 --extra mps --group dev pytest
```

Training needs the chart corpus, which is not distributed. Generation needs the
released weights; the [README](README.md#generate-a-chart) has the download.

### Pull requests

- One piece of work per pull request, with tests for the behaviour it changes
  and the documentation it affects.
- Match the style of the surrounding code and documentation. A claim about
  charts or speed names the check or measurement behind it.
- Say whether the change alters generated charts, and how you checked.

## Licence

Code contributions are made under the repository's licence, `AGPL-3.0-only`.
The released weights on Hugging Face are licensed separately, Ensomi R2's under
CC BY-NC-SA 4.0; see the [README](README.md#license).
