# The Ensomi repositories: where shared facts live and what a change affects

Ensomi is six repositories and a model repository on Hugging Face. Some facts
are stated in several of them: the released model's identity, files and
licence, versions, the messages programs exchange, the rules of generation.
This page says where each kind of fact is defined, so that its current value
is read there, and what a change in one repository affects in the others. It
names places, not values.

[`ensomi-cross-repo-changes`](../.agents/skills/ensomi-cross-repo-changes/SKILL.md)
says how to make such a change.

## Repositories

| Repository | Owns | Takes from the others |
| --- | --- | --- |
| ensomi-model | the models, their training and evaluation, the released Python system, the model service, the exporter that writes model packages, this page | the protocol (the model service); beatmap-lens's annotation dataset releases |
| runtime | the Rust core every client binds to: generation, gameplay rules, the model-package format and its reader; the Swift and npm packages | the protocol, whose messages it mirrors; model packages |
| protocol | EnsomiProtocol: the messages programs exchange, with their error codes and phases | — |
| ensomi | the macOS and iOS app | the runtime and the protocol; the model package on Hugging Face; the model service, its chart-server engine |
| ensomi-web | the site and its in-browser generator | the runtime; the model package on Hugging Face |
| beatmap-lens | the chart renderer and the annotation tools | ensomi-model, whose code some of its tools run |
| the model repository on Hugging Face | the released model's files, its package manifest and its card | ensomi-model's exporter and release step write it |

Each repository's README and `LICENSE` give its URL, default branch and
licence.

## Where to look things up

| What you need | Where it is defined |
| --- | --- |
| The released model's identity, files, sizes, hashes, licence and licence notices | its package manifest on Hugging Face, at the release tag the clients pin. ensomi-model's exporter writes it; the runtime owns the format and documents it. |
| The model revision a client downloads | that client's model source: the app's model downloader, the site's model URL and its cache name |
| What a package holds besides the manifest (graph interfaces, constants) | the package config the exporter writes, which the runtime's model loader reads |
| A runtime or protocol version, and who pins it | that package's own metadata; each consumer pins an exact version in its own manifest (the app's project file, the site's `package.json`, the extras in ensomi-model's `pyproject.toml`) |
| Python and dependency ranges | ensomi-model's `pyproject.toml`; the protocol's `pyproject.toml` for the range its Python package accepts |
| Messages, error codes and phases | the protocol's `.proto` files |
| Generation behaviour: star range, controls and their statuses, section rules, arrangement passes, random numbers, the decode loop | for a model in development, ensomi-model's Python system, which the model service serves to the app; for a released model, its package and the runtime's pipeline for it; the R2 report for the evidence behind each control's status |
| Licences | the Hugging Face card and the manifest for the model's files; each repository's `LICENSE` for its code; the runtime's `licenses/` for the third-party notices it ships |
| A speed, size or compatibility claim | the measurement or check that backs it, which the claim names |
| Research decisions and their reasons | ensomi-model's `relay-notes` branch |

## What a change affects

Each row lists, in order, what follows a change of that kind. A release or an
upload comes before the repositories that take it.

| Change | What follows |
| --- | --- |
| New or changed model files | ensomi-model exports them and writes the manifest; they are uploaded to Hugging Face under a new tag; each client moves its pinned revision; the card; the report, if behaviour changed |
| The package format or the package config | the runtime, which owns the format, and its release; ensomi-model's exporter; a new package on Hugging Face; the clients take the runtime release and the new revision |
| A protocol message, error code or phase | the protocol's release; the runtime's mirror of the messages, and its release; ensomi-model's model service and its pin; the app; the site |
| The runtime's API or generation behaviour | the runtime's release; the app and the site take it; the report, the card and client text wherever they describe the behaviour |
| A generation rule or a control's status | for a model in development, nothing outside ensomi-model; for a released model, a release of that model: its package, the runtime's pipeline if the behaviour changes, the protocol if a message documents the rule, then the report, the card, client checks and text |
| A licence, or the files a licence covers | the card and the manifest; every README and client screen that states it; the files the clients download |
| A dependency range | the owner's metadata; every consumer's pin; a fresh resolution in each consumer |
| A measured claim (size, speed) | the measurement first; then every place that states it |

## Where a model's rules live

A model's rules (its star range, controls and their statuses, section rules,
arrangement passes, seed handling and decode loop) are written while it is
developed, in ensomi-model's Python system, and change there freely. The model
service serves the current Python system to the app; it does not have to
refuse what the runtime refuses. The runtime runs released models for users
and is where strictness belongs: package checks, request validation and
conformance. Its model-specific behaviour comes from a released Python system;
it defines none of its own.

The Python system and the runtime are reconciled only when a model is
released: ensomi-model's exporter writes the package from the Python system,
and the runtime's generator replay parity (step 5 of its `RELEASING.md`) shows
that its pipeline reproduces the Python system's decisions on recorded draws.
Between releases they are not kept in step, and a change to a model in
development needs no Rust or Wasm build.

R2's host code is compiled into the runtime as its built-in pipeline, and the
runtime restates some of R2's values, so these rules are written in more than
one program: the model identity, the controls with their statuses and the star
range in ensomi-model's model service and in the runtime; section rules in
ensomi-model's system, its model service and the runtime; the `.osz` layout in
the Python system and in the runtime's Rust and JavaScript. Whether those
values move into the package, and whether a package carries its model's host
code, are decided once the next model's architecture is known.
