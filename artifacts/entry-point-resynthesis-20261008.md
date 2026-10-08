# Why the entry point's problem map and movement were re-synthesised (2026-10-08)

<a id="d-resynthesis"></a>**Agent re-synthesis at the human's request, 2026-10-08 ([private, local](private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-4)); not yet reviewed.** The previous text is kept unchanged in [entry-point-before-20261008](entry-point-before-20261008.md).

## What changed in the problem map, and on what grounds

- **New node `representation`.** The human made the representation of chart sections the first problem ([d-represent-first](r2-representation-20261008.md#d-represent-first)), and the round showed it is a problem of its own:
  - level representations missed what the human marks;
  - relational ones recognise more ([s-represent-answer](r2-representation-20261008.md#s-represent-answer)).
  - The node needs only `contract` and `data`, and the nodes that judge or steer charts need it: `evaluation`, `style`, and `selection` through `evaluation`. The old map had no place for it; it was implicit inside `evaluation`.
- **New node `identity`.** The chart-level choice was spread across `proposal` and `rollout` in the old map. Three results make it a node:
  - R2 copies its recent level, so the level random-walks ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer));
  - the rows fix at most 41 % of identity ([o-skeleton-identity](r2-represent-skeleton-20261008.md#o-skeleton-identity));
  - the human agrees a whole-chart identity vector is probably needed ([d-represent-first](r2-representation-20261008.md#d-represent-first)).
  - It needs `data`, `time` (what the rows already fix) and `representation` (the space it lives in). `proposal`, `rollout` and `control` need it.
- **Edges added; none removed.**

  | Edge | Why |
  |---|---|
  | `evaluation` → `representation` | A measure is a function on a representation |
  | `selection` → `evaluation` | A selector needs a validated score; the decode rounds failed on that ([s-decode-design](r2-bakeoff-night-20261007.md#s-decode-design)) |
  | `style` → `representation` | The Lens concepts are recognisable only in a relational representation ([o-fable-lens](r2-represent-fable-20261008.md#o-fable-lens)) |
  | `control` → `identity` | Chart-level requests act through the chart-level choice |
  | `control` → `evaluation` | A request is checked by measuring the output |
  | `proposal` → `identity` | The arrangement is decided given the chart-level choice |
  | `rollout` → `identity` | Holding quality over the model's own history means holding that choice |

- **Statement changes:**
  - `time` now also asks what the rows decide about the chart ([s-skeleton-achieves](r2-represent-skeleton-20261008.md#s-skeleton-achieves)).
  - `proposal` says "given the rows' times" instead of "timed".
  - `data` names the human labels and the same-rows human pairs.
- **Status change, (proposed):** `control` parked, following the human's decision that controls are a later problem ([d-collapse-primary](r2-collapse-20261007.md#d-collapse-primary)). Parking needs the human's review.
- **Cells shortened.** Each cell keeps one-line claims and links; the earlier cell text, with its history of phase N, the LN level and the guards, is in [old-map-20261008](entry-point-before-20261008.md#old-map-20261008).
- **Grouping.** The map was reordered into three layers: what a chart is and what is known, how it is described and judged, how it is made.

## What changed in the current movement

- The section had become a log of the 2026-10-07/08 rounds. It now holds:
  - the focus the human set (collapse first, controls later; representations first, then measures);
  - three items: in focus, next, and deferred;
  - the running step-1 work;
  - one paragraph of where things stand.
- The rounds' accounts stay in their materials, reached through [old-movement-20261008](entry-point-before-20261008.md#old-movement-20261008).
- **Proposed:** the style formulation, focus item 2 of 2026-10-06, is listed as deferred with `control`. The 2026-10-07 and 2026-10-08 redirects did not mention it, and it has had no work since.

## The human's review of the two proposals (2026-10-08 about 04:33 UTC, [private, local](private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-5))

- <a id="d-control-after-eval"></a>**Decision (human): `control` is not parked.** Control is still part of R2's goal. It is solved after the representation, measure and evaluation work. The node stays `open`, and the "parked (proposed)" status is withdrawn.
- <a id="d-style-parallel"></a>**Decision (human): the style formulation stays in focus.** It is a separate branch, pushed in parallel as formulation work. The proposal to defer it with `control` is withdrawn.
