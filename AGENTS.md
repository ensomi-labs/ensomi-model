# Ensomi Coding Agent Guide

## Start here

Use `README.md` for the project purpose, the released model and where earlier
work lives. Inspect the relevant source and nearby tests before editing.

## Task routing

Use task-specific guidance only when its scope matches the work.

| Task | Resource |
| --- | --- |
| Repository prose writing, review, trimming, restoration, or comment and documentation coverage | `.agents/skills/ensomi-prose-standard/SKILL.md` |
| Evidence-backed simplification surveys, dead or duplicate surface audits, and scoped cleanup proposals | `.agents/skills/ensomi-find-simplifications/SKILL.md` |
| Outgoing-diff test selection, pre-push evidence, force-with-lease safety, or readiness claims | `.agents/skills/ensomi-pre-push-checks/SKILL.md` |
| Authoring-session, review, PR, change-narration, or reasoning-transcript leakage in durable prose | `.agents/skills/ensomi-trim-cot-leakage/SKILL.md` |
| Packaged Hydra configs, mapper training presets, inference profiles, config adapters, CLI entrypoints, or Hydra tests | `.agents/skills/hydra-conventions/SKILL.md` |
| ML research direction, analogue search, hypothesis branching, bounded experiment design, or result evaluation | `.agents/skills/research-triage/SKILL.md` |
| Root-cause analyses, performance investigations, or postmortems | `docs/guides/technical_analysis_writing.md` |

## Repository context

- `main` holds the V3 formulation and the released R1 lineage. Mapper v2/v2.1,
  the timing stack, Control V3 and the websocket inference service are on the
  `legacy/v2` branch. `docs/research/README.md` maps earlier work to its refs.
- Research questions, human decisions and outcomes are recorded on the
  `relay-notes` branch, entry point `RESEARCH.md`. The `agent-notes` branch is
  archived read-only at `refs/archive/heads/agent-notes`; do not recreate it or
  write new Agent Notes.
- `artifacts/` and `dataset/` are ignored. Generated evaluations, caches,
  checkpoints, datasets, and run snapshots are not repository sources of truth
  and may be absent in a fresh clone. Do not scan them broadly unless the user
  names one.
- Put durable conclusions and reusable constraints in curated `docs/`
  documentation.

## Verification

Run the smallest relevant check first. Keep accelerator extras explicit for
model-backed commands: `mps` on Apple Silicon and `cuda` on Linux with NVIDIA.
