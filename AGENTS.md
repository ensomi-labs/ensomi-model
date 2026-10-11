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
| Planning, briefing, launching or reading an experiment, training run, evaluation or design round; choosing a measure, baseline or decision rule | `.agents/skills/ensomi-research-guardrails/SKILL.md` |
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

## Research guardrails

Agents here have repeated the same research failures under new names. Before
planning, briefing, launching or reading an experiment, read
`.agents/skills/ensomi-research-guardrails/SKILL.md` and apply the guardrails
whose trigger holds. The core:

- A measure decides only after it is shown to recognise the human's labels.
  Until then it characterises, and the human's screen decides.
- Measure the seed and checkpoint noise of a decision statistic before setting
  a margin. Never use a selected checkpoint as the only baseline.
- Before training on a new input, show with a teacher-forced ablation that the
  model uses it.
- Work out what the fixed inputs let the model learn and the evaluation see:
  supplied head rows, warm start, panel and scope.
- Before a launch, put the human's contradicting evidence to the human. If the
  human cannot be reached, run only measurements.
- Briefs for subagents and Astra name the skill instead of copying the rules.

## Verification

Run the smallest relevant check first. Keep accelerator extras explicit for
model-backed commands: `mps` on Apple Silicon and `cuda` on Linux with NVIDIA.
