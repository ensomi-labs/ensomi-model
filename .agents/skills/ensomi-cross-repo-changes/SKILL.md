---
name: ensomi-cross-repo-changes
description: Keep facts that several Ensomi repositories state in agreement. Use when a change touches a released model's files, revision, manifest, licence or generation rules, versions and pins, or protocol messages, when it states such a fact somewhere new, and before a release, a tag or a Hugging Face upload. A model in development, and the model service that serves it, stay in ensomi-model and are out of scope.
---

# Ensomi cross-repository changes

Whoever changes a shared fact makes every Ensomi repository agree with it in
the same round and leaves none of that work to someone else.
[`docs/ecosystem.md`](../../../docs/ecosystem.md) says where each fact is
defined and what a change affects.

1. **Change the source, then everything it reaches.** Edit the fact where it
   is defined and regenerate derived files (the manifest, `SHA256SUMS`,
   bindings, lock files) with their own commands. Sweep ensomi-model,
   ensomi-runtime, ensomi-protocol, ensomi, ensomi-web and beatmap-lens
   (`git -C ../<repo> grep -n -i '<old>'`) and the Hub card's raw `README.md`
   for the old value and wording, and fix every hit in the same round. Open
   one pull request per repository, in the order `docs/ecosystem.md` gives: a
   release or upload before the repositories that take it.
2. **Point rather than copy.** A document links to a value's source; where it
   must restate the value, it names the source. A reference to another
   repository's code, files or packages pins a tag, a commit or an exact
   version, except a link to a document meant to stay current, such as the R2
   report, which may follow `main`.
3. **Claim only what is checked.** Resolve a dependency change from what is
   declared (`uv lock`, a fresh environment or CI), not from an environment
   changed by hand. State a size, speed or compatibility only with the check
   behind it.
4. **Keep `docs/ecosystem.md` current** when an owner, or a dependency between
   repositories, appears, moves or goes.

Each pull request names its sweep, what it found and the round's other pull
requests.
