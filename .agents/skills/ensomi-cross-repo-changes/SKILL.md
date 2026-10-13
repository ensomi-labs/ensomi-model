---
name: ensomi-cross-repo-changes
description: Keep facts that several Ensomi repositories state in agreement, and make changes that cross repositories in the right order. Use when a change touches something other Ensomi repositories also state or depend on (the released model's files, revision, manifest or licence; versions and pins; protocol messages; a released model's generation rules), when a change states such a fact somewhere new, and before a release, a tag or a Hugging Face upload. A change to a model in development stays in ensomi-model and is out of scope.
---

# Ensomi cross-repository changes

[`docs/ecosystem.md`](../../../docs/ecosystem.md) says where each kind of
shared fact is defined and what a change affects in the other repositories.
A change that edits one place and leaves the others breaks Ensomi even when
every repository's own checks pass.

## Rules

1. **A model in development stays in ensomi-model.** Its rules change in
   ensomi-model's Python system and need no change, sweep or pull request in
   another repository, and the model service need not match the runtime. They
   reach the other repositories when the model is released
   ([`docs/ecosystem.md`](../../../docs/ecosystem.md), "Where a model's rules
   live").
2. **Source first.** Find where the fact is defined (`docs/ecosystem.md` says
   where to look), read its current value there, and change it there. Update
   the places that restate it after, never instead. A derived file (the
   manifest, `SHA256SUMS`, generated bindings, a lock file) is regenerated with
   its owning command, not edited by hand.
3. **Sweep before opening a pull request.** Search every repository and the
   Hub card for the old value and its wording, for example:

   ```sh
   for r in ensomi-model ensomi-runtime ensomi-protocol ensomi ensomi-web beatmap-lens; do
     git -C "../$r" grep -n -i -e '<old value or phrase>'
   done
   curl -sL 'https://huggingface.co/<model repository>/raw/main/README.md' | grep -n -i -e '<old value or phrase>'
   ```

   Fix every hit in the same round, or list it in the pull request with the
   repository that will fix it.
4. **Follow the order in `docs/ecosystem.md`.** One pull request per
   repository, each linking the others and stating the merge order. A release
   or an upload comes before the repositories that take it.
5. **Pin references across repositories** to a tag, a commit or an exact
   version, never `main`. Moving a pin is a change of its own, named in the
   pull request.
6. **Resolve dependencies from what is declared.** Check a dependency change
   with `uv lock`, a fresh environment or CI. An environment changed by hand
   (`uv pip install`, `uv run --no-sync`) cannot show that the declared
   constraints resolve.
7. **State a measurement with its source.** A size, speed or compatibility
   names the measurement or check behind it. A claim nothing checks is not
   made.
8. **Prefer a link to a copy.** Where a document can point to the source of a
   value, it points; where it must restate the value, it names the source.
9. **Keep `docs/ecosystem.md` current** when a kind of fact moves to another
   owner, a dependency between repositories appears or goes, or a repository
   joins.

## Report

Each pull request lists the sweep (patterns and repositories), what it found,
and the other pull requests of the round.
