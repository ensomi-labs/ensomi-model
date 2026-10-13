---
name: ensomi-cross-repo-changes
description: Keep facts that several Ensomi repositories state in agreement. Use when a change touches anything docs/ecosystem.md lists (the released model's files, revision and manifest, licences, versions and pins, generation rules, the model service), when a change states such a fact somewhere new, and before a release, a tag or a Hugging Face upload.
---

# Ensomi cross-repository changes

Several facts are stated in more than one Ensomi repository.
[`docs/ecosystem.md`](../../../docs/ecosystem.md) names each fact's owner and
every place that restates it. A change that edits one place and leaves the
others breaks Ensomi even when every repository's own checks pass.

## Rules

1. **Owner first.** Find the fact in `docs/ecosystem.md` and change it at its
   owner; update the restatements after, never instead. A derived file (the
   manifest, `SHA256SUMS`, generated bindings, a lock file) is regenerated with
   its owning command, not edited by hand.
2. **Sweep before opening a pull request.** Search every repository and the
   Hub card for the old value and its wording, for example:

   ```sh
   for r in ensomi-model ensomi-runtime ensomi-protocol ensomi ensomi-web beatmap-lens; do
     git -C "../$r" grep -n -i -e '<old value or phrase>'
   done
   curl -sL https://huggingface.co/sed-i/ensomi-r2/raw/main/README.md | grep -n -i -e '<old value or phrase>'
   ```

   Fix every hit in the same round, or list it in the pull request with the
   repository that will fix it.
3. **One pull request per repository,** each linking the others and stating
   the merge order. Messages change before the programs that speak them:
   protocol, then the runtime and its release, then ensomi-model and the app,
   then ensomi-web, then the Hub card and its tag.
4. **Pin references across repositories** to a tag, a commit or an exact
   version: Hub `resolve/<tag>` and `--revision <tag>`, `git+...@<commit>`,
   `exactVersion`, an exact npm version. Never `main`. Moving a pin is a change
   of its own, named in the pull request.
5. **Resolve dependencies from what is declared.** Check a dependency change
   with `uv lock`, a fresh environment or CI. An environment changed by hand
   (`uv pip install`, `uv run --no-sync`) cannot show that the declared
   constraints resolve.
6. **State a measurement with its source.** A size, speed or compatibility
   names the measurement or check behind it. A claim nothing checks is not
   made.
7. **Keep the map current.** A change that states a shared fact somewhere new
   adds that place to `docs/ecosystem.md` in the same pull request, or links
   to the owner instead.

## Report

Each pull request lists the sweep (patterns and repositories), what it found,
and the other pull requests of the round.
