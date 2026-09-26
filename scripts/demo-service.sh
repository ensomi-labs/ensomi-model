#!/bin/sh
# Resolve local research assets from the repository shared by managed worktrees.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
common_git_dir=$(git -C "$project_dir" rev-parse --path-format=absolute --git-common-dir)
asset_dir=$(dirname -- "$common_git_dir")
export ENSOMI_DEMO_CHECKPOINT="${ENSOMI_DEMO_CHECKPOINT:-$asset_dir/artifacts/joint-audio/20260926-common-prefix-outcomes-r1-v1/actor-128/step-128.pt}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$asset_dir/.venv}"
export PYTHONPATH="$project_dir/src"
cd "$project_dir"
exec uv run --no-sync --extra mps python -m ensomi_model.serving.hydra "$@"
