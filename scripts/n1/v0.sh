#!/usr/bin/env bash
# N1 v0 on the mac, as resumable stages. From the project root:
#   bash <src>/scripts/n1/v0.sh data  <data-dir> <src> [processes]          curation, then caches (CPU)
#   bash <src>/scripts/n1/v0.sh chain <data-dir> <run-dir> <src> <seed> <total-min> [max-jobs]
#        refs once, then training jobs of at most 50 minutes each until the run's verdict.json exists
#   bash <src>/scripts/n1/v0.sh eval  <data-dir> <run-dir> <src> [extra args for ensomi_n.run.v0 eval]
# A stage whose marker <dir>/stage-<name>.done exists is skipped. Every stage appends to <dir>/stages.jsonl,
# and a heartbeat line goes to <dir>/progress.jsonl every 60 s while the script lives.
set -uo pipefail
MODE=${1:?data, chain or eval}
PY=.venv/bin/python
note() { echo "{\"t\": $(date +%s), \"event\": \"$2\"}" >> "$1/stages.jsonl"; echo "[$(date -u +%H:%M:%S)] $2"; }
heartbeat() { ( while kill -0 $$ 2>/dev/null; do echo "{\"t\": $(date +%s), \"event\": \"heartbeat\", \"mode\": \"$MODE\"}" >> "$1/progress.jsonl"; sleep 60; done ) & HB=$!; trap 'kill $HB 2>/dev/null' EXIT; }
stage() {
  local dir=$1 name=$2; shift 2
  if [ -f "$dir/stage-$name.done" ]; then note "$dir" "skip $name"; return 0; fi
  note "$dir" "start $name"
  if "$@"; then touch "$dir/stage-$name.done"; note "$dir" "done $name"; else note "$dir" "FAILED $name"; return 1; fi
}
if [ "$MODE" = data ]; then
  DATA=${2:?data dir}; SRC=${3:?src}; NP=${4:-6}
  export PYTHONPATH="$SRC/src:$SRC/n/src"
  mkdir -p "$DATA"; heartbeat "$DATA"
  stage "$DATA" curate $PY -m ensomi_n.run.v0 curate --data "$DATA" --src "$SRC" --processes "$NP" || exit 1
  stage "$DATA" cache $PY -m ensomi_n.run.v0 cache --data "$DATA" --processes "$NP" || exit 1
  note "$DATA" "data finished"
elif [ "$MODE" = chain ]; then
  DATA=${2:?data dir}; RUN=${3:?run dir}; SRC=${4:?src}; SEED=${5:?seed}; TOTAL=${6:?total minutes}; MAXJ=${7:-8}
  export PYTHONPATH="$SRC/n/src"
  mkdir -p "$RUN"; heartbeat "$RUN"
  SHA=$(basename "$SRC")
  stage "$RUN" refs $PY -m ensomi_n.run.v0 refs --data "$DATA" --run "$RUN" || exit 1
  for j in $(seq 1 "$MAXJ"); do
    if [ -f "$RUN/verdict.json" ]; then break; fi
    if [ -f "$RUN/STOP" ]; then note "$RUN" "STOP file found"; break; fi
    note "$RUN" "start job $j"
    if PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 $PY -m ensomi_n.run.v0 train \
        --data "$DATA" --run "$RUN" --seed "$SEED" --total-min "$TOTAL" --job-min 50 --src-sha "$SHA" >> "$RUN/train.log" 2>&1; then
      note "$RUN" "done job $j"
    else
      note "$RUN" "FAILED job $j"; break
    fi
  done
  note "$RUN" "chain finished"
elif [ "$MODE" = eval ]; then
  DATA=${2:?data dir}; RUN=${3:?run dir}; SRC=${4:?src}; shift 4
  export PYTHONPATH="$SRC/src:$SRC/n/src"
  heartbeat "$RUN"
  note "$RUN" "start eval $*"
  if $PY -m ensomi_n.run.v0 eval --data "$DATA" --run "$RUN" "$@" >> "$RUN/eval.log" 2>&1; then note "$RUN" "done eval $*"; else note "$RUN" "FAILED eval $*"; fi
fi
