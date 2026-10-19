#!/usr/bin/env bash
# N1 rung 1 torch smoke on the mac, before any long training job (about 15 minutes).
# Usage, from the project root: bash <src>/scripts/n1/r1_smoke.sh <scratch-r1-run> <r0-run> <src>
#   1. retry the Mel cache's failed files (present files are skipped);
#   2. rung 0 group a again: only stages without a marker run (the throughput probe, MPS, alone), then
#      NE-M throughput at smaller batches and 30 s crops;
#   3. r1 refs, then r1 train with a 1-minute budget per arm (seed 0), then r1 score of the last
#      checkpoint on val-n1's random half, all into the scratch run.
set -uo pipefail
RUN=${1:?scratch r1 run}; R0=${2:?r0 run}; SRC=${3:?src}
PY=.venv/bin/python
mkdir -p "$RUN"
( while kill -0 $$ 2>/dev/null; do echo "{\"t\": $(date +%s), \"event\": \"heartbeat\", \"step\": \"smoke\"}" >> "$RUN/progress.jsonl"; sleep 60; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT
note() { echo "{\"t\": $(date +%s), \"event\": \"$1\"}" >> "$RUN/stages.jsonl"; echo "[$(date -u +%H:%M:%S)] $1"; }
stage() {
  local name=$1; shift
  if [ -f "$RUN/stage-$name.done" ]; then note "skip $name"; return 0; fi
  note "start $name"
  if "$@"; then touch "$RUN/stage-$name.done"; note "done $name"; else note "FAILED $name"; return 1; fi
}
export N1_CODE_COMMIT=$(basename "$SRC")
stage cache_retry env PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.r0 cache --run "$R0" --processes 2
stage r0a_rest bash "$SRC/scripts/n1/r0.sh" a "$R0" "$SRC"
# NE-M did not fit at 60 s crops and batch 8 under the 17.8 GB cap: measure smaller batches and crops.
WM="PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0"
stage throughput_nem_60s env $WM PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.throughput --only NE-M --batches 1,2,4 \
    --out "$R0/throughput-nem-60s.json"
stage throughput_nem_30s env $WM PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.throughput --only NE-M --frames 3000 \
    --batches 2,4,8 --out "$R0/throughput-nem-30s.json"
stage refs env PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.r1 refs --run "$RUN" --r0 "$R0" || exit 1
stage train env PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.r1 train --run "$RUN" --r0 "$R0" --seed 0 --arms A1,A2 --budget-min 1 || exit 1
stage score env PYTHONPATH="$SRC/n/src" $PY -m ensomi_n.run.r1 score --run "$RUN" --r0 "$R0" --seed 0 --arms A1,A2 --last 1 --panels val_random
note "smoke finished"
