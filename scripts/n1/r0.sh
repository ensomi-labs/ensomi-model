#!/usr/bin/env bash
# N1 rung 0 on the mac, in two resumable groups of stages (each under 55 minutes).
# Usage, from the project root: bash <src>/scripts/n1/r0.sh <a|b> <run-dir> <src>
#   a: labels and manifest, Mel cache, clock fixtures, onset check, throughput (MPS, alone),
#      stored BeatThis logits mapped, BeatThis final0 for val-n1 (MPS)
#   b: R2 as exported (CPU, in the background) beside BeatThis final1 and final2 for the panel (MPS)
# A stage whose marker <run>/stage-<name>.done exists is skipped; each stage is itself resumable.
set -uo pipefail
GROUP=${1:?a or b}
RUN=${2:?run dir}
SRC=${3:?source dir (tarball)}
PY=.venv/bin/python
BULK=artifacts/bulk/n1-cache
export PYTHONPATH="$SRC/src:$SRC/n/src"
mkdir -p "$RUN"
# A heartbeat line in <run>/progress.jsonl every 60 s for the job's life: other workers on the mac use
# the GPU only when no runs/*/progress.jsonl was written in the last 3 minutes.
( while kill -0 $$ 2>/dev/null; do echo "{\"t\": $(date +%s), \"event\": \"heartbeat\", \"group\": \"$GROUP\"}" >> "$RUN/progress.jsonl"; sleep 60; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT
note() { echo "{\"t\": $(date +%s), \"group\": \"$GROUP\", \"event\": \"$1\"}" >> "$RUN/stages.jsonl"; echo "[$(date -u +%H:%M:%S)] $1"; }
stage() {
  local name=$1; shift
  if [ -f "$RUN/stage-$name.done" ]; then note "skip $name"; return 0; fi
  note "start $name"
  if "$@"; then touch "$RUN/stage-$name.done"; note "done $name"; else note "FAILED $name"; return 1; fi
}
if [ "$GROUP" = a ]; then
  stage labels $PY -m ensomi_n.run.r0 labels --run "$RUN" --bulk $BULK --processes 4 || exit 1
  stage cache $PY -m ensomi_n.run.r0 cache --run "$RUN" --bulk $BULK --processes 4 || exit 1
  stage clock $PY -m ensomi_n.run.r0 clock --run "$RUN" --bulk $BULK
  stage onsets $PY -m ensomi_n.run.r0 onsets --run "$RUN" --bulk $BULK
  if $PY -c 'import pytest' 2>/dev/null; then
    stage pytest_n $PY -m pytest -q "$SRC/n/tests" -p no:cacheprovider
  else
    note "pytest not importable in .venv; N tests not run here"
  fi
  # The watermark caps MPS allocations at the recommended working set, so an oversized batch raises
  # (recorded as an error row) instead of swapping.
  stage throughput env PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 $PY -m ensomi_n.run.throughput --out "$RUN/throughput.json"
  stage bt_map_fitdev $PY "$SRC/scripts/n1/bt_logits.py" map --stored fitdev --ckpt final0 --split "$RUN/split.json" \
      --subset panel --out "$RUN/bt-map-fitdev.json"
  stage bt_map_reset $PY "$SRC/scripts/n1/bt_logits.py" map --stored reset --ckpt final0 --split "$RUN/split.json" \
      --subset reset --out "$RUN/bt-map-reset.json"
  stage bt_final0_val $PY "$SRC/scripts/n1/bt_logits.py" compute --ckpt final0 --split "$RUN/split.json" \
      --subset val_hard,val_random,panel --verify 5 --max-minutes 30 --run "$RUN" --out "$RUN/bt-final0-val.json"
elif [ "$GROUP" = b ]; then
  F0=$(ls -d $BULK/bt-final0-* | head -1)
  ( stage r2_export $PY "$SRC/scripts/n1/r2_export.py" --split "$RUN/split.json" --subset panel,val_hard,val_random \
      --logits "$F0" --out $BULK/r2-export --processes 3 --max-minutes 50 --result "$RUN/r2-export.json" ) &
  R2PID=$!
  stage bt_final1_panel $PY "$SRC/scripts/n1/bt_logits.py" compute --ckpt final1 --split "$RUN/split.json" \
      --subset panel --max-minutes 25 --run "$RUN" --out "$RUN/bt-final1-panel.json"
  stage bt_final2_panel $PY "$SRC/scripts/n1/bt_logits.py" compute --ckpt final2 --split "$RUN/split.json" \
      --subset panel --max-minutes 25 --run "$RUN" --out "$RUN/bt-final2-panel.json"
  wait $R2PID
fi
note "group $GROUP finished"
