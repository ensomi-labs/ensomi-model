#!/usr/bin/env bash
# N1 v0.1 on the mac: training chains, their evaluation, and the reader-neutral analysis. From the project root:
#   bash <src>/scripts/n1/v01.sh night <src> <analysis-src> <date>
#        seed 0: chain, evaluation, control evaluation; then seed 1's chain on the GPU while seed 0's
#        analysis runs on the CPU (nice 19); then seed 1's evaluations and analysis.
#   bash <src>/scripts/n1/v01.sh gpu <seed> <run-dir> <src> <control-minute>
#   bash <src>/scripts/n1/v01.sh cpu <seed> <run-dir> <analysis-src> <control-minute>
# <analysis-src> holds scripts/n1/v0_analysis.py (branch n1/v0-eval). The control is the run's checkpoint
# nearest <control-minute> training minutes, evaluated alone into <run>/eval-m<minute>/.
# Stop: create <run>/STOP (no new training job starts; v0.sh) and, before seed 1, runs/STOP-v01-s1.
set -uo pipefail
MODE=${1:?night, gpu or cpu}
PY=.venv/bin/python
DATA=artifacts/n1-night-20261018/runs/v0data2-20261019
TUNE=artifacts/n1-night-20261018/runs/v0-s0-20261019/eval
RUNS=artifacts/n1-night-20261018/runs
note() { echo "{\"t\": $(date +%s), \"event\": \"$2\"}" >> "$1/stages.jsonl"; echo "[$(date -u +%H:%M:%S)] $2"; }
nearest() {  # checkpoint tag nearest a training minute
  ls "artifacts/bulk/n1-runs/$(basename "$1")/" | sed -n 's/^ckpt-\([0-9]*\)\.pt$/\1/p' | \
    awk -v t="$(( $2 * 60 ))" '{d = $1 - t; if (d < 0) d = -d; if (best == "" || d < bd) {best = $1; bd = d}} END {print best}'
}
gpu() {
  local SEED=$1 RUN=$2 SRC=$3 CM=$4
  bash "$SRC/scripts/n1/v0.sh" chain "$DATA" "$RUN" "$SRC" "$SEED" 150 >> "$RUN/chain.log" 2>&1
  if [ -f "$RUN/STOP" ]; then note "$RUN" "STOP: no evaluation"; return 1; fi
  bash "$SRC/scripts/n1/v0.sh" eval "$DATA" "$RUN" "$SRC" --panels panel,val_hard --arms v0,r2 --checkpoints last3 \
    --decode-kw "auto:$TUNE" --processes 8
  local CK; CK=$(nearest "$RUN" "$CM")
  note "$RUN" "control checkpoint $CK (nearest minute $CM)"
  bash "$SRC/scripts/n1/v0.sh" eval "$DATA" "$RUN" "$SRC" --panels panel,val_hard --arms v0 --checkpoints "$CK" \
    --decode-kw "auto:$TUNE" --processes 8 --out "$RUN/eval-m$CM"
}
cpu() {  # the final checkpoint first, then the control, then the other two of the last three
  local SEED=$1 RUN=$2 ASRC=$3 CM=$4 NAME; NAME=$(basename "$RUN")
  local MAPS=artifacts/bulk/n1-v01-analysis/$NAME CK; CK=$(nearest "$RUN" "$CM")
  local L3; L3=$(ls "artifacts/bulk/n1-runs/$NAME/" | sed -n 's/^ckpt-\([0-9]*\)\.pt$/\1/p' | sort | tail -3)
  local LAST; LAST=$(echo "$L3" | tail -1)
  local OTHERS; OTHERS=$(echo "$L3" | head -2 | paste -sd, -)
  mkdir -p "$RUN/analysis" "$RUN/analysis-m$CM" "$MAPS/last3" "$MAPS/m$CM"
  run() { PYTHONPATH="$ASRC/src:$ASRC/n/src" OMP_NUM_THREADS=4 nice -n 19 $PY "$ASRC/scripts/n1/v0_analysis.py" "$@"; }
  step() {  # <out> <maps> <checkpoints> <eval-dir> <label>
    note "$RUN" "start analysis $5 ($3)"
    if run maps --run-name "$NAME" --maps "$2" --out "$1" --threads 4 --checkpoints "$3" >> "$1/run.log" 2>&1 && \
       run items --run-name "$NAME" --maps "$2" --out "$1" --eval-dir "$4" >> "$1/run.log" 2>&1; then
      note "$RUN" "done analysis $5"
    else
      note "$RUN" "FAILED analysis $5"
    fi
  }
  step "$RUN/analysis" "$MAPS/last3" "$LAST" "$RUN/eval" last
  mkdir -p "$RUN/analysis/last" && cp "$RUN"/analysis/items-*.jsonl "$RUN/analysis/last/" 2>/dev/null
  step "$RUN/analysis-m$CM" "$MAPS/m$CM" "$CK" "$RUN/eval-m$CM" control
  step "$RUN/analysis" "$MAPS/last3" "$OTHERS" "$RUN/eval" last3
}
if [ "$MODE" = gpu ]; then
  mkdir -p "$3"; gpu "$2" "$3" "$4" "$5"
elif [ "$MODE" = cpu ]; then
  cpu "$2" "$3" "$4" "$5"
elif [ "$MODE" = night ]; then
  SRC=${2:?src}; ASRC=${3:?analysis src}; DATE=${4:?date}
  R0=$RUNS/v01-s0-$DATE; R1=$RUNS/v01-s1-$DATE
  mkdir -p "$R0"
  note "$R0" "night start (src $(basename "$SRC"), analysis $(basename "$ASRC"))"
  gpu 0 "$R0" "$SRC" 100 || { note "$R0" "night stopped after seed 0's chain"; exit 0; }
  note "$R0" "seed 0 GPU part finished"
  cpu 0 "$R0" "$ASRC" 100 &
  CPU0=$!
  if [ -f "$RUNS/STOP-v01-s1" ]; then
    note "$R0" "STOP-v01-s1: seed 1 not started"
  else
    mkdir -p "$R1"
    note "$R1" "seed 1 start"
    gpu 1 "$R1" "$SRC" 80 && note "$R1" "seed 1 GPU part finished"
  fi
  wait "$CPU0"
  note "$R0" "seed 0 analysis finished"
  if [ -f "$R1/eval-m80/compact-val_hard.json" ]; then
    cpu 1 "$R1" "$ASRC" 80
    note "$R1" "seed 1 analysis finished"
  fi
  note "$R0" "night finished"
fi
