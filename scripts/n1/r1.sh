#!/usr/bin/env bash
# N1 rung 1 on the mac (design §5.1 R1). Usage, from the project root:
#   bash <src>/scripts/n1/r1.sh <step> <r1-run-dir> <r0-run-dir> <src> [seed]
# steps: refs | train (both arms of one seed, about 28 min) | score (one seed's last three checkpoints)
#        | controls | verdict
set -uo pipefail
STEP=${1:?step}; RUN=${2:?r1 run}; R0=${3:?r0 run}; SRC=${4:?src}; SEED=${5:-0}
PY=.venv/bin/python
export PYTHONPATH="$SRC/n/src"
export N1_CODE_COMMIT=$(basename "$SRC")
mkdir -p "$RUN"
echo "{\"t\": $(date +%s), \"event\": \"start $STEP s$SEED\"}" >> "$RUN/stages.jsonl"
case $STEP in
  refs) $PY -m ensomi_n.run.r1 refs --run "$RUN" --r0 "$R0" ;;
  train) $PY -m ensomi_n.run.r1 train --run "$RUN" --r0 "$R0" --seed "$SEED" --arms A1,A2 --budget-min 11 ;;
  score) $PY -m ensomi_n.run.r1 score --run "$RUN" --r0 "$R0" --seed "$SEED" --arms A1,A2 --last 3 ;;
  controls) $PY -m ensomi_n.run.r1 controls --run "$RUN" --r0 "$R0" ;;
  verdict) $PY -m ensomi_n.run.r1 verdict --run "$RUN" --r0 "$R0" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
RC=$?
echo "{\"t\": $(date +%s), \"event\": \"end $STEP s$SEED\", \"rc\": $RC}" >> "$RUN/stages.jsonl"
exit $RC
