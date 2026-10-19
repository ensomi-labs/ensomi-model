#!/bin/bash
# E1 on the mac (ensomi-model root as the working directory): render synthetic suite v0, then, when no
# training job is active, run the baselines. Stage lines go to <truth>/stages.log, which syncs back.
#   bash <src>/scripts/n1/e1_mac.sh <src> <commit> [render|baselines|all]
set -u
SRC=$1
SHA=$2
WHAT=${3:-all}
OUT=artifacts/n1-night-20261018/synth-v0
AUDIO=artifacts/bulk/n1-synth/v0
LOGITS=artifacts/bulk/n1-synth/v0-logits
PY=.venv/bin/python
mkdir -p "$OUT" "$AUDIO"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$OUT/stages.log"; }
log "start $SHA $WHAT pid $$"
if [ "$WHAT" = render ] || [ "$WHAT" = all ]; then
  PYTHONPATH="$SRC/n/src" nice -n 19 "$PY" -m ensomi_n.synth render --audio "$AUDIO" --truth "$OUT" --commit "$SHA" \
    > "$OUT/render.log" 2>&1 || { log "render FAILED"; exit 1; }
  log "render done: $(wc -l < "$OUT/manifest.jsonl") clips"
fi
if [ "$WHAT" = baselines ] || [ "$WHAT" = all ]; then
  for i in $(seq 0 12); do
    recent=$(find artifacts/n1-night-20261018/runs -name progress.jsonl -mmin -3 2>/dev/null | head -1)
    [ -z "$recent" ] && break
    if [ "$i" -eq 12 ]; then log "training still active after 2 h ($recent): baselines not run"; exit 2; fi
    log "training active ($recent): waiting 10 min"
    sleep 600
  done
  log "baselines start"
  PYTHONPATH="$SRC/src:$SRC/n/src" nice -n 19 "$PY" "$SRC/scripts/n1/r2_on_clips.py" --audio "$AUDIO" \
    --manifest "$OUT/manifest.jsonl" --out "$OUT/preds" --logits "$LOGITS" --device mps \
    --result "$OUT/baselines-result.json" > "$OUT/baselines.log" 2>&1
  status=$?
  log "baselines done, exit $status"
fi
log "end"
