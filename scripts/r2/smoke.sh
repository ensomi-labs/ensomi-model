#!/usr/bin/env bash
# End-to-end R2 smoke on the real cache (mac, repo root): launch through the frozen-code
# supervisor, kill the trainer after its first non-initial checkpoint, let the supervisor
# resume it, and finish with per-checkpoint fit_dev CE and free-run reports.
# Usage: scripts/r2/smoke.sh <run-id> [device] [total_exposures] [checkpoint_every]
set -euo pipefail
RUN=${1:?run id}
DEVICE=${2:-cpu}
TOTAL=${3:-60000}
EVERY=${4:-20000}
DIR=artifacts/r2-runs/$RUN
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch --run-id "$RUN" \
  --config src/ensomi_model/r2/configs/ce_v1.json --set device="\"$DEVICE\"" \
  --set total_exposures=$TOTAL --set checkpoint_every=$EVERY --set warmup_exposures=5000 &
SUP=$!
until ls "$DIR"/checkpoints/ckpt-*.pt 2>/dev/null | grep -v 'ckpt-0000000000' | grep -q .; do
  kill -0 $SUP 2>/dev/null || { echo "supervisor exited before the first checkpoint"; tail -30 "$DIR/trainer.log"; exit 1; }
  sleep 1
done
sleep 3
echo "killing trainer of $RUN to exercise resume"
pkill -9 -f "ensomi_model.r2.train_ce --config .*$RUN/config.json" || true
wait $SUP || true
echo "--- events"; cat "$DIR/events.jsonl"
echo "--- run.json"; cat "$DIR/run.json"
echo "--- checkpoints"; ls -la "$DIR/checkpoints"
echo "--- train.jsonl (last 3)"; tail -3 "$DIR/train.jsonl"
echo "--- evals"; python3 - "$DIR" <<'PY'
import json, sys
for line in open(sys.argv[1] + '/evals.jsonl'):
    r = json.loads(line)
    print(r['checkpoint'], json.dumps(r['fit_dev']))
    for c in r.get('freerun', []):
        for seed, s in c['seeds'].items():
            print('  ', c['sha256'][:12], seed, {k: s.get(k) for k in ('violations', 'heads_present', 'eos_closed', 'ln_share', 'hold_le40', 'seconds', 'error')}, 'source ln_share', round(c['source']['ln_share'], 3))
PY
