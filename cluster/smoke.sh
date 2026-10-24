#!/usr/bin/env bash
# Smoke tests to run on ONE cluster GPU before every full job, with the full job's exact training arguments:
#   bash cluster/smoke.sh <run name> [svr.train args...]
# 1. unit checks   2. tiny end-to-end run (2 epochs, beam eval, logging, model export)
# 3. hard crash + resume   4. peak VRAM of the worst micro-batches of the whole run   5. peak RAM
# Exits non-zero on any failure. Takes ~10 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; shift
PY=${PY:-venv-svr/bin/python}; [ -x "$PY" ] || PY=venv/bin/python  # venv: cluster copies set up before venv-svr
OUT=checkpoints/_smoke_cluster
rm -rf "$OUT/$NAME"
mkdir -p "$OUT"
fail() { echo "SMOKE FAIL: $*"; exit 1; }
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
TINY="--epochs 2 --max-micro-batches-per-epoch 4 --eval-limit 3 --beam-every 1 --ckpt-minutes 0 --monitor-seconds 5"

echo "== 1. unit checks"
$PY -m tests.smoke_train_units || fail "unit checks"

echo "== 2. tiny end-to-end run"
$TIME $PY -m svr.train --name "$NAME" --out-root "$OUT" "$@" $TINY --crash-at-step 3 \
  > "$OUT/$NAME.stdout" 2> "$OUT/$NAME.time" ; code=$?
[ $code -eq 1 ] || fail "expected the simulated crash (exit 1), got $code"
echo "== 3. resume after the crash"
$TIME $PY -m svr.train --name "$NAME" --out-root "$OUT" "$@" $TINY \
  >> "$OUT/$NAME.stdout" 2>> "$OUT/$NAME.time" || fail "resumed run"
grep -q "resumed from" "$OUT/$NAME.stdout" || fail "did not resume from a checkpoint"
[ -f "$OUT/$NAME/DONE" ] || fail "no DONE marker"
$PY -c "from svr.model import load_gaddy_checkpoint; load_gaddy_checkpoint('$OUT/$NAME/model.pt')" 2>/dev/null \
  || echo "note: model.pt is not a Gaddy-architecture checkpoint (fine for SVR-1)"
tail -2 "$OUT/$NAME/log.jsonl"

echo "== 4. worst micro-batches of the full run (peak VRAM)"
MICRO=$(printf '%s\n' "$@" | grep -A1 -- '--micro-batch-samples' | tail -1)
case "$MICRO" in ''|--*) MICRO=432000 ;; esac
$PY -m tests.smoke_worst_batches "$MICRO" | tail -1 || fail "worst-batch memory test"

echo "== 5. peak RAM (max resident set size, kB)"
grep "Maximum resident set size" "$OUT/$NAME.time" | sort -k6 -n | tail -1 || true
[ -z "$TIME" ] && echo "(/usr/bin/time not available: read RAM from the Erbium report in $OUT/$NAME/erbium/)"
echo "SMOKE PASS: request ~1.5x the peak RAM above; VRAM must stay under ~90% of the card"
