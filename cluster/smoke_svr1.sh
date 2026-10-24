#!/usr/bin/env bash
# SVR-1 smoke tests on ONE cluster GPU before every full job, with the full job's exact trainer arguments:
#   bash cluster/smoke_svr1.sh <run name> [svr.train_svr1 args...]
# 1. unit checks (padding invariance, masks, CTC feasibility, finite grads, FA2 == SDPA)
# 2. tiny end-to-end run with a simulated crash   3. resume   4. benchmark chain on the tiny model
# 5. peak VRAM of the worst micro-batches of the full run   6. peak RAM
# Exits non-zero on any failure. Takes ~15 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; shift
PY=venv/bin/python
OUT=checkpoints/_smoke_cluster
rm -rf "$OUT/$NAME"
mkdir -p "$OUT"
fail() { echo "SMOKE FAIL: $*"; exit 1; }
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
TINY="--epochs 2 --max-micro-batches-per-epoch 4 --eval-limit 3 --ckpt-minutes 0 --monitor-seconds 5 --judge-start 0 --judge-ramp 1"

echo "== 0. articulator targets"
if printf '%s\n' "$@" | grep -q -- '--no-artic'; then echo "--no-artic: not needed"
else [ -f data/svr1_targets/stats.json ] || fail "no articulator targets: run python -m svr.precompute_svr1 sparc, then align"
fi

echo "== 1. unit checks"
$PY -m tests.smoke_svr1_units || fail "unit checks"

echo "== 2. tiny end-to-end run"
$TIME $PY -m svr.train_svr1 --name "$NAME" --out-root "$OUT" "$@" $TINY --crash-at-step 3 \
  > "$OUT/$NAME.stdout" 2> "$OUT/$NAME.time" ; code=$?
[ $code -eq 1 ] || { tail -30 "$OUT/$NAME.stdout"; fail "expected the simulated crash (exit 1), got $code"; }
echo "== 3. resume after the crash"
$TIME $PY -m svr.train_svr1 --name "$NAME" --out-root "$OUT" "$@" $TINY \
  >> "$OUT/$NAME.stdout" 2>> "$OUT/$NAME.time" || { tail -30 "$OUT/$NAME.stdout"; fail "resumed run"; }
grep -q "resumed from" "$OUT/$NAME.stdout" || fail "did not resume from a checkpoint"
[ -f "$OUT/$NAME/DONE" ] || fail "no DONE marker"
tail -2 "$OUT/$NAME/log.jsonl"
$PY -c "
import json; r = [json.loads(l) for l in open('$OUT/$NAME/log.jsonl')][-1]
assert 'train_judge' in r or 'no-judge' in '$*', 'judge loss never ran'
import math; assert all(math.isfinite(v) for k, v in r.items() if isinstance(v, float)), r
print('log fields ok:', sorted(k for k in r if k.startswith('train_')))" || fail "log.jsonl"

echo "== 4. benchmark chain on the tiny model (GPU only, 6ch)"
$PY -m bench.run_model --name "_smoke_$NAME" --checkpoint "$OUT/$NAME/model.pt" --channels 6ch --devices cuda \
  --speeds "" > "$OUT/$NAME.bench" 2>&1 || { tail -20 "$OUT/$NAME.bench"; fail "bench.run_model"; }
$PY -m bench.score "_smoke_${NAME}_gpu" >> "$OUT/$NAME.bench" 2>&1 || { tail -20 "$OUT/$NAME.bench"; fail "bench.score"; }
grep -q '"candidate_scores"' benchmark_results/_smoke_${NAME}_gpu/raw/open_vocab_test__6ch.json \
  || fail "SVR-1 candidate scores missing from the raw benchmark output"
rm -rf "benchmark_results/_smoke_${NAME}_gpu"

echo "== 5. worst micro-batches of the full run (peak VRAM)"
$PY -m tests.smoke_svr1_vram "$@" | tail -7 || fail "worst-batch memory test"

echo "== 6. peak RAM (max resident set size, kB)"
grep "Maximum resident set size" "$OUT/$NAME.time" | sort -k6 -n | tail -1 || true
[ -z "$TIME" ] && echo "(/usr/bin/time not available: read RAM from the Erbium report in $OUT/$NAME/erbium/)"
echo "SMOKE PASS: request ~1.5x the peak RAM above; VRAM must stay under ~90% of the card"
