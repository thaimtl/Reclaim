#!/usr/bin/env bash
# Supervise one training run: restart from the last checkpoint after a crash, then benchmark.
# TRAINER=svr.train_svr1 supervises SVR-1 (default svr.train, the Gaddy-architecture baseline).
#
#   setsid nohup svr/supervise.sh <run name> <benchmark channels> [svr.train args...] \
#       > checkpoints/<run name>/supervisor.log 2>&1 < /dev/null &
#
# exit code 0 from training -> run bench.run_model + bench.score + bench.report
# exit code 3 (stopped by SIGTERM / Ctrl-C, checkpoint saved) -> stop, no restart, no benchmarks
# anything else (crash, kill -9, OOM) -> restart from the checkpoint, up to MAX_RESTARTS times;
#   give up if two crashes in a row happen at the same optimizer step (no progress)
set -u
cd "$(dirname "$0")/.."
NAME=$1; BENCH_CHANNELS=$2; shift 2
PY=venv/bin/python
TRAINER=${TRAINER:-svr.train}
MAX_RESTARTS=${MAX_RESTARTS:-5}
OUT_ROOT=${OUT_ROOT:-checkpoints}
RUN_DIR=$OUT_ROOT/$NAME
mkdir -p "$RUN_DIR"
log() { echo "[supervisor $(date '+%F %T')] $*"; }

step_of_checkpoint() {
  $PY -c "import json,sys
try:
    import torch; s = torch.load('$RUN_DIR/ckpt/latest.pt', map_location='cpu', weights_only=False); print(s['opt_step'])
except Exception: print(-1)" 2>/dev/null
}

restarts=0; last_crash_step=-2
while true; do
  log "starting training (restart $restarts): $PY -m $TRAINER --name $NAME --out-root $OUT_ROOT $*"
  $PY -m "$TRAINER" --name "$NAME" --out-root "$OUT_ROOT" "$@" >> "$RUN_DIR/train_stdout.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then log "training finished"; break; fi
  if [ $code -eq 3 ]; then log "training stopped on request (checkpoint saved); not restarting"; exit 3; fi
  step=$(step_of_checkpoint)
  log "training crashed with exit code $code; last checkpoint at optimizer step $step"
  if [ "$step" = "$last_crash_step" ]; then log "no progress since the previous crash; giving up"; exit 1; fi
  last_crash_step=$step
  restarts=$((restarts + 1))
  if [ $restarts -gt $MAX_RESTARTS ]; then log "too many restarts; giving up"; exit 1; fi
  sleep 30
done

if [ "${SKIP_BENCHMARK:-0}" = "1" ]; then log "SKIP_BENCHMARK=1, done"; exit 0; fi
BENCH_NAME=${BENCH_NAME:-${NAME}_ours}
log "benchmarking $RUN_DIR/model.pt as $BENCH_NAME ($BENCH_CHANNELS)"
$PY -m bench.run_model --name "$BENCH_NAME" --checkpoint "$RUN_DIR/model.pt" --channels "$BENCH_CHANNELS" \
    --description "${BENCH_DESCRIPTION:-Run $NAME ($TRAINER) trained on the Gaddy & Klein data; training config in $RUN_DIR/config.json}" \
  && $PY -m bench.score "${BENCH_NAME}_cpu" "${BENCH_NAME}_gpu" \
  && $PY -m bench.report \
  && log "benchmarks done" || { log "benchmark chain failed"; exit 1; }
