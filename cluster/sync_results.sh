#!/usr/bin/env bash
# Copy training results to the private bucket. Never deletes anything there (no --delete).
#   bash cluster/sync_results.sh SVR1_4ch [more runs]           once
#   bash cluster/sync_results.sh --watch SVR1_4ch [more runs]   every 30 min until every run is finished, then once more
# Synced: checkpoints/<run>/ except ckpt/ (model.pt = the EMA weights, config.json, log.jsonl, progress and supervisor
# logs), benchmark_results/ (the run's benchmarks and the leaderboard), and data/svr1_targets/ (scale.json and the
# articulator targets). train_job.sh runs it once when a run finishes; run --watch on a login node (in tmux) too, since
# compute nodes may have no internet.
set -u
cd "$(dirname "$0")/.."
. cluster/env.sh
B=${RECLAIM_BUCKET:-hf://buckets/austinkomachi/Austin_CH}
HF=$(command -v hf || true)
[ -n "$HF" ] || HF=venv-svr/bin/hf
[ -x "$HF" ] || HF=venv/bin/hf
WATCH=0
if [ "${1:-}" = "--watch" ]; then WATCH=1; shift; fi
[ $# -gt 0 ] || { echo "usage: bash cluster/sync_results.sh [--watch] <run> [run...]"; exit 1; }
INTERVAL=${SYNC_INTERVAL:-1800}
MAX_HOURS=${SYNC_MAX_HOURS:-36}
log() { echo "[sync $(date '+%F %T')] $*"; }

finished() {  # trained and benchmarked (or the benchmark failed), the supervisor gave up, or the run was skipped
  local sup=checkpoints/$1/supervisor.log
  [ -f "checkpoints/_skipped_$1" ] && return 0
  { [ -f checkpoints/$1/DONE ] && grep -qE "benchmarks done|benchmark chain failed|SKIP_BENCHMARK" "$sup" 2>/dev/null; } \
    || grep -q "giving up" "$sup" 2>/dev/null
}

sync_all() {
  local fail=0 r
  for r in "$@"; do
    if [ -d "checkpoints/$r" ]; then
      "$HF" sync "checkpoints/$r" "$B/checkpoints/$r" --exclude "ckpt/*" || fail=1
    else
      log "no checkpoints/$r yet"
    fi
  done
  [ -d benchmark_results ] && { "$HF" sync benchmark_results "$B/results/benchmark_results" --exclude "*_smoke*" || fail=1; }
  [ -d data/svr1_targets ] && { "$HF" sync data/svr1_targets "$B/data/svr1_targets" || fail=1; }
  [ $fail -eq 0 ] && log "synced: $*" || log "a sync failed (no internet here, or not logged in: hf auth login)"
  return $fail
}

if [ $WATCH -eq 0 ]; then sync_all "$@"; exit $?; fi
start=$(date +%s)
while true; do
  all=1
  for r in "$@"; do finished "$r" || all=0; done
  sync_all "$@"
  if [ $all -eq 1 ]; then log "every run finished and synced: $*"; exit 0; fi
  if [ $(( $(date +%s) - start )) -gt $(( MAX_HOURS * 3600 )) ]; then log "giving up after $MAX_HOURS h"; exit 1; fi
  sleep "$INTERVAL"
done
