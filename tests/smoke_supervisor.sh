#!/usr/bin/env bash
# Smoke-test svr/supervise.sh:
#   A. a crash mid-run is restarted from the checkpoint, the run completes, then the benchmark
#      chain (run_model + score + report) runs on the trained checkpoint
#   B. a SIGTERM to the trainer is an intentional stop: exit 3, no restart, no benchmarks
# Benchmark output goes to benchmark_results/SMOKE_sup_{cpu,gpu}; it is removed afterwards and the
# leaderboard rebuilt without it.
set -u
cd "$(dirname "$0")/.."
OUT=checkpoints/_smoke_sup
rm -rf "$OUT" benchmark_results/SMOKE_sup_cpu benchmark_results/SMOKE_sup_gpu
mkdir -p "$OUT"
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
SMALL="--epochs 2 --max-micro-batches-per-epoch 4 --eval-limit 3 --beam-every 1000 --ckpt-minutes 0 --monitor-seconds 5"

# A: crash once at step 3, auto-restart, finish, benchmark
OUT_ROOT=$OUT BENCH_NAME=SMOKE_sup svr/supervise.sh supA 8ch $SMALL --crash-at-step 3 > "$OUT/supA.log" 2>&1
codeA=$?
check "A supervisor exit code 0" "[ $codeA -eq 0 ]"
check "A crash detected" "grep -q 'crashed with exit code 1' $OUT/supA.log"
check "A restarted once" "[ \$(grep -c 'starting training' $OUT/supA.log) -eq 2 ]"
check "A resumed from checkpoint" "grep -q 'resumed from' $OUT/supA/train_stdout.log"
check "A training finished (DONE)" "[ -f $OUT/supA/DONE ]"
check "A benchmarks ran" "grep -q 'benchmarks done' $OUT/supA.log"
check "A per-case results written" "[ -f benchmark_results/SMOKE_sup_gpu/open_vocab__8ch.json ] && [ -f benchmark_results/SMOKE_sup_cpu/summary.json ]"
check "A leaderboard includes the run" "grep -q SMOKE_sup_gpu benchmark_results/LEADERBOARD.md"
check "A no orphaned monitor after the crash" "! ps -eo args | grep -q '[s]vr.train --name supA'"
check "A Erbium report written" "ls $OUT/supA/erbium/*.txt > /dev/null 2>&1"

# C: kill -9 the trainer -> its Erbium monitor must die with it
venv/bin/python -m svr.train --name supC --out-root $OUT $SMALL --epochs 50 > "$OUT/supC.log" 2>&1 &
trainer=$!
until ls $OUT/supC/erbium/*.txt > /dev/null 2>&1; do sleep 1; done
sleep 3
children=$(pgrep -P $trainer | wc -l)
kill -9 $trainer; wait $trainer 2>/dev/null
sleep 3
check "C trainer had a monitor child" "[ $children -ge 1 ]"
check "C no orphaned monitor after kill -9" "! ps -eo args | grep -q '[s]vr.train --name supC'"

# B: SIGTERM to the trainer -> exit 3, no restart, no benchmarks
OUT_ROOT=$OUT BENCH_NAME=SMOKE_supB svr/supervise.sh supB 8ch --epochs 50 --max-micro-batches-per-epoch 4 \
  --eval-limit 3 --beam-every 1000 --ckpt-minutes 0 --monitor-seconds 5 > "$OUT/supB.log" 2>&1 &
sup=$!
until grep -q '"epoch": 1' "$OUT/supB/log.jsonl" 2>/dev/null; do sleep 2; done
trainer=$(pgrep -f "svr.train --name supB")
kill -TERM $trainer
wait $sup
codeB=$?
check "B supervisor exit code 3" "[ $codeB -eq 3 ]"
check "B no restart" "[ \$(grep -c 'starting training' $OUT/supB.log) -eq 1 ]"
check "B stop reported" "grep -q 'stopped on request' $OUT/supB.log"
check "B checkpoint saved on SIGTERM" "grep -q 'checkpoint saved' $OUT/supB/train_stdout.log"
check "B no benchmarks" "! grep -q 'benchmarking' $OUT/supB.log && [ ! -d benchmark_results/SMOKE_supB_gpu ]"

rm -rf benchmark_results/SMOKE_sup_cpu benchmark_results/SMOKE_sup_gpu
venv/bin/python -m bench.report > /dev/null
check "cleanup: leaderboard no longer lists the smoke run" "! grep -q SMOKE_sup benchmark_results/LEADERBOARD.md"
[ $fail -eq 0 ] && echo "ALL SUPERVISOR CHECKS PASSED" || echo "SUPERVISOR CHECKS FAILED"
exit $fail
