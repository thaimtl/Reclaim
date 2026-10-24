#!/usr/bin/env bash
# PBS template for one training run on one GPU (Conrad / Moira). Two ways to submit:
#   bash cluster/submit.sh [--svr1] <run name> <benchmark channels> [trainer args...]     (arguments via a file)
#   qsub -v "ARGS='<run name> <benchmark channels> [svr.train args...]'" cluster/train_job.sh   (quoted: -v splits on commas)
#PBS -N Austin-Train
#PBS -q development
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=12:00:00
#PBS -j oe
# Job listings: pass -o <dir> through QSUB_OPTS (cluster/submit.sh) to keep them out of $HOME.
# Resource numbers are starting points: replace them with ~1.5x what the smoke test measured.
# Baselines (G6_baseline, B1T_exact, 128k x 2): smoke peak RAM 4.8 GB; worst-batch VRAM 8.2 GB (Moira H200).
# SVR-1 holds more in RAM (all of Berkeley's inputs, phonemes, articulator targets, two look-alike lexicons): 32gb.
# Override with QSUB_OPTS="-l select=1:ncpus=8:mem=<n>gb:ngpus=1:vntype=gpu" (cluster/submit.sh).
#
# At the wall time PBS sends SIGTERM (then SIGKILL after kill_delay). The trap asks the trainer to checkpoint and
# exit 3; if the run is not finished, the job resubmits itself and the next job resumes mid-epoch.
set -u
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}"
. cluster/env.sh  # caches, tmp and the HF token inside the working copy (+ RECLAIM_CLUSTER_ENV)
if [ -n "${ARGS_FILE:-}" ]; then ARGS=$(cat "$ARGS_FILE"); fi
set -- $ARGS
NAME=$1
export TRAINER=${TRAINER:-svr.train}
mkdir -p "checkpoints/$NAME"
trap 'echo "wall time: asking the trainer to checkpoint"; pkill -TERM -f "svr\.train[_a-z0-9]* --name $NAME "' TERM

SKIP_BENCHMARK=${SKIP_BENCHMARK:-0} svr/supervise.sh "$@" >> "checkpoints/$NAME/supervisor.log" 2>&1 &
child=$!
wait $child; code=$?
if [ $code -gt 128 ]; then wait $child; code=$?; fi  # the trap interrupted wait; collect the real exit code

if [ -f "checkpoints/$NAME/DONE" ]; then  # finished: weights, logs and benchmarks to the bucket (best effort)
  bash cluster/sync_results.sh "$NAME" >> "checkpoints/$NAME/sync.log" 2>&1 || true
fi
if [ ! -f "checkpoints/$NAME/DONE" ] && { [ $code -eq 3 ] || [ $code -gt 128 ]; }; then
  echo "run $NAME not finished (exit $code): resubmitting to resume from the last checkpoint"
  if [ -n "${ARGS_FILE:-}" ]; then
    qsub $(cat "$ARGS_FILE.qsub" 2>/dev/null) -N Austin-Train -o "$PWD/cluster/jobs/$NAME.pbs.out" \
      -v TRAINER="$TRAINER",ARGS_FILE="$ARGS_FILE" \
      "$PWD/cluster/train_job.sh"
  else
    qsub -N Austin-Train -v "TRAINER=$TRAINER,ARGS='$ARGS'" "$PWD/cluster/train_job.sh"
  fi
fi
exit $code
