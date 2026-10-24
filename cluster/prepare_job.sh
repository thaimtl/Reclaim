#!/usr/bin/env bash
# PBS job, submitted by cluster/pretrain.sh: what the main SVR-1 run needs, then the run itself.
#   1. causal cache + scale statistics (svr.precompute_svr1 cache)
#   2. which attention kernel will run (FlashAttention-2 if it imports, else PyTorch SDPA)
#   3. the SVR-1 smoke test with the run's exact arguments
#   4. submit the training run (cluster/submit.sh), unless LAUNCH=0
# The main run trains without articulator targets (MAIN_ARGS, default --no-artic); cluster/sparc_job.sh makes them on
# the side and starts SVR1_4ch_artic. Log: checkpoints/_prepare/prepare.log, ending in PREPARE PASS or PREPARE FAIL.
#PBS -N Austin-Train
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=03:00:00
#PBS -j oe
set -u
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}"
. cluster/env.sh
RUN=${RUN:-SVR1_4ch}
MAIN_ARGS=${MAIN_ARGS---no-artic}
P=checkpoints/_prepare
mkdir -p $P
exec >> $P/prepare.log 2>&1
PY=venv-svr/bin/python; [ -x "$PY" ] || PY=venv/bin/python
step() { echo "== $(date '+%F %T') $*"; }
fail() { echo "PREPARE FAIL: $*"; exit 1; }
step "start on $(hostname), run $RUN $MAIN_ARGS, $(git log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step "1. causal cache + scale statistics"
$PY -m svr.precompute_svr1 cache || fail "cache"

step "2. attention kernel"
$PY -c "import flash_attn; print('FlashAttention-2', flash_attn.__version__, '(used on H100/H200)')" \
  || echo "flash_attn does not import: PyTorch SDPA will run (slower, same results)"

step "3. smoke test: bash cluster/smoke_svr1.sh $RUN $MAIN_ARGS"
bash cluster/smoke_svr1.sh "$RUN" $MAIN_ARGS || fail "smoke test (details above)"

if [ "${LAUNCH:-1}" = "1" ]; then
  step "4. submitting $RUN"
  QSUB_OPTS=$(cat checkpoints/_qsub_opts 2>/dev/null || true)
  export QSUB_OPTS
  bash cluster/submit.sh --svr1 "$RUN" 4ch $MAIN_ARGS || fail "qsub of the training job"
else
  step "4. LAUNCH=0: submit it yourself: bash cluster/submit.sh --svr1 $RUN 4ch $MAIN_ARGS"
fi
echo "PREPARE PASS"
