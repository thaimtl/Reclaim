#!/usr/bin/env bash
# PBS job, submitted by cluster/pretrain.sh, on its own GPU: articulator targets, then a second run that uses them.
#   1. SPARC on every voiced Berkeley recording (venv-sparc; resumable: rerunning skips what is done)
#   2. wait for the causal cache (built by cluster/prepare_job.sh), then align the targets to it
#   3. the SVR-1 smoke test for the second run   4. submit it (RUN, default SVR1_4ch_artic: SVR1_4ch + articulators)
# It never blocks the main run. On failure it writes checkpoints/_skipped_<run> with the reason (the bucket watcher
# then stops waiting for that run). Log: checkpoints/_sparc/sparc.log, ending in SPARC PASS or SPARC FAIL.
#PBS -N Austin-Train
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=08:00:00
#PBS -j oe
set -u
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}"
. cluster/env.sh
export HF_HUB_OFFLINE=1  # the SPARC weights were downloaded on the login node; compute nodes may have no internet
RUN=${RUN:-SVR1_4ch_artic}
P=checkpoints/_sparc
mkdir -p $P
exec >> $P/sparc.log 2>&1
PY=venv-svr/bin/python; [ -x "$PY" ] || PY=venv/bin/python
step() { echo "== $(date '+%F %T') $*"; }
fail() { echo "SPARC FAIL: $*"; echo "$*" > "checkpoints/_skipped_$RUN"; exit 1; }
rm -f "checkpoints/_skipped_$RUN"
step "start on $(hostname), run $RUN"

step "1. SPARC articulator targets"
venv-sparc/bin/python -m svr.precompute_svr1 sparc --device cuda || fail "SPARC (see above)"
echo "SPARC files: $(find data/svr1_targets/sparc -name '*.npy' | wc -l)"

step "2. waiting for the causal cache, then align"
for _ in $(seq 240); do [ -f data/svr1_targets/scale.json ] && break; sleep 60; done
[ -f data/svr1_targets/scale.json ] || fail "the causal cache never appeared (cluster/prepare_job.sh)"
$PY -m svr.precompute_svr1 align || fail "align"

step "3. smoke test: bash cluster/smoke_svr1.sh $RUN"
bash cluster/smoke_svr1.sh "$RUN" || fail "smoke test (details above)"

step "4. submitting $RUN"
QSUB_OPTS=$(cat checkpoints/_qsub_opts 2>/dev/null || true)
export QSUB_OPTS
bash cluster/submit.sh --svr1 "$RUN" 4ch || fail "qsub of the training job"
echo "SPARC PASS"
