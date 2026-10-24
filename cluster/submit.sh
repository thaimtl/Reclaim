#!/usr/bin/env bash
# Submit one supervised training run. Arguments travel in a file because qsub -v splits values on commas
# (--keep-channels 0,1,2,3,4,5 and "8ch,6ch" would break).
#   bash cluster/submit.sh <run name> <benchmark channels> [svr.train args...]            Gaddy-architecture baseline
#   bash cluster/submit.sh --svr1 <run name> <benchmark channels> [svr.train_svr1 args...]  SVR-1
# Extra qsub options (queue, account, resources) go in QSUB_OPTS, e.g. QSUB_OPTS="-q gpu -A myproj".
# Every job of this project is named Austin-Train (Austin's rule); tell them apart by cluster/jobs/<run>.pbs.out and
# qstat -f (Variable_List shows ARGS_FILE / RUN).
set -eu
cd "$(dirname "$0")/.."
TRAINER=svr.train
if [ "$1" = "--svr1" ]; then TRAINER=svr.train_svr1; shift; fi
mkdir -p cluster/jobs
F=$PWD/cluster/jobs/$1.args
printf '%s\n' "$*" > "$F"
printf '%s\n' "${QSUB_OPTS:-}" > "$F.qsub"  # reused when the job resubmits itself
echo "$TRAINER $(cat "$F")"
qsub ${QSUB_OPTS:-} -N Austin-Train -o "$PWD/cluster/jobs/$1.pbs.out" -v TRAINER="$TRAINER",ARGS_FILE="$F" \
  "$PWD/cluster/train_job.sh"
