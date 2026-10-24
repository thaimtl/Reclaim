#!/usr/bin/env bash
# PBS wrapper for cluster/smoke.sh on one GPU (the login nodes have none).
#   qsub -v ARGS="<run name> [svr.train args...]" cluster/smoke_job.sh
#   qsub -v SMOKE=smoke_svr1,ARGS="SVR1_full" cluster/smoke_job.sh        (SVR-1; args with commas: quote as in train_job.sh)
# Generous memory on purpose: the smoke test measures peak RAM, which then sizes train_job.sh.
#PBS -N svr_smoke
#PBS -q development
#PBS -l select=1:ncpus=8:mem=96gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=01:00:00
#PBS -j oe
set -u
# Cluster-specific setup (caches, tmp, PATH) lives outside the repo: export RECLAIM_CLUSTER_ENV=/path/to/env.sh
[ -n "${RECLAIM_CLUSTER_ENV:-}" ] && . "$RECLAIM_CLUSTER_ENV"
cd "${REPO:-${PBS_O_WORKDIR:-$(dirname "$0")/..}}"
echo "host $(hostname)  job ${PBS_JOBID:-none}  args: $ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash cluster/${SMOKE:-smoke}.sh $ARGS
