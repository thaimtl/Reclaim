#!/usr/bin/env bash
# PBS wrapper for an evaluation-only entry of cluster/baselines.sh on one GPU (bench.run_model runs CPU + GPU passes).
#   qsub -v WHAT=B1_6ch cluster/eval_job.sh
#PBS -N svr_eval
#PBS -q development
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=02:00:00
#PBS -j oe
set -u
# Cluster-specific setup (caches, tmp, PATH) lives outside the repo: export RECLAIM_CLUSTER_ENV=/path/to/env.sh
[ -n "${RECLAIM_CLUSTER_ENV:-}" ] && . "$RECLAIM_CLUSTER_ENV"
cd "${REPO:-${PBS_O_WORKDIR:-$(dirname "$0")/..}}"
echo "host $(hostname)  job ${PBS_JOBID:-none}  what: $WHAT"
bash cluster/baselines.sh "$WHAT"
