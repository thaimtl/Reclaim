#!/usr/bin/env bash
# PBS wrapper for the camera-chain smoke test (camera.smoke on Lipflow's sample video) on one GPU.
#   qsub cluster/camera_smoke_job.sh
#PBS -N Austin-Train
#PBS -q development
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:vntype=gpu
#PBS -l place=free
#PBS -l walltime=01:00:00
#PBS -j oe
set -u
# Cluster-specific setup (caches, tmp, PATH) lives outside the repo: export RECLAIM_CLUSTER_ENV=/path/to/env.sh
[ -n "${RECLAIM_CLUSTER_ENV:-}" ] && . "$RECLAIM_CLUSTER_ENV"
export HF_HUB_OFFLINE=1  # weights were fetched on the login node; never download from a compute node
cd "${REPO:-${PBS_O_WORKDIR:-$(dirname "$0")/..}}"
echo "host $(hostname)  job ${PBS_JOBID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$CH_ROOT/venv-camera/bin/python" -m camera.smoke third_party/lipflow/samples/2016-03-12.mov
