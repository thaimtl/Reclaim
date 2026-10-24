#!/usr/bin/env bash
# Set up or update the working copy on a cluster (PBS), from the private bucket only (no git needed):
#   hf sync hf://buckets/austinkomachi/Austin_CH/repo <working copy>    # the code first: this script is part of it
#   bash <working copy>/cluster/setup.sh
# Code: the bucket's repo/ (a snapshot of the Reclaim repo uploaded from Austin's machine). Data: data/ (the Berkeley
# tarball, the causal cache + scale statistics, the lexicon) and Gaddy's checkpoints. Builds venv-svr if missing (venv-camera
# only with WITH_CAMERA=1). Rerunning is safe: sync only adds and updates, never deletes.
# Never sync the bucket's calhacks2026/ folder here: it is a full snapshot of Austin's machine, environments included.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=$PWD
B=hf://buckets/austinkomachi/Austin_CH
. cluster/env.sh  # from here on, caches, tmp and the HF token stay inside the working copy
HF=$(command -v hf || true); [ -n "$HF" ] || HF=venv-svr/bin/hf
"$HF" sync $B/repo .
echo "code: $(cat cluster/VERSION 2>/dev/null || echo 'snapshot from the bucket')"

# Data (sync only adds and updates; never pass --delete)
"$HF" sync $B/data ./data
mkdir -p baseline_gaddy/checkpoints silent_speech
"$HF" sync $B/checkpoints/gaddy baseline_gaddy/checkpoints
[ -d silent_speech/emg_data ] || tar -C silent_speech -xzf data/emg_data.tar.gz

# Python 3.12 + the exact package set of the working environment (torch 2.11, CUDA 13 build).
# A copy set up before the rename keeps its old venv/ (the scripts fall back to it).
if [ ! -x venv-svr/bin/python ] && [ ! -x venv/bin/python ]; then
  if command -v uv > /dev/null; then
    uv venv --python 3.12 venv-svr
    uv pip install --python venv-svr/bin/python -r requirements-cluster.txt
  else
    python3.12 -m venv venv-svr
    venv-svr/bin/pip install -r requirements-cluster.txt
  fi
  # FlashAttention-2 for H100/H200 (SVR-1 uses it; Gaddy-architecture baselines do not need it).
  { command -v uv > /dev/null && uv pip install --python venv-svr/bin/python flash-attn --no-build-isolation; } \
    || venv-svr/bin/python -m pip install flash-attn --no-build-isolation \
    || echo "flash-attn not installed: SVR-1 will fall back to SDPA"
fi
PY=venv-svr/bin/python; [ -x "$PY" ] || PY=venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Camera baselines (optional, WITH_CAMERA=1): separate environment (MediaPipe needs numpy<2), then their weights.
# Skipped by default: SVR-1 doesn't need them, and some weights come from Google Drive, which can fail (404, quota).
if [ "${WITH_CAMERA:-0}" = "1" ] && [ ! -x venv-camera/bin/python ]; then
  if command -v uv > /dev/null; then
    uv venv --python 3.11 venv-camera && uv pip install --python venv-camera/bin/python -r requirements-camera.txt
  else
    python3.11 -m venv venv-camera && venv-camera/bin/pip install -r requirements-camera.txt
  fi
  PY=venv-camera/bin/python bash camera/download_weights.sh
fi
echo "setup done in $WORK"
