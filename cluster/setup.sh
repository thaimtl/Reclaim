#!/usr/bin/env bash
# One-time setup of a working copy on a cluster (Moira or Conrad, PBS). Run on a login or interactive node:
#   bash setup.sh /path/to/calhacks2026
# Pulls code, data and checkpoints from the private bucket, extracts the Berkeley dataset, builds the venv.
set -euo pipefail
WORK=${1:-$PWD/calhacks2026}
B=hf://buckets/austinkomachi/Austin_CH
mkdir -p "$WORK" && cd "$WORK"

hf sync $B/repo .
hf sync $B/data ./data
mkdir -p baseline_gaddy/checkpoints silent_speech
hf sync $B/checkpoints/gaddy baseline_gaddy/checkpoints
[ -d silent_speech/emg_data ] || tar -C silent_speech -xzf data/emg_data.tar.gz

# Python 3.12 + the exact package set of the working environment (torch 2.11, CUDA 13 build).
if command -v uv > /dev/null; then
  uv venv --python 3.12 venv
  uv pip install --python venv/bin/python -r requirements-cluster.txt
else
  python3.12 -m venv venv
  venv/bin/pip install -r requirements-cluster.txt
fi
# FlashAttention-2 for H100/H200 (SVR-1 uses it; Gaddy-architecture baselines do not need it).
{ command -v uv > /dev/null && uv pip install --python venv/bin/python flash-attn --no-build-isolation; } \
  || venv/bin/python -m pip install flash-attn --no-build-isolation \
  || echo "flash-attn not installed: SVR-1 will fall back to SDPA"

venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Camera baselines: separate environment (MediaPipe needs numpy<2), then all their weights
if command -v uv > /dev/null; then
  uv venv --python 3.11 venv-camera && uv pip install --python venv-camera/bin/python -r requirements-camera.txt
else
  python3.11 -m venv venv-camera && venv-camera/bin/pip install -r requirements-camera.txt
fi
PY=venv-camera/bin/python bash camera/download_weights.sh
echo "setup done in $WORK"
