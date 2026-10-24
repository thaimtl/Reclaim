#!/usr/bin/env bash
# One-time setup of a working copy on a cluster (PBS). Run on a login or interactive node:
#   bash setup.sh /path/to/calhacks2026 [branch, default main]
# The working copy is a git checkout of the Reclaim repo (the code) with the data beside it in git-ignored folders:
# the Berkeley dataset, Gaddy's checkpoints and data/ from the private bucket. Builds venv-svr (and venv-camera).
# Rerunning it is safe; it also updates the code of an existing copy to the branch (data folders are untouched).
set -euo pipefail
WORK=${1:-$PWD/calhacks2026}
BRANCH=${2:-main}
REPO=https://github.com/thaimtl/Reclaim.git
B=hf://buckets/austinkomachi/Austin_CH
mkdir -p "$WORK" && cd "$WORK"

# Code. An older copy synced from the bucket has no .git: its code files are replaced by the repo's.
if [ ! -d .git ]; then git init -q && git remote add origin "$REPO"; fi
git fetch -q origin "$BRANCH"
git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
echo "code: $(git log --oneline -1)"

# Data (sync only adds and updates; never pass --delete)
hf sync $B/data ./data
mkdir -p baseline_gaddy/checkpoints silent_speech
hf sync $B/checkpoints/gaddy baseline_gaddy/checkpoints
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

# Camera baselines: separate environment (MediaPipe needs numpy<2), then all their weights
if [ ! -x venv-camera/bin/python ]; then
  if command -v uv > /dev/null; then
    uv venv --python 3.11 venv-camera && uv pip install --python venv-camera/bin/python -r requirements-camera.txt
  else
    python3.11 -m venv venv-camera && venv-camera/bin/pip install -r requirements-camera.txt
  fi
  PY=venv-camera/bin/python bash camera/download_weights.sh
fi
echo "setup done in $WORK"
