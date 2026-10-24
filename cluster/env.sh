# Keep everything this project writes on the cluster inside the working copy: temp files, every cache (pip, uv, Hugging
# Face, torch, triton, numba, matplotlib), uv's Python installs, the Hugging Face login token, tmux sockets.
# Sourced by every cluster script; by hand: source cluster/env.sh. .cache/ and tmp/ are git-ignored.
# Anything cluster-specific (modules, PATH) can go in a file outside the repo named by RECLAIM_CLUSTER_ENV; it is
# sourced first, and the settings below win.
RECLAIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RECLAIM_ROOT
if [ -n "${RECLAIM_CLUSTER_ENV:-}" ] && [ -f "$RECLAIM_CLUSTER_ENV" ]; then . "$RECLAIM_CLUSTER_ENV"; fi
mkdir -p "$RECLAIM_ROOT/.cache" "$RECLAIM_ROOT/tmp"
export TMPDIR=$RECLAIM_ROOT/tmp TMP=$RECLAIM_ROOT/tmp TEMP=$RECLAIM_ROOT/tmp TMUX_TMPDIR=$RECLAIM_ROOT/tmp
export XDG_CACHE_HOME=$RECLAIM_ROOT/.cache
export HF_HOME=$RECLAIM_ROOT/.cache/huggingface
export PIP_CACHE_DIR=$RECLAIM_ROOT/.cache/pip UV_CACHE_DIR=$RECLAIM_ROOT/.cache/uv
export UV_PYTHON_INSTALL_DIR=$RECLAIM_ROOT/.cache/uv-python
export TORCH_HOME=$RECLAIM_ROOT/.cache/torch TRITON_CACHE_DIR=$RECLAIM_ROOT/.cache/triton
export NUMBA_CACHE_DIR=$RECLAIM_ROOT/.cache/numba MPLCONFIGDIR=$RECLAIM_ROOT/.cache/matplotlib
export CUDA_CACHE_PATH=$RECLAIM_ROOT/.cache/nv
# The Hugging Face token lives in $HF_HOME too: bring in an existing login once (file mode 600, git-ignored folder).
if [ ! -f "$HF_HOME/token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  mkdir -p "$HF_HOME" && cp "$HOME/.cache/huggingface/token" "$HF_HOME/token" && chmod 600 "$HF_HOME/token"
fi
