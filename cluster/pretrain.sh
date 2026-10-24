#!/usr/bin/env bash
# Tonight's SVR-1 pretraining in one command, on the cluster's login node, from the working copy (cluster/RUNBOOK.md):
#   export QSUB_OPTS="-q <queue> [-A <account>]"   # only how this cluster takes jobs; resources are in the job scripts
#   export FLASH_ATTN_WHEEL=/path/to/flash_attn-...whl   # optional: a prebuilt FlashAttention-2 wheel for venv-svr
#   bash cluster/pretrain.sh
# 1. here: checks; FlashAttention-2 into venv-svr from the wheel; the SPARC environment + its weights (needs internet)
# 2. submits cluster/prepare_job.sh (1 GPU): cache, smoke test, then the main run SVR1_4ch (no articulator targets)
# 3. submits cluster/sparc_job.sh (1 GPU, on the side): articulator targets, then SVR1_4ch_artic (skip: SKIP_SPARC=1)
# 4. starts the bucket watcher here (tmux session reclaim_sync, or nohup): both runs' weights, logs and benchmarks go to
#    the bucket every 30 min and once more when each run is done
# Follow: checkpoints/_prepare/prepare.log, checkpoints/_sparc/sparc.log, python -m svr.status SVR1_4ch,
# checkpoints/_sync_watch.log
set -euo pipefail
cd "$(dirname "$0")/.."
. cluster/env.sh
RUN=${RUN:-SVR1_4ch}
ARTIC_RUN=${ARTIC_RUN:-${RUN}_artic}
PY=venv-svr/bin/python; [ -x "$PY" ] || PY=venv/bin/python
HF=$(command -v hf || true); [ -n "$HF" ] || HF=$(dirname "$PY")/hf
pip_install() {  # into venv-svr, without touching its torch
  if command -v uv > /dev/null; then uv pip install -q --python "$PY" "$@"; else "$PY" -m pip install -q "$@"; fi
}
command -v qsub > /dev/null || { echo "no qsub here: run this on the cluster's login node"; exit 1; }
[ -x "$PY" ] || { echo "no Python environment: run cluster/setup.sh first"; exit 1; }
[ -d silent_speech/emg_data ] || { echo "no Berkeley data (silent_speech/emg_data): run cluster/setup.sh first"; exit 1; }
if ! "$HF" auth whoami > /dev/null 2>&1; then  # training matters more: start anyway, sync once someone logs in
  echo "WARNING: not logged in to Hugging Face under $HF_HOME: training starts, but nothing reaches the bucket until"
  echo "         hf auth login (then: bash cluster/sync_results.sh $RUN ${RUN}_artic)"
fi
for r in "$RUN" "$ARTIC_RUN"; do
  if [ -d "checkpoints/$r" ]; then
    echo "checkpoints/$r exists: it was started before. Resume it (cluster/RUNBOOK.md, 'If something fails'),"
    echo "or start fresh under other names: RUN=<name> bash cluster/pretrain.sh"
    exit 1
  fi
done
mkdir -p checkpoints cluster/jobs
printf '%s\n' "${QSUB_OPTS:-}" > checkpoints/_qsub_opts

echo "== FlashAttention-2"
if ! "$PY" -c "import flash_attn" 2> /dev/null && [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
  pip_install --no-deps "$FLASH_ATTN_WHEEL" && pip_install einops || echo "wheel install failed"
fi
"$PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2> /dev/null \
  || echo "flash_attn does not import in $PY: PyTorch SDPA will run (slower, same results)"

SPARC=0
if [ "${SKIP_SPARC:-0}" != "1" ]; then
  echo "== SPARC environment and weights (a few minutes; downloaded here since compute nodes may have no internet)"
  if [ ! -x venv-sparc/bin/python ]; then
    if command -v uv > /dev/null; then
      uv venv -q --python 3.12 venv-sparc && uv pip install -q --python venv-sparc/bin/python speech-articulatory-coding soundfile || true
    else
      python3 -m venv venv-sparc && venv-sparc/bin/pip install -q speech-articulatory-coding soundfile || true
    fi
  fi
  if venv-sparc/bin/python -c "from sparc import load_model; load_model('en', device='cpu')" > checkpoints/_sparc_check.log 2>&1; then
    SPARC=1
  else
    echo "SPARC not usable (checkpoints/_sparc_check.log): no $ARTIC_RUN tonight"
    echo "SPARC environment failed: see checkpoints/_sparc_check.log" > "checkpoints/_skipped_$ARTIC_RUN"
  fi
else
  echo "SKIP_SPARC=1" > "checkpoints/_skipped_$ARTIC_RUN"
fi

echo "== submitting the preparation job: cache, smoke test, then $RUN"
mkdir -p checkpoints/_prepare checkpoints/_sparc
qsub ${QSUB_OPTS:-} -N Austin-Train -o "$PWD/checkpoints/_prepare/pbs.out" -v RUN="$RUN" "$PWD/cluster/prepare_job.sh"
if [ $SPARC = 1 ]; then
  echo "== submitting the SPARC job: articulator targets, then $ARTIC_RUN"
  qsub ${QSUB_OPTS:-} -N Austin-Train -o "$PWD/checkpoints/_sparc/pbs.out" -v RUN="$ARTIC_RUN" "$PWD/cluster/sparc_job.sh"
fi

echo "== bucket watcher"
W="bash cluster/sync_results.sh --watch $RUN $ARTIC_RUN >> checkpoints/_sync_watch.log 2>&1"
if command -v tmux > /dev/null; then
  tmux kill-session -t reclaim_sync 2> /dev/null || true
  tmux new-session -d -s reclaim_sync "$W" && echo "tmux session reclaim_sync"
else
  setsid nohup bash -c "$W" < /dev/null > /dev/null 2>&1 &
  echo "nohup watcher, pid $!"
fi
echo "done. Follow: tail -f checkpoints/_prepare/prepare.log   then   $PY -m svr.status $RUN"
