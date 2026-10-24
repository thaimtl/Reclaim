#!/usr/bin/env bash
# Every baseline run, as the exact commands to put inside PBS jobs (one GPU each; queue/resources set on the cluster).
# Run cluster/smoke.sh with the same arguments first. Each training run benchmarks itself when it finishes.
#
# EMG baselines on Berkeley data (Gaddy's recipe: 128k micro-batch x 2 accumulation = ~50k optimizer steps):
#   G6_baseline  Gaddy's architecture on our 6 sites (1-6)            -> the model SVR-1 must beat
#   B1T_exact    Gaddy's architecture, all 8 channels, his batch size  -> our training code reproduces his 27%?
#   B1_ours 6ch  Gaddy's released 8-channel model fed only our 6 channels (reference point, no training)
#
#   bash cluster/smoke.sh G6_baseline --keep-channels 0,1,2,3,4,5 --micro-batch-samples 128000 --grad-accum 2
#   svr/supervise.sh G6_baseline 6ch --keep-channels 0,1,2,3,4,5 --micro-batch-samples 128000 --grad-accum 2
#
#   bash cluster/smoke.sh B1T_exact --micro-batch-samples 128000 --grad-accum 2
#   svr/supervise.sh B1T_exact 8ch,6ch --micro-batch-samples 128000 --grad-accum 2
#
#   venv-svr/bin/python -m bench.run_model --name B1_ours --checkpoint baseline_gaddy/checkpoints/model.pt --channels 6ch
#   venv-svr/bin/python -m bench.score B1_ours_cpu B1_ours_gpu && venv-svr/bin/python -m bench.report
#
# Camera-only baselines (need the wearer's recordings in recordings/, day 2; smoke-test now on a sample video):
#   bash third_party/lipflow/scripts/download-models.sh --samples
#   venv-camera/bin/python -m camera.smoke third_party/lipflow/samples/2016-03-12.mov
#   venv-camera/bin/python -m camera.run_camera_baselines            # Auto-AVSR, +LLM cleanup, +LLM rescore,
#                                                                    # BRAVEn, LipLearner 5-shot (fixed sentences)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-venv-svr/bin/python}; [ -x "$PY" ] || PY=venv/bin/python  # venv: cluster copies set up before venv-svr
case "${1:-}" in
  G6_baseline) svr/supervise.sh G6_baseline 6ch --keep-channels 0,1,2,3,4,5 --micro-batch-samples 128000 --grad-accum 2 ;;
  B1T_exact)   svr/supervise.sh B1T_exact 8ch,6ch --micro-batch-samples 128000 --grad-accum 2 ;;
  B1_6ch)      $PY -m bench.run_model --name B1_ours --checkpoint baseline_gaddy/checkpoints/model.pt \
                 --channels 6ch && $PY -m bench.score B1_ours_cpu B1_ours_gpu && $PY -m bench.report ;;
  camera)      venv-camera/bin/python -m camera.run_camera_baselines ;;
  *) echo "usage: cluster/baselines.sh G6_baseline | B1T_exact | B1_6ch | camera"; exit 1 ;;
esac
