#!/usr/bin/env bash
# SVR-1 runs, in priority order (one GPU each). Smoke-test first: bash cluster/smoke_svr1.sh SVR1_full
#   bash cluster/svr1_sweep.sh list          print the commands
#   bash cluster/svr1_sweep.sh <run name>    submit one run
#   bash cluster/svr1_sweep.sh all           submit every run that has no checkpoints/<run> yet
# All are 6-channel models (slots 0-5) benchmarked at 6ch, except the 8-channel pretraining run.
set -eu
cd "$(dirname "$0")/.."
declare -A RUNS=(
  [SVR1_full]="6ch"
  [SVR1_8ch_pre]="8ch,6ch --keep-channels 0,1,2,3,4,5,6,7 --channel-dropout 0.25"
  [SVR1_no_judge]="6ch --no-judge"
  [SVR1_no_verifier]="6ch --no-verifier"
  [SVR1_no_artic]="6ch --no-artic"
  [SVR1_no_phone]="6ch --no-phone"
  [SVR1_no_rate]="6ch --no-rate"
  [SVR1_no_fastaug]="6ch --no-speed --no-undershoot"
  [SVR1_no_hwsim]="6ch --no-hwsim"
  [SVR1_small]="6ch --model {\"d_model\":384,\"n_layers\":8,\"n_heads\":6,\"ffn_hidden\":1024}"
  [SVR1_large]="6ch --model {\"d_model\":768,\"n_layers\":16,\"n_heads\":12,\"ffn_hidden\":2048}"
)
ORDER="SVR1_full SVR1_8ch_pre SVR1_no_judge SVR1_no_verifier SVR1_no_artic SVR1_no_phone SVR1_no_rate SVR1_no_fastaug SVR1_no_hwsim SVR1_small SVR1_large"
# after SVR1_8ch_pre finishes: 6-channel fine-tune of the 8-channel model
#   bash cluster/submit.sh --svr1 SVR1_8to6 6ch --init-from checkpoints/SVR1_8ch_pre/model.pt --epochs 50 --warmup-steps 300 --judge-start 0 --judge-ramp 1000
case "${1:-list}" in
  list) for r in $ORDER; do echo "bash cluster/submit.sh --svr1 $r ${RUNS[$r]}"; done ;;
  all) for r in $ORDER; do [ -d "checkpoints/$r" ] || bash cluster/submit.sh --svr1 $r ${RUNS[$r]}; done ;;
  *) [ -n "${RUNS[$1]:-}" ] || { echo "unknown run $1"; exit 1; }; bash cluster/submit.sh --svr1 $1 ${RUNS[$1]} ;;
esac
