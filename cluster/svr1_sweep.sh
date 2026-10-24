#!/usr/bin/env bash
# Pretraining runs on Berkeley's data at the rig's 4 sites (slots 0, 1, 2, 5), in priority order, one GPU each
# (docs/SVR1_SPEC.md section 4, cluster/RUNBOOK.md). Smoke-test first: bash cluster/smoke_svr1.sh SVR1_4ch
#   bash cluster/svr1_sweep.sh list          print the commands
#   bash cluster/svr1_sweep.sh <run name>    submit one run
#   bash cluster/svr1_sweep.sh all           submit every run that has no checkpoints/<run> yet
# SVR1_4ch is the main run; G4 is Gaddy's architecture and recipe on the same 4 slots and inputs (the baseline and the
# fallback). Both read the causal cache (python -m svr.precompute_svr1 cache). The rest are optional ablations.
set -eu
cd "$(dirname "$0")/.."
declare -A RUNS=(
  [SVR1_4ch]="--svr1 SVR1_4ch 4ch"
  [G4]="G4 4ch --keep-channels 0,1,2,5 --micro-batch-samples 128000 --grad-accum 2"
  [SVR1_8ch_pre]="--svr1 SVR1_8ch_pre 8ch,4ch --keep-channels 0,1,2,3,4,5,6,7 --channel-dropout 0.25"
  [SVR1_no_judge]="--svr1 SVR1_no_judge 4ch --no-judge"
  [SVR1_no_prefix]="--svr1 SVR1_no_prefix 4ch --no-prefix"
  [SVR1_no_delivery]="--svr1 SVR1_no_delivery 4ch --no-delivery"
  [SVR1_no_artic]="--svr1 SVR1_no_artic 4ch --no-artic"
)
ORDER="SVR1_4ch G4 SVR1_8ch_pre SVR1_no_judge SVR1_no_prefix SVR1_no_delivery SVR1_no_artic"
# after SVR1_8ch_pre finishes: 50 epochs on the 4 slots, then compare with SVR1_4ch
#   bash cluster/submit.sh --svr1 SVR1_8to4 4ch --init-from checkpoints/SVR1_8ch_pre/model.pt --epochs 50 \
#     --warmup-steps 300 --judge-start 0 --judge-ramp 1000 --align-start 0
case "${1:-list}" in
  list) for r in $ORDER; do echo "bash cluster/submit.sh ${RUNS[$r]}"; done ;;
  all) for r in $ORDER; do [ -d "checkpoints/$r" ] || bash cluster/submit.sh ${RUNS[$r]}; done ;;
  *) [ -n "${RUNS[$1]:-}" ] || { echo "unknown run $1"; exit 1; }; bash cluster/submit.sh ${RUNS[$1]} ;;
esac
