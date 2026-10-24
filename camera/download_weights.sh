#!/usr/bin/env bash
# Download every camera-baseline weight file. Run once, from the repo root, inside venv-camera:
#   bash camera/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-venv-camera/bin/python}

# Auto-AVSR visual model + subword LM + MediaPipe face landmarker, where Lipflow's LipReader expects them
L=third_party/lipflow/models
get() { [ -s "$2" ] && { echo "have $2"; return; }; mkdir -p "$(dirname "$2")"; curl -fL -o "$2.part" "$1" && mv "$2.part" "$2"; }
HF=https://huggingface.co
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json $L/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  $L/vsr/model.pth
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  $L/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   $L/lm/model.pth
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model $L/lm/unigram5000.model
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    $L/face_landmarker.task

# BRAVEn (Google Drive): Large, self-trained, LRS3+Vox2+AVS, high-resource (LRS3 VSR 20.9%; 20.1% with LM) + its LM
W=data/camera_weights
mkdir -p $W
gd() { [ -s "$2" ] && { echo "have $2"; return; }; $PY -m gdown "$1" -O "$2"; }
gd 1bU-bFxEiXNXNoOLKaAz6_9j7XuE7f_6r $W/braven_vsr_large_lrs3vox2avs_self.pth
gd 1mTeynSf6Sryh_mnVabpw-UwTRrh-mDc1 $W/braven_lm.download
# The LM link may be an archive: unpack it and keep the model file as braven_lm.pth
if [ ! -s $W/braven_lm.pth ]; then
  # A torch.save checkpoint is itself a zip (archive/data.pkl): it is the model, not an archive to unpack.
  # (no grep -q: it exits early, unzip gets SIGPIPE, and pipefail makes the test false)
  if unzip -Z1 $W/braven_lm.download 2>/dev/null | grep '/data.pkl$' > /dev/null; then cp $W/braven_lm.download $W/braven_lm.pth; fi
fi
if [ ! -s $W/braven_lm.pth ]; then
  case "$(file -b $W/braven_lm.download)" in
    *Zip*) (cd $W && unzip -o braven_lm.download && find . -name "rnnlm.model.best" -exec cp {} braven_lm.pth \;) ;;
    *gzip*|*tar*) (cd $W && tar xf braven_lm.download && find . -name "rnnlm.model.best" -exec cp {} braven_lm.pth \;) ;;
    *) cp $W/braven_lm.download $W/braven_lm.pth ;;
  esac
fi

# LipLearner contrastive encoder (Google Drive)
gd 1RK7no1QH8QFZzjVrmg64X8wuCLRfJzu3 $W/LipLearner_pretrained_model.pt

# Local LLM for cleanup and rescoring
$PY - <<'EOF'
from huggingface_hub import snapshot_download
from camera.llm import DEFAULT_MODEL
print(snapshot_download(DEFAULT_MODEL))
EOF
echo "camera weights ready"
