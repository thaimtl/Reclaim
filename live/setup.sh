#!/usr/bin/env bash
# Set up the live camera reader: venv-live, Lipflow (pinned), model weights, the emotion model, the sample clip.
# Run once from the repo root (Linux, NVIDIA driver with CUDA 13):
#   bash live/setup.sh                                  # everything, next to the code
#   RECLAIM_WORKDIR=/path/to/workdir bash live/setup.sh  # data and venv in the working directory (README, "Code and data")
#   SKIP_VENV=1 bash live/setup.sh                      # weights only, venv already there
# third_party/ and venv-live/ go in RECLAIM_WORKDIR (default: the repo, where they are gitignored). Re-running skips
# what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
W="${RECLAIM_WORKDIR:-$PWD}"

LIPFLOW_REPO=https://github.com/amywork777/lipflow.git
LIPFLOW_COMMIT=e37453b1e1e125ca7130e4abf2635117635f6e0b   # the version live/ was built and tested against (MIT)
HF=https://huggingface.co
L="$W/third_party/lipflow/models"

get() { [ -s "$2" ] && { echo "have $2"; return; }; mkdir -p "$(dirname "$2")"; curl -fL -o "$2.part" "$1" && mv "$2.part" "$2"; }

# 1. Lipflow (face crop + LipReader + its bundled ESPnet), pinned
if [ ! -d "$W/third_party/lipflow/.git" ]; then
  git clone "$LIPFLOW_REPO" "$W/third_party/lipflow"
fi
git -C "$W/third_party/lipflow" fetch -q origin
git -C "$W/third_party/lipflow" checkout -q "$LIPFLOW_COMMIT"

# 2. Weights where Lipflow's LipReader expects them: Auto-AVSR LRS3_V_WER19.1, subword LM, MediaPipe face landmarker
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json $L/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  $L/vsr/model.pth
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  $L/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   $L/lm/model.pth
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model $L/lm/unigram5000.model
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    $L/face_landmarker.task

# 3. E5 emotion classifier (EmotiEffLib / HSEmotion enet_b0_8_va_mtl, ONNX), for live/emotion_net.py
get https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx \
    "$W/third_party/emotiefflib/enet_b0_8_va_mtl.onnx"
echo "c43e056ad388d4a8dc911832b8291435b2af537f967e5870ebd731574ec7e812  $W/third_party/emotiefflib/enet_b0_8_va_mtl.onnx" \
  | sha256sum -c -

# 4. Lipflow's sample clip (tests and --replay without a camera)
[ -s "$W/third_party/lipflow/samples/2016-03-12.mov" ] || bash "$W/third_party/lipflow/scripts/download-models.sh" --samples

# 5. venv-live: Python 3.11, torch cu130, the camera requirements (MediaPipe needs numpy < 2)
if [ -z "${SKIP_VENV:-}" ] && [ ! -x "$W/venv-live/bin/python" ]; then
  uv venv --python 3.11 "$W/venv-live"
  uv pip install --python "$W/venv-live/bin/python" torch --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$W/venv-live/bin/python" -r requirements-camera.txt pytest
fi

echo "done. check: $W/venv-live/bin/python -m pytest tests/live -q ; laptop: bash live/bench/laptop_check.sh"
