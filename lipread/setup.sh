#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
W="${RECLAIM_WORKDIR:-$PWD}"

LIPFLOW_REPO=https://github.com/amywork777/lipflow.git
LIPFLOW_COMMIT=e37453b1e1e125ca7130e4abf2635117635f6e0b
VSR=https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/c0b413f56b164565865a63f00774ee55a9738cc5
LM=https://huggingface.co/Amanvir/lm_en_subword/resolve/f4e39b6d0904b2a900c53a2fff0377796c797bdc
SPM=https://raw.githubusercontent.com/mpc001/auto_avsr/182b62837773ab01052d4ac21ef1d2203ea7d267/spm/unigram
# facial expression network for the emotion tag (lipread/expression.py): HSEmotion, Apache-2.0
HSE=https://github.com/HSE-asavchenko/face-emotion-recognition/raw/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/onnx
HSE_SHA256=fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898
TORCH=torch==2.11.0
L="$W/third_party/lipflow/models"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PY="$W/venv-live/Scripts/python.exe" ;;
  *) PY="$W/venv-live/bin/python" ;;
esac

get() { [ -s "$2" ] && { echo "have $2"; return; }; mkdir -p "$(dirname "$2")"; curl -fL -o "$2.part" "$1" && mv "$2.part" "$2"; }

if [ ! -d "$W/third_party/lipflow/.git" ]; then
  git clone "$LIPFLOW_REPO" "$W/third_party/lipflow"
fi
git -C "$W/third_party/lipflow" fetch -q origin
git -C "$W/third_party/lipflow" checkout -q "$LIPFLOW_COMMIT"

get $VSR/model.json $L/vsr/model.json
get $VSR/model.pth  $L/vsr/model.pth
get $LM/model.json  $L/lm/model.json
get $LM/model.pth   $L/lm/model.pth
get $SPM/unigram5000.model $L/lm/unigram5000.model
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    $L/face_landmarker.task
# optional: without it the emotion tag falls back to the blendshape rules, so a failure here only warns
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi   # macOS has only shasum
if ! { get $HSE/enet_b0_8_best_vgaf.onnx $L/hsemotion/enet_b0_8_best_vgaf.onnx &&
       echo "$HSE_SHA256  $L/hsemotion/enet_b0_8_best_vgaf.onnx" | $SHA -c - ; }; then
  rm -f "$L/hsemotion/enet_b0_8_best_vgaf.onnx"
  echo "WARNING: expression model download or checksum failed: emotion will use the blendshape rules" >&2
fi

[ -s "$W/third_party/lipflow/samples/2016-03-12.mov" ] || bash "$W/third_party/lipflow/scripts/download-models.sh" --samples

if [ -z "${SKIP_VENV:-}" ] && [ ! -x "$PY" ]; then
  uv venv --python 3.11 "$W/venv-live"
  uv pip install --python "$PY" "$TORCH" --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$PY" -r requirements-camera.txt pytest
fi

echo "done. check: $PY -m pytest tests/lipread tests/bus -q ; on the laptop: $PY -m lipread.bench.laptop_check"
