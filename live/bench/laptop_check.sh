#!/usr/bin/env bash
# Laptop smoke test for the live camera reader (live/README.md; E0/E3 in docs/EVAN_REALTIME_LIP_READING.md). Run from anywhere, paste the
# whole output back:
#   bash live/bench/laptop_check.sh                  # venv-live, /dev/video0
#   VENV=venv-camera CAM=1 SIZE=1280x720 bash live/bench/laptop_check.sh
#   SKIP_GPU=1 bash live/bench/laptop_check.sh     # no GPU load (matmul + encoder sections SKIP)
# Every check is independent: OK / FAIL / SKIP lines, never aborts. Summary at the end.
set -u
cd "$(dirname "$0")/../.." || exit 1
VENV=${VENV:-venv-live}
PY=${PY:-$VENV/bin/python}
CAM=${CAM:-0}
SIZE=${SIZE:-640x480}
SECS=${SECS:-5}
SKIP_GPU=${SKIP_GPU:-0}
DEV=/dev/video$CAM
case $CAM in /dev/*) DEV=$CAM ;; esac
RES=$(mktemp)
trap 'rm -f "$RES"' EXIT

ok()   { echo "OK   $*"; echo OK >> "$RES"; }
fail() { echo "FAIL $*"; echo FAIL >> "$RES"; }
skip() { echo "SKIP $*"; echo SKIP >> "$RES"; }
have() { command -v "$1" > /dev/null 2>&1; }
hdr()  { echo; echo "== $*"; }
# run python with a timeout; prints its output indented
pyrun() { local t=$1; shift; timeout "$t" "$PY" "$@" 2>&1 | sed 's/^/    /'; return "${PIPESTATUS[0]}"; }

echo "laptop_check $(date '+%F %T') host=$(hostname) kernel=$(uname -r) repo=$(pwd)"
[ -f /etc/os-release ] && . /etc/os-release && echo "os: ${PRETTY_NAME:-?}"
grep -qi microsoft /proc/version 2>/dev/null && echo "note: WSL2 (no webcam expected)"

hdr "1. NVIDIA driver (needs CUDA 13 for torch cu130)"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,pstate --format=csv,noheader 2>&1 | sed 's/^/    /'
  cv=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9.]+' | awk '{print $NF}')
  if [ -z "$cv" ]; then fail "nvidia-smi gave no CUDA version"
  elif [ "${cv%%.*}" -ge 13 ]; then ok "driver supports CUDA $cv"
  else fail "driver supports CUDA $cv, torch cu130 needs >= 13.0 (update the driver)"; fi
else
  fail "nvidia-smi not found (no NVIDIA driver?)"
fi

hdr "2. Python env ($PY)"
if [ -x "$PY" ]; then
  ok "$("$PY" --version 2>&1) at $PY"
else
  fail "no python at $PY (set VENV=... or PY=...)"
fi

hdr "3. torch cu130 matmul on the GPU"
if [ "$SKIP_GPU" = 1 ]; then skip "torch matmul: SKIP_GPU=1"
elif [ -x "$PY" ]; then
  if pyrun 120 - <<'EOF'
import time, torch
print(f'torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
p = torch.cuda.get_device_properties(0)
print(f'{p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB, arch list {torch.cuda.get_arch_list()}')
for dt in (torch.float32, torch.float16):
    a = torch.randn(4096, 4096, device='cuda', dtype=dt); b = torch.randn_like(a)
    (a @ b).sum().item()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize(); s = (time.perf_counter() - t) / 20
    print(f'{str(dt):14s} 4096^3 matmul {s * 1000:.2f} ms = {2 * 4096 ** 3 / s / 1e12:.1f} TFLOPS')
ref = (a.float().cpu()[:64] @ b.float().cpu()[:, :64])
err = (c.float().cpu()[:64, :64] - ref).abs().max().item()
assert err < 1.0, f'GPU result off by {err}'
EOF
  then ok "torch matmul on GPU"; else fail "torch matmul on GPU (see above)"; fi
else skip "torch matmul: no python"; fi

hdr "4. import mediapipe, cv2"
if [ -x "$PY" ]; then
  if pyrun 120 -c "import mediapipe, cv2, numpy; print('mediapipe', mediapipe.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__); print('cv2 V4L2 backend:', [l.strip() for l in cv2.getBuildInformation().splitlines() if 'v4l' in l.lower()])"
  then ok "import mediapipe, cv2"; else fail "import mediapipe, cv2"; fi
  if pyrun 60 -c "import zmq, msgpack, live.bus; print('pyzmq', zmq.__version__, 'libzmq', zmq.zmq_version(), 'msgpack', msgpack.version)"
  then ok "import zmq, msgpack (team bus)"; else fail "import zmq, msgpack (uv pip install --python \$PY pyzmq msgpack)"; fi
  if [ -f third_party/lipflow/models/face_landmarker.task ]; then
    if pyrun 120 - <<'EOF'
import time, numpy as np
import live
from lipflow.face import FaceTracker
tr = FaceTracker(live.LIPFLOW_MODELS + '/face_landmarker.task')
img = np.zeros((480, 640, 3), np.uint8)
t = time.perf_counter()
for i in range(20):
    tr.detect(img, i * 40)
print(f'FaceLandmarker VIDEO mode runs, {(time.perf_counter() - t) / 20 * 1000:.1f} ms/frame on a blank 640x480')
EOF
    then ok "MediaPipe FaceLandmarker"; else fail "MediaPipe FaceLandmarker"; fi
  else skip "FaceLandmarker: third_party/lipflow/models/face_landmarker.task missing (camera/download_weights.sh)"; fi
else skip "imports: no python"; fi

hdr "5. Webcam formats and controls ($DEV)"
if [ ! -e "$DEV" ]; then
  fail "no $DEV ($(ls /dev/video* 2>/dev/null | tr '\n' ' ' || true)) (set CAM=n)"
elif ! have v4l2-ctl; then
  skip "v4l2-ctl missing (sudo apt install v4l-utils); camera controls can't be locked without it"
else
  v4l2-ctl -d "$DEV" --info 2>&1 | sed 's/^/    /' | head -8
  v4l2-ctl -d "$DEV" --list-formats-ext 2>&1 | sed 's/^/    /'
  v4l2-ctl -d "$DEV" -l 2>&1 | sed 's/^/    /'
  if v4l2-ctl -d "$DEV" --list-formats-ext 2>/dev/null | grep -q "'MJPG'"; then ok "MJPEG offered"
  else fail "no MJPEG format listed"; fi
  ctl=$(v4l2-ctl -d "$DEV" -l 2>/dev/null)
  if echo "$ctl" | grep -qE '^\s*(auto_exposure|exposure_auto) '; then ok "manual exposure control present"
  else fail "no auto_exposure/exposure_auto control"; fi
  if echo "$ctl" | grep -qE '^\s*(focus_automatic_continuous|focus_auto) '; then ok "autofocus control present"
  else skip "no autofocus control (fixed-focus camera: fine)"; fi
fi

hdr "6. Webcam grab, MJPEG, exposure/autofocus locked, ${SECS}s per rate ($SIZE)"
if [ ! -x "$PY" ]; then skip "webcam grab: no python"
elif [ ! -e "$DEV" ]; then fail "webcam grab: no $DEV"
else
  rates="25 30"
  if have v4l2-ctl; then
    w=${SIZE%x*}; h=${SIZE#*x}
    offered=$(v4l2-ctl -d "$DEV" --list-formats-ext 2>/dev/null | awk -v sz="${w}x${h}" '
      /\x27MJPG\x27/ {m=1; next} /\x27[A-Z0-9]{4}\x27/ {m=0}
      m && /Size:/ {s=($3==sz)} m && s && /fps/ {gsub(/[()]/,"",$4); print int($4+0.5)}' | sort -un | tr '\n' ' ')
    echo "    MJPG $SIZE offers: ${offered:-?}"
    for r in 50 60; do echo " $offered " | grep -q " $r " && rates="$rates $r"; done
  fi
  for r in $rates 0; do
    label=$r; [ "$r" = 0 ] && label="auto (what CameraSource picks)"
    out=$(timeout $((SECS + 20)) "$PY" -m live.source --probe --device "$CAM" --fps "$r" --seconds "$SECS" --size "$SIZE" 2>&1)
    echo "$out" | sed 's/^/    /'
    fps=$(echo "$out" | grep '25fps out' | grep -o "'fps': [0-9.]*" | awk '{print $2}')
    p95=$(echo "$out" | grep '25fps out' | grep -o "'p95_ms': [0-9.]*" | awk '{print $2}')
    if [ -n "$fps" ] && awk -v f="$fps" 'BEGIN{exit !(f >= 24 && f <= 26.5)}'; then
      ok "grab @ $label -> $fps fps out, p95 interval $p95 ms"
    else
      fail "grab @ $label -> ${fps:-no} fps out (want ~25)"
    fi
  done
fi

hdr "7. Which GPU drives the display"
if have prime-select; then echo "    prime-select: $(prime-select query 2>&1)"; fi
if have glxinfo; then glxinfo -B 2>/dev/null | grep -E 'OpenGL renderer|OpenGL vendor' | sed 's/^/    /'; fi
for c in /sys/class/drm/card*/device; do
  [ -e "$c/vendor" ] && echo "    $(basename "$(dirname "$c")") vendor $(cat "$c/vendor") boot_vga $(cat "$c/boot_vga" 2>/dev/null || echo ?)"
done
if have nvidia-smi; then
  procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  gfx=$(nvidia-smi 2>/dev/null | awk '/Processes:/,0' | grep -E ' G ' || true)
  echo "    compute procs: ${procs:-none}"
  echo "    graphics procs on NVIDIA: $(echo "$gfx" | grep -c . ) ${gfx:+(listed below)}"
  [ -n "$gfx" ] && echo "$gfx" | sed 's/^/      /'
  disp=$(nvidia-smi --query-gpu=display_active --format=csv,noheader 2>/dev/null)
  echo "    nvidia display_active: $disp"
  if echo "$disp" | grep -qiE '^(Enabled|Yes)'; then skip "display on the NVIDIA GPU (fine, but it shares the 8 GB and SMs)"
  else ok "display not on the NVIDIA GPU (or unknown: $disp)"; fi
else skip "display GPU: no nvidia-smi"; fi

hdr "8. Power"
for a in /sys/class/power_supply/A*/online /sys/class/power_supply/AC*/online; do
  [ -e "$a" ] && echo "    $(dirname "$a" | xargs basename) online=$(cat "$a")" && break
done
if have powerprofilesctl; then
  pp=$(powerprofilesctl get 2>&1); echo "    powerprofilesctl: $pp"
  if [ "$pp" = performance ]; then ok "power profile performance"
  else fail "power profile '$pp' (powerprofilesctl set performance)"; fi
else skip "powerprofilesctl missing"; fi
if have nvidia-smi; then
  nvidia-smi --query-gpu=power.draw,power.limit,power.max_limit,clocks.sm,clocks.max.sm,temperature.gpu \
    --format=csv 2>&1 | sed 's/^/    /'
  ok "nvidia power readout"
else skip "nvidia power: no nvidia-smi"; fi

hdr "9. groups (evdev endpoint key needs 'input')"
echo "    $(id -nG)"
if id -nG | tr ' ' '\n' | grep -qx input; then ok "user in group input"
else fail "not in group input (sudo usermod -aG input \$USER, then log out/in)"; fi
ev=$(ls /dev/input/event* 2>/dev/null | head -1)
if [ -n "$ev" ]; then
  if [ -r "$ev" ]; then ok "$ev readable"; else fail "$ev not readable"; fi
else skip "no /dev/input/event*"; fi

hdr "10. Encoder ms, 250-frame dummy input (fp32 / fp16, warm-up then median of 10)"
if [ "$SKIP_GPU" = 1 ]; then skip "encoder: SKIP_GPU=1"
elif [ ! -x "$PY" ]; then skip "encoder: no python"
elif [ ! -f third_party/lipflow/models/vsr/model.json ]; then skip "encoder: third_party/lipflow/models/vsr/model.json missing"
else
  if pyrun 600 - <<'EOF'
import argparse, json, os, statistics, time
import torch
import live
assert torch.cuda.is_available(), 'no CUDA'
T = 250
enc, how = None, ''
try:
    import live.model as lm
    if hasattr(lm, 'bench_encoder'):  # lane A's own benchmark hook, if it has one
        for k, v in lm.bench_encoder(T).items():
            print(f'live.model.bench_encoder {k}: {v}')
        raise SystemExit(0)
except ImportError:
    pass
w = os.path.join(live.LIPFLOW_MODELS, 'vsr', 'model.pth')
if os.path.exists(w):
    from lipflow.vsr import LipReader
    enc, how = LipReader(device='cuda', use_lm=False, personal=False).model.encoder, 'LipReader encoder, real weights'
else:  # same architecture, random weights: same speed
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    cfg = json.load(open(os.path.join(live.LIPFLOW_MODELS, 'vsr', 'model.json')))
    args = argparse.Namespace(**(cfg if isinstance(cfg, dict) else cfg[2]))
    enc, how = E2E(5049, args).encoder, 'E2E encoder from model.json, RANDOM weights (model.pth missing)'
enc = enc.cuda().eval()
print(how, f'{sum(p.numel() for p in enc.parameters()) / 1e6:.1f} M params')
x = torch.randn(1, 1, T, 88, 88, device='cuda')
for name, dt in (('fp32', torch.float32), ('fp16', torch.float16)):
    e, xi = enc.to(dt), x.to(dt)
    torch.cuda.reset_peak_memory_stats()
    ms = []
    with torch.inference_mode():
        for i in range(13):
            torch.cuda.synchronize(); t = time.perf_counter()
            out, _ = e(xi, None)
            torch.cuda.synchronize()
            if i >= 3:
                ms.append((time.perf_counter() - t) * 1000)
    print(f'{name}: median {statistics.median(ms):.1f} ms, min {min(ms):.1f}, max {max(ms):.1f}, '
          f'out {tuple(out.shape)}, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
EOF
  then ok "encoder benchmark"; else fail "encoder benchmark (see above)"; fi
fi

echo
echo "== summary: $(grep -c '^OK' "$RES") OK, $(grep -c '^FAIL' "$RES") FAIL, $(grep -c '^SKIP' "$RES") SKIP"
