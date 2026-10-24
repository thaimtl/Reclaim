"""Checks Austin's laptop can run the lip reader: driver, CUDA, imports, webcam, encoder speed. Windows and Linux.

    venv-live\\Scripts\\python -m lipread.bench.laptop_check            # Windows
    venv-live/bin/python -m lipread.bench.laptop_check                 # Linux
    ... --cam 1 --seconds 5 --skip-gpu

Prints OK / FAIL / SKIP per check and a summary; exits 1 if anything failed.
"""
import argparse
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import lipread

RESULTS = []


def report(status, text):
    RESULTS.append(status)
    print(f'{status:4s} {text}', flush=True)


def check(name, fn, *a):
    print(f'\n== {name}', flush=True)
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        report('FAIL', f'{name}: {type(e).__name__}: {e}')


def driver():
    if not shutil.which('nvidia-smi'):
        return report('FAIL', 'nvidia-smi not found (no NVIDIA driver?)')
    print('    ' + subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                   '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
    m = re.search(r'CUDA (?:UMD )?Version: ([\d.]+)', subprocess.run(['nvidia-smi'], capture_output=True,
                                                                     text=True).stdout)
    if not m:
        return report('FAIL', 'nvidia-smi gave no CUDA version')
    major = int(m.group(1).split('.')[0])
    report('OK' if major >= 13 else 'FAIL', f'driver supports CUDA {m.group(1)}'
           + ('' if major >= 13 else ', torch cu130 needs >= 13.0: update the NVIDIA driver'))


def gpu_matmul():
    import torch
    print(f'    torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        return report('FAIL', 'torch.cuda.is_available() is False')
    p = torch.cuda.get_device_properties(0)
    print(f'    {p.name} sm_{p.major}{p.minor} {p.total_memory / 2 ** 30:.1f} GiB, arch list {torch.cuda.get_arch_list()}')
    for dt in (torch.float32, torch.float16):
        a = torch.randn(4096, 4096, device='cuda', dtype=dt)
        b = torch.randn_like(a)
        (a @ b).sum().item()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            c = a @ b
        torch.cuda.synchronize()
        s = (time.perf_counter() - t) / 20
        print(f'    {str(dt):14s} 4096^3 matmul {s * 1000:.2f} ms = {2 * 4096 ** 3 / s / 1e12:.1f} TFLOPS')
    err = (c.float().cpu()[:64, :64] - a.float().cpu()[:64] @ b.float().cpu()[:, :64]).abs().max().item()
    report('OK' if err < 1.0 else 'FAIL', f'torch matmul on the GPU (max error {err:.3f})')


def imports():
    import cv2
    import mediapipe
    import msgpack
    import numpy
    import zmq
    print(f'    mediapipe {mediapipe.__version__} cv2 {cv2.__version__} numpy {numpy.__version__} '
          f'pyzmq {zmq.__version__} msgpack {msgpack.version}')
    report('OK', 'imports')
    path = os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task')
    if not os.path.exists(path):
        return report('SKIP', f'FaceLandmarker: {path} missing (bash lipread/setup.sh)')
    import numpy as np
    from lipflow.face import FaceTracker
    tr = FaceTracker(path)
    img = np.zeros((720, 1280, 3), np.uint8)
    t = time.perf_counter()
    for i in range(20):
        tr.detect(img, i * 40)
    report('OK', f'FaceLandmarker runs, {(time.perf_counter() - t) / 20 * 1000:.1f} ms/frame on a blank 1280x720')


def webcam(cam, seconds):
    from lipread.source import CameraSource, interval_stats
    src = CameraSource(cam)
    print(f'    driver {src.cam_fps:g} fps, {src.fourcc} {src.size[0]}x{src.size[1]}, locked {src.locked}')
    t_end, got = time.perf_counter() + 1.5 + seconds, []
    for f in src.frames():
        if f.t > t_end:
            break
        if f.t > t_end - seconds:
            got.append(f.t)
    src.close()
    st = interval_stats(got)
    print(f'    measured {src.measured_fps or 0:.1f} fps from the camera, {st} out')
    report('OK' if src.size == (1920, 1080) and src.fourcc == 'MJPG' else 'FAIL',
           f'1920x1080 MJPG (got {src.size[0]}x{src.size[1]} {src.fourcc})')
    report('OK' if (src.measured_fps or 0) >= 28 else 'FAIL',
           f'camera delivers {src.measured_fps or 0:.1f} fps (want 30: more light, or lock exposure with '
           f'--camera-settings)')
    report('OK' if 24 <= st['fps'] <= 26.5 else 'FAIL', f'{st["fps"]} fps out to the reader (want 25)')


def on_battery():
    """True / False / None (unknown)."""
    if sys.platform == 'win32':
        class Status(ctypes.Structure):
            _fields_ = [('ac', ctypes.c_byte), ('flag', ctypes.c_byte), ('pct', ctypes.c_byte),
                        ('saver', ctypes.c_byte), ('life', ctypes.c_ulong), ('full', ctypes.c_ulong)]
        s = Status()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return {0: True, 1: False}.get(s.ac)
        return None
    for d in ('/sys/class/power_supply/AC', '/sys/class/power_supply/ACAD', '/sys/class/power_supply/AC0'):
        if os.path.exists(f'{d}/online'):
            return open(f'{d}/online').read().strip() == '0'
    return None


def power():
    b = on_battery()
    if b is None:
        return report('SKIP', 'could not tell whether the charger is plugged in')
    report('OK' if b else 'FAIL', 'on battery' if b else 'charger plugged in: unplug it before pads go on skin')


def encoder():
    import torch
    if not torch.cuda.is_available():
        return report('FAIL', 'no CUDA')
    if not os.path.exists(os.path.join(lipread.LIPFLOW_MODELS, 'vsr', 'model.pth')):
        return report('SKIP', 'vsr/model.pth missing (bash lipread/setup.sh)')
    from lipread.model import bench_encoder
    for name, line in bench_encoder(250).items():
        print(f'    {name}: {line}')
    report('OK', 'encoder on 250 frames (then: python -m lipread.bench.tick_bench)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--skip-gpu', action='store_true')
    args = ap.parse_args(argv)
    print(f'laptop_check {time.strftime("%F %T")} {platform.platform()} python {platform.python_version()} '
          f'data {lipread.WORK}')
    check('1. NVIDIA driver', driver)
    if not args.skip_gpu:
        check('2. torch on the GPU', gpu_matmul)
    check('3. imports and MediaPipe', imports)
    check(f'4. webcam {args.cam}', webcam, args.cam, args.seconds)
    check('5. power', power)
    if not args.skip_gpu:
        check('6. encoder speed', encoder)
    print(f'\n== summary: {RESULTS.count("OK")} OK, {RESULTS.count("FAIL")} FAIL, {RESULTS.count("SKIP")} SKIP')
    return 1 if 'FAIL' in RESULTS else 0


if __name__ == '__main__':
    sys.exit(main())
