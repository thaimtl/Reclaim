"""E1: crop smoothing modes against the training-style centred window, and the cost of blendshapes.

Landmarks are detected once per frame over the whole sample video (live order, VIDEO mode) and shared by every mode.
Each mode's crops are cut into 8 s windows (windows with < 90% faces skipped), encoded with LipReader (fp32, GPU) and
read greedily. Reference: k = 6 symmetric, bit-exact with Lipflow's mouth_rois (= the training crop).

    venv-live/bin/python -m lipread.bench.smooth_check [--video ...] [--window 8]
"""
import argparse
import time

import jiwer
import numpy as np

import lipread
from lipread.crop import BlendTracker, LiveCropper
from lipread.source import ReplaySource
from lipread.types import FPS

MODES = {
    'ref k6 symmetric': dict(k=6, symmetric=True),
    'box k2': dict(k=2),
    'box k0': dict(k=0),
    'ema 0.3': dict(k=0, smooth='ema', ema_alpha=0.3),
    'ema 0.5': dict(k=0, smooth='ema', ema_alpha=0.5),
    'ema 0.7': dict(k=0, smooth='ema', ema_alpha=0.7),
}


def crops_for(frames, cache, face_model, **kw):
    c = LiveCropper(face_model, **kw)
    c.detect = lambda image, t: cache[t]
    out = []
    for f in frames:
        out += c.push(f)
    out += c.flush()
    c.close()
    return out


def blend_cost(frames, face_model, n=300):
    """Median detect ms/frame with blendshapes off and on, over n consecutive frames from the first window that has
    faces in every frame (frames without a face return early and would hide the cost)."""
    out = {}
    for on in (False, True):
        tr = BlendTracker(face_model, blendshapes=on)
        ms = []
        for f in frames[:n]:
            t0 = time.perf_counter()
            tr.detect(f.image, int(round(f.t * 1000)))
            ms.append((time.perf_counter() - t0) * 1000)
        tr.close()
        out[on] = np.median(ms[20:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', default=lipread.SAMPLE_VIDEO)
    ap.add_argument('--window', type=float, default=8.0)
    args = ap.parse_args()
    face_model = f'{lipread.LIPFLOW_MODELS}/face_landmarker.task'

    src = ReplaySource(args.video, realtime=False)
    frames = list(src.frames())
    src.close()
    tr = BlendTracker(face_model)
    t0 = time.perf_counter()
    cache = {f.t: tr.detect(f.image, int(round(f.t * 1000))) for f in frames}
    det_ms = (time.perf_counter() - t0) * 1000 / len(frames)
    tr.close()
    print(f'{len(frames)} frames ({len(frames) / FPS:.1f} s), faces {sum(o is not None for o in cache.values())}, '
          f'detect + blendshapes {det_ms:.2f} ms/frame (incl. RGB convert)')
    has = np.array([cache[f.t] is not None for f in frames])
    start = next(i for i in range(len(frames) - 300) if has[i:i + 300].all())
    cost = blend_cost(frames[start:], face_model)
    print(f'detect median ms/frame: blendshapes off {cost[False]:.2f}, on {cost[True]:.2f}')

    crops = {name: crops_for(frames, cache, face_model, **kw) for name, kw in MODES.items()}
    ref = crops['ref k6 symmetric']
    n = int(args.window * FPS)
    wins = [(i, i + n) for i in range(0, len(ref) - n // 2, n)
            if np.mean([c.face_ok for c in ref[i:i + n]]) >= 0.9]
    print(f'{len(wins)} windows of {args.window:.0f} s with >= 90% faces')

    from lipflow.vsr import LipReader
    reader = LipReader(device='cuda', use_lm=False, personal=False)
    texts = {}
    for name, cs in crops.items():
        texts[name] = [reader.greedy(reader.encode(np.stack([c.roi96 for c in cs[a:b]]))) for a, b in wins]

    print(f'\n{"mode":18s} {"lat ms":>6s} {"px mean":>7s} {"px p99":>6s} {"px max":>6s} {"WER vs ref":>10s} '
          f'{"same text":>9s}')
    r = np.stack([c.roi96 for c in ref]).astype(np.int16)
    for name, cs in crops.items():
        d = np.abs(np.stack([c.roi96 for c in cs]).astype(np.int16) - r).mean(axis=(1, 2))
        pairs = [(a, b) for a, b in zip(texts['ref k6 symmetric'], texts[name]) if a]
        wer = jiwer.wer([a for a, _ in pairs], [b for _, b in pairs])
        same = sum(a == b for a, b in pairs)
        lat = MODES[name]['k'] * 1000 / FPS
        print(f'{name:18s} {lat:6.0f} {d.mean():7.3f} {np.percentile(d, 99):6.2f} {d.max():6.2f} {wer:10.3f} '
              f'{same:4d}/{len(pairs):<4d}')
    print('\nexample window (ref | box k2 | ema 0.5):')
    for k in range(min(3, len(wins))):
        print(f'  {texts["ref k6 symmetric"][k]}\n  {texts["box k2"][k]}\n  {texts["ema 0.5"][k]}\n')


if __name__ == '__main__':
    main()
