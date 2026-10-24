"""E3 on the laptop: tick time as the live loop sees it, fp32 vs fp16, for segments growing to 2, 4 and 8 s.

Each "segment" grows on the live tick schedule (a tick at the due points 3, 5, 8, 10, 13, ... frames for 100 ms) with
a --gap pause between ticks (as live: the GPU idles in between), using real mouth crops from the sample when present
(else random crops: same speed). Every tick goes through live.run.Runner.tick, so "total" is what the tick loop
spends: stack the crops + read (encoder + CTC + copy) + build the CamTick + encode it for the bus (msgpack; the zmq
send itself is ~0.02 ms). Reports p50 / p95 / max of the total per length band, the timing_ms split (front end,
Conformer, CTC), peak GPU memory, and whether the 40 ms p95 target at 8 s holds.

    venv-live/bin/python -m live.bench.tick_bench                 # ~2 min; paste the output back
    venv-live/bin/python -m live.bench.tick_bench --gap 0         # back to back (GPU kept busy), for comparison
    venv-live/bin/python -m live.bench.tick_bench --tick-ms 200   # the old 200 ms schedule
"""
import argparse
import os
import time

import numpy as np
import torch

import live
from live.bus import encode
from live.endpoint import Segmenter
from live.model import LiveModel
from live.run import Runner
from live.types import FPS, Crop

BANDS = ((2, 50), (4, 100), (8, 200))   # seconds, frames
TARGET_MS = 40.0


def sample_rois(n=200):
    face = os.path.join(live.LIPFLOW_MODELS, 'face_landmarker.task')
    if not (os.path.exists(live.SAMPLE_VIDEO) and os.path.exists(face)):
        return np.random.default_rng(0).integers(0, 255, (n, 96, 96), dtype=np.uint8), 'random crops'
    from live.crop import LiveCropper
    from live.source import ReplaySource
    src = ReplaySource(live.SAMPLE_VIDEO, live.SAMPLE_START, live.SAMPLE_START + n / 25 + 1, realtime=False)
    c = LiveCropper(face, k=0)
    out = [x.roi96 for f in src.frames() for x in c.push(f)]
    src.close()
    c.close()
    return np.stack(out[:n]), 'sample crops'


def run(model, rois, gap, reps, tick_ms, fe_cache=True):
    """Per band: TickStats of a segment growing on the tick schedule up to the band's length, reps times."""
    out = {}
    t = 1000.0
    for s, n in BANDS:
        stats = []
        for _ in range(reps):
            r = Runner(model, encode, Segmenter(never_end=True, cap_s=n / FPS + 1), tick_ms=tick_ms, fe_cache=fe_cache)
            for i in range(n):
                t += 1 / FPS
                got = r.on_crops([(Crop(t=t, roi96=rois[i], mouth_open=0.0, face_ok=True), 0.0)])
                if got:
                    stats += got
                    time.sleep(gap)
        out[s] = stats
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gap', type=float, default=0.1, help='seconds between ticks (0.1 = live)')
    ap.add_argument('--tick-ms', type=int, choices=(100, 200), default=100)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--only', choices=('fp32', 'fp16'), default=None)
    ap.add_argument('--no-fe-cache', action='store_true', help='re-run the front end on the whole segment')
    args = ap.parse_args()
    rois, what = sample_rois()
    print(f'GPU {torch.cuda.get_device_name(0)}, {what}, ticks every {args.tick_ms} ms, gap {args.gap * 1000:.0f} ms, '
          f'reps {args.reps}')
    for half in (False, True):
        name = 'fp16' if half else 'fp32'
        if args.only and args.only != name:
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        m = LiveModel(half=half)
        load = time.perf_counter() - t0
        res = run(m, rois, args.gap, args.reps, args.tick_ms, not args.no_fe_cache)
        print(f'\n{name}: load + warm-up {load:.1f} s (warm-up {m.warmup_s:.1f} s), '
              f'peak GPU {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB')
        for s, st in res.items():
            tot = [x.total_ms for x in st]
            p = {k: np.percentile([getattr(x, k) for x in st], 50) for k in ('frontend_ms', 'conformer_ms', 'ctc_ms')}
            print(f'  segments up to {s} s: total p50 {np.percentile(tot, 50):5.1f}  p95 {np.percentile(tot, 95):5.1f}'
                  f'  max {max(tot):5.1f} ms  ({len(st)} ticks; p50 front end {p["frontend_ms"]:.1f}, '
                  f'Conformer {p["conformer_ms"]:.1f}, CTC {p["ctc_ms"]:.1f})')
        p95_8 = np.percentile([x.total_ms for x in res[8]], 95)
        print(f'  target p95 < {TARGET_MS:.0f} ms at 8 s: {"OK" if p95_8 < TARGET_MS else "MISS"} ({p95_8:.1f} ms)')
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
