"""Precompute SVR-1's articulator targets for the Berkeley data. Two stages:

1. sparc  (any env with `pip install speech-articulatory-coding soundfile`):
       python -m svr.precompute_svr1 sparc [--device cuda]
   SPARC acoustic-to-articulatory inversion (Cho et al. 2024; as used for EMG by Lee et al. 2025) on every voiced
   recording's audio (16 kHz): 12 EMA trajectories + loudness at 50 Hz -> data/svr1_targets/sparc/<session>/<idx>.npy

2. align  (main venv):
       venv/bin/python -m svr.precompute_svr1 align
   Resamples each voiced recording's SPARC output to its model frames (86.13 Hz, the length of the cached input).
   Silent recordings have no audio: their targets come from the voiced twin of the same sentence, aligned by dynamic
   time warping on per-frame EMG energy (Gaddy & Klein's idea of transferring voiced targets to silent EMG).
   Writes data/svr1_targets/artic/<session>/<idx>.npy (frames, 13) and stats.json (per-dimension mean/std).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
DATA_ROOT = os.path.join(WORK, 'silent_speech', 'emg_data')
OUT = os.path.join(WORK, 'data', 'svr1_targets')


def stage_sparc(device):
    from sparc import load_model  # speech-articulatory-coding
    coder = load_model('en', device=device)
    audio = sorted(glob.glob(os.path.join(DATA_ROOT, 'voiced_parallel_data', '*', '*_audio_clean.flac'))
                   + glob.glob(os.path.join(DATA_ROOT, 'nonparallel_data', '*', '*_audio_clean.flac')))
    print(len(audio), 'voiced recordings', flush=True)
    for n, path in enumerate(audio, 1):
        session = os.path.relpath(os.path.dirname(path), DATA_ROOT)
        idx = os.path.basename(path).split('_')[0]
        out = os.path.join(OUT, 'sparc', session, f'{idx}.npy')
        if os.path.exists(out):
            continue
        code = coder.encode(path)
        ema = np.asarray(code['ema'], dtype=np.float32).reshape(-1, 12)
        loud = np.asarray(code['loudness'], dtype=np.float32).reshape(-1, 1)[:len(ema)]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, np.concatenate([ema, loud], 1))
        if n % 500 == 0:
            print(f'  {n}/{len(audio)}', flush=True)


def energy_features(x):
    """Per-frame log energy of each channel from a cached model input (frames*8, channels)."""
    f = x.reshape(-1, 8, x.shape[1]).astype(np.float64)
    e = np.log((f ** 2).mean(1) + 1e-6)
    return (e - e.mean(0)) / (e.std(0) + 1e-6)


_DTW = None


def _compile_dtw():
    from numba import njit

    @njit(cache=True)
    def dtw(cost):
        n, m = cost.shape
        acc = np.full((n + 1, m + 1), np.inf)
        acc[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        i, j = n, m
        match = np.zeros(n, np.int64)
        while i > 0 and j > 0:
            match[i - 1] = j - 1
            diag, up, left = acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            if diag <= up and diag <= left:
                i, j = i - 1, j - 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        return match
    return dtw


def dtw_path(a, b):
    """Monotonic alignment of frame sequences a (n, d) and b (m, d): for each frame of a, a frame of b."""
    global _DTW
    if _DTW is None:
        _DTW = _compile_dtw()
    return _DTW(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def resample_to(y, n):
    if len(y) == n:
        return y
    src = np.linspace(0, len(y) - 1, n)
    return np.stack([np.interp(src, np.arange(len(y)), y[:, d]) for d in range(y.shape[1])], 1).astype(np.float32)


def stage_align():
    sys.path.insert(0, ROOT)
    from svr.data import CACHE_ROOT, list_split
    entries = list_split('train') + list_split('dev')
    by_location = {}
    for s, i, info in entries:
        by_location.setdefault((info['book'], info['sentence_index']), []).append((s, i))
    voiced_targets, written = {}, 0
    for s, i, info in entries:
        src = os.path.join(OUT, 'sparc', s, f'{i}.npy')
        if 'silent' in s or not os.path.exists(src):
            continue
        x = np.load(os.path.join(CACHE_ROOT, s, f'{i}.npy'))
        y = resample_to(np.load(src), x.shape[0] // 8)
        voiced_targets[(s, i)] = (x, y)
        out = os.path.join(OUT, 'artic', s, f'{i}.npy')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, y)
        written += 1
    print('voiced targets', written, flush=True)
    silent_written = 0
    for s, i, info in entries:
        if 'silent' not in s:
            continue
        twins = [t for t in by_location.get((info['book'], info['sentence_index']), []) if t in voiced_targets]
        if not twins:
            continue
        xv, yv = voiced_targets[twins[0]]
        xs = np.load(os.path.join(CACHE_ROOT, s, f'{i}.npy'))
        match = dtw_path(energy_features(xs), energy_features(xv))
        out = os.path.join(OUT, 'artic', s, f'{i}.npy')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, yv[match])
        silent_written += 1
    print('silent targets via DTW', silent_written, flush=True)
    files = sorted(glob.glob(os.path.join(OUT, 'artic', '*', '*', '*.npy')))
    allv = np.concatenate([np.load(p) for p in files[::max(1, len(files) // 3000)]])  # spread over all sessions
    with open(os.path.join(OUT, 'stats.json'), 'w') as f:
        json.dump({'mean': allv.mean(0).tolist(), 'std': (allv.std(0) + 1e-6).tolist()}, f)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'sparc':
        stage_sparc(sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cuda')
    elif stage == 'align':
        stage_align()
    else:
        sys.exit('usage: python -m svr.precompute_svr1 sparc|align')
