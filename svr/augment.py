"""On-the-fly augmentation of SVR-1 model inputs (689 Hz, scaled, shape (frames*8, channels)).

speed        resample time by 1/factor (0.8-1.6x): faster or slower speech                         [E]
undershoot   articulatory undershoot of fast speech: compress burst peaks, overlap neighbouring bursts  [E, new]
hardware     our rig vs Berkeley's lab amplifier: per-channel gain, small channel mixing, random first-order
             high-pass (2-70 Hz, the BioAmp's band is not known exactly), 10-bit quantization, noise
time_mask    zero a few short spans
channel_drop zero random channels (8-to-6 pretraining variant)
Every function takes a numpy Generator, so a batch's augmentation is reproducible from (seed, epoch, batch).
"""
import numpy as np
from scipy.signal import lfilter

RATE = 689.06


def speed(x, factor):
    n = len(x)
    new = max(8, int(round(n / factor)) // 8 * 8)
    if new == n:
        return x
    src = np.linspace(0, n - 1, new)
    return np.stack([np.interp(src, np.arange(n), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def undershoot(x, rng, max_compress=0.35, max_overlap=0.35):
    """Smaller and overlapping muscle bursts, as in fast articulation."""
    u = rng.uniform(0, max_compress)
    k = 35  # ~50 ms envelope window
    env = np.sqrt(np.apply_along_axis(lambda c: np.convolve(c ** 2, np.ones(k) / k, 'same'), 0, x) + 1e-8)
    ref = np.median(env, 0, keepdims=True) + 1e-6
    y = x * np.maximum(env / ref, 1.0) ** (-u)
    a = rng.uniform(0, max_overlap)
    d = int(rng.uniform(0.010, 0.040) * RATE)
    shifted = np.zeros_like(y)
    shifted[d:] = y[:-d] if d else y
    return (y + a * shifted).astype(np.float32)


def hardware(x, rng):
    c = x.shape[1]
    mix = np.eye(c) + rng.normal(0, 0.03, (c, c))
    y = x @ mix.T * rng.normal(1.0, 0.15, (1, c))
    if rng.random() < 0.5:
        fc = rng.uniform(2, 70)
        alpha = 1 / (1 + 2 * np.pi * fc / RATE)
        y = lfilter([alpha, -alpha], [1, -alpha], y, axis=0)
    scale = np.abs(y).max() * rng.uniform(1.0, 2.0) + 1e-6
    step = 2 * scale / 1024
    y = np.round(y / step) * step
    y = y + rng.normal(0, 0.02 * (y.std() + 1e-6), y.shape)
    return y.astype(np.float32)


def time_mask(x, rng, n_max=3, max_frames=8):
    y = x.copy()
    frames = len(x) // 8
    for _ in range(rng.integers(0, n_max + 1)):
        w = int(rng.integers(1, max_frames + 1))
        if frames > w + 1:
            s = int(rng.integers(0, frames - w))
            y[s * 8:(s + w) * 8] = 0
    return y


def channel_drop(keep, rng, p, min_keep=3):
    """Random subset of the available channel slots (keep: list of slot indices)."""
    keep = list(keep)
    if p <= 0 or len(keep) <= min_keep:
        return keep
    kept = [c for c in keep if rng.random() > p]
    if len(kept) < min_keep:
        kept = list(rng.choice(keep, min_keep, replace=False))
    return sorted(kept)
