"""A tiny SVR-1 checkpoint and fake rig signal for the live EMG reader's tests (CPU, random weights, no dataset)."""
from dataclasses import asdict

import numpy as np
import torch

from svr.preprocess import robust_spread, rig_gain, rig_unscaled
from svr.svr1 import SVR1Config, build

TINY = dict(d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, verifier_dim=64, verifier_layers=1)
TARGET = [14.0, 29.2, 15.7, 17.9, 16.0, 18.8, 13.9, 10.2]   # SVR1_4ch's scale_target, rounded


def raw_signal(seconds, seed=0, bursts=()):
    """(n, 4) uint16 Uno readings around mid-scale: noise, plus louder (t0, t1) bursts like mouthing."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 1000)
    x = 512 + 4 * rng.standard_normal((n, 4))
    for t0, t1 in bursts:
        a, b = int(t0 * 1000), int(t1 * 1000)
        x[a:b] += 40 * rng.standard_normal((b - a, 4))
    return np.clip(np.round(x), 0, 1023).astype(np.uint16)


def tiny_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    cfg = asdict(SVR1Config(**TINY))
    model = build(cfg)
    gain = rig_gain(robust_spread(rig_unscaled(raw_signal(10, seed))), TARGET)
    torch.save({'config': cfg, 'model': model.state_dict(), 'preprocess': 'causal', 'scale_target': TARGET,
                'session_gains': {'s1': gain.tolist()}, 'effort_ref': {'s1': 1.0}, 'pace_ref': {'s1': 1.0},
                'default_session': 's1'}, path)
    return path


def raw_msgs(raw, t_start=100.0, batch=10, counter0=0):
    """rig/publish.py's emg.raw messages for a raw signal, 10 samples each, on the sample clock."""
    for i in range(0, len(raw), batch):
        yield {'t0': t_start + i / 1000, 'fs': 1000, 'samples': raw[i:i + batch], 'counter': counter0 + i}
