"""Helpers shared by the benchmark scripts. Must stay Python 3.9 compatible (the B0 env)."""
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
SPLITS_DIR = os.path.join(ROOT, 'bench', 'splits')
RESULTS_DIR = os.path.join(WORK, 'benchmark_results')
DATA_ROOT = os.path.join(WORK, 'silent_speech', 'emg_data')
GADDY_CHECKPOINT = os.path.join(WORK, 'baseline_gaddy', 'checkpoints', 'model.pt')
LM_PATH = os.path.join(WORK, 'baseline_gaddy', 'checkpoints', 'lm.binary')

FRAME_RATE = 689.06 / 8  # model output frames per second (86.13)
CLIP_SECONDS = 3.0

# Gaddy electrode sites 1..8 are data channels 0..7. Our rig uses sites 1, 2, 3, 6.
CHANNEL_CONFIGS = {
    '8ch': {'keep_channels': [0, 1, 2, 3, 4, 5, 6, 7], 'sites': [1, 2, 3, 4, 5, 6, 7, 8]},
    '6ch': {'keep_channels': [0, 1, 2, 3, 4, 5], 'sites': [1, 2, 3, 4, 5, 6]},
    '4ch': {'keep_channels': [0, 1, 2, 5], 'sites': [1, 2, 3, 6]},
}
EVAL_SPLITS = ['open_vocab_test', 'closed_vocab_silent']


def load_split(name):
    with open(os.path.join(SPLITS_DIR, name + '.json')) as f:
        return json.load(f)


def npz_key(item_id):
    return item_id.replace('/', '__')


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def cpu_name():
    try:
        out = subprocess.run(['lscpu'], stdout=subprocess.PIPE, universal_newlines=True).stdout
        for line in out.splitlines():
            if line.startswith('Model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment_info(torch):
    info = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'numpy': np.__version__,
        'cpu': cpu_name(),
        'cpu_threads_used': torch.get_num_threads(),
        'platform': platform.platform(),
    }
    return info


class Timer:
    def __init__(self, cuda_sync=None):
        self.cuda_sync = cuda_sync

    def __enter__(self):
        if self.cuda_sync:
            self.cuda_sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda_sync:
            self.cuda_sync()
        self.ms = (time.perf_counter() - self.start) * 1000


def latency_stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(v)), 'mean': float(v.mean()), 'p90': float(np.percentile(v, 90)),
            'min': float(v.min()), 'max': float(v.max()), 'n': int(len(v))}


def save_raw(out_dir, split, channels, meta, items, logits, inputs):
    """Raw model outputs for one split x channel config: <split>__<ch>.json + .npz."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, '%s__%s' % (split, channels))
    np.savez_compressed(base + '_logits.npz', **{npz_key(k): v for k, v in logits.items()})
    np.savez_compressed(base + '_inputs.npz', **{npz_key(k): v for k, v in inputs.items()})
    with open(base + '.json', 'w') as f:
        json.dump(dict(meta, items=items), f, indent=1)
