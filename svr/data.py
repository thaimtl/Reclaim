"""Gaddy's dataset: train/dev/test splits, a preprocessed-input cache, and his length-aware batching.

Splits follow his EMGDataset with testset_largedev.json:
    dev / test  silent recordings whose (book, sentence_index) is in the dev / test list
    train       every silent and voiced recording whose (book, sentence_index) is in neither list
Recordings marked sentence_index < 0 (silence between sentences) are never used.

Model inputs are preprocessed once (svr.preprocess, all 8 channels) and cached as .npy:
    causal  data/preprocessed_causal/   rig/filters.py, unscaled; scaled at load time with the session gain
                                        (target_c / spread_c, data/svr1_targets/scale.json; SVR1_SPEC section 1)
    gaddy   data/preprocessed/          Gaddy's zero-phase filters, already scaled (the B0/B1 parity baselines)
Fewer channels are simulated at load time by zeroing; that matches Gaddy's --remove_channels exactly because the
input scaling maps 0 to 0.
"""
import json
import multiprocessing as mp
import os
import random
import string

import numpy as np

from svr.preprocess import apply_gain, load_unscaled, load_utterance, robust_spread
from svr.text import text_to_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
DATA_ROOT = os.path.join(WORK, 'silent_speech', 'emg_data')
CACHE_ROOT = os.path.join(WORK, 'data', 'preprocessed')  # gaddy
CACHE_ROOTS = {'gaddy': CACHE_ROOT, 'causal': os.path.join(WORK, 'data', 'preprocessed_causal')}
SCALE_FILE = os.path.join(WORK, 'data', 'svr1_targets', 'scale.json')
TESTSET_FILE = os.path.join(WORK, 'silent_speech', 'testset_largedev.json')
SILENT_DIRS = ['silent_parallel_data']
VOICED_DIRS = ['voiced_parallel_data', 'nonparallel_data']
MAX_BATCH_EMG_SAMPLES = 128000  # Gaddy's SizeAwareSampler budget, in raw 1000 Hz samples


def list_split(split):
    """[(session_dir relative to DATA_ROOT, index, info dict)] for 'train', 'dev' or 'test'."""
    with open(TESTSET_FILE) as f:
        lists = json.load(f)
    held_out = {tuple(x) for x in lists['dev']} | {tuple(x) for x in lists['test']}
    wanted = {tuple(x) for x in lists[split]} if split != 'train' else None
    groups = SILENT_DIRS if split != 'train' else SILENT_DIRS + VOICED_DIRS
    out = []
    for group in groups:
        for session in sorted(os.listdir(os.path.join(DATA_ROOT, group))):
            session_dir = os.path.join(group, session)
            full = os.path.join(DATA_ROOT, session_dir)
            for fname in sorted(os.listdir(full)):
                if not fname.endswith('_info.json'):
                    continue
                with open(os.path.join(full, fname)) as f:
                    info = json.load(f)
                if info['sentence_index'] < 0:
                    continue
                location = (info['book'], info['sentence_index'])
                if (split == 'train' and location not in held_out) or (split != 'train' and location in wanted):
                    out.append((session_dir, int(fname.split('_')[0]), info))
    return out


def cache_path(session_dir, index, preprocess='gaddy'):
    return os.path.join(CACHE_ROOTS[preprocess], session_dir, f'{index}.npy')


def _preprocess_one(args):
    session_dir, index, preprocess = args
    path = cache_path(session_dir, index, preprocess)
    if not os.path.exists(path):
        full = os.path.join(DATA_ROOT, session_dir)
        x = load_unscaled(full, index) if preprocess == 'causal' else load_utterance(full, index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npy'  # two jobs may build the same cache
        np.save(tmp, x)
        os.replace(tmp, path)
    return path


def build_cache(entries, processes=None, preprocess='gaddy'):
    todo = [(s, i, preprocess) for s, i, _ in entries if not os.path.exists(cache_path(s, i, preprocess))]
    if todo:
        print(f'preprocessing {len(todo)} recordings ({preprocess}) into {CACHE_ROOTS[preprocess]}', flush=True)
        with mp.Pool(processes or max(1, (os.cpu_count() or 2) - 2)) as pool:
            for n, _ in enumerate(pool.imap_unordered(_preprocess_one, todo, chunksize=16), 1):
                if n % 1000 == 0:
                    print(f'  preprocessed {n}/{len(todo)}', flush=True)


def compute_scale_stats(stride=4):
    """Robust spread of the causal-filtered Berkeley signal per channel: per session (over every recording of the
    session, all splits, like Austin's sessions) and pooled over all sessions (the targets). Writes SCALE_FILE."""
    entries = list_split('train') + list_split('dev') + list_split('test')
    build_cache(entries, preprocess='causal')
    by_session = {}
    for s, i, _ in entries:
        by_session.setdefault(s, []).append(np.load(cache_path(s, i, 'causal'))[::stride])
    spreads = {s: robust_spread(np.concatenate(xs)).tolist() for s, xs in sorted(by_session.items())}
    pooled = np.concatenate([x for xs in by_session.values() for x in xs])[::2]  # weighted by duration
    stats = {'target': robust_spread(pooled).tolist(), 'sessions': spreads,
             'definition': '1.4826 * median(|x - median(x)|) per channel of the rig/filters.py-filtered signal at '
                           '689.06 Hz; gain = target / session spread, applied before the soft clip'}
    os.makedirs(os.path.dirname(SCALE_FILE), exist_ok=True)
    tmp = f'{SCALE_FILE}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(stats, f, indent=1)
    os.replace(tmp, SCALE_FILE)
    return stats


def scale_stats():
    if not os.path.exists(SCALE_FILE):
        return compute_scale_stats()
    with open(SCALE_FILE) as f:
        return json.load(f)


_EXTRA_SPREADS = {}


def session_spread(session_dir, stats, max_recordings=60, stride=4):
    """Robust spread of a session in the stats, or (sessions outside the splits, e.g. closed_vocab) computed from its
    own recordings."""
    if session_dir in stats['sessions']:
        return np.array(stats['sessions'][session_dir])
    if session_dir not in _EXTRA_SPREADS:
        full = os.path.join(DATA_ROOT, session_dir)
        idx = sorted(int(f.split('_')[0]) for f in os.listdir(full) if f.endswith('_emg.npy'))[:max_recordings]
        x = np.concatenate([load_unscaled(full, i, use_audio_length=False)[::stride] for i in idx])
        _EXTRA_SPREADS[session_dir] = robust_spread(x)
    return _EXTRA_SPREADS[session_dir]


def session_gain(session_dir, stats=None):
    """Per-channel gain of one Berkeley session (target / spread); channels with no signal get 0."""
    stats = stats or scale_stats()
    spread = session_spread(session_dir, stats)
    target = np.array(stats['target'])
    return np.where(spread > 1e-9, target / np.maximum(spread, 1e-9), 0.0).astype(np.float32)


class EMGTextDataset:
    """In-memory model inputs + character targets for one split."""

    def __init__(self, split, keep_channels=None, preprocess='causal'):
        self.entries = list_split(split)
        build_cache(self.entries, preprocess=preprocess)
        stats = scale_stats() if preprocess == 'causal' else None
        self.keep_channels = keep_channels
        self.preprocess = preprocess
        self.inputs, self.targets, self.texts, self.ids, self.emg_lengths = [], [], [], [], []
        drop = None if keep_channels is None else [c for c in range(8) if c not in set(keep_channels)]
        for session_dir, index, info in self.entries:
            x = np.load(cache_path(session_dir, index, preprocess))
            if stats is not None:
                x = apply_gain(x, session_gain(session_dir, stats))
            if drop:
                x = x.copy()
                x[:, drop] = 0
            self.inputs.append(x)
            self.targets.append(np.array(text_to_ids(info['text']), dtype=np.int64))
            self.texts.append(info['text'])
            self.ids.append(f'{session_dir}/{index}')
            self.emg_lengths.append(sum(c[0] for c in info['chunks']))

    def __len__(self):
        return len(self.inputs)

    def batches(self, rng=random, max_samples=MAX_BATCH_EMG_SAMPLES):
        """Gaddy's SizeAwareSampler: shuffle, skip texts without letters, fill up to the sample
        budget (raw 1000 Hz EMG samples), drop the last incomplete batch."""
        order = list(range(len(self)))
        rng.shuffle(order)
        batch, total = [], 0
        for i in order:
            if not any(ch in string.ascii_letters for ch in self.texts[i]):
                continue
            length = self.emg_lengths[i]
            if length + total > max_samples:
                if batch:
                    yield batch
                batch, total = [], 0
            batch.append(i)
            total += length
