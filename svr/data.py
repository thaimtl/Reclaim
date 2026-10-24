"""Gaddy's dataset: train/dev/test splits, a preprocessed-input cache, and his length-aware batching.

Splits follow his EMGDataset with testset_largedev.json:
    dev / test  silent recordings whose (book, sentence_index) is in the dev / test list
    train       every silent and voiced recording whose (book, sentence_index) is in neither list
Recordings marked sentence_index < 0 (silence between sentences) are never used.

Model inputs are preprocessed once (svr.preprocess, all 8 channels) and cached as .npy under
data/preprocessed/. Fewer channels are simulated at load time by zeroing; that matches Gaddy's
--remove_channels exactly because the input scaling maps 0 to 0.
"""
import json
import multiprocessing as mp
import os
import random
import string

import numpy as np

from svr.preprocess import load_utterance
from svr.text import text_to_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
DATA_ROOT = os.path.join(WORK, 'silent_speech', 'emg_data')
CACHE_ROOT = os.path.join(WORK, 'data', 'preprocessed')
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


def cache_path(session_dir, index):
    return os.path.join(CACHE_ROOT, session_dir, f'{index}.npy')


def _preprocess_one(args):
    session_dir, index = args
    path = cache_path(session_dir, index)
    if not os.path.exists(path):
        x = load_utterance(os.path.join(DATA_ROOT, session_dir), index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + '.tmp.npy', x)
        os.replace(path + '.tmp.npy', path)
    return path


def build_cache(entries, processes=None):
    todo = [(s, i) for s, i, _ in entries if not os.path.exists(cache_path(s, i))]
    if todo:
        with mp.Pool(processes or max(1, (os.cpu_count() or 2) - 2)) as pool:
            for n, _ in enumerate(pool.imap_unordered(_preprocess_one, todo, chunksize=16), 1):
                if n % 1000 == 0:
                    print(f'  preprocessed {n}/{len(todo)}', flush=True)


class EMGTextDataset:
    """In-memory model inputs + character targets for one split."""

    def __init__(self, split, keep_channels=None):
        self.entries = list_split(split)
        build_cache(self.entries)
        self.keep_channels = keep_channels
        self.inputs, self.targets, self.texts, self.ids, self.emg_lengths = [], [], [], [], []
        drop = None if keep_channels is None else [c for c in range(8) if c not in set(keep_channels)]
        for session_dir, index, info in self.entries:
            x = np.load(cache_path(session_dir, index))
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
