"""Freeze the benchmark test sets and the phrase-ID candidate lists.

Run once with the B0 environment, from baseline_gaddy/silent_speech, so the open-vocab
test set comes from Gaddy's own EMGDataset(test=True):

    cd baseline_gaddy/silent_speech
    ../.venv-svr/bin/python ../../bench/make_splits.py

Writes bench/splits/{open_vocab_test,closed_vocab_silent,phrase_id_openvocab,phrase_id_closedvocab}.json.
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.getcwd())  # Gaddy's modules
from svr.text import clean_text, spell_out_date_time  # noqa: E402

SPLITS = os.path.join(ROOT, 'bench', 'splits')
N_CANDIDATES = 10
SEED = 0


def item(session_dir, index, text, reference):
    rel = os.path.relpath(session_dir, 'emg_data')
    return {'id': f'{rel}/{index}', 'session_dir': rel, 'index': int(index), 'text': text, 'reference': reference}


def open_vocab_items():
    from absl import flags
    from read_emg import EMGDataset
    flags.FLAGS(['make_splits'])
    testset = EMGDataset(test=True)
    out = []
    for directory_info, index in testset.example_indices:
        with open(os.path.join(directory_info.directory, f'{index}_info.json')) as f:
            text = json.load(f)['text']
        out.append(item(os.path.normpath(directory_info.directory), index, text, clean_text(text)))
    return out


def closed_vocab_items():
    out = []
    for path in sorted(glob.glob('emg_data/closed_vocab/silent/*/*_info.json')):
        with open(path) as f:
            info = json.load(f)
        if info['sentence_index'] < 0 or not info['text'].strip():
            continue
        index = int(re.match(r'(\d+)_info.json', os.path.basename(path)).group(1))
        out.append(item(os.path.dirname(path), index, info['text'], spell_out_date_time(info['text'])))
    out.sort(key=lambda x: (x['session_dir'], x['index']))
    return out


def phrase_template(text):
    if re.fullmatch(r'\d\d:\d\d [AP]M', text):
        return 'time'
    if re.fullmatch(r'[A-Za-z]+ \d\d? \d{4}', text):
        return 'month_day_year'
    if re.fullmatch(r'[A-Za-z]+ [A-Za-z]+ \d\d?', text):
        return 'weekday_month_day'
    return 'other'


def phrase_id(items, group_fn, description):
    rng = np.random.RandomState(SEED)
    pools = {}
    for it in items:
        pools.setdefault(group_fn(it), set()).add(it['reference'])
    pools = {k: sorted(v) for k, v in pools.items()}
    trials = []
    for it in items:
        others = [p for p in pools[group_fn(it)] if p != it['reference']]
        distractors = [others[i] for i in rng.choice(len(others), N_CANDIDATES - 1, replace=False)]
        candidates = distractors + [it['reference']]
        order = rng.permutation(N_CANDIDATES)
        candidates = [candidates[i] for i in order]
        trials.append({'id': it['id'], 'candidates': candidates, 'answer_index': candidates.index(it['reference'])})
    return {'description': description, 'n_candidates': N_CANDIDATES, 'seed': SEED, 'trials': trials}


def write(name, obj):
    os.makedirs(SPLITS, exist_ok=True)
    with open(os.path.join(SPLITS, name), 'w') as f:
        json.dump(obj, f, indent=1)
    print('wrote', name)


def main():
    ov = open_vocab_items()
    write('open_vocab_test.json', {
        'description': "Gaddy's silent open-vocabulary test set: EMGDataset(test=True) with testset_largedev.json, "
                       'in his evaluation order. reference = his clean_text(text).',
        'data_root': 'silent_speech/emg_data', 'items': ov})

    cv = closed_vocab_items()
    write('closed_vocab_silent.json', {
        'description': 'All silent closed-vocabulary (date/time) recordings with text. Never used in training. '
                       'reference = text spelled out as spoken, by svr.text.spell_out_date_time.',
        'data_root': 'silent_speech/emg_data', 'items': cv})

    write('phrase_id_openvocab.json', phrase_id(
        ov, lambda it: 'all',
        'For each open-vocab test recording: its sentence + 9 other test sentences, shuffled.'))
    by_id = {it['id']: it for it in cv}
    write('phrase_id_closedvocab.json', phrase_id(
        cv, lambda it: phrase_template(by_id[it['id']]['text']),
        'For each closed-vocab recording: its phrase + 9 other phrases of the same template '
        '(time / weekday month day / month day year), shuffled.'))


if __name__ == '__main__':
    main()
