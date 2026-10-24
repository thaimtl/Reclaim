"""Parity check: does our reimplementation (B1) reproduce Gaddy's exact setup (B0)?

Compares one difference at a time, with the pass bars from the benchmark plan:

  step 1  B0 (CPU, his code, PyTorch 2.0)  vs  B1 CPU (our code, PyTorch 2.11)
          -> greedy transcripts identical on >= 98% of open-vocab test sentences
  step 2  B1 CPU  vs  B1 GPU (TF32 off)
          -> same bar
  step 3  ctcdecode  vs  our beam/LM decoder, both on B0's log-probs
          -> beam/LM WER within 2 points

Also reports how close the model inputs and the per-frame log-probs are, at 8 and 4 channels.
Run after score.py:

    venv-svr/bin/python -m bench.parity
"""
import json
import os

import numpy as np

from bench.common import CHANNEL_CONFIGS, EVAL_SPLITS, RESULTS_DIR, npz_key

B0, B1_CPU, B1_GPU = 'B0_gaddy_exactenv_cpu', 'B1_ours_cpu', 'B1_ours_gpu'
GREEDY_IDENTICAL_BAR = 0.98
BEAM_WER_BAR = 0.02


def load(model, split, channels, kind):
    return np.load(os.path.join(RESULTS_DIR, model, 'raw', '%s__%s_%s.npz' % (split, channels, kind)))


def load_cases(model, name):
    with open(os.path.join(RESULTS_DIR, model, name)) as f:
        return json.load(f)


def array_diff(a_model, b_model, split, channels, kind):
    a, b = load(a_model, split, channels, kind), load(b_model, split, channels, kind)
    per, shape_mismatch, exact = [], [], 0
    argmax_agree, frames = 0, 0
    for key in a.files:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            shape_mismatch.append(key)
            continue
        d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
        per.append(d)
        exact += d == 0.0
        if kind == 'logits':
            argmax_agree += int((x.argmax(-1) == y.argmax(-1)).sum())
            frames += x.shape[0]
    out = {'n': len(a.files), 'shape_mismatches': shape_mismatch, 'bit_identical': exact,
           'max_abs_diff': max(per) if per else None, 'median_max_abs_diff': float(np.median(per)) if per else None}
    if kind == 'logits':
        out['frame_argmax_agreement'] = argmax_agree / frames if frames else None
    return out


def transcript_compare(a_model, b_model, channels, field_a='greedy', field_b='greedy', labels=None):
    a = load_cases(a_model, 'open_vocab__%s.json' % channels)
    b = load_cases(b_model, 'open_vocab__%s.json' % channels)
    label_a, label_b = labels or (a_model, b_model)
    bc = {c['id']: c for c in b['cases']}
    diffs = [{'id': c['id'], 'reference': c['reference'], label_a: c[field_a], label_b: bc[c['id']][field_b]}
             for c in a['cases'] if c[field_a] != bc[c['id']][field_b]]
    n = len(a['cases'])
    return {'n': n, 'identical': n - len(diffs), 'identical_fraction': (n - len(diffs)) / n, 'differences': diffs}


def phrase_agreement(a_model, b_model, channels, bench):
    a = load_cases(a_model, '%s__%s.json' % (bench, channels))
    b = load_cases(b_model, '%s__%s.json' % (bench, channels))
    bc = {c['id']: c for c in b['cases']}
    same = sum(c['predicted_index'] == bc[c['id']]['predicted_index'] for c in a['cases'])
    return {'n': len(a['cases']), 'same_prediction': same,
            'top1_accuracy': {a_model: a['metrics']['top1_accuracy'], b_model: b['metrics']['top1_accuracy']}}


def model_step(a_model, b_model):
    step = {}
    for channels in CHANNEL_CONFIGS:
        g = transcript_compare(a_model, b_model, channels)
        a = load_cases(a_model, 'open_vocab__%s.json' % channels)['metrics']
        b = load_cases(b_model, 'open_vocab__%s.json' % channels)['metrics']
        step[channels] = {
            'model_inputs': {s: array_diff(a_model, b_model, s, channels, 'inputs') for s in EVAL_SPLITS},
            'log_probs': {s: array_diff(a_model, b_model, s, channels, 'logits') for s in EVAL_SPLITS},
            'greedy_transcripts': g,
            'greedy_wer': {a_model: a['greedy']['wer'], b_model: b['greedy']['wer']},
            'phrase_id_openvocab': phrase_agreement(a_model, b_model, channels, 'phrase_id_openvocab'),
            'phrase_id_closedvocab': phrase_agreement(a_model, b_model, channels, 'phrase_id_closedvocab'),
            'pass': g['identical_fraction'] >= GREEDY_IDENTICAL_BAR,
        }
    return step


def decoder_step():
    step = {}
    for channels in CHANNEL_CONFIGS:
        cases = load_cases(B0, 'open_vocab__%s.json' % channels)
        m = cases['metrics']
        t = transcript_compare(B0, B0, channels, 'beam_lm', 'beam_lm_svr', labels=('ctcdecode', 'svr_decoder'))
        gap = abs(m['beam_lm']['wer'] - m['beam_lm_svr_decoder']['wer'])
        step[channels] = {
            'beam_lm_wer': {'ctcdecode': m['beam_lm']['wer'], 'svr_decoder': m['beam_lm_svr_decoder']['wer']},
            'wer_gap_points': gap * 100,
            'identical_transcripts': t['identical'], 'n': t['n'],
            'differences': t['differences'],
            'pass': gap <= BEAM_WER_BAR,
        }
    return step


def main():
    report = {
        'pass_bars': {'steps_1_2': 'greedy transcripts identical on >= %d%% of open-vocab test sentences'
                                   % (GREEDY_IDENTICAL_BAR * 100),
                      'step_3': 'beam/LM WER within %d points' % (BEAM_WER_BAR * 100)},
        'step1_B0_vs_B1_cpu': model_step(B0, B1_CPU),
        'step2_B1_cpu_vs_B1_gpu': model_step(B1_CPU, B1_GPU),
        'step3_ctcdecode_vs_svr_decoder_on_B0_logits': decoder_step(),
    }
    official = os.path.join(RESULTS_DIR, B0, 'official_eval.json')
    if os.path.exists(official):
        with open(official) as f:
            report['B0_official_script'] = json.load(f)
    report['all_pass'] = all(v['pass'] for k in ('step1_B0_vs_B1_cpu', 'step2_B1_cpu_vs_B1_gpu',
                                                 'step3_ctcdecode_vs_svr_decoder_on_B0_logits')
                             for v in report[k].values())
    path = os.path.join(RESULTS_DIR, 'parity_B0_vs_B1.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=1)
    print('wrote', path)
    for k in ('step1_B0_vs_B1_cpu', 'step2_B1_cpu_vs_B1_gpu', 'step3_ctcdecode_vs_svr_decoder_on_B0_logits'):
        print(k, {ch: v['pass'] for ch, v in report[k].items()})
    print('all_pass', report['all_pass'])


if __name__ == '__main__':
    main()
