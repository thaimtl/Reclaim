"""Score raw model outputs into per-case JSON files and a summary.

For each model folder benchmark_results/<model>/raw (written by b0_dump.py / b1_run.py):

    open_vocab__<ch>.json             per test recording: reference, greedy + beam/LM transcripts, WER, CER, timings
    phrase_id_openvocab__<ch>.json    per recording: 10 candidate sentences with CTC log-likelihoods, pick, rank
    phrase_id_closedvocab__<ch>.json  same for the closed-vocab date/time phrases
    summary.json                      every metric, latency, parameter count, environment

Beam search with the LM uses our decoder (svr.decoders.BeamLMDecoder). For B0 the headline
beam/LM transcript is Gaddy's own ctcdecode output; our decoder's output on the same B0
log-probs is kept alongside it (beam_lm_svr) for the decoder parity check.

    venv-svr/bin/python -m bench.score B0_gaddy_exactenv_cpu B1_ours_cpu B1_ours_gpu
"""
import json
import multiprocessing as mp
import os
import sys
import time

import jiwer
import numpy as np

from bench.common import CHANNEL_CONFIGS, LM_PATH, RESULTS_DIR, SPLITS_DIR, latency_stats, load_split, npz_key
from svr.decoders import BeamLMDecoder, greedy_decode, score_phrases

_DECODER = None

# (raw outputs to score, candidate trials) pairs; the @Nx raw splits are the time-compressed fast-speech inputs
CANDIDATE_BENCHMARKS = [
    ('open_vocab_test', 'phrase_id_openvocab'),
    ('closed_vocab_silent', 'phrase_id_closedvocab'),
    ('open_vocab_test', 'homophene_10'),
    ('open_vocab_test', 'homophene_50'),
    ('open_vocab_test', 'nearduplicate_10'),
    ('open_vocab_test@1.25x', 'homophene_10'),
    ('open_vocab_test@1.5x', 'homophene_10'),
]


def _beam_worker(log_probs):
    start = time.perf_counter()
    text = _DECODER.decode(log_probs)
    return text, (time.perf_counter() - start) * 1000


def beam_decode_all(log_prob_list, processes):
    global _DECODER
    if _DECODER is None:
        _DECODER = BeamLMDecoder(LM_PATH)
    with mp.get_context('fork').Pool(processes) as pool:
        return pool.map(_beam_worker, log_prob_list, chunksize=1)


def wer(ref, hyp):
    return jiwer.wer(ref, hyp)


def cer(ref, hyp):
    return jiwer.cer(ref, hyp)


def corpus_rates(refs, hyps):
    return {'wer': jiwer.wer(refs, hyps), 'cer': jiwer.cer(refs, hyps)}


def load_raw(model, split, channels):
    base = os.path.join(RESULTS_DIR, model, 'raw', '%s__%s' % (split, channels))
    with open(base + '.json') as f:
        meta = json.load(f)
    logits = np.load(base + '_logits.npz')
    return meta, logits


def write(model, name, obj):
    path = os.path.join(RESULTS_DIR, model, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    print('wrote', os.path.relpath(path, RESULTS_DIR), flush=True)


def score_open_vocab(model, channels, processes):
    meta, logits = load_raw(model, 'open_vocab_test', channels)
    split = {it['id']: it for it in load_split('open_vocab_test')['items']}
    rows = meta['items']
    lps = [logits[npz_key(r['id'])] for r in rows]
    beams = beam_decode_all(lps, processes)
    is_b0 = 'ctcdecode_text' in rows[0]
    cases, refs, greedy_hyps, beam_hyps, svr_hyps = [], [], [], [], []
    for r, lp, (svr_text, svr_ms) in zip(rows, lps, beams):
        ref = split[r['id']]['reference']
        start = time.perf_counter()
        g = greedy_decode(lp)
        greedy_ms = (time.perf_counter() - start) * 1000
        beam = r['ctcdecode_text'] if is_b0 else svr_text
        case = {
            'id': r['id'], 'text': split[r['id']]['text'], 'reference': ref,
            'greedy': g, 'beam_lm': beam,
            'wer_greedy': wer(ref, g), 'cer_greedy': cer(ref, g),
            'wer_beam_lm': wer(ref, beam), 'cer_beam_lm': cer(ref, beam),
            'n_frames': r['n_frames'], 'duration_s': r['duration_s'],
            'latency_ms': {'preprocess': r['preprocess_ms'], 'forward': r['forward_ms'], 'greedy_decode': greedy_ms,
                           'beam_lm_decode': r['ctcdecode_ms'] if is_b0 else svr_ms},
        }
        if is_b0:
            case['beam_lm_decoder'] = 'ctcdecode'
            case['beam_lm_svr'] = svr_text
            case['wer_beam_lm_svr'] = wer(ref, svr_text)
            case['latency_ms']['beam_lm_svr_decode'] = svr_ms
        else:
            case['beam_lm_decoder'] = 'svr.decoders.BeamLMDecoder'
        cases.append(case)
        refs.append(ref)
        greedy_hyps.append(g)
        beam_hyps.append(beam)
        svr_hyps.append(svr_text)
    metrics = {'n': len(cases), 'greedy': corpus_rates(refs, greedy_hyps), 'beam_lm': corpus_rates(refs, beam_hyps)}
    if is_b0:
        metrics['beam_lm_svr_decoder'] = corpus_rates(refs, svr_hyps)
    latency = {k: latency_stats([c['latency_ms'][k] for c in cases]) for k in cases[0]['latency_ms']}
    write(model, 'open_vocab__%s.json' % channels, {
        'model': model, 'benchmark': 'open_vocab', 'channels': channels, 'channel_config': meta['channel_config'],
        'split': 'bench/splits/open_vocab_test.json',
        'metric_notes': 'Corpus WER/CER = total errors / total reference words (chars), as jiwer.wer(list, list) '
                        "in Gaddy's evaluation. Per-case rates are per sentence.",
        'metrics': metrics, 'latency_ms': latency, 'cases': cases})
    return metrics, latency, meta


def score_phrase_id(model, channels, raw_split, trials_name):
    meta, logits = load_raw(model, raw_split, channels)
    split = {it['id']: it for it in load_split(raw_split.split('@')[0])['items']}
    trials = load_split(trials_name)
    rows = {r['id']: r for r in meta['items']}
    cases, correct, top3, ranks = [], 0, 0, []
    skipped = 0
    for t in trials['trials']:
        if len(t['candidates']) < 2:  # no look-alike exists for any word: nothing to judge
            skipped += 1
            continue
        lp = logits[npz_key(t['id'])]
        start = time.perf_counter()
        provided = rows[t['id']].get('candidate_scores', {}).get(trials_name)
        scores = provided if provided is not None else score_phrases(lp, t['candidates'])
        score_ms = (time.perf_counter() - start) * 1000
        if provided is not None:
            score_ms = rows[t['id']].get('scoring_ms', score_ms)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        rank = order.index(t['answer_index']) + 1
        best_other = max(s for i, s in enumerate(scores) if i != t['answer_index'])
        cases.append({
            'id': t['id'], 'text': split[t['id']]['text'], 'reference': split[t['id']]['reference'],
            'candidates': [{'text': c, 'log_likelihood': s} for c, s in zip(t['candidates'], scores)],
            'answer_index': t['answer_index'], 'predicted_index': order[0], 'predicted': t['candidates'][order[0]],
            'correct': rank == 1, 'true_rank': rank, 'margin': scores[t['answer_index']] - best_other,
            'greedy': greedy_decode(lp),
            'latency_ms': {'preprocess': rows[t['id']]['preprocess_ms'], 'forward': rows[t['id']]['forward_ms'],
                           'phrase_scoring': score_ms},
        })
        correct += rank == 1
        top3 += rank <= 3
        ranks.append(rank)
    n = len(cases)
    metrics = {'n': n, 'skipped_single_candidate': skipped, 'n_candidates': trials['n_candidates'],
               'mean_candidates': float(np.mean([len(c['candidates']) for c in cases])),
               'chance': float(np.mean([1 / len(c['candidates']) for c in cases])),
               'top1_accuracy': correct / n, 'top3_accuracy': top3 / n, 'mean_true_rank': float(np.mean(ranks))}
    latency = {k: latency_stats([c['latency_ms'][k] for c in cases]) for k in cases[0]['latency_ms']}
    bench_name = trials_name + ('@' + raw_split.split('@')[1] if '@' in raw_split else '')
    write(model, '%s__%s.json' % (bench_name, channels), {
        'model': model, 'benchmark': bench_name, 'channels': channels, 'channel_config': meta['channel_config'],
        'split': 'bench/splits/%s.json' % trials_name, 'description': trials['description'],
        'metric_notes': 'Each candidate is scored by its CTC log-likelihood under the model (natural log); '
                        'the highest score is the prediction. margin = true score - best wrong score.',
        'metrics': metrics, 'latency_ms': latency, 'cases': cases})
    return metrics, latency


def score_model(model, processes):
    summary = {'model': model, 'benchmarks': {}}
    raw_dir = os.path.join(RESULTS_DIR, model, 'raw')
    available = [ch for ch in CHANNEL_CONFIGS
                 if os.path.exists(os.path.join(raw_dir, 'open_vocab_test__%s.json' % ch))]
    for channels in available:
        ov_metrics, ov_latency, meta = score_open_vocab(model, channels, processes)
        summary['benchmarks']['open_vocab__' + channels] = {'metrics': ov_metrics, 'latency_ms': ov_latency}
        for raw_split, trials in CANDIDATE_BENCHMARKS:
            if not (os.path.exists(os.path.join(raw_dir, '%s__%s.json' % (raw_split, channels)))
                    and os.path.exists(os.path.join(SPLITS_DIR, trials + '.json'))):
                continue
            m, lat = score_phrase_id(model, channels, raw_split, trials)
            name = trials + ('@' + raw_split.split('@')[1] if '@' in raw_split else '')
            summary['benchmarks']['%s__%s' % (name, channels)] = {'metrics': m, 'latency_ms': lat}
        summary.setdefault('latency_3s_clip_forward_ms', {})[channels] = meta['latency_3s_clip_forward_ms']
    for key in ('description', 'checkpoint', 'checkpoint_md5', 'device', 'environment', 'parameters', 'ctcdecode'):
        if key in meta:
            summary[key] = meta[key]
    summary['channel_configs'] = CHANNEL_CONFIGS
    summary['channel_mapping_note'] = ('Electrode site N (Gaddy 2020, Table 3) is assumed to be data channel N-1. '
                                       '4ch keeps sites 1, 2, 3, 6 and zeroes the other channels. For a model '
                                       'trained on 8 channels, 4ch shows the loss, not a 4ch model.')
    write(model, 'summary.json', summary)


def main():
    processes = max(1, (os.cpu_count() or 2) - 2)
    for model in sys.argv[1:]:
        score_model(model, processes)


if __name__ == '__main__':
    main()
