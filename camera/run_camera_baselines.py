"""Camera-only baselines on the wearer's recordings: accuracy vs speaking speed, latency, effective WPM.

    venv-camera/bin/python -m camera.run_camera_baselines            # everything
    venv-camera/bin/python -m camera.run_camera_baselines --skip-llm # lip readers only

Systems (all on identical mouth crops from camera.mouth):
    CAM_autoavsr            Auto-AVSR top-1                       (the model inside Lipflow and Chaplin)
    CAM_autoavsr_llmclean   Auto-AVSR top-5 -> local LLM cleanup  (Lipflow's design: camera + LLM)
    CAM_autoavsr_llmrescore Auto-AVSR top-20 reranked by camera score + w * LLM log-prob; w tuned on set C
    CAM_braven              BRAVEn top-1                          (second camera model)
    CAM_liplearner_fixed    LipLearner 5-shot on the fixed sentences, mouth open and mouth closed

Data: free-speech sets C (paced, used for tuning) and E (held out, the honest numbers); fixed sentences set A.
Metrics per utterance and per pace bucket (target ~100 / ~150 / 180+ WPM):
    WER, CER (after the same text normalization as the EMG benchmarks)
    spoken WPM      = prompt words / speech duration
    latency_ms      = encoder + beam search (+ LLM), measured per utterance after a warm-up
    effective WPM   = prompt words / (speech duration + latency)
    entry WPM       = (characters of the output - 1) / (duration + latency) x 60 / 5   (MELDER's text-entry formula)
Writes benchmark_results/CAM_<system>/{summary.json, cases_<set>.json} and benchmark_results/CAMERA_BASELINES.md.
"""
import argparse
import json
import os
import subprocess
import sys

import jiwer
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
sys.path.insert(0, ROOT)
from svr.recordings import load_manifest, speech_window, spoken_wpm  # noqa: E402
from svr.text import clean_text  # noqa: E402

RESULTS = os.path.join(WORK, 'benchmark_results')
RAW = os.path.join(RESULTS, 'CAM_raw')
CACHE = os.path.join(WORK, 'data', 'camera_cache')
FREE_SETS = ('C', 'E')
PACES = ((0, 125, '~100 WPM'), (125, 165, '~150 WPM'), (165, 10 ** 6, '180+ WPM'))


def pace_bucket(r):
    t = r.get('target_wpm')
    if t is None:
        return 'unpaced'
    return next(name for lo, hi, name in PACES if lo <= t < hi)


def norm(text):
    return ' '.join(clean_text(text).split())


def run_worker(system, ids, out):
    ids_file = out + '.ids'
    with open(ids_file, 'w') as f:
        f.write('\n'.join(ids))
    subprocess.run([sys.executable, '-m', 'camera.vsr_worker', '--system', system, '--ids', ids_file, '--out', out],
                   cwd=ROOT, check=True)


def utterance_duration(r):
    d = np.load(os.path.join(CACHE, f"{r['id']}.npz"))
    clip = float(d['t'][-1] - d['t'][0]) if len(d['t']) > 1 else 0.0
    t0, t1 = speech_window(r, clip)
    return t1 - t0, float(d['preprocess_ms'])


def case(r, hyp, latency_ms, extra=None):
    ref = norm(r['prompt'])
    hyp = norm(hyp)
    dur, pre_ms = utterance_duration(r)
    words = len(ref.split())
    total = dur + latency_ms / 1000
    out = {'id': r['id'], 'set': r['set'], 'session': r['session'], 'mouth': r.get('mouth'),
           'target_wpm': r.get('target_wpm'), 'pace': pace_bucket(r), 'reference': ref, 'hypothesis': hyp,
           'wer': jiwer.wer(ref, hyp) if ref else None, 'cer': jiwer.cer(ref, hyp) if ref else None,
           'speech_seconds': dur, 'spoken_wpm': spoken_wpm(r, dur), 'latency_ms': latency_ms,
           'preprocess_ms': pre_ms, 'effective_wpm': 60.0 * words / total if total > 0 else 0.0,
           'entry_wpm': max(len(hyp) - 1, 0) / total * 60 / 5 if total > 0 else 0.0}
    if extra:
        out.update(extra)
    return out


def summarize(cases):
    def block(cs):
        if not cs:
            return None
        refs = [c['reference'] for c in cs]
        hyps = [c['hypothesis'] for c in cs]
        return {'n': len(cs), 'wer': jiwer.wer(refs, hyps), 'cer': jiwer.cer(refs, hyps),
                'spoken_wpm_median': float(np.median([c['spoken_wpm'] for c in cs])),
                'effective_wpm_median': float(np.median([c['effective_wpm'] for c in cs])),
                'entry_wpm_median': float(np.median([c['entry_wpm'] for c in cs])),
                'latency_ms_median': float(np.median([c['latency_ms'] for c in cs]))}
    out = {}
    for s in FREE_SETS:
        cs = [c for c in cases if c['set'] == s]
        out[s] = {'all': block(cs), **{name: block([c for c in cs if c['pace'] == name]) for _, _, name in PACES}}
    return out


def write_system(name, description, cases, extra=None):
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    for s in sorted({c['set'] for c in cases}):
        with open(os.path.join(d, f'cases_{s}.json'), 'w') as f:
            json.dump([c for c in cases if c['set'] == s], f, indent=1)
    summary = {'system': name, 'description': description, 'by_set': summarize(cases), **(extra or {})}
    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', name)
    return summary


def latency(row):
    return row['encode_ms'] + row['decode_ms']


def liplearner_fixed(records, emb_file, shots=5):
    """Few-shot nearest-centroid over the fixed sentences: enroll `shots` open-mouth reps per sentence from the
    first session, test on every other rep, mouth open and mouth closed."""
    d = np.load(emb_file)
    emb = {u: e for u, e in zip(d['ids'], d['emb'])}
    enc_ms = {u: t for u, t in zip(d['ids'], d['encode_ms'])}
    first = sorted({r['session'] for r in records})[0]
    by_sentence = {}
    for r in records:
        by_sentence.setdefault(norm(r['prompt']), []).append(r)
    centroids, enrolled = {}, set()
    for sent, rs in by_sentence.items():
        pool = [r for r in rs if r['session'] == first and r.get('mouth') == 'open'][:shots]
        if pool:
            c = np.mean([emb[r['id']] for r in pool], 0)
            centroids[sent] = c / np.linalg.norm(c)
            enrolled |= {r['id'] for r in pool}
    names = sorted(centroids)
    mat = np.stack([centroids[n] for n in names])
    cases = []
    for r in records:
        if r['id'] in enrolled or norm(r['prompt']) not in centroids:
            continue
        sims = mat @ emb[r['id']]
        pred = names[int(sims.argmax())]
        cases.append({'id': r['id'], 'session': r['session'], 'mouth': r.get('mouth'), 'reference': norm(r['prompt']),
                      'predicted': pred, 'correct': pred == norm(r['prompt']), 'latency_ms': float(enc_ms[r['id']])})
    acc = {m: (float(np.mean([c['correct'] for c in cases if c['mouth'] == m])) if any(c['mouth'] == m for c in cases)
               else None) for m in ('open', 'closed')}
    return cases, {'shots': shots, 'n_sentences': len(names), 'chance': 1 / max(len(names), 1),
                   'top1_accuracy_mouth_open': acc['open'], 'top1_accuracy_mouth_closed': acc['closed']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--systems', default='auto_avsr,braven,liplearner')
    p.add_argument('--skip-llm', action='store_true')
    p.add_argument('--llm', default=None, help='HF model id; default camera.llm.DEFAULT_MODEL')
    p.add_argument('--recordings', default=None, help='recordings root (default: <repo>/recordings)')
    a = p.parse_args()
    root = {'root': a.recordings} if a.recordings else {}
    os.makedirs(RAW, exist_ok=True)
    systems = a.systems.split(',')

    free = [r for r in load_manifest(sets=FREE_SETS, **root) if r.get('video') and r.get('prompt')]
    fixed = [r for r in load_manifest(sets=('A',), **root) if r.get('video') and r.get('prompt')]
    missing = [r['id'] for r in free + fixed if not os.path.exists(os.path.join(CACHE, f"{r['id']}.npz"))]
    if missing:
        subprocess.run([sys.executable, '-m', 'camera.mouth', '--sets', 'A,C,E']
                       + (['--recordings', a.recordings] if a.recordings else []), cwd=ROOT, check=True)
    print(f'{len(free)} free-speech utterances (sets C, E), {len(fixed)} fixed-sentence utterances (set A)')

    summaries = {}
    raw = {}
    for system in [s for s in systems if s in ('auto_avsr', 'braven')]:
        out = os.path.join(RAW, f'{system}.json')
        run_worker(system, [r['id'] for r in free], out)
        with open(out) as f:
            raw[system] = json.load(f)['utterances']
        name = {'auto_avsr': 'CAM_autoavsr', 'braven': 'CAM_braven'}[system]
        cases = [case(r, raw[system][r['id']]['nbest'][0]['text'], latency(raw[system][r['id']])) for r in free]
        summaries[name] = write_system(name, f'{system} top-1, camera only', cases)

    if 'auto_avsr' in raw and not a.skip_llm:
        from camera.llm import DEFAULT_MODEL, LocalLLM
        llm = LocalLLM(a.llm or DEFAULT_MODEL)
        llm.cleanup(['warm up'])
        clean_cases, rescored = [], {}
        for r in free:
            row = raw['auto_avsr'][r['id']]
            texts = [h['text'] for h in row['nbest']]
            fixed_text, llm_ms = llm.cleanup(texts[:5])
            clean_cases.append(case(r, fixed_text, latency(row) + llm_ms, {'candidates': texts[:5]}))
            scores, score_ms = llm.rescore(texts)
            rescored[r['id']] = (texts, [h['score'] for h in row['nbest']], scores, latency(row) + score_ms)
        summaries['CAM_autoavsr_llmclean'] = write_system(
            'CAM_autoavsr_llmclean', f'Auto-AVSR top-5 cleaned by {llm.model_id} with Lipflow\'s prompt', clean_cases)

        def pick(uid, w):
            texts, cam, lm, _ = rescored[uid]
            return texts[int(np.argmax(np.asarray(cam) + w * np.asarray(lm)))]
        grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
        tune = [r for r in free if r['set'] == 'C'] or free
        best_w = min(grid, key=lambda w: jiwer.wer([norm(r['prompt']) for r in tune],
                                                    [norm(pick(r['id'], w)) for r in tune]))
        cases = [case(r, pick(r['id'], best_w), rescored[r['id']][3], {'llm_weight': best_w}) for r in free]
        summaries['CAM_autoavsr_llmrescore'] = write_system(
            'CAM_autoavsr_llmrescore', f'Auto-AVSR top-20 reranked: camera score + {best_w} x {llm.model_id} log-prob '
            '(weight tuned on set C)', cases, {'llm_weight': best_w})

    if 'liplearner' in systems and fixed:
        out = os.path.join(RAW, 'liplearner.npz')
        run_worker('liplearner', [r['id'] for r in fixed], out)
        cases, metrics = liplearner_fixed(fixed, out)
        d = os.path.join(RESULTS, 'CAM_liplearner_fixed')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'cases_A.json'), 'w') as f:
            json.dump(cases, f, indent=1)
        with open(os.path.join(d, 'summary.json'), 'w') as f:
            json.dump({'system': 'CAM_liplearner_fixed', 'metrics': metrics}, f, indent=1)
        summaries['CAM_liplearner_fixed'] = {'metrics': metrics}
        print('wrote CAM_liplearner_fixed', metrics)

    report(summaries)


def report(summaries):
    lines = ['# Camera-only baselines', '', 'Generated by `camera/run_camera_baselines.py`. Set E is held out; set C '
             'was used to tune the rescoring weight.', '',
             '| System | Set | WER all | WER ~100 | WER ~150 | WER 180+ | spoken WPM | latency ms | effective WPM |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    fmt = lambda b, k, scale=100: '-' if not b else ('%.1f' % (b[k] * scale) if scale != 1 else '%.0f' % b[k])
    for name, s in summaries.items():
        if 'by_set' not in s:
            continue
        for st, blocks in s['by_set'].items():
            al = blocks['all']
            if not al:
                continue
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, st, fmt(al, 'wer'), fmt(blocks['~100 WPM'], 'wer'), fmt(blocks['~150 WPM'], 'wer'),
                fmt(blocks['180+ WPM'], 'wer'), fmt(al, 'spoken_wpm_median', 1), fmt(al, 'latency_ms_median', 1),
                fmt(al, 'effective_wpm_median', 1)))
    if 'CAM_liplearner_fixed' in summaries:
        m = summaries['CAM_liplearner_fixed']['metrics']
        lines += ['', f"LipLearner {m['shots']}-shot on {m['n_sentences']} fixed sentences: mouth open "
                      f"{m['top1_accuracy_mouth_open']}, mouth closed {m['top1_accuracy_mouth_closed']} "
                      f"(chance {m['chance']:.2f})."]
    with open(os.path.join(RESULTS, 'CAMERA_BASELINES.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote CAMERA_BASELINES.md')


if __name__ == '__main__':
    main()
