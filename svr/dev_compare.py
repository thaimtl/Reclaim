"""Compare SVR-1 checkpoints on the fine-tuning dev split only (session 3 stays unseen): step 3 of the plan.

    venv-svr/bin/python -m svr.dev_compare checkpoints/W_s12_svr1_4ch/model.pt checkpoints/W_s12_svr1_4ch_artic/model.pt \
        checkpoints/SVR1_4ch/model.pt checkpoints/SVR1_4ch_artic/model.pt

The dev split is exactly svr.finetune's: the same clip filter (--max-filled), the same seed, 10% of the last training
session's sentences. Each checkpoint is scored on:
    WER / CER        greedy transcription, EMG alone
    look-alike top-1 the true sentence against 9 lip look-alikes, scored by S_emg (letter + phoneme CTC + verifier):
                     SVR-1's job in the joint search. Each clip gets --sets different look-alike sets (fixed seeds), so
                     20 dev clips give 100 trials; chance is 10%. The interval is a 95% Wilson interval.
Every checkpoint sees the same inputs: the session gains are computed from the training sessions' clips, as svr.finetune
does. Writes benchmark_results/DEV_COMPARE/summary.json.
"""
import argparse
import json
import math
import os

import jiwer
import numpy as np
import torch

from svr.decoders import greedy_decode
from svr.finetune import (RESULTS, WearerData, build_model, forward_all, load_init, make_splits, phrase_scores,
                          session_gains, usable_record)
from svr.preprocess import KEEP, rig_unscaled
from svr.recordings import load_manifest


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--train-sessions', default='session_01,session_02')
    p.add_argument('--max-filled', type=float, default=0.05)
    p.add_argument('--sets', type=int, default=5, help='look-alike sets per dev clip')
    p.add_argument('--seed', type=int, default=0, help="svr.finetune's split seed")
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    records = [r for r in load_manifest() if usable_record(r, False, args.max_filled)]
    train_r, dev_r, _ = make_splits(records, args.train_sessions.split(','), args.seed)
    used = train_r + dev_r
    unscaled = {r['id']: rig_unscaled(np.load(r['emg'])) for r in used}
    from svr.train_svr1 import CandidateMaker
    maker = CandidateMaker(extra_texts=[r['prompt'] for r in used if r.get('prompt')])
    cmask = torch.zeros(8, device=device)
    cmask[KEEP] = 1
    print(f'dev: {len(dev_r)} clips from {sorted({r["session"] for r in dev_r})}; look-alike trials per model: '
          f'{len(dev_r) * args.sets}', flush=True)
    results = {}
    for ckpt in args.checkpoints:
        init = load_init(ckpt)
        gains = session_gains(unscaled, used, init['scale_target'])
        dev = WearerData(dev_r, unscaled, gains)
        model = build_model(argparse.Namespace(arch='svr1', init_from=ckpt), init).to(device).eval()
        outs = forward_all(model, dev, cmask, device)
        refs = [dev.clean[k] for k in range(len(dev)) if dev.clean[k]]
        hyps = [greedy_decode(o['letter_logp'][0].float().cpu().numpy()) for k, (o, _) in enumerate(outs) if dev.clean[k]]
        correct = n = 0
        ranks = []
        for k, (o, _) in enumerate(outs):
            if not dev.clean[k]:
                continue
            for s in range(args.sets):
                cands = maker(dev.clean[k], 9, np.random.default_rng(1000 * s + k))
                if len(cands) < 2:
                    continue
                sc = phrase_scores(model, o, cands, argparse.Namespace(arch='svr1'), device)
                correct += int(np.argmax(sc) == 0)
                ranks.append(int((sc > sc[0]).sum()) + 1)
                n += 1
        lo, hi = wilson(correct, n)
        results[ckpt] = {'wer': jiwer.wer(refs, hyps), 'cer': jiwer.cer(refs, hyps), 'lookalike_top1': correct / max(n, 1),
                         'lookalike_ci95': [lo, hi], 'lookalike_trials': n, 'lookalike_mean_rank': float(np.mean(ranks)),
                         'chance_top1': 0.1}
        r = results[ckpt]
        print(f'{ckpt}: WER {100 * r["wer"]:.1f}%  CER {100 * r["cer"]:.1f}%  look-alike top-1 {100 * r["lookalike_top1"]:.0f}% '
              f'(95% {100 * lo:.0f}-{100 * hi:.0f}%, mean rank {r["lookalike_mean_rank"]:.2f} of ~10, {n} trials)',
              flush=True)
    out = os.path.join(RESULTS, 'DEV_COMPARE')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({'dev_clips': [r['id'] for r in dev_r], 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
