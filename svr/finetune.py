"""Fine-tune on our wearer's recordings (day 2) and benchmark on the held-out ones.

    venv/bin/python -m svr.finetune --name W_svr1 --init-from checkpoints/SVR1_full/model.pt
    venv/bin/python -m svr.finetune --name W_svr1_cal --init-from checkpoints/SVR1_full/model.pt --mode calibrate
    venv/bin/python -m svr.finetune --name W_gaddy6 --arch gaddy --init-from checkpoints/G6_baseline/model.pt

Data (svr/recordings.py manifest): 6-channel EMG in slots 0-5 of the 8 input slots, filtered and scaled exactly
like the Berkeley data. Splits:
    test (open vocabulary)   set E                                        never trained on
    test (fixed sentences)   sets A + B of the last session (or a seeded 20% of them with one session)
    dev                      a seeded 10% of the remaining labelled training utterances (model selection,
                             the "none" threshold of the Gaddy baseline, the camera+EMG fusion weight)
    train                    everything else: A, B (-> "none"), C, D, F
Fixed sentences: the distinct prompts of --fixed-sets (default A, D); set B (rest / junk) is the "none" class.

SVR-1 losses: letter CTC, phoneme CTC, fixed-sentence classifier (+ none), judge loss (true sentence vs lip
look-alikes, vs the other fixed sentences, and vs the camera's own Auto-AVSR n-best when CAM_raw exists), and the
closed-lips transfer loss [contribution F]: a mouth-closed utterance's letter posteriors are pulled (KL) towards the
DTW-aligned posteriors of a mouth-open utterance of the same sentence (teacher side detached).
--mode calibrate trains only the session adapter, the fixed-sentence head and the score weights (demo-day refit).
--arch gaddy fine-tunes a Gaddy-architecture checkpoint with letter CTC only (the baseline); its fixed-sentence pick
is CTC phrase scoring with a "none" threshold tuned on dev.

Writes checkpoints/<name>/{model.pt, log.jsonl, config.json} and benchmark_results/WEARER_<name>/
{summary.json, cases_*.json}.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import jiwer
import numpy as np
import torch
import torch.nn.functional as F

from svr import augment
from svr.decoders import greedy_decode, score_phrases
from svr.phonemes import PHONE_BLANK, sentence_phonemes
from svr.precompute_svr1 import dtw_path, energy_features
from svr.preprocess import highpass, notch_harmonics, prepare_emg
from svr.recordings import load_manifest, spoken_wpm
from svr.text import BLANK, clean_text, text_to_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
RESULTS = os.path.join(WORK, 'benchmark_results')
CAM_NBEST = os.path.join(RESULTS, 'CAM_raw', 'auto_avsr.json')
FRAME_RATE = 86.13


# ------------------------------------------------------------------------------------------------ data
def wearer_input(r):
    """Model input (frames*8, 8) of one recording: notch + high-pass, resample, scale; 6 channels -> slots 0-5."""
    emg = np.load(r['emg']).astype(np.float64)
    x = prepare_emg(highpass(notch_harmonics(emg)))
    out = np.zeros((len(x), 8), np.float32)
    out[:, :x.shape[1]] = x
    return out


class WearerData:
    def __init__(self, records, fixed):
        self.r = records
        self.inputs = [wearer_input(r) for r in records]
        self.clean = [clean_text(r.get('prompt') or '') for r in records]
        self.targets = [np.array(text_to_ids(c), np.int64) for c in self.clean]
        self.phones = [sentence_phonemes(c) if c else None for c in self.clean]
        index = {clean_text(p): i for i, p in enumerate(fixed)}
        self.label = [len(fixed) if r['set'] == 'B' else index.get(c, -1) for r, c in zip(records, self.clean)]
        self.lengths = [len(x) for x in self.inputs]

    def __len__(self):
        return len(self.r)

    def batches(self, rng, budget):
        order = list(range(len(self)))
        rng.shuffle(order)
        chunks = [sorted(order[i:i + 32], key=lambda k: self.lengths[k]) for i in range(0, len(order), 32)]
        out, cur, longest = [], [], 0
        for k in (k for c in chunks for k in c):
            new = max(longest, self.lengths[k])
            if cur and new * (len(cur) + 1) > budget:
                out.append(cur)
                cur, new = [], self.lengths[k]
            cur.append(k)
            longest = new
        if cur:
            out.append(cur)
        rng.shuffle(out)
        return out


def make_splits(records, seed):
    rng = random.Random(seed)
    test_e = [r for r in records if r['set'] == 'E']
    ab = [r for r in records if r['set'] in ('A', 'B')]
    sessions = sorted({r['session'] for r in ab})
    if len(sessions) > 1:
        test_fixed = [r for r in ab if r['session'] == sessions[-1]]
    else:
        test_fixed = rng.sample(ab, len(ab) // 5)
    held = {r['id'] for r in test_e + test_fixed}
    rest = [r for r in records if r['id'] not in held and (r.get('prompt') or r['set'] == 'B')]
    rng.shuffle(rest)
    n_dev = max(1, len(rest) // 10)
    return rest[n_dev:], rest[:n_dev], test_e, test_fixed


def camera_nbest(path=CAM_NBEST):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v['nbest'] for k, v in json.load(f)['utterances'].items()}


def transfer_pairs(data):
    """(closed index, open twin index, DTW match) for closed-mouth utterances whose sentence was also mouthed open."""
    open_by_text = {}
    for k, r in enumerate(data.r):
        if r.get('mouth') == 'open' and not r.get('voiced') and data.clean[k]:
            open_by_text.setdefault(data.clean[k], k)
    pairs = []
    for k, r in enumerate(data.r):
        j = open_by_text.get(data.clean[k])
        if r.get('mouth') == 'closed' and j is not None:
            pairs.append((k, j, dtw_path(energy_features(data.inputs[k]), energy_features(data.inputs[j]))))
    return pairs


# ------------------------------------------------------------------------------------------------ models
class GaddyWrap(torch.nn.Module):
    """Gaddy-architecture model behind SVR-1's forward signature (one utterance at a time; no padding support)."""

    def __init__(self, path):
        super().__init__()
        from svr.model import load_gaddy_checkpoint
        self.m = load_gaddy_checkpoint(path)

    def forward(self, x, lengths, channel_mask=None):
        assert x.shape[0] == 1
        if channel_mask is not None:
            x = x * channel_mask.to(x.dtype).reshape(1, 1, -1)
        return {'letter_logp': F.log_softmax(self.m(x).float(), -1), 'lengths': lengths}


def build_model(args, n_fixed):
    if args.arch == 'gaddy':
        return GaddyWrap(args.init_from)
    from svr.svr1 import build
    st = torch.load(args.init_from, map_location='cpu', weights_only=False)
    cfg = dict(st['config'], n_fixed=n_fixed, attention='sdpa')
    model = build(cfg)
    state = {k: v for k, v in st['model'].items() if not k.startswith(('fixed.', 'fixed_query'))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('initialised from', args.init_from, 'new:', missing, 'ignored:', unexpected, flush=True)
    return model


# ------------------------------------------------------------------------------------------------ training
def train_step(model, data, idx, rng, args, cmask, device, maker, fixed_clean, cam, pairs):
    is_svr1 = args.arch == 'svr1'
    xs, factors = [], []
    for k in idx:
        x, f = data.inputs[k], 1.0
        if not args.no_speed and rng.random() < 0.5:
            f = float(np.exp(rng.uniform(np.log(0.8), np.log(1.5))))
            cand = augment.speed(x, f)
            if len(data.targets[k]) == 0 or len(cand) // 8 >= 2 * len(data.targets[k]) + 1:
                x = cand
            else:
                f = 1.0
        if not args.no_undershoot and f > 1.1 and rng.random() < 0.5:
            x = augment.undershoot(x, rng)
        xs.append(augment.time_mask(x, rng))
        factors.append(f)
    lengths = torch.tensor([len(x) // 8 for x in xs])
    xb = np.zeros((len(xs), max(len(x) for x in xs), 8), np.float32)
    for i, x in enumerate(xs):
        xb[i, :len(x)] = x
    losses = {}
    if not is_svr1:  # Gaddy model: one utterance per forward, letter CTC only
        terms = []
        for i, k in enumerate(idx):
            if len(data.targets[k]):
                out = model(torch.from_numpy(xb[i:i + 1, :len(xs[i])]).to(device), lengths[i:i + 1], cmask)
                terms.append(F.ctc_loss(out['letter_logp'].transpose(0, 1), torch.from_numpy(data.targets[k])[None].to(device),
                                        lengths[i:i + 1], torch.tensor([len(data.targets[k])]), blank=BLANK,
                                        zero_infinity=True))
        if terms:
            losses['letter'] = torch.stack(terms).mean()
        return losses
    from svr.svr1 import INFEASIBLE, score_texts
    lengths_d = lengths.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        out = model(torch.from_numpy(xb).to(device), lengths_d, cmask)
    rows = [i for i, k in enumerate(idx) if len(data.targets[k])]
    if rows:
        tg = [torch.from_numpy(data.targets[idx[i]]) for i in rows]
        losses['letter'] = F.ctc_loss(out['letter_logp'][rows].transpose(0, 1), torch.cat(tg).to(device),
                                      lengths[rows], torch.tensor([len(t) for t in tg]), blank=BLANK,
                                      zero_infinity=True)
    prow = [i for i in rows if data.phones[idx[i]]]
    if prow and args.mode == 'full':
        pt = [torch.tensor(data.phones[idx[i]]) for i in prow]
        losses['phone'] = F.ctc_loss(out['phone_logp'][prow].transpose(0, 1), torch.cat(pt).to(device),
                                     lengths[prow], torch.tensor([len(t) for t in pt]), blank=PHONE_BLANK,
                                     zero_infinity=True)
    lab = torch.tensor([data.label[k] for k in idx])
    if 'fixed_logits' in out and (lab >= 0).any():
        sel = (lab >= 0).nonzero(as_tuple=True)[0]
        losses['fixed'] = F.cross_entropy(out['fixed_logits'][sel.to(device)], lab[sel].to(device))
    if not args.no_judge:
        terms = []
        for i in rng.permutation(rows)[:args.judge_rows] if rows else []:
            k = idx[i]
            true = data.clean[k]
            negs = []
            if data.label[k] >= 0 and data.label[k] < len(fixed_clean):
                negs += [c for c in fixed_clean if c != true]
            negs += [clean_text(h['text']) for h in cam.get(data.r[k]['id'], [])[:args.judge_k]]
            negs += maker(true, args.judge_k, rng)[1:]
            cands = [true] + list(dict.fromkeys(n for n in negs if n and n != true))[:2 * args.judge_k]
            if len(cands) < 2:
                continue
            s = score_texts(model, out, int(i), cands, device)
            if s[0] > INFEASIBLE / 2:
                terms.append(F.cross_entropy(s[None], torch.zeros(1, dtype=torch.long, device=device)))
        if terms:
            losses['judge'] = torch.stack(terms).mean()
    if pairs and not args.no_transfer:
        terms = []
        for p in rng.permutation(len(pairs))[:args.transfer_rows]:
            kc, ko, match = pairs[p]
            xc = torch.from_numpy(data.inputs[kc])[None].to(device)
            xo = torch.from_numpy(data.inputs[ko])[None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                lc = model(xc, torch.tensor([xc.shape[1] // 8], device=device), cmask)['letter_logp'][0]
                with torch.no_grad():
                    lo = model(xo, torch.tensor([xo.shape[1] // 8], device=device), cmask)['letter_logp'][0]
            teacher = lo[torch.from_numpy(match).to(device)].float()
            terms.append(F.kl_div(lc.float(), teacher, log_target=True, reduction='batchmean'))
        losses['transfer'] = torch.stack(terms).mean()
    return losses


WEIGHTS = {'letter': 1.0, 'phone': 0.5, 'fixed': 1.0, 'judge': 0.5, 'transfer': 0.5}


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def forward_all(model, data, cmask, device):
    model.eval()
    outs = []
    for k in range(len(data)):
        x = torch.from_numpy(data.inputs[k])[None].to(device)
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model(x, torch.tensor([x.shape[1] // 8], device=device), cmask)
        if device == 'cuda':
            torch.cuda.synchronize()
        outs.append((out, (time.perf_counter() - t0) * 1000))
    return outs


def phrase_scores(model, out, cands, args, device):
    """S_emg for SVR-1 (CTC + verifier), plain letter-CTC log-likelihood for the Gaddy model."""
    if args.arch == 'svr1':
        from svr.svr1 import score_texts
        return score_texts(model, out, 0, cands, device).detach().float().cpu().numpy()
    return np.nan_to_num(np.array(score_phrases(out['letter_logp'][0].cpu().numpy(), cands)), neginf=-1e4)


def dev_wer(model, data, cmask, device):
    outs = forward_all(model, data, cmask, device)
    pairs = [(data.clean[k], greedy_decode(o['letter_logp'][0].float().cpu().numpy()))
             for k, (o, _) in enumerate(outs) if data.clean[k]]
    model.train()
    return jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 1.0


@torch.no_grad()
def none_threshold(model, dev, fixed_clean, args, cmask, device):
    """Gaddy baseline: per-frame CTC log-likelihood below which the pick is "none" (best dev accuracy)."""
    outs = forward_all(model, dev, cmask, device)
    rows = []
    for k, (o, _) in enumerate(outs):
        if dev.label[k] >= 0:
            s = phrase_scores(model, o, fixed_clean, args, device)
            rows.append((float(s.max()) / max(1, int(o['lengths'][0])), int(s.argmax()), dev.label[k]))
    if not rows:
        return -math.inf
    n = len(fixed_clean)

    def acc(t):
        return np.mean([(n if s < t else a) == y for s, a, y in rows])
    grid = sorted({r[0] for r in rows}) + [math.inf]
    return max([-math.inf] + grid, key=acc)


@torch.no_grad()
def evaluate(model, args, tests, dev, fixed_clean, cmask, device, maker):
    test_e, test_fixed = tests
    summary, cases = {}, {}
    # 1. open vocabulary (set E): WER/CER overall, by pace and by mouth; effective words per minute
    if len(test_e):
        outs = forward_all(model, test_e, cmask, device)
        rows = []
        for k, (o, ms) in enumerate(outs):
            r = test_e.r[k]
            dur = len(test_e.inputs[k]) / 8 / FRAME_RATE
            hyp = greedy_decode(o['letter_logp'][0].float().cpu().numpy())
            rows.append({'id': r['id'], 'reference': test_e.clean[k], 'greedy': hyp, 'mouth': r.get('mouth'),
                         'target_wpm': r.get('target_wpm'), 'spoken_wpm': spoken_wpm(r, dur), 'forward_ms': ms})

        def wer_of(sub):
            return {'n': len(sub), 'wer': jiwer.wer([c['reference'] for c in sub], [c['greedy'] for c in sub]),
                    'cer': jiwer.cer([c['reference'] for c in sub], [c['greedy'] for c in sub]),
                    'spoken_wpm': float(np.mean([c['spoken_wpm'] for c in sub])),
                    'forward_ms_median': float(np.median([c['forward_ms'] for c in sub]))}
        summary['open_vocab_E'] = {'all': wer_of(rows)}
        for key in ('target_wpm', 'mouth'):
            for v in sorted({str(c[key]) for c in rows}):
                summary['open_vocab_E'][f'{key}={v}'] = wer_of([c for c in rows if str(c[key]) == v])
        # 2. homophene judge: true sentence vs 9 lip look-alikes (fixed seed)
        rng = np.random.default_rng(7)
        correct = n = 0
        for k, (o, _) in enumerate(outs):
            cands = maker(test_e.clean[k], 9, rng)
            if len(cands) > 1:
                s = phrase_scores(model, o, cands, args, device)
                correct += int(np.argmax(s) == 0)
                n += 1
                rows[k]['homophene_correct'] = bool(np.argmax(s) == 0)
        summary['homophene10_E'] = {'n': n, 'top1': correct / max(n, 1)}
        # 3. camera + EMG: rerank Auto-AVSR's n-best with S_emg (weight tuned on dev utterances that have an n-best)
        cam = camera_nbest(args.camera_nbest)
        if cam and any(r['id'] in cam for r in test_e.r):
            dev_outs = forward_all(model, dev, cmask, device)

            def picks(data, outs_, w):
                hyps, refs = [], []
                for k, (o, _) in enumerate(outs_):
                    nb = cam.get(data.r[k]['id'])
                    if not nb or not data.clean[k]:
                        continue
                    texts = [clean_text(h['text']) for h in nb]
                    s_cam = np.array([h['score'] for h in nb], np.float64)
                    s_emg = phrase_scores(model, o, texts, args, device)
                    hyps.append(texts[int(np.argmax(s_cam + w * s_emg))])
                    refs.append(data.clean[k])
                return refs, hyps
            grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
            tune = [(w, picks(dev, dev_outs, w)) for w in grid]
            tune = [(w, jiwer.wer(rh[0], rh[1])) for w, rh in tune if rh[0]]
            w = min(tune, key=lambda t: t[1])[0] if tune else 0.3
            refs, hyps = picks(test_e, outs, w)
            refs0, hyps0 = picks(test_e, outs, 0.0)
            summary['camera_nbest_E'] = {'n': len(refs), 'emg_weight': w, 'wer_camera_top1': jiwer.wer(refs0, hyps0),
                                         'wer_camera_plus_emg': jiwer.wer(refs, hyps)}
        cases['open_vocab_E'] = rows
    # 4. fixed sentences (+ none), held-out A/B, by mouth
    if len(test_fixed):
        outs = forward_all(model, test_fixed, cmask, device)
        thr = none_threshold(model, dev, fixed_clean, args, cmask, device) if args.arch == 'gaddy' else None
        n_fixed = len(fixed_clean)
        rows = []
        for k, (o, ms) in enumerate(outs):
            y = test_fixed.label[k]
            s = phrase_scores(model, o, fixed_clean, args, device)
            if 'fixed_logits' in o:
                pred = int(o['fixed_logits'][0].argmax())
            else:
                pred = n_fixed if float(s.max()) / max(1, int(o['lengths'][0])) < thr else int(s.argmax())
            rows.append({'id': test_fixed.r[k]['id'], 'mouth': test_fixed.r[k].get('mouth'), 'label': y,
                         'pred': pred, 'phrase_scoring_pred': int(s.argmax()), 'forward_ms': ms})

        def acc_of(sub):
            real = [c for c in sub if c['label'] < n_fixed]
            junk = [c for c in sub if c['label'] == n_fixed]
            return {'n': len(sub), 'accuracy': float(np.mean([c['pred'] == c['label'] for c in sub])),
                    'sentence_accuracy': float(np.mean([c['pred'] == c['label'] for c in real])) if real else None,
                    'phrase_scoring_accuracy': float(np.mean([c['phrase_scoring_pred'] == c['label'] for c in real]))
                    if real else None,
                    'none_recall': float(np.mean([c['pred'] == n_fixed for c in junk])) if junk else None,
                    'false_none_rate': float(np.mean([c['pred'] == n_fixed for c in real])) if real else None}
        summary['fixed_sentences'] = {'all': acc_of(rows), 'n_classes': n_fixed + 1}
        for v in sorted({str(c['mouth']) for c in rows}):
            summary['fixed_sentences'][f'mouth={v}'] = acc_of([c for c in rows if str(c['mouth']) == v])
        if thr is not None:
            summary['fixed_sentences']['none_threshold_per_frame'] = thr
        cases['fixed_sentences'] = rows
    return summary, cases


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--init-from', required=True)
    p.add_argument('--arch', default='svr1', choices=['svr1', 'gaddy'])
    p.add_argument('--mode', default='full', choices=['full', 'calibrate'])
    p.add_argument('--recordings', default=None, help='default: recordings/')
    p.add_argument('--fixed-sets', default='A,D')
    p.add_argument('--camera-nbest', default=CAM_NBEST, help='Auto-AVSR n-best JSON (camera.run_camera_baselines)')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--micro-batch-samples', type=int, default=96000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--new-lr-mult', type=float, default=10.0, help='lr multiplier of new layers (adapter, fixed head)')
    p.add_argument('--judge-k', type=int, default=8)
    p.add_argument('--judge-rows', type=int, default=4)
    p.add_argument('--transfer-rows', type=int, default=2)
    for flag in ('no-judge', 'no-transfer', 'no-speed', 'no-undershoot'):
        p.add_argument('--' + flag, action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-only', action='store_true', help='benchmark the init model without fine-tuning')
    p.add_argument('--out-root', default=os.path.join(WORK, 'checkpoints'))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    records = load_manifest(**({'root': args.recordings} if args.recordings else {}))
    if not records:
        sys.exit('no recordings found (expected recordings/<session>/manifest.jsonl)')
    fixed_sets = set(args.fixed_sets.split(','))
    fixed = sorted({r['prompt'] for r in records if r['set'] in fixed_sets and r.get('prompt')})
    fixed_clean = [clean_text(p) for p in fixed]
    train_r, dev_r, test_e_r, test_fixed_r = make_splits(records, args.seed)
    print(f'{len(fixed)} fixed sentences; train {len(train_r)}  dev {len(dev_r)}  test E {len(test_e_r)}  '
          f'test fixed {len(test_fixed_r)}', flush=True)
    train, dev = WearerData(train_r, fixed), WearerData(dev_r, fixed)
    tests = (WearerData(test_e_r, fixed), WearerData(test_fixed_r, fixed))
    from svr.train_svr1 import CandidateMaker
    maker = CandidateMaker(extra_texts=[r['prompt'] for r in train_r if r.get('prompt')])
    cam = camera_nbest(args.camera_nbest)
    pairs = transfer_pairs(train) if args.arch == 'svr1' and not args.no_transfer else []
    print(f'camera n-best for {sum(r["id"] in cam for r in train_r)} training utterances; '
          f'{len(pairs)} closed/open transfer pairs', flush=True)

    model = build_model(args, len(fixed)).to(device)
    cmask = torch.zeros(8, device=device)
    cmask[:6] = 1
    new = ('adapter.', 'fixed.', 'fixed_query', 'mix')
    if args.mode == 'calibrate':
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(n_.startswith(new))
    params_new = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and n_.startswith(new)]
    params_old = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and not n_.startswith(new)]
    optim = torch.optim.AdamW([{'params': params_old, 'lr': args.lr},
                               {'params': params_new, 'lr': args.lr * args.new_lr_mult}], weight_decay=0.01)
    base_lrs = [g['lr'] for g in optim.param_groups]
    rng = np.random.default_rng(args.seed)
    n_steps = args.epochs * len(train.batches(np.random.default_rng(0), args.micro_batch_samples))
    best, best_state, step = math.inf, copy.deepcopy(model.state_dict()), 0
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), fixed_sentences=fixed, n_train=len(train), n_dev=len(dev), steps=n_steps), f,
                  indent=1)
    model.train()
    for epoch in range(0 if args.eval_only else args.epochs):
        t0, ep_losses = time.time(), []
        for idx in train.batches(rng, args.micro_batch_samples):
            losses = train_step(model, train, idx, rng, args, cmask, device, maker, fixed_clean, cam, pairs)
            if not losses:
                continue
            loss = sum(WEIGHTS[k] * v for k, v in losses.items())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = min(1.0, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / max(1, n_steps))))
            for g, lr in zip(optim.param_groups, base_lrs):
                g['lr'] = lr * scale
            optim.step()
            step += 1
            ep_losses.append({k: float(v.detach()) for k, v in losses.items()})
        wer = dev_wer(model, dev, cmask, device)
        rec = {'epoch': epoch + 1, 'step': step, 'dev_wer_greedy': wer, 'seconds': time.time() - t0}
        for k in WEIGHTS:
            vals = [d[k] for d in ep_losses if k in d]
            if vals:
                rec['train_' + k] = float(np.mean(vals))
        with open(os.path.join(out_dir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
        if wer <= best:
            best, best_state = wer, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    if args.arch == 'svr1':
        torch.save({'config': model.config_dict(), 'model': model.state_dict(), 'keep_channels': list(range(6)),
                    'fixed_sentences': fixed}, os.path.join(out_dir, 'model.pt'))
    else:
        torch.save(model.m.state_dict(), os.path.join(out_dir, 'model.pt'))
    summary, cases = evaluate(model, args, tests, dev, fixed_clean, cmask, device, maker)
    summary = {'model': args.name, 'arch': args.arch, 'mode': args.mode, 'init_from': args.init_from,
               'best_dev_wer_greedy': best, 'device': device, 'benchmarks': summary}
    res = os.path.join(RESULTS, f'WEARER_{args.name}')
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for k, v in cases.items():
        with open(os.path.join(res, f'cases_{k}.json'), 'w') as f:
            json.dump(v, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
