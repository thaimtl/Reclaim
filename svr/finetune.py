"""Fine-tune SVR-1 on Austin's recordings (stage 2 of docs/SVR1_SPEC.md section 4) and benchmark on session 3.

    venv-svr/bin/python -m svr.finetune --name W_svr1 --init-from checkpoints/SVR1_4ch/model.pt
    venv-svr/bin/python -m svr.finetune --name W_svr1_cal --init-from checkpoints/W_svr1/model.pt --mode calibrate \
        --train-sessions session_04
    venv-svr/bin/python -m svr.finetune --name W_g4 --arch gaddy --init-from checkpoints/G4/model.pt

Data: recorder/record.py's format (svr/recordings.py, RECORDING.md). Each clip's raw Uno readings (samples, 4: pins
A0..A3 = Gaddy sites 3, 2, 6, 1) go through rig/filters.py in order, into model slots [2, 1, 5, 0], are resampled to
689.06 Hz, multiplied by the session's gain (Berkeley's spread / this session's robust spread, per slot) and soft
clipped: the same path as pretraining and the live loop (svr.preprocess, svr.live). Splits:
    test    set E (session 3, demo pace, 10 smile + 10 frown)                     never trained on
    dev     a seeded 10% of the last training session's sentences                early stopping (EMA weights)
    train   everything else in --train-sessions: set C sentences, and set B rest clips as speaking negatives
Clips whose recorder warnings say a channel was flat or pinned, or the EMG was short, are skipped (--keep-warned).

SVR-1 losses (svr.train_svr1.train_losses, so pretraining and fine-tuning cannot drift apart): letter CTC (rest clips:
all blank), phoneme CTC, rate, speaking BCE (forced alignments with the EMA model at the start and every
--align-every epochs; rest clips all 0), partial-sentence cuts, and the judge loss, whose negatives are lip look-alikes
plus the camera's own wrong guesses (Auto-AVSR's n-best on each clip's recorded video, when CAM_raw exists).
--mode calibrate trains only the session adapter for --calib-steps steps (demo-day refit; the session gain is
recomputed from the calibration clips). --arch gaddy fine-tunes a Gaddy-architecture checkpoint (G4) with letter CTC
only, the comparison baseline.

Writes checkpoints/<name>/{model.pt, log.jsonl, config.json} and benchmark_results/WEARER_<name>/
{summary.json, cases_*.json}. model.pt also holds what svr.live needs: the session gains, the scale target and the
slot map.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import jiwer
import numpy as np
import torch
import torch.nn.functional as F

from svr import augment
from svr.decoders import greedy_decode, score_phrases
from svr.phonemes import sentence_phonemes
from svr.preprocess import KEEP, RIG_TO_SLOT, apply_gain, rig_gain, rig_unscaled, robust_spread
from svr.recordings import load_manifest, spoken_wpm
from svr.text import BLANK, clean_text, text_to_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
RESULTS = os.path.join(WORK, 'benchmark_results')
CAM_NBEST = os.path.join(RESULTS, 'CAM_raw', 'auto_avsr.json')
FRAME_RATE = 86.13
BAD_WARNINGS = ('flat', 'pinned', 'EMG short')


# ------------------------------------------------------------------------------------------------ data
def usable_record(r, keep_warned=False):
    if not r.get('emg') or not os.path.exists(r['emg']):
        return False
    return keep_warned or not any(b in w for w in r.get('warnings') or [] for b in BAD_WARNINGS)


def session_gains(unscaled_by_record, records, target):
    """{session: per-slot gain} from the robust spread of all of that session's clips (SVR1_SPEC section 1)."""
    by_session = {}
    for r in records:
        by_session.setdefault(r['session'], []).append(unscaled_by_record[r['id']][::2])
    return {s: rig_gain(robust_spread(np.concatenate(xs)), target) for s, xs in by_session.items()}


class WearerData:
    """Austin's clips with the interface svr.train_svr1.make_batch / train_losses expect."""

    def __init__(self, records, unscaled, gains):
        self.r = records
        self.inputs = [apply_gain(unscaled[r['id']], gains[r['session']], KEEP) for r in records]
        self.clean = [' '.join(clean_text(r.get('prompt') or '').split()) for r in records]
        self.targets = [np.array(text_to_ids(c), np.int64) for c in self.clean]
        self.phones = [sentence_phonemes(c) if c else None for c in self.clean]
        self.artic = [None] * len(records)  # no articulator targets for silent clips
        self.align = [None] * len(records)
        self.speak_neg = [r.get('set') == 'B' or not c for r, c in zip(records, self.clean)]
        self.lengths = [len(x) for x in self.inputs]
        self.usable = list(range(len(records)))

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


def effort_refs(data):
    """{session: median 100 ms RMS of the speaking frames}: voice.say's effort = 1.0 (svr.live). Speaking frames come
    from the forced alignments; clips without one count from 0.5 s in to 0.5 s before the end."""
    from svr.align import speak_target
    from svr.live import frame_rms
    by_session = {}
    for k, r in enumerate(data.r):
        if not data.clean[k]:
            continue
        rms = frame_rms(data.inputs[k])
        if data.align[k] is not None:
            on = speak_target(data.align[k], len(rms)) > 0
        else:
            on = np.zeros(len(rms), bool)
            on[int(0.5 * FRAME_RATE):max(0, len(rms) - int(0.5 * FRAME_RATE))] = True
        by_session.setdefault(r['session'], []).append(rms[on])
    return {s: float(np.median(np.concatenate(v))) for s, v in by_session.items() if sum(map(len, v))}


def make_splits(records, train_sessions, seed):
    """train, dev (10% of the last training session's sentences), test (set E)."""
    rng = random.Random(seed)
    test = [r for r in records if r.get('set') == 'E']
    train = [r for r in records if r['session'] in train_sessions and r.get('set') != 'E']
    last = max(train_sessions, key=lambda s: (any(r['session'] == s for r in train), s))
    pool = sorted((r for r in train if r['session'] == last and r.get('prompt')), key=lambda r: r['id'])
    dev_ids = {r['id'] for r in rng.sample(pool, max(1, len(pool) // 10))} if len(pool) > 1 else set()
    return [r for r in train if r['id'] not in dev_ids], [r for r in train if r['id'] in dev_ids], test


def camera_nbest(path=CAM_NBEST):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: v['nbest'] for k, v in json.load(f)['utterances'].items()}


class WearerNegatives:
    """Judge-loss negatives: the camera's own n-best for this sentence (from any clip of it), then lip look-alikes."""

    def __init__(self, maker, cam, data):
        self.maker = maker
        self.by_text = {}
        for r, c in zip(data.r, data.clean):
            for h in cam.get(r['id'], []):
                t = ' '.join(clean_text(h['text']).split())
                if t and t != c:
                    self.by_text.setdefault(c, []).append(t)

    def __call__(self, clean, k, rng):
        negs = list(dict.fromkeys(self.by_text.get(clean, [])))[:k] + self.maker(clean, k, rng)[1:]
        return [clean] + list(dict.fromkeys(n for n in negs if n and n != clean))[:2 * k]


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


def load_init(path):
    st = torch.load(path, map_location='cpu', weights_only=False)
    return st if isinstance(st, dict) and 'config' in st else {}


def build_model(args, init):
    if args.arch == 'gaddy':
        return GaddyWrap(args.init_from)
    from svr.svr1 import build, config_of
    cfg = dict(config_of(init), n_fixed=0, attention='sdpa', use_speak=True)
    model = build(cfg)
    state = {k: v for k, v in init['model'].items() if not k.startswith(('fixed.', 'fixed_query'))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('initialised from', args.init_from, 'new:', missing, 'ignored:', unexpected, flush=True)
    return model


def gaddy_losses(model, data, idx, rng, args, cmask, device):
    """Letter CTC, one utterance per forward, speed augmentation only (the G4 baseline's recipe)."""
    from svr.train_svr1 import ctc_feasible
    terms = []
    for k in idx:
        x, ids = data.inputs[k], data.targets[k]
        if not args.no_speed and rng.random() < 0.5:
            cand = augment.speed(x, float(np.exp(rng.uniform(np.log(0.8), np.log(1.5)))))
            x = cand if ctc_feasible(len(cand) // 8, ids) else x
        x = torch.from_numpy(augment.time_mask(x, rng))[None].to(device)
        n = torch.tensor([x.shape[1] // 8])
        out = model(x, n, cmask)
        terms.append(F.ctc_loss(out['letter_logp'].transpose(0, 1), torch.from_numpy(ids)[None].to(device), n,
                                torch.tensor([len(ids)]), blank=BLANK, zero_infinity=True))
    return {'letter': torch.stack(terms).mean()} if terms else {}


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def forward_all(model, data, cmask, device):
    was_training = model.training
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
    model.train(was_training)
    return outs


def phrase_scores(model, out, cands, args, device):
    """S_emg for SVR-1 (CTC + verifier), plain letter-CTC log-likelihood for the Gaddy model."""
    if args.arch == 'svr1':
        from svr.svr1 import score_texts
        return score_texts(model, out, 0, cands, device).detach().float().cpu().numpy()
    return np.nan_to_num(np.array(score_phrases(out['letter_logp'][0].float().cpu().numpy(), cands)), neginf=-1e4)


def dev_wer(model, data, cmask, device):
    outs = forward_all(model, data, cmask, device)
    pairs = [(data.clean[k], greedy_decode(o['letter_logp'][0].float().cpu().numpy()))
             for k, (o, _) in enumerate(outs) if data.clean[k]]
    return jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 1.0


@torch.no_grad()
def evaluate(model, args, test, dev, cmask, device, maker):
    summary, cases = {}, {}
    rows = [k for k in range(len(test)) if test.clean[k]]
    if not rows:
        return summary, cases
    outs = forward_all(model, test, cmask, device)
    # 1. open vocabulary (set E): WER/CER overall and by delivery (pace, expression, effort; RECORDING.md)
    table = []
    for k in rows:
        o, ms = outs[k]
        r = test.r[k]
        dur = len(test.inputs[k]) / 8 / FRAME_RATE
        row = {'id': r['id'], 'reference': test.clean[k], 'greedy': greedy_decode(o['letter_logp'][0].float().cpu().numpy()),
               'pace': r.get('pace'), 'expression': r.get('expression'), 'effort': r.get('effort'),
               'spoken_wpm': spoken_wpm(r, dur), 'forward_ms': ms}
        if 'speak_logit' in o:  # share of frames the speaking head calls speech (clips include ~0.5 s still each end)
            row['speaking_share'] = float(torch.sigmoid(o['speak_logit'][0]).gt(0.5).float().mean())
        table.append(row)

    def wer_of(sub):
        return {'n': len(sub), 'wer': jiwer.wer([c['reference'] for c in sub], [c['greedy'] for c in sub]),
                'cer': jiwer.cer([c['reference'] for c in sub], [c['greedy'] for c in sub]),
                'spoken_wpm': float(np.mean([c['spoken_wpm'] for c in sub])),
                'forward_ms_median': float(np.median([c['forward_ms'] for c in sub]))}
    summary['open_vocab_E'] = {'all': wer_of(table)}
    for key in ('pace', 'expression', 'effort'):
        for v in sorted({str(c[key]) for c in table}):
            summary['open_vocab_E'][f'{key}={v}'] = wer_of([c for c in table if str(c[key]) == v])
    # 2. homophene judge: true sentence vs 9 lip look-alikes (fixed seed)
    rng = np.random.default_rng(7)
    correct = n = 0
    for c, k in zip(table, rows):
        cands = maker(test.clean[k], 9, rng)
        if len(cands) > 1:
            s = phrase_scores(model, outs[k][0], cands, args, device)
            c['homophene_correct'] = bool(np.argmax(s) == 0)
            correct += int(c['homophene_correct'])
            n += 1
    summary['homophene10_E'] = {'n': n, 'top1': correct / max(n, 1)}
    # 3. camera + EMG: rerank Auto-AVSR's n-best with S_emg (weight tuned on dev clips that have an n-best)
    cam = camera_nbest(args.camera_nbest)
    if cam and any(test.r[k]['id'] in cam for k in rows):
        dev_outs = forward_all(model, dev, cmask, device)

        def picks(data, outs_, w):
            hyps, refs = [], []
            for k, (o, _) in enumerate(outs_):
                nb = cam.get(data.r[k]['id'])
                if not nb or not data.clean[k]:
                    continue
                texts = [' '.join(clean_text(h['text']).split()) for h in nb]
                s_cam = np.array([h['score'] for h in nb], np.float64)
                s_emg = phrase_scores(model, o, texts, args, device)
                hyps.append(texts[int(np.argmax(s_cam + w * s_emg))])
                refs.append(data.clean[k])
            return refs, hyps
        grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
        tune = [(w, picks(dev, dev_outs, w)) for w in grid]
        tune = [(w, jiwer.wer(rh[0], rh[1])) for w, rh in tune if rh[0]]
        w = min(tune, key=lambda t: t[1])[0] if tune else 0.3
        refs, hyps = picks(test, outs, w)
        refs0, hyps0 = picks(test, outs, 0.0)
        summary['camera_nbest_E'] = {'n': len(refs), 'emg_weight': w, 'wer_camera_top1': jiwer.wer(refs0, hyps0),
                                     'wer_camera_plus_emg': jiwer.wer(refs, hyps)}
    cases['open_vocab_E'] = table
    return summary, cases


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--init-from', required=True)
    p.add_argument('--arch', default='svr1', choices=['svr1', 'gaddy'])
    p.add_argument('--mode', default='full', choices=['full', 'calibrate'])
    p.add_argument('--recordings', default=None, help='default: $RECLAIM_WORKDIR/recordings')
    p.add_argument('--train-sessions', default='session_01,session_02')
    p.add_argument('--keep-warned', action='store_true', help='also train on clips with recorder warnings')
    p.add_argument('--camera-nbest', default=CAM_NBEST, help='Auto-AVSR n-best JSON (camera.run_camera_baselines)')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--calib-steps', type=int, default=50)
    p.add_argument('--micro-batch-samples', type=int, default=96000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--new-lr-mult', type=float, default=10.0, help='lr multiplier of the session adapter')
    p.add_argument('--ema', type=float, default=0.998)
    p.add_argument('--judge-k', type=int, default=8)
    p.add_argument('--judge-rows', type=int, default=4)
    p.add_argument('--align-every', type=int, default=10)
    p.add_argument('--prefix-p', type=float, default=0.3)
    p.add_argument('--slow-p', type=float, default=0.1)
    p.add_argument('--effort-p', type=float, default=0.3)
    p.add_argument('--expression-p', type=float, default=0.1)
    for flag in ('no-judge', 'no-speed', 'no-undershoot', 'no-phone', 'no-speak', 'no-prefix', 'no-delivery'):
        p.add_argument('--' + flag, action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-only', action='store_true', help='benchmark the init model without fine-tuning')
    p.add_argument('--out-root', default=os.path.join(WORK, 'checkpoints'))
    return p.parse_args(argv)


def loss_args(args):
    """The settings svr.train_svr1.make_batch / train_losses read: real rig data, so no hardware simulation, no
    channel dropout, no articulator targets."""
    return SimpleNamespace(**vars(args), no_hwsim=True, no_artic=True, channel_dropout=0.0,
                           w_phone=0.5, w_artic=0.0, w_rate=0.1, w_speak=0.2, w_judge=0.5)


def main(argv=None):
    args = parse_args(argv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    records = load_manifest(**({'root': args.recordings} if args.recordings else {}))
    if not records:
        sys.exit('no recordings found (expected recordings/session_0N/manifest.jsonl)')
    n_all = len(records)
    records = [r for r in records if usable_record(r, args.keep_warned)]
    train_sessions = args.train_sessions.split(',')
    train_r, dev_r, test_r = make_splits(records, train_sessions, args.seed)
    print(f'{n_all} clips, {len(records)} usable; train {len(train_r)} (rest {sum(r["set"] == "B" for r in train_r)})  '
          f'dev {len(dev_r)}  test {len(test_r)}', flush=True)
    if not train_r and not args.eval_only:
        sys.exit(f'no training clips in {train_sessions}')

    init = load_init(args.init_from)
    target = init.get('scale_target')
    if target is None:
        from svr.data import scale_stats
        target = scale_stats()['target']
    used = train_r + dev_r + test_r
    unscaled = {r['id']: rig_unscaled(np.load(r['emg'])) for r in used}
    gains = session_gains(unscaled, used, target)
    print('session gains:', {s: [round(float(g[c]), 3) for c in KEEP] for s, g in gains.items()}, flush=True)
    train, dev, test = WearerData(train_r, unscaled, gains), WearerData(dev_r, unscaled, gains), \
        WearerData(test_r, unscaled, gains)
    from svr.train_svr1 import LOSS_WEIGHTS, CandidateMaker, compute_alignments, score_parts, train_losses
    maker = CandidateMaker(extra_texts=[r['prompt'] for r in train_r if r.get('prompt')])
    cam = camera_nbest(args.camera_nbest)
    negatives = WearerNegatives(maker, cam, train)
    print(f'camera n-best for {sum(r["id"] in cam for r in train_r)} training clips', flush=True)

    model = build_model(args, init).to(device)
    cmask = torch.zeros(8, device=device)
    cmask[KEEP] = 1
    is_svr1 = args.arch == 'svr1'
    new = ('adapter.',)
    if args.mode == 'calibrate':
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(n_.startswith(new) if is_svr1 else n_.startswith('m.'))
    params_new = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and n_.startswith(new)]
    params_old = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and not n_.startswith(new)]
    optim = torch.optim.AdamW([{'params': params_old, 'lr': args.lr},
                               {'params': params_new, 'lr': args.lr * args.new_lr_mult}], weight_decay=0.01)
    base_lrs = [g['lr'] for g in optim.param_groups]
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = len(train.batches(np.random.default_rng(0), args.micro_batch_samples)) if len(train) else 0
    n_steps = args.calib_steps if args.mode == 'calibrate' else args.epochs * steps_per_epoch
    epochs = 0 if args.eval_only else (math.ceil(n_steps / max(1, steps_per_epoch)) if args.mode == 'calibrate'
                                       else args.epochs)
    la = loss_args(args)
    parts = score_parts(SimpleNamespace(no_phone=args.no_phone, no_verifier=False, no_judge=args.no_judge))
    best, best_state, step = math.inf, copy.deepcopy(ema.state_dict()), 0
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), n_train=len(train), n_dev=len(dev), n_test=len(test), steps=n_steps,
                       session_gains={s: g.tolist() for s, g in gains.items()}), f, indent=1)
    model.train()
    for epoch in range(epochs):
        t0, ep_losses = time.time(), []
        if is_svr1 and not (args.no_speak and args.no_prefix) and epoch % args.align_every == 0:
            train.align, n_ok = compute_alignments(ema, train, KEEP, device, args.micro_batch_samples)
            print(f'epoch {epoch + 1}: forced alignments for {n_ok}/{sum(len(t) > 0 for t in train.targets)} clips',
                  flush=True)
        for idx in train.batches(rng, args.micro_batch_samples):
            if step >= n_steps:
                break
            if is_svr1:
                losses, _ = train_losses(model, train, idx, rng, la, KEEP, device, negatives, parts,
                                         judge=not args.no_judge, prefix_p=0.0 if args.no_prefix else args.prefix_p)
            else:
                losses = gaddy_losses(model, train, idx, rng, args, cmask, device)
            if not losses:
                continue
            loss = sum(LOSS_WEIGHTS[k](la) * v for k, v in losses.items())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = min(1.0, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / max(1, n_steps))))
            for g, lr in zip(optim.param_groups, base_lrs):
                g['lr'] = lr * scale
            optim.step()
            step += 1
            with torch.no_grad():
                decay = min(args.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            ep_losses.append({k: float(v.detach()) for k, v in losses.items()})
        wer = dev_wer(ema, dev, cmask, device) if len(dev) else math.nan
        rec = {'epoch': epoch + 1, 'step': step, 'dev_wer_greedy': wer, 'seconds': time.time() - t0}
        for k in ('letter', 'phone', 'rate', 'speak', 'judge'):
            vals = [d[k] for d in ep_losses if k in d]
            if vals:
                rec['train_' + k] = float(np.mean(vals))
        with open(os.path.join(out_dir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
        if not len(dev) or wer <= best:
            best, best_state = wer, copy.deepcopy(ema.state_dict())
    ema.load_state_dict(best_state)
    if is_svr1:
        torch.save({'config': ema.config_dict(), 'model': ema.state_dict(), 'keep_channels': KEEP,
                    'preprocess': 'causal', 'scale_target': list(map(float, target)), 'rig_to_slot': RIG_TO_SLOT,
                    'session_gains': {s: g.tolist() for s, g in gains.items()}, 'effort_ref': effort_refs(train),
                    'default_session': max(r['session'] for r in train_r) if train_r else None},
                   os.path.join(out_dir, 'model.pt'))
    else:
        torch.save(ema.m.state_dict(), os.path.join(out_dir, 'model.pt'))
    eval_maker = CandidateMaker(extra_texts=[r['prompt'] for r in used if r.get('prompt')])
    summary, cases = evaluate(ema, args, test, dev, cmask, device, eval_maker)
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
