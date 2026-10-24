"""Train SVR-1 on the Berkeley data (pretraining) with every contribution behind a switch for the ablation sweep.

    venv/bin/python -m svr.train_svr1 --name SVR1_full --keep-channels 0,1,2,3,4,5
    venv/bin/python -m svr.train_svr1 --name SVR1_8to6 --keep-channels 0,1,2,3,4,5,6,7 --channel-dropout 0.25
    venv/bin/python -m svr.train_svr1 --name SVR1_no_judge --keep-channels 0,1,2,3,4,5 --no-judge

Losses per micro-batch (each can be switched off):
    letter CTC                                   always
    phoneme CTC                                  --no-phone
    articulator regression (SPARC targets)       --no-artic       (needs svr.precompute_svr1)
    rate (predict log speed factor)              --no-rate        (model config use_rate)
    homophene-contrastive judge loss             --no-judge       contribution A: true sentence vs K lip look-alikes,
                                                                  scored by letter + phoneme CTC + verifier
    verifier                                     --no-verifier    contribution B (part of the judge loss)
Augmentation: speed 0.8-1.6x (--no-speed), undershoot (--no-undershoot), hardware simulation (--no-hwsim),
time masking, channel dropout (--channel-dropout p, for 8-channel pretraining that is later used with 6).

Batches: one utterance per row, padded; the padded size (rows x longest model-input length, 689 Hz) stays under
--micro-batch-samples, the knob cluster/smoke.sh tunes for VRAM. Everything random in a batch derives from
(seed, epoch, batch index), so a resumed run reproduces the uninterrupted one. Checkpointing, resume, SIGTERM and
Erbium monitoring are shared with svr/train.py. model.pt holds the EMA weights + config (svr.svr1.load).
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import signal
import string
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench.lookalikes import Lexicon, corpus_vocabulary, homophene_candidates  # noqa: E402
from svr import augment  # noqa: E402
from svr.data import DATA_ROOT, EMGTextDataset  # noqa: E402
from svr.decoders import greedy_decode  # noqa: E402
from svr.phonemes import PHONE_BLANK, sentence_phonemes  # noqa: E402
from svr.svr1 import INFEASIBLE, build, score_texts  # noqa: E402
from svr.text import BLANK, clean_text, text_to_ids  # noqa: E402
from svr.train import (EXIT_INTERRUPTED, atomic_save, load_checkpoint, log_line, save_checkpoint,  # noqa: E402
                       start_resource_monitor)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
TARGETS = os.path.join(WORK, 'data', 'svr1_targets')
RATE_CENTER, RATE_SCALE = 2.34, 0.37  # log letters/s over Berkeley train (mean, std): the rate head's target scale
LOSS_WEIGHTS = {'letter': lambda a: 1.0, 'phone': lambda a: a.w_phone, 'artic': lambda a: a.w_artic,
                'rate': lambda a: a.w_rate, 'judge': lambda a: a.w_judge}
RESUME_KEYS = ('keep_channels', 'micro_batch_samples', 'grad_accum', 'lr', 'epochs', 'seed', 'model')


# ------------------------------------------------------------------------------------------------ data
class TrainData:
    """Berkeley inputs + letter/phoneme targets + optional articulator targets."""

    def __init__(self, split, use_artic):
        self.ds = EMGTextDataset(split)
        # single spaces: a double space in the true sentence but not in its look-alikes would be a judge shortcut
        self.clean = [' '.join(clean_text(t).split()) for t in self.ds.texts]
        self.targets = [np.array(text_to_ids(c), np.int64) for c in self.clean]
        self.phones = [sentence_phonemes(c) for c in self.clean]
        self.artic = [None] * len(self.ds)
        self.artic_stats = None
        if use_artic and not os.path.exists(os.path.join(TARGETS, 'stats.json')):
            sys.exit('articulator targets missing (data/svr1_targets/stats.json): run svr.precompute_svr1 sparc + '
                     'align, or pass --no-artic')
        if use_artic:
            with open(os.path.join(TARGETS, 'stats.json')) as f:
                st = json.load(f)
            self.artic_stats = (np.array(st['mean'], np.float32), np.array(st['std'], np.float32))
            for k, (s, i, _) in enumerate(self.ds.entries):
                p = os.path.join(TARGETS, 'artic', s, f'{i}.npy')
                if os.path.exists(p):
                    self.artic[k] = ((np.load(p) - self.artic_stats[0]) / self.artic_stats[1]).astype(np.float32)
        self.usable = [k for k, t in enumerate(self.ds.texts) if any(ch in string.ascii_letters for ch in t)]
        self.lengths = [len(x) for x in self.ds.inputs]  # model-input samples (689 Hz), what padding costs

    def batches(self, seed, epoch, budget, chunk=1024):
        """Shuffle, sort within chunks of 1024 by length, fill micro-batches up to a padded-size budget, never
        mixing two chunks (chunks of 64 without a flush padded batches to ~52% efficiency; this gives ~93%)."""
        rng = random.Random(seed * 100003 + epoch)
        order = list(self.usable)
        rng.shuffle(order)
        lens = self.lengths
        chunks = [sorted(order[i:i + chunk], key=lambda k: lens[k]) for i in range(0, len(order), chunk)]
        batches, cur, longest = [], [], 0
        for chunk_ in chunks:
            if cur:
                batches.append(cur)
                cur, longest = [], 0
            for k in chunk_:
                new_longest = max(longest, lens[k])
                if cur and new_longest * (len(cur) + 1) > budget:
                    batches.append(cur)
                    cur, new_longest = [], lens[k]
                cur.append(k)
                longest = new_longest
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches


def ctc_feasible(frames, ids):
    repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return frames >= len(ids) + repeats


def make_batch(data, idx, rng, args, keep, force_speed=None):
    """Augment and pad one micro-batch. Returns tensors (CPU) and the per-row speed factors.
    force_speed: use this factor for every row (the VRAM test uses the slowest, i.e. longest, 0.8)."""
    xs, factors, artic = [], [], []
    absent = [c for c in range(8) if c not in keep]
    for k in idx:
        x = data.ds.inputs[k]
        if absent:  # electrodes we will not have: zero them before hardware() mixes channels
            x = x.copy()
            x[:, absent] = 0
        y = data.artic[k]
        f = 1.0
        if force_speed or (not args.no_speed and rng.random() < 0.7):
            f = force_speed or float(np.exp(rng.uniform(np.log(0.8), np.log(1.6))))
            cand = augment.speed(x, f)
            if ctc_feasible(len(cand) // 8, data.targets[k]):
                x = cand
                if y is not None:
                    y = augment.speed(y.repeat(8, 0), f)[::8][:len(x) // 8] if len(y) else y
            else:
                f = 1.0
        if not args.no_undershoot and f > 1.1 and rng.random() < 0.6:
            x = augment.undershoot(x, rng)
        if not args.no_hwsim and rng.random() < 0.8:
            x = augment.hardware(x, rng)
        x = augment.time_mask(x, rng)
        xs.append(x)
        factors.append(f)
        artic.append(y)
    t8 = max(len(x) for x in xs)
    b = len(xs)
    xb = np.zeros((b, t8, 8), np.float32)
    for i, x in enumerate(xs):
        xb[i, :len(x), :x.shape[1]] = x
    lengths = torch.tensor([len(x) // 8 for x in xs])
    cmask = np.zeros((b, 8), np.float32)
    for i in range(b):
        cmask[i, augment.channel_drop(keep, rng, args.channel_dropout)] = 1
    tf = t8 // 8
    am = torch.zeros(b, tf, dtype=torch.bool)
    at = torch.zeros(b, tf, 13)
    for i, y in enumerate(artic):
        if y is not None and len(y):
            n = min(len(y), int(lengths[i]))
            at[i, :n] = torch.from_numpy(y[:n])
            am[i, :n] = True
    return torch.from_numpy(xb), lengths, torch.from_numpy(cmask), at, am, torch.tensor(factors)


# ------------------------------------------------------------------------------------------------ judge
def _cached(fn):
    cache = {}

    def wrapper(word):
        if word not in cache:
            cache[word] = fn(word)
        return cache[word]
    return wrapper


class CandidateMaker:
    """Lip look-alike negatives for the judge loss, from the Berkeley vocabulary (cached per word)."""

    def __init__(self, texts=None, extra_texts=()):
        """texts: sentences whose words may be substituted in (default: every Berkeley recording, for evaluation
        candidates; training passes the train sentences only so dev/test words never become negatives)."""
        if texts is None:
            texts = []
            for p in sorted(glob.glob(os.path.join(DATA_ROOT, '*', '*', '*_info.json'))):
                with open(p) as f:
                    t = json.load(f)['text']
                if t:
                    texts.append(t)
        texts = list(texts) + list(extra_texts)
        vocab = corpus_vocabulary(texts, clean_text)
        vocab = {w: c for w, c in vocab.items() if (len(w) > 1 or w in ('a', 'i')) and not w.isdigit()}
        self.lex = Lexicon(vocab)
        self.lex.lookalikes = _cached(self.lex.lookalikes)
        self.lex.near_lookalikes = _cached(self.lex.near_lookalikes)

    def __call__(self, clean, k, rng):
        negs, _ = homophene_candidates(clean, self.lex, k + 1, random.Random(int(rng.integers(1 << 30))))
        return [clean] + negs


def judge_scores(model, out, row, cands, device, parts):
    return score_texts(model, out, row, cands, device, parts)


def train_losses(model, train, idx, rng, args, keep, device, maker, parts, force_speed=None, judge=True):
    """Forward pass and every enabled loss for one micro-batch (shared by the trainer and the VRAM smoke test)."""
    xb, lengths, cmask, at, am, factors = make_batch(train, idx, rng, args, keep, force_speed)
    xb, lengths, cmask = xb.to(device, non_blocking=True), lengths.to(device), cmask.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(xb, lengths, cmask)
    losses = {}
    tg = [torch.tensor(train.targets[k]) for k in idx]
    losses['letter'] = F.ctc_loss(out['letter_logp'].transpose(0, 1), torch.cat(tg).to(device), lengths.cpu(),
                                  torch.tensor([len(t) for t in tg]), blank=BLANK, zero_infinity=True)
    if not args.no_phone:
        rows = [i for i, k in enumerate(idx) if train.phones[k]]
        if rows:
            pt = [torch.tensor(train.phones[idx[i]]) for i in rows]
            losses['phone'] = F.ctc_loss(out['phone_logp'][rows].transpose(0, 1), torch.cat(pt).to(device),
                                         lengths[rows].cpu(), torch.tensor([len(t) for t in pt]),
                                         blank=PHONE_BLANK, zero_infinity=True)
    if not args.no_artic and am.any():
        am_d = am.to(device)
        diff = (out['artic'] - at.to(device)) ** 2
        losses['artic'] = diff[am_d].mean()
    if model.cfg.use_rate:  # absolute speaking rate (letters/s after augmentation), not just the factor
        lps = torch.tensor([len(tg_) for tg_ in tg], dtype=torch.float32) / (lengths.cpu().float() / 86.13)
        losses['rate'] = F.mse_loss(out['rate'].float(), ((lps.log() - RATE_CENTER) / RATE_SCALE).to(device))
    if judge and not args.no_judge:
        rows = rng.permutation(len(idx))[:args.judge_rows]
        terms = []
        for i in rows:
            cands = maker(train.clean[idx[i]], args.judge_k, rng)
            if len(cands) < 2:
                continue
            s = judge_scores(model, out, int(i), cands, device, parts)
            if s[0] > INFEASIBLE / 2:
                terms.append(F.cross_entropy(s[None], torch.zeros(1, dtype=torch.long, device=device)))
        if terms:
            losses['judge'] = torch.stack(terms).mean()
    return losses


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, dev, dev_cands, device, keep, parts, limit=None):
    model.eval()
    refs, hyps, correct, n_judged = [], [], 0, 0
    cmask = torch.zeros(8, device=device)
    cmask[keep] = 1
    n = len(dev.ds) if limit is None else min(limit, len(dev.ds))
    for k in range(n):
        x = torch.from_numpy(dev.ds.inputs[k])[None].to(device)
        lengths = torch.tensor([x.shape[1] // 8], device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model(x, lengths, cmask)
        refs.append(dev.clean[k])
        hyps.append(greedy_decode(out['letter_logp'][0].cpu().numpy()))
        cands = dev_cands[k]
        if len(cands) > 1:
            s = judge_scores(model, out, 0, cands, device, parts)
            correct += int(bool(s[0] > s[1:].max()))  # strict: a tie (e.g. all infeasible) is a miss
            n_judged += 1
    model.train()
    return {'dev_wer_greedy': jiwer.wer(refs, hyps), 'dev_homophene10_top1': correct / max(n_judged, 1)}


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--keep-channels', default='0,1,2,3,4,5')
    p.add_argument('--channel-dropout', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--micro-batch-samples', type=int, default=112000,
                   help='padded model-input samples per micro-batch; 112000 -> ~405 steps/epoch (~61k steps), ~97%% real')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model', default='{}', help='JSON overrides of SVR1Config, e.g. {"d_model": 384}')
    p.add_argument('--judge-k', type=int, default=8, help='look-alike negatives per utterance')
    p.add_argument('--judge-rows', type=int, default=8, help='utterances per micro-batch that get a judge loss')
    p.add_argument('--w-phone', type=float, default=0.5)
    p.add_argument('--w-artic', type=float, default=0.5)
    p.add_argument('--w-rate', type=float, default=0.1)
    p.add_argument('--w-judge', type=float, default=0.5)
    p.add_argument('--judge-start', type=int, default=3000,
                   help='optimizer step where the judge loss starts (ranking with an untrained CTC is noise)')
    p.add_argument('--judge-ramp', type=int, default=3000, help='steps to ramp the judge weight from 0 to 1')
    for flag in ('no-phone', 'no-artic', 'no-rate', 'no-judge', 'no-verifier', 'no-speed', 'no-undershoot',
                 'no-hwsim'):
        p.add_argument('--' + flag, action='store_true')
    p.add_argument('--init-from', default=None, help='model.pt to start from (e.g. 8-channel pretraining)')
    p.add_argument('--attention', default='auto', choices=['auto', 'sdpa', 'fa2'])
    p.add_argument('--vram-gb', type=float, default=None, help='cap; default 90%% of the card')
    p.add_argument('--ckpt-minutes', type=float, default=15.0)
    p.add_argument('--monitor-seconds', type=float, default=15.0)
    p.add_argument('--eval-limit', type=int, default=None)
    p.add_argument('--max-micro-batches-per-epoch', type=int, default=None)
    p.add_argument('--crash-at-step', type=int, default=None)
    p.add_argument('--out-root', default=os.path.join(WORK, 'checkpoints'))
    return p.parse_args(argv)


def make_model(args):
    overrides = json.loads(args.model)
    if args.attention == 'auto':
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
        overrides.setdefault('attention', 'fa2' if 'H100' in gpu or 'H200' in gpu else 'sdpa')
    else:
        overrides['attention'] = args.attention
    overrides.setdefault('use_rate', not args.no_rate)
    overrides.setdefault('use_verifier', not args.no_verifier)
    return build(**overrides)


def score_parts(args):
    """Terms of S_emg. Without the judge loss the verifier and the mix weights are never trained, so the verifier
    is left out rather than adding a random logit to the --no-judge ablation's scores."""
    return tuple(p for p, off in (('letter', False), ('phone', args.no_phone),
                                  ('verifier', args.no_verifier or args.no_judge)) if not off)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.out_root, args.name)
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    os.makedirs(out_dir, exist_ok=True)
    keep = [int(c) for c in args.keep_channels.split(',')]
    start_resource_monitor(os.path.join(out_dir, 'erbium'), args.monitor_seconds)
    device = 'cuda'
    total_mem = torch.cuda.get_device_properties(0).total_memory
    cap = args.vram_gb * 2 ** 30 / total_mem if args.vram_gb else 0.9
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = make_model(args).to(device)
    if args.init_from:
        st = torch.load(args.init_from, map_location='cpu', weights_only=False)
        missing = model.load_state_dict(st['model'], strict=False)
        print('initialised from', args.init_from, missing, flush=True)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)

    train = TrainData('train', use_artic=not args.no_artic)
    dev = TrainData('dev', use_artic=False)
    maker = CandidateMaker(texts=[train.ds.texts[k] for k in train.usable])
    dev_rng = np.random.default_rng(12345)
    eval_maker = CandidateMaker()
    dev_cands = [eval_maker(c, 9, dev_rng) for c in dev.clean]
    parts = score_parts(args)

    no_decay = [p_ for n_, p_ in model.named_parameters() if p_.ndim < 2 or n_.startswith('adapter.')]
    decay = [p_ for n_, p_ in model.named_parameters() if p_.ndim >= 2 and not n_.startswith('adapter.')]
    optim = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                               {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train.batches(args.seed, 0, args.micro_batch_samples)) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

    config = {k: getattr(args, k) for k in RESUME_KEYS}
    epoch, micro_in_epoch, micro_idx, opt_step, gpu_seconds = 0, 0, 0, 0, 0.0
    state, state_path = load_checkpoint(ckpt_dir)
    if state is not None:
        if state['config'] != config:
            sys.exit(f'refusing to resume from {state_path}: config differs')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optim.load_state_dict(state['optim'])
        epoch, micro_in_epoch, micro_idx, opt_step = (state['epoch'], state['micro_in_epoch'], state['micro_idx'],
                                                      state['opt_step'])
        gpu_seconds = state['gpu_seconds']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        print(f'resumed from {state_path}: epoch {epoch + 1}, micro-batch {micro_in_epoch}, step {opt_step}',
              flush=True)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), model_config=model.config_dict(), steps_per_epoch=steps_per_epoch,
                       total_steps=total_steps, train_recordings=len(train.usable),
                       articulator_targets=sum(a is not None for a in train.artic),
                       parameters=sum(p_.numel() for p_ in model.parameters()),
                       torch=torch.__version__, gpu=torch.cuda.get_device_name(0)), f, indent=1)

    def snapshot(epoch_losses):
        return {'model': model.state_dict(), 'ema': ema.state_dict(), 'optim': optim.state_dict(),
                'config': config, 'epoch': epoch, 'micro_in_epoch': micro_in_epoch, 'micro_idx': micro_idx,
                'opt_step': opt_step, 'gpu_seconds': gpu_seconds, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(), 'epoch_losses': epoch_losses}

    model.train()
    last_ckpt = time.time()
    epoch_losses = state.get('epoch_losses', []) if state else []
    while epoch < args.epochs:
        batches = train.batches(args.seed, epoch, args.micro_batch_samples)
        if args.max_micro_batches_per_epoch:
            batches = batches[:args.max_micro_batches_per_epoch]
        t_epoch = time.time()
        torch.cuda.reset_peak_memory_stats()
        while micro_in_epoch < len(batches):
            t0 = time.time()
            rng = np.random.default_rng([args.seed, epoch, micro_in_epoch])
            idx = batches[micro_in_epoch]
            ramp = min(1.0, max(0.0, (opt_step - args.judge_start) / max(1, args.judge_ramp)))
            losses = train_losses(model, train, idx, rng, args, keep, device, maker, parts, judge=ramp > 0)
            loss = sum(LOSS_WEIGHTS[k](args) * (ramp if k == 'judge' else 1.0) * v for k, v in losses.items())
            (loss / args.grad_accum).backward()
            epoch_losses.append({k: float(v) for k, v in losses.items()})
            micro_idx += 1
            micro_in_epoch += 1
            if micro_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for g in optim.param_groups:
                    g['lr'] = lr_at(opt_step)
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1
                with torch.no_grad():
                    decay = min(args.ema, (1 + opt_step) / (10 + opt_step))  # EMA warm-up
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                gpu_seconds += time.time() - t0
                if args.crash_at_step == opt_step and not os.path.exists(os.path.join(out_dir, f'.crashed_{opt_step}')):
                    open(os.path.join(out_dir, f'.crashed_{opt_step}'), 'w').close()
                    print(f'simulated crash at optimizer step {opt_step}', flush=True)
                    os._exit(1)
                if stop['flag']:
                    save_checkpoint(ckpt_dir, snapshot(epoch_losses))
                    print(f'checkpoint saved at optimizer step {opt_step}; exiting', flush=True)
                    sys.exit(EXIT_INTERRUPTED)
                if time.time() - last_ckpt >= args.ckpt_minutes * 60:
                    save_checkpoint(ckpt_dir, snapshot(epoch_losses))
                    last_ckpt = time.time()
            else:
                gpu_seconds += time.time() - t0
            if micro_in_epoch % 50 == 0:
                recent = epoch_losses[-50:]
                log_line(os.path.join(out_dir, 'progress.jsonl'), {
                    'epoch': epoch + 1, 'micro_batch': micro_in_epoch, 'micro_batches_in_epoch': len(batches),
                    'opt_step': opt_step, 'recent_loss': float(np.mean([sum(LOSS_WEIGHTS[k](args) * v for k, v in d.items())
                                                                        for d in recent])),
                    'time': time.time()})

        t_eval = time.time()
        record = {'epoch': epoch + 1, 'opt_step': opt_step, 'lr': lr_at(opt_step), 'micro_batches': len(batches),
                  'epoch_train_seconds': t_eval - t_epoch,
                  'peak_vram_reserved_gb': torch.cuda.max_memory_reserved() / 2 ** 30}
        for k in ('letter', 'phone', 'artic', 'rate', 'judge'):
            vals = [d[k] for d in epoch_losses if k in d]
            if vals:
                record['train_' + k] = float(np.mean(vals))
        record.update(evaluate(ema, dev, dev_cands, device, keep, parts, args.eval_limit))
        gpu_seconds += time.time() - t_eval
        record['eval_seconds'] = time.time() - t_eval
        record['gpu_hours_total'] = gpu_seconds / 3600
        record['epoch_wall_seconds'] = time.time() - t_epoch
        log_line(os.path.join(out_dir, 'log.jsonl'), record)
        atomic_save({'config': ema.config_dict(), 'model': ema.state_dict(), 'keep_channels': keep},
                    os.path.join(out_dir, 'model.pt'))
        epoch += 1
        micro_in_epoch, epoch_losses = 0, []
        save_checkpoint(ckpt_dir, snapshot(epoch_losses))
        last_ckpt = time.time()
        if stop['flag']:
            sys.exit(EXIT_INTERRUPTED)
    with open(os.path.join(out_dir, 'DONE'), 'w') as f:
        f.write(json.dumps({'epochs': epoch, 'opt_step': opt_step, 'gpu_hours': gpu_seconds / 3600}) + '\n')
    print('training complete', flush=True)


if __name__ == '__main__':
    main()
