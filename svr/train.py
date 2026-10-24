"""Train the EMG-to-text model with Gaddy's recipe (recognition_model.py), in our code.

Recipe, as in his train_model():
    * micro-batches of whole recordings up to a raw-EMG-sample budget (his: 128,000), joined end
      to end and cut into 1,600-sample (200-frame) windows; outputs re-assembled per recording for CTC
    * random 0-7 sample left shift of each micro-batch (augmentation)
    * CTC loss (blank = 37), AdamW lr 3e-4, no weight decay
    * gradient accumulation (his: 2 micro-batches per optimizer step)
    * linear warm-up over the first 128M raw samples of training (his: 1,000 micro-batches x 128,000),
      converted to optimizer steps for the chosen micro-batch size and accumulation;
      lr halved after epochs 125, 150 and 175
    * 200 epochs; the final epoch's weights are the result
Differences: TF32 matmuls; the accumulated loss is divided by the accumulation count (Adam is
insensitive to that scale); learning rate is a pure function of (step, epoch) so resume is exact.

Fault tolerance:
    * checkpoints/<name>/ckpt/latest.pt every --ckpt-minutes and at every epoch end, written to a
      temp file then renamed (a crash mid-write never corrupts it); the previous one is kept as prev.pt
    * resume is automatic: model, optimizer, position inside the epoch, step counters, all RNG states
    * SIGTERM / Ctrl-C saves a checkpoint at the next optimizer step and exits with code 3
Efficiency logging: per epoch in log.jsonl, GPU-hours, achieved TFLOPS (FLOPs counted once per
window with torch's FlopCounterMode) and PyTorch peak VRAM; system CPU / RAM / GPU / VRAM utilization
by Erbium's ResourceMonitor, as a report + plot under checkpoints/<name>/erbium/.

    venv-svr/bin/python -m svr.train --name B1T
"""
import argparse
import json
import os
import random
import signal
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from erbium.api import ResourceMonitor  # noqa: E402
from torch import nn  # noqa: E402

from svr.data import EMGTextDataset  # noqa: E402
from svr.decoders import BeamLMDecoder, greedy_decode  # noqa: E402
from svr.model import EMGModel  # noqa: E402
from svr.text import BLANK, clean_text  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
LM_PATH = os.path.join(WORK, 'baseline_gaddy', 'checkpoints', 'lm.binary')
WINDOW = 200 * 8
MILESTONES = (125, 150, 175)
EXIT_INTERRUPTED = 3
RESUME_KEYS = ('keep_channels', 'micro_batch_samples', 'grad_accum', 'lr', 'warmup_samples', 'epochs', 'seed',
               'preprocess')


def combine_fixed_length(tensors, length):
    total = sum(t.shape[0] for t in tensors)
    if total % length:
        tensors = list(tensors) + [tensors[0].new_zeros(length - total % length, *tensors[0].shape[1:])]
        total += length - total % length
    return torch.cat(tensors, 0).view(total // length, length, *tensors[0].shape[1:])


def learning_rate(base, step, epoch, warmup_steps):
    """step: 1-based optimizer step about to be taken; epoch: 0-based."""
    lr = base * min(1.0, step / warmup_steps)
    return lr * 0.5 ** sum(epoch >= m for m in MILESTONES)


def epoch_batches(dataset, seed, epoch, max_samples):
    return list(dataset.batches(random.Random(seed * 100003 + epoch), max_samples))


def _erbium_child(monitor):
    """Erbium's sampling loop, made to die with the trainer: the kernel sends SIGTERM when the
    parent exits for any reason, including kill -9 or a crash (Erbium's daemon flag only covers
    a clean interpreter exit, so hard crashes used to leave the monitor running forever)."""
    import ctypes
    ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    if os.getppid() == 1:  # parent already gone before prctl took effect
        os._exit(0)
    monitor._run()


def start_resource_monitor(report_dir, interval):
    import multiprocessing
    monitor = ResourceMonitor(report_dir, interval=interval)
    monitor._process = multiprocessing.Process(target=_erbium_child, args=(monitor,), name='Resource Monitor',
                                               daemon=True)
    monitor.start()
    return monitor


def flops_per_window(model, device):
    """Training FLOPs (forward + backward) for one 200-frame window, counted by PyTorch.

    Runs on a copy with the RNG state preserved, so it changes neither the weights, the BatchNorm
    statistics nor the random stream of the real run.
    """
    import copy
    from torch.utils.flop_counter import FlopCounterMode
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        clone = copy.deepcopy(model)
        x = torch.randn(4, WINDOW, 8, device=device)
        counter = FlopCounterMode(display=False)
        with counter:
            clone(x).float().sum().backward()
    del clone
    torch.cuda.empty_cache()
    return counter.get_total_flops() / 4


def evaluate(model, dataset, device, decoder=None, limit=None):
    model.eval()
    refs, greedy, beam = [], [], []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    with torch.no_grad():
        for x, text in zip(dataset.inputs[:n], dataset.texts[:n]):
            lp = F.log_softmax(model(torch.from_numpy(x)[None].to(device)), -1)[0].float().cpu().numpy()
            refs.append(clean_text(text))
            greedy.append(greedy_decode(lp))
            if decoder is not None:
                beam.append(decoder.decode(lp))
    model.train()
    out = {'dev_wer_greedy': jiwer.wer(refs, greedy)}
    if decoder is not None:
        out['dev_wer_beam_lm'] = jiwer.wer(refs, beam)
    return out


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(ckpt_dir, state):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest, prev = os.path.join(ckpt_dir, 'latest.pt'), os.path.join(ckpt_dir, 'prev.pt')
    tmp = latest + '.tmp'
    torch.save(state, tmp)
    if os.path.exists(latest):
        os.replace(latest, prev)
    os.replace(tmp, latest)


def load_checkpoint(ckpt_dir):
    for name in ('latest.pt', 'prev.pt'):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            try:
                return torch.load(path, map_location='cpu', weights_only=False), path
            except Exception as e:
                print(f'could not read {path}: {e}', flush=True)
    return None, None


def log_line(path, record):
    print(json.dumps(record), flush=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--keep-channels', default='0,1,2,3,4,5,6,7')
    p.add_argument('--preprocess', default='causal', choices=['causal', 'gaddy'],
                   help='causal: rig/filters.py, as SVR-1 and the live rig (G4); gaddy: his zero-phase filters (B1T)')
    p.add_argument('--epochs', type=int, default=200)
    # 432k x 1: worst micro-batch of the 200-epoch run peaks at 26.0 GB (28 GB cap); Gaddy used 128k x 2
    p.add_argument('--micro-batch-samples', type=int, default=432000, help='raw 1000 Hz EMG samples per micro-batch')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup-samples', type=int, default=128_000_000,
                   help='raw EMG samples of linear warm-up (Gaddy: 1,000 micro-batches x 128,000)')
    p.add_argument('--beam-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--vram-gb', type=float, default=28.0, help='hard cap on this process\'s GPU memory')
    p.add_argument('--ckpt-minutes', type=float, default=15.0)
    p.add_argument('--progress-minutes', type=float, default=5.0)
    p.add_argument('--monitor-seconds', type=float, default=15.0, help='Erbium sampling interval')
    # smoke-test controls
    p.add_argument('--max-micro-batches-per-epoch', type=int, default=None)
    p.add_argument('--eval-limit', type=int, default=None)
    p.add_argument('--crash-at-step', type=int, default=None, help='simulate one hard crash after this optimizer step')
    p.add_argument('--stop-after-steps', type=int, default=None, help='exit cleanly after this optimizer step')
    p.add_argument('--step-log', action='store_true', help='write every micro-batch to steps.jsonl (smoke tests)')
    p.add_argument('--out-root', default=os.path.join(WORK, 'checkpoints'))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.out_root, args.name)
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    log_path = os.path.join(out_dir, 'log.jsonl')
    progress_path = os.path.join(out_dir, 'progress.jsonl')
    os.makedirs(out_dir, exist_ok=True)
    keep = [int(c) for c in args.keep_channels.split(',')]
    warmup_steps = max(1, round(args.warmup_samples / (args.micro_batch_samples * args.grad_accum)))
    start_resource_monitor(os.path.join(out_dir, 'erbium'), args.monitor_seconds)
    device = 'cuda'
    total_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2 ** 30 / total_mem))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stop_requested = {'flag': False}

    def on_signal(signum, frame):
        stop_requested['flag'] = True
        print(f'signal {signum}: will checkpoint and exit at the next optimizer step', flush=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = EMGTextDataset('train', keep_channels=keep, preprocess=args.preprocess)
    dev = EMGTextDataset('dev', keep_channels=keep, preprocess=args.preprocess)
    model = EMGModel().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    config = {k: getattr(args, k) for k in RESUME_KEYS if k != 'keep_channels'}
    config['keep_channels'] = keep

    # ---- resume ----
    state, state_path = load_checkpoint(ckpt_dir)
    if state is not None:
        saved = dict(state['config'])
        saved.setdefault('preprocess', 'gaddy')  # checkpoints from before --preprocess were all Gaddy's pipeline
        mismatched = {k: (saved.get(k), config[k]) for k in RESUME_KEYS if saved.get(k) != config[k]}
        if mismatched:
            sys.exit(f'refusing to resume from {state_path}: config differs {mismatched}')
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        if state.get('grads') is not None:  # gradients accumulated across an epoch boundary
            for p, g in zip(model.parameters(), state['grads']):
                p.grad = None if g is None else g.to(device)
        epoch, micro_in_epoch, micro_idx, opt_step = (state['epoch'], state['micro_in_epoch'],
                                                       state['micro_idx'], state['opt_step'])
        shift_rng = random.Random()
        shift_rng.setstate(state['shift_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        gpu_seconds, wall_start_offset = state['gpu_seconds'], state.get('epoch_train_seconds', 0.0)
        epoch_losses = state.get('epoch_losses', [])
        epoch_windows = state.get('epoch_windows', 0)
        print(f'resumed from {state_path}: epoch {epoch + 1}, micro-batch {micro_in_epoch} of the epoch, '
              f'optimizer step {opt_step}', flush=True)
    else:
        epoch, micro_in_epoch, micro_idx, opt_step = 0, 0, 0, 0
        shift_rng = random.Random(args.seed + 1)
        gpu_seconds, wall_start_offset, epoch_losses, epoch_windows = 0.0, 0.0, [], 0

    fpw = flops_per_window(model, device)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), keep_channels=keep, warmup_steps=warmup_steps, train_recordings=len(train), dev_recordings=len(dev),
                       parameters=sum(p.numel() for p in model.parameters()), train_flops_per_window=fpw,
                       torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                       vram_cap_fraction=args.vram_gb * 2 ** 30 / total_mem), f, indent=1)

    decoder = None
    model.train()  # no zero_grad here: a resume may have restored half-accumulated gradients
    last_ckpt = last_progress = time.time()

    def snapshot():
        pending = micro_idx % args.grad_accum != 0
        grads = [None if p.grad is None else p.grad.detach().cpu() for p in model.parameters()] if pending else None
        return {'model': model.state_dict(), 'optim': optim.state_dict(), 'config': config, 'grads': grads,
                'epoch': epoch, 'micro_in_epoch': micro_in_epoch, 'micro_idx': micro_idx, 'opt_step': opt_step,
                'shift_rng': shift_rng.getstate(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(), 'gpu_seconds': gpu_seconds,
                'epoch_train_seconds': epoch_train_seconds, 'epoch_losses': epoch_losses,
                'epoch_windows': epoch_windows, 'saved_at': time.time()}

    while epoch < args.epochs:
        batches = epoch_batches(train, args.seed, epoch, args.micro_batch_samples)
        if args.max_micro_batches_per_epoch:
            batches = batches[:args.max_micro_batches_per_epoch]
        torch.cuda.reset_peak_memory_stats()
        t_epoch = time.time()
        epoch_train_seconds = wall_start_offset
        wall_start_offset = 0.0
        while micro_in_epoch < len(batches):
            t0 = time.time()
            batch = batches[micro_in_epoch]
            xs = [torch.from_numpy(train.inputs[i]) for i in batch]
            lengths = [x.shape[0] // 8 for x in xs]
            x = combine_fixed_length(xs, WINDOW).to(device, non_blocking=True)
            r = shift_rng.randrange(8)
            if r > 0:
                x[:, :-r, :] = x[:, r:, :].clone()
                x[:, -r:, :] = 0
            pred = F.log_softmax(model(x), 2)
            pred = pred.reshape(-1, pred.shape[-1])
            pieces, start = [], 0
            for length in lengths:
                pieces.append(pred[start:start + length])
                start += length
            pred = nn.utils.rnn.pad_sequence(pieces, batch_first=False)
            targets = [torch.from_numpy(train.targets[i]) for i in batch]
            y = nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)
            loss = F.ctc_loss(pred, y, lengths, [len(t) for t in targets], blank=BLANK)
            (loss / args.grad_accum).backward()
            epoch_losses.append(loss.item())
            epoch_windows += x.shape[0]
            micro_idx += 1
            micro_in_epoch += 1
            stepped = False
            if micro_idx % args.grad_accum == 0:
                opt_step += 1
                for g in optim.param_groups:
                    g['lr'] = learning_rate(args.lr, opt_step, epoch, warmup_steps)
                optim.step()
                optim.zero_grad(set_to_none=True)
                stepped = True
            if args.step_log:
                with open(os.path.join(out_dir, 'steps.jsonl'), 'a') as f:
                    f.write(json.dumps({'micro_idx': micro_idx, 'opt_step': opt_step, 'epoch': epoch + 1,
                                        'micro_in_epoch': micro_in_epoch, 'first_id': train.ids[batch[0]],
                                        'n_recordings': len(batch), 'shift': r, 'loss': epoch_losses[-1],
                                        'lr': optim.param_groups[0]['lr'] if stepped else None}) + '\n')
            dt = time.time() - t0
            gpu_seconds += dt
            epoch_train_seconds += dt

            now = time.time()
            if now - last_progress >= args.progress_minutes * 60:
                last_progress = now
                with open(progress_path, 'a') as f:
                    f.write(json.dumps({'time': now, 'epoch': epoch + 1, 'micro_batch': micro_in_epoch,
                                        'micro_batches_in_epoch': len(batches), 'opt_step': opt_step,
                                        'recent_loss': float(np.mean(epoch_losses[-50:])),
                                        'gpu_hours': gpu_seconds / 3600}) + '\n')
            if stepped:
                crash_marker = os.path.join(out_dir, f'.crashed_at_step_{opt_step}')
                if args.crash_at_step == opt_step and not os.path.exists(crash_marker):
                    open(crash_marker, 'w').close()  # crash once per run directory
                    print(f'simulated crash at optimizer step {opt_step}', flush=True)
                    os._exit(1)
                if stop_requested['flag'] or (args.stop_after_steps and opt_step >= args.stop_after_steps):
                    save_checkpoint(ckpt_dir, snapshot())
                    print(f'checkpoint saved at optimizer step {opt_step}; exiting', flush=True)
                    sys.exit(EXIT_INTERRUPTED)
                if now - last_ckpt >= args.ckpt_minutes * 60:
                    save_checkpoint(ckpt_dir, snapshot())
                    last_ckpt = time.time()

        # ---- end of epoch ----
        n_windows = epoch_windows
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(epoch_losses)), 'micro_batches': len(batches),
                  'opt_step': opt_step, 'lr': optim.param_groups[0]['lr'],
                  'epoch_train_seconds': epoch_train_seconds,
                  'windows': n_windows, 'achieved_tflops': fpw * n_windows / max(epoch_train_seconds, 1e-9) / 1e12,
                  'peak_vram_allocated_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                  'peak_vram_reserved_gb': torch.cuda.max_memory_reserved() / 2 ** 30}
        use_beam = (epoch + 1) % args.beam_every == 0 or epoch + 1 == args.epochs
        if use_beam and decoder is None:
            try:
                decoder = BeamLMDecoder(LM_PATH)
            except (ImportError, OSError) as e:  # kenlm or lm.binary missing: keep training, greedy dev WER only
                print(f'beam/LM dev evaluation disabled: {e}', flush=True)
                decoder = False
        use_beam = use_beam and bool(decoder)
        t_eval = time.time()
        record.update(evaluate(model, dev, device, decoder if use_beam else None, args.eval_limit))
        gpu_seconds += time.time() - t_eval
        record['eval_seconds'] = time.time() - t_eval
        record['epoch_wall_seconds'] = time.time() - t_epoch
        record['gpu_hours_total'] = gpu_seconds / 3600
        log_line(log_path, record)

        atomic_save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
        if (epoch + 1) % 50 == 0:
            atomic_save(model.state_dict(), os.path.join(out_dir, f'model_epoch{epoch + 1}.pt'))
        epoch += 1
        micro_in_epoch, epoch_losses, epoch_windows, epoch_train_seconds = 0, [], 0, 0.0
        save_checkpoint(ckpt_dir, snapshot())
        last_ckpt = time.time()
        if stop_requested['flag']:
            print('checkpoint saved at epoch end; exiting', flush=True)
            sys.exit(EXIT_INTERRUPTED)
    with open(os.path.join(out_dir, 'DONE'), 'w') as f:
        f.write(json.dumps({'epochs': epoch, 'opt_step': opt_step, 'gpu_hours': gpu_seconds / 3600}) + '\n')
    print('training complete', flush=True)


if __name__ == '__main__':
    main()
