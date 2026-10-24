"""Unit-level smoke checks for svr.train (no full training).

    venv/bin/python -m tests.smoke_train_units
"""
import copy
import random
import time

import torch

from svr import train as T
from svr.data import EMGTextDataset
from svr.model import EMGModel


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    # learning-rate schedule: warm-up and milestones
    lr = T.learning_rate
    check('warmup step 1', abs(lr(3e-4, 1, 0, 500) - 3e-4 / 500) < 1e-15)
    check('warmup step 250', abs(lr(3e-4, 250, 0, 500) - 1.5e-4) < 1e-15)
    check('after warmup', lr(3e-4, 501, 3, 500) == 3e-4 and lr(3e-4, 9999, 124, 500) == 3e-4)
    check('milestone epoch 125 (0-based)', lr(3e-4, 9999, 125, 500) == 1.5e-4)
    check('milestone 150 / 175', lr(3e-4, 9999, 150, 500) == 7.5e-5 and lr(3e-4, 9999, 199, 500) == 3.75e-5)
    # Gaddy: lr at micro-batch i (1-based) during warm-up = i * lr / 1000; his step k uses micro-batch 2k
    check('matches Gaddy warm-up at accum 2', all(abs(lr(3e-4, k, 0, 500) - 2 * k * 3e-4 / 1000) < 1e-15
                                                  for k in range(1, 501)))

    # sampler: deterministic per epoch, within budget, different across epochs
    train = EMGTextDataset('train')
    a = T.epoch_batches(train, 0, 5, 128000)
    b = T.epoch_batches(train, 0, 5, 128000)
    c = T.epoch_batches(train, 0, 6, 128000)
    check('epoch batch order deterministic', a == b)
    check('epochs differ', a != c)
    worst = max(sum(train.emg_lengths[i] for i in bt) for bt in a)
    check('every micro-batch within budget', worst <= 128000, f'largest {worst}')
    used = {i for bt in a for i in bt}
    check('epoch covers most recordings once', len(used) == sum(len(bt) for bt in a) and len(used) > 0.95 * len(train),
          f'{len(used)}/{len(train)} recordings, {len(a)} micro-batches')
    big = T.epoch_batches(train, 0, 5, 384000)
    check('larger budget -> fewer micro-batches', len(big) < len(a), f'{len(big)} vs {len(a)}')

    # windowing + shift augmentation
    xs = [torch.arange(10 * 8, dtype=torch.float32).view(10, 8), torch.ones(5, 8)]
    w = T.combine_fixed_length(xs, 4)
    check('combine_fixed_length pads to whole windows', w.shape == (4, 4, 8) and w[-1, -1].abs().sum() == 0)
    x = torch.arange(2 * 16 * 1, dtype=torch.float32).view(2, 16, 1)
    r = 3
    y = x.clone()
    y[:, :-r, :] = y[:, r:, :].clone()
    y[:, -r:, :] = 0
    check('shift moves samples left and zero-fills', torch.equal(y[:, :-r], x[:, r:]) and y[:, -r:].abs().sum() == 0)

    # FLOP counter leaves the model and RNG untouched
    torch.manual_seed(0)
    model = EMGModel().cuda().train()
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.cuda.get_rng_state().clone()
    fpw = T.flops_per_window(model, 'cuda')
    same = all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    check('FLOP count leaves weights and BatchNorm stats unchanged', same)
    check('FLOP count leaves CUDA RNG unchanged', torch.equal(rng_before, torch.cuda.get_rng_state()))
    check('FLOPs per window plausible', 1e9 < fpw < 1e12, f'{fpw / 1e9:.1f} GFLOP per window (fwd+bwd)')

    # resource monitor
    from erbium.api import ResourceMonitor
    import tempfile, glob
    d = tempfile.mkdtemp()
    ResourceMonitor(d, interval=0.5).start()
    time.sleep(3)
    reports = glob.glob(d + '/*.txt')
    text = open(reports[0]).read() if reports else ''
    check('Erbium monitor writes a report with CPU and GPU sections', 'CPUs' in text and 'GPUs' in text and
          'Utilization' in text, reports[0] if reports else 'no report')
    random.seed(0)
    print('ALL UNIT CHECKS PASSED')


if __name__ == '__main__':
    main()
