"""Measure peak VRAM and throughput of one training step at a given micro-batch budget.

Uses the worst real micro-batch (most 200-frame windows) over 5 epochs of the sampler, so the
measured peak is an upper bound for the full run. Runs fwd + bwd + AdamW step 4 times under the
same per-process VRAM cap as training.

    venv-svr/bin/python -m tests.smoke_vram_sweep 384000
"""
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from svr import train as T  # noqa: E402
from svr.data import EMGTextDataset  # noqa: E402
from svr.model import EMGModel  # noqa: E402
from svr.text import BLANK  # noqa: E402


def main():
    budget = int(sys.argv[1])
    vram_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 28.0
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(vram_gb * 2 ** 30 / total)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = EMGTextDataset('train')

    def windows(batch):
        n = sum(train.inputs[i].shape[0] for i in batch)
        return -(-n // T.WINDOW)
    candidates = [b for e in range(5) for b in T.epoch_batches(train, 0, e, budget)]
    worst = max(candidates, key=windows)
    n_per_epoch = len(T.epoch_batches(train, 0, 0, budget))

    model = EMGModel().cuda().train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    fpw = T.flops_per_window(model, 'cuda')
    torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        xs = [torch.from_numpy(train.inputs[i]) for i in worst]
        lengths = [x.shape[0] // 8 for x in xs]
        x = T.combine_fixed_length(xs, T.WINDOW).cuda()
        pred = F.log_softmax(model(x), 2).reshape(-1, 38)
        pieces, s = [], 0
        for L in lengths:
            pieces.append(pred[s:s + L])
            s += L
        pred = nn.utils.rnn.pad_sequence(pieces, batch_first=False)
        targets = [torch.from_numpy(train.targets[i]) for i in worst]
        y = nn.utils.rnn.pad_sequence(targets, batch_first=True).cuda()
        loss = F.ctc_loss(pred, y, lengths, [len(t) for t in targets], blank=BLANK)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    w = x.shape[0]
    step = min(times[1:])
    out = {'budget_samples': budget, 'worst_windows': w, 'recordings_in_worst': len(worst),
           'micro_batches_per_epoch': n_per_epoch,
           'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
           'step_seconds': round(step, 3), 'ms_per_window': round(1000 * step / w, 2),
           'achieved_tflops': round(fpw * w / step / 1e12, 1)}
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
