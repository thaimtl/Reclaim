"""Run the worst micro-batches the full 200-epoch run will ever see, under a VRAM cap (default 90% of the card).

The sampler is deterministic (seed, epoch), so every micro-batch of the run is known in advance.
Candidates: the top 3 by window count (activations) and the top 3 by B * T_max * (2 * S_max + 1)
(CTC lattice). Each runs fwd + bwd + AdamW step; peak memory is reset between candidates.

    venv/bin/python -m tests.smoke_worst_batches 448000 [epochs] [vram cap in GB, default 90% of the card]
"""
import json
import os
import sys

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
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    total = torch.cuda.get_device_properties(0).total_memory
    cap_gb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9 * total / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(cap_gb * 2 ** 30 / total)
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'vram_cap_gb': round(cap_gb, 1)}), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = EMGTextDataset('train')

    def stats(batch):
        lens = [train.inputs[i].shape[0] for i in batch]
        frames = [n // 8 for n in lens]
        s_max = max(len(train.targets[i]) for i in batch)
        return {'windows': -(-sum(lens) // T.WINDOW), 'ctc': len(batch) * max(frames) * (2 * s_max + 1)}

    every = [(e, b) for e in range(epochs) for b in T.epoch_batches(train, 0, e, budget)]
    scored = [(e, b, stats(b)) for e, b in every]
    by_w = sorted(scored, key=lambda t: -t[2]['windows'])[:3]
    by_c = sorted(scored, key=lambda t: -t[2]['ctc'])[:3]
    print(json.dumps({'budget': budget, 'epochs_scanned': epochs, 'micro_batches_scanned': len(every)}), flush=True)

    model = EMGModel().cuda().train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    worst_peak = 0.0
    for kind, cands in (('windows', by_w), ('ctc', by_c)):
        for e, batch, st in cands:
            torch.cuda.reset_peak_memory_stats()
            xs = [torch.from_numpy(train.inputs[i]) for i in batch]
            lengths = [x.shape[0] // 8 for x in xs]
            x = T.combine_fixed_length(xs, T.WINDOW).cuda()
            pred = F.log_softmax(model(x), 2).reshape(-1, 38)
            pieces, s = [], 0
            for L in lengths:
                pieces.append(pred[s:s + L])
                s += L
            pred = nn.utils.rnn.pad_sequence(pieces, batch_first=False)
            targets = [torch.from_numpy(train.targets[i]) for i in batch]
            y = nn.utils.rnn.pad_sequence(targets, batch_first=True).cuda()
            loss = F.ctc_loss(pred, y, lengths, [len(t) for t in targets], blank=BLANK)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            del x, pred, y, loss, pieces
            peak = torch.cuda.max_memory_reserved() / 2 ** 30
            worst_peak = max(worst_peak, peak)
            print(json.dumps({'by': kind, 'epoch': e + 1, 'recordings': len(batch), **st,
                              'peak_reserved_gb': round(peak, 2)}), flush=True)
    print(json.dumps({'budget': budget, 'worst_peak_reserved_gb': round(worst_peak, 2)}), flush=True)


if __name__ == '__main__':
    main()
