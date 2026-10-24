"""Peak VRAM of SVR-1's worst micro-batches over the whole run, with the full job's arguments.

The sampler is deterministic (seed, epoch), so every micro-batch is known in advance. Candidates: the 3 largest
padded sizes (rows x longest input: activations, attention) and the 3 with the most rows (CTC, judge loss).
Each runs at speed factor 0.8 (the largest padded size the sampler allows: slower rows are clamped to it), judge and
speaking losses on (evenly spaced stand-in alignments), fwd + bwd + AdamW step.

    venv-svr/bin/python -m tests.smoke_svr1_vram [svr.train_svr1 args...]
"""
import json
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from svr import train_svr1 as T  # noqa: E402


def main():
    args = T.parse_args(['--name', '_vram'] + sys.argv[1:])
    total = torch.cuda.get_device_properties(0).total_memory
    cap = args.vram_gb * 2 ** 30 / total if args.vram_gb else 0.9
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap))
    torch.backends.cuda.matmul.allow_tf32 = True
    keep = [int(c) for c in args.keep_channels.split(',')]
    train = T.TrainData('train', use_artic=not args.no_artic, preprocess=args.preprocess)
    maker = T.CandidateMaker()
    every = [(e, b) for e in range(args.epochs) for b in train.batches(args.seed, e, args.micro_batch_samples)]
    padded = lambda b: len(b) * max(train.lengths[k] for k in b)  # noqa: E731
    worst = sorted(every, key=lambda eb: -padded(eb[1]))[:3] + sorted(every, key=lambda eb: -len(eb[1]))[:3]
    model = T.make_model(args).cuda().train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'vram_cap_gb': round(cap * total / 2 ** 30, 1),
                      'attention': model.cfg.attention, 'parameters': sum(p.numel() for p in model.parameters()),
                      'micro_batches_scanned': len(every)}), flush=True)
    for e, b in worst:  # stand-in alignments: letters evenly spread over the clip
        for k in b:
            n, f = len(train.targets[k]), train.lengths[k] // 8
            train.align[k] = (np.arange(n) * f // max(n, 1)).astype(np.int64)
    worst_peak = 0.0
    for e, b in worst:
        torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(0)
        losses, _ = T.train_losses(model, train, b, rng, args, keep, 'cuda', maker, T.score_parts(args),
                                   force_speed=0.8, judge=True)
        loss = sum(T.LOSS_WEIGHTS[k](args) * v for k, v in losses.items())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / 2 ** 30
        worst_peak = max(worst_peak, peak)
        print(json.dumps({'epoch': e + 1, 'rows': len(b), 'padded_samples': padded(b),
                          'losses': {k: round(float(v), 3) for k, v in losses.items()},
                          'peak_reserved_gb': round(peak, 2)}), flush=True)
    print(json.dumps({'micro_batch_samples': args.micro_batch_samples, 'worst_peak_reserved_gb': round(worst_peak, 2),
                      'card_gb': round(total / 2 ** 30, 1)}), flush=True)


if __name__ == '__main__':
    main()
