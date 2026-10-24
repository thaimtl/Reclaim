"""Unit checks for SVR-1 (CPU or GPU; FA2-vs-SDPA parity only where flash-attn and a CUDA bf16 GPU exist).

    venv/bin/python -m tests.smoke_svr1_units
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from svr import augment
from svr.phonemes import sentence_phonemes
from svr.svr1 import INFEASIBLE, build, candidate_tensors, ctc_fits, flash_attn_varlen_func, score_texts
from svr.text import text_to_ids

TINY = dict(d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, verifier_dim=64, verifier_layers=1)


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    m = build(TINY).to(device).eval()
    cm = torch.zeros(8, device=device)
    cm[:6] = 1

    # shapes; frames = samples / 8 exactly
    x = torch.randn(2, 80 * 8, 8, device=device)
    lengths = torch.tensor([80, 50], device=device)
    with torch.no_grad():
        out = m(x, lengths, cm)
    check('letter/phone/artic shapes', out['letter_logp'].shape == (2, 80, 38) and out['phone_logp'].shape == (2, 80, 40)
          and out['artic'].shape == (2, 80, 13))

    # padding invariance: row 1 alone == row 1 padded next to a longer row (no batch statistics, masked attention)
    with torch.no_grad():
        alone = m(x[1:2, :50 * 8], lengths[1:2], cm)['letter_logp'][0]
    err = (alone - out['letter_logp'][1, :50]).abs().max().item()
    check('padding does not change a row', err < 1e-4, f'max diff {err:.2e}')

    # absent channels have no effect (masked before and after the adapter)
    x2 = x.clone()
    x2[..., 6:] = torch.randn_like(x2[..., 6:]) * 10
    with torch.no_grad():
        err = (m(x2, lengths, cm)['letter_logp'] - out['letter_logp']).abs().max().item()
    check('masked channels are ignored', err < 1e-5, f'max diff {err:.2e}')

    # per-row channel masks
    with torch.no_grad():
        o2 = m(x, lengths, cm[None].expand(2, -1))['letter_logp']
    check('(B, C) channel mask == (C,) mask', (o2 - out['letter_logp']).abs().max().item() < 1e-5)

    # ctc_fits agrees with torch's CTC (inf loss <=> infeasible)
    for txt, frames in (('hello', 5), ('hello', 6), ('abc', 3), ('aa', 2), ('aa', 3)):
        ids = torch.tensor(text_to_ids(txt))
        lp = torch.randn(frames, 1, 38).log_softmax(-1)
        inf = torch.isinf(F.ctc_loss(lp, ids[None], [frames], [len(ids)], blank=37, reduction='none'))[0].item()
        fits = ctc_fits(ids, torch.tensor([len(ids)]), torch.tensor([frames]))[0].item()
        check(f'ctc_fits({txt!r}, {frames} frames) = {fits}', fits == (not inf))

    # candidate scoring: infeasible candidate -> INFEASIBLE, finite gradients everywhere
    m.train()
    out = m(x, torch.tensor([80, 12], device=device), cm)
    cands = ['the cat sat', 'the bat sat', 'the cat sat on a very long mat that cannot possibly fit here']
    s, detail = m.score_candidates(out, 1, *candidate_tensors(cands, device))
    check('too-long candidate scored INFEASIBLE', s[2].item() == INFEASIBLE and not detail['feasible'][2].item())
    loss = F.cross_entropy(s[None], torch.zeros(1, dtype=torch.long, device=device))
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    check('finite gradients with an infeasible candidate', not bad, ', '.join(bad[:3]))
    got = {n.split('.')[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    check('judge gradients reach verifier, mix, encoder', {'verifier', 'mix', 'blocks', 'front'} <= got, str(sorted(got)))
    m.zero_grad()

    # out-of-dictionary candidate does not get a free phoneme score
    oov = 'the cat zzqxv'
    check('test word is out of dictionary', sentence_phonemes(oov) is None)
    with torch.no_grad():
        out = m.eval()(x, lengths, cm)
        s_mixed, d_mixed = m.score_candidates(out, 0, *candidate_tensors(['the cat sat', oov], device))
    check('phone term dropped when any candidate is OOV', 'phone' not in d_mixed)
    check('mix weights on the simplex', abs(m.mix.softmax(0).sum().item() - 1) < 1e-6)
    check('score_texts finite', torch.isfinite(score_texts(m, out, 0, ['the cat sat', 'a bat sat'], device)).all())

    # augmentation: speed keeps whole frames and channel layout; channel_drop keeps >= 3
    rng = np.random.default_rng(0)
    xa = np.random.randn(800, 8).astype(np.float32)
    for f in (0.8, 1.0, 1.6):
        y = augment.speed(xa, f)
        check(f'speed {f}: whole frames', len(y) % 8 == 0 and y.shape[1] == 8)
    check('channel_drop min 3', all(len(augment.channel_drop(range(6), rng, 0.9)) >= 3 for _ in range(50)))

    # under autocast the attention inputs are bf16 (FA2 needs them; fp32 here silently fell back to SDPA)
    seen = []
    hook = m.blocks[0].attn.qkv.register_forward_hook(lambda mod, i, o: seen.append(o.dtype))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        o = m(x, lengths, cm)
    hook.remove()
    check('qkv output is bf16 under autocast', seen == [torch.bfloat16], str(seen))
    check('heads return fp32 log-probs', o['letter_logp'].dtype == torch.float32)

    # FA2 vs SDPA (bf16, H100/H200)
    if device == 'cuda' and flash_attn_varlen_func is not None:
        torch.manual_seed(0)
        ms = build(TINY, attention='sdpa').cuda().eval()
        mf = build(TINY, attention='fa2').cuda().eval()
        mf.load_state_dict(ms.state_dict())
        xb = torch.randn(3, 300 * 8, 8, device='cuda')
        lb = torch.tensor([300, 211, 97], device='cuda')
        import svr.svr1 as S
        calls = []
        real = S.flash_attn_varlen_func
        S.flash_attn_varlen_func = lambda *a_, **k_: calls.append(1) or real(*a_, **k_)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = ms(xb, lb, cm)['letter_logp'], mf(xb, lb, cm)['letter_logp']
        S.flash_attn_varlen_func = real
        check('FA2 kernel actually runs', len(calls) == TINY['n_layers'], f'{len(calls)} calls')
        mask = torch.arange(300, device='cuda')[None] < lb[:, None]
        err = (a - b).abs()[mask].max().item()
        check('FA2 == SDPA on real frames (bf16)', err < 0.05, f'max diff {err:.3f}')
    else:
        print('SKIP FA2 parity (needs CUDA + flash-attn)')
    print('SVR1 UNITS PASS')


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    main()
