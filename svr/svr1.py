"""SVR-1: the EMG model of the Silent Voice Rig.

    raw EMG (B, T*8, 8 slots) x channel mask
      -> session adapter (8x8 channel mix + gain, identity at init; refit on demo day)
      -> conv front end: 3 strided residual blocks (per-frame LayerNorm), 8x downsampling to 86 frames/s
      -> rate token: speaking-rate estimate from the EMG envelope, FiLM into every block       [contribution E]
      -> pre-norm Transformer: RMSNorm, RoPE, SwiGLU; attention = FlashAttention-2 (varlen) or PyTorch SDPA
      -> heads: letter CTC, phoneme CTC, articulator regression (SPARC targets), fixed-sentence classifier + none
    disputed-span verifier: candidate phonemes (+ disputed flags) cross-attend to encoder frames  [contribution B]
    score_candidates(): letter CTC + phoneme CTC + verifier, with learned combination weights; this is the
    S_emg term of the fused system and what the homophene-contrastive judge loss trains            [contribution A]

Inputs use 8 channel slots so 8-channel pretraining (Berkeley) and 6-channel use (our rig, slots 0-5) share one
model; missing channels are zeroed by the mask. One utterance per batch row, padded; attention never crosses
utterances.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from svr.phonemes import (N_PHONE_CLASSES, N_TOKENS, PHONE_BLANK, TOKEN_PAD, disputed_words, sentence_phonemes,
                          verifier_tokens)
from svr.text import BLANK as LETTER_BLANK
from svr.text import text_to_ids
from svr.text import NUM_CLASSES as N_LETTER_CLASSES

INFEASIBLE = -1e4  # score of a candidate too long for the clip (no CTC path)

try:  # FlashAttention-2 on H100/H200; everything else uses SDPA
    from flash_attn import flash_attn_varlen_func
except Exception:  # noqa: BLE001
    flash_attn_varlen_func = None


@dataclass
class SVR1Config:
    in_channels: int = 8
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_hidden: int = 1376          # SwiGLU hidden width (~8/3 x d_model, multiple of 32)
    dropout: float = 0.1
    n_artic: int = 13               # 12 EMA trajectories + loudness
    n_fixed: int = 0                # fixed sentences (fine-tuning); +1 "none" class when > 0
    use_rate: bool = True           # contribution E
    use_verifier: bool = True       # contribution B
    verifier_dim: int = 256
    verifier_layers: int = 2
    verifier_heads: int = 4
    attention: str = 'sdpa'         # 'sdpa' or 'fa2'


# ---------------------------------------------------------------------------------------------- building blocks
class FrameNorm(nn.LayerNorm):
    """LayerNorm over channels at each time step of a (B, C, T) tensor. Unlike BatchNorm (Gaddy's front end) it
    has no batch statistics: padding never leaks into the normalisation, training and inference behave the same,
    and a new rig / a 30-utterance calibration set / a growing streaming clip cannot shift running statistics."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1, stride=stride)
        self.n1 = FrameNorm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.n2 = FrameNorm(cout)
        self.res = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride), FrameNorm(cout))

    def forward(self, x, m):
        """x (B, Cin, T) with padding already zero; m (B, 1, T/stride) float mask of real steps. Padding is
        re-zeroed after every conv so the last real frames see exactly what they see when the clip is alone."""
        y = F.gelu(self.n1(self.conv1(x))) * m
        return F.gelu(self.n2(self.conv2(y)) + self.res(x)) * m


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
                * self.weight.to(x.dtype))


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, Dh), positions 0..T-1."""
    _, _, t, d = x.shape
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.backend = cfg.attention

    def forward(self, x, mask, packing=None):
        """x (B, T, D); mask (B, T) bool, True = real frame; packing (cu_seqlens, frame index, max length) for FA2,
        computed once per forward in SVR1.forward."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # (B, H, T, Dh) each
        q, k = rope(q), rope(k)
        p = self.dropout if self.training else 0.0
        if packing is not None and q.dtype in (torch.float16, torch.bfloat16):  # q: the qkv output, bf16 under autocast
            cu, idx, max_len = packing
            pack = lambda z: z.transpose(1, 2).reshape(b * t, self.h, self.dh)[idx]  # (total, H, Dh)
            o = flash_attn_varlen_func(pack(q), pack(k), pack(v), cu, cu, max_len, max_len, dropout_p=p)
            full = o.new_zeros(b * t, self.h, self.dh)
            full[idx] = o
            o = full.view(b, t, d)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], dropout_p=p)
            o = o.transpose(1, 2).reshape(b, t, d)
        return self.out(o)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.w12 = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.film = nn.Linear(1, 2 * cfg.d_model) if cfg.use_rate else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, mask, rate, packing=None):
        # rate FiLM modulates the normalised branch inputs (adaLN style), not the residual stream, so it cannot
        # compound over the 12 blocks; zero-initialised = identity at the start
        if self.film is not None:
            gamma, beta = self.film(rate[:, None].float()).unsqueeze(1).chunk(2, -1)
            film = lambda z: z * (1 + gamma.to(z.dtype)) + beta.to(z.dtype)  # noqa: E731
        else:
            film = lambda z: z  # noqa: E731
        x = x + self.drop(self.attn(film(self.norm1(x)), mask, packing))
        a, g = self.w12(film(self.norm2(x))).chunk(2, -1)
        return x + self.drop(self.w3(F.silu(a) * g))


class RateEstimator(nn.Module):
    """Speaking-rate token from the EMG envelope (per-frame RMS), trained to predict log speed factor."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=2, dilation=1), nn.GELU())
        self.head = nn.Linear(64, 1)

    def forward(self, x, mask):
        b, t8, c = x.shape
        mt = mask[:, None].to(x.dtype)  # (B, 1, T)
        env = x.reshape(b, t8 // 8, 8, c).pow(2).mean(2).add(1e-6).log().transpose(1, 2) * mt  # (B, C, T)
        h = self.net[1](self.net[0](env)) * mt
        h = self.net[3](self.net[2](h)).transpose(1, 2)
        m = mask[..., None].to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)[:, 0]


class Verifier(nn.Module):
    """Candidate phonemes (+ disputed flags) cross-attend to EMG frames -> one match score per candidate."""

    def __init__(self, cfg):
        super().__init__()
        dv = cfg.verifier_dim
        self.tok = nn.Embedding(N_TOKENS, dv, padding_idx=TOKEN_PAD)
        self.disp = nn.Embedding(2, dv)
        self.pos = nn.Embedding(512, dv)
        self.mem = nn.Linear(cfg.d_model, dv)
        layer = nn.TransformerDecoderLayer(dv, cfg.verifier_heads, 4 * dv, dropout=cfg.dropout, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(layer, cfg.verifier_layers)
        self.norm = nn.LayerNorm(dv)
        self.score = nn.Linear(dv, 1)

    def forward(self, memory, mem_mask, tokens, disputed):
        """memory (N, T, D) encoder states; mem_mask (N, T) True = real; tokens/disputed (N, L)."""
        tok_mask = tokens != TOKEN_PAD
        pos = torch.arange(tokens.shape[1], device=tokens.device).clamp(max=511)
        y = self.tok(tokens) + self.disp(disputed.clamp(0, 1)) + self.pos(pos)[None]
        y = self.dec(y, self.mem(memory), tgt_key_padding_mask=~tok_mask, memory_key_padding_mask=~mem_mask)
        y = self.norm(y)
        w = (disputed.bool() & tok_mask).to(y.dtype)
        w = torch.where(w.sum(1, keepdim=True) > 0, w, tok_mask.to(y.dtype))  # nothing disputed: pool all
        pooled = (y * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
        return self.score(pooled)[:, 0]


# ---------------------------------------------------------------------------------------------- the model
class SVR1(nn.Module):
    def __init__(self, cfg: SVR1Config):
        super().__init__()
        self.cfg = cfg
        c, d = cfg.in_channels, cfg.d_model
        self.adapter = nn.Linear(c, c)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(c))
            self.adapter.bias.zero_()
        self.front = nn.ModuleList([ResBlock(c, d, 2), ResBlock(d, d, 2), ResBlock(d, d, 2)])
        self.proj = nn.Linear(d, d)
        self.rate = RateEstimator(c) if cfg.use_rate else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(d)
        self.letter = nn.Linear(d, N_LETTER_CLASSES)
        self.phone = nn.Linear(d, N_PHONE_CLASSES)
        self.artic = nn.Linear(d, cfg.n_artic)
        if cfg.n_fixed:
            self.fixed_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
            self.fixed = nn.Linear(d, cfg.n_fixed + 1)
        self.verifier = Verifier(cfg) if cfg.use_verifier else None
        # learned weights of letter CTC, phoneme CTC and verifier inside S_emg, kept on the simplex (softmax):
        # unconstrained weights let the judge loss shrink by inflating every score instead of ranking better
        self.mix = nn.Parameter(torch.zeros(3))

    def forward(self, x, lengths, channel_mask=None):
        """x (B, T*8, C) model inputs; lengths (B,) in frames; channel_mask (C,) or (B, C) in {0, 1}."""
        b, t8, c = x.shape
        t = t8 // 8
        mask = torch.arange(t, device=x.device)[None] < lengths[:, None]
        if channel_mask is not None:
            cm = channel_mask.to(x.dtype)
            cm = cm[None, None] if cm.dim() == 1 else cm[:, None]
            x = x * cm
        x = self.adapter(x)
        if channel_mask is not None:
            x = x * cm
        x = x * mask.repeat_interleave(8, 1)[..., None].to(x.dtype)
        rate = self.rate(x, mask) if self.rate is not None else x.new_zeros(b)
        h = x.transpose(1, 2)
        for i, blk in enumerate(self.front):  # sample-level masks at 1/2, 1/4, 1/8 resolution
            h = blk(h, mask.repeat_interleave(4 >> i, 1)[:, None].to(h.dtype))
        h = self.proj(h.transpose(1, 2)).float()  # fp32 residual stream; matmuls still run in bf16 under autocast
        packing = None
        if self.cfg.attention == 'fa2' and flash_attn_varlen_func is not None and h.is_cuda:
            lens = mask.sum(1, dtype=torch.int32)
            packing = (F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0)), mask.reshape(-1).nonzero(as_tuple=True)[0],
                       int(lens.max()))
        for blk in self.blocks:
            h = blk(h, mask, rate, packing)
        h = self.norm(h)
        out = {'h': h, 'mask': mask, 'lengths': lengths, 'rate': rate,
               'letter_logp': F.log_softmax(self.letter(h).float(), -1),
               'phone_logp': F.log_softmax(self.phone(h).float(), -1),
               'artic': self.artic(h).float()}
        if self.cfg.n_fixed:
            att = torch.einsum('btd,d->bt', h.float(), self.fixed_query.float()).masked_fill(~mask, -1e9).softmax(1)
            out['fixed_logits'] = self.fixed(torch.einsum('bt,btd->bd', att.to(h.dtype), h)).float()
        return out

    # ------------------------------------------------------------------ candidate scoring (S_emg)
    def ctc_loglik(self, logp, lengths, targets, target_lengths, blank):
        """Per-row CTC log-likelihood; logp (N, T, C) batch-first log-probs; INFEASIBLE where the target cannot fit.
        zero_infinity=True: an infeasible row would otherwise give inf loss and NaN gradients, and NaN * 0 is still
        NaN, so masking the score afterwards would not protect the encoder."""
        nll = F.ctc_loss(logp.float().transpose(0, 1), targets, lengths.long().cpu(), target_lengths.long().cpu(),
                         blank=blank, reduction='none', zero_infinity=True)
        ok = ctc_fits(targets, target_lengths, lengths)
        return torch.where(ok, -nll, torch.full_like(nll, INFEASIBLE)), ok

    def score_candidates(self, out, row, letters, letter_lens, phones, phone_lens, phone_ok, tokens, disputed,
                         parts=('letter', 'phone', 'verifier')):
        """Scores of K candidates for batch row `row`. letters/phones: concatenated targets; phone_ok (K,) bool
        marks candidates with a phoneme target; tokens/disputed (K, L) verifier inputs. Returns (K,) and parts."""
        k = letter_lens.shape[0]
        n = out['lengths'][row].expand(k)
        # the phoneme term is used only when every candidate has a pronunciation: giving out-of-dictionary
        # candidates a 0 instead of a (negative) log-likelihood would make them win for free
        active = [p for p in ('letter', 'phone', 'verifier') if p in parts
                  and (p != 'phone' or bool(phone_ok.all())) and (p != 'verifier' or self.verifier is not None)]
        w = dict(zip(active, self.mix[[('letter', 'phone', 'verifier').index(p) for p in active]].softmax(0)))
        s_letter, feasible = self.ctc_loglik(out['letter_logp'][row:row + 1].expand(k, -1, -1), n, letters,
                                             letter_lens, LETTER_BLANK)
        total = w['letter'] * s_letter if 'letter' in w else torch.zeros_like(s_letter)
        detail = {'letter': s_letter}
        if 'phone' in w:
            s_phone, ok = self.ctc_loglik(out['phone_logp'][row:row + 1].expand(k, -1, -1), n, phones, phone_lens,
                                          PHONE_BLANK)
            feasible = feasible & ok
            total = total + w['phone'] * s_phone
            detail['phone'] = s_phone
        if 'verifier' in w:
            mem = out['h'][row:row + 1].float().expand(k, -1, -1)
            mmask = out['mask'][row:row + 1].expand(k, -1)
            s_ver = self.verifier(mem, mmask, tokens, disputed).float()
            total = total + w['verifier'] * s_ver
            detail['verifier'] = s_ver
        total = torch.where(feasible, total, torch.full_like(total, INFEASIBLE))
        detail['feasible'] = feasible
        return total, detail

    def config_dict(self):
        return asdict(self.cfg)


def ctc_fits(targets, target_lengths, lengths):
    """(N,) bool: a CTC path exists (frames >= labels + adjacent repeats). targets concatenated (sum lengths,)."""
    tl = target_lengths.long().cpu().tolist()
    frames = lengths.long().cpu().tolist()
    t = targets.cpu()
    ok, s = [], 0
    for i, n in enumerate(tl):
        y = t[s:s + n]
        s += n
        ok.append(frames[i] >= n + int((y[1:] == y[:-1]).sum()))
    return torch.tensor(ok, device=target_lengths.device)


def candidate_tensors(cands, device):
    """Targets for score_candidates from cleaned candidate sentences: letters, phonemes, verifier tokens."""
    letters = [torch.tensor(text_to_ids(c), dtype=torch.long) for c in cands]
    phones = [sentence_phonemes(c) for c in cands]
    phone_ok = torch.tensor([p is not None and len(p) > 0 for p in phones], device=device)
    ptens = [torch.tensor(p if p else [0], dtype=torch.long) for p in phones]
    disp = disputed_words(cands)
    toks = [verifier_tokens(c, d) for c, d in zip(cands, disp)]
    width = max(len(t) for t, _ in toks)
    tok = torch.full((len(cands), width), TOKEN_PAD, dtype=torch.long)
    dis = torch.zeros((len(cands), width), dtype=torch.long)
    for i, (t, f) in enumerate(toks):
        tok[i, :len(t)] = torch.tensor(t)
        dis[i, :len(f)] = torch.tensor(f)
    return (torch.cat(letters).to(device), torch.tensor([len(x) for x in letters], device=device),
            torch.cat(ptens).to(device), torch.tensor([len(x) for x in ptens], device=device), phone_ok,
            tok.to(device), dis.to(device))


def score_texts(model, out, row, cands, device, parts=('letter', 'phone', 'verifier')):
    """S_emg of each cleaned candidate sentence for batch row `row` (INFEASIBLE where CTC cannot fit it)."""
    s, _ = model.score_candidates(out, row, *candidate_tensors(cands, device), parts=parts)
    return s


def build(cfg_dict=None, **overrides):
    cfg = SVR1Config(**{**(cfg_dict or {}), **overrides})
    return SVR1(cfg)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = build(state['config'])
    model.load_state_dict(state['model'])
    return model.to(device).eval()
