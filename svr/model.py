"""EMG encoder with a CTC character head.

Same architecture and parameter names as Gaddy's architecture.Model, so his released
checkpoint loads directly with load_state_dict(strict=True):

    raw EMG (B, T*8, C)
      -> 3 strided residual conv blocks (x8 downsampling)  -> (B, T, 768)
      -> linear
      -> 6 post-norm Transformer layers, 8 heads, relative position bias within +-99 frames
      -> linear to 38 classes (37 characters + CTC blank)

Training-time augmentation (Gaddy's random 0-7 sample time shift) lives in the training
loop, not here, so the forward pass is the same in train and eval apart from dropout/BN.
"""
import torch
from torch import nn
import torch.nn.functional as F

from svr.text import NUM_CLASSES


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.residual_path is None else self.res_norm(self.residual_path(x))
        return F.relu(out + res)


class RelativePositionBias(nn.Module):
    """Learned per-head relative position embeddings, dotted with the query.

    Relative offsets beyond +-(max_distance - 1) get a -1e8 bias, so each frame only
    attends to frames within that window.
    """

    def __init__(self, num_heads, head_dim, max_distance=100):
        super().__init__()
        self.max_distance = max_distance
        # (heads, 2 * max_distance - 1 offsets, head_dim, 1); offset r is stored at index r + max_distance - 1
        self.embeddings = nn.Parameter(torch.zeros(num_heads, 2 * max_distance - 1, head_dim, 1))
        nn.init.normal_(self.embeddings, mean=0.0, std=head_dim ** -0.5)

    def forward(self, q):
        """q: (batch, heads, length, head_dim) -> bias (batch, heads, length, length)."""
        length = q.shape[2]
        emb = self.embeddings[..., 0]
        # logits against every stored offset: (b, h, query, offset)
        logits = torch.einsum('bhqd,hmd->bhqm', q, emb)
        offsets = torch.arange(length, device=q.device)
        rel = offsets[None, :] - offsets[:, None]  # rel[i, j] = j - i
        in_range = rel.abs() < self.max_distance
        index = (rel + self.max_distance - 1).clamp(0, 2 * self.max_distance - 2)
        index = index.expand(*logits.shape[:2], length, length)
        bias = torch.gather(logits, 3, index)
        return bias.masked_fill(~in_range, 0.0) - (~in_range).to(bias.dtype) * 1e8


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, relative_positional_distance=100):
        super().__init__()
        assert d_model % n_head == 0
        self.d_qkv = d_model // n_head
        self.w_q = nn.Parameter(torch.empty(n_head, d_model, self.d_qkv))
        self.w_k = nn.Parameter(torch.empty(n_head, d_model, self.d_qkv))
        self.w_v = nn.Parameter(torch.empty(n_head, d_model, self.d_qkv))
        self.w_o = nn.Parameter(torch.empty(n_head, self.d_qkv, d_model))
        for w in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_normal_(w)
        self.dropout = nn.Dropout(dropout)
        self.relative_positional = RelativePositionBias(n_head, self.d_qkv, relative_positional_distance)

    def forward(self, x):
        """x: (time, batch, d_model) -> (time, batch, d_model)."""
        q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
        k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
        v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
        logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)
        logits = logits + self.relative_positional(q)
        probs = self.dropout(F.softmax(logits, dim=-1))
        o = torch.einsum('bhqk,bhka->bhqa', probs, v)
        return torch.einsum('bhta,haf->tbf', o, self.w_o)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout, relative_positional_distance):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, relative_positional_distance)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.self_attn(x)))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout2(ff))


class TransformerStack(nn.Module):
    def __init__(self, num_layers, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(**layer_kwargs) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EMGModel(nn.Module):
    def __init__(self, in_channels=8, d_model=768, num_layers=6, n_head=8, dim_feedforward=3072,
                 dropout=0.2, relative_positional_distance=100, num_outs=NUM_CLASSES):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, d_model, 2),
            ResBlock(d_model, d_model, 2),
            ResBlock(d_model, d_model, 2),
        )
        self.w_raw_in = nn.Linear(d_model, d_model)
        self.transformer = TransformerStack(
            num_layers, d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, relative_positional_distance=relative_positional_distance)
        self.w_out = nn.Linear(d_model, num_outs)

    def forward(self, x_raw):
        """x_raw: (batch, time * 8, channels) -> logits (batch, time, num_outs)."""
        x = self.conv_blocks(x_raw.transpose(1, 2)).transpose(1, 2)
        x = self.w_raw_in(x)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.w_out(x)


def load_gaddy_checkpoint(path, device='cpu'):
    model = EMGModel()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
