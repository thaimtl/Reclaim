"""Auto-AVSR (Lipflow's LipReader weights) as the camera reader needs it: re-read a whole segment, CTC top-k per frame.

    model = LiveModel(half=False)          # loads LRS3_V_WER19.1 on CUDA, warms up
    r = model.read(rois)                   # rois (T, 96, 96) uint8 crops at 25 fps -> Reading
    r.topk_ids (T, 32) int32, r.topk_logp (T, 32) float32, r.blank_logp (T,), r.greedy 'UPPERCASE TEXT', r.enc_ms ...
    r = model.read(rois, keep_enc=True)    # also r.enc, the (T, D) encoder output on the GPU (the final pass reuses it)

The encoder (3D-conv/ResNet front end + 12 Conformer blocks) runs on the whole segment every call (r.frontend_ms +
r.conformer_ms = r.enc_ms, split with CUDA events); the CTC head
(log_softmax, blank = 0) and top-k run on the GPU, one device-to-host copy per call. half=True runs the encoder in
fp16 via .half() (measured faster than autocast, CHANGELOG T0); the CTC head stays fp32.
bench_encoder(T) is the hook live/bench/laptop_check.sh calls.
"""
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

import live  # noqa: F401  (puts Lipflow on sys.path)
from live.bus import bus_text
from live.types import BLANK, TOPK


@dataclass(frozen=True)
class Reading:
    topk_ids: np.ndarray    # (T, K) int32, most likely first
    topk_logp: np.ndarray   # (T, K) float32
    blank_logp: np.ndarray  # (T,) float32
    greedy: str             # CTC best path, UPPERCASE words
    enc_ms: float           # encoder (GPU, synchronised) = frontend_ms + conformer_ms
    ctc_ms: float           # CTC head + top-k + copy to host
    enc: object = None      # read(keep_enc=True): (T, D) torch tensor on the GPU, encoder dtype; else None
    frontend_ms: float = 0.0   # host -> GPU upload + 3D conv / ResNet front end (enc_ms - conformer_ms)
    conformer_ms: float = 0.0  # embed + 12 Conformer blocks + final norm (CUDA events; 0.0 on CPU)


FE_CONTEXT = 2              # the front end's Conv3d (kernel 5 in time) sees each frame and 2 neighbours on each side


class FrontendCache:
    """Front-end features of one segment, computed once per frame (E3 step 2).

    A frame's features are final once the 2 frames after it exist; the newest 2 are recomputed each read with the
    same zero padding a whole-segment encode uses, so the result matches an encode of the whole segment
    (tests/live/test_run.py). One cache per segment: a new segment (also after a cap) starts a new one, because
    its first frames are zero-padded on the left."""

    def __init__(self, seg_id=None):
        self.seg_id = seg_id
        self.feats = None       # (1, n_final, D) front-end features of frames 0..n_final-1
        self.n_final = 0


class LiveModel:
    def __init__(self, half=False, device='cuda', warmup=range(1, 201), cudnn_benchmark=False):
        from lipflow.vsr import LipReader
        # LM only matters for the final beam pass (E6); the live re-read is CTC only
        self.reader = LipReader(device=device, use_lm=False, personal=False)
        self.device = self.reader.enc_device
        self.encoder = self.reader.model.encoder
        self.ctc = self.reader.model.ctc
        self.half = half
        # cudnn_benchmark: cuDNN times every kernel per input length during the warm-up (measured on the 5090:
        # CHANGELOG "E3"); off by default because the search can take tens of seconds and GBs of scratch memory
        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.dtype = torch.float16 if half else torch.float32
        if half:
            self.encoder.half()
        self.token_list = self.reader.token_list
        # front end / Conformer split of the encoder time: two CUDA events per read, no extra synchronise
        self._events = ((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        if self.device.type == 'cuda' else None)
        # every segment length once: the first call at a new length is slow (kernel selection, measured up to ~60 ms
        # on the 5090), and a live segment passes through all lengths up to the 8 s cap (200 frames)
        t0 = time.perf_counter()
        for t in warmup:
            self.read(np.zeros((t, 96, 96), np.uint8))
        self.warmup_s = time.perf_counter() - t0

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(self, rois, events=None):
        """(T, 96, 96) uint8 -> (T, D) encoder output in the encoder's dtype: the whole segment, no cache."""
        return self.encode_cached(rois, FrontendCache(), events)

    @torch.inference_mode()
    def encode_cached(self, rois, cache, events=None):
        """encode() with the front end run only on frames it hasn't finalised yet (plus FE_CONTEXT frames of left
        context). rois: (T, 96, 96) array or a list of (96, 96) crops; only the slice it needs is stacked.

        After the front end: the same steps as the espnet Encoder.forward (Linear+pos-enc embed, Conformer blocks,
        after_norm; tests/live/test_run.py checks it against the encoder's own forward), split from the front end so
        that events = (start, end) CUDA events time the Conformer part."""
        e = self.encoder
        T, a = len(rois), cache.n_final
        lo = max(0, a - FE_CONTEXT)
        x = self.reader.to_tensor(np.stack(rois[lo:T])).unsqueeze(0).to(self.device, self.dtype)
        new = e.frontend(x)[:, a - lo:]                               # frames a..T-1
        feats = new if cache.feats is None else torch.cat([cache.feats, new], dim=1)
        n_final = max(a, T - FE_CONTEXT)                              # frames with their 2 right neighbours seen
        cache.feats, cache.n_final = feats[:, :n_final], n_final
        if events is not None:
            events[0].record()
        x = e.embed(feats)
        x, _ = e.encoders(x, None)
        if isinstance(x, tuple):
            x = x[0]
        if e.normalize_before:
            x = e.after_norm(x)
        if events is not None:
            events[1].record()
        return x[0]

    @torch.inference_mode()
    def read(self, rois, keep_enc=False, cache=None):
        """cache: a FrontendCache for this segment (front end once per frame), or None (whole segment each read)."""
        events = self._events
        t0 = time.perf_counter()
        enc = self.encode_cached(rois, FrontendCache() if cache is None else cache, events)
        self._sync()
        t1 = time.perf_counter()
        conformer = events[0].elapsed_time(events[1]) if events is not None else 0.0
        logp = self.ctc.log_softmax(enc.float().unsqueeze(0))[0]          # (T, V)
        top = logp.topk(TOPK, dim=-1)
        out = torch.cat([top.values, logp[:, BLANK:BLANK + 1], top.indices.float()], dim=1).cpu().numpy()
        t2 = time.perf_counter()
        ids = out[:, TOPK + 1:].astype(np.int32)
        return Reading(ids, np.ascontiguousarray(out[:, :TOPK], np.float32),
                       np.ascontiguousarray(out[:, TOPK], np.float32), self.text(ids[:, 0]),
                       (t1 - t0) * 1000, (t2 - t1) * 1000, enc if keep_enc else None,
                       max(0.0, (t1 - t0) * 1000 - conformer), conformer)

    def text(self, best):
        """CTC best path (argmax id per frame) -> bus text: repeats collapsed, then live.bus.bus_text."""
        out, prev = [], None
        for i in best.tolist():
            if i != prev:
                out.append(self.token_list[i])
            prev = i
        return bus_text(''.join(out))


def bench_encoder(T=250, n=10):
    """Median ms of LiveModel.read on T zero frames, fp32 and fp16 (.half()), plus peak GPU memory."""
    out = {}
    for half in (False, True):
        torch.cuda.reset_peak_memory_stats()
        m = LiveModel(half=half, warmup=(T,))
        x = np.zeros((T, 96, 96), np.uint8)
        rs = [m.read(x) for _ in range(n)]
        name = 'fp16' if half else 'fp32'
        out[name] = (f'read median {statistics.median(r.enc_ms + r.ctc_ms for r in rs):.1f} ms (encoder '
                     f'{statistics.median(r.enc_ms for r in rs):.1f}), peak mem '
                     f'{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB')
        del m
        torch.cuda.empty_cache()
    return out
