"""E6 final pass: submit never blocks, queue policy, cam.final valid on the bus, clean close, and the real beam + LM on
the sample (the fixes greedy misses).

    venv-live/bin/python -m pytest tests/live/test_final.py -q
"""
import threading
import time
from collections import Counter

import numpy as np
import pytest

import live
from live.bus import decode, encode, validate
from live.final import FinalPass
from live.types import CamFinal

from .conftest import needs_cuda, needs_sample, needs_weights


class SlowBeam:
    """Fake beam: sleeps, records what it decoded; seg ids come from the rois' first pixel."""

    def __init__(self, s=0.2, out=(('HELLO WORLD', -1.5), ('HELLO WORD', -2.5))):
        self.s, self.out, self.calls = s, out, []
        self.started = threading.Event()

    def __call__(self, rois, enc):
        self.started.set()
        self.calls.append(int(rois[0, 0, 0]))
        time.sleep(self.s)
        return self.out


def rois(seg_id, T=50):
    return np.full((T, 96, 96), seg_id, np.uint8)


def make(beam, **kw):
    sent = []
    return FinalPass(None, sent.append, beam=beam, **kw), sent


def test_submit_never_blocks_while_busy():
    beam = SlowBeam(0.3)
    fp, sent = make(beam, max_pending=2)
    ms = []
    for i in range(1, 8):
        t0 = time.perf_counter()
        assert fp.submit(i, rois(i))
        ms.append((time.perf_counter() - t0) * 1000)
        if i == 1:
            assert beam.started.wait(1.0)
    assert max(ms) < 5, ms
    assert fp.close(timeout=3.0)


def test_queue_keeps_the_newest_and_counts_drops():
    beam = SlowBeam(0.2)
    fp, sent = make(beam, max_pending=2)
    fp.submit(1, rois(1))
    assert beam.started.wait(1.0)           # 1 running
    for i in (2, 3, 4, 5):                  # 2 waiting at most: 2 and 3 dropped, 4 and 5 kept
        fp.submit(i, rois(i))
    assert fp.wait_idle(3.0)
    assert fp.dropped == [2, 3]
    assert beam.calls == [1, 4, 5]
    assert [m.seg_id for m in sent] == [1, 4, 5]
    assert [s['seg_id'] for s in fp.stats] == [1, 4, 5]
    assert all(s['T'] == 50 and s['beam_ms'] >= 150 and s['queued_ms'] >= 0 for s in fp.stats)
    assert fp.stats[1]['queued_ms'] >= 150  # waited for segment 1's beam
    assert fp.close(1.0)


def test_cap_then_continuation_both_get_a_final():
    """A capped 8 s segment and its continuation closing while the first beam still runs: nothing dropped."""
    beam = SlowBeam(0.2)
    fp, sent = make(beam)
    fp.submit(1, rois(1, 200))
    fp.submit(2, rois(2, 60))
    assert fp.close(3.0)
    assert [m.seg_id for m in sent] == [1, 2] and fp.dropped == []


def test_published_finals_are_valid_on_the_bus():
    outs = [(('HELLO WORLD', -1.5), ('HELLO WORD', -2.5)), (('', -3.0),), (("IT'S 1943", -0.25),)]
    for out in outs:
        fp, sent = make(SlowBeam(0.0, out))
        fp.submit(7, rois(7))
        assert fp.close(2.0)
        assert len(sent) == 1 and isinstance(sent[0], CamFinal)
        topic, msg = decode(encode(sent[0], seq=1))
        validate(topic, msg)
        assert topic == 'cam.final' and msg['seg_id'] == 7
        assert [(n['text'], n['score']) for n in msg['nbest']] == list(out)


def test_close_finishes_or_drops_and_refuses_new_work():
    beam = SlowBeam(0.3)
    fp, sent = make(beam)
    for i in (1, 2, 3):
        fp.submit(i, rois(i))
    assert beam.started.wait(1.0)
    t0 = time.perf_counter()
    done = fp.close(timeout=0.45)           # 1 finishes, 2 starts (cannot be interrupted), 3 is dropped
    assert time.perf_counter() - t0 < 0.6
    assert not done or fp.dropped
    assert 3 in fp.dropped and not fp.submit(4, rois(4))
    fp.thread.join(1.0)
    assert not fp.thread.is_alive()
    assert [m.seg_id for m in sent] == [1, 2]


def test_a_failing_beam_is_logged_and_the_worker_goes_on():
    calls = []

    def beam(r, enc):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('boom')
        return (('OK', -1.0),)

    fp, sent = make(beam)
    fp.submit(1, rois(1))
    fp.submit(2, rois(2))
    assert fp.close(2.0)
    assert fp.errors == [1] and [m.seg_id for m in sent] == [2]


def sample_rois(sample_video, face_model):
    from live.crop import LiveCropper
    from live.source import ReplaySource
    src = ReplaySource(sample_video, live.SAMPLE_START, live.SAMPLE_END, realtime=False)
    cropper = LiveCropper(face_model, k=0)
    crops = [c for f in src.frames() for c in cropper.push(f)] + list(cropper.flush())
    src.close()
    cropper.close()
    return np.stack([c.roi96 for c in crops])


def check_sample_nbest(nbest):
    texts = [t for t, _ in nbest]
    scores = [s for _, s in nbest]
    assert 1 <= len(nbest) <= 5 and len(set(texts)) == len(texts)
    assert scores == sorted(scores, reverse=True) and all(np.isfinite(scores))
    assert any('AND RAISED MOSTLY' in t for t in texts), texts
    assert any('NINETEEN FORTY THREE' in t for t in texts), texts
    truth = Counter(live.SAMPLE_TEXT.split())
    hit = sum((truth & Counter(texts[0].split())).values())
    assert hit >= 15, (hit, texts[0])


@needs_weights
@needs_cuda
@needs_sample
def test_real_beam_on_the_sample(sample_video, face_model):
    """The default design: the tick's encoder output -> beam + LM in the CPU process. Then the in-process GPU Beam on
    the rois gives the same 5-best."""
    from live.final import Beam
    from live.model import LiveModel
    x = sample_rois(sample_video, face_model)
    assert len(x) == 192
    model = LiveModel(warmup=(25,))
    r = model.read(x, keep_enc=True)
    assert r.enc is not None and tuple(r.enc.shape) == (192, model.reader.model.adim)
    assert model.read(x).enc is None                        # default unchanged
    sent = []
    fp = FinalPass(model, sent.append)
    t0 = time.perf_counter()
    assert fp.submit(3, x, enc=r.enc)
    assert (time.perf_counter() - t0) * 1000 < 5
    assert fp.close(timeout=30.0)
    assert len(sent) == 1 and sent[0].seg_id == 3
    validate(*decode(encode(sent[0])))
    check_sample_nbest(sent[0].nbest)
    gpu = Beam(model)(x)
    check_sample_nbest(gpu)
    assert gpu[0][0] == sent[0].nbest[0][0]                 # CPU and GPU agree on the best
    print(f'\nbeam {fp.stats[0]["beam_ms"]:.0f} ms (process, CPU), best: {sent[0].nbest[0]}')
