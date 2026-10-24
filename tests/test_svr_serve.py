"""svr.serve: the live EMG reader's guard rails (gaps, silence, restarts, channel quality), its message, calibration
and the loop over real TCP sockets. A tiny random SVR-1 on CPU; needs the SVR-1 venv (skipped without it)."""
import json
import socket
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('svr.live', reason='needs the SVR-1 venv (scipy, soundfile, unidecode)')

from bus.team import Pub, Sub, decode, encode  # noqa: E402
from svr import serve as S  # noqa: E402
from svr.live import LiveSVR1  # noqa: E402
from tests.svr_fixture import raw_msgs, raw_signal, tiny_checkpoint  # noqa: E402


@pytest.fixture(scope='module')
def ckpt(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp('svr') / 'model.pt')


def reader(ckpt):
    return S.EmgReader(LiveSVR1(str(ckpt), device='cpu'))


def run(r, msgs, every=1):
    """Feed messages, polling after every `every`-th one at the message's sample time. Returns what poll sent."""
    out = []
    for i, m in enumerate(msgs):
        r.on_raw(m, m['t0'])
        if i % every == 0:
            out += r.poll(m['t0'] + len(m['samples']) / 1000)
    return out


def reads(out):
    return [m for t, m in out if t == 'emg.read']


def test_reads_every_100_ms_on_the_sample_clock(ckpt):
    r = reader(ckpt)
    got = reads(run(r, raw_msgs(raw_signal(1.5, bursts=[(0.5, 1.2)]), t_start=100.0)))
    assert 13 <= len(got) <= 15 and all(m['ok'] for m in got)
    m = got[-1]
    F = len(m['frame_t'])
    assert F > 0 and m['letter_logp'].shape == (F, 38) and m['letter_logp'].dtype == np.float16
    assert m['speak_prob'].shape == (F,) and m['speak_prob'].dtype == np.float32
    assert 100.0 <= m['frame_t'][0] < m['frame_t'][-1] <= 101.5 and np.all(np.diff(m['frame_t']) > 0)
    assert m['gap_ms'] == 0.0 and m['quality']['pinned'] == [0.0] * 4 and not any(m['quality']['flat'])
    assert isinstance(m['pace'], float) and isinstance(m['effort'], float)


def test_message_round_trips_on_the_bus(ckpt):
    r = reader(ckpt)
    m = reads(run(r, raw_msgs(raw_signal(0.6))))[-1]
    topic, back = decode(encode('emg.read', m, seq=1))
    assert topic == 'emg.read' and back['v'] == 2 and back['seg_start'] == m['seg_start']
    assert np.array_equal(back['letter_logp'], m['letter_logp']) and back['letter_logp'].flags.writeable


def test_short_gap_is_filled_without_a_reset(ckpt):
    r = reader(ckpt)
    msgs = list(raw_msgs(raw_signal(1.0)))
    run(r, msgs[:40])
    floor = r.live.floor
    del msgs[40:43]                                   # 30 samples lost
    got = reads(run(r, msgs[40:]))
    assert r.live.floor == floor and r.live.n == 1000  # the fill kept the sample index continuous
    assert sum(m['gap_ms'] for m in got) == 30.0 and all(m['ok'] for m in got)


def test_long_gap_or_restart_resets(ckpt):
    r = reader(ckpt)
    msgs = list(raw_msgs(raw_signal(2.0)))
    run(r, msgs[:60])
    n = r.live.n
    run(r, msgs[110:111])                             # 500 samples lost
    assert r.live.floor == n and r.live.n == n + 10
    n = r.live.n
    run(r, list(raw_msgs(raw_signal(0.05, seed=1), t_start=103.0))[:1])  # counter back to 0: publisher restarted
    assert r.live.floor == n


def test_silence_sends_one_not_ok_then_recovers(ckpt):
    r = reader(ckpt)
    msgs = list(raw_msgs(raw_signal(1.0)))
    run(r, msgs[:50])
    t_last = msgs[49]['t0'] + 0.01
    out = r.poll(t_last + 0.31) + r.poll(t_last + 0.5) + r.poll(t_last + 1.0)
    assert len(out) == 1 and out[0][0] == 'emg.read' and out[0][1]['ok'] is False
    assert len(out[0][1]['frame_t']) == 0 and out[0][1]['letter_logp'].shape == (0, 38)
    later = list(raw_msgs(raw_signal(0.6, seed=2), t_start=t_last + 2.0, counter0=500))
    got = reads(run(r, later))
    assert got and all(m['ok'] for m in got)


def test_never_connected_rig_reports_not_ok_once(ckpt):
    r = reader(ckpt)
    out = r.poll(1.0) + r.poll(2.0)
    assert [m['ok'] for _, m in out] == [False]


def test_quality_flags_flat_and_pinned_channels():
    raw = raw_signal(0.5).astype(np.int64)
    q, ok = S.quality(raw)
    assert ok and not any(q['flat'])
    raw[:, 1] = 512                                   # a loose cable: flat
    q, ok = S.quality(raw)
    assert q['flat'] == [False, True, False, False] and ok   # one bad channel is tolerated
    raw[:200, 3] = 1023                               # a pad off the skin: pinned 40 %
    q, ok = S.quality(raw)
    assert q['pinned'][3] == pytest.approx(0.4) and not ok


def test_camera_mouthing_reaches_the_reader(ckpt):
    r = reader(ckpt)
    seen = []
    r.live.read = lambda camera_active=False: seen.append(camera_active)
    r.on_tick({'activity': np.array([0.1, 0.9, 0.9, 0.9])}, 100.0)
    msgs = list(raw_msgs(raw_signal(0.3), t_start=100.0))
    run(r, msgs[:5])
    r.cam = (100.0 - 1.0, True)                       # an old tick: no longer counts
    run(r, msgs[10:20])
    assert seen[0] is True and seen[-1] is False


class FakeSub:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    def recv(self, timeout_ms=None):
        return self.msgs.pop(0) if self.msgs else (None, None)


class FakePub:
    def __init__(self):
        self.sent = []

    def send(self, topic, msg):
        self.sent.append((topic, msg))


def test_serve_survives_a_failing_read_and_bad_messages(ckpt):
    r = reader(ckpt)
    calls = iter(range(4))

    def boom():
        raise RuntimeError('CUDA error: device-side assert')
    r.poll = lambda now: boom()
    pub = FakePub()
    S.serve(r, FakeSub([('emg.raw', {'samples': np.zeros((10, 4))})]), pub, lambda: next(calls, None) is None)
    stages = [m['stage'] for t, m in pub.sent if t == 'timing.svr']
    assert stages.count('svr.error') == 4 and 'svr.bad_msg' in stages   # no t0 / counter: counted, not raised


def test_calibration_saves_and_reloads(ckpt, tmp_path):
    live = LiveSVR1(str(ckpt), device='cpu')
    sub = FakeSub([('emg.raw', m) for m in raw_msgs(raw_signal(9.0, bursts=[(1, 4), (5, 8)]))])
    path = S.calibrate(live, sub, 8.0, str(tmp_path))
    saved = json.load(open(path))
    assert len(saved['gain']) == 8 and saved['gain'] == pytest.approx(live.gain.tolist())
    fresh = LiveSVR1(str(ckpt), device='cpu')
    S.load_gains(fresh, path)
    assert fresh.gain.tolist() == pytest.approx(live.gain.tolist()) and fresh.effort_ref == live.effort_ref


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_end_to_end_over_tcp(ckpt):
    emg_addr, svr_addr = (f'tcp://127.0.0.1:{free_port()}' for _ in range(2))
    rig, out = Pub(emg_addr), Pub(svr_addr)
    sub, listen = Sub([emg_addr], ['emg.raw', 'cam.tick']), Sub([svr_addr], ['emg.read'])
    done = threading.Event()
    th = threading.Thread(target=S.serve, args=(reader(ckpt), sub, out, done.is_set))
    th.start()
    try:
        time.sleep(0.3)                               # PUB/SUB drops until the subscriptions arrive
        t0 = time.perf_counter()
        for m in raw_msgs(raw_signal(1.0), t_start=t0):
            rig.send('emg.raw', m)
            time.sleep(0.002)
        got = []
        while listen.sock.poll(1500):
            got.append(listen.recv())
            if sum(m['ok'] for _, m in got) >= 3:
                break
        assert all(t == 'emg.read' and m['v'] == 2 for t, m in got)
        assert [m['ok'] for _, m in got][-3:] == [True] * 3   # a not-ok first if it polled before the rig's samples
    finally:
        done.set()
        th.join()
        for s in (rig, out, sub, listen):
            s.close()
