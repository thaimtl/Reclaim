"""Team bus: numpy-in-msgpack round trip, validation, the stub, and real PUB/SUB sockets (tcp ephemeral port, ipc).

    venv-live/bin/python -m pytest tests/live/test_bus.py -q
"""
import time

import msgpack
import numpy as np
import pytest

import live
from live.bus import (DTYPES, FINAL, TICK, VERSION, Publisher, Subscriber, bus_text, decode, encode, from_msg, pack_nd,
                      unpack_nd, validate)
from live.stub_bus import Stub, Tokenizer
from live.types import EMOTIONS, TIMING_KEYS, TOPK, CamFinal, CamTick, Emotion


def make_tick(T=10, seg_id=1, t0=100.0):
    """A valid tick: distinct ids per row (encode refuses repeats)."""
    rng = np.random.default_rng(0)
    logp = np.sort(np.log(rng.uniform(1e-4, 1.0, (T, TOPK))), axis=1)[:, ::-1]
    scores = {e: 0.2 for e in EMOTIONS}
    return CamTick(seg_id, t0, t0 + np.arange(T) / 25, np.argsort(rng.random((T, 5049)), axis=1)[:, :TOPK], logp,
                   np.full(T, -0.1), 'WHAT DO', np.full(T, 0.5), True, Emotion('neutral', 0.2, scores),
                   timing_ms={'T': T, 'mediapipe': 14.5, 'frontend': 3.0, 'conformer': 9.25, 'ctc': 0.5, 'total': 28.0})


FINAL_MSG = CamFinal(1, (('WHAT DO YOU NEED', -4.8), ('WHY DO YOU NEED', -6.1)))


def test_pack_nd_round_trip():
    for a in (np.arange(6, dtype=np.int32).reshape(2, 3), np.linspace(0, 1, 5), np.zeros((0, 32), np.float32)):
        d = msgpack.unpackb(msgpack.packb(pack_nd(a), use_bin_type=True), raw=False)
        assert set(d) == {'nd', 'dtype', 'shape', 'data'}
        b = unpack_nd(d)
        assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)
    f = np.asfortranarray(np.arange(6).reshape(2, 3))
    assert np.array_equal(unpack_nd(pack_nd(f)), f)


def test_encode_decode_tick():
    tick = make_tick()
    frames = encode(tick, seq=7, t_pub=101.0)
    assert frames[0] == b'cam.tick'
    topic, msg = decode(frames)
    validate(topic, msg)
    assert msg['seq'] == 7 and msg['t_pub'] == 101.0 and msg['v'] == VERSION == 2
    for k, dt in DTYPES.items():
        if k in ('pts', 'blend', 'pose'):
            continue
        assert msg[k].dtype == np.dtype(dt)
    back = from_msg(topic, msg)
    assert np.array_equal(back.topk_ids, tick.topk_ids) and np.allclose(back.topk_logp, tick.topk_logp)
    assert np.array_equal(back.frame_t, tick.frame_t)
    assert back.emotion == tick.emotion and back.greedy == 'WHAT DO' and back.face_ok is True
    assert back.timing_ms == tick.timing_ms and type(back.timing_ms['T']) is int


def test_encode_decode_final():
    topic, msg = decode(encode(FINAL_MSG, seq=1))
    validate(topic, msg)
    assert from_msg(topic, msg) == FINAL_MSG


def bad(tick=None, **change):
    topic, msg = decode(encode(tick or make_tick()))
    msg.update(change)
    with pytest.raises(ValueError):
        validate(topic, msg)


def test_validate_rejects():
    T = 10
    bad(topk_ids=np.zeros((T, 16), np.int32))
    bad(topk_ids=np.zeros((T, TOPK), np.int64))
    bad(topk_logp=np.zeros((T + 1, TOPK), np.float32))
    bad(blank_logp=np.full(T, 0.5, np.float32))
    bad(activity=np.full(T, 1.5, np.float32))
    bad(frame_t=np.full(T, 100.0))
    bad(topk_ids=np.full((T, TOPK), 5049, np.int32))
    bad(greedy='what do')
    bad(seg_id=0)
    bad(emotion={'label': 'excited', 'confidence': 0.5, 'scores': {}})
    t = make_tick()
    bad(t, topk_logp=np.ascontiguousarray(t.topk_logp[:, ::-1]).astype(np.float32))
    topic, msg = decode(encode(make_tick()))
    del msg['emotion']
    with pytest.raises(ValueError):
        validate(topic, msg)
    topic, msg = decode(encode(FINAL_MSG))
    msg['nbest'] = []
    with pytest.raises(ValueError):
        validate(topic, msg)


def test_tokenizer_uses_model_ids():
    tok = Tokenizer()
    assert tok.encode('WATER') == [4899]
    assert tok.text([0, 4899, 4899, 0, 3571]) == 'WATER PLEASE'


def test_stub_messages_valid_and_cover_edge_cases():
    stub = Stub()
    now, ticks, finals = 0.0, [], []
    while now < 80:
        for m in stub.tick(now):
            topic, msg = decode(encode(m))
            validate(topic, msg)
            (ticks if topic == TICK else finals).append(msg)
        now += stub.period
    seg_ids = {t['seg_id'] for t in ticks}
    final_ids = [f['seg_id'] for f in finals]
    assert len(final_ids) == len(set(final_ids))
    assert seg_ids - set(final_ids) <= {max(seg_ids)}
    assert all(len(f['nbest']) >= 2 for f in finals)
    assert max(len(t['frame_t']) for t in ticks) == 200
    assert not all(t['face_ok'] for t in ticks)
    assert len({t['emotion']['label'] for t in ticks}) >= 3
    by_seg = {}
    for t in ticks:
        by_seg[t['seg_id']] = t
    capped = [s for s, t in by_seg.items() if len(t['frame_t']) == 200]
    nxt = by_seg[capped[0] + 1]
    assert nxt['frame_t'][0] == pytest.approx(by_seg[capped[0]]['frame_t'][-1] - 49 / 25)
    assert by_seg[1]['greedy'] == 'WHAT DO YOU NEED'


def recv_until(sub, pub, msg, want, timeout=3.0):
    """Send msg until the subscriber gets `want` messages (PUB drops while the SUB is still connecting)."""
    got, end = [], time.monotonic() + timeout
    while len(got) < want and time.monotonic() < end:
        pub.send(msg)
        r = sub.recv(timeout_ms=50)
        if r:
            got.append(r)
    return got


def test_pub_sub_real_sockets():
    pub = Publisher('tcp://127.0.0.1:*')
    a, b = Subscriber(pub.addr), Subscriber(pub.addr)
    only_final = Subscriber(pub.addr, topics=('cam.final',))
    try:
        assert recv_until(a, pub, make_tick(), 1)
        assert recv_until(b, pub, make_tick(), 1)
        assert recv_until(only_final, pub, FINAL_MSG, 1)[0][0] == FINAL
        while a.recv(timeout_ms=50):
            pass
        pub.send(make_tick())
        pub.send(FINAL_MSG)
        t1, m1 = a.recv(timeout_ms=1000)
        t2, m2 = a.recv(timeout_ms=1000)
        assert (t1, t2) == (TICK, FINAL)
        assert validate(t1, m1)['seq'] == pub.seq[TICK] and validate(t2, m2)['seq'] == pub.seq[FINAL]
        assert 0 <= time.monotonic() - m1['t_pub'] < 1.0
        while True:
            r = only_final.recv(timeout_ms=100)
            if r is None:
                break
            assert r[0] == FINAL
    finally:
        for s in (a, b, only_final, pub):
            s.close()


def test_validate_rejects_nan_bools_and_wrong_types():
    T = 10
    nan_scores = {e: float('nan') for e in EMOTIONS}
    for change in (dict(topk_logp=np.full((T, TOPK), np.nan, np.float32)),
                   dict(blank_logp=np.full(T, np.nan, np.float32)),
                   dict(activity=np.full(T, np.nan, np.float32)),
                   dict(seg_id=True), dict(seg_start='x'), dict(seg_start=float('nan')), dict(t_pub='x'),
                   dict(topk_ids=np.zeros((T, TOPK), np.int32)),
                   dict(face_ok=1), dict(greedy=None), dict(frame_t=[1.0]), dict(emotion=[1]),
                   dict(emotion={'label': 'neutral', 'confidence': float('nan'), 'scores': nan_scores}),
                   dict(emotion={'label': 'neutral', 'confidence': 0.5, 'scores': {e: 7.0 for e in EMOTIONS}})):
        bad(**change)
    topic, msg = decode(encode(make_tick()))
    msg['topk_logp'] = msg['topk_logp'].copy()
    msg['topk_logp'][:, 16:] = -np.inf
    validate(topic, msg)
    for score in (float('nan'), True, 'x'):
        topic, msg = decode(encode(FINAL_MSG))
        msg['nbest'] = [{'text': 'A', 'score': score}]
        with pytest.raises(ValueError):
            validate(topic, msg)


def test_decode_garbage_is_value_error():
    for frames in ([b'cam.tick', b'\xc1'], [b'cam.tick'], [b'a', b'b', b'c'], [b'cam.tick', msgpack.packb([1, 2])],
                   [b'\xff', msgpack.packb({})],
                   [b'cam.tick', msgpack.packb({'frame_t': {'nd': 1, 'dtype': '<f8', 'shape': [3], 'data': b'12'}})],
                   [b'cam.tick', msgpack.packb({'x': {'nd': 1, 'dtype': '|O', 'shape': [1], 'data': b'\0' * 8}})]):
        with pytest.raises(ValueError):
            decode(frames)


def test_encode_refuses_lossy_casts():
    import dataclasses
    tick = make_tick()
    with pytest.raises(TypeError):
        encode(dataclasses.replace(tick, topk_ids=tick.topk_ids + 0.7))
    with pytest.raises(TypeError):
        encode(dataclasses.replace(tick, topk_ids=np.full((10, TOPK), 2 ** 40)))
    encode(dataclasses.replace(tick, topk_logp=tick.topk_logp.astype(np.float64)))


def test_two_threads_publish_in_order():
    import threading
    pub = Publisher('tcp://127.0.0.1:*')
    sub = Subscriber(pub.addr, rcvhwm=0)
    try:
        assert recv_until(sub, pub, FINAL_MSG, 1)
        while sub.recv(timeout_ms=50):
            pass
        start = dict(pub.seq)
        threads = [threading.Thread(target=lambda o=o: [pub.send(o) for _ in range(500)])
                   for o in (make_tick(), FINAL_MSG)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = {TICK: [], FINAL: []}
        while (r := sub.recv(timeout_ms=300)) is not None:
            got[r[0]].append(validate(*r)['seq'])
        assert {k: pub.seq[k] for k in (TICK, FINAL)} == {k: start[k] + 500 for k in (TICK, FINAL)}
        for seqs in got.values():
            assert seqs and all(b > a for a, b in zip(seqs, seqs[1:]))
    finally:
        sub.close()
        pub.close()


def test_stub_rereads_change_only_the_newest_frames():
    stub = Stub()
    now, prev, steps = 0.0, {}, set()
    while now < 40:
        for m in stub.tick(now):
            if isinstance(m, CamTick):
                p = prev.get(m.seg_id)
                if p is not None:
                    n = len(p.frame_t)
                    steps.add(len(m.frame_t) - n)
                    assert np.array_equal(p.frame_t, m.frame_t[:n])
                    assert np.array_equal(p.topk_ids[:max(0, n - 4)], m.topk_ids[:max(0, n - 4)])
                prev[m.seg_id] = m
        now += stub.period
    assert steps <= {1, 2, 3}


def test_last_final_survives_close():
    for _ in range(10):
        pub = Publisher('tcp://127.0.0.1:*')
        sub = Subscriber(pub.addr)
        try:
            assert recv_until(sub, pub, make_tick(), 1)
            while sub.recv(timeout_ms=20):
                pass
            for _ in range(10):
                pub.send(make_tick(T=200))
            pub.send(FINAL_MSG)
            pub.close()
            topics = []
            while (r := sub.recv(timeout_ms=300)) is not None:
                topics.append(r[0])
            assert topics[-1] == FINAL
        finally:
            sub.close()


def test_bus_listen_counts_a_publisher_restart(tmp_path):
    import subprocess
    import sys
    addr = f'ipc://{tmp_path}/bus/cam.tick'
    listen = subprocess.Popen([sys.executable, '-m', 'live.bus_listen', '--sub', addr, '--quiet', '--duration', '3'],
                              cwd=live.ROOT, stdout=subprocess.PIPE, text=True)
    try:
        time.sleep(0.8)
        for _ in range(2):
            pub = Publisher(addr)
            time.sleep(0.3)
            for _ in range(20):
                pub.send(make_tick())
                time.sleep(0.01)
            pub.close()
        out = listen.communicate(timeout=10)[0]
    finally:
        listen.kill()
    summary = out.strip().splitlines()[-1]
    assert 'dropped=0 restarts=1' in summary, out
    assert 'pass p50=28.0 ms p95=28.0 ms' in summary, out



def test_timing_ms_round_trip_and_plain_types():
    import dataclasses
    tick = dataclasses.replace(make_tick(), timing_ms={'T': np.int64(10), 'mediapipe': np.float32(12.5),
                                                          'frontend': 2, 'conformer': 8.0, 'ctc': 0.25,
                                                          'total': np.float64(23.75)})
    topic, msg = decode(encode(tick))
    validate(topic, msg)
    assert msg['timing_ms'] == {'T': 10, 'mediapipe': 12.5, 'frontend': 2, 'conformer': 8.0, 'ctc': 0.25,
                                'total': 23.75}
    assert type(msg['timing_ms']['T']) is int and set(msg['timing_ms']) == set(TIMING_KEYS)
    assert from_msg(topic, msg).timing_ms == msg['timing_ms']
    assert live.bus.zero_timing(5) == {'T': 5, 'mediapipe': 0.0, 'frontend': 0.0, 'conformer': 0.0, 'ctc': 0.0,
                                       'total': 0.0}


def test_timing_ms_validate_rejects():
    ok = make_tick().timing_ms
    for tm in (None, [], 'x', {}, {k: v for k, v in ok.items() if k != 'ctc'},
               {**ok, 'extra': 1.0},
               {**ok, 'T': 9}, {**ok, 'T': 10.0}, {**ok, 'T': True},
               {**ok, 'total': -0.1}, {**ok, 'conformer': float('nan')}, {**ok, 'ctc': float('inf')},
               {**ok, 'mediapipe': '3'}, {**ok, 'frontend': True}, {**ok, 'total': None}):
        bad(timing_ms=tm)
    topic, msg = decode(encode(make_tick()))
    del msg['timing_ms']
    with pytest.raises(ValueError, match='timing_ms'):
        validate(topic, msg)
    bad(v=1)
    validate(*decode(encode(FINAL_MSG)))
    import dataclasses
    with pytest.raises(ValueError, match='timing_ms'):
        validate(*decode(encode(dataclasses.replace(make_tick(), timing_ms=None))))


def test_default_address_is_ipc():
    assert live.bus.DEFAULT_CAM_PUB == 'ipc:///tmp/reclaim/cam.tick'
    assert live.bus.TICK_S == 0.1 and VERSION == 2


def test_ipc_round_trip_makes_dir_and_cleans_up(tmp_path):
    path = tmp_path / 'a' / 'b' / 'cam.tick'
    pub = Publisher(f'ipc://{path}')
    sub = Subscriber(f'ipc://{path}')
    try:
        assert pub.addr == f'ipc://{path}' and path.exists()
        got = recv_until(sub, pub, make_tick(), 1)
        assert got and validate(*got[0])['timing_ms']['total'] == 28.0
        pub.send(FINAL_MSG)
        while (r := sub.recv(timeout_ms=300)) and r[0] != FINAL:
            pass
        assert r and r[0] == FINAL
        import zmq
        with pytest.raises(zmq.ZMQError):
            Publisher(f'ipc://{path}')
        assert path.exists()
    finally:
        sub.close()
        pub.close()
    assert not path.exists()
    pub = Publisher(f'ipc://{path}')
    pub.close()


def test_ipc_bind_over_stale_socket_file_after_crash(tmp_path):
    import signal
    import subprocess
    import sys
    path = tmp_path / 'cam.tick'
    addr = f'ipc://{path}'
    code = ('import sys, time; from live.bus import Publisher; p = Publisher(sys.argv[1]); '
            'print("up", flush=True); time.sleep(60)')
    crashed = subprocess.Popen([sys.executable, '-c', code, addr], cwd=live.ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert crashed.stdout.readline().strip() == 'up'
        crashed.send_signal(signal.SIGKILL)
        crashed.wait(timeout=5)
    finally:
        crashed.kill()
    assert path.exists()
    pub = Publisher(addr)
    sub = Subscriber(addr)
    try:
        assert recv_until(sub, pub, make_tick(), 1)
    finally:
        sub.close()
        pub.close()
    import socket
    u = socket.socket(socket.AF_UNIX)
    u.bind(str(path))
    u.close()
    assert path.exists()
    Publisher(addr).close()


def test_tcp_still_works_with_env_override():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, '-c', 'from live import bus; print(bus.CAM_PUB)'], cwd=live.ROOT,
                         env={**__import__('os').environ, 'RECLAIM_CAM_PUB': 'tcp://127.0.0.1:5599'},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == 'tcp://127.0.0.1:5599'


def test_close_while_sending_from_another_thread():
    import threading
    for _ in range(20):
        pub = Publisher('tcp://127.0.0.1:*')
        errors, sent, started = [], [], threading.Event()

        def final_pass():
            try:
                started.set()
                for _ in range(2000):
                    sent.append(pub.send(make_tick(T=50)))
            except Exception as e:                           # noqa: BLE001
                errors.append(e)

        th = threading.Thread(target=final_pass)
        th.start()
        started.wait()
        time.sleep(0.001)
        pub.close(linger_ms=0)
        th.join(timeout=10)
        assert not th.is_alive() and not errors, errors
        assert pub.closed and pub.send(FINAL_MSG) == 0
        assert sent and sent[-1] == 0 and pub.dropped_after_close >= 1 and len(sent) == 2000
        pub.close()


def test_stub_ticks_every_100ms_with_3_2_frames():
    import math
    stub = Stub()
    assert stub.period == 0.1
    now, ticks, timings = 0.0, [], []
    while now < 80:
        for m in stub.tick(now):
            topic, msg = decode(encode(m))
            validate(topic, msg)
            if topic == TICK:
                ticks.append(m)
                timings.append(msg['timing_ms'])
        now += stub.period
    first = [len(t.frame_t) for t in ticks if t.seg_id == 1]
    assert first[:-1] == [math.ceil(k * 2.5) for k in range(1, len(first))]
    steps = [b - a for a, b in zip(first, first[1:-1])]
    assert steps[:6] == [2, 3, 2, 3, 2, 3]
    assert all(t.frame_t[-1] <= t.frame_t[0] + k * 0.1 + 1e-9 for k, t in enumerate(ticks[:len(first)], 1))
    assert max(len(t.frame_t) for t in ticks) == 200
    assert not all(t.face_ok for t in ticks)
    assert all(tm['T'] > 0 and 0 < tm['total'] < 60 for tm in timings)
    assert max(tm['conformer'] for tm in timings) > 2 * min(tm['conformer'] for tm in timings)


def test_invalid_send_raises_and_uses_no_seq():
    import dataclasses
    pub = Publisher('tcp://127.0.0.1:*')
    try:
        pub.send(make_tick())
        with pytest.raises(ValueError, match='greedy'):
            pub.send(dataclasses.replace(make_tick(), greedy='what do'))
        assert pub.seq[TICK] == 1
        pub.send(make_tick())
        assert pub.seq[TICK] == 2
    finally:
        pub.close()


def test_bus_text_is_what_validate_accepts():
    assert bus_text('▁BORN▁IN▁NEW<eos>') == 'BORN IN NEW'
    assert bus_text("▁IT'S▁<unk>▁ok<eos>") == "IT'S OK"
    assert bus_text('<blank><eos>') == ''
    assert bus_text('▁NA<blank>N<unk>CY▁ÉTÉ-1') == 'NANCY T1'
    for raw in ('▁WHAT▁DO', '<unk>', '▁x▁', "▁DON'T"):
        validate(*decode(encode(CamFinal(1, ((bus_text(raw), -1.0),)))))
