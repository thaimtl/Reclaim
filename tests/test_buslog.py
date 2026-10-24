"""buslog: record -> read round trip, rotation, crash-cut files, replay timing and topic filter. Uses inproc://
sockets on a shared context, never the real endpoints: python -m pytest tests/test_buslog.py"""
import contextlib
import io
import threading
import time

import msgpack
import numpy as np
import zmq

import bus
import buslog
from bus.team import Pub, Sub, decode, encode
from buslog import Writer, read, read_raw
from buslog.record import record
from buslog.replay import replay, retime

CTX = zmq.Context.instance()


def wait_until(pred, action=lambda: None, deadline_s=5.0):
    end = bus.now() + deadline_s
    while bus.now() < end:
        action()
        if pred():
            return True
        time.sleep(0.05)
    return False


def logged(folder, topic=None):
    with contextlib.redirect_stderr(io.StringIO()):  # the logger may be mid-record: no "cut short" noise
        return [r for r in read_raw(folder) if topic is None or r[2][0] == topic]


def test_record_then_read(tmp_path):
    eps = {'emg': 'inproc://t-emg', 'dec': 'inproc://t-dec'}
    emg, dec = Pub(eps['emg'], CTX), Pub(eps['dec'], CTX)
    done, folder = threading.Event(), tmp_path / 's'
    th = threading.Thread(target=record, args=(folder,), kwargs=dict(endpoints=eps, ctx=CTX, stop=done.is_set))
    th.start()
    try:
        # probe both endpoints until the logger has them on disk: then it is subscribed
        assert wait_until(lambda: folder.exists() and {r[1] for r in logged(folder, b'probe')} == {'emg', 'dec'},
                          lambda: (emg.send('probe', {}), dec.send('probe', {})))
        samples = np.arange(40, dtype=np.uint16).reshape(10, 4)
        for i in range(5):
            emg.send('emg.raw', {'t0': i * 0.01, 'fs': 1000, 'samples': samples + i, 'counter': 10 * i})
        dec.send('dec.lock', {'utt_id': 1, 'words': [{'w': 'hi'}]})
        dec.sock.send_multipart([b'dec.hyp', b'\xc1'])  # malformed: kept raw, skipped by read()
        assert wait_until(lambda: len(logged(folder)) - len(logged(folder, b'probe')) == 7)
    finally:
        done.set()
        th.join()
        emg.close(0), dec.close(0)

    raw = [r for r in read_raw(folder) if r[2][0] != b'probe']
    assert len(raw) == 7 and {r[1] for r in raw} == {'emg', 'dec'}
    assert raw[-1][2] == [b'dec.hyp', b'\xc1']
    got = [g for g in read(folder) if g[1] != 'probe']
    assert [t for _, t, _ in got] == ['emg.raw'] * 5 + ['dec.lock']
    assert all(a[0] <= b[0] for a, b in zip(got, got[1:]))  # t_recv in order
    assert np.array_equal(got[4][2]['samples'], samples + 4) and got[4][2]['counter'] == 40
    seqs = [m['seq'] for _, t, m in got if t == 'emg.raw']
    assert seqs == list(range(seqs[0], seqs[0] + 5))  # in order, none lost
    assert list(read(folder, topics=['dec.'])) == got[5:]


def test_rotation_and_crash_cut_tail(tmp_path, monkeypatch, capsys):
    w = Writer(tmp_path / 's', rotate_bytes=200)
    for i in range(20):
        w.write(float(i), 'dec', encode('dec.hyp', {'i': i}, seq=i + 1))
    w.close()
    files = sorted((tmp_path / 's').glob('bus-*.msgpack'))
    assert len(files) > 2
    last = files[-1]
    last.write_bytes(last.read_bytes() + msgpack.packb([99.0, 'dec', [b'dec.hyp', b'x' * 50]])[:-10])  # cut short
    assert [m['i'] for _, _, m in read(tmp_path / 's')] == list(range(20))
    assert 'cut short' in capsys.readouterr().err
    monkeypatch.setenv('RECLAIM_WORKDIR', str(tmp_path / 'work'))
    Writer(buslog.log_root() / '20261024-120000').close()
    assert buslog.resolve('latest') == tmp_path / 'work' / 'logs' / '20261024-120000'


def test_replay_no_retime_keeps_timing_and_bytes(tmp_path):
    # original gaps 0.4 s and 0.8 s; at speed 4 they come back as 0.1 s and 0.2 s
    w = Writer(tmp_path / 's')
    sent = [encode('cam.tick', {'i': i}, seq=i + 1) for i in range(3)]
    for t, frames in zip([10.0, 10.4, 11.2], sent):
        w.write(t, 'cam', frames)
    w.write(10.5, 'dec', encode('dec.hyp', {}))  # filtered out below
    w.close()
    eps = {'cam': 'inproc://r-cam', 'dec': 'inproc://r-dec'}
    sub = Sub([eps['cam'], eps['dec']], [''], CTX)  # inproc lets us connect before replay binds
    th = threading.Thread(target=replay, args=(tmp_path / 's',),
                          kwargs=dict(speed=4.0, topics=['cam.'], endpoints=eps, ctx=CTX, wait_s=0.3,
                                      retime_times=False))
    th.start()
    got = []
    while len(got) < 3:
        assert sub.sock.poll(3000), 'replay sent nothing'
        got.append((bus.now(), sub.sock.recv_multipart()))
    th.join()
    assert not sub.sock.poll(100)  # dec.hyp was not played
    sub.close()
    assert [f for _, f in got] == sent  # byte for byte
    gaps = [b[0] - a[0] for a, b in zip(got, got[1:])]
    assert abs(gaps[0] - 0.1) < 0.06 and abs(gaps[1] - 0.2) < 0.06 and gaps[1] > gaps[0], gaps


def test_replay_retimes_onto_the_replay_clock(tmp_path):
    w = Writer(tmp_path / 's')
    t = 5000.0  # an old session's clock
    tick = {'seg_id': 1, 'seg_start': t - 1.0, 'frame_t': np.array([t - 0.04, t]), 'greedy': 'HI'}
    w.write(t, 'cam', encode('cam.tick', tick, seq=7, t_pub=t))
    w.write(t + 0.1, 'cam', [b'cam.tick', b'\xc1'])  # malformed: forwarded as is
    w.close()
    sub = Sub(['inproc://rt-cam'], [''], CTX)
    th = threading.Thread(target=replay, args=(tmp_path / 's',),
                          kwargs=dict(endpoints={'cam': 'inproc://rt-cam'}, ctx=CTX, wait_s=0.3))
    th.start()
    assert sub.sock.poll(3000)
    now = bus.now()
    topic, msg = decode(sub.sock.recv_multipart(), validate=False)
    assert sub.sock.poll(3000)
    bad = sub.sock.recv_multipart()
    th.join()
    sub.close()
    assert topic == 'cam.tick' and msg['seq'] == 7 and msg['greedy'] == 'HI'
    assert abs(msg['t_pub'] - now) < 0.1 and abs(msg['frame_t'][-1] - now) < 0.1
    assert abs(msg['frame_t'][0] - msg['frame_t'][-1] + 0.04) < 1e-9 and abs(msg['seg_start'] - (now - 1.0)) < 0.1
    assert bad == [b'cam.tick', b'\xc1']


def test_retime_nested_and_untouched_fields():
    lock = {'utt_id': 3, 'words': [{'w': 'hi', 't': 100.0, 'conf': 0.9}, {'w': 'there', 't': 100.5}]}
    topic, msg = decode(retime(encode('dec.lock', lock, seq=4, t_pub=101.0), 10.0))
    assert [w['t'] for w in msg['words']] == [110.0, 110.5] and msg['t_pub'] == 111.0
    assert msg['seq'] == 4 and msg['utt_id'] == 3 and msg['words'][0]['conf'] == 0.9
    raw = {'t0': 2.0, 'fs': 1000, 'samples': np.zeros((10, 4), np.uint16), 'counter': 7}
    _, msg = decode(retime(encode('emg.raw', raw, seq=1, t_pub=2.0), 1.0))
    assert msg['t0'] == 3.0 and msg['fs'] == 1000 and msg['counter'] == 7
    _, msg = decode(retime(encode('dec.hyp', {'utt_id': 3, 't0': 50.0, 'locked': ['hi']}, seq=2, t_pub=51.0), 5.0))
    assert msg['t0'] == 55.0 and msg['t_pub'] == 56.0 and msg['locked'] == ['hi']


def test_read_raw_skips_bad_records_and_warns(tmp_path, capsys):
    w = Writer(tmp_path / 's')
    w.write(1.0, 'dec', encode('dec.hyp', {'i': 1}))
    w.f.write(msgpack.packb(['not', 'a record']))
    w.write(2.0, 'dec', encode('dec.hyp', {'i': 2}))
    w.close()
    assert [m['i'] for _, _, m in read(tmp_path / 's')] == [1, 2]
    assert 'skipped 1 malformed' in capsys.readouterr().err
