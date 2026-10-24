"""Frame sources: the webcam's stall / unplug handling and thread-safe close(), and replay of recordings with
per-frame timestamps (recorder/record.py + svr/recordings.py layout, and RECORDING.md's <id>.video_t.npy + labels.csv).

    venv-live/bin/python -m pytest tests/lipread/test_source.py -q
"""
import json
import threading
import time

import cv2
import numpy as np
import pytest

import lipread.source as source
from bus import now
from lipread.source import MAX_REPEAT, ReplaySource, Resampler, find_ts, load_ts, resolve_recording
from lipread.types import FPS

SIZE = (64, 48)


class StallCap:
    """cv2.VideoCapture stand-in at 50 fps. `script` (class attribute, shared by reopens): the number of good grabs
    each opened device gives before failing for good, or None when the device does not open."""
    script = []
    opened = 0
    lock = threading.Lock()
    in_call = 0
    races = 0

    def __init__(self, *a):
        cls = type(self)
        self.good = cls.script[cls.opened] if cls.opened < len(cls.script) else None
        cls.opened += 1
        self.released = False
        self.props = {cv2.CAP_PROP_FPS: 50.0, cv2.CAP_PROP_FRAME_WIDTH: SIZE[0], cv2.CAP_PROP_FRAME_HEIGHT: SIZE[1],
                      cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*'MJPG'))}

    def isOpened(self):
        return self.good is not None

    def set(self, k, v):
        self.props[k] = float(v)
        return True

    def get(self, k):
        return self.props.get(k, 0.0)

    def _call(self, fn):
        cls = type(self)
        with cls.lock:
            if self.released:
                cls.races += 1
            cls.in_call += 1
        try:
            return fn()
        finally:
            with cls.lock:
                cls.in_call -= 1

    def grab(self):
        def g():
            time.sleep(0.02)
            if self.good:
                self.good -= 1
                return True
            return False
        return self._call(g)

    def retrieve(self):
        return self._call(lambda: (True, np.zeros((SIZE[1], SIZE[0], 3), np.uint8)))

    def release(self):
        with type(self).lock:
            if type(self).in_call:
                type(self).races += 1
            self.released = True


@pytest.fixture
def stall_cam(monkeypatch):
    monkeypatch.setattr(StallCap, 'opened', 0)
    monkeypatch.setattr(StallCap, 'races', 0)
    monkeypatch.setattr(source.cv2, 'VideoCapture', StallCap)
    monkeypatch.setattr(source, 'v4l2_fps', lambda *a: [])
    monkeypatch.setattr(source, 'lock_controls', lambda *a: {})

    def make(script, **kw):
        monkeypatch.setattr(StallCap, 'script', script)
        return source.CameraSource(0, *SIZE, **{'stall_s': 0.2, 'reopen_tries': 2, 'backoff_s': 0.05, **kw})
    return make


def drain(src, limit_s=5.0):
    w0, got = now(), []
    for f in src.frames():
        got.append(f)
        assert now() - w0 < limit_s, 'frames() did not end'
    return got, now() - w0


def test_camera_unplug_ends_the_stream(stall_cam):
    src = stall_cam([10])
    got, wall = drain(src)
    src.close()
    assert 0 < len(got) <= 10 and wall < 2.0
    assert src.error and 'lost' in src.error and src.reopens == 0 and src.grab_failures > 0
    assert StallCap.opened == 3
    assert StallCap.races == 0


def test_camera_recovers_after_a_stall(stall_cam):
    src = stall_cam([10, 20])
    got, _ = drain(src)
    src.close()
    assert src.reopens == 1 and src.error
    assert len(got) > 12
    assert [f.index for f in got] == list(range(len(got)))
    assert StallCap.races == 0


def test_camera_close_from_another_thread(stall_cam):
    src = stall_cam([10 ** 6])
    threading.Timer(0.3, src.close).start()
    got, wall = drain(src)
    assert got and wall < 1.5 and src.error is None
    src.close()
    assert StallCap.races == 0


def test_camera_close_during_a_stall(stall_cam):
    src = stall_cam([5], stall_s=10.0)
    threading.Timer(0.4, src.close).start()
    got, wall = drain(src)
    assert wall < 1.5 and src.error is None and StallCap.races == 0


def level(i):
    return 10 + 12 * (i % 20)


def write_clip(path, n, fps=30):
    """Flat frames whose grey level encodes the frame number (mp4v; flat frames decode within a level or two)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, SIZE)
    for i in range(n):
        w.write(np.full((SIZE[1], SIZE[0], 3), level(i), np.uint8))
    w.release()


def level_of(im):
    return int(round((im.mean() - 10) / 12))


def vfr_ts(n, rng, t0=1234.5, gaps=()):
    """Jittered ~30 fps capture times on a monotonic clock, with extra gaps {after frame i: seconds}."""
    d = rng.uniform(0.7, 1.3, n - 1) / 30
    for i, g in dict(gaps).items():
        d[i] += g
    return t0 + np.concatenate([[0.0], np.cumsum(d)])


def expected(ts, max_repeat=MAX_REPEAT):
    rs = Resampler(max_repeat=max_repeat)
    return [i for i, t in enumerate(ts - ts[0]) for _ in range(rs.push(t))]


def test_replay_uses_sibling_timestamps(tmp_path):
    rng = np.random.default_rng(0)
    n = 90
    ts = vfr_ts(n, rng, gaps={30: 0.08, 60: 0.4})
    write_clip(tmp_path / 'u1.mp4', n)
    np.save(tmp_path / 'u1_video_ts.npy', ts)
    src = ReplaySource(str(tmp_path / 'u1.mp4'), realtime=False)
    assert src.ts_path == str(tmp_path / 'u1_video_ts.npy') and src.t_offset == ts[0] and src.ts_gaps == 2
    got = list(src.frames())
    src.close()
    idx = expected(ts)
    assert [f.t for f in got] == pytest.approx(list(ts[idx] - ts[0]))
    assert [level_of(f.image) for f in got] == [i % 20 for i in idx]
    assert [f.index for f in got] == list(range(len(got)))
    d = np.diff([f.t for f in got])
    steady = d[d < 0.1]
    assert abs(steady.mean() - 1 / FPS) < 0.004
    assert d.max() > 0.4
    plain = ReplaySource(str(tmp_path / 'u1.mp4'), realtime=False, ts=None)
    assert plain.ts is None and [f.t for f in plain.frames()] == pytest.approx(
        [i / 30 for i in expected(np.arange(n) / 30, None)])
    plain.close()


def test_replay_start_end_are_seconds_into_the_clip(tmp_path):
    rng = np.random.default_rng(1)
    ts = vfr_ts(60, rng)
    write_clip(tmp_path / 'u.mp4', 60)
    np.save(tmp_path / 'u_video_ts.npy', ts)
    src = ReplaySource(str(tmp_path / 'u.mp4'), 0.5, 1.2, realtime=False)
    got = list(src.frames())
    src.close()
    rel = ts - ts[0]
    first = int(np.argmax(rel >= 0.5))
    assert got[0].t == pytest.approx(rel[first]) and level_of(got[0].image) == first % 20
    assert got[-1].t <= 1.2 and got[-1].t > 1.2 - 2 / 30


def test_replay_realtime_follows_the_recorded_times(tmp_path):
    rng = np.random.default_rng(2)
    ts = vfr_ts(20, rng, gaps={10: 0.06})
    write_clip(tmp_path / 'u.mp4', 20)
    np.save(tmp_path / 'u_video_ts.npy', ts)
    src = ReplaySource(str(tmp_path / 'u.mp4'))
    w0 = now()
    got = [(now() - w0, f.t) for f in src.frames()]
    src.close()
    lag = [w - t for w, t in got]
    assert max(lag) < 0.05 and min(lag) > -0.005
    assert got[-1][0] == pytest.approx(ts[-1] - ts[0], abs=0.05)


def test_replay_more_frames_than_timestamps(tmp_path):
    ts = 5.0 + np.arange(10) / 30
    write_clip(tmp_path / 'u.mp4', 15)
    np.save(tmp_path / 'u_video_ts.npy', ts)
    src = ReplaySource(str(tmp_path / 'u.mp4'), realtime=False)
    assert max(level_of(f.image) for f in src.frames()) == 9
    src.close()


def test_replay_close_from_another_thread(tmp_path):
    write_clip(tmp_path / 'u.mp4', 120)
    src = ReplaySource(str(tmp_path / 'u.mp4'))
    threading.Timer(0.3, src.close).start()
    w0 = now()
    got = list(src.frames())
    assert now() - w0 < 0.6 and 5 < len(got) < 15
    time.sleep(0.1)
    assert src.cap is None
    src.close()


def test_resolve_manifest_id_svr_layout(tmp_path):
    """svr/recordings.py / recorder/record.py: recordings/<session>/manifest.jsonl + <id>.mp4 + <id>_video_ts.npy."""
    d = tmp_path / 's1'
    d.mkdir()
    write_clip(d / 's1_C_0001.mp4', 30)
    np.save(d / 'ts_elsewhere.npy', vfr_ts(30, np.random.default_rng(3)))
    (d / 'manifest.jsonl').write_text(json.dumps({'id': 's1_B_0001', 'video': None}) + '\n' + json.dumps(
        {'id': 's1_C_0001', 'session': 's1', 'video': 's1_C_0001.mp4', 'video_ts': 'ts_elsewhere.npy'}) + '\n')
    assert resolve_recording('s1_C_0001', str(tmp_path)) == (str(d / 's1_C_0001.mp4'), str(d / 'ts_elsewhere.npy'))
    src = ReplaySource('s1_C_0001', realtime=False, root=str(tmp_path))
    assert src.ts is not None and len(list(src.frames())) > 20
    src.close()
    with pytest.raises(FileNotFoundError):
        resolve_recording('nope', str(tmp_path))


def test_resolve_recording_md_layout(tmp_path):
    """RECORDING.md: recordings/labels.csv + session_01/s1_001.mp4 + s1_001.video_t.npy."""
    d = tmp_path / 'session_01'
    d.mkdir()
    write_clip(d / 's1_001.mp4', 30)
    ts = vfr_ts(30, np.random.default_rng(4))
    np.save(d / 's1_001.video_t.npy', ts)
    (tmp_path / 'labels.csv').write_text('id,session,set,pace,sentence,emg,video,speaker\n'
                                         's1_001,1,train,natural,"Hi, there.",session_01/s1_001.emg.npy,'
                                         'session_01/s1_001.mp4,austin\n')
    assert resolve_recording('s1_001', str(tmp_path)) == (str(d / 's1_001.mp4'), str(d / 's1_001.video_t.npy'))
    assert resolve_recording(str(d / 's1_001'), str(tmp_path))[0] == str(d / 's1_001.mp4')
    src = ReplaySource(str(d / 's1_001.mp4'), 0.0, None, realtime=False)
    assert src.ts_path == str(d / 's1_001.video_t.npy')
    src.close()


def test_load_ts_csv(tmp_path):
    p = tmp_path / 'u_video_ts.csv'
    p.write_text('t,frame\n1.5,0\n1.53,1\n\n1.6,2\n')
    np.testing.assert_allclose(load_ts(str(p)), [1.5, 1.53, 1.6])
    (tmp_path / 'u.mp4').write_bytes(b'')
    assert find_ts(str(tmp_path / 'u.mp4')) == str(p)
