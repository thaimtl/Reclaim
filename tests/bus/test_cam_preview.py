"""cam.preview: the dashboard's mirror of the webcam, with each frame's own face points, on the camera bus.

    venv-live/bin/python -m pytest tests/bus/test_cam_preview.py -q
"""
import threading
import time

import cv2
import numpy as np
import pytest

import bus
from bus import PREVIEW, CamPreview, preview_msg
from lipread.preview import PreviewWorker

JPEG = b'\xff\xd8' + b'\x00' * 100
RNG = np.random.default_rng(2)


def face(w=1280, h=720, n=478):
    return (3.5, RNG.random((n, 2)).astype(np.float32) * [w, h], None), RNG.normal(0, 0.05, n).astype(np.float32)


def test_roundtrip_with_and_without_face():
    last_face, z = face()
    m = preview_msg(3.5, JPEG, 1280, 720, last_face, z)
    topic, msg = bus.decode(bus.encode(m, 1))
    bus.validate(topic, msg)
    assert topic == PREVIEW and msg['jpeg'] == JPEG and msg['pts'].shape == (478, 3)
    assert np.allclose(msg['pts'][:, :2], last_face[1] / [1280, 720], atol=1e-3)     # normalised like cam.face
    assert np.allclose(msg['pts'][:, 2], z, atol=1e-3)
    assert isinstance(bus.from_msg(topic, msg), CamPreview)
    topic, msg = bus.decode(bus.encode(preview_msg(3.6, JPEG, 1280, 720, None, None), 2))
    assert msg['pts'] is None


@pytest.mark.parametrize('bad', [CamPreview(1.0, 640, 360, b'not a jpeg'),
                                 CamPreview(1.0, 640, 360, JPEG, np.zeros((478, 2), np.float32)),
                                 CamPreview(1.0, 640, 360, b'\xff\xd8' + b'\x00' * (bus.PREVIEW_MAX_BYTES + 1))])
def test_encode_refuses_what_the_dashboard_cant_use(bad):
    with pytest.raises(ValueError):
        bus.encode(bad, 1)


def test_publisher_counts_previews_separately():
    pub = bus.Publisher('inproc://cam-preview-test')
    try:
        assert pub.send(preview_msg(1.0, JPEG, 640, 360, None, None), block=False) > 0
        assert pub.seq[PREVIEW] == 1 and pub.seq[bus.FACE] == 0
    finally:
        pub.close()


def test_worker_never_blocks_the_crop_thread_and_sends_the_newest_frame():
    sent, gate = [], threading.Event()

    def slow_publish(m):                       # a slow socket: the first send waits until we let it go
        gate.wait(2)
        sent.append(m)
        return 1
    w = PreviewWorker(slow_publish, hz=1000)
    img = np.zeros((720, 1280, 3), np.uint8)
    last_face, z = face()
    t0 = time.perf_counter()
    for k in range(50):                        # the crop thread hands over 50 frames while the socket is stuck
        w.submit(float(k), img, (float(k), last_face[1], None), z)
    assert time.perf_counter() - t0 < 0.05     # submit() never waits
    gate.set()
    time.sleep(0.3)
    w.close()
    assert 1 <= len(sent) <= 3 and sent[-1].t == 49.0          # skipped to the newest, not 50 sends
    assert sent[-1].w == 1280 and sent[-1].pts.shape == (478, 3)
    img_out = cv2.imdecode(np.frombuffer(sent[-1].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img_out.shape == (360, 640, 3)      # shrunk to 640 px wide


def test_worker_drops_a_face_from_another_frame():
    sent = []
    w = PreviewWorker(lambda m: sent.append(m) or 1, hz=1000)
    last_face, z = face()
    w.submit(5.0, np.zeros((360, 640, 3), np.uint8), (4.9, last_face[1], None), z)     # the face of frame 4.9
    time.sleep(0.2)
    w.close()
    assert sent and sent[-1].pts is None       # no points rather than points of the wrong frame


def test_a_failing_encode_or_send_never_stops_the_worker():
    calls = []

    def bad(m):
        calls.append(m)
        raise RuntimeError('socket gone')
    w = PreviewWorker(bad, hz=1000)
    for k in range(3):
        w.submit(float(k), np.zeros((36, 64, 3), np.uint8), None, None)
        time.sleep(0.05)
    w.close()
    assert len(calls) == 3 and w.errors == 3


def test_close_during_the_rate_limit_wait_never_hangs():
    w = PreviewWorker(lambda m: 1, hz=5)       # 200 ms between frames
    img = np.zeros((36, 64, 3), np.uint8)
    w.submit(0.0, img, None, None)
    time.sleep(0.02)
    w.submit(0.1, img, None, None)             # the worker now waits out the rate limit
    time.sleep(0.005)
    t0 = time.perf_counter()
    w.close()
    assert time.perf_counter() - t0 < 0.5 and not w.thread.is_alive()


def test_crop_thread_hands_every_frame_and_its_face_to_the_preview():
    import queue
    from lipread.run import crop_thread
    from lipread.types import Frame

    class Source:
        def frames(self):
            for k in range(5):
                yield Frame(float(k), np.zeros((36, 64, 3), np.uint8))

        def close(self):
            pass

    class Cropper:
        last_face, last_z = (0.0, np.zeros((478, 2), np.float32), None), np.zeros(478, np.float32)

        def push(self, frame):
            self.last_face = (frame.t,) + self.last_face[1:]
            return []

        def flush(self):
            return []

        def close(self):
            pass

    got = []

    class Preview:
        def submit(self, t, image, last_face, last_z):
            got.append((t, last_face[0], last_z is not None))
    crop_thread(Source(), Cropper(), queue.Queue(), threading.Event(), preview=Preview())
    assert got == [(float(k), float(k), True) for k in range(5)]


def test_preview_hz_must_be_positive():
    from lipread.run import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--preview-hz', '0'])
