"""cam.face: the dashboard's face (landmarks, blendshapes, optional head pose) on the team bus.

    venv-live/bin/python -m pytest tests/live/test_cam_face.py -q
"""
import os
import queue
import threading

import numpy as np
import pytest

import live
from live import bus
from live.bus import FACE, CamFace, face_msg

RNG = np.random.default_rng(1)


def face(n=478, pose=True):
    pts = RNG.random((n, 2)).astype(np.float32) * [1280, 720]
    return face_msg(12.5, (12.5, pts, RNG.random(52).astype(np.float32)),
                    np.array([0.1, -0.2, 0.05], np.float32) if pose else None, 1280, 720)


def roundtrip(m, seq=1):
    topic, msg = bus.decode(bus.encode(m, seq))
    bus.validate(topic, msg)
    return topic, msg


def test_roundtrip_with_and_without_pose():
    for n in (478, 468):
        for pose in (True, False):
            m = face(n, pose)
            topic, msg = roundtrip(m)
            assert topic == FACE and msg['face_ok'] is True and msg['w'] == 1280 and msg['h'] == 720
            assert msg['pts'].dtype == np.float16 and msg['pts'].shape == (n, 2)
            assert np.allclose(msg['pts'], m.pts, atol=1e-3)
            assert (msg['pose'] is None) == (not pose)
            back = bus.from_msg(topic, msg)
            assert isinstance(back, CamFace) and back.face_ok


def test_no_face_is_small_and_valid():
    m = face_msg(3.0, None, None, 1280, 720)
    frames = bus.encode(m, 1)
    assert len(frames[1]) < 200
    topic, msg = bus.decode(frames)
    bus.validate(topic, msg)
    assert msg['face_ok'] is False and msg['pts'] is None


def test_size_per_message():
    assert len(bus.encode(face(), 1)[1]) < 2500


@pytest.mark.parametrize('bad', [
    dict(pts=np.full((478, 2), np.nan, np.float32)),
    dict(pts=np.zeros((100, 2), np.float32)),
    dict(pts=np.full((478, 2), 5.0, np.float32)),
    dict(blend=np.full(52, 1.5, np.float32)),
    dict(blend=np.zeros(10, np.float32)),
    dict(pose=np.array([0, 0, 9], np.float32)),
])
def test_validate_rejects(bad):
    m = face()
    fields = {**m.__dict__, **bad}
    with pytest.raises(ValueError):
        bus.encode(CamFace(**fields), 1)


def test_no_face_must_have_no_arrays():
    with pytest.raises(ValueError):
        bus.encode(CamFace(1.0, 1280, 720, False, np.zeros((478, 2), np.float32)), 1)


def test_tick_and_final_still_need_seg_id():
    topic, msg = bus.decode(bus.encode(face(), 1))
    msg = dict(msg)
    with pytest.raises(ValueError):
        bus.validate(bus.TICK, {k: v for k, v in msg.items()})


def test_nonblocking_send_drops_instead_of_waiting(tmp_path):
    pub = bus.Publisher(f'ipc://{tmp_path}/face')
    try:
        assert pub.send(face(), block=False) > 0
        pub.lock.acquire()
        try:
            assert pub.send(face(), block=False) == 0
            assert pub.dropped_busy == 1
        finally:
            pub.lock.release()
        assert pub.seq[FACE] == 1, 'a dropped message uses no seq number'
    finally:
        pub.close()


class Frame:
    def __init__(self, t):
        self.t, self.image = t, np.zeros((720, 1280, 3), np.uint8)


class FaceSource:
    def __init__(self, n):
        self.n = n

    def frames(self):
        for i in range(self.n):
            yield Frame(i / 25)

    def close(self):
        pass


class FaceCropper:
    last_pose = None

    def __init__(self):
        self.last_face = None

    def push(self, frame):
        self.last_face = None if int(frame.t * 25) % 5 == 0 else (frame.t, RNG.random((478, 2)) * [1280, 720], None)
        return []

    def flush(self):
        return []

    def close(self):
        pass


def test_crop_thread_publishes_every_second_frame_and_survives_errors():
    from live.run import crop_thread
    sent = []
    crop_thread(FaceSource(20), FaceCropper(), queue.Queue(), threading.Event(), face_pub=sent.append)
    assert len(sent) == 10 and all(isinstance(m, CamFace) for m in sent)
    for m in sent:
        bus.validate(*bus.decode(bus.encode(m, 1)))
    assert any(not m.face_ok for m in sent) and any(m.face_ok for m in sent)

    def boom(m):
        raise RuntimeError('subscriber exploded')
    q = queue.Queue()
    crop_thread(FaceSource(10), FaceCropper(), q, threading.Event(), face_pub=boom)
    items = []
    while not q.empty():
        items.append(q.get())
    assert items and getattr(items[-1], 'error', None) is None


HAVE = os.path.exists(os.path.join(live.LIPFLOW_MODELS, 'face_landmarker.task')) and os.path.exists(live.SAMPLE_VIDEO)


@pytest.mark.skipif(not HAVE, reason='face model or sample missing')
def test_pose_flag_keeps_landmarks_bit_identical():
    import cv2
    from live.crop import BlendTracker
    cap = cv2.VideoCapture(live.SAMPLE_VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
    a, b = BlendTracker(os.path.join(live.LIPFLOW_MODELS, 'face_landmarker.task')), \
        BlendTracker(os.path.join(live.LIPFLOW_MODELS, 'face_landmarker.task'), pose=True)
    for i in range(10):
        ok, img = cap.read()
        assert ok
        oa, ob = a.detect(img, i * 40), b.detect(img, i * 40)
        assert np.array_equal(oa.pts, ob.pts) and np.array_equal(oa.blend, ob.blend)
        assert oa.pose is None and ob.pose.shape == (3,) and np.all(np.abs(ob.pose) < np.pi / 2)
