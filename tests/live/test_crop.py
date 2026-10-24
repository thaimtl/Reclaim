"""Live crop and replay source against Lipflow's offline pipeline, on the sample's test segment (20.4-28.1 s).

    venv-live/bin/python -m pytest tests/live/test_crop.py -q -s     # -s prints the measured numbers
"""
import time

import cv2
import numpy as np
import pytest

import live
from live.crop import BLEND_NAMES, SMOOTHING, LiveCropper
from live.source import Resampler, ReplaySource
from live.types import FPS

from .conftest import needs_sample


@pytest.fixture(scope='module')
def frames(sample_video):
    src = ReplaySource(sample_video, live.SAMPLE_START, live.SAMPLE_END, realtime=False)
    out = list(src.frames())
    src.close()
    return out


@pytest.fixture(scope='module')
def reference(frames, face_model):
    """Lipflow mouth_rois on the same 25 fps frames, landmarks from a full-res FaceTracker in VIDEO mode."""
    from lipflow.face import FaceTracker, mouth_rois
    tr = FaceTracker(face_model)
    anchors = []
    for f in frames:
        obs = tr.detect(f.image, int(round(f.t * 1000)))
        anchors.append(obs.anchors if obs is not None else None)
    tr.close()
    return mouth_rois([cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in frames], anchors), anchors


def run(frames, face_model, **kw):
    c = LiveCropper(face_model, **kw)
    out = []
    for f in frames:
        out += c.push(f)
    ms = list(c.times)
    out += c.flush()
    c.close()
    return out, ms


def frame_diff(crops, ref):
    """Mean abs pixel difference per frame (0-255 scale)."""
    r = np.stack([c.roi96 for c in crops]).astype(np.int16)
    return np.abs(r - ref.astype(np.int16)).mean(axis=(1, 2))


@needs_sample
def test_k6_symmetric_reproduces_mouth_rois(frames, face_model, reference):
    ref, _ = reference
    crops, _ = run(frames, face_model, k=6, symmetric=True)
    d = frame_diff(crops, ref)
    print(f'\nk=6 symmetric vs mouth_rois: {len(crops)} frames, mean {d.mean():.3f}, max {d.max():.3f}')
    assert len(crops) == len(ref)
    assert d.max() == 0


@needs_sample
def test_k6_default_matches_mouth_rois(frames, face_model, reference):
    ref, _ = reference
    crops, _ = run(frames, face_model, k=6)
    d = frame_diff(crops, ref)
    m = 6
    within = float((d <= 1.0).mean())
    print(f'\nk=6 default vs mouth_rois: mean {d.mean():.3f}, max {d.max():.3f} (edge frames), '
          f'{within:.1%} of frames <= 1.0, interior max {d[m:-m].max():.3f}')
    assert d[m:-m].max() == 0
    assert d.mean() < 0.5
    assert within >= 1 - 2 * m / len(d)
    k2, _ = run(frames, face_model, k=2)
    d2 = frame_diff(k2, ref)
    print(f'(for scale) k=2 default vs mouth_rois: mean {d2.mean():.3f}, max {d2.max():.3f}')


@needs_sample
def test_k2_counts_and_times(frames, face_model):
    c = LiveCropper(face_model, k=2)
    out = []
    for i, f in enumerate(frames):
        got = c.push(f)
        assert len(got) == (1 if i >= 2 else 0)
        out += got
    out += c.flush()
    c.close()
    assert len(out) == len(frames)
    t = [x.t for x in out]
    assert t == [f.t for f in frames]
    assert all(b > a for a, b in zip(t, t[1:]))
    assert all(x.roi96.shape == (96, 96) and x.roi96.dtype == np.uint8 for x in out)
    assert all(x.face_ok and x.mouth_open > 0 for x in out)
    assert c.crop_ms > 0


@needs_sample
def test_blendshapes_and_mouth_points(frames, face_model):
    out, _ = run(frames[:40], face_model, k=0)
    assert len(out) == 40 and all(x.face_ok for x in out)
    assert len(BLEND_NAMES) == 52 and 'browInnerUp' in BLEND_NAMES and 'mouthSmileLeft' in BLEND_NAMES
    for x in out:
        assert x.blend.shape == (52,) and x.blend.dtype == np.float32 and 0 <= x.blend.min() <= x.blend.max() <= 1
        assert x.mouth_pts.shape == (4, 2) and x.eye_dist > 10
        p = x.mouth_pts
        assert x.mouth_open == pytest.approx(np.linalg.norm(p[0] - p[1]) / (np.linalg.norm(p[2] - p[3]) + 1e-6),
                                             rel=1e-4)
    jaw = np.array([x.blend[BLEND_NAMES.index('jawOpen')] for x in out])
    assert jaw.std() > 0.01


@needs_sample
def test_smoothing_presets(frames, face_model, reference):
    ref, _ = reference
    n = 60
    with pytest.raises(ValueError):
        LiveCropper(face_model, k=2, smooth='ema')
    for name, kw in SMOOTHING.items():
        c = LiveCropper(face_model, **kw)
        got = [len(c.push(f)) for f in frames[:n]]
        out = c.flush()
        c.close()
        k = kw['k']
        assert got == [1 if i >= k else 0 for i in range(n)], name
        assert len(out) == k
    ema, _ = run(frames, face_model, **SMOOTHING['ema'])
    box, _ = run(frames, face_model, **SMOOTHING['k0'])
    d_ema, d_box = frame_diff(ema, ref), frame_diff(box, ref)
    print(f'\nvs mouth_rois: ema mean {d_ema.mean():.3f}, k0 mean {d_box.mean():.3f}')
    assert 0 < d_ema.mean() < 5 and 0 < d_box.mean() < 5


@needs_sample
def test_face_gap_fields_consistent(frames, face_model):
    seq = [type(f)(t=f.t, image=np.full_like(f.image, 100), index=f.index) if 40 <= i < 60 else f
           for i, f in enumerate(frames[:100])]
    for name, kw in SMOOTHING.items():
        out, _ = run(seq, face_model, **kw)
        assert len(out) == 100 and sum(x.face_ok for x in out) == 80, name
        for x in out:
            assert (x.blend is not None) == (x.mouth_pts is not None) == (x.eye_dist > 0) == x.face_ok, name
        assert all(out[i].roi96.any() for i in range(40, 60)), name


@needs_sample
def test_no_face_before_first_and_held_after(frames, face_model):
    blank = [type(f)(t=f.t, image=np.full_like(f.image, 100), index=f.index) for f in frames[:10]]
    seq = blank + list(frames[10:30]) + [type(f)(t=f.t + 10, image=np.full_like(f.image, 100), index=f.index + 1000)
                                         for f in frames[:5]]
    out, _ = run(seq, face_model, k=2)
    assert len(out) == len(seq)
    first = [i for i, x in enumerate(out) if x.face_ok][0]
    assert all(not x.face_ok and x.mouth_open == 0.0 for x in out[:first])
    assert all(not x.roi96.any() for x in out[:first - 2])
    assert all(x.roi96.any() for x in out[first - 2:first])
    assert all(not x.face_ok and x.mouth_open == 0.0 and x.roi96.any() for x in out[-5:])


@needs_sample
def test_replay_25fps_steady(frames):
    t = np.array([f.t for f in frames])
    d = np.diff(t)
    span = t[-1] - t[0]
    print(f'\nreplay: {len(frames)} frames over {span:.3f} s, mean interval {d.mean() * 1000:.2f} ms, '
          f'min {d.min() * 1000:.1f}, max {d.max() * 1000:.1f}')
    assert [f.index for f in frames] == list(range(len(frames)))
    assert abs(t[0] - live.SAMPLE_START) < 1e-6
    assert len(frames) == int(np.floor(span * FPS + 1e-6)) + 1
    assert abs(d.mean() - 1 / FPS) < 1e-3
    grid = t[0] + np.arange(len(t)) / FPS
    lag = t - grid
    assert lag.min() >= -1e-9 and lag.max() < 1 / 29.0


@needs_sample
def test_replay_same_frames_as_lipflow(frames, face_model):
    from lipflow.face import FaceTracker
    from lipflow.offline import load_clip
    from lipflow.vsr import LipReader
    tr = FaceTracker(face_model)
    ts, grays, _ = load_clip(live.SAMPLE_VIDEO, tr, live.SAMPLE_START, live.SAMPLE_END)
    tr.close()
    idx = LipReader.resample(ts, len(ts))
    assert [ts[i] for i in idx] == [f.t for f in frames]
    assert all(np.array_equal(grays[i], cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)) for i, f in zip(idx, frames))


@needs_sample
def test_replay_realtime_paces():
    src = ReplaySource(live.SAMPLE_VIDEO, live.SAMPLE_START, live.SAMPLE_START + 1.0, realtime=True)
    w0 = time.monotonic()
    got = [(time.monotonic() - w0, f.t) for f in src.frames()]
    src.close()
    lag = [w - (t - got[0][1]) for w, t in got]
    print(f'\nrealtime replay: {len(got)} frames, wall {got[-1][0]:.3f} s for {got[-1][1] - got[0][1]:.3f} s of video, '
          f'max lag {max(lag) * 1000:.1f} ms')
    assert len(got) == 25 or len(got) == 26
    assert max(lag) < 0.05 and min(lag) > -0.005


def test_resampler_matches_lipreader():
    from lipflow.vsr import LipReader
    rng = np.random.default_rng(0)
    for rate in (30, 29.97, 25, 60, 15):
        ts = np.cumsum(rng.uniform(0.6, 1.4, 300) / rate) + 100.0
        rs = Resampler()
        idx = [i for i, t in enumerate(ts) for _ in range(rs.push(t))]
        assert idx == LipReader.resample(list(ts), len(ts)), rate


@needs_sample
def test_downscaled_landmarks_close(frames, face_model, reference):
    _, full = reference
    for kw, p95_max in (({'detect_scale': 0.5}, 2.0), ({'face_roi': True}, 4.0)):
        c = LiveCropper(face_model, **kw)
        err = []
        for f, a in zip(frames, full):
            obs = c.detect(f.image, f.t)
            if obs is not None and a is not None:
                err.append(np.linalg.norm(obs.anchors - a, axis=1))
        c.close()
        err = np.array(err)
        print(f'\n{kw} anchors vs full-res: found {len(err)}/{len(frames)}, median {np.median(err):.2f} px, '
              f'p95 {np.percentile(err, 95):.2f} px, max {err.max():.2f} px (frame {frames[0].image.shape[1]}x'
              f'{frames[0].image.shape[0]})')
        assert len(err) == len(frames)
        assert np.percentile(err, 95) < p95_max


class FakeCap:
    """cv2.VideoCapture stand-in: grabs at the rate set with CAP_PROP_FPS (no webcam on the dev box)."""

    def __init__(self, *a):
        self.props = {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                      cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*'MJPG'))}
        self.next = None

    def isOpened(self):
        return True

    def set(self, k, v):
        if k == cv2.CAP_PROP_FPS and v not in self.offered:
            return False
        self.props[k] = float(v)
        return True

    def get(self, k):
        return self.props.get(k, 0.0)

    def grab(self):
        now = time.monotonic()
        self.next = max(self.next or now, now) + 1 / self.props[cv2.CAP_PROP_FPS]
        time.sleep(max(0.0, self.next - now))
        return True

    def retrieve(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        pass


@pytest.mark.parametrize('offered,cam_fps,keep', [((30, 60), 60, 1), ((30, 50, 60), 50, 2), ((30,), 30, 1)])
def test_camera_source_rates(monkeypatch, offered, cam_fps, keep):
    import live.source as source
    monkeypatch.setattr(FakeCap, 'offered', offered, raising=False)
    monkeypatch.setattr(source.cv2, 'VideoCapture', FakeCap)
    monkeypatch.setattr(source, 'v4l2_fps', lambda *a: [])
    monkeypatch.setattr(source, 'lock_controls', lambda *a: {})
    src = source.CameraSource(0)
    assert (src.cam_fps, src.keep_every) == (cam_fps, keep)
    got, t_end = [], time.monotonic() + 1.5
    for f in src.frames():
        got.append(f)
        if f.t > t_end:
            break
    src.close()
    got = got[5:]
    st = source.interval_stats([f.t for f in got])
    print(f'\nfake {cam_fps} fps camera, keep every {keep}: out {st}')
    assert abs(st['fps'] - FPS) < 1.0
    assert [f.index for f in got] == list(range(got[0].index, got[0].index + len(got)))


def test_slow_camera_is_resampled_to_25fps(monkeypatch):
    """A camera that only does 15 fps (readback) gets frames repeated onto the 25 fps grid, as a 15 fps replay does."""
    import live.source as source

    class SlowCap(FakeCap):
        offered = ()

        def __init__(self, *a):
            super().__init__(*a)
            self.props[cv2.CAP_PROP_FPS] = 15.0

    monkeypatch.setattr(source.cv2, 'VideoCapture', SlowCap)
    monkeypatch.setattr(source, 'v4l2_fps', lambda *a: [])
    monkeypatch.setattr(source, 'lock_controls', lambda *a: {})
    src = source.CameraSource(0)
    assert (src.cam_fps, src.keep_every) == (15.0, 1)
    got, t_end = [], time.monotonic() + 1.5
    for f in src.frames():
        got.append(f)
        if f.t > t_end:
            break
    src.close()
    ts = [f.t for f in got[5:]]
    assert len(ts) / (ts[-1] - ts[0]) == pytest.approx(FPS, abs=2.0)


@needs_sample
def test_crop_view_on_a_15fps_clip(tmp_path, face_model):
    """A video slower than 25 fps repeats frame times on the grid; crop_view pairs frames and crops FIFO (no KeyError)."""
    from live import crop_view
    cap = cv2.VideoCapture(live.SAMPLE_VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, live.SAMPLE_START * 1000)
    clip = str(tmp_path / 'clip15.mp4')
    w = None
    for i in range(60):
        ok, im = cap.read()
        if not ok:
            break
        if i % 2:
            continue
        if w is None:
            w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*'mp4v'), 15, (im.shape[1], im.shape[0]))
        w.write(im)
    w.release()
    cap.release()
    src = ReplaySource(clip, realtime=False)
    ts = [f.t for f in src.frames()]
    src.close()
    assert len(ts) > len(set(ts))
    c = LiveCropper(face_model)
    src = ReplaySource(clip, realtime=False)
    pairs = list(crop_view.paired(src, c))
    src.close()
    c.close()
    assert [f.t for f, _ in pairs] == [cr.t for _, cr in pairs] == ts
    assert [f.index for f, _ in pairs] == list(range(len(ts)))
    out = tmp_path / 'view.mp4'
    crop_view.main(['--replay', clip, '--out', str(out)])
    assert out.exists() and out.stat().st_size > 0
