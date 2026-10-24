"""lipread/emotion_net.py: fusion logic with numbers (no model), plus one model check on the sample when weights exist.

    venv-live/bin/python -m pytest tests/lipread/test_emotion_net.py -q
"""
import os

import numpy as np
import pytest

import lipread
import bus
from lipread.emotion_net import (CLASSES, MODEL_PATH, EmgTonic, EmotionEstimator, face_box, preprocess)
from bus import EMOTIONS

RNG = np.random.default_rng(0)
REST = np.array([0.5, -1.0, -1.0, -1.5, -0.5, 3.0, 0.8, -0.5])


def calibrated(**kw):
    est = EmotionEstimator(calib_reads=40, **kw)
    for _ in range(40):
        est.update(REST + RNG.normal(0, 0.3, 8), still=True)
    assert est.calibrated
    return est


def bump(cls, amount):
    x = REST.copy()
    x[CLASSES.index(cls)] += amount
    return x


def test_neutral_before_and_after_calibration():
    est = EmotionEstimator(calib_reads=40)
    assert est.update(REST, still=True).label == 'neutral'
    est = calibrated()
    for _ in range(20):
        e = est.update(REST + RNG.normal(0, 0.3, 8), still=True)
    assert e.label == 'neutral'


def test_resting_face_bias_is_not_anger():
    """Anger logit high in this person's resting face: calibration absorbs it."""
    est = calibrated()
    for _ in range(20):
        e = est.update(REST, still=True)
    assert e.label == 'neutral'


def test_sad_when_still():
    est = calibrated()
    labels = [est.update(bump('sadness', 3.0), still=True).label for _ in range(10)]
    assert labels[-1] == 'sad'
    assert labels[0] == 'neutral'


def test_mouthing_cannot_vote_surprise():
    est = calibrated()
    for _ in range(15):
        e = est.update(bump('surprise', 4.0), still=False)
    assert e.label == 'neutral'


def test_mouthing_cannot_create_sadness():
    """Negative labels update only on still reads; a sad-looking mouthing frame stream stays neutral."""
    est = calibrated()
    est.update(REST, still=True)
    for _ in range(15):
        e = est.update(bump('sadness', 3.0), still=False)
    assert e.label == 'neutral'


def test_sadness_read_before_talking_persists_briefly():
    est = calibrated()
    for _ in range(10):
        est.update(bump('sadness', 3.0), still=True)
    assert est.current.label == 'sad'
    for _ in range(5):
        e = est.update(REST, still=False)
    assert e.label == 'sad'


def test_emg_tonic_smile():
    est = calibrated()
    for _ in range(40):
        est.update(REST, still=True, emg=1.0 + RNG.normal(0, 0.05))
    for _ in range(10):
        e = est.update(REST, still=False, emg=2.0)
    assert e.label == 'happy'


def test_no_flicker_on_noise():
    est = calibrated()
    labels = [est.update(REST + RNG.normal(0, 0.3, 8), still=True).label
              for _ in range(300)]
    assert sum(a != b for a, b in zip(labels, labels[1:])) <= 2


def test_lost_face_decays_to_neutral():
    est = calibrated()
    for _ in range(10):
        est.update(bump('sadness', 3.0), still=True)
    assert est.current.label == 'sad'
    for _ in range(30):
        e = est.update(None)
    assert e.label == 'neutral'


def test_scores_shape():
    est = calibrated()
    e = est.update(REST, still=True)
    assert set(e.scores) == set(EMOTIONS)
    assert abs(sum(e.scores.values()) - 1) < 1e-6 and 0 <= e.confidence <= 1


def test_emg_tonic_separates_bursts_from_held_level():
    fs = 1000
    t = np.arange(4 * fs) / fs
    bursts = (np.sin(2 * np.pi * 4 * t) > 0.6) * RNG.normal(0, 1, t.size)
    held = RNG.normal(0, 1, t.size)
    a, b = EmgTonic(1), EmgTonic(1)
    for i in range(0, t.size, 100):
        va = a.push(bursts[i:i + 100, None])[0]
        vb = b.push(held[i:i + 100, None])[0]
    assert vb > 2 * va


def test_face_box_and_preprocess():
    pts = np.array([[100, 80], [180, 200]], np.float32)
    assert face_box(pts, (360, 640)) is not None
    assert face_box(np.array([[0, 0], [3, 3]], np.float32), (360, 640)) is None
    x = preprocess(np.zeros((50, 40, 3), np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32


HAVE_MODEL = os.path.exists(MODEL_PATH) and os.path.exists(lipread.SAMPLE_VIDEO)


@pytest.mark.skipif(not HAVE_MODEL, reason='model or sample missing')
def test_model_on_sample_frame():
    import cv2
    from lipread.emotion_net import EmotionNet
    cap = cv2.VideoCapture(lipread.SAMPLE_VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
    ok, img = cap.read()
    assert ok
    logits, val, aro = EmotionNet().run(img[40:330, 180:460])
    assert logits.shape == (8,) and np.isfinite(logits).all() and -1.5 < val < 1.5 and -1.5 < aro < 1.5


def test_calibrates_on_someone_who_never_stops_talking():
    est = EmotionEstimator(calib_reads=40)
    for _ in range(40):
        est.update(REST + RNG.normal(0, 0.3, 8), still=False)
    assert est.calibrated
    for _ in range(10):
        e = est.update(REST, still=False, tag={'happy': 0.6, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0})
    assert e.label == 'happy'


def test_tagger_smile_drives_happy_while_mouthing():
    est = calibrated()
    for _ in range(10):
        e = est.update(REST, still=False, tag={'happy': 0.6, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0})
    assert e.label == 'happy'


def test_tagger_below_its_margin_stays_neutral():
    est = calibrated()
    for _ in range(20):
        e = est.update(REST, still=False, tag={'happy': 0.2, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0})
    assert e.label == 'neutral'


def test_emphasis_brows_alone_are_not_anger():
    est = calibrated()
    for _ in range(20):
        e = est.update(REST, still=False, tag={'happy': 0.0, 'sad': 0.0, 'angry': 0.6, 'surprised': 0.0})
    assert e.label == 'neutral'


def test_anger_when_both_agree():
    est = calibrated()
    for _ in range(20):
        e = est.update(bump('anger', 3.0), still=True, tag={'happy': 0.0, 'sad': 0.0, 'angry': 0.6, 'surprised': 0.0})
    assert e.label == 'angry'


def test_emg_tonic_warms_up_before_use():
    t = EmgTonic(1, warmup_s=2.0)
    t.push(np.ones((1000, 1)))
    assert not t.ready
    t.push(np.ones((1000, 1)))
    assert t.ready


def test_recalibrate_starts_over():
    est = calibrated()
    for _ in range(10):
        est.update(bump('sadness', 3.0), still=True)
    est.recalibrate()
    assert not est.calibrated and est.current.label == 'neutral'


@pytest.mark.skipif(not HAVE_MODEL, reason='model or sample missing')
def test_worker_end_to_end_on_sample():
    """Real model + tagger + stillness through the worker thread, on 12 s of the sample at its own frame rate."""
    import time
    import cv2
    from lipread.crop import BlendTracker
    from lipread.emotion_net import EmotionWorker
    tracker = BlendTracker(os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task'))
    w = EmotionWorker(classifier=True)
    cap = cv2.VideoCapture(lipread.SAMPLE_VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, 100000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        for i in range(int(12 * fps)):
            ok, img = cap.read()
            assert ok
            t = 100 + i / fps
            obs = tracker.detect(img, t * 1000)
            w.submit(t, img, None if obs is None else (t, obs.pts, obs.blend))
            time.sleep(0.005)
    finally:
        w.close()
    assert w.reads > 20 and w.est.calibrated
    assert set(w.current().scores) == set(EMOTIONS)


def _passes_bus(emotion):
    """The Emotion survives encode -> decode -> validate exactly as cam.tick sends it."""
    import bus
    from tests.lipread.test_emotion import tick_with
    bus.validate(*bus.decode(bus.encode(tick_with(emotion))))


def test_non_finite_inputs_never_reach_the_bus():
    est = calibrated()
    for bad in (np.inf, -np.inf, np.nan):
        e = est.update(REST, still=True, emg=bad)
        e = est.update(np.full(8, bad), still=True)
        assert all(np.isfinite(v) for v in e.scores.values()) and np.isfinite(e.confidence)
        _passes_bus(e)
    for _ in range(10):
        e = est.update(REST, still=True)
    assert e.label == 'neutral'
    _passes_bus(e)


def test_emg_tonic_survives_bad_samples_and_1d_input():
    t = EmgTonic(4, warmup_s=0.0)
    t.push(np.full((100, 4), np.inf))
    assert np.isfinite(t.zi).all() and np.isfinite(t.value).all()
    t1 = EmgTonic(1, warmup_s=0.0)
    assert np.isfinite(t1.push(np.ones(50))).all()


def test_first_read_after_calibration_cannot_lock_in_anger_while_mouthing():
    """A talker calibrates while mouthing, then one emphatic read: the label must stay neutral."""
    est = EmotionEstimator(calib_reads=40)
    for _ in range(40):
        est.update(REST + RNG.normal(0, 0.3, 8), still=False)
    emphasis = {'happy': 0, 'sad': 0, 'angry': 0.85, 'surprised': 0.9}
    labels = [est.update(bump('anger', 1.0), still=False, tag=emphasis).label]
    labels += [est.update(REST, still=False, tag={'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0}).label
               for _ in range(30)]
    assert set(labels) == {'neutral'}


def test_hysteresis_needs_the_same_challenger_in_a_row():
    est = calibrated()
    seq = [bump('sadness', 4.0), bump('anger', 4.0)] * 8
    labels = [est.update(x, still=True, tag={'happy': 0, 'sad': 0.6, 'angry': 0.6, 'surprised': 0}).label for x in seq]
    assert labels[-1] in ('neutral', 'sad', 'angry')
    assert labels.count('neutral') >= 5, 'no switch on the first reads of an alternating pair'


def test_emg_floor_is_relative_to_its_level():
    est = calibrated()
    for _ in range(40):
        est.update(REST, still=True, emg=1000.0)
    for _ in range(10):
        e = est.update(REST, still=True, emg=1001.0)
    assert e.label == 'neutral'


def test_face_crop_is_square_at_the_frame_edge():
    from lipread.emotion_net import face_crop
    img = np.full((100, 200, 3), 255, np.uint8)
    box = face_box(np.array([[150, 10], [230, 90]], np.float32), img.shape)
    crop = face_crop(img, box)
    assert crop.shape[0] == crop.shape[1]


class FakeNet:
    def __init__(self):
        self.calls = 0

    def run(self, face):
        self.calls += 1
        return REST + RNG.normal(0, 0.3, 8), 0.0, 0.0


def _face(t):
    pts = np.tile(np.array([[600.0, 300.0]], np.float32), (478, 1)) + RNG.normal(0, 40, (478, 2)).astype(np.float32)
    return (t, pts, None)


def test_worker_reads_every_second_frame_at_25_fps_even_far_from_t0():
    import time
    from lipread.emotion_net import EmotionWorker
    net = FakeNet()
    w = EmotionWorker(net=net)
    img = np.zeros((720, 1280, 3), np.uint8)
    try:
        t0 = 183456.7
        for i in range(100):
            w.submit(t0 + i / 25, img, _face(t0 + i / 25))
            time.sleep(0.004)
        time.sleep(0.2)
    finally:
        w.close()
    assert 45 <= net.calls <= 50, net.calls


def test_worker_recalibrate_is_applied_on_the_worker_thread():
    import time
    from lipread.emotion_net import EmotionWorker
    w = EmotionWorker(net=FakeNet(), estimator=EmotionEstimator(calib_reads=10))
    img = np.zeros((720, 1280, 3), np.uint8)
    try:
        for i in range(60):
            w.submit(i / 25, img, _face(i / 25))
            time.sleep(0.004)
        time.sleep(0.1)
        assert w.est.calibrated
        w.recalibrate()
        w.submit(3.0, img, _face(3.0))
        w.submit(3.04, img, _face(3.04))
        time.sleep(0.1)
        assert not w.est.calibrated and w.current().label == 'neutral'
    finally:
        w.close()


def test_submit_never_raises():
    from lipread.emotion_net import EmotionWorker
    w = EmotionWorker(net=FakeNet())
    try:
        w.submit(0.0, np.zeros((10, 10, 3), np.uint8), (0.0, np.zeros((3, 2), np.float32), None))
        w.submit(0.04, None, 'garbage')
    finally:
        w.close()


def test_default_worker_is_the_spec_tagger():
    """Repo spec (EVAN_REALTIME_LIP_READING.md E5): blendshapes only. No model, no thread, label = EmotionTagger's."""
    from lipread.emotion import EmotionTagger
    from lipread.emotion_net import EmotionWorker
    w = EmotionWorker()
    ref = EmotionTagger()
    blend = np.zeros(52, np.float32)
    try:
        assert not w.classifier and w.est is None and not hasattr(w, 'net')
        for i in range(120):
            w.submit(i / 25, None, (i / 25, np.zeros((478, 2), np.float32), blend))
            want = ref.push(__import__('types').SimpleNamespace(t=i / 25, blend=blend))
        assert w.current() == want
        _passes_bus(w.current())
    finally:
        w.close()
