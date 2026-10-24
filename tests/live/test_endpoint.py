"""Mouthing activity and segments (E4) on synthetic lip landmarks, plus the real sample.

    venv-live/bin/python -m pytest tests/live/test_endpoint.py -q -s     # -s prints the sample's activity stats
"""
import math

import numpy as np
import pytest

from live.endpoint import CAP, END, START, Activity, Segmenter, frames
from live.types import FPS, Crop

from .conftest import needs_sample

ROI = np.zeros((96, 96), np.uint8)
EYE = 65.0


def lips(gap, width=0.9, lift=-0.06, eye=EYE, center=(320.0, 300.0), angle=0.0, pitch=1.0):
    """mouth_pts (13, 14, 61, 291) for a lip shape in eye distances, placed rigidly in the image. pitch < 1
    foreshortens the vertical (a nod)."""
    shape = np.array([[0, -gap / 2], [0, gap / 2], [-width / 2, -gap / 2 - lift], [width / 2, -gap / 2 - lift]])
    shape[:, 1] *= pitch
    c, s = math.cos(angle), math.sin(angle)
    return (shape @ np.array([[c, s], [-s, c]]) * eye + np.asarray(center)).astype(np.float32)


def crop(i, gap=0.02, face_ok=True, eye=EYE, t0=0.0, **kw):
    if not face_ok:
        return Crop(t=t0 + i / FPS, roi96=ROI, mouth_open=0.0, face_ok=False)
    return Crop(t=t0 + i / FPS, roi96=ROI, mouth_open=gap, face_ok=True, mouth_pts=lips(gap, eye=eye, **kw),
                eye_dist=eye)


def still(n, rng, start=0, **kw):
    """Closed mouth with ~0.2 px landmark noise."""
    return [crop(start + i, gap=0.02 + rng.normal(0, 0.003), width=0.9 + rng.normal(0, 0.003), **kw)
            for i in range(n)]


def talk(n, start=0, **kw):
    """Syllables at ~4.5 Hz: the gap opens and closes, the corners narrow and widen."""
    out = []
    for i in range(start, start + n):
        ph = 2 * math.pi * 4.5 * i / FPS
        out.append(crop(i, gap=0.12 + 0.09 * math.sin(ph), width=0.88 + 0.04 * math.sin(0.7 * ph), **kw))
    return out


def run(seg, crops):
    """[(frame, kind, seg)] and the per-frame activity."""
    events, acts = [], []
    for i, c in enumerate(crops):
        events += [(i, k, s) for k, s in seg.push(c)]
        acts.append(seg.activity)
    return events, np.array(acts)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_frame_counts():
    assert (frames(0.3), frames(0.5), frames(8.0), frames(2.0), frames(0.4)) == (8, 13, 200, 50, 10)


def test_still_face_no_segment(rng):
    seg = Segmenter()
    events, acts = run(seg, still(250, rng))
    assert events == [] and seg.segment is None
    assert acts.max() < seg.still_thr


def test_burst_one_segment_with_preroll_and_end(rng):
    crops = still(25, rng) + talk(50, start=25) + still(40, rng, start=75)
    seg = Segmenter()
    events, acts = run(seg, crops)
    assert [(k, s.seg_id) for _, k, s in events] == [(START, 1), (END, 1)]
    rise = int(np.argmax(acts >= seg.start_thr))
    assert 25 <= rise <= 27
    (i_start, _, s), (i_end, _, closed) = events
    assert i_start == rise + seg.n_min - 1
    assert s is closed and closed.crops[0] is crops[rise - 8]
    assert closed.crops[-1] is crops[i_end] and len(closed.crops) == len(closed.activity)
    last_active = int(np.nonzero(acts >= seg.still_thr)[0].max())
    assert i_end == last_active + 13
    assert 0.5 <= (i_end - 74) / FPS <= 0.5 + 0.3 + 0.1
    assert seg.segment is None and seg.activity < seg.still_thr


def test_scale_invariance(rng):
    a = run(Segmenter(), talk(60))[1]
    b = run(Segmenter(), talk(60, eye=2 * EYE, center=(640.0, 600.0)))[1]
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a.max() > 0.9


def test_rigid_head_motion_no_segment(rng):
    """Nodding and turning move all lip points together: translation, tilt, vertical foreshortening."""
    crops = []
    for i in range(200):
        ph = 2 * math.pi * 1.5 * i / FPS
        crops.append(crop(i, gap=0.02 + rng.normal(0, 0.003), center=(320 + 40 * math.sin(ph), 300 + 50 * math.cos(ph)),
                          angle=0.2 * math.sin(ph), pitch=0.95 + 0.05 * math.cos(ph)))
    seg = Segmenter()
    events, acts = run(seg, crops)
    assert events == [] and acts.max() < seg.still_thr


def test_blink_no_segment(rng):
    """A blink moves the eyes only; MediaPipe's eye centres wobble a little, so eye_dist dips ~3%."""
    crops = []
    for i in range(150):
        eye = EYE * (0.97 if i % 40 in (10, 11, 12) else 1.0)
        crops.append(crop(i, gap=0.02 + rng.normal(0, 0.003), eye=eye))
    events, acts = run(Segmenter(), crops)
    assert events == [] and acts.max() < 0.25


def test_single_frame_jitter_no_segment(rng):
    crops = still(150, rng)
    for i in (30, 70, 110):
        crops[i] = crop(i, gap=0.35, width=0.7)
    seg = Segmenter()
    events, acts = run(seg, crops)
    assert events == [] and acts.max() < seg.still_thr


def test_two_frame_glitch_rejected_by_min_active(rng):
    crops = still(100, rng)
    crops[40], crops[41] = crop(40, gap=0.35), crop(41, gap=0.35)
    seg = Segmenter()
    events, acts = run(seg, crops)
    assert acts.max() >= seg.start_thr
    assert events == [] and seg.rejected == 1


def test_face_lost_mid_segment(rng):
    crops = still(10, rng) + talk(40, start=10) + [crop(50 + i, face_ok=False) for i in range(25)] \
        + talk(40, start=75) + still(40, rng, start=115)
    seg = Segmenter()
    events, acts = run(seg, crops)
    assert [(k, s.seg_id) for _, k, s in events] == [(START, 1), (END, 1), (START, 2), (END, 2)]
    assert (acts[50:75] == 0).all()
    assert events[1][0] == 50 + 12
    assert not events[1][2].crops[-1].face_ok
    assert events[2][2].crops[0].t >= crops[75 - 8].t
    assert seg.face_lost == 25


def test_cap_and_carry_over(rng):
    crops = still(25, rng) + talk(300, start=25) + still(40, rng, start=325)
    seg = Segmenter()
    events, _ = run(seg, crops)
    assert [(k, s.seg_id) for _, k, s in events] == [(START, 1), (CAP, 1), (START, 2), (END, 2)]
    (_, _, first), (i_cap, _, capped), (i_new, _, new), (_, _, ended) = events
    assert first is capped and len(capped.crops) == 200 and i_cap == i_new
    assert new is ended and new.seg_id == 2
    assert all(a is b for a, b in zip(ended.crops[:50], capped.crops[-50:]))
    assert ended.activity[:50] == capped.activity[-50:]
    assert ended.crops[50] is crops[i_cap + 1]
    assert ended.crops[50].t - capped.crops[-1].t == pytest.approx(1 / FPS)


def test_cap_segment_is_current_after_cap(rng):
    seg = Segmenter(cap_s=2.0, carry_s=0.4)
    seen = []
    for c in talk(150):
        for k, s in seg.push(c):
            if k == CAP:
                seen.append((s.seg_id, seg.segment.seg_id, len(seg.segment.crops)))
    assert seen[0] == (1, 2, 10) and [a for a, _, _ in seen] == list(range(1, len(seen) + 1))


def test_flush(rng):
    seg = Segmenter()
    run(seg, talk(40))
    assert seg.segment is not None
    out = seg.flush()
    assert [(k, s.seg_id) for k, s in out] == [(END, 1)] and len(out[0][1].crops) > 0
    assert seg.segment is None and seg.flush() == []
    events, _ = run(seg, talk(40, start=100))
    assert events[0][1] == START and events[0][2].seg_id == 2


def test_activity_zero_without_landmarks():
    act = Activity()
    assert act.push(Crop(t=0.0, roi96=ROI, mouth_open=0.0, face_ok=True)) == 0.0
    assert act.push(crop(1, face_ok=False)) == 0.0


@needs_sample
def test_sample_one_segment(sample_video, face_model):
    import live
    from live.crop import LiveCropper
    from live.source import ReplaySource
    cropper = LiveCropper(face_model, k=0)
    src = ReplaySource(sample_video, live.SAMPLE_START, live.SAMPLE_END, realtime=False)
    seg = Segmenter()
    crops, events, acts = [], [], []
    try:
        for frame in src.frames():
            for c in cropper.push(frame):
                crops.append(c)
                events += seg.push(c)
                acts.append(seg.activity)
        for c in cropper.flush():
            crops.append(c)
            events += seg.push(c)
            acts.append(seg.activity)
        events += seg.flush()
    finally:
        cropper.close()
    acts = np.array(acts)
    ends = [s for k, s in events if k == END]
    print(f'\nsample: {len(crops)} crops, activity p5/p25/p50/p75/p95 '
          f'{np.round(np.percentile(acts, [5, 25, 50, 75, 95]), 2).tolist()}, '
          f'frames < still_thr {int((acts < seg.still_thr).sum())}, '
          f'segments {[(s.seg_id, round(s.t_start - crops[0].t, 2), round(s.duration_s, 2)) for s in ends]}')
    assert [k for k, _ in events] == [START, END]
    assert ends[0].duration_s >= 0.9 * len(crops) / FPS


@pytest.mark.parametrize('pause', [20, 22, 28])
def test_back_to_back_segments_share_no_crop(rng, pause):
    """A pause just over still_s: the second segment's pre-roll only holds crops after the first one's last crop."""
    crops = talk(40) + still(pause, rng, start=40) + talk(40, start=40 + pause) + still(30, rng, start=80 + pause)
    seg = Segmenter()
    events, _ = run(seg, crops)
    ended = [s for _, k, s in events if k == END]
    assert [s.seg_id for s in ended] == [1, 2]
    first, second = ended
    assert not {id(c) for c in first.crops} & {id(c) for c in second.crops}
    assert second.crops[0].t > first.crops[-1].t
    assert len(second.crops) == len(second.activity)


def test_never_end_opens_on_the_first_crop_and_only_caps():
    seg = Segmenter(never_end=True, cap_s=4.0, carry_s=1.0)
    events = []
    for i in range(130):
        events += [(k, s.seg_id) for k, s in seg.push(crop(i, face_ok=i < 60))]
    assert events == [(START, 1), (CAP, 1), (START, 2)]
    assert len(seg.segment.crops) == 130 - 100 + 25
    assert [(k, s.seg_id) for k, s in seg.flush()] == [(END, 2)]
