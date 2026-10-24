"""Emotion tagger (E5) on synthetic blendshape sequences, plus the label distribution on the sample newsreader.

    venv-live/bin/python -m pytest tests/lipread/test_emotion.py -q -s     # -s prints the measured numbers
"""
import collections
import time

import numpy as np
import pytest

import lipread
import bus
from bus import decode, encode, validate
from lipread.emotion import CANON, MEDIAPIPE_NAMES, SUSTAIN_S, EmotionTagger
from bus import CamTick, EMOTIONS, TOPK
from lipread.types import FPS, Crop

from .conftest import needs_sample

Z = np.zeros((96, 96), np.uint8)
IX = {n: i for i, n in enumerate(MEDIAPIPE_NAMES)}
REST = {'browDownLeft': 0.55, 'browDownRight': 0.5, 'eyeSquintLeft': 0.5, 'eyeSquintRight': 0.45,
        'eyeBlinkLeft': 0.12, 'eyeBlinkRight': 0.1, 'mouthPressLeft': 0.1}
BROW_RAISE = {'browInnerUp': 0.6, 'browOuterUpLeft': 0.5, 'browOuterUpRight': 0.5, 'browDownLeft': 0.05,
              'browDownRight': 0.05, 'eyeWideLeft': 0.2, 'eyeWideRight': 0.2}
SMILE = {'mouthSmileLeft': 0.7, 'mouthSmileRight': 0.7, 'eyeSquintLeft': 0.75, 'eyeSquintRight': 0.7,
         'cheekSquintLeft': 0.2, 'cheekSquintRight': 0.2}
SAD = {'mouthFrownLeft': 0.45, 'mouthFrownRight': 0.45, 'browInnerUp': 0.45}
ANGRY = {'browDownLeft': 0.9, 'browDownRight': 0.88}


class Seq:
    """Builds a 25 fps Crop stream: rest/speech noise plus held expressions."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.crops = []

    def add(self, seconds, expr=None, speech=True, face=True):
        for _ in range(int(round(seconds * FPS))):
            t = len(self.crops) / FPS
            if not face:
                self.crops.append(Crop(t, Z, 0.0, False))
                continue
            b = np.zeros(52, np.float32)
            for n, v in {**REST, **(expr or {})}.items():
                b[IX[n]] = v
            if speech:
                b[IX['jawOpen']] = 0.15 + 0.15 * np.sin(2 * np.pi * 4 * t)
                b[IX['mouthPucker']] = max(0.0, 0.3 * np.sin(2 * np.pi * 3 * t))
                b += self.rng.normal(0, 0.03, 52).astype(np.float32)
            self.crops.append(Crop(t, Z, 0.1, True, blend=np.clip(b, 0, 1)))
        return self

    def run(self, tagger=None):
        tagger = tagger or EmotionTagger()
        return [tagger.push(c) for c in self.crops]


def labels(out, start_s=0.0):
    return [e.label for e in out[int(start_s * FPS):]]


def first(out, label, start_s=0.0):
    """Seconds after start_s at which `label` is first published, None if never."""
    for i, e in enumerate(out[int(start_s * FPS):]):
        if e.label == label:
            return i / FPS
    return None


def test_names_match_mediapipe():
    from lipread.crop import BLEND_NAMES
    assert len(MEDIAPIPE_NAMES) == 52 and len(set(MEDIAPIPE_NAMES)) == 52
    if BLEND_NAMES:
        assert tuple(BLEND_NAMES) == MEDIAPIPE_NAMES


def test_neutral_stays_neutral():
    out = Seq().add(3).add(12).run()
    assert set(labels(out)) == {'neutral'}
    assert min(e.confidence for e in out) > 0.8


def test_neutral_with_high_resting_brow_down():
    """A resting browDown of 0.75 (raw-threshold 'angry') is the person's neutral."""
    hi = {'browDownLeft': 0.78, 'browDownRight': 0.75}
    assert set(labels(Seq().add(3, hi).add(12, hi).run())) == {'neutral'}


def test_brow_raise_is_surprised():
    out = Seq().add(3).add(1).add(3, BROW_RAISE).run()
    lag = first(out, 'surprised', 4.0)
    print(f'\nsurprised after {lag:.2f} s')
    assert lag is not None and 0.5 <= lag <= 1.5
    assert labels(out)[-1] == 'surprised' and out[-1].confidence > 0.5


def test_smile_with_squint_is_happy():
    out = Seq().add(3).add(1).add(3, SMILE).run()
    assert first(out, 'happy', 4.0) is not None and labels(out)[-1] == 'happy'
    assert set(labels(out)[:4 * FPS]) == {'neutral'}


def test_brief_mouth_smiles_while_talking_stay_neutral():
    """Mouthed syllables pull the corners up for 0.2-0.4 s: never happy."""
    s = Seq().add(3)
    for k in range(12):
        s.add(0.2 + 0.1 * (k % 3), {'mouthSmileLeft': 0.75, 'mouthSmileRight': 0.7}).add(0.3)
    assert 0.4 < SUSTAIN_S
    assert set(labels(s.run())) == {'neutral'}


def test_sustained_mouth_smile_alone_is_happy():
    out = Seq().add(3).add(3, {'mouthSmileLeft': 0.75, 'mouthSmileRight': 0.7}).run()
    assert labels(out)[-1] == 'happy'


def test_frown_with_inner_brow_is_sad():
    out = Seq().add(3).add(1).add(3, SAD).run()
    assert first(out, 'sad', 4.0) is not None and labels(out)[-1] == 'sad'


def test_brow_down_is_angry():
    out = Seq().add(3).add(1).add(3, ANGRY).run()
    assert first(out, 'angry', 4.0) is not None and labels(out)[-1] == 'angry'


def test_back_to_neutral():
    out = Seq().add(3).add(3, BROW_RAISE).add(4).run()
    assert labels(out)[6 * FPS - 1] == 'surprised' and labels(out)[-1] == 'neutral'


def test_flicker_suppressed():
    """Expressions alternating every 0.2 s never move the published label."""
    s = Seq().add(3)
    for k in range(25):
        s.add(0.2, BROW_RAISE if k % 2 else ANGRY)
    assert set(labels(s.run())) == {'neutral'}
    s = Seq().add(3).add(3, ANGRY)
    for _ in range(5):
        s.add(0.2, BROW_RAISE).add(0.4, ANGRY)
    assert set(labels(s.run(), 6.0)) == {'angry'}


def test_face_lost_holds_then_decays():
    out = Seq().add(3).add(3, BROW_RAISE).add(8, face=False).run()
    lost = out[6 * FPS:]
    assert all(e.label == 'surprised' for e in lost[:2 * FPS])
    conf = [e.confidence for e in lost if e.label == 'surprised']
    assert all(b <= a + 1e-9 for a, b in zip(conf, conf[1:])) and conf[-1] < conf[0] * 0.5
    assert lost[-1].label == 'neutral' and lost[-1].confidence > 0.9


def test_no_face_before_baseline_and_baseline_from_face_frames_only():
    s = Seq().add(2, face=False).add(3).add(3, BROW_RAISE)
    t = EmotionTagger()
    out = [t.push(c) for c in s.crops[:5 * FPS - 1]]
    assert t.base is None and set(labels(out)) == {'neutral'}
    out += [t.push(c) for c in s.crops[5 * FPS - 1:]]
    assert t.base is not None and labels(out)[-1] == 'surprised'


def test_reset_baseline():
    t = EmotionTagger()
    out = Seq().add(3).add(3, ANGRY).run(t)
    assert out[-1].label == 'angry'
    t.reset_baseline()
    assert t.push(Seq().add(0.04, ANGRY).crops[0]).label == 'neutral'


def test_label_map():
    names = ('calm', 'joy', 'sadness', 'anger')
    lm = {'neutral': 'calm', 'happy': 'joy', 'sad': 'sadness', 'angry': 'anger', 'surprised': None}
    out = Seq().add(3).add(3, BROW_RAISE).add(1).add(3, SMILE).run(EmotionTagger(labels=names, label_map=lm))
    assert all(set(e.scores) == set(names) for e in out)
    assert set(labels(out)[:7 * FPS]) == {'calm'} and labels(out)[-1] == 'joy'
    with pytest.raises(ValueError):
        EmotionTagger(labels=names, label_map={**lm, 'happy': 'happy'})
    with pytest.raises(ValueError):
        EmotionTagger(labels=names)


def tick_with(emotion, T=3):
    logp = np.tile(-np.arange(TOPK, dtype=np.float32), (T, 1))
    ids = np.tile(np.arange(TOPK, dtype=np.int32), (T, 1))
    return CamTick(1, 100.0, 100.0 + np.arange(T) / FPS, ids, logp, np.full(T, -0.1, np.float32), 'HI',
                   np.full(T, 0.5, np.float32), True, emotion, timing_ms=bus.zero_timing(T))


def test_every_output_passes_bus_validate():
    seqs = [Seq(1).add(3).add(2, BROW_RAISE).add(2, SMILE).add(2, SAD).add(2, ANGRY).add(3, face=False).add(2),
            Seq(2).add(0.5, face=False).add(4)]
    n = 0
    for s in seqs:
        for e in s.run():
            assert e.label in EMOTIONS and set(e.scores) == set(EMOTIONS)
            validate(*decode(encode(tick_with(e))))
            n += 1
    assert n > 500 and set(CANON) == set(EMOTIONS)


def test_deterministic_and_cheap():
    s = Seq(3).add(3).add(2, BROW_RAISE).add(2, SMILE).add(20)
    a = s.run()
    t = EmotionTagger()
    t.push(s.crops[0])
    t0 = time.perf_counter()
    b = [t.push(c) for c in s.crops[1:]]
    ms = (time.perf_counter() - t0) * 1000 / len(b)
    print(f'\nemotion tagger {ms * 1000:.1f} us/frame')
    assert a[1:] == b and ms < 0.2


@needs_sample
def test_sample_newsreader_mostly_neutral(sample_video, face_model):
    from lipread.crop import BLEND_NAMES, LiveCropper
    from lipread.source import ReplaySource
    src = ReplaySource(sample_video, lipread.SAMPLE_START, lipread.SAMPLE_END, realtime=False)
    c = LiveCropper(face_model, k=0)
    crops = []
    for f in src.frames():
        crops += c.push(f)
    crops += c.flush()
    c.close()
    src.close()
    assert tuple(BLEND_NAMES) == MEDIAPIPE_NAMES
    t = EmotionTagger()
    out = [t.push(x) for x in crops]
    dist = collections.Counter(labels(out))
    peak = {k: round(max(e.scores[k] for e in out), 2) for k in EMOTIONS}
    print(f'\nsample {len(out)} frames: labels {dict(dist)}, peak scores {peak}')
    for e in out:
        validate(*decode(encode(tick_with(e))))
    assert dist['neutral'] >= 0.8 * len(out)
