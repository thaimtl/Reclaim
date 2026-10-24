"""Facial expression -> emotion tag (E5): one Emotion per Crop from MediaPipe's 52 blendshapes, for cam.tick.

The mouth is busy mouthing words, so the tag leans on the upper face; mouth shapes count only when held longer than
a mouthed syllable (SUSTAIN_S). Everything is scored as a change from the person's own neutral face: the median of
each blendshape over the first baseline_s of frames with a face (a resting news speaker already reads browDown ~0.6,
eyeSquint ~0.5, so raw thresholds fail; the median survives talking during the baseline).

Evidence per frame, d(x) = max(0, x - baseline x), ramp(v, lo, hi) = clip((v - lo) / (hi - lo), 0, 1),
a | b = 1 - (1 - a)(1 - b) (noisy-or), L/R pairs averaged:
    surprised  0.7 ramp(d browOuterUp) | 0.3 ramp(d browInnerUp) | 0.5 ramp(d eyeWide), times (1 - frown)
    angry      0.85 ramp(d browDown) | 0.3 squint
    happy      0.8 smile | 0.5 squint (1 - ramp(d browDown))          squint = ramp(max(d eyeSquint, d cheekSquint))
    sad        frown (0.55 + 0.45 ramp(d browInnerUp) (1 - ramp(d browOuterUp)))
smile / frown: ramp(d mouthSmile / d mouthFrown) once it has stayed above SMILE_ON / FROWN_ON for SUSTAIN_S (dips
shorter than GAP_S are bridged), ramping in over SUSTAIN_RAMP_S; 0 before.
Scores are EMA-smoothed with time constant smooth_s / 2 (on crop.t deltas); neutral = 1 - max(other scores). The
candidate is the best non-neutral label if its score >= MARGIN, else neutral; the published label switches only once
the same candidate has won for switch_s. No face: the label is held while the scores decay to 0 with NOFACE_TAU_S;
once the held label's score is below MARGIN / 2 it drops to neutral. Before the baseline is complete: neutral.

Labels: the canonical set CANON is scored internally; label_map {canonical: published name or None} renames them
(None drops one, e.g. if the TTS has no 'surprised'); two canonicals may share a name (max wins). Output scores have
exactly the `labels` keys, all in [0, 1], so every Emotion passes bus.validate.

    tagger = EmotionTagger()                                    # labels=EMOTIONS, label_map={'happy': 'joy', ...}
    for crop in crops:
        emotion = tagger.push(crop)                             # Emotion(label, confidence, scores), ~30 us
    tagger.reset_baseline()                                     # new person in front of the camera
"""
import math

import numpy as np

from bus import EMOTIONS, Emotion
from lipread.types import FPS

CANON = ('neutral', 'happy', 'sad', 'angry', 'surprised')
NEUTRAL = 'neutral'
MEDIAPIPE_NAMES = (
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
    'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
    'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight')
FEATURES = {'brow_down': ('browDownLeft', 'browDownRight'), 'brow_inner': ('browInnerUp',),
            'brow_outer': ('browOuterUpLeft', 'browOuterUpRight'), 'eye_wide': ('eyeWideLeft', 'eyeWideRight'),
            'eye_squint': ('eyeSquintLeft', 'eyeSquintRight'), 'cheek_squint': ('cheekSquintLeft', 'cheekSquintRight'),
            'smile': ('mouthSmileLeft', 'mouthSmileRight'), 'frown': ('mouthFrownLeft', 'mouthFrownRight')}
RAMP = {'brow_down': (0.12, 0.30), 'brow_inner': (0.20, 0.50), 'brow_outer': (0.10, 0.35), 'eye_wide': (0.08, 0.25),
        'squint': (0.10, 0.25), 'smile': (0.25, 0.50), 'frown': (0.15, 0.35)}
SMILE_ON, FROWN_ON = 0.25, 0.15
SUSTAIN_S = 0.6
SUSTAIN_RAMP_S = 0.3
GAP_S = 0.12
MARGIN = 0.35
NOFACE_TAU_S = 2.0
MAX_DT = 0.5


def _ramp(v, lo_hi):
    lo, hi = lo_hi
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def _or(*xs):
    p = 1.0
    for x in xs:
        p *= 1.0 - x
    return 1.0 - p


class _Sustain:
    """Seconds a value has stayed above `on` (dips < GAP_S bridged) -> 0..1 gate."""

    def __init__(self, on):
        self.on = on
        self.since = self.last = None

    def push(self, v, t):
        if v >= self.on:
            if self.since is None or t - self.last > GAP_S:
                self.since = t
            self.last = t
        elif self.last is not None and t - self.last > GAP_S:
            self.since = self.last = None
        if self.since is None:
            return 0.0
        return min(1.0, max(0.0, (t - self.since - SUSTAIN_S) / SUSTAIN_RAMP_S + 1.0 / FPS / SUSTAIN_RAMP_S))

    def reset(self):
        self.since = self.last = None


class EmotionTagger:
    def __init__(self, baseline_s=3.0, smooth_s=1.0, switch_s=0.5, labels=EMOTIONS, label_map=None, margin=MARGIN):
        self.baseline_n = max(1, int(round(baseline_s * FPS)))
        self.tau = smooth_s / 2
        self.switch_s, self.margin = switch_s, margin
        self.labels = tuple(labels)
        self.label_map = {c: c for c in CANON if c in self.labels} if label_map is None else dict(label_map)
        if self.label_map.get(NEUTRAL) not in self.labels:
            raise ValueError(f'label_map must map {NEUTRAL!r} to one of {self.labels}')
        bad = {c: n for c, n in self.label_map.items() if c not in CANON or (n is not None and n not in self.labels)}
        if bad:
            raise ValueError(f'label_map: canonical names {CANON} -> one of {self.labels} or None, got {bad}')
        self.active = [c for c in CANON[1:] if self.label_map.get(c) is not None]
        self.idx = None
        self.reset_baseline()

    def reset_baseline(self):
        """Forget the person: collect a new baseline from the next baseline_s of face frames, back to neutral."""
        self.base_frames, self.base = [], None
        self.smooth = dict.fromkeys(CANON[1:], 0.0)
        self.label, self.cand, self.cand_since, self.t = NEUTRAL, NEUTRAL, None, None
        self.smile, self.frown = _Sustain(SMILE_ON), _Sustain(FROWN_ON)

    def _indices(self):
        from lipread.crop import BLEND_NAMES
        names = list(BLEND_NAMES) or list(MEDIAPIPE_NAMES)
        self.idx = {f: [names.index(n) for n in ns] for f, ns in FEATURES.items()}

    def _features(self, blend):
        b = blend.tolist()
        return {f: sum(b[i] for i in ix) / len(ix) for f, ix in self.idx.items()}

    def raw(self, f, t):
        """Per-frame evidence {canonical: 0..1} from feature rises over baseline f (dict, as _features)."""
        d = {k: max(0.0, v - self.base[k]) for k, v in f.items()}
        down, inner, outer = _ramp(d['brow_down'], RAMP['brow_down']), _ramp(d['brow_inner'], RAMP['brow_inner']), \
            _ramp(d['brow_outer'], RAMP['brow_outer'])
        wide = _ramp(d['eye_wide'], RAMP['eye_wide'])
        squint = _ramp(max(d['eye_squint'], d['cheek_squint']), RAMP['squint'])
        smile = self.smile.push(d['smile'], t) * _ramp(d['smile'], RAMP['smile'])
        frown = self.frown.push(d['frown'], t) * _ramp(d['frown'], RAMP['frown'])
        return {'surprised': _or(0.7 * outer, 0.3 * inner, 0.5 * wide) * (1.0 - frown),
                'angry': _or(0.85 * down, 0.3 * squint * down),
                'happy': _or(0.8 * smile, 0.5 * squint * (1.0 - down)),
                'sad': frown * (0.55 + 0.45 * inner * (1.0 - outer))}

    def push(self, crop):
        t = float(crop.t)
        dt = 1.0 / FPS if self.t is None else min(MAX_DT, max(0.0, t - self.t))
        self.t = t
        face = crop.blend is not None
        if face and self.base is None:
            if self.idx is None:
                self._indices()
            self.base_frames.append(self._features(crop.blend))
            if len(self.base_frames) >= self.baseline_n:
                self.base = {k: float(np.median([fr[k] for fr in self.base_frames])) for k in FEATURES}
                self.base_frames = []
            return self._emit()
        if self.base is None:
            return self._emit()
        if face:
            target, tau = self.raw(self._features(crop.blend), t), self.tau
        else:
            target, tau = dict.fromkeys(self.smooth, 0.0), NOFACE_TAU_S
            self.smile.reset()
            self.frown.reset()
        a = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        for k in self.smooth:
            self.smooth[k] += a * (target[k] - self.smooth[k])
        if face:
            self._switch(t)
        elif self.label != NEUTRAL and self.smooth[self.label] < self.margin / 2:
            self.label, self.cand, self.cand_since = NEUTRAL, NEUTRAL, None
        return self._emit()

    def _switch(self, t):
        best = max(self.active, key=self.smooth.get, default=NEUTRAL)
        cand = best if best != NEUTRAL and self.smooth[best] >= self.margin else NEUTRAL
        if cand != self.cand:
            self.cand, self.cand_since = cand, t
        if cand != self.label and t - self.cand_since >= self.switch_s - 1e-9:
            self.label = cand

    def _emit(self):
        """Canonical state -> Emotion in the published label names."""
        s = {c: self.smooth[c] for c in self.active}
        s[NEUTRAL] = 1.0 - max(s.values(), default=0.0)
        scores = dict.fromkeys(self.labels, 0.0)
        for c, v in s.items():
            n = self.label_map[c]
            scores[n] = max(scores[n], min(1.0, max(0.0, float(v))))
        label = self.label_map[self.label]
        return Emotion(label, scores[label], scores)
