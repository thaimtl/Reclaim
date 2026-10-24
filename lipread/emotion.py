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

EmotionWorker wraps it for the crop thread: submit() every frame (never raises), current() from the tick thread.
"""
import logging
import math
import os
import threading
import time
from types import SimpleNamespace

import numpy as np

from bus import EMOTIONS, Emotion
from lipread.types import FPS

log = logging.getLogger(__name__)

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
BASELINE_S = 3.0        # the speaker's neutral face at the start of the call (both taggers, and emotion_fit)


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
    def __init__(self, baseline_s=BASELINE_S, smooth_s=1.0, switch_s=0.5, labels=EMOTIONS, label_map=None,
                 margin=MARGIN):
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


RESTING = Emotion(NEUTRAL, 1.0, {e: float(e == NEUTRAL) for e in EMOTIONS})


class NetTagger:
    """Expression network (lipread/expression.py) -> Emotion, with the same baseline and switching as EmotionTagger.

    push(t, image, face) per processed frame, face = (t, pts, blend) or None. The first baseline_s of face frames are
    the person's neutral: their mean features are subtracted from every later frame. Probabilities are EMA-smoothed
    (tau), a non-neutral label needs a smoothed prob >= margin and must win for hold_s before the published label
    switches; with no face the label is held while the scores decay (NOFACE_TAU_S), then drops to neutral. With a head
    (fitted on the speaker's sessions) margin, hold and tau are the head's own; without one, GENERIC_SMOOTHING.
    """

    GENERIC_SMOOTHING = (0.6, 0.5, 1.0)     # margin, hold_s, tau_s: best on RAVDESS without a head

    def __init__(self, net, head=None, baseline_s=BASELINE_S, labels=EMOTIONS):
        from lipread.expression import GENERIC
        self.net, self.head, self.baseline_s, self.labels = net, head, baseline_s, tuple(labels)
        self.own = head.labels if head else tuple(GENERIC.values())
        self.margin, self.hold_s, self.tau = (head.margin, head.hold_s, head.tau_s) if head else self.GENERIC_SMOOTHING
        if NEUTRAL not in self.own or NEUTRAL not in self.labels:
            raise ValueError(f'{NEUTRAL!r} must be a label of the head and of the bus')
        self.reset_baseline()

    def reset_baseline(self):
        self.base, self.base_feats, self.base_t0 = None, [], None
        self.smooth = None
        self.label, self.cand, self.cand_since, self.t = NEUTRAL, NEUTRAL, None, None

    def _probs(self, feats):
        from lipread.expression import generic_probs
        x = feats - self.base
        if self.head:
            return self.head.predict(x)[0]
        return generic_probs(x[:8])

    def push(self, t, image, face):
        """One processed frame: run the network on the face (if any), then push_features."""
        feats = None
        if face is not None and face[2] is not None:
            logits = self.net.logits(image, face[1])
            if logits is not None:
                feats = np.concatenate([logits, np.asarray(face[2], np.float32)])
        return self.push_features(t, feats)

    def push_features(self, t, feats):
        """[8 logits, 52 blendshapes] or None (no face): the baseline first, then step() with the probabilities."""
        if feats is not None and self.base is None:
            self.base_t0 = t if self.base_t0 is None else self.base_t0
            self.base_feats.append(feats)
            if t - self.base_t0 >= self.baseline_s:
                self.base = np.mean(self.base_feats, 0)
                self.base_feats = []
            self.t = float(t)
            return self._emit()
        if self.base is None:
            self.t = float(t)
            return self._emit()
        return self.step(t, None if feats is None else self._probs(feats))

    def step(self, t, probs):
        """Probabilities over self.own (or None: no face) -> smoothing and switching -> Emotion. lipread/emotion_fit.py
        drives this directly with held-out probabilities, so what it scores is what runs live."""
        t = float(t)
        dt = 1.0 / FPS if self.t is None else min(MAX_DT, max(0.0, t - self.t))
        self.t = t
        if probs is not None:
            target, tau = np.asarray(probs, np.float32), self.tau
        else:
            target, tau = np.zeros(len(self.own), np.float32), NOFACE_TAU_S
        a = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        self.smooth = target.copy() if self.smooth is None else self.smooth + a * (target - self.smooth)
        if probs is not None:
            others = [(self.smooth[i], c) for i, c in enumerate(self.own) if c != NEUTRAL and c in self.labels]
            p, best = max(others, default=(0.0, NEUTRAL))
            cand = best if p >= self.margin else NEUTRAL
            if cand != self.cand:
                self.cand, self.cand_since = cand, t
            if cand != self.label and t - self.cand_since >= self.hold_s - 1e-9:
                self.label = cand
        else:
            self.cand, self.cand_since = self.label, None     # no face: a pending switch starts its hold again
            if self.label != NEUTRAL and self.smooth[self.own.index(self.label)] < self.margin / 2:
                self.label = self.cand = NEUTRAL
        return self._emit()

    def _emit(self):
        scores = dict.fromkeys(self.labels, 0.0)
        if self.smooth is None:
            scores[NEUTRAL] = 1.0
        else:
            for c, v in zip(self.own, self.smooth):
                if c in scores:
                    scores[c] = min(1.0, max(0.0, float(v)))
            if self.label == NEUTRAL:
                scores[NEUTRAL] = max(scores[NEUTRAL], 1.0 - max(v for c, v in scores.items() if c != NEUTRAL))
        return Emotion(self.label, scores[self.label], scores)


class EmotionWorker:
    """The emotion for cam.tick, fed from the crop thread. face = (t, pts, blend or None) as LiveCropper.last_face.

    submit() every frame (never raises): the rule tagger runs inline (~30 us). The expression network, when its model
    is there, sees at most NET_HZ frames per second of frame time. Live it runs on its own thread on the newest frame,
    so it never slows the crop thread; with sync=True (a replay faster than real time) it runs inline on the same
    frames every run, so replays are deterministic. current(): without the network, the rule tagger; with it, the
    network's emotion, or neutral while it has failed NET_MAX_FAILURES times in a row or (live) not answered for
    NET_STALE_S: one calm change for the voice, not a jump between two taggers with different smoothing.
    """

    NET_HZ = 15
    NET_MAX_FAILURES = 10
    NET_STALE_S = 2.0
    JOIN_S = 1.0

    def __init__(self, net='auto', head='auto', sync=False):
        self.tagger = EmotionTagger()
        self.emotion = RESTING
        self.failed = False
        self.sync = sync
        self.net_tagger, self.net_failures, self._slot, self._last_t = None, 0, None, None
        self._net = (RESTING, -math.inf)         # (emotion, monotonic time): one tuple, swapped in one assignment
        self._lock = threading.Lock()            # NetTagger.push vs reset_baseline from another thread
        self._slot_lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None
        try:
            self.net_tagger = self._make_net(net, head)
        except Exception:  # noqa: BLE001
            log.exception('expression network failed to start: rule tagger only')
        if self.net_tagger is not None and not sync:
            self._thread = threading.Thread(target=self._net_loop, name='emotion-net', daemon=True)
            self._thread.start()

    @staticmethod
    def _make_net(net, head):
        from lipread import expression
        if net == 'auto':
            if not expression.ExpressionNet.available():
                log.info('no verified %s: emotion from the rule tagger (lipread/setup.sh downloads it)',
                         expression.MODEL_PATH)
                return None
            net = expression.ExpressionNet()
        if net is None:
            return None
        if head == 'auto':
            head = expression.Head.load() if os.path.isfile(expression.HEAD_PATH) else None
            log.info('emotion: expression network, %s', f'head {expression.HEAD_PATH} ({head.info})' if head else
                     'no head (lipread/emotion_fit.py makes one from recorded sessions): generic labels')
        return NetTagger(net, head)

    def submit(self, t, image, face):
        try:
            self.emotion = self.tagger.push(SimpleNamespace(t=t, blend=None if face is None else face[2]))
        except Exception:  # noqa: BLE001
            if not self.failed:
                self.failed = True
                log.exception('emotion failed (logged once; the lip reader goes on)')
        if self.net_tagger is None:
            return
        if self.sync:
            if self._last_t is None or t - self._last_t >= 1.0 / self.NET_HZ - 1e-9:
                self._last_t = t
                self._run_net((t, image, face))
            return
        with self._slot_lock:
            self._slot = (t, image, face)
        self._wake.set()

    def _run_net(self, slot):
        try:
            with self._lock:
                emotion = self.net_tagger.push(*slot)
            self._net = (emotion, time.monotonic())
            self.net_failures = 0
        except Exception:  # noqa: BLE001
            self.net_failures += 1
            if self.net_failures == 1:
                log.exception('expression network failed (neutral until it recovers)')

    def _net_loop(self):
        period = 1.0 / self.NET_HZ
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self._slot_lock:
                slot, self._slot = self._slot, None
            if slot is None:
                continue
            started = time.monotonic()
            self._run_net(slot)
            self._stop.wait(max(0.0, started + period - time.monotonic()))

    def reset_baseline(self):
        self.tagger.reset_baseline()
        if self.net_tagger is not None:
            with self._lock:
                self.net_tagger.reset_baseline()
                self._net = (RESTING, time.monotonic())

    def current(self):
        if self.net_tagger is None:
            return self.emotion
        emotion, t = self._net
        if self.net_failures >= self.NET_MAX_FAILURES:
            return RESTING
        if not self.sync and time.monotonic() - t >= self.NET_STALE_S:
            return RESTING
        return emotion

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self.JOIN_S)
