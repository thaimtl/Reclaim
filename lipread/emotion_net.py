"""Facial emotion for the camera reader (E5): webcam face -> types.Emotion on cam.tick.

Default (the repo's spec, docs/EVAN_REALTIME_LIP_READING.md E5): the 52 MediaPipe blendshapes, upper face first,
smoothed ~1 s with a 0.5 s hold, which is lipread/emotion.py's EmotionTagger; EmotionWorker() just runs it per frame.
Opt-in (EmotionWorker(classifier=True), run.py --emotion-classifier): an extra experiment, not in the repo plan, to be
switched on only if it measurably beats the tagger on Austin's clips. Three signals, fused per classifier read (12.5/s):

    classifier  EmotiEffLib enet_b0_8_va_mtl (ONNX, AffectNet, 8 classes + valence/arousal, Apache-2.0), on an RGB
                face crop cut from the MediaPipe landmarks the crop thread already has. ~4.4 ms on 2 CPU threads.
    tagger      lipread/emotion.py EmotionTagger, every frame (~10 us): blendshape rules vs a median baseline; a mouth
                shape counts only once held 0.6 s, so mouthed syllables don't read as smiles. Owns smile/frown timing.
    EMG         optional: tonic (sustained) level of pad A3 = Gaddy site 1, near the smile muscles. Speech makes short
                bursts, a held smile a sustained level, so the slow envelope separates them (EmgTonic).

Fusion: happy = max(classifier, tagger, EMG) and updates any time; sad / angry / surprised = mean(classifier, tagger),
so both must agree, and update only on still reads (while mouthing they hold their last still value, fading).

Why (measured on the sample): mouthing halves the classifier's Neutral (65% -> 39%), a resting
face reads Anger/Sadness 10-20% of the time, and the tagger alone reads emphasis brows as angry 54% of the time. So
each signal is a z-score against the wearer's resting face (median/MAD over the first CALIB_READS reads, still ones
preferred), a label needs Z_THR, mouthing reads are down-weighted and cannot vote surprise, and a new label must lead
for HOLD reads. Fused, sample 100-170 s: 92% neutral, 4 label changes.

    worker = EmotionWorker()                       # the spec: EmotionTagger only, no model, no thread
    worker = EmotionWorker(classifier=True)        # the experiment: loads the model, starts its thread
    worker.submit(frame.t, frame.image, cropper.last_face)   # crop thread, every frame; never blocks
    worker.push_emg(t, samples)                    # optional, (n, 4) filtered EMG at 1 kHz, pins A0-A3
    tick_emotion = worker.current()                # types.Emotion, any thread
    worker.close()

    venv-live/bin/python -m lipread.emotion_net --replay third_party/lipflow/samples/2016-03-12.mov --start 100 --end 130
    venv-live/bin/python -m lipread.emotion_net --cam 0
"""
import argparse
import logging
import os
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

import lipread
from lipread.emotion import MARGIN as TAG_MARGIN
from lipread.endpoint import STILL_THR
from bus import EMOTIONS, Emotion, now
from lipread.types import FPS

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(lipread.WORK, 'third_party', 'emotiefflib', 'enet_b0_8_va_mtl.onnx')
CLASSES = ('anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')
TO_LABEL = {'anger': 'angry', 'contempt': 'angry', 'disgust': 'angry', 'fear': 'surprised', 'happiness': 'happy',
            'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG = 224
BOX_MARGIN = 0.12

CALIB_READS = 60
Z_THR = 2.0
SD_FLOOR = 0.25
EMG_SD_FLOOR = 1e-6
EMG_REL_FLOOR = 0.1
MOUTH_W = 0.35
EMA = 0.4
MOUTH_DECAY = 0.973
LEAD = 0.5
HOLD = 6
MAX_HZ = 12.5
EMG_WARMUP_S = 4.0
EV_CLIP = 20.0
NEUTRAL = Emotion('neutral', 1.0, {e: float(e == 'neutral') for e in EMOTIONS})
HAPPY = np.array([k == 'happy' for k in EMOTIONS])


def face_box(pts, shape, margin=BOX_MARGIN):
    """Square box (x0, y0, x1, y1) around the landmarks plus margin, clipped to the image; None if degenerate."""
    h, w = shape[:2]
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    if side < 16:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bx1, by1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return (bx0, by0, bx1, by1) if bx1 - bx0 >= 16 and by1 - by0 >= 16 else None


def face_crop(image, box):
    """The square crop for `box`, padded with black where the face runs off the frame (no stretching at the edges)."""
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    if h == w:
        return crop
    d = abs(h - w)
    if h < w:
        return cv2.copyMakeBorder(crop, d // 2, d - d // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return cv2.copyMakeBorder(crop, 0, 0, d // 2, d - d // 2, cv2.BORDER_CONSTANT, value=0)


def preprocess(face_bgr):
    """BGR face crop -> (1, 3, 224, 224) float32, the model's ImageNet normalisation (same as hsemotion-onnx)."""
    x = cv2.resize(face_bgr, (IMG, IMG))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255)
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class EmotionNet:
    """ONNX Runtime session, CPU only (the GPU belongs to the lip reader). run() -> (logits (8,), valence, arousal)."""

    def __init__(self, path=MODEL_PATH, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name

    def run(self, face_bgr):
        y = self.sess.run(None, {self.inp: preprocess(face_bgr)})[0][0]
        return y[:8].astype(np.float32), float(y[8]), float(y[9])


class Stillness:
    """Mouthing activity from the face landmarks with lipread.endpoint's own measure (same thresholds as the segmenter)."""

    def __init__(self):
        from lipread.endpoint import Activity
        self.act = Activity()

    def push(self, pts):
        if pts is None:
            self.act.reset()
            return 0.0
        eyes = np.linalg.norm(pts[[33, 133]].mean(0) - pts[[362, 263]].mean(0))
        crop = SimpleNamespace(face_ok=True, mouth_pts=pts[[13, 14, 61, 291]], eye_dist=float(eyes))
        return self.act.push(crop)


class EmgTonic:
    """Slow (tonic) envelope of each EMG channel: rectify, then a 2nd-order low-pass at cutoff_hz.
    Input: (n, C) samples already through rig/filters.py (high-pass 20 Hz + notches), at fs. Keeps filter state."""

    def __init__(self, channels=4, fs=1000, cutoff_hz=0.5, warmup_s=EMG_WARMUP_S):
        from scipy.signal import butter
        self.sos = butter(2, cutoff_hz, fs=fs, output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2, channels))
        self.value = np.zeros(channels)
        self.seen, self.warmup = 0, int(warmup_s * fs)

    @property
    def ready(self):
        """True once the envelope has settled (it starts at 0 and rises for a few seconds)."""
        return self.seen >= self.warmup

    def push(self, samples):
        from scipy.signal import sosfilt
        x = np.abs(np.nan_to_num(np.asarray(samples, np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        if x.size == 0:
            return self.value
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        if not np.isfinite(self.zi).all():
            self.zi[:] = 0.0
        self.seen += len(x)
        self.value = y[-1]
        return self.value


class Baseline:
    """The wearer's resting face: robust centre and spread of a vector over the first n reads, then frozen.
    Median and 1.4826 * MAD, so talking or a passing expression during calibration barely moves it; still reads are
    used when at least a third of the window was still, else all reads (someone who never stops talking)."""

    def __init__(self, n, floor, rel_floor=0.0):
        self.n, self.floor, self.rel_floor, self.rows, self.still = n, floor, rel_floor, [], []
        self.mu = self.sd = None

    @property
    def ready(self):
        return self.mu is not None

    def push(self, v, still=True):
        if self.ready:
            return
        self.rows.append(np.asarray(v, np.float64))
        self.still.append(bool(still))
        if len(self.rows) >= self.n:
            a, m = np.array(self.rows), np.array(self.still)
            if m.sum() >= self.n // 3:
                a = a[m]
            self.mu = np.median(a, axis=0)
            floor = np.maximum(self.floor, self.rel_floor * np.abs(self.mu))
            self.sd = np.maximum(1.4826 * np.median(np.abs(a - self.mu), axis=0), floor)
            self.rows, self.still = [], []

    def z(self, v):
        return (np.asarray(v, np.float64) - self.mu) / self.sd


class EmotionEstimator:
    """Pure fusion logic (no model, no threads): one call per read. Deterministic, so tests drive it with numbers."""

    def __init__(self, calib_reads=CALIB_READS, z_thr=Z_THR, mouth_w=MOUTH_W, ema=EMA, lead=LEAD, hold=HOLD):
        self.z_thr, self.mouth_w, self.ema, self.lead, self.hold = z_thr, mouth_w, ema, lead, hold
        self.base_logits = Baseline(calib_reads, SD_FLOOR)
        self.base_emg = Baseline(calib_reads, EMG_SD_FLOOR, EMG_REL_FLOOR)
        self.ev = None
        self.label, self.cand, self.run = 'neutral', None, 0
        self.valence = self.arousal = 0.0
        self.current = NEUTRAL

    @property
    def calibrated(self):
        return self.base_logits.ready

    def recalibrate(self):
        """New person in front of the camera: collect a new resting face, back to neutral."""
        n = self.base_logits.n
        self.base_logits = Baseline(n, SD_FLOOR)
        self.base_emg = Baseline(n, EMG_SD_FLOOR, EMG_REL_FLOOR)
        self.ev, self.label, self.cand, self.run = None, 'neutral', None, 0
        self.valence = self.arousal = 0.0
        self.current = NEUTRAL

    def update(self, logits=None, still=True, emg=None, va=None, tag=None):
        """logits: (8,) model logits or None (no face);
        still: mouth not moving; emg: tonic level of the smile pad (A3) or None; va: (valence, arousal) or None;
        tag: EmotionTagger.smooth {label: 0..1} or None. Returns types.Emotion.
        Non-finite inputs are treated as missing, so the published scores are always finite."""
        if logits is None or not np.isfinite(logits).all():
            return self._decay()
        if emg is not None and not np.isfinite(emg):
            emg = None
        if not self.calibrated:
            self.base_logits.push(logits, still)
            if emg is not None:
                self.base_emg.push([emg], still)
            if self.calibrated:
                self.ev = np.zeros(len(EMOTIONS))
            return self.current
        z = self.base_logits.z(logits)
        w = 1.0 if still else self.mouth_w
        e = dict.fromkeys(EMOTIONS, 0.0)
        for c, zc in zip(CLASSES, z):
            lab = TO_LABEL[c]
            if lab != 'neutral':
                e[lab] = max(e[lab], w * zc)
        if tag is not None:
            for k, v in tag.items():
                if k not in e or k == 'neutral':
                    continue
                tz = v / TAG_MARGIN * self.z_thr
                e[k] = max(e[k], tz) if k == 'happy' else 0.5 * (e[k] + tz)
        if not still:
            e['surprised'] = 0.0
        if emg is not None and self.base_emg.ready:
            e['happy'] = max(e['happy'], float(self.base_emg.z([emg])[0]))
        if not self.base_emg.ready and emg is not None:
            self.base_emg.push([emg], still)
        vec = np.array([e[k] for k in EMOTIONS])
        new = (1 - self.ema) * self.ev + self.ema * vec
        if not still:
            new = np.where(HAPPY, new, self.ev * MOUTH_DECAY)
        self.ev = np.clip(new, -EV_CLIP, EV_CLIP)
        if va is not None:
            self.valence += self.ema * (va[0] - self.valence)
            self.arousal += self.ema * (va[1] - self.arousal)
        return self._decide()

    def _decay(self):
        """No face: let the evidence fade so a lost face ends in neutral, not a frozen label."""
        if self.ev is not None:
            self.ev = (1 - self.ema) * self.ev
            return self._decide()
        return self.current

    def _decide(self):
        """Label with hysteresis: the same challenger must lead the current label by `lead` for `hold` reads in a row.
        confidence = the held label's softmax score, so while a label is held it can be lower than a rising one's."""
        ev = dict(zip(EMOTIONS, np.clip(np.nan_to_num(self.ev), -EV_CLIP, EV_CLIP)))
        ev['neutral'] = self.z_thr
        best = max(EMOTIONS, key=lambda k: ev[k])
        if best != self.label and ev[best] - ev[self.label] > self.lead:
            self.run = self.run + 1 if best == self.cand else 1
            self.cand = best
            if self.run >= self.hold:
                self.label, self.cand, self.run = best, None, 0
        else:
            self.cand, self.run = None, 0
        x = np.array([ev[k] for k in EMOTIONS])
        p = np.exp(x - x.max())
        p /= p.sum()
        scores = {k: float(v) for k, v in zip(EMOTIONS, p)}
        self.current = Emotion(self.label, scores[self.label], scores)
        return self.current


class EmotionWorker:
    """classifier=False (default, the repo's E5 spec): the EmotionTagger runs per frame in submit() and current() is
    its label. classifier=True: also runs the model off the crop thread (latest-frame mailbox) and fuses (module doc).
    current() returns the latest Emotion from any thread."""

    def __init__(self, model_path=MODEL_PATH, max_hz=MAX_HZ, threads=2, estimator=None, net=None, classifier=False):
        from lipread.emotion import EmotionTagger
        self.classifier = classifier or net is not None
        self.tagger = EmotionTagger()
        self._tag_emotion = NEUTRAL
        if not self.classifier:
            self.est = None
            self.read_ms, self.reads, self.last_still = deque(maxlen=1), 0, True
            return
        self.net = net or EmotionNet(model_path, threads)
        self.est = estimator or EmotionEstimator()
        self._tag = None
        self.still = Stillness()
        self.tonic = EmgTonic()
        self.every = max(1, round(FPS / max_hz))
        self._n = 0
        self._slot, self._emg = None, None
        self._lock, self._wake = threading.Lock(), threading.Event()
        self._stop = threading.Event()
        self._recal = False
        self._failed = False
        self.read_ms = deque(maxlen=50)
        self.reads = 0
        self.last_still = True
        self._th = threading.Thread(target=self._loop, name='emotion', daemon=True)
        self._th.start()

    def submit(self, t, image, face):
        """face = (t, pts (N, 2) pixels, blend (52,) or None) of this frame (LiveCropper.last_face), or None.
        Call from one thread (the crop thread), every frame. Non-blocking: the tagger and the stillness measure run
        here per frame (~10 us each, at the 25 fps they were tuned for); the classifier read goes to the worker. Never
        raises: an error is logged once and the lip reader keeps going."""
        try:
            self._tag_emotion = self.tagger.push(SimpleNamespace(t=t, blend=None if face is None else face[2]))
            if not self.classifier:
                return
            self._tag = dict(self.tagger.smooth) if self.tagger.base is not None else None
            pts = None if face is None else np.asarray(face[1], np.float32)
            self.last_still = self.still.push(pts) < STILL_THR
            self._n += 1
            if self._n % self.every:
                return
            with self._lock:
                self._slot = (image, pts, self.last_still, self._tag)
            self._wake.set()
        except Exception:  # noqa: BLE001
            if not self._failed:
                self._failed = True
                log.exception('emotion submit failed (logged once; emotion may stay neutral)')

    def recalibrate(self):
        """New person: both baselines start over. Call from the crop thread; the worker resets its estimator before its
        next read, so the two threads never touch it at once."""
        self.tagger.reset_baseline()
        if not self.classifier:
            return
        with self._lock:
            self._slot = None
            self._recal = True

    def push_emg(self, t, samples):
        """(n, 4) filtered EMG at 1 kHz, pins A0-A3, from one thread; keeps the tonic level of A3 (Gaddy site 1, smile
        side) once the envelope has settled. Classifier mode only (EMG emotion is not in the repo plan)."""
        if not self.classifier:
            return
        v = self.tonic.push(samples)[3]
        if self.tonic.ready:
            self._emg = float(v)

    def current(self):
        return self.est.current if self.classifier else self._tag_emotion

    def _loop(self):
        while not self._stop.is_set():
            if not self._wake.wait(0.2):
                continue
            with self._lock:
                slot, self._slot = self._slot, None
                self._wake.clear()
            if slot is None:
                continue
            try:
                self._read(*slot)
            except Exception:  # noqa: BLE001
                log.exception('emotion read failed')

    def _read(self, image, pts, still, tag):
        t0 = time.perf_counter()
        with self._lock:
            recal, self._recal = self._recal, False
        if recal:
            self.est.recalibrate()
            return
        box = None if pts is None else face_box(pts, image.shape)
        if box is None:
            self.est.update(None)
            return
        logits, val, aro = self.net.run(face_crop(image, box))
        self.est.update(logits, still=still, emg=self._emg, va=(val, aro), tag=tag)
        self.reads += 1
        self.read_ms.append((time.perf_counter() - t0) * 1000)

    def close(self):
        if not self.classifier:
            return
        self._stop.set()
        self._wake.set()
        self._th.join(timeout=1.0)


def draw(img, worker, still):
    """Overlay for --show / --out: label, the five scores as bars, calibration and mouth state."""
    e, est = worker.current(), worker.est
    pad = 12
    cv2.rectangle(img, (0, 0), (300, 40 + 26 * len(EMOTIONS) + 52), (18, 15, 20), -1)
    if est is None:
        head = e.label.upper()
    else:
        head = e.label.upper() if est.calibrated else f'CALIBRATING {len(est.base_logits.rows)}/{est.base_logits.n}'
    cv2.putText(img, head, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (122, 210, 255), 2, cv2.LINE_AA)
    for i, k in enumerate(EMOTIONS):
        y = 52 + 26 * i
        v = e.scores.get(k, 0.0)
        cv2.putText(img, k, (pad, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 235, 240), 1, cv2.LINE_AA)
        cv2.rectangle(img, (110, y), (110 + 170, y + 14), (60, 50, 60), -1)
        color = (122, 210, 255) if k == e.label else (255, 184, 143)
        cv2.rectangle(img, (110, y), (110 + int(170 * v), y + 14), color, -1)
    y = 52 + 26 * len(EMOTIONS) + 10
    va = '' if est is None else f'  valence {est.valence:+.2f}  arousal {est.arousal:+.2f}'
    cv2.putText(img, f'{"still" if still else "mouthing"}{va}',
                (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _main():
    ap = argparse.ArgumentParser(description='Facial emotion on a replay or the webcam; prints label changes.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--replay')
    src.add_argument('--cam', type=int)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--show', action='store_true', help='window with the overlay (needs a display); q quits')
    ap.add_argument('--out', default=None, help='write an MP4 with the overlay')
    ap.add_argument('--classifier', action='store_true', help='the opt-in classifier fusion (default: the spec tagger)')
    args = ap.parse_args()
    from lipread.crop import BlendTracker
    tracker = BlendTracker(os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task'))
    worker = EmotionWorker(classifier=args.classifier)
    cap = cv2.VideoCapture(args.replay if args.replay else args.cam)
    if args.replay:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    i, last = 0, None
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = args.start + i / fps if args.replay else now()
            i += 1
            if args.end is not None and t > args.end:
                break
            obs = tracker.detect(img, t * 1000)
            face = None if obs is None else (t, obs.pts, obs.blend)
            worker.submit(t, img, face)
            if args.replay and not args.out:
                time.sleep(1 / fps)
            elif args.replay:
                time.sleep(0.02)
            e = worker.current()
            if e.label != last:
                est = worker.est
                extra = '' if est is None else (f'  valence {est.valence:+.2f} arousal {est.arousal:+.2f}  '
                                                f'calibrated {est.calibrated}')
                print(f'{t:7.2f}s  {e.label:9s} conf {e.confidence:.2f}{extra}', flush=True)
                last = e.label
            if args.show or args.out:
                vis = draw(img.copy(), worker, worker.last_still)
                if args.out:
                    if writer is None:
                        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, vis.shape[1::-1])
                    writer.write(vis)
                if args.show:
                    cv2.imshow('Reclaim emotion', vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    ms = list(worker.read_ms)
    if ms:
        print(f'{worker.reads} reads, read ms median {np.median(ms):.1f} p95 {np.percentile(ms, 95):.1f}')


if __name__ == '__main__':
    _main()
