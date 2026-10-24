"""Live mouth crops: one 96x96 grayscale patch per 25 fps frame, aligned like Lipflow's mouth_rois (D12).

Each frame gets MediaPipe landmarks (Lipflow FaceTracker, VIDEO mode) on the full frame, a downscaled copy
(detect_scale < 1) or a square face ROI around the last face (face_roi=True), mapped back to full-frame coordinates.
The frame is cropped once, k frames late: the transform is smoothed over the past window_margin // 2 crops plus the
k frames of lookahead, then frozen (re-cropping a growing clip would change past crops, D12). Missing faces are
interpolated or held within that window; before the first face the patch is zeros with face_ok False.

Smoothing (smooth=): 'box' averages the past window_margin // 2 anchors plus k frames of lookahead (k = 2: 80 ms
late); 'ema' weights the past window by (1 - ema_alpha) ** age and needs no lookahead (k = 0, crop on arrival).
BlendTracker adds MediaPipe's 52 blendshapes and the mouth points E4 (activity) and E5 (emotion) read.

With symmetric=True the window is the same +-m frames as mouth_rois (m = min(window_margin // 2, past, future)), so
k = 6 with full-res detection reproduces mouth_rois exactly on a clip without long face gaps (tests/live/test_crop.py).
Measured on the sample (640x360) and a 2x upscale (1280x720), 16-core CPU: detection ~4.5 ms/frame at full res, no
faster downscaled (MediaPipe resizes internally) and slower with face_roi, both less accurate: full res is the default.

    crop = LiveCropper(face_model, k=2)                  # or smooth='ema', k=0
    for frame in source.frames():
        for c in crop.push(frame):
            ...                  # live.types.Crop, k frames behind the newest frame
    rest = crop.flush()          # end of the utterance / clip: the last k crops with the lookahead that exists
"""
import os
import time
from collections import deque, namedtuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import live  # noqa: F401  (puts Lipflow on sys.path)
from lipflow.face import STABLE_REFERENCE, FaceObs, FaceTracker, _interpolate
from live.types import FPS, Crop

CROP = 96
REF = STABLE_REFERENCE.astype(np.float32)
MOUTH_PTS = [13, 14, 61, 291]   # inner lip top, bottom, corners (Lipflow FaceObs.mouth_open uses the same points)
BLEND_NAMES = []                # MediaPipe's 52 blendshape names, in the order of Crop.blend (filled on first face)
# Named smoothing settings (--smooth); measured on the sample by live/bench/smooth_check.py (CHANGELOG "Measured")
SMOOTHING = {'k0': dict(k=0), 'k2': dict(k=2), 'ema': dict(k=0, smooth='ema', ema_alpha=0.3)}
DEFAULT_SMOOTHING = 'k0'
Pending = namedtuple('Pending', 't gray anchors mouth_open blend mouth_pts eye_dist')


class LiveFaceObs(FaceObs):
    __slots__ = ('blend', 'pose')

    def __init__(self, pts, blend=None, pose=None):
        super().__init__(pts)
        self.blend = blend
        self.pose = pose            # (3,) float32 yaw, pitch, roll in radians, or None (BlendTracker pose=False)


def head_pose(matrix):
    """MediaPipe's 4x4 facial transformation matrix -> (yaw, pitch, roll) radians, float32."""
    m = np.asarray(matrix, np.float64)[:3, :3]
    return np.array([np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2])), np.arctan2(m[2, 1], m[2, 2]),
                     np.arctan2(m[1, 0], m[0, 0])], np.float32)


class BlendTracker(FaceTracker):
    """Lipflow FaceTracker (same options, VIDEO mode) with output_face_blendshapes=True; pose=True adds the head pose
    (output_facial_transformation_matrixes) for cam.face. Neither option changes the landmarks."""

    def __init__(self, model_path, blendshapes=True, pose=False):
        self.pose = pose
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.abspath(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=pose,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_bgr, ts_ms):
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing time
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
        blend = None
        if res.face_blendshapes:
            cats = res.face_blendshapes[0]
            if not BLEND_NAMES:
                BLEND_NAMES.extend(c.category_name for c in cats)
            blend = np.array([c.score for c in cats], dtype=np.float32)
        pose = head_pose(res.facial_transformation_matrixes[0]) if self.pose and res.facial_transformation_matrixes \
            else None
        return LiveFaceObs(pts, blend, pose)


def align(gray, lms_i, window, crop=CROP, weights=None):
    """One frame of Lipflow mouth_rois: smoothed anchors -> similarity transform to the mean face -> patch at the mouth.

    lms_i: this frame's 4x2 anchors; window: the anchors to average (including lms_i), optionally weighted.
    Same ops as face.py mouth_rois when weights is None.
    """
    half = crop // 2
    smoothed = np.mean(window, axis=0) if weights is None else np.average(window, axis=0, weights=weights)
    smoothed += lms_i.mean(axis=0) - smoothed.mean(axis=0)
    tf, _ = cv2.estimateAffinePartial2D(smoothed.astype(np.float32), REF, method=cv2.LMEDS)
    if tf is None:
        tf = cv2.estimateAffinePartial2D(lms_i.astype(np.float32), REF)[0]
    mouth = smoothed[3] @ tf[:, :2].T + tf[:, 2]
    warped = cv2.warpAffine(gray, tf, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    cx = int(round(np.clip(mouth[0], half, 256 - half)))
    cy = int(round(np.clip(mouth[1], half, 256 - half)))
    return warped[cy - half:cy + half, cx - half:cx + half]


class LiveCropper:
    def __init__(self, face_model, k=2, detect_scale=1.0, face_roi=False, roi_px=256, roi_margin=1.6,
                 window_margin=12, symmetric=False, smooth='box', ema_alpha=0.5, blendshapes=True, pose=False):
        if smooth not in ('box', 'ema') or (smooth == 'ema' and (k != 0 or symmetric)):
            raise ValueError(f'smooth={smooth!r} k={k} symmetric={symmetric}: box (any k) or ema (k = 0)')
        self.smooth, self.ema_alpha = smooth, ema_alpha
        self.k, self.detect_scale, self.face_roi = k, detect_scale, face_roi
        self.roi_px, self.roi_margin = roi_px, roi_margin
        self.half_window, self.symmetric = window_margin // 2, symmetric
        self.tracker = BlendTracker(face_model, blendshapes, pose)
        self.roi_tracker = (BlendTracker(face_model, blendshapes, pose)   # own VIDEO-mode state for the ROI images
                            if face_roi else None)
        self.box = None             # (x0, y0, side) of the face ROI, full-frame pixels
        self.past = deque(maxlen=self.half_window)  # anchors used for the last crops (None before the first face)
        self.pending = deque()      # detected, not yet cropped (at most k)
        self.times = deque(maxlen=FPS)  # per-frame crop ms, last second at 25 fps
        self.last_face = None       # see push()
        self.last_pose = None       # (3,) yaw, pitch, roll of the newest frame, with pose=True; else None

    @property
    def crop_ms(self):
        """Mean per-frame crop time (detect + align) over the last 25 frames, for Status.crop_ms."""
        return float(np.mean(self.times)) if self.times else 0.0

    def detect(self, image, t):
        """FaceObs-like landmarks (pts in full-frame pixels) or None, per detect_scale / face_roi."""
        ts = int(round(t * 1000))
        obs = None
        if self.face_roi and self.box is not None:
            x0, y0, side = self.box
            s = self.roi_px / side
            sub = cv2.resize(image[y0:y0 + side, x0:x0 + side], (self.roi_px, self.roi_px),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            obs = self.roi_tracker.detect(sub, ts)
            if obs is not None:
                obs.pts = obs.pts / s + np.array([x0, y0], np.float32)
        if obs is None:
            s = self.detect_scale
            small = image if s == 1 else cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            obs = self.tracker.detect(small, ts)
            if obs is not None and s != 1:
                obs.pts = obs.pts * np.array([image.shape[1] / small.shape[1], image.shape[0] / small.shape[0]],
                                             np.float32)
        if self.face_roi:
            self.box = self._box(obs.pts, image.shape) if obs is not None else None
        return obs

    def _box(self, pts, shape):
        h, w = shape[:2]
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        side = int(min(max(x1 - x0, y1 - y0) * self.roi_margin, w, h))
        bx = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
        by = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
        return bx, by, side

    def push(self, frame):
        """Add one Frame; returns the crops that became final (0 or 1, k frames behind)."""
        t0 = time.perf_counter()
        obs = self.detect(frame.image, frame.t)
        # newest frame's face for live/emotion_net.py (E5): (t, pts (478, 2) pixels, blend (52,) or None), or None
        self.last_face = None if obs is None else (frame.t, obs.pts, obs.blend)
        self.last_pose = None if obs is None else obs.pose
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if obs is None:
            self.pending.append(Pending(frame.t, gray, None, 0.0, None, None, 0.0))
        else:
            a = obs.anchors
            self.pending.append(Pending(frame.t, gray, a, obs.mouth_open, obs.blend, obs.pts[MOUTH_PTS].copy(),
                                        float(np.linalg.norm(a[0] - a[1]))))
        out = [self._emit(self.k)] if len(self.pending) > self.k else []
        self.times.append((time.perf_counter() - t0) * 1000)
        return out

    def flush(self):
        """Crop every pending frame with the lookahead that exists, then start a fresh smoothing window."""
        out = [self._emit(len(self.pending) - 1) for _ in range(len(self.pending))]
        self.past.clear()
        return out

    def _emit(self, future):
        past = list(self.past)
        lms = _interpolate(past + [p.anchors for p in list(self.pending)[:future + 1]])
        p = self.pending.popleft()
        if lms is None:
            self.past.append(None)
            return Crop(t=p.t, roi96=np.zeros((CROP, CROP), np.uint8), mouth_open=0.0, face_ok=False)
        i = len(past)
        weights = None
        if self.symmetric:
            m = min(self.half_window, i, future)
            window = lms[i - m:i + m + 1]
        elif self.smooth == 'ema':
            window = lms[max(0, i - self.half_window):i + 1]
            weights = (1 - self.ema_alpha) ** np.arange(len(window) - 1, -1, -1)
        else:
            window = lms[max(0, i - self.half_window):i + future + 1]
        self.past.append(lms[i])
        face_ok = p.anchors is not None
        return Crop(t=p.t, roi96=align(p.gray, lms[i], window, weights=weights),
                    mouth_open=p.mouth_open if face_ok else 0.0, face_ok=face_ok, blend=p.blend,
                    mouth_pts=p.mouth_pts, eye_dist=p.eye_dist)

    def close(self):
        self.tracker.close()
        if self.roi_tracker is not None:
            self.roi_tracker.close()
