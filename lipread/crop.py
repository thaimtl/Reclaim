"""Live mouth crops: one 96x96 grayscale patch per 25 fps frame, aligned like Lipflow's mouth_rois (D12).

Each frame gets MediaPipe landmarks (Lipflow FaceTracker, VIDEO mode) on the full frame. The frame is cropped once,
k frames late: the transform is averaged over the past window_margin // 2 anchors plus k frames of lookahead, then
frozen (re-cropping a growing clip would change past crops, D12). The live default is k = 0: past frames only, a crop
as soon as its frame arrives. Missing faces are interpolated or held within the window; before the first face the
patch is zeros with face_ok False. BlendTracker adds MediaPipe's 52 blendshapes and the mouth points that activity
(endpoint.py) and emotion read.

With symmetric=True the window is the same +-m frames as mouth_rois (m = min(window_margin // 2, past, future)), so
k = 6 reproduces mouth_rois exactly on a clip without long face gaps (tests/lipread/test_crop.py): the check that the
alignment matches what Auto-AVSR was trained on.

    crop = LiveCropper(face_model)
    for frame in source.frames():
        for c in crop.push(frame):
            ...                  # lipread.types.Crop, k frames behind the newest frame
    rest = crop.flush()          # end of the clip: the last k crops with the lookahead that exists
"""
import os
import time
from collections import deque, namedtuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import lipread  # noqa: F401  (puts Lipflow on sys.path)
from lipflow.face import STABLE_REFERENCE, FaceObs, FaceTracker, _interpolate
from lipread.types import FPS, Crop

CROP = 96
REF = STABLE_REFERENCE.astype(np.float32)
MOUTH_PTS = [13, 14, 61, 291]
BLEND_NAMES = []
Pending = namedtuple('Pending', 't gray anchors mouth_open blend mouth_pts eye_dist')


class LiveFaceObs(FaceObs):
    __slots__ = ('blend', 'pose')

    def __init__(self, pts, blend=None, pose=None):
        super().__init__(pts)
        self.blend = blend
        self.pose = pose


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
        ts_ms = max(int(ts_ms), self._last_ts + 1)
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


def align(gray, lms_i, window, crop=CROP):
    """One frame of Lipflow mouth_rois: smoothed anchors -> similarity transform to the mean face -> patch at the mouth.

    lms_i: this frame's 4x2 anchors; window: the anchors to average (including lms_i). Same ops as face.py mouth_rois.
    """
    half = crop // 2
    smoothed = np.mean(window, axis=0)
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
    def __init__(self, face_model, k=0, window_margin=12, symmetric=False, blendshapes=True, pose=False):
        self.k = k
        self.half_window, self.symmetric = window_margin // 2, symmetric
        self.tracker = BlendTracker(face_model, blendshapes, pose)
        self.past = deque(maxlen=self.half_window)
        self.pending = deque()
        self.times = deque(maxlen=FPS)
        self.last_face = None
        self.last_pose = None

    def detect(self, image, t):
        return self.tracker.detect(image, int(round(t * 1000)))

    def push(self, frame):
        """Add one Frame; returns the crops that became final (0 or 1, k frames behind)."""
        t0 = time.perf_counter()
        obs = self.detect(frame.image, frame.t)
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
        if self.symmetric:
            m = min(self.half_window, i, future)
            window = lms[i - m:i + m + 1]
        else:
            window = lms[max(0, i - self.half_window):i + future + 1]
        self.past.append(lms[i])
        face_ok = p.anchors is not None
        return Crop(t=p.t, roi96=align(p.gray, lms[i], window),
                    mouth_open=p.mouth_open if face_ok else 0.0, face_ok=face_ok, blend=p.blend,
                    mouth_pts=p.mouth_pts, eye_dist=p.eye_dist)

    def close(self):
        self.tracker.close()
