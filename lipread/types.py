"""Types the lip reader's modules share.

Clock: every `t` is seconds on the team clock, bus.now() (time.perf_counter). CameraSource stamps it at grab;
ReplaySource stamps the video's own time (frame i -> i / fps), and lipread/run.py rebases those replay times onto
bus.now() (t + offset fixed at start), so bus times are on the laptop's clock either way.
Model frames run at FPS = 25; frame i of a segment covers [t0 + i / FPS, t0 + (i + 1) / FPS).

Data flow (lipread/run.py):
    source   -> Frame     (one per captured frame, BGR)
    crop     -> Crop      (one per 25 fps model frame)
    endpoint -> Segment   (activity segments, 8 s cap with 2 s carried over)
    model    -> CamTick   (the open segment re-read every 100 ms)       -> bus cam.tick
    final    -> CamFinal  (beam + LM over a closed segment)             -> bus cam.final
Message types and the wire format: bus/__init__.py and bus/schema.md.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

FPS = 25


@dataclass(frozen=True)
class Frame:
    t: float
    image: np.ndarray
    index: int = 0


@dataclass(frozen=True)
class Crop:
    t: float
    roi96: np.ndarray
    mouth_open: float
    face_ok: bool
    blend: np.ndarray = None
    mouth_pts: np.ndarray = None
    eye_dist: float = 0.0


def fit(frame, size):
    """The frame shrunk (INTER_AREA, aspect kept) to fit size = (w, h); unchanged if it fits or size is None."""
    h, w = frame.image.shape[:2]
    if size is None or (w <= size[0] and h <= size[1]):
        return frame
    k = min(size[0] / w, size[1] / h)
    img = cv2.resize(frame.image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return replace(frame, image=img)
