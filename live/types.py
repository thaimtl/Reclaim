"""Contracts every module of the live engine codes against. Owned by the lead; lanes read, never edit.

Clock: every `t` is seconds on the capture clock, time.monotonic(). CameraSource stamps time.monotonic() at grab;
ReplaySource stamps the video's own time (frame i -> i / fps), and live/run.py rebases those replay times onto
time.monotonic() (t + offset fixed at start), so bus times are monotonic seconds on the laptop either way.
Model frames run at FPS = 25; frame i of an utterance covers [t0 + i / FPS, t0 + (i + 1) / FPS).

Data flow (live/run.py):
    source   -> Frame     (one per captured frame, BGR)
    crop     -> Crop      (one per 25 fps model frame, emitted k = 2 frames late because of the crop lookahead)
    endpoint -> Segment   (live/endpoint.py: activity segments, 8 s cap with 2 s carried over)
    model    -> CamTick   (the open segment re-read every 100 ms)       -> bus cam.tick
    final    -> CamFinal  (beam + LM over a closed segment)             -> bus cam.final
Wire format of the bus: live/bus_schema.md.

LiveHyp / Word / Update are the camera-only fallback's types (live/commit.py: faint -> solid words), kept until
Austin decides whether the camera reader also commits words itself.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

FPS = 25
FAINT, SOLID = 'faint', 'solid'
WordState = Literal['faint', 'solid']


def frame_time(t0, i):
    """Capture time at the END of model frame i of an utterance that starts at t0."""
    return t0 + (i + 1) / FPS


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


@dataclass(frozen=True)
class WordEnd:
    w: str
    end_frame: int


@dataclass(frozen=True)
class LiveHyp:
    """One live decode of the whole utterance so far: input of commit.py (camera-only fallback)."""
    utt_id: int
    t0: float
    n_frames: int
    words: tuple
    trailing_blanks: int

    @property
    def t(self):
        """Capture time of the newest frame in this update."""
        return frame_time(self.t0, self.n_frames - 1)


@dataclass(frozen=True)
class Word:
    w: str
    state: WordState
    t_end: float


@dataclass(frozen=True)
class Update:
    utt_id: int
    t: float
    words: tuple





TOPK = 32
BLANK = 0
VOCAB = 5049
TIMING_KEYS = ('T', 'mediapipe', 'frontend', 'conformer', 'ctc', 'total')
EMOTIONS = ('neutral', 'happy', 'sad', 'angry', 'surprised')


@dataclass(frozen=True)
class Emotion:
    label: str
    confidence: float
    scores: dict


@dataclass(frozen=True)
class CamTick:
    """One re-read of the whole current segment, every 100 ms while a segment is open (topic cam.tick)."""
    seg_id: int
    seg_start: float
    frame_t: np.ndarray
    topk_ids: np.ndarray
    topk_logp: np.ndarray
    blank_logp: np.ndarray
    greedy: str
    activity: np.ndarray
    face_ok: bool
    emotion: Emotion
    timing_ms: dict


@dataclass(frozen=True)
class CamFinal:
    """Beam search + LM over the finished segment (topic cam.final), best first."""
    seg_id: int
    nbest: tuple
