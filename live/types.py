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
    t: float                # capture time of the grab
    image: np.ndarray       # (H, W, 3) uint8 BGR, as cv2 returns it
    index: int = 0          # running count of frames from this source (gaps = dropped frames)


@dataclass(frozen=True)
class Crop:
    t: float                # capture time of the frame this crop came from
    roi96: np.ndarray       # (96, 96) uint8 grayscale mouth patch, aligned like Lipflow mouth_rois (zeros if no face yet)
    mouth_open: float       # inner-lip gap / mouth width (Lipflow FaceObs.mouth_open); 0.0 when face_ok is False
    face_ok: bool           # landmarks found on this frame (False: crop interpolated or held from the last face)
    blend: np.ndarray = None    # (52,) float32 MediaPipe blendshape scores, order live.crop.BLEND_NAMES; None: no face
    mouth_pts: np.ndarray = None  # (4, 2) float32 pixels: inner lip top 13, bottom 14, corners 61, 291; None: no face
    eye_dist: float = 0.0       # distance between the eye centres (pixels), the face-size scale; 0.0: no face


@dataclass(frozen=True)
class WordEnd:
    w: str                  # one word, UPPERCASE, as LipReader prints it
    end_frame: int          # model frame (0 = utterance start) of the word's last non-blank CTC frame


@dataclass(frozen=True)
class LiveHyp:
    """One live decode of the whole utterance so far: input of commit.py (camera-only fallback)."""
    utt_id: int
    t0: float               # capture time of the utterance's first model frame
    n_frames: int           # model frames encoded in this update
    words: tuple            # tuple[WordEnd, ...] in reading order
    trailing_blanks: int    # CTC blank frames at the end of the utterance (best path), for the auto endpoint

    @property
    def t(self):
        """Capture time of the newest frame in this update."""
        return frame_time(self.t0, self.n_frames - 1)


@dataclass(frozen=True)
class Word:
    w: str
    state: WordState
    t_end: float            # capture time of the word's CTC end (frame_time(t0, end_frame))


@dataclass(frozen=True)
class Update:
    utt_id: int
    t: float                # capture time of the newest frame the update saw
    words: tuple            # tuple[Word, ...]: the full live text of the utterance, solid words first





# Team bus (docs/PLAN.md#interfaces, live/bus_schema.md): what the camera reader publishes. Times are time.monotonic()
# seconds on the laptop. T = 25 fps model frames of the current segment, K = TOPK tokens per frame.
TOPK = 32
BLANK = 0                   # CTC blank id; Auto-AVSR vocabulary: 0 blank, 1..5047 unigram5000 units, 5048 <eos>
VOCAB = 5049
TIMING_KEYS = ('T', 'mediapipe', 'frontend', 'conformer', 'ctc', 'total')  # CamTick.timing_ms
EMOTIONS = ('neutral', 'happy', 'sad', 'angry', 'surprised')  # placeholder until Frank confirms Cartesia's names


@dataclass(frozen=True)
class Emotion:
    label: str              # one of EMOTIONS
    confidence: float       # 0..1, of the current label
    scores: dict            # {label: 0..1} for every label in EMOTIONS


@dataclass(frozen=True)
class CamTick:
    """One re-read of the whole current segment, every 100 ms while a segment is open (topic cam.tick)."""
    seg_id: int             # >= 1, new segment = new id (an 8 s cap also starts a new id, carrying 2 s over)
    seg_start: float        # capture time of the segment's first model frame
    frame_t: np.ndarray     # (T,) float64, capture time of each model frame, increasing
    topk_ids: np.ndarray    # (T, K) int32 token ids, most likely first (blank may appear)
    topk_logp: np.ndarray   # (T, K) float32 CTC log-probs of topk_ids
    blank_logp: np.ndarray  # (T,) float32 CTC log-prob of the blank
    greedy: str             # CTC best path as text, UPPERCASE, for debugging only
    activity: np.ndarray    # (T,) float32 mouthing activity 0..1
    face_ok: bool           # face found on the newest frame
    emotion: Emotion
    timing_ms: dict         # cost of this tick in ms, keys TIMING_KEYS: {'T': int (== len(frame_t)), 'mediapipe',
    #                         'frontend', 'conformer', 'ctc', 'total': float >= 0}; live.bus.zero_timing(T) if untimed


@dataclass(frozen=True)
class CamFinal:
    """Beam search + LM over the finished segment (topic cam.final), best first."""
    seg_id: int
    nbest: tuple            # tuple[(text, score), ...], UPPERCASE text, score = log-prob (higher is better)
