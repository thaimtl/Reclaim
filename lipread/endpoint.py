"""Mouthing activity and automatic segments (E4, docs/EVAN_REALTIME_LIP_READING.md): what the camera re-reads every
100 ms.

Activity (0..1, per Crop): how much the lips move over the last window_s. Three lip measures per frame, all distances
between lip points divided by eye_dist, so head distance (scale), translation, in-plane tilt and nodding (all points
move together) cancel out:
    gap    inner-lip gap 13-14 (as Lipflow's FaceObs.mouth_open, but over eye distance)
    width  corner to corner 61-291
    lift   top lip 13 above the corner line (corners rise in a smile, fall in a frown)
Each is median-filtered over 3 frames (one-frame landmark jitter vanishes), then
    activity = min(1, (ptp(gap) + 0.5 ptp(width) + 0.5 ptp(lift)) / ACT_FULL)    ptp over the last window_s
A crop without a face has activity 0 and clears the history (no jump when the face comes back).

Segments, with hysteresis: activity >= start_thr opens a candidate; it becomes a segment once activity stayed >=
still_thr for min_active_s (a twitch or a two-frame glitch shorter than that is rejected), and the segment includes
the preroll_s of crops before the rise (only crops after the previous segment's last one: back-to-back segments
never share a crop). It ends after still_s of activity < still_thr (lost face counts as still).
At cap_s it is closed and a new segment (next seg_id) opens with the last carry_s of crops. All durations become
frame counts at FPS = 25 (one Crop per model frame); time comes only from crop.t, so replays are deterministic.
never_end=True (run.py --segments stream: debugging, replays of continuous speech): the first crop opens a segment
and stillness never ends it; only the cap (and flush) closes it.

    seg = Segmenter()
    for crop in crops:
        for kind, s in seg.push(crop):      # ('start', open seg) | ('cap', closed seg) | ('end', closed seg)
            ...
        if seg.segment: reread(seg.segment.crops)                # every 100 ms in the runner
    seg.flush()                                                  # end of a replay: ('end', seg) if one is open
"""
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lipread.types import FPS

log = logging.getLogger(__name__)

START, CAP, END = 'start', 'cap', 'end'
ACT_FULL = 0.15
W_WIDTH, W_LIFT = 0.5, 0.5
START_THR = 0.25
STILL_THR = 0.12
MEDIAN_N = 3


def frames(seconds):
    """Duration in 25 fps model frames, rounded half up (0.3 s -> 8, 0.5 s -> 13, 8 s -> 200)."""
    return max(1, int(seconds * FPS + 0.5))


def lip_features(crop):
    """(gap, width, lift) in eye distances from crop.mouth_pts, or None without a usable face."""
    if not crop.face_ok or crop.mouth_pts is None or not crop.eye_dist > 0:
        return None
    top, bot, left, right = np.asarray(crop.mouth_pts, np.float64)
    axis = right - left
    width = float(np.hypot(*axis))
    if width <= 0:
        return None
    rel = top - (left + right) / 2
    lift = (axis[0] * rel[1] - axis[1] * rel[0]) / width
    return np.array([np.hypot(*(top - bot)), width, lift]) / crop.eye_dist


class Activity:
    """Per-frame mouthing activity 0..1: act = Activity(); a = act.push(crop)."""

    def __init__(self, window_s=0.3, act_full=ACT_FULL):
        self.act_full = act_full
        self.weights = np.array([1.0, W_WIDTH, W_LIFT])
        self.raw = deque(maxlen=MEDIAN_N)
        self.window = deque(maxlen=frames(window_s))

    def reset(self):
        self.raw.clear()
        self.window.clear()

    def push(self, crop):
        f = lip_features(crop)
        if f is None:
            self.reset()
            return 0.0
        self.raw.append(f)
        self.window.append(np.median(np.array(self.raw), axis=0))
        w = np.array(self.window)
        return float(min(1.0, (np.ptp(w, axis=0) @ self.weights) / self.act_full))


@dataclass
class Segment:
    seg_id: int
    crops: list = field(default_factory=list)
    activity: list = field(default_factory=list)

    @property
    def t_start(self):
        return self.crops[0].t

    @property
    def duration_s(self):
        return len(self.crops) / FPS


class Segmenter:
    def __init__(self, start_thr=START_THR, still_thr=STILL_THR, preroll_s=0.3, still_s=0.5, cap_s=8.0, carry_s=2.0,
                 window_s=0.3, min_active_s=0.4, act_full=ACT_FULL, never_end=False):
        if not still_thr <= start_thr:
            raise ValueError(f'still_thr {still_thr} must be <= start_thr {start_thr}')
        self.start_thr, self.still_thr = start_thr, still_thr
        self.n_pre, self.n_still, self.n_min = frames(preroll_s), frames(still_s), frames(min_active_s)
        self.n_cap, self.n_carry = frames(cap_s), frames(carry_s)
        if not self.n_carry < self.n_cap:
            raise ValueError(f'carry_s {carry_s} must be shorter than cap_s {cap_s}')
        self.never_end = never_end
        self.act = Activity(window_s, act_full)
        self.next_id = 1
        self.rejected = 0
        self.face_lost = 0
        self.reset()

    def reset(self):
        """Forget activity history, the pre-roll buffer and any open segment (no event)."""
        self.act.reset()
        self.segment = None
        self.activity = 0.0
        self.buf = deque(maxlen=self.n_pre + self.n_min)
        self.run = 0
        self.quiet = 0

    def push(self, crop):
        """Add one Crop; returns the events it caused, in order."""
        a = self.activity = self.act.push(crop)
        if not crop.face_ok:
            self.face_lost += 1
        self.buf.append((crop, a))
        if self.segment is None and self.never_end:
            return [self._open([(crop, a)])]
        if self.segment is None:
            return self._idle(crop, a)
        seg = self.segment
        seg.crops.append(crop)
        seg.activity.append(a)
        self.quiet = self.quiet + 1 if a < self.still_thr else 0
        if self.quiet >= self.n_still and not self.never_end:
            return [self._end(crop.t)]
        if len(seg.crops) >= self.n_cap:
            new = Segment(self._new_id(), seg.crops[-self.n_carry:], seg.activity[-self.n_carry:])
            log.info('segment %d capped at %.2f (%.2f s); segment %d carries %d frames', seg.seg_id, crop.t,
                     seg.duration_s, new.seg_id, self.n_carry)
            self.segment = new
            return [(CAP, seg), (START, new)]
        return []

    def _idle(self, crop, a):
        if self.run == 0:
            self.run = 1 if a >= self.start_thr else 0
        elif a >= self.still_thr:
            self.run += 1
        else:
            log.debug('candidate rejected at %.2f after %d frames', crop.t, self.run)
            self.rejected += 1
            self.run = 0
        if self.run < self.n_min:
            return []
        items = list(self.buf)[-(self.n_pre + self.run):]
        log.info('rise at %.2f, %d pre-roll frames', items[-self.n_min][0].t, len(items) - self.n_min)
        return [self._open(items)]

    def _open(self, items):
        """items: (crop, activity) of the new segment's first crops."""
        self.segment = Segment(self._new_id(), [c for c, _ in items], [x for _, x in items])
        self.run = self.quiet = 0
        log.info('segment %d start at %.2f', self.segment.seg_id, self.segment.t_start)
        return START, self.segment

    def _new_id(self):
        self.next_id += 1
        return self.next_id - 1

    def _end(self, t):
        seg, self.segment = self.segment, None
        self.quiet = 0
        self.buf.clear()
        log.info('segment %d end at %.2f: %.2f s, mean activity %.2f', seg.seg_id, t, seg.duration_s,
                 float(np.mean(seg.activity)))
        return END, seg

    def flush(self):
        """End of input: ('end', seg) for an open segment, then a clean state."""
        out = [self._end(self.segment.crops[-1].t)] if self.segment is not None else []
        self.reset()
        return out
