"""One lock score per word, and the threshold it must cross. Hand-set v1, to be replaced by the learned lock model.

The score takes a feature dict, so a model trained on sessions 1 and 2 ("was this word right and final?") can
replace LockModel without touching the decoder:

    cam       camera confidence 0..1
    emg       EMG support 0..1, or None until EMG exists
    claude    Claude's predicted weight for the word 0..1 (0 when no live prediction)
    finished  a later word has started, the lips paused, or the segment ended
    passes    passes the word survived unchanged
    face_ok   the face was found on the newest frame

    sensor = gate * cam + (1 - gate) * emg          the readers' support; gate = 1 (camera only) while emg is None
    score  = W_SENSOR * sensor + W_STABLE * stability + min(W_CLAUDE * claude, CLAUDE_CAP)
             times UNFINISHED if not finished, times FACE_LOST if not face_ok

Claude alone can never lock a word: with sensor = 0 the score is at most W_STABLE + CLAUDE_CAP = 0.30, below
the lowest threshold, and a word also needs sensor >= SENSOR_MIN whatever Claude says.

The threshold moves with Claude's predictability: THRESHOLD at 0.5 (or with no live prediction), down to
THRESHOLD - SWING when the next words are fully predictable, up to THRESHOLD + SWING when anything could come.
"""
W_SENSOR = 0.70
W_STABLE = 0.15
PASSES_FULL = 3       # passes for full stability credit (300 ms at 100 ms ticks)
W_CLAUDE = 0.20
CLAUDE_CAP = 0.15     # Claude's share of the score never exceeds this
UNFINISHED = 0.5      # a word still being mouthed scores low by design
FACE_LOST = 0.8
SENSOR_MIN = 0.45     # readers' support below this never locks
THRESHOLD = 0.72
SWING = 0.08


def gate(f):
    """Camera weight 0..1 against EMG. Today EMG is None, so the camera carries everything.

    Hook for EMG: a learned gate will look at camera confidence, face_ok, EMG signal quality and pace. Until
    then, with EMG present, the camera counts half when the face is visible and little when it is not.
    """
    if f.get('emg') is None:
        return 1.0
    return 0.5 if f.get('face_ok', True) else 0.1


def sensor(f):
    g = gate(f)
    return g * f['cam'] + (1 - g) * (f['emg'] or 0.0)


class LockModel:
    """Hand-set weights. A learned model only needs score(f) and locks(f, threshold)."""

    def score(self, f):
        s = W_SENSOR * sensor(f) + W_STABLE * min(f['passes'], PASSES_FULL) / PASSES_FULL \
            + min(W_CLAUDE * f['claude'], CLAUDE_CAP)
        if not f['finished']:
            s *= UNFINISHED
        if not f.get('face_ok', True):
            s *= FACE_LOST
        return s

    def locks(self, f, threshold):
        return sensor(f) >= SENSOR_MIN and self.score(f) >= threshold


def threshold(predictability=None):
    """predictability 0..1 from the newest live claude.predict, None without one."""
    if predictability is None:
        return THRESHOLD
    return THRESHOLD - SWING * (2 * min(max(predictability, 0.0), 1.0) - 1)
