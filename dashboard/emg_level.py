"""Pad meter levels from emg.raw: filtered as rig/filters.py, measured above each pad's resting floor."""
import numpy as np

# rig.filters.design(fs=1000, mains=60) and its sosfilt_zi, copied since the dashboard venv has no scipy
SOS = np.array([
    [0.848475295524359, -1.696950591048718, 0.848475295524359, 1.0, -1.778313488139435, 0.7924474718329468],
    [1.0, -2.0, 1.0, 1.0, -1.8934156010225003, 0.9084644129492953],
    [0.9814970254751076, -1.8251457105120343, 0.9814970254751075, 1.0, -1.8251457105120346, 0.9629940509502157],
    [0.9636538841655016, -1.404946898498875, 0.9636538841655018, 1.0, -1.404946898498875, 0.9273077683310031],
    [0.9464238710518653, -0.8059353706734738, 0.9464238710518652, 1.0, -0.8059353706734735, 0.8928477421037307]])
ZI = np.array([[-0.8484752955243585, 0.8484752955243586], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

FLOOR_DOWN, FLOOR_UP = 0.2, 0.002   # floor drops fast, rises slowly
FLOOR_MARGIN = 2.0                  # activity starts at this many times the floor
MIN_SPAN = 2.0                      # smallest RMS range a full meter stands for
PEAK_DECAY = 0.995                  # running peak decay per message
SMOOTH = 0.35                       # level EMA per message (10 ms)
POWER = 0.3                         # power EMA per message (~30 ms)
SETTLE = 30                         # messages a pad reads 0 after regaining contact


class EmgFilter:
    """rig.filters.EmgFilter in numpy, sample-for-sample identical."""

    def __init__(self):
        self.zi = None

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.zi is None:
            self.zi = ZI[:, :, None] * x[0]
        y = np.empty_like(x)
        for n, v in enumerate(x):                       # direct form II transposed
            for s, (b0, b1, b2, _, a1, a2) in enumerate(SOS):
                out = b0 * v + self.zi[s, 0]
                self.zi[s, 0] = b1 * v - a1 * out + self.zi[s, 1]
                self.zi[s, 1] = b2 * v - a2 * out
                v = out
            y[n] = v
        return y


class PadLevels:
    """push(samples n x pads) -> each pad's level in [0, 1]."""

    def __init__(self, pads):
        self.filter = EmgFilter()
        self.power = None
        self.contact = np.zeros(pads, bool)
        self.settle = np.zeros(pads, int)
        self.floor = np.full(pads, np.inf)
        self.peak = np.zeros(pads)
        self.level = np.zeros(pads)

    def push(self, x, contact):
        back = contact & ~self.contact
        self.contact = contact.copy()
        self.settle[back] = SETTLE
        p = (self.filter(x) ** 2).mean(0)
        self.power = p if self.power is None else self.power + POWER * (p - self.power)
        rms = np.sqrt(self.power)
        live = contact & (self.settle == 0)
        self.settle = np.maximum(0, self.settle - 1)
        contact = live
        first = ~np.isfinite(self.floor) & contact
        self.floor[first] = rms[first]
        with np.errstate(invalid='ignore'):     # inf floor = never on skin yet
            rate = np.where(rms < self.floor, FLOOR_DOWN, FLOOR_UP)
            self.floor = np.where(contact, self.floor + rate * (rms - self.floor), self.floor)
            rest = FLOOR_MARGIN * self.floor
            self.peak = np.where(contact, np.maximum(self.peak * PEAK_DECAY, rms), self.peak)
            span = np.maximum(MIN_SPAN, self.peak - rest)
            target = np.where(contact & np.isfinite(rest), np.clip((rms - rest) / span, 0.0, 1.0), 0.0)
        self.level += SMOOTH * (target - self.level)
        return self.level.copy()
