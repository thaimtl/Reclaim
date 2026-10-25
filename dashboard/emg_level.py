"""How hard each pad's muscle is working, for the dashboard's pad meters, from raw emg.raw samples.

The raw signal is mostly mains hum and drift: on Austin's session 3, the 10 ms RMS of the raw chin channel was 6.6 at
rest and 10.6 while speaking (90th percentile), so a meter on raw RMS looks busy when he is still. So, like everything
that reads EMG (CLAUDE.md), the samples first go through rig/filters.py's filter: 20 Hz high-pass and notches at 60,
120 and 180 Hz, causal, in order. Its coefficients are copied here (the dashboard's venv has no scipy); a test (where
scipy is installed) checks the output still matches rig/filters.py sample for sample. Then each pad's level is how far its RMS is above FLOOR_MARGIN times
its own resting floor, against its own recent range: a still face reads 0, real activity fills the meter.
"""
import numpy as np

# rig.filters.design(fs=1000, mains=60): second-order sections (b0 b1 b2 a0 a1 a2), and sosfilt_zi for a unit step
SOS = np.array([
    [0.848475295524359, -1.696950591048718, 0.848475295524359, 1.0, -1.778313488139435, 0.7924474718329468],
    [1.0, -2.0, 1.0, 1.0, -1.8934156010225003, 0.9084644129492953],
    [0.9814970254751076, -1.8251457105120343, 0.9814970254751075, 1.0, -1.8251457105120346, 0.9629940509502157],
    [0.9636538841655016, -1.404946898498875, 0.9636538841655018, 1.0, -1.404946898498875, 0.9273077683310031],
    [0.9464238710518653, -0.8059353706734738, 0.9464238710518652, 1.0, -0.8059353706734735, 0.8928477421037307]])
ZI = np.array([[-0.8484752955243585, 0.8484752955243586], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

FLOOR_DOWN, FLOOR_UP = 0.2, 0.002   # the resting floor follows a quieter pad at once, a louder one only slowly
FLOOR_MARGIN = 2.0                  # activity starts at this many times the floor. Measured on sessions 2 and 3: a
                                    # still face lights a pad 0-1% of the time, speech 36% (raw RMS, as before: 10-34%)
MIN_SPAN = 2.0                      # filtered RMS units: the smallest range a full meter stands for
PEAK_DECAY = 0.995                  # per message: the running peak a pad's level is scaled against
SMOOTH = 0.35                       # level EMA per message (10 ms)
POWER = 0.3                         # the muscle's power, an EMA per message (~30 ms): a 10-sample RMS is too jumpy
SETTLE = 30                         # messages (300 ms) a pad's filter settles, from the start or back on the skin:
                                    # its meter reads 0 and its floor and peak hold meanwhile


class EmgFilter:
    """rig.filters.EmgFilter in numpy: causal, stateful, the same output sample for sample (n x channels)."""

    def __init__(self):
        self.zi = None

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.zi is None:
            self.zi = ZI[:, :, None] * x[0]
        y = np.empty_like(x)
        for n, v in enumerate(x):                       # direct form II transposed, section by section
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
        back = contact & ~self.contact      # on the skin from now (or from the start): the filter rings for a while
        self.contact = contact.copy()       # (a jump from a rail, the hum it starts in), so that pad settles first
        self.settle[back] = SETTLE
        p = (self.filter(x) ** 2).mean(0)
        self.power = p if self.power is None else self.power + POWER * (p - self.power)
        rms = np.sqrt(self.power)
        live = contact & (self.settle == 0)
        self.settle = np.maximum(0, self.settle - 1)
        contact = live
        first = ~np.isfinite(self.floor) & contact
        self.floor[first] = rms[first]
        with np.errstate(invalid='ignore'):     # a pad never on the skin yet has no floor (inf): held, level 0
            rate = np.where(rms < self.floor, FLOOR_DOWN, FLOOR_UP)
            # off the skin (on a rail, or flat) a pad would drag its floor and peak far off: they hold until it's back
            self.floor = np.where(contact, self.floor + rate * (rms - self.floor), self.floor)
            rest = FLOOR_MARGIN * self.floor
            self.peak = np.where(contact, np.maximum(self.peak * PEAK_DECAY, rms), self.peak)
            span = np.maximum(MIN_SPAN, self.peak - rest)
            target = np.where(contact & np.isfinite(rest), np.clip((rms - rest) / span, 0.0, 1.0), 0.0)
        self.level += SMOOTH * (target - self.level)
        return self.level.copy()
