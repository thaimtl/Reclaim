import numpy as np
from scipy import signal

FS = 1000
MAINS = 60


def design(fs=FS, mains=MAINS, q=10):
    highpass = signal.butter(4, 20, "highpass", fs=fs, output="sos")
    notches = [signal.tf2sos(*signal.iirnotch(f, q, fs)) for f in (mains, 2 * mains, 3 * mains)]
    return np.vstack([highpass, *notches])


class EmgFilter:
    """Causal and stateful, so training must run it over raw recordings in order, never with filtfilt."""

    def __init__(self, fs=FS, mains=MAINS):
        self.sos = design(fs, mains)
        self.zi = None

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y
