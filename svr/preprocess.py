"""Raw EMG -> model input.

Reimplements the input path of Gaddy's read_emg.load_utterance + EMGDataset.__getitem__:

1. Concatenate the previous and next recordings of the session around the target one,
   so the zero-phase filters have context at both edges.
2. Notch 60 Hz and its harmonics up to 420 Hz (Q=30), then a 3rd-order 2 Hz high-pass.
   Both run forward and backward (filtfilt), so there is no phase shift.
3. Cut the target recording back out and linearly resample 1000 Hz -> 689.06 Hz,
   which gives 8 input samples per 86.13 Hz output frame.
4. Trim to a whole number of output frames, skipping the first 8 samples.
5. Scale: x / 20, then soft-clip with 50 * tanh(x / 50).

The frame count in step 4 follows Gaddy exactly: it is the smaller of the frame count of his
hand-crafted features (computed at 516.79 Hz, 16-sample windows, hop 6) and the frame count
of the parallel audio's mel spectrogram (22050 Hz, hop 256). The audio only sets the length;
pass audio_path=None to derive it from the EMG alone, as the live rig will.
"""
import math
import os

import numpy as np
import scipy.signal
import soundfile as sf

EMG_RATE = 1000
RAW_RATE = 689.06
FEATURE_RATE = 516.79
SAMPLES_PER_FRAME = 8
NOTCH_FREQ = 60
NOTCH_HARMONICS = 7
NOTCH_Q = 30
HIGHPASS_HZ = 2
HIGHPASS_ORDER = 3


def notch_harmonics(x, freq=NOTCH_FREQ, fs=EMG_RATE):
    for harmonic in range(1, NOTCH_HARMONICS + 1):
        b, a = scipy.signal.iirnotch(freq * harmonic, NOTCH_Q, fs)
        x = scipy.signal.filtfilt(b, a, x, axis=0)
    return x


def highpass(x, fs=EMG_RATE):
    b, a = scipy.signal.butter(HIGHPASS_ORDER, HIGHPASS_HZ, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, x, axis=0)


def resample_linear(x, new_rate, old_rate=EMG_RATE):
    times = np.arange(len(x)) / old_rate
    new_times = np.arange(0, times[-1], 1 / new_rate)
    return np.stack([np.interp(new_times, times, x[:, c]) for c in range(x.shape[1])], 1)


def resampled_length(n, new_rate, old_rate=EMG_RATE):
    return len(np.arange(0, (n - 1) / old_rate, 1 / new_rate))


def load_filtered_emg(session_dir, index):
    """Filtered 1000 Hz EMG of one recording, (time, channels), float64."""
    def load(i):
        path = os.path.join(session_dir, f'{i}_emg.npy')
        return np.load(path) if os.path.exists(path) else None

    emg = load(index)
    before = load(index - 1)
    after = load(index + 1)
    n_before = 0 if before is None else len(before)
    n_after = 0 if after is None else len(after)
    parts = [p for p in (before, emg, after) if p is not None]
    x = np.concatenate(parts, 0)
    x = highpass(notch_harmonics(x))
    return x[n_before:len(x) - n_after]


def audio_frame_count(audio_path):
    info = sf.info(audio_path)
    n = info.frames
    if info.samplerate == 16000:
        n = int(math.ceil(n * 22050 / 16000))
    else:
        assert info.samplerate == 22050
    return n // 256


def emg_frame_count(n_emg_samples):
    n = resampled_length(n_emg_samples, FEATURE_RATE)
    return 1 + (n - 16) // 6


def scale(raw):
    raw = raw.astype(np.float32) / 20
    return 50 * np.tanh(raw / 50.)


def prepare_emg(filtered_emg, n_frames=None, keep_channels=None):
    """Filtered 1000 Hz EMG -> scaled model input at 689.06 Hz, (n_frames * 8, channels), float32.

    keep_channels: channel indices to keep; the others are set to zero (simulates fewer electrodes).
    """
    x = resample_linear(filtered_emg, RAW_RATE)
    if keep_channels is not None:
        drop = [c for c in range(x.shape[1]) if c not in set(keep_channels)]
        x[:, drop] = 0
    if n_frames is None:
        n_frames = emg_frame_count(len(filtered_emg))
    x = x[SAMPLES_PER_FRAME:SAMPLES_PER_FRAME + SAMPLES_PER_FRAME * n_frames]
    assert len(x) == SAMPLES_PER_FRAME * n_frames, (len(x), n_frames)
    return scale(x)


def load_utterance(session_dir, index, keep_channels=None, use_audio_length=True):
    """Model input for one recording in Gaddy's dataset layout."""
    emg = load_filtered_emg(session_dir, index)
    n_frames = emg_frame_count(len(emg))
    if use_audio_length:
        n_frames = min(n_frames, audio_frame_count(os.path.join(session_dir, f'{index}_audio_clean.flac')))
    return prepare_emg(emg, n_frames, keep_channels)
