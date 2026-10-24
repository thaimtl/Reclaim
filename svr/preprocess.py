"""Raw EMG -> model input. Two preprocessings:

causal (SVR-1, everything since docs/SVR1_SPEC.md section 1): the rig's own filter, `rig/filters.py` (20 Hz high-pass
+ 60/120/180 Hz notches, sosfilt in time order), so training sees exactly what the live loop sees.

1. Warm the filter up on the first second of the recording played backwards (it joins the recording without a jump;
   live, the filter has been running for minutes), then filter the recording itself in order. Never filtfilt.
2. Resample 1000 Hz -> 689.06 Hz and trim to whole frames, as below.
3. The cache holds this unscaled signal. Scaling happens at load time: multiply channel c by the session's gain
   target_c / spread_c (robust spread, `robust_spread`; targets in data/svr1_targets/scale.json), then the soft clip.

gaddy (the Gaddy-architecture baselines B0/B1, kept for parity with his released model) reimplements the input path
of Gaddy's read_emg.load_utterance + EMGDataset.__getitem__:

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
import importlib.util
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
WARMUP_SAMPLES = 1000  # 1 s: the 20 Hz high-pass and the Q=10 notches settle in ~0.1 s
PREPROCESSINGS = ('causal', 'gaddy')

# rig/filters.py is the one filter used live and in training. Loaded by path (the repo's rig/ is not a package, and
# the working directory may reach svr/ through a symlink).
_spec = importlib.util.spec_from_file_location(
    'reclaim_rig_filters', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'rig', 'filters.py'))
rig_filters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rig_filters)


def notch_harmonics(x, freq=NOTCH_FREQ, fs=EMG_RATE):
    for harmonic in range(1, NOTCH_HARMONICS + 1):
        b, a = scipy.signal.iirnotch(freq * harmonic, NOTCH_Q, fs)
        x = scipy.signal.filtfilt(b, a, x, axis=0)
    return x


def highpass(x, fs=EMG_RATE):
    b, a = scipy.signal.butter(HIGHPASS_ORDER, HIGHPASS_HZ, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, x, axis=0)


def causal_filter(x, warmup=WARMUP_SAMPLES):
    """rig/filters.py over one recording (time, channels), in order, after warming up on its first `warmup` samples
    reversed. Same output as the live filter would give once settled."""
    x = np.asarray(x, dtype=np.float64)
    f = rig_filters.EmgFilter()
    if warmup:
        f(x[:min(warmup, len(x))][::-1])
    return f(x)


def robust_spread(x):
    """Per-channel robust standard deviation, 1.4826 * median absolute deviation, of (time, channels)."""
    x = np.asarray(x, dtype=np.float64)
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


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


def unscale(y):
    """Inverse of scale() (up to the clip), for augmentations that act on muscle amplitude."""
    return 20 * 50 * np.arctanh(np.clip(np.asarray(y, np.float32) / 50, -0.999, 0.999))


def frames_of(filtered_emg, n_frames=None):
    """Filtered 1000 Hz EMG -> unscaled 689.06 Hz signal trimmed to whole frames, (n_frames * 8, channels), float32."""
    x = resample_linear(filtered_emg, RAW_RATE)
    if n_frames is None:
        n_frames = emg_frame_count(len(filtered_emg))
    x = x[SAMPLES_PER_FRAME:SAMPLES_PER_FRAME + SAMPLES_PER_FRAME * n_frames]
    assert len(x) == SAMPLES_PER_FRAME * n_frames, (len(x), n_frames)
    return x.astype(np.float32)


def prepare_emg(filtered_emg, n_frames=None, keep_channels=None, gain=None):
    """Filtered 1000 Hz EMG -> scaled model input at 689.06 Hz, (n_frames * 8, channels), float32.

    keep_channels: channel indices to keep; the others are set to zero (simulates fewer electrodes).
    gain: per-channel scale factors applied before the soft clip (causal preprocessing; None = 1).
    """
    return apply_gain(frames_of(filtered_emg, n_frames), gain, keep_channels)


def n_frames_of(session_dir, index, n_samples, use_audio_length=True):
    n_frames = emg_frame_count(n_samples)
    if use_audio_length:
        n_frames = min(n_frames, audio_frame_count(os.path.join(session_dir, f'{index}_audio_clean.flac')))
    return n_frames


def load_unscaled(session_dir, index, use_audio_length=True):
    """Causal preprocessing up to (not including) scaling, for one recording in Gaddy's layout: what the cache holds."""
    emg = causal_filter(np.load(os.path.join(session_dir, f'{index}_emg.npy')))
    return frames_of(emg, n_frames_of(session_dir, index, len(emg), use_audio_length))


def apply_gain(x, gain, keep_channels=None):
    """Unscaled cached signal -> model input: per-channel gain, soft clip, absent channels zeroed."""
    y = scale(x if gain is None else x * np.asarray(gain, np.float32)[None])
    if keep_channels is not None:
        y[:, [c for c in range(y.shape[1]) if c not in set(keep_channels)]] = 0
    return y


def load_utterance(session_dir, index, keep_channels=None, use_audio_length=True, preprocess='gaddy', gain=None):
    """Model input for one recording in Gaddy's dataset layout. preprocess='causal' needs the session's gain
    (svr.data.session_gain); 'gaddy' is his original zero-phase pipeline."""
    if preprocess == 'causal':
        if gain is None:
            raise ValueError('causal preprocessing needs the session gain (svr.data.session_gain)')
        return apply_gain(load_unscaled(session_dir, index, use_audio_length), gain, keep_channels)
    emg = load_filtered_emg(session_dir, index)
    return prepare_emg(emg, n_frames_of(session_dir, index, len(emg), use_audio_length), keep_channels)


# ------------------------------------------------------------------------------------------------ our rig
# Rig pins A0..A3 = Gaddy sites 3, 2, 6, 1 -> model slots (site - 1); the other 4 slots stay zero and masked.
RIG_TO_SLOT = [2, 1, 5, 0]
KEEP = sorted(RIG_TO_SLOT)
N_SLOTS = 8


def rig_to_slots(x):
    """(time, 4) rig columns in pin order -> (time, 8) Berkeley channel slots."""
    out = np.zeros((len(x), N_SLOTS), np.float32 if x.dtype == np.float32 else np.float64)
    out[:, RIG_TO_SLOT] = x
    return out


def rig_unscaled(raw):
    """Raw Uno readings of one clip (samples, 4), any dtype -> filtered, resampled, unscaled (frames * 8, 8 slots)."""
    return frames_of(rig_to_slots(causal_filter(np.asarray(raw, np.float64))))


def rig_gain(spread, target, keep=KEEP):
    """Per-slot gain target / spread on the slots in use (0 elsewhere): spread from robust_spread over a session's
    unscaled signal (or the live calibration minute), target = Berkeley's (scale.json / the checkpoint)."""
    spread, target = np.asarray(spread, np.float64), np.asarray(target, np.float64)
    gain = np.zeros(N_SLOTS, np.float32)
    for c in keep:
        gain[c] = target[c] / spread[c] if spread[c] > 1e-9 else 0.0
    return gain
