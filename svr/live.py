"""SVR-1 in the live loop (docs/SVR1_SPEC.md section 5): the EMG half of the fusion process.

    live = LiveSVR1('checkpoints/W_svr1/model.pt')       # session gains and effort reference from fine-tuning
    live.calibrate(raw_minute)                            # demo day: ~1 min of rig data -> new gains + effort reference
    on every emg.raw message:   live.feed(msg['samples'], msg['t0'])
    every 100 ms:               out = live.read(camera_active=...)
    for disputed words:         live.verify(['i need to go', 'i need to know'])

No bus code here: the fusion process subscribes to `emg.raw` and calls this. The filter is rig/filters.py, run once per
sample as samples arrive (state kept across messages), exactly as in training; slots, gain and soft clip are
svr.preprocess's. Each read re-runs SVR-1 over the whole current segment (up to --max-seconds; past that the window
restarts with the last --context-seconds), or over the last --idle-seconds while nobody is speaking, to catch the
start of speech.

read() returns per-frame letter and phoneme log-probabilities, the speaking probability, frame times, and per read the
rate z-score, `pace` and `effort` in voice.say's units (1.0 = Austin's normal; docs/TEAM_TASKS_AND_INTERFACES.md):
    pace    exp(0.37 * rate z-score) / the calibration's median: relative speaking rate
    effort  RMS of the scaled signal over the last 100 ms / the session's speaking median: relative muscle effort
Segments start when the speaking probability (or the camera's mouth activity) rises, 300 ms earlier, and end after
0.5 s with neither; the read where one starts says so (segment_started) and the read where one ends too (segment_ended).
"""
import time

import numpy as np
import torch

from svr.preprocess import (EMG_RATE, KEEP, RAW_RATE, SAMPLES_PER_FRAME, apply_gain, resample_linear, rig_filters,
                            rig_gain, rig_to_slots, rig_unscaled, robust_spread)
from svr.svr1 import build, config_of, score_texts
from svr.text import clean_text

FRAME_RATE = RAW_RATE / SAMPLES_PER_FRAME
RATE_SCALE = 0.37            # svr.train_svr1.RATE_SCALE: rate head z-score -> log letters per second
EFFORT_FRAMES = int(round(0.1 * FRAME_RATE))


def frame_rms(x):
    """Per-frame RMS over the slots in use of a scaled model input (frames * 8, 8), smoothed over 100 ms."""
    f = x.reshape(-1, SAMPLES_PER_FRAME, x.shape[1])[:, :, KEEP]
    e = (f.astype(np.float64) ** 2).mean((1, 2))
    k = min(EFFORT_FRAMES, len(e)) or 1
    return np.sqrt(np.convolve(e, np.ones(k) / k, 'full')[:len(e)])  # causal: each frame sees the 100 ms before it


class LiveSVR1:
    def __init__(self, checkpoint, device=None, session=None, max_seconds=8.0, context_seconds=2.0,
                 idle_seconds=2.0, buffer_seconds=60.0, speak_threshold=0.5, pre_roll=0.3, end_silence=0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('preprocess', 'gaddy') != 'causal':
            raise ValueError(f'{checkpoint} was not trained on the causal preprocessing (rig/filters.py)')
        self.model = build(config_of(state), attention='sdpa')
        self.model.load_state_dict(state['model'])
        self.model.to(self.device).eval()
        self.target = np.asarray(state['scale_target'], np.float64)
        # until calibrate(): the gain and effort reference of `session`, default the last training session
        session = session or state.get('default_session')
        gains = state.get('session_gains') or {}
        self.gain = np.asarray(gains.get(session, gains[sorted(gains)[-1]]), np.float32) if gains else None
        refs = state.get('effort_ref') or {}
        self.effort_ref = float(refs.get(session, refs[sorted(refs)[-1]])) if refs else None
        paces = state.get('pace_ref') or {}  # Austin's normal pace; 1.0 = Berkeley's average speaker without it
        self.pace_ref = float(paces.get(session, paces[sorted(paces)[-1]])) if paces else 1.0
        self.cmask = torch.zeros(8, device=self.device)
        self.cmask[KEEP] = 1
        self.filter = rig_filters.EmgFilter()
        self.size = int(buffer_seconds * EMG_RATE)
        self.ring = np.zeros((self.size, 4), np.float64)   # filtered 1 kHz signal, rig column order
        self.n = 0                                         # samples received so far (absolute index of the next one)
        self.clock = None                                  # (absolute index, monotonic time) of the latest message
        self.max_s, self.ctx_s, self.idle_s = (int(s * EMG_RATE) for s in (max_seconds, context_seconds, idle_seconds))
        self.thr, self.pre_roll, self.end_silence = speak_threshold, pre_roll, end_silence
        self.seg_id, self.seg_start, self.win_start, self.last_active = 0, None, None, None
        self.last_out = None

    # ------------------------------------------------------------------ input
    def feed(self, samples, t0=None):
        """One emg.raw message: samples (n, 4) raw readings in pin order A0..A3; t0 = monotonic time of the first."""
        x = np.asarray(samples, np.float64).reshape(-1, 4)
        if not len(x):
            return
        y = self.filter(x)
        idx = (self.n + np.arange(len(y))) % self.size
        self.ring[idx] = y
        self.clock = (self.n, time.monotonic() - len(x) / EMG_RATE if t0 is None else float(t0))
        self.n += len(x)

    def time_of(self, i):
        i0, t0 = self.clock
        return t0 + (i - i0) / EMG_RATE

    def _window(self, start):
        start = max(start, self.n - self.size + 1, 0)
        idx = np.arange(start, self.n) % self.size
        return start, self.ring[idx]

    # ------------------------------------------------------------------ calibration
    def calibrate(self, raw, speak_threshold=None):
        """~1 min of raw rig data (samples, 4) with some mouthing: the session gain (Berkeley's spread / this spread
        per slot), then the effort and pace references from the frames SVR-1 calls speech."""
        u = rig_unscaled(raw)
        self.gain = rig_gain(robust_spread(u), self.target)
        x = apply_gain(u, self.gain, KEEP)
        rms, z = [], []
        win = int(4 * FRAME_RATE) * SAMPLES_PER_FRAME
        for s in range(0, len(x) - 2 * SAMPLES_PER_FRAME + 1, win):
            chunk = x[s:s + win]
            chunk = chunk[:len(chunk) // SAMPLES_PER_FRAME * SAMPLES_PER_FRAME]
            out = self._forward(chunk)
            r = frame_rms(chunk)
            speaking = (torch.sigmoid(out['speak_logit'][0]).float().cpu().numpy() > (speak_threshold or self.thr)
                        if 'speak_logit' in out else np.ones(len(r), bool))
            if speaking.mean() > 0.3:
                rms.append(r[speaking])
                z.append(float(out['rate'][0]))
        if rms:
            self.effort_ref = float(np.median(np.concatenate(rms)))
            self.pace_ref = float(np.median(np.exp(RATE_SCALE * np.array(z))))
        return {'gain': self.gain.tolist(), 'effort_ref': self.effort_ref, 'pace_ref': self.pace_ref}

    # ------------------------------------------------------------------ model
    @torch.no_grad()
    def _forward(self, x):
        n = len(x) // SAMPLES_PER_FRAME
        xt = torch.from_numpy(np.ascontiguousarray(x[:n * SAMPLES_PER_FRAME]))[None].to(self.device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=str(self.device).startswith('cuda')):
            return self.model(xt, torch.tensor([n], device=self.device), self.cmask)

    def read(self, camera_active=False):
        """Re-read the current segment (or the idle window) and update the segmentation. None until enough data."""
        if self.gain is None:
            raise RuntimeError('no session gain: load a fine-tuned checkpoint or call calibrate() first')
        if self.clock is None:
            return None
        t_start = time.perf_counter()
        in_seg = self.seg_start is not None
        if in_seg and self.n - self.win_start > self.max_s:
            self.win_start = self.n - self.ctx_s
        start, f1k = self._window(self.win_start if in_seg else self.n - self.idle_s)
        if len(f1k) < 2 * SAMPLES_PER_FRAME * EMG_RATE / RAW_RATE:
            return None
        x = resample_linear(rig_to_slots(f1k), RAW_RATE)
        x = apply_gain(x[:len(x) // SAMPLES_PER_FRAME * SAMPLES_PER_FRAME].astype(np.float32), self.gain, KEEP)
        out = self._forward(x)
        self.last_out = out
        frames = int(out['lengths'][0])
        frame_t = self.time_of(start) + (np.arange(frames) * SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2) / RAW_RATE
        speak = (torch.sigmoid(out['speak_logit'][0]).float().cpu().numpy() if 'speak_logit' in out
                 else np.ones(frames, np.float32))
        rms = frame_rms(x)
        z = float(out['rate'][0])
        now_t = self.time_of(self.n)
        recent = speak[-EFFORT_FRAMES:]
        active = bool(recent.mean() > self.thr) or camera_active
        ended = started = False
        if not in_seg and active:  # speech started: open a segment 300 ms before the run of speaking frames
            run = len(speak)
            while run > 0 and speak[run - 1] > self.thr:
                run -= 1
            t_on = frame_t[run] if run < frames else now_t
            self.seg_id += 1
            self.seg_start = t_on - self.pre_roll
            self.win_start = max(start, self.clock[0] + int(round((self.seg_start - self.clock[1]) * EMG_RATE)))
            self.last_active = now_t
            started = True
        elif in_seg:
            if active:
                self.last_active = now_t
            elif now_t - self.last_active >= self.end_silence:
                ended = True
                self.seg_start = self.win_start = None
        # on the read where speech starts, the segment is reported at once (this read covered the idle window;
        # frame_t says where each frame falls); the next read covers the segment from seg_start
        return {'seg_id': self.seg_id, 'in_segment': (in_seg and not ended) or started, 'segment_started': started,
                'segment_ended': ended, 'seg_start': self.seg_start if (in_seg or started) else None, 'frame_t': frame_t,
                'letter_logp': out['letter_logp'][0].float().cpu().numpy(),
                'phone_logp': out['phone_logp'][0].float().cpu().numpy(),
                'speak_prob': speak, 'rate_z': z, 'pace': float(np.exp(RATE_SCALE * z)) / self.pace_ref,
                'effort': float(rms[-1] / self.effort_ref) if self.effort_ref else None,
                'timing_ms': (time.perf_counter() - t_start) * 1000}

    @torch.no_grad()
    def verify(self, candidates):
        """S_emg of each candidate sentence (letter + phoneme CTC + verifier) on the last read's window."""
        if self.last_out is None:
            raise RuntimeError('read() first')
        cands = [' '.join(clean_text(c).split()) for c in candidates]
        return score_texts(self.model, self.last_out, 0, cands, self.device).float().cpu().numpy()
