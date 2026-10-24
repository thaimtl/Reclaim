"""SVR-1 offline, through exactly the live code path: replay recorded clips into svr.serve.EmgReader (LiveSVR1 plus
its guard rails) as emg.raw messages on the recording's own clock, and log every emg.read it would have sent.

    python -m svr.infer --ckpt CKPT --out LOGDIR [--recordings DIR] [--sessions s1,s2,s3]
    python -m svr.infer --ckpt CKPT --bench                 # ms per read at 2, 5 and 8 s windows

The log is a buslog session (endpoint 'svr', t_pub = the read's sample time on the recorder's bus.now() clock), so
fusion.evaluate --decode --emg LOGDIR joins it to the camera's log of the same clips and scores camera + EMG
against camera alone. Clips play in recording order; the reader resets between clips (the rig streamed between
them live, but those samples weren't recorded). Each session uses its gain from the checkpoint (svr.finetune
stores one per session, a held-out session's from its first minute, like the demo-day calibration); a session
the checkpoint doesn't know is calibrated with LiveSVR1.calibrate on its first CALIB_S of sentence clips.

Also, for the board's API: load(path) is svr.svr1.load with the causal-preprocessing check, and score(live, cands)
is LiveSVR1.verify on the last read.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from svr.live import LiveSVR1
from svr.serve import EmgReader

FS = 1000
CHUNK = 10        # samples per emg.raw message, as rig/publish.py
CALIB_S = 60.0


def load(path, device='cpu'):
    """SVR-1 from a checkpoint trained on the causal preprocessing (rig/filters.py), else ValueError."""
    from svr.svr1 import load as load_model
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('preprocess', 'gaddy') != 'causal':
        raise ValueError(f'{path} was not trained on the causal preprocessing (rig/filters.py)')
    return load_model(path, device)


def score(live, candidates):
    """S_emg of each candidate sentence on the last read's window (letter + phoneme CTC + verifier)."""
    return live.verify(candidates)


def clip_times(r, n):
    """Per-sample bus.now() times of a clip: its emg_ts, else clip_t0 + i / FS."""
    if r.get('emg_ts') and os.path.exists(r['emg_ts']):
        ts = np.load(r['emg_ts']).astype(np.float64)
        if len(ts) == n:
            return ts
    return float(r.get('clip_t0') or 0.0) + np.arange(n) / FS


def replay_clip(reader, raw, ts, counter0=0):
    """Feed one clip as emg.raw messages; yields (t, topic, msg) for what the reader sends, t on the clip's clock."""
    for i in range(0, len(raw), CHUNK):
        rows = raw[i:i + CHUNK]
        t_end = ts[min(i + CHUNK, len(ts)) - 1] + 1 / FS
        reader.on_raw({'t0': float(ts[i]), 'fs': FS, 'samples': rows, 'counter': counter0 + i}, t_end)
        for topic, msg in reader.poll(t_end):
            yield t_end, topic, msg


def session_reader(ckpt, session, records, device):
    live = LiveSVR1(ckpt, device=device, session=session)
    state_gains = torch.load(ckpt, map_location='cpu', weights_only=False).get('session_gains') or {}
    if session not in state_gains:
        raws, total = [], 0
        for r in records:
            if r.get('prompt') and total < CALIB_S * FS:
                x = np.load(r['emg'])
                raws.append(x)
                total += len(x)
        if raws:
            print(f'svr.infer: {session} is not in the checkpoint, calibrating on {total / FS:.0f} s',
                  file=sys.stderr)
            live.calibrate(np.concatenate(raws))
    return EmgReader(live)


def replay_sessions(ckpt, records, out, device=None):
    """Replay every clip of `records` (svr.recordings.load_manifest rows) into a buslog session at `out`."""
    import buslog
    from bus.team import encode

    w = buslog.Writer(out, meta={'source': 'svr.infer', 'ckpt': os.path.abspath(ckpt),
                                 'clips': [r['id'] for r in records]})
    seq, n_reads, n_clips = 0, 0, 0
    try:
        by_session = {}
        for r in records:
            by_session.setdefault(r['session'], []).append(r)
        for session, rs in sorted(by_session.items()):
            rs.sort(key=lambda r: r.get('clip_t0') or 0.0)
            reader = session_reader(ckpt, session, rs, device)
            for r in rs:
                if not r.get('emg') or not os.path.exists(r['emg']):
                    continue
                raw = np.load(r['emg']).reshape(-1, 4)
                reader.reset()
                for t, topic, msg in replay_clip(reader, raw, clip_times(r, len(raw))):
                    seq += 1
                    w.write(t, 'svr', encode(topic, msg, seq=seq, t_pub=t))
                    n_reads += topic == 'emg.read'
                n_clips += 1
    finally:
        w.close()
    return n_clips, n_reads


def bench(ckpt, device=None, windows=(2.0, 5.0, 8.0), reps=20):
    """Median ms per read with the segment window at each length (fp16 on CUDA)."""
    rng = np.random.default_rng(0)
    out = {}
    for win in windows:
        live = LiveSVR1(ckpt, device=device, max_seconds=max(windows) + 1)
        if live.gain is None:
            live.gain = np.ones(8, np.float32)
        raw = np.clip(512 + 40 * rng.standard_normal((int((win + 1.0) * FS), 4)), 0, 1023).astype(np.uint16)
        live.feed(raw[:500], 0.0)
        live.read(camera_active=True)                 # opens a segment
        live.feed(raw[500:], 0.5)
        live.read(camera_active=True)                 # warm-up (cuDNN, autocast)
        ms = []
        for _ in range(reps):
            if str(live.device).startswith('cuda'):
                torch.cuda.synchronize()
            t = time.perf_counter()
            live.read(camera_active=True)
            if str(live.device).startswith('cuda'):
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000)
        out[win] = float(np.median(ms))
        print(f'svr.infer bench: {win:.0f} s window: {out[win]:.1f} ms per read ({live.device})')
    return out


def main(argv=None):
    from svr.recordings import RECORDINGS, load_manifest

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', help='buslog session folder to write (must not exist)')
    ap.add_argument('--recordings', default=RECORDINGS)
    ap.add_argument('--sessions', default=None, help="comma-separated manifest session ids, e.g. s1,s2 (default: all)")
    ap.add_argument('--device', default=None)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args(argv)
    if args.bench:
        return bench(args.ckpt, args.device)
    if not args.out:
        ap.error('--out is required (or --bench)')
    records = load_manifest(args.recordings, sessions=args.sessions.split(',') if args.sessions else None)
    if not records:
        ap.error(f'no clips under {args.recordings}')
    n_clips, n_reads = replay_sessions(args.ckpt, records, args.out, args.device)
    print(f'svr.infer: {n_clips} clips, {n_reads} emg.read -> {args.out}')
    return n_clips, n_reads


if __name__ == '__main__':
    main()
