"""The live EMG reader on the bus: emg.raw (and the camera's mouth activity from cam.tick) in, emg.read and
timing.svr out (endpoint 'svr'). SVR-1 itself is svr/live.py's LiveSVR1; this is the bus side and the guard rails.

    python -m svr.serve --ckpt $RECLAIM_WORKDIR/checkpoints/W_svr1/model.pt [--calibrate 60] [--gains PATH]

Its own process, so torch and the GPU stay out of the decoder: a CUDA error or a pulled Uno can't stop the captions,
and fusion.run only ever sees emg.read, which the logger keeps (fusion.evaluate --decode replays camera + EMG with no
GPU). One thread: drain what has arrived, feed it to LiveSVR1, re-read every 100 ms.

Guard rails, all in EmgReader (svr/infer.py replays recordings through the same class):
- A counter gap of up to GAP_FILL samples is filled by repeating the last raw row, so the filter and the sample
  clock stay continuous. A longer gap, a counter that goes back (rig/publish.py restarted), or no emg.raw for
  STALE_S resets LiveSVR1 and sends ok = False until fresh samples arrive.
- Quality over the last QUALITY_S, per channel: the share of samples pinned at the rails (rig/signal_report.py's
  "clipped": <= 2 or >= 1021) and whether the channel is flat (robust spread under FLAT_SPREAD ADC counts: a loose
  cable). ok is False when more than MAX_BAD channels are bad (noisy channels hurt more than missing ones).
- An exception in a read is reported on timing.svr (stage svr.error) and the loop goes on.

emg.read: seg_id, in_segment, segment_started, segment_ended, seg_start (None outside a segment), frame_t (<f8, F),
letter_logp (<f2, F x 38), speak_prob (<f4, F), rate_z, pace, effort (None until an effort reference exists),
quality {pinned, flat} (4 each, pin order A0..A3), ok, gap_ms (samples filled since the last read, in ms),
timing_ms. With ok False only seg_id, ok, gap_ms and quality are meaningful, and the arrays are empty.

--calibrate S: the first S seconds of emg.raw (pads on, Austin mouthing a few sentences) go to LiveSVR1.calibrate;
the result is saved to $RECLAIM_WORKDIR/calib/<timestamp>.json, and --gains PATH loads it after a restart.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import deque

import numpy as np

import bus

FS = 1000
READ_EVERY_S = 0.1
STALE_S = 0.3        # no emg.raw for this long: the rig is gone
GAP_FILL = 50        # samples (50 ms): a gap up to this is filled, a longer one resets
QUALITY_S = 0.5
PIN_LO, PIN_HI = 2, 1021
PINNED_MAX = 0.2     # share of pinned samples that makes a channel bad
FLAT_SPREAD = 0.5    # ADC counts
MAX_BAD = 1
CAM_FRESH_S = 0.3
CAM_ACTIVE = 0.5     # the camera's mouth activity over its last 3 frames above this counts as mouthing


def quality(raw):
    """(n, 4) raw readings -> per-channel pinned share and flat flag, and whether the window is usable."""
    raw = np.asarray(raw, np.float64)
    if not len(raw):
        return {'pinned': [0.0] * 4, 'flat': [True] * 4}, False
    pinned = ((raw <= PIN_LO) | (raw >= PIN_HI)).mean(0)
    spread = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0)
    flat = spread < FLAT_SPREAD
    bad = int(((pinned > PINNED_MAX) | flat).sum())
    return {'pinned': [round(float(p), 4) for p in pinned], 'flat': [bool(f) for f in flat]}, bad <= MAX_BAD


def to_msg(out, q, ok, gap_ms):
    """LiveSVR1.read() output -> the emg.read message."""
    return {'seg_id': int(out['seg_id']), 'in_segment': bool(out['in_segment']),
            'segment_started': bool(out['segment_started']), 'segment_ended': bool(out['segment_ended']),
            'seg_start': None if out['seg_start'] is None else float(out['seg_start']),
            'frame_t': np.asarray(out['frame_t'], np.float64),
            'letter_logp': np.asarray(out['letter_logp'], np.float16),
            'speak_prob': np.asarray(out['speak_prob'], np.float32),
            'rate_z': float(out['rate_z']), 'pace': float(out['pace']),
            'effort': None if out['effort'] is None else float(out['effort']),
            'quality': q, 'ok': bool(ok), 'gap_ms': float(gap_ms), 'timing_ms': float(out['timing_ms'])}


def not_ok(seg_id, q, gap_ms):
    empty = np.zeros(0)
    return {'seg_id': int(seg_id), 'in_segment': False, 'segment_started': False, 'segment_ended': False,
            'seg_start': None, 'frame_t': empty.astype(np.float64), 'letter_logp': np.zeros((0, 38), np.float16),
            'speak_prob': empty.astype(np.float32), 'rate_z': 0.0, 'pace': 1.0, 'effort': None, 'quality': q,
            'ok': False, 'gap_ms': float(gap_ms), 'timing_ms': 0.0}


class EmgReader:
    """LiveSVR1 plus the guard rails. on_raw / on_tick per message, poll(now) every loop: returns (topic, msg) to send.
    `now` is bus.now() live and the recording's sample time in svr/infer.py."""

    def __init__(self, live, read_every=READ_EVERY_S):
        self.live, self.read_every = live, read_every
        self.next_counter = self.last_row = self.last_raw_t = None
        self.last_read = -np.inf
        self.recent = deque(maxlen=int(QUALITY_S * FS))   # raw rows, for quality()
        self.gap_samples = 0
        self.fresh = False                                 # samples arrived since the last reset
        self.stale_sent = False
        self.cam = (None, False)                           # (time of the newest cam.tick, mouthing)

    def reset(self):
        self.live.reset()
        self.next_counter = self.last_row = None
        self.recent.clear()
        self.fresh = False

    def on_raw(self, msg, now):
        x = np.asarray(msg['samples']).reshape(-1, 4)
        if not len(x):
            return
        counter, t0 = int(msg['counter']), float(msg['t0'])
        if self.next_counter is not None:
            gap = counter - self.next_counter
            if gap < 0 or gap > GAP_FILL:   # the publisher restarted, or too much is missing to bridge
                self.reset()
            elif gap:
                fill = np.repeat(self.last_row[None], gap, 0)
                self.live.feed(fill, t0 - gap / FS)
                self.recent.extend(fill)
                self.gap_samples += gap
        self.live.feed(x, t0)
        self.recent.extend(x)
        self.next_counter, self.last_row, self.last_raw_t = counter + len(x), x[-1].copy(), now
        self.fresh, self.stale_sent = True, False

    def on_tick(self, msg, now):
        act = np.asarray(msg.get('activity', ()), np.float64)
        self.cam = (now, bool(len(act) and act[-3:].mean() > CAM_ACTIVE))

    def poll(self, now):
        out = []
        if self.fresh and now - self.last_raw_t > STALE_S:
            self.reset()
        if not self.fresh:
            if not self.stale_sent:  # once per outage, so the decoder drops EMG at once
                q, _ = quality(())
                out.append(('emg.read', not_ok(self.live.seg_id, q, 0.0)))
                self.stale_sent = True
            return out
        if now - self.last_read < self.read_every:
            return out
        self.last_read = now
        q, ok = quality(np.array(self.recent))
        cam_t, mouthing = self.cam
        r = self.live.read(camera_active=bool(mouthing and cam_t is not None and now - cam_t < CAM_FRESH_S))
        gap_ms, self.gap_samples = self.gap_samples * 1000 / FS, 0
        if r is not None:
            out.append(('emg.read', to_msg(r, q, ok, gap_ms)))
        return out


def drain(sub, poll_ms, bad):
    msgs, timeout = [], poll_ms
    while True:
        try:
            topic, msg = sub.recv(timeout_ms=timeout)
        except ValueError as e:
            bad.append(str(e))
            timeout = 0
            continue
        if topic is None:
            return msgs
        msgs.append((topic, msg))
        timeout = 0


def serve(reader, sub, pub, stop=lambda: False, poll_ms=5):
    errors = 0
    while not stop():
        bad = []
        for topic, msg in drain(sub, poll_ms, bad):
            try:
                if topic == 'emg.raw':
                    reader.on_raw(msg, bus.now())
                elif topic == 'cam.tick':
                    reader.on_tick(msg, bus.now())
            except Exception as e:  # a message missing or mistyping fields: count it, keep going
                bad.append(f'{topic}: {type(e).__name__}: {e}')
        try:
            out = reader.poll(bus.now())
        except Exception as e:  # a failed read (CUDA, a bug): report, keep reading
            traceback.print_exc()
            errors += 1
            out = [('timing.svr', {'stage': 'svr.error', 'ms': 0.0, 'error': f'{type(e).__name__}: {e}',
                                   'total': errors})]
        if bad:
            print(f'svr.serve: {len(bad)} bad message(s): {bad[-1]}', file=sys.stderr)
            out.append(('timing.svr', {'stage': 'svr.bad_msg', 'ms': 0.0, 'count': len(bad), 'errors': bad[:10]}))
        for topic, msg in out:
            pub.send(topic, msg)


def calibrate(live, sub, seconds, out_dir):
    """Collect `seconds` of emg.raw, refit the gains and references, save them. Returns the saved path."""
    rows, need = [], int(seconds * FS)
    print(f'svr.serve: calibrating, mouth a few sentences for {seconds:.0f} s', file=sys.stderr)
    while sum(map(len, rows)) < need:
        topic, msg = sub.recv(timeout_ms=1000)
        if topic == 'emg.raw':
            rows.append(np.asarray(msg['samples']).reshape(-1, 4))
    res = live.calibrate(np.concatenate(rows)[:need])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime('%Y%m%d-%H%M%S') + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'svr.serve: calibration saved to {path}: {res}', file=sys.stderr)
    return path


def load_gains(live, path):
    with open(path) as f:
        res = json.load(f)
    live.gain = np.asarray(res['gain'], np.float32)
    live.effort_ref = res.get('effort_ref') or live.effort_ref
    live.pace_ref = res.get('pace_ref') or live.pace_ref


def main(argv=None):
    from bus.team import Pub, Sub
    from svr.live import LiveSVR1

    work = os.environ.get('RECLAIM_WORKDIR', '.')
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ckpt', default=os.path.join(work, 'checkpoints', 'W_svr1', 'model.pt'))
    ap.add_argument('--device', default=None, help='default: cuda if available')
    ap.add_argument('--session', default=None, help="the checkpoint's session gains to start from (default: its last)")
    ap.add_argument('--calibrate', type=float, default=0.0, metavar='S', help='calibrate on the first S s of emg.raw')
    ap.add_argument('--gains', default=None, help='a calibration JSON saved by --calibrate')
    ap.add_argument('--calib-dir', default=os.path.join(work, 'calib'))
    args = ap.parse_args(argv)
    live = LiveSVR1(args.ckpt, device=args.device, session=args.session)
    if args.gains:
        load_gains(live, args.gains)
    if live.gain is None and not args.calibrate:
        ap.error(f'{args.ckpt} has no session gains (not fine-tuned): pass --calibrate 60 or --gains PATH')
    sub = Sub(['emg', 'cam'], ['emg.raw', 'cam.tick'], rcvhwm=2000)
    pub = Pub('svr')
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    try:
        if args.calibrate:
            calibrate(live, sub, args.calibrate, args.calib_dir)
        print(f'svr.serve: ready on {live.device}, publishing emg.read', file=sys.stderr, flush=True)
        serve(EmgReader(live), sub, pub)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()


if __name__ == '__main__':
    main()
