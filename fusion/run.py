"""The fusion process on the bus: camera + EMG + Claude -> dec.hyp, dec.lock, dec.repeat, voice.say, timing.dec.

    venv-svr/bin/python -m fusion.run                       # SVR-1 W_s12_svr1_4ch, fitted gate and lock model
    venv-svr/bin/python -m fusion.run --calibrate 60        # first 60 s of emg.raw (Austin mouthing) refit the gains
    venv-svr/bin/python -m fusion.run --no-emg              # camera + LM + Claude only (pads off)

Connects to 'cam' (cam.tick, cam.final), 'emg' (emg.raw), 'claude' (claude.predict, claude.repair) and 'call'
(partner.text); binds 'dec' (bus/team.py). The logic is fusion/core.py; this file only moves messages. Messages are
drained before each pass, and a cam.tick with a newer one of the same segment already waiting is skipped, so a slow
pass never builds a backlog: the next pass reads the newest frames.

Needs, under RECLAIM_WORKDIR: checkpoints/W_s12_svr1_4ch/model.pt, data/lm/conversational.binary,
data/lexicon/cmudict.dict (look-alikes), data/fusion/params.json (python -m fusion.offline fit-gate / fit-lock;
without it the unfitted defaults run), and the camera's units file and SentencePiece model in third_party/.
"""
import argparse
import os
import signal
import sys
import time

import numpy as np

import bus
from bus.team import Pub, Sub
from fusion.offline import PARAMS, WORK, build_search, load_params

DEFAULT_SVR = os.path.join(WORK, 'checkpoints', 'W_s12_svr1_4ch', 'model.pt')
TOPICS = ['cam.tick', 'cam.final', 'emg.raw', 'claude.predict', 'claude.repair', 'partner.text']
STEP_S = 0.02


def drain(sub, first):
    """`first` plus everything already queued, with stale cam.ticks (a newer one of the same segment waiting) out."""
    msgs = [first]
    while True:
        topic, msg = sub.recv(timeout_ms=0)
        if topic is None:
            break
        msgs.append((topic, msg))
    newest = {}
    for k, (topic, msg) in enumerate(msgs):
        if topic == 'cam.tick':
            newest[int(msg['seg_id'])] = k
    return [(t, m) for k, (t, m) in enumerate(msgs) if t != 'cam.tick' or newest[int(m['seg_id'])] == k]


def calibrate(live, sub, seconds):
    print(f'fusion: calibrating on {seconds:.0f} s of emg.raw: mouth a few sentences now', file=sys.stderr)
    raw, t_end = [], bus.now() + seconds
    while bus.now() < t_end:
        topic, msg = sub.recv(timeout_ms=100)
        if topic == 'emg.raw':
            raw.append(np.asarray(msg['samples']).reshape(-1, 4))
    if not raw:
        sys.exit('fusion: no emg.raw arrived: is rig/publish.py running?')
    print(f'fusion: calibration {live.calibrate(np.concatenate(raw))}', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--svr', default=DEFAULT_SVR, help='SVR-1 checkpoint')
    ap.add_argument('--params', default=PARAMS, help='fitted gate and lock model (fusion.offline)')
    ap.add_argument('--repair-s', type=float, default=0.3, help="repair window before a locked word is spoken")
    ap.add_argument('--no-emg', action='store_true', help='camera + LM + Claude only')
    ap.add_argument('--no-claude', action='store_true', help="ignore Claude's predictions and repairs")
    ap.add_argument('--calibrate', type=float, default=0.0, metavar='S', help='refit the EMG gains on the first S s')
    ap.add_argument('--sub', action='append', default=[], help='endpoints to connect to (default cam emg claude call)')
    ap.add_argument('--pub', default='dec')
    args = ap.parse_args()

    from fusion.core import Fusion
    gate, lock, meta = load_params(args.params)
    if not meta:
        print(f'fusion: {args.params} missing: unfitted gate and lock model', file=sys.stderr)
    live = None
    if not args.no_emg:
        from svr.live import LiveSVR1
        live = LiveSVR1(args.svr)
    search = build_search()
    topics = [t for t in TOPICS if not (args.no_claude and t.startswith('claude.'))]
    sub = Sub(args.sub or ['cam', 'emg', 'claude', 'call'], topics, rcvhwm=2000)
    pub = Pub(args.pub)
    if live is not None and args.calibrate > 0:
        calibrate(live, sub, args.calibrate)
    core = Fusion(search, gate, lock, svr=live, repair_s=args.repair_s, use_hints=not args.no_claude)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(f'fusion: running (EMG {"off" if live is None else os.path.basename(os.path.dirname(args.svr))}, '
          f'repair window {args.repair_s} s)', file=sys.stderr)
    next_step = bus.now()
    try:
        while True:
            topic, msg = sub.recv(timeout_ms=int(STEP_S * 1000))
            out = []
            if topic is not None:
                for t, m in drain(sub, (topic, msg)):
                    try:
                        out += core.feed(t, m, bus.now())
                    except Exception as e:  # noqa: BLE001  one bad message never stops the captions
                        print(f'fusion: {t}: {type(e).__name__}: {e}', file=sys.stderr)
            now = bus.now()
            if now >= next_step:
                out += core.step(now)
                next_step = now + STEP_S
            for t, m in out:
                pub.send(t, m)
                if t == 'dec.lock':
                    print('locked:', ' '.join(x['w'] for x in m['words']), file=sys.stderr)
                elif t == 'dec.repeat':
                    print('please repeat:', m['reason'].get('reader'), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        time.sleep(0.1)


if __name__ == '__main__':
    main()
