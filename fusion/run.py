"""The fusion decoder on the bus: cam.tick, cam.final, claude.predict, claude.repair in; dec.hyp, dec.lock,
dec.repeat, voice.say and timing.dec out (endpoint 'dec').

One thread, one socket each way. Every loop drains what has arrived, keeps only the newest cam.tick per seg_id
(each tick re-reads the whole segment), hands messages to the Decoder in arrival order, then pumps it so the
repair window and sentence end advance even when nothing arrives. A malformed message is counted, logged on
timing.dec (stage dec.bad_msg) and skipped, and so is an exception from the decoder: the loop never dies on one.
SIGTERM stops it like Ctrl-C.

    RECLAIM_WORKDIR=... python -m fusion.run [--units PATH] [--window 0.3]
"""
import argparse
import signal
import sys
import traceback

import bus
from bus.team import Pub, Sub
from fusion.decoder import REPAIR_WINDOW_S, Decoder
from fusion.words import SPM, UNITS, load_lookalikes, load_units

SOURCES = ('cam', 'claude')
TOPICS = ('cam.tick', 'cam.final', 'claude.predict', 'claude.repair')
POLL_MS = 10


def drain(sub, poll_ms, bad):
    """Everything waiting now (after up to poll_ms for the first); malformed messages go into `bad`."""
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


def newest_ticks(msgs):
    """Drop every cam.tick that a later tick of the same seg_id replaces."""
    last = {m['seg_id']: i for i, (t, m) in enumerate(msgs) if t == 'cam.tick' and 'seg_id' in m}
    return [(t, m) for i, (t, m) in enumerate(msgs) if t != 'cam.tick' or last.get(m.get('seg_id')) == i]


def handle(dec, topic, msg, now):
    if topic == 'cam.tick':
        return dec.on_tick(msg, now)
    if topic == 'cam.final':
        return dec.on_final(msg, now)
    if topic == 'claude.predict':
        return dec.on_predict(msg, now)
    if topic == 'claude.repair':
        return dec.on_repair(msg, now)
    return []


def serve(dec, sub, pub, stop=lambda: False, poll_ms=POLL_MS):
    bad_total = 0
    while not stop():
        bad = []
        msgs = newest_ticks(drain(sub, poll_ms, bad))
        now = bus.now()
        out = []
        for topic, msg in msgs:
            try:
                out += handle(dec, topic, msg, now)
            except Exception as e:  # a message missing or mistyping fields, or a decoder bug: log, keep going
                traceback.print_exc()
                bad.append(f'{topic}: {type(e).__name__}: {e}')
        try:
            out += dec.pump(now)
        except Exception as e:
            traceback.print_exc()
            bad.append(f'pump: {type(e).__name__}: {e}')
        if bad:
            bad_total += len(bad)
            print(f'fusion: {len(bad)} bad message(s): {bad[-1]}', file=sys.stderr)
            out.append(('timing.dec', {'stage': 'dec.bad_msg', 'ms': 0.0, 'count': len(bad), 'total': bad_total,
                                       'errors': bad[:10]}))
        for topic, msg in out:
            pub.send(topic, msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--units', default=UNITS, help='the camera\'s unigram5000_units.txt (line number = token id); '
                    'found through $RECLAIM_WORKDIR (default %(default)s)')
    ap.add_argument('--spm', default=SPM, help='the camera\'s unigram5000.model, for look-alike word frequencies (default %(default)s)')
    ap.add_argument('--window', type=float, default=REPAIR_WINDOW_S, help='repair window in seconds')
    ap.add_argument('--no-lookalikes', action='store_true', help='no lip look-alikes in the alts')
    args = ap.parse_args()
    units = load_units(args.units)
    dec = Decoder(units, None if args.no_lookalikes else load_lookalikes(units, args.spm),
                  repair_window_s=args.window)
    sub, pub = Sub(SOURCES, TOPICS), Pub('dec')
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    try:
        serve(dec, sub, pub)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()


if __name__ == '__main__':
    main()
