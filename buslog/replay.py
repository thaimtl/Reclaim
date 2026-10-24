"""Play a logged session back onto the bus with its original timing, so a bug can be reproduced without anyone
mouthing in front of the laptop.

    python -m buslog.replay SESSION [--speed 1.0] [--topics cam. emg.] [--wait 1.0] [--no-retime]

SESSION is a session folder or 'latest'. Replay binds the endpoints of the topics it plays, so it stands in
for those producers: stop them first (bus.Pub refuses an ipc:// path a live publisher holds), then start the
parts under test, which connect as usual. --topics keeps only topics starting with one of the prefixes, e.g.
replay cam. and emg. to test a new decoder live.

The parts compare message times with time.monotonic() (the repair window, sentence end), so by default every
time field listed in TIME_FIELDS is moved onto the replay's clock by one constant offset, replay start minus
the first logged t_recv; seq stays the original. A message that does not decode goes out as it was logged.
At --speed other than 1 the times keep their original spacing, so they drift from the clock as it plays.
--no-retime sends every message byte for byte, times on the recording's clock.
"""
import argparse
import signal
import sys
import time

import numpy as np

from bus import ENDPOINTS, Pub, decode, encode
from buslog import matches, read_raw, topic_of

SNDHWM = 10000  # a fast replay must not drop at the default 100
# topic prefix -> fields holding time.monotonic() seconds (numbers or arrays), from the Interfaces table and
# fusion/decoder.py; 'words[].t' is the t of every item in the list words. t_pub is in every message.
TIME_FIELDS = {
    '': ('t_pub',),
    'emg.raw': ('t0',),
    'cam.tick': ('frame_t', 'seg_start'),
    'cam.face': ('t',),
    'dec.lock': ('words[].t',),
    'claude.': ('t',),
    'partner.text': ('t',),
    'timing.': ('t',),
}


def _shift(d, path, offset):
    head, _, rest = path.partition('.')
    if head.endswith('[]'):
        for item in d.get(head[:-2]) or []:
            if isinstance(item, dict):
                _shift(item, rest, offset)
        return
    v = d.get(head)
    if isinstance(v, np.ndarray):
        d[head] = (v + offset).astype(v.dtype)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        d[head] = v + offset


def retime(frames, offset):
    """The message with every TIME_FIELDS entry shifted by offset and its seq kept; unchanged if it does not decode."""
    try:
        topic, msg = decode(frames)
    except ValueError:
        return frames
    for prefix, fields in TIME_FIELDS.items():
        if topic.startswith(prefix):
            for f in fields:
                _shift(msg, f, offset)
    return encode(topic, msg, seq=msg.get('seq', 0), t_pub=msg.get('t_pub'))  # no t_pub: stamped now


def replay(session, speed=1.0, topics=None, endpoints=None, ctx=None, wait_s=1.0, retime_times=True,
           stop=lambda: False):
    """Re-publish the session; returns the number of messages sent."""
    endpoints = dict(endpoints or ENDPOINTS)
    # a first pass finds the endpoints to bind, the second streams, so a long session never sits in memory
    names = {name for _, name, frames in read_raw(session) if matches(topic_of(frames), topics)}
    pubs = {name: Pub(endpoints[name], ctx=ctx, sndhwm=SNDHWM) for name in names if name in endpoints}
    if names - set(pubs):
        print(f'replay: no endpoint for {sorted(names - set(pubs))}, skipping those', file=sys.stderr)
    time.sleep(wait_s)  # let subscribers connect, or the first messages are lost
    count, start, first = 0, time.monotonic(), None
    try:
        for t_recv, name, frames in read_raw(session):
            if stop():
                break
            if name not in pubs or not matches(topic_of(frames), topics):
                continue
            first = t_recv if first is None else first
            delay = start + (t_recv - first) / speed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            pubs[name].sock.send_multipart(retime(frames, start - first) if retime_times else frames)
            count += 1
    finally:
        for pub in pubs.values():
            pub.close()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', help="session folder, or 'latest'")
    ap.add_argument('--speed', type=float, default=1.0, help='2 plays twice as fast')
    ap.add_argument('--topics', nargs='+', help='topic prefixes to play (default: all)')
    ap.add_argument('--wait', type=float, default=1.0, help='seconds to wait for subscribers before playing')
    ap.add_argument('--no-retime', action='store_true', help='keep the recorded times, send exact bytes')
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    try:
        n = replay(args.session, args.speed, args.topics, wait_s=args.wait, retime_times=not args.no_retime)
        print(f'replayed {n} messages', file=sys.stderr)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
