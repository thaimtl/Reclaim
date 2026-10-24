"""Record every topic on every endpoint to a new session folder, until Ctrl-C.

    python -m buslog.record [--out FOLDER] [--rotate-mb 256]

One SUB socket per endpoint, so each record knows which endpoint it came from (replay binds the same one).
Producers never wait on us: PUB/SUB drops for a slow subscriber instead of blocking, and ZeroMQ's own IO thread
queues up to RCVHWM messages per endpoint while we write: about 80 s of cam traffic (cam.tick and cam.face)
and 100 s of emg.raw, so if the disk stalls the worst case held in memory is about 110 MB of cam.tick (55 KB
each) and a few MB for the rest. Frames are written as
they arrive, without decoding (a malformed message is logged too), and flushed to the OS every FLUSH_S, so a
crash of this process loses at most the last FLUSH_S (a crash of the laptop may lose what the OS had not yet
written).
"""
import argparse
import os
import signal
import sys
import time

import zmq

from bus import ENDPOINTS, Sub
from buslog import ROTATE_BYTES, Writer, log_root

FLUSH_S = 0.2
RCVHWM = {'cam': 2000}  # messages queued per endpoint: about 80 s of cam.tick + cam.face (~22/s)
RCVHWM_DEFAULT = 10000  # 100 s of emg.raw at 100 Hz


def session_folder():
    """logs/<date-time>-<ms>-<pid>: sorts by start time, and two loggers started together never share one."""
    t = time.time()
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(t))
    return log_root() / f'{stamp}-{int(t * 1000) % 1000:03d}-{os.getpid()}'


def record(folder, endpoints=None, ctx=None, stop=lambda: False, rotate_bytes=ROTATE_BYTES):
    """Log until stop() is true; returns the number of messages written."""
    endpoints = dict(endpoints or ENDPOINTS)
    writer = Writer(folder, {'endpoints': endpoints}, rotate_bytes)
    subs = {name: Sub([addr], [''], ctx=ctx, rcvhwm=RCVHWM.get(name, RCVHWM_DEFAULT)) for name, addr in endpoints.items()}
    poller = zmq.Poller()
    for sub in subs.values():
        poller.register(sub.sock, zmq.POLLIN)
    name_of = {sub.sock: name for name, sub in subs.items()}
    count, flushed = 0, time.monotonic()
    try:
        while not stop():
            for sock, _ in poller.poll(50):
                while True:  # drain this endpoint
                    try:
                        frames = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    writer.write(time.monotonic(), name_of[sock], frames)
                    count += 1
            if time.monotonic() - flushed >= FLUSH_S:
                writer.flush()
                flushed = time.monotonic()
    finally:
        writer.close()
        for sub in subs.values():
            sub.close()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', help='session folder (default: $RECLAIM_WORKDIR/logs/<timestamp>)')
    ap.add_argument('--rotate-mb', type=float, default=ROTATE_BYTES / 2**20)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    folder = args.out or session_folder()
    print(f'logging to {folder}', file=sys.stderr)
    try:
        record(folder, rotate_bytes=int(args.rotate_mb * 2**20))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
