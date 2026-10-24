"""Publish the rig on the bus: emg.raw every 10 ms on endpoint 'emg', raw and unfiltered (consumers run
rig/filters.py in order, the same as on the recordings).

    python -m rig.publish PORT        # e.g. COM5 or /dev/ttyACM0

Each message is one run of consecutive samples: t0 (time.monotonic() of the first), fs = 1000, samples
(n x 4 uint16, 0 to 1023, pins A0 to A3) and counter (running index of the first sample, dropped frames
included). rig/reader.py reports drops as a running total, so a gap shows as counter + n of one message falling
short of the next one's counter, and a message never spans a gap: sample i is always at t0 + i / fs. The reader
does not say where in a chunk frames went missing, so the gap is put at the start of that chunk.

Timestamps come from the sample clock, not from when a read returned: USB hands over samples in bursts a few
ms late, so each sample's time is offset + index / fs, with offset the lower envelope of t_read - index / fs
(the read with the least delay). The envelope rises by LEAK s per s, so it follows the Uno's resonator if that
runs slow (up to 0.5%); if it runs fast the envelope follows on its own.
"""
import argparse
import signal
import time

import numpy as np

from bus import Pub

FS = 1000       # rig/uno samples every 1000 us
BATCH = 10      # samples per message: 10 ms at 1 kHz
LEAK = 0.005    # s per s the clock offset may rise, covering a 0.5% slow resonator


def batches(chunks, fs=FS, size=BATCH, leak=LEAK):
    """(t_read, rows, dropped) chunks -> emg.raw messages of at most `size` consecutive samples. rows is n x 4,
    t_read the time.monotonic() right after it was read, dropped the running total of frames lost so far."""
    seen = lost = 0                    # samples received, frames lost: the next sample's index is seen + lost
    offset = t_prev = None
    buf, start = [], 0                 # samples waiting to go out, and the index of the first
    for t_read, rows, dropped in chunks:
        if not len(rows):
            continue
        if dropped > lost and buf:     # a gap: what came before it goes out on its own
            yield _msg(offset, start, buf, fs)
            buf = []
        lost = max(lost, dropped)
        first = seen + lost
        seen += len(rows)
        floor = t_read - (first + len(rows) - 1) / fs
        offset = floor if offset is None else min(offset + leak * (t_read - t_prev), floor)
        t_prev = t_read
        for k, row in enumerate(rows):
            if not buf:
                start = first + k
            buf.append(row)
            if len(buf) == size:
                yield _msg(offset, start, buf, fs)
                buf = []
    if buf:
        yield _msg(offset, start, buf, fs)


def _msg(offset, start, rows, fs):
    return {'t0': offset + start / fs, 'fs': fs, 'samples': np.array(rows, dtype=np.uint16), 'counter': start}


def stamped(port, stats):
    """The rig's chunks as (t_read, rows, dropped)."""
    from rig.reader import frames  # here, so batches() is testable without pyserial

    for rows in frames(port, stats):
        yield time.monotonic(), rows, stats.get('dropped', 0)


def main():
    ap = argparse.ArgumentParser(description='Publish the rig as emg.raw on the bus (endpoint emg, RECLAIM_EMG_PUB).')
    ap.add_argument('port', help='the Uno\'s serial port, e.g. COM5 or /dev/ttyACM0')
    args = ap.parse_args()
    pub, stats = Pub('emg'), {}
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    try:
        for msg in batches(stamped(args.port, stats)):
            pub.send('emg.raw', msg)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()


if __name__ == '__main__':
    main()
