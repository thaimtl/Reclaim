"""The error log: every bus message on disk, so any session can be read back offline or replayed.

    python -m buslog.record                       # until Ctrl-C, into logs/<timestamp>/
    python -m buslog.replay latest --speed 1.0    # re-publish it, see buslog/replay.py

    for t_recv, topic, msg in buslog.read('latest', topics=['dec.lock']):
        ...

A session is a folder under $RECLAIM_WORKDIR/logs/ (./logs without it): meta.json (endpoints, start time) and
bus-0000.msgpack, bus-0001.msgpack, ... rotated by size. Each file is a plain stream of msgpack arrays
[t_recv, endpoint, frames]: t_recv is bus.now() when the logger got the message, endpoint the ENDPOINTS
name it came from, frames the raw ZeroMQ frames ([topic, payload] for a good message), never decoded, so a
malformed message is kept as it was and replay sends the same bytes, seq and t_pub included. Appending needs no
index, and a crash of the recorder only cuts the last record short, which the reader skips with a warning.
"""
import json
import os
import sys
import time
from pathlib import Path

import msgpack

import bus
from bus.team import decode

ROTATE_BYTES = 256 << 20


def log_root():
    work = os.environ.get('RECLAIM_WORKDIR')
    return Path(work) / 'logs' if work else Path('logs')


def resolve(session):
    """A session folder, or 'latest' for the newest one under log_root()."""
    if str(session) != 'latest':
        return Path(session)
    folders = sorted(p for p in log_root().iterdir() if (p / 'meta.json').exists())
    if not folders:
        raise FileNotFoundError(f'no sessions in {log_root()}')
    return folders[-1]


class Writer:
    """Appends records to a session folder, rotating files at rotate_bytes. Not thread-safe; one writer per session."""

    def __init__(self, folder, meta=None, rotate_bytes=ROTATE_BYTES):
        self.folder, self.rotate_bytes, self.index, self.f = Path(folder), rotate_bytes, -1, None
        self.folder.mkdir(parents=True, exist_ok=False)  # never append to someone else's session
        meta = dict(meta or {}, t_now=bus.now(), t_wall=time.time())
        (self.folder / 'meta.json').write_text(json.dumps(meta, indent=2))
        self._next_file()

    def _next_file(self):
        if self.f:
            self.f.close()
        self.index += 1
        self.f, self.size = open(self.folder / f'bus-{self.index:04d}.msgpack', 'ab'), 0

    def write(self, t_recv, endpoint, frames):
        rec = msgpack.packb([t_recv, endpoint, [bytes(f) for f in frames]], use_bin_type=True)
        self.f.write(rec)
        self.size += len(rec)
        if self.size >= self.rotate_bytes:
            self._next_file()

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def read_raw(session):
    """Yield (t_recv, endpoint, frames) in the order they were logged; frames[0] is the topic. A record that is
    not [t_recv, endpoint, frames] is skipped; a file that stops decoding (cut short, corrupt) ends there. Both
    are counted and reported on stderr."""
    for path in sorted(resolve(session).glob('bus-*.msgpack')):
        skipped, broken = 0, False
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            try:
                for rec in unpacker:
                    if (isinstance(rec, list) and len(rec) == 3 and isinstance(rec[0], (int, float))
                            and isinstance(rec[2], list) and rec[2] and all(isinstance(x, bytes) for x in rec[2])):
                        yield rec[0], rec[1], rec[2]
                    else:
                        skipped += 1
            except (ValueError, msgpack.UnpackException) as e:  # msgpack cannot resync, so the rest is lost
                print(f'buslog: {path.name}: unreadable from byte {unpacker.tell()}: {e}', file=sys.stderr)
                broken = True
            tail = os.path.getsize(path) - unpacker.tell()
        if tail > 0 and not broken:  # the file ends inside its last record
            print(f'buslog: {path.name}: last {tail} bytes cut short', file=sys.stderr)
        if skipped:
            print(f'buslog: {path.name}: skipped {skipped} malformed records', file=sys.stderr)


def matches(topic, prefixes):
    """ZeroMQ's subscription rule: a topic matches when it starts with one of the prefixes (None: everything)."""
    return prefixes is None or any(topic.startswith(p) for p in prefixes)


def topic_of(frames):
    return frames[0].decode(errors='replace')


def read(session, topics=None):
    """Yield (t_recv, topic, msg) with msg decoded as on the bus (numpy arrays restored); skips malformed ones."""
    for t_recv, _, frames in read_raw(session):
        if not matches(topic_of(frames), topics):
            continue
        try:
            yield (t_recv, *decode(frames, validate=False))
        except ValueError:
            continue
