"""The rest of the team bus: every topic besides the camera's, on the same wire format as bus/__init__.py.

One endpoint per producer on the laptop's loopback (TCP, because ipc:// doesn't work on Windows), each overridable by
RECLAIM_<NAME>_PUB. Each producer binds its endpoint (a second one on the same port fails with EADDRINUSE);
consumers connect to the ones they need and name their topics. Messages carry v, seq (per topic from 1) and t_pub
(bus.now() at send). cam.tick / cam.final / cam.face are checked with bus.validate, exactly as bus.Subscriber does
(decode(..., validate=False) skips it, for logs).

ZeroMQ sockets are not thread-safe: use one Pub or Sub per thread and never share one. Decoded arrays are
writeable copies.

    pub = Pub('dec')                          # binds ENDPOINTS['dec']
    pub.send('dec.lock', {'utt_id': 3, 'words': [...]})
    sub = Sub(['cam', 'claude'], ['cam.tick', 'cam.final', 'claude.'])
    topic, msg = sub.recv(timeout_ms=100)     # (None, None) on timeout; ValueError on a malformed message
"""
import os

import msgpack
import numpy as np
import zmq

import bus
from bus import now, pack_nd

VERSION = bus.VERSION
ACCEPTED = (1, 2)  # v1: the same envelope for plain maps; the call service stamped it until 2026-10-24, so recorded
                  # sessions hold it and replay sends it as logged (cam.* messages are checked by their schema)
CAM_TOPICS = (bus.TICK, bus.FINAL, bus.FACE)
_DEFAULTS = {
    'emg': 'tcp://127.0.0.1:5555',      # emg.raw (rig/publish.py)
    'cam': bus.DEFAULT_CAM_PUB,         # cam.tick, cam.final, cam.face (lipread/run.py), 5556
    'dec': 'tcp://127.0.0.1:5557',      # dec.hyp, dec.lock, dec.repeat, voice.say, timing.dec (Austin's decoder)
    'claude': 'tcp://127.0.0.1:5558',   # claude.predict, claude.repair, timing.claude (claude_service)
    'call': 'tcp://127.0.0.1:5559',     # partner.text, timing.call (call/)
    'svr': 'tcp://127.0.0.1:5560',      # emg.read, timing.svr (Austin's live EMG reader)
}
ENDPOINTS = {k: os.environ.get(f'RECLAIM_{k.upper()}_PUB', v) for k, v in _DEFAULTS.items()}


def unpack_nd(d):
    return bus.unpack_nd(d).copy()


_is_nd = bus._is_nd  # one definition of the array-map format


def _out(v):
    if isinstance(v, np.ndarray):
        return pack_nd(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _out(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_out(x) for x in v]
    return v


def _in(v):
    if _is_nd(v):
        return unpack_nd(v)
    if isinstance(v, dict):
        return {k: _in(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_in(x) for x in v]
    return v


def encode(topic, msg, seq=0, t_pub=None, v=VERSION):
    """v: the schema version to stamp (replay passes the logged one, so it sends what was recorded)."""
    body = dict(_out(msg), v=v, seq=seq, t_pub=now() if t_pub is None else t_pub)
    return [topic.encode(), msgpack.packb(body, use_bin_type=True)]


def decode(frames, validate=True):
    """[topic, payload] -> (topic, dict with numpy arrays restored). Raises ValueError on anything malformed.
    validate=False skips the cam.* schema check: logs keep whatever was sent, and tools read them as they are."""
    try:
        if len(frames) != 2:
            raise ValueError(f'expected 2 frames, got {len(frames)}')
        msg = msgpack.unpackb(frames[1], raw=False)
        if not isinstance(msg, dict):
            raise ValueError('payload is not a map')
        if msg.get('v') not in ACCEPTED:
            raise ValueError(f"schema version {msg.get('v')!r}, expected one of {ACCEPTED}")
        topic, msg = bytes(frames[0]).decode(), _in(msg)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f'malformed message: {type(e).__name__}: {e}') from e
    if validate and topic in CAM_TOPICS:
        bus.validate(topic, msg)
    return topic, msg


def _addr(name_or_addr):
    return ENDPOINTS.get(name_or_addr, name_or_addr)


class Pub:
    def __init__(self, name_or_addr, ctx=None, sndhwm=100):
        self.sock = (ctx or zmq.Context.instance()).socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, sndhwm)  # a slow subscriber loses messages instead of building a backlog
        self.sock.bind(_addr(name_or_addr))
        self.seq = {}

    def send(self, topic, msg):
        self.seq[topic] = self.seq.get(topic, 0) + 1
        self.sock.send_multipart(encode(topic, msg, self.seq[topic]))

    def close(self, linger_ms=500):
        if not self.sock.closed:
            self.sock.close(linger=linger_ms)


class Sub:
    def __init__(self, names_or_addrs, topics, ctx=None, rcvhwm=100):
        """topics: topic prefixes to receive, e.g. ['cam.tick', 'cam.final']; '' receives everything (the logger)."""
        self.sock = (ctx or zmq.Context.instance()).socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, rcvhwm)
        for a in names_or_addrs:
            self.sock.connect(_addr(a))
        for t in topics:
            self.sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    def recv(self, timeout_ms=None):
        if timeout_ms is not None and not self.sock.poll(timeout_ms):
            return None, None
        return decode(self.sock.recv_multipart())

    def close(self):
        self.sock.close(linger=0)
