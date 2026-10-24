"""The team bus: ZeroMQ PUB/SUB over local ipc:// sockets, msgpack payloads, numpy arrays as raw bytes.

Same wire format as the camera reader (live/bus.py, live/bus_schema.md): multipart [topic, payload], payload a
msgpack map with the message fields plus v (schema version), seq (per-topic counter from 1) and t_pub
(time.monotonic() at send). Every numpy array is the map {'nd': 1, 'dtype': '<f4', 'shape': [...], 'data': <bin>}.

Each producer binds one endpoint; consumers connect to the ones they need and name their topics. RECLAIM_<NAME>_PUB
overrides an endpoint (e.g. RECLAIM_DEC_PUB), and tcp:// works too. Binding an ipc path that a live publisher holds
is an error, never a silent steal; a socket file left by a crashed publisher is removed.

ZeroMQ sockets are not thread-safe: use one Pub or Sub per thread and never share one. Decoded arrays are
writeable copies.

    pub = Pub('dec')                          # binds ENDPOINTS['dec']
    pub.send('dec.lock', {'utt_id': 3, 'words': [...]})
    sub = Sub(['cam', 'claude'], ['cam.tick', 'cam.final', 'claude.'])
    topic, msg = sub.recv(timeout_ms=100)     # (None, None) on timeout; ValueError on a malformed message
"""
import errno
import os
import socket
import stat
import time

import msgpack
import numpy as np
import zmq

VERSION = 2
_DEFAULTS = {
    'emg': 'ipc:///tmp/reclaim/emg',          # emg.raw (rig/publish.py)
    'cam': 'ipc:///tmp/reclaim/cam.tick',     # cam.tick, cam.final, cam.face (live/run.py)
    'dec': 'ipc:///tmp/reclaim/dec',          # dec.hyp, dec.lock, dec.repeat, voice.say
    'claude': 'ipc:///tmp/reclaim/claude',    # claude.predict, claude.repair, timing.claude
    'call': 'ipc:///tmp/reclaim/call',        # partner.text, timing.call
}
ENDPOINTS = {k: os.environ.get(f'RECLAIM_{k.upper()}_PUB', v) for k, v in _DEFAULTS.items()}


def pack_nd(a):
    a = np.ascontiguousarray(a)
    return {'nd': 1, 'dtype': a.dtype.str, 'shape': list(a.shape), 'data': a.tobytes()}


def _is_nd(v):
    return isinstance(v, dict) and v.get('nd') == 1 and 'data' in v


def unpack_nd(d):
    return np.frombuffer(d['data'], dtype=np.dtype(d['dtype'])).reshape(d['shape']).copy()


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


def encode(topic, msg, seq=0, t_pub=None):
    body = dict(_out(msg), v=VERSION, seq=seq, t_pub=time.monotonic() if t_pub is None else t_pub)
    return [topic.encode(), msgpack.packb(body, use_bin_type=True)]


def decode(frames):
    """[topic, payload] -> (topic, dict with numpy arrays restored). Raises ValueError on anything malformed."""
    try:
        if len(frames) != 2:
            raise ValueError(f'expected 2 frames, got {len(frames)}')
        msg = msgpack.unpackb(frames[1], raw=False)
        if not isinstance(msg, dict):
            raise ValueError('payload is not a map')
        if msg.get('v') != VERSION:
            raise ValueError(f"schema version {msg.get('v')!r}, expected {VERSION}")
        return bytes(frames[0]).decode(), _in(msg)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f'malformed message: {type(e).__name__}: {e}') from e


def _addr(name_or_addr):
    return ENDPOINTS.get(name_or_addr, name_or_addr)


def _prepare_ipc(path):
    """Make the socket's folder; remove a file left by a crashed publisher, refuse one a live publisher holds."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not os.path.exists(path):
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        probe.connect(path)
    except OSError as e:
        if e.errno not in (errno.ECONNREFUSED, errno.ENOENT, errno.ENOTSOCK):
            raise
        if not stat.S_ISSOCK(os.stat(path).st_mode):  # never delete a file that isn't a socket
            raise FileExistsError(f'{path}: exists and is not a socket') from None
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return
    finally:
        probe.close()
    raise zmq.ZMQError(errno.EADDRINUSE, f'{path}: another publisher is bound here')


class Pub:
    def __init__(self, name_or_addr, ctx=None, sndhwm=100):
        addr = _addr(name_or_addr)
        self.sock = (ctx or zmq.Context.instance()).socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, sndhwm)  # a slow subscriber loses old messages instead of building a backlog
        self._ipc = None
        if addr.startswith('ipc://'):
            path = addr[len('ipc://'):]
            _prepare_ipc(path)
            self.sock.bind(addr)
            self._ipc = (path, os.stat(path).st_ino)
        else:
            self.sock.bind(addr)
        self.seq = {}

    def send(self, topic, msg):
        self.seq[topic] = self.seq.get(topic, 0) + 1
        self.sock.send_multipart(encode(topic, msg, self.seq[topic]))

    def close(self, linger_ms=500):
        if self.sock.closed:
            return
        self.sock.close(linger=linger_ms)
        if self._ipc:  # libzmq leaves the socket file behind; remove it if it is still ours
            path, ino = self._ipc
            try:
                if os.stat(path).st_ino == ino:
                    os.unlink(path)
            except FileNotFoundError:
                pass


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
