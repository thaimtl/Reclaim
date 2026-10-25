"""Team bus for the call service: ZeroMQ PUB/SUB, multipart [topic, msgpack body], body carries v, seq, t_pub
(same envelope as bus/__init__.py). We publish partner.text, timing.voice and call.state; we read voice.say and cam.tick
(any version: only its emotion map {label, confidence, scores} is used, unchanged in bus v2).

The envelope (VERSION, t_pub = time.perf_counter()) is bus/__init__.py's; tests/call checks they stay equal. Decoding
is plain msgpack on purpose: cam.tick's numpy maps stay raw bytes, never converted, so reading its emotion at 10 Hz
costs almost nothing.
"""
import time

import msgpack
import zmq
import zmq.asyncio

VERSION = 2      # the team bus envelope (bus/__init__.py): plain msgpack maps are the same in v1 and v2
PARTNER_TEXT, TIMING_VOICE, CALL_STATE = 'partner.text', 'timing.voice', 'call.state'
VOICE_SAY, CAM_TICK = 'voice.say', 'cam.tick'


def encode(topic, body, seq):
    msg = {'v': VERSION, 'seq': int(seq), 't_pub': time.perf_counter(), **body}
    return [topic.encode(), msgpack.packb(msg, use_bin_type=True)]


def decode(frames):
    """[topic, payload] -> (topic str, dict). Raises ValueError on anything malformed."""
    try:
        topic, payload = frames
        msg = msgpack.unpackb(payload, raw=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'bad bus message: {exc}') from exc
    if not isinstance(msg, dict):
        raise ValueError('bad bus message: map expected')
    return topic.decode(), msg


class Publisher:
    """Binds once; send() never blocks (a missing subscriber drops messages, as PUB does)."""

    def __init__(self, addr, ctx=None):
        self.ctx = ctx or zmq.asyncio.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 100)
        self.sock.setsockopt(zmq.LINGER, 200)
        try:
            self.sock.bind(addr)
        except zmq.ZMQError:
            self.sock.close()       # an open socket would block the context's term() at exit
            raise
        self.addr = addr
        self.seq = {}

    async def send(self, topic, body):
        self.seq[topic] = self.seq.get(topic, 0) + 1
        await self.sock.send_multipart(encode(topic, body, self.seq[topic]), flags=zmq.NOBLOCK)

    def close(self):
        self.sock.close()


class Subscriber:
    """Connects to one or more publishers and yields (topic, msg) for the given topic prefixes."""

    def __init__(self, addrs, topics, ctx=None):
        self.ctx = ctx or zmq.asyncio.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 1000)
        for a in addrs:
            self.sock.connect(a)
        for t in topics:
            self.sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    async def recv(self):
        while True:
            frames = await self.sock.recv_multipart()
            try:
                return decode(frames)
            except ValueError:
                continue    # one bad message must not stop the call

    def close(self):
        self.sock.close()
