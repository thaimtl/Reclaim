"""Team bus for the call service: ZeroMQ PUB/SUB, multipart [topic, msgpack body], body carries v, seq, t_pub
(the team bus's envelope, bus/schema.md, v2; t_pub is bus.now() = time.perf_counter()). We publish partner.text,
timing.voice and call.state; we read voice.say and cam.tick (only its emotion map {label, confidence, scores}).

Only plain msgpack is read here (cam.tick's numpy maps stay as bytes and are ignored), so this needs no numpy.
"""
import os
import time

import msgpack
import zmq
import zmq.asyncio

VERSION = 2
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
        if addr.startswith('ipc://'):           # the socket file's folder must exist before bind
            os.makedirs(os.path.dirname(addr[len('ipc://'):]) or '.', exist_ok=True)
        self.sock.bind(addr)
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
