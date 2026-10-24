"""Team bus: ZeroMQ PUB/SUB on the laptop's loopback, msgpack payloads, numpy arrays as raw bytes.

Wire format (bus/schema.md): multipart [topic, payload], topic b'cam.tick', b'cam.final' or b'cam.face',
payload a msgpack map with the fields of docs/TEAM_TASKS_AND_INTERFACES.md plus v (schema version), seq (per-topic
counter) and t_pub (now() at send).
Every numpy array is the map {'nd': 1, 'dtype': '<f4', 'shape': [T, 32], 'data': <bin, C order>}.

    pub = Publisher()                       # binds CAM_PUB (env RECLAIM_CAM_PUB overrides)
    pub.send(cam_tick)                      # CamTick / CamFinal / CamFace, from any thread; a no-op after close()
    pub.send(cam_face, block=False)         # the crop thread: drops the message (returns 0) instead of waiting
    sub = Subscriber()                      # connects; cam.tick + cam.final (add FACE for the dashboard's cam.face)
    topic, msg = sub.recv(timeout_ms=1000)  # msg: validated dict with numpy arrays; None on timeout
    finals, ticks = sub.drain()             # everything waiting: every cam.final, the newest cam.tick per segment

encode/decode/validate work without sockets (tests, logs); encode refuses what validate refuses.
"""
from dataclasses import dataclass
import logging
import math
import os
import re
import threading
import time

import msgpack
import numpy as np
import zmq

log = logging.getLogger(__name__)
now = time.perf_counter
DEFAULT_CAM_PUB = 'tcp://127.0.0.1:5556'
CAM_PUB = os.environ.get('RECLAIM_CAM_PUB', DEFAULT_CAM_PUB)
TICK, FINAL, FACE = 'cam.tick', 'cam.final', 'cam.face'
VERSION = 2
TICK_S = 0.1
SNDHWM = 20
RCVHWM = 100
TOPK = 32
BLANK = 0
VOCAB = 5049
TIMING_KEYS = ('T', 'mediapipe', 'frontend', 'conformer', 'ctc', 'total')
EMOTIONS = ('neutral', 'happy', 'sad', 'angry', 'surprised')
_TEXT = re.compile(r"^([A-Z0-9']+( [A-Z0-9']+)*)?$")
_SPECIAL = re.compile(r'<[a-z]+>')
_NOT_TEXT = re.compile(r"[^A-Z0-9' ]")
DTYPES = {'frame_t': '<f8', 'topk_ids': '<i4', 'topk_logp': '<f4', 'blank_logp': '<f4', 'activity': '<f4',
          'pts': '<f2', 'blend': '<f2', 'pose': '<f4'}
FACE_POINTS = (468, 478)
FACE_RANGE = (-1.0, 2.0)


@dataclass(frozen=True)
class Emotion:
    label: str
    confidence: float
    scores: dict


@dataclass(frozen=True)
class CamTick:
    """One re-read of the whole current segment, every 100 ms while a segment is open (topic cam.tick)."""
    seg_id: int
    seg_start: float
    frame_t: np.ndarray
    topk_ids: np.ndarray
    topk_logp: np.ndarray
    blank_logp: np.ndarray
    greedy: str
    activity: np.ndarray
    face_ok: bool
    emotion: Emotion
    timing_ms: dict


@dataclass(frozen=True)
class CamFinal:
    """Beam search + LM over the finished segment (topic cam.final), best first."""
    seg_id: int
    nbest: tuple


@dataclass(frozen=True)
class CamFace:
    """The newest face for the dashboard (topic cam.face, ~12.5 Hz), from the crop thread. pts are normalised to the
    processed frame (x / w, y / h), so the dashboard needs no frame size; w, h say what they are relative to."""
    t: float
    w: int
    h: int
    face_ok: bool
    pts: np.ndarray = None
    blend: np.ndarray = None
    pose: np.ndarray = None


def face_msg(t, last_face, last_pose, w, h):
    """LiveCropper.last_face / last_pose of a w x h frame -> CamFace (the run loop's one-liner)."""
    if last_face is None:
        return CamFace(float(t), int(w), int(h), False)
    _, pts, blend = last_face
    pts = np.asarray(pts, np.float32) / np.array([w, h], np.float32)
    return CamFace(float(t), int(w), int(h), True, pts, blend, last_pose)


def cast_nd(a, dtype=None):
    """Contiguous array; dtype: cast to it, but only within a kind (float64 -> float32, int64 -> int32 in range),
    never float -> int."""
    a = np.asarray(a)
    if dtype is not None:
        dt = np.dtype(dtype)
        if not np.can_cast(a.dtype, dt, 'same_kind'):
            raise TypeError(f'{a.dtype} -> {dt}: not a same-kind cast')
        if dt.kind in 'iu' and a.size and (a.min() < np.iinfo(dt).min or a.max() > np.iinfo(dt).max):
            raise TypeError(f'{a.dtype} -> {dt}: values out of range')
    return np.ascontiguousarray(a, dtype=dtype)


def pack_nd(a, dtype=None):
    a = cast_nd(a, dtype)
    return {'nd': 1, 'dtype': a.dtype.str, 'shape': list(a.shape), 'data': a.tobytes()}


def unpack_nd(d):
    """Read-only view on the payload bytes (no copy)."""
    return np.frombuffer(d['data'], dtype=np.dtype(d['dtype'])).reshape(d['shape'])


def _is_nd(v):
    return isinstance(v, dict) and v.get('nd') == 1 and 'data' in v


def bus_text(pieces):
    """Joined SentencePiece pieces (CTC best path or a beam hypothesis) -> the text validate accepts: '▁' starts a
    word, special tokens are dropped (as LipReader's greedy text, which never shows <unk>), UPPERCASE, nothing but
    A-Z, 0-9 and '. The one place that decides it."""
    text = _SPECIAL.sub('', pieces.replace('▁', ' ')).upper()
    return ' '.join(_NOT_TEXT.sub('', text).split())


def zero_timing(T):
    """timing_ms for a tick nobody timed (tests, tools): every stage 0 ms."""
    return {'T': int(T), **{k: 0.0 for k in TIMING_KEYS[1:]}}


def to_msg(obj):
    """CamTick / CamFinal -> (topic, dict of plain msgpack types and numpy arrays)."""
    if isinstance(obj, CamTick):
        e = obj.emotion
        return TICK, {'seg_id': int(obj.seg_id), 'seg_start': float(obj.seg_start),
                      'frame_t': obj.frame_t, 'topk_ids': obj.topk_ids, 'topk_logp': obj.topk_logp,
                      'blank_logp': obj.blank_logp, 'greedy': obj.greedy, 'activity': obj.activity,
                      'face_ok': bool(obj.face_ok), 'timing_ms': _timing_out(obj.timing_ms),
                      'emotion': {'label': e.label, 'confidence': float(e.confidence),
                                  'scores': {k: float(v) for k, v in e.scores.items()}}}
    if isinstance(obj, CamFinal):
        return FINAL, {'seg_id': int(obj.seg_id), 'nbest': [{'text': t, 'score': float(s)} for t, s in obj.nbest]}
    if isinstance(obj, CamFace):
        return FACE, {'t': float(obj.t), 'w': int(obj.w), 'h': int(obj.h), 'face_ok': bool(obj.face_ok),
                      'pts': obj.pts, 'blend': obj.blend, 'pose': obj.pose}
    raise TypeError(f'not a bus message: {type(obj).__name__}')


def _timing_out(tm):
    """timing_ms -> plain msgpack map (numpy scalars to int / float). None stays None (encode's validate refuses it)."""
    if tm is None:
        return None
    def plain(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v
    return {k: plain(v) for k, v in dict(tm).items()}


def from_msg(topic, msg):
    """Decoded dict -> CamTick / CamFinal (inverse of to_msg)."""
    if topic == TICK:
        e = msg['emotion']
        return CamTick(msg['seg_id'], msg['seg_start'], msg['frame_t'], msg['topk_ids'], msg['topk_logp'],
                       msg['blank_logp'], msg['greedy'], msg['activity'], msg['face_ok'],
                       Emotion(e['label'], e['confidence'], dict(e['scores'])),
                       dict(msg['timing_ms']))
    if topic == FINAL:
        return CamFinal(msg['seg_id'], tuple((n['text'], n['score']) for n in msg['nbest']))
    if topic == FACE:
        return CamFace(msg['t'], msg['w'], msg['h'], msg['face_ok'], msg.get('pts'), msg.get('blend'), msg.get('pose'))
    raise ValueError(f'unknown topic {topic!r}')


def encode(obj, seq=0, t_pub=None):
    """CamTick / CamFinal -> [topic bytes, msgpack bytes], arrays cast to DTYPES. Validates first: a message
    validate() would refuse raises ValueError here (TypeError for a lossy array cast), so none reaches the wire."""
    topic, msg = to_msg(obj)
    body = {'v': VERSION, 'seq': int(seq), 't_pub': now() if t_pub is None else float(t_pub)}
    for k, v in msg.items():
        body[k] = cast_nd(v, DTYPES[k]) if k in DTYPES and v is not None else v
    validate(topic, body)
    return [topic.encode(), msgpack.packb({k: pack_nd(v) if k in DTYPES and v is not None else v
                                           for k, v in body.items()}, use_bin_type=True)]


def decode(frames):
    """[topic bytes, msgpack bytes] -> (topic, dict) with every array map turned back into numpy.
    Any malformed message raises ValueError."""
    try:
        topic, payload = frames
        msg = msgpack.unpackb(payload, raw=False)
        if not isinstance(msg, dict):
            raise ValueError(f'map expected, got {type(msg).__name__}')
        return bytes(topic).decode(), {k: unpack_nd(v) if _is_nd(v) else v for k, v in msg.items()}
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001  (msgpack, unicode, numpy errors: one type for listeners)
        raise ValueError(f'undecodable message: {type(e).__name__}: {e}') from e


def _need(msg, keys, where):
    missing = [k for k in keys if k not in msg]
    if missing:
        raise ValueError(f'{where}: missing {missing}')


def _arr(msg, key, ndim, where):
    a = msg[key]
    if not isinstance(a, np.ndarray) or a.dtype != np.dtype(DTYPES[key]) or a.ndim != ndim:
        got = f'{a.dtype}{a.shape}' if isinstance(a, np.ndarray) else type(a).__name__
        raise ValueError(f'{where}.{key}: {ndim}-d {DTYPES[key]} array expected, got {got}')
    return a


def _int(msg, key, where, lo):
    v = msg[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f'{where}.{key}: int >= {lo} expected, got {v!r}')


def _num(v, where, lo=-math.inf, hi=math.inf):
    """A finite number in [lo, hi] (bools are not numbers here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f'{where}: finite number in [{lo}, {hi}] expected, got {v!r}')


def validate(topic, msg):
    """Raise ValueError unless msg (as decode returns it) is a valid cam.tick / cam.final / cam.face. Returns msg.

    NaN anywhere is rejected; -inf log-probs are allowed (log 0)."""
    if not isinstance(msg, dict):
        raise ValueError(f'{topic}: map expected, got {type(msg).__name__}')
    _need(msg, ('v', 'seq', 't_pub'), topic)
    if msg['v'] != VERSION:
        raise ValueError(f'{topic}.v: {VERSION} expected, got {msg["v"]!r}')
    _int(msg, 'seq', topic, 0)
    _num(msg['t_pub'], f'{topic}.t_pub')
    if topic == FACE:
        return _validate_face(msg)
    _need(msg, ('seg_id',), topic)
    _int(msg, 'seg_id', topic, 1)
    if topic == TICK:
        _need(msg, ('seg_start', 'frame_t', 'topk_ids', 'topk_logp', 'blank_logp', 'greedy', 'activity',
                    'face_ok', 'emotion', 'timing_ms'), topic)
        _num(msg['seg_start'], f'{topic}.seg_start')
        ft = _arr(msg, 'frame_t', 1, topic)
        T = len(ft)
        if T < 1:
            raise ValueError(f'{topic}.frame_t: at least one frame expected')
        if not np.all(np.isfinite(ft)) or (T > 1 and not np.all(np.diff(ft) > 0)):
            raise ValueError(f'{topic}.frame_t: finite and increasing expected')
        if ft[0] < msg['seg_start'] - 1e-6:
            raise ValueError(f'{topic}.frame_t: starts before seg_start')
        for key, shape in (('topk_ids', (T, TOPK)), ('topk_logp', (T, TOPK)), ('blank_logp', (T,)),
                           ('activity', (T,))):
            a = _arr(msg, key, len(shape), topic)
            if a.shape != shape:
                raise ValueError(f'{topic}.{key}: shape {shape} expected, got {a.shape}')
        ids, logp, blank = msg['topk_ids'], msg['topk_logp'], msg['blank_logp']
        if ids.min() < 0 or ids.max() >= VOCAB:
            raise ValueError(f'{topic}.topk_ids: ids in [0, {VOCAB}) expected')
        if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError(f'{topic}.topk_ids: ids in a row must be distinct')
        if np.isnan(logp).any() or np.isnan(blank).any():
            raise ValueError(f'{topic}: NaN log-prob')
        if np.any(logp > 1e-4) or np.any(blank > 1e-4):
            raise ValueError(f'{topic}: log-probs must be <= 0')
        with np.errstate(invalid='ignore'):
            if np.any(np.diff(logp, axis=1) > 1e-4):
                raise ValueError(f'{topic}.topk_logp: each row must be sorted, most likely first')
        act = msg['activity']
        if not np.all((act >= 0) & (act <= 1)):
            raise ValueError(f'{topic}.activity: values in [0, 1] expected')
        if not isinstance(msg['greedy'], str) or not _TEXT.match(msg['greedy']):
            raise ValueError(f'{topic}.greedy: UPPERCASE words expected, got {msg["greedy"]!r}')
        if not isinstance(msg['face_ok'], bool):
            raise ValueError(f'{topic}.face_ok: bool expected')
        e = msg['emotion']
        if not isinstance(e, dict) or e.get('label') not in EMOTIONS:
            raise ValueError(f'{topic}.emotion.label: one of {EMOTIONS} expected, got {e!r}')
        _num(e.get('confidence'), f'{topic}.emotion.confidence', 0, 1)
        if not isinstance(e.get('scores'), dict) or set(e['scores']) != set(EMOTIONS):
            raise ValueError(f'{topic}.emotion.scores: one score per label in {EMOTIONS} expected')
        for k, v in e['scores'].items():
            _num(v, f'{topic}.emotion.scores.{k}', 0, 1)
        tm = msg['timing_ms']
        if not isinstance(tm, dict) or set(tm) != set(TIMING_KEYS):
            raise ValueError(f'{topic}.timing_ms: map with exactly the keys {TIMING_KEYS} expected, got {tm!r}')
        if isinstance(tm['T'], bool) or not isinstance(tm['T'], int) or tm['T'] != T:
            raise ValueError(f'{topic}.timing_ms.T: int == len(frame_t) = {T} expected, got {tm["T"]!r}')
        for k in TIMING_KEYS[1:]:
            _num(tm[k], f'{topic}.timing_ms.{k}', 0)
    elif topic == FINAL:
        _need(msg, ('nbest',), topic)
        nb = msg['nbest']
        if not isinstance(nb, list) or not nb:
            raise ValueError(f'{topic}.nbest: at least one entry expected')
        for i, n in enumerate(nb):
            if not isinstance(n, dict) or not isinstance(n.get('text'), str) or not _TEXT.match(n['text']):
                raise ValueError(f'{topic}.nbest[{i}].text: UPPERCASE words expected, got {n!r}')
            _num(n.get('score'), f'{topic}.nbest[{i}].score')
    else:
        raise ValueError(f'unknown topic {topic!r}')
    return msg


def _validate_face(msg):
    where = FACE
    _need(msg, ('t', 'w', 'h', 'face_ok', 'pts', 'blend', 'pose'), where)
    _num(msg['t'], f'{where}.t')
    _int(msg, 'w', where, 1)
    _int(msg, 'h', where, 1)
    if not isinstance(msg['face_ok'], bool):
        raise ValueError(f'{where}.face_ok: bool expected')
    if not msg['face_ok']:
        if any(msg[k] is not None for k in ('pts', 'blend', 'pose')):
            raise ValueError(f'{where}: no face, so pts / blend / pose must be nil')
        return msg
    pts = _arr(msg, 'pts', 2, where)
    if pts.shape[0] not in FACE_POINTS or pts.shape[1] != 2:
        raise ValueError(f'{where}.pts: shape (468 or 478, 2) expected, got {pts.shape}')
    if not np.all(np.isfinite(pts)) or pts.min() < FACE_RANGE[0] or pts.max() > FACE_RANGE[1]:
        raise ValueError(f'{where}.pts: finite normalised coordinates in {FACE_RANGE} expected')
    if msg['blend'] is not None:
        b = _arr(msg, 'blend', 1, where)
        if b.shape != (52,) or not np.all((b >= 0) & (b <= 1)):
            raise ValueError(f'{where}.blend: 52 scores in [0, 1] expected')
    if msg['pose'] is not None:
        p = _arr(msg, 'pose', 1, where)
        if p.shape != (3,) or not np.all(np.abs(p) <= math.pi + 1e-3):
            raise ValueError(f'{where}.pose: (yaw, pitch, roll) radians in [-pi, pi] expected')
    return msg


class Publisher:
    """One PUB socket for the camera reader. send() is thread-safe (tick thread and final-pass thread), and close()
    takes the same lock: a send racing close() either goes out before it or is dropped (send returns 0), never
    touches a closed socket."""

    def __init__(self, addr=CAM_PUB, bind=True, ctx=None):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, SNDHWM)
        self.sock.setsockopt(zmq.LINGER, 0)
        try:
            (self.sock.bind if bind else self.sock.connect)(addr)
        except zmq.ZMQError:
            self.sock.close()
            raise
        self.addr = self.sock.getsockopt_string(zmq.LAST_ENDPOINT) if bind else addr
        self.seq = {TICK: 0, FINAL: 0, FACE: 0}
        self.dropped_busy = 0
        self.lock = threading.Lock()
        self.closed = False
        self.dropped_after_close = 0

    def send(self, obj, block=True):
        """Returns the number of payload bytes sent; 0 after close() (the message is dropped and counted).
        block=False (the crop thread's cam.face): if another thread is sending, drop this message and return 0."""
        topic = TICK if isinstance(obj, CamTick) else FACE if isinstance(obj, CamFace) else FINAL
        frames = None
        if not self.lock.acquire(blocking=block):
            self.dropped_busy += 1
            return 0
        try:
            if self.closed:
                self.dropped_after_close += 1
                if self.dropped_after_close == 1:
                    log.warning('bus send after close: %s dropped', topic)
                return 0
            frames = encode(obj, self.seq[topic] + 1)
            self.sock.send_multipart(frames, copy=False)
            self.seq[topic] += 1
            return len(frames[1])
        finally:
            self.lock.release()

    def close(self, linger_ms=500):
        """Waits up to linger_ms for queued messages (the last cam.final) to go out; LINGER is 0 until now.
        Idempotent; later send() calls are no-ops."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.sock.close(linger=linger_ms)


class Subscriber:
    """SUB socket for teammates' processes. Each cam.tick is a full snapshot of its segment, so keep only the newest
    one per seg_id, but never drop a cam.final: drain() does both. The default topics leave out cam.face (~12.5 Hz):
    a subscriber that doesn't draw faces shouldn't queue them.

    ZeroMQ queues up to SNDHWM messages at the publisher and rcvhwm here, and past that drops the NEWEST ones, so a
    subscriber that falls behind must call drain() rather than recv() one message at a time. A message that fails
    decode or validate is logged once, counted in .bad and skipped, never raised."""

    def __init__(self, addr=CAM_PUB, topics=(TICK, FINAL), ctx=None, rcvhwm=RCVHWM):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, rcvhwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        for t in topics:
            self.sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        self.sock.connect(addr)
        self.bad = 0

    def _valid(self, frames):
        try:
            topic, msg = decode(frames)
            return topic, validate(topic, msg)
        except ValueError as e:
            self.bad += 1
            if self.bad == 1:
                log.warning('bad bus message skipped (later ones are only counted): %s', e)
            return None

    def recv(self, timeout_ms=None):
        """(topic, msg) or None on timeout; msg as decode() returns it, already validated."""
        while timeout_ms is None or self.sock.poll(timeout_ms):
            got = self._valid(self.sock.recv_multipart())
            if got:
                return got
        return None

    def drain(self):
        """Everything already queued, without waiting: (finals, ticks). finals: every cam.final in arrival order.
        ticks: {seg_id: newest cam.tick}, plus {'cam.face': newest cam.face} if subscribed."""
        finals, ticks = [], {}
        while True:
            try:
                frames = self.sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return finals, ticks
            got = self._valid(frames)
            if got is None:
                continue
            topic, msg = got
            if topic == FINAL:
                finals.append(msg)
            else:
                ticks[msg['seg_id'] if topic == TICK else topic] = msg

    def close(self):
        self.sock.close()
