"""The judge's words: call audio (mu-law 8 kHz, straight from Twilio) -> noise gate -> Cartesia Ink (ink-2, turn
detection) -> partner.text {text, is_final, t}. turn.update carries the whole turn so far (not a delta); turn.end the
final text.

NoiseGate: at the expo the judge's phone also hears everyone nearby, and Ink transcribes them too (first live call:
bystanders' sentences in the judge's captions, and turns that never ended). The judge holds the phone to their ear, so
their voice is far louder at the mic than voices across the room. The gate tracks the background level and passes only
audio clearly above it; the rest becomes mu-law silence, so Ink hears the caller alone and turns end when they stop.
"""
import asyncio
import logging
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

CHUNK_BYTES = 800           # 100 ms of mu-law at 8 kHz: Ink's recommended chunk size
FRAME = 160                 # 20 ms (Twilio's frame)
SILENCE = 0xFF              # mu-law code for 0
OPEN_DB = 12.0              # caller must be this far above the background level
MIN_DBFS = -50.0            # never open below this (line noise)
HANGOVER_FRAMES = 15        # keep open 300 ms after the last loud frame (word endings, short pauses)
FLOOR_WINDOW = 150          # background level = 20th percentile of the last 150 quiet frames' levels
LEARN_FRAMES = 25           # the first 0.5 s always count, so a noisy room still sets a floor
LOUD_EVERY = 5              # 1 in 5 loud frames feeds the floor too


def _mulaw_table():
    """G.711 mu-law byte -> linear sample (int16 range)."""
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign, exp, mant = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    mag = (((mant << 3) + 0x84) << exp) - 0x84
    return np.where(sign != 0, -mag, mag).astype(np.float32)


MULAW = _mulaw_table()


def level_dbfs(frame):
    x = MULAW[np.frombuffer(frame, np.uint8)]
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return 20 * np.log10(max(rms, 1.0) / 32768.0)


class NoiseGate:
    """Per 20 ms frame: pass it if it is OPEN_DB above the running background level (and above MIN_DBFS), with a
    hangover; otherwise replace it with silence. The background is a low percentile of recent levels, so a talking
    caller does not raise it."""

    def __init__(self, open_db=OPEN_DB, min_dbfs=MIN_DBFS, hangover=HANGOVER_FRAMES, window=FLOOR_WINDOW):
        self.open_db, self.min_dbfs, self.hangover = open_db, min_dbfs, hangover
        self.levels = deque(maxlen=window)
        self.hold = 0
        self.passed = self.total = 0

    @property
    def floor(self):
        return float(np.percentile(self.levels, 20)) if self.levels else -90.0

    def push(self, frame):
        lv = level_dbfs(frame)
        loud = lv >= max(self.floor + self.open_db, self.min_dbfs)
        # the floor learns from background, not from the caller's speech (a long sentence would raise it and shut the
        # gate), but loud frames still count 1 in LOUD_EVERY, so a room that gets louder lifts the floor over ~10 s
        if not loud or len(self.levels) < LEARN_FRAMES or self.total % LOUD_EVERY == 0:
            self.levels.append(lv)
        self.hold = self.hangover if loud else max(0, self.hold - 1)
        self.total += 1
        if loud or self.hold:
            self.passed += 1
            return frame
        return bytes([SILENCE]) * len(frame)


class Ink:
    """feed(bytes) from the Twilio handler; on_text(text, is_final, t) for every update."""

    def __init__(self, cfg, on_text, client=None, gate=True, record=None):
        self.cfg, self.on_text, self.client = cfg, on_text, client
        self.q = asyncio.Queue(maxsize=200)     # ~20 s of audio; if Ink stalls we drop rather than grow
        self.buf = bytearray()
        self.gate = NoiseGate() if gate else None
        self.raw = bytearray()
        self.record = record                    # debug only: path for (raw, gated) call audio, off by default
        self.tasks = []
        self._cm = self.conn = None
        self._behind_logged = 0.0
        self.alive = True

    async def start(self):
        if self.client is None:
            from cartesia import AsyncCartesia
            self.client = AsyncCartesia(api_key=self.cfg.cartesia_key)
        self._cm = self.client.stt.auto_finalize.websocket(model=self.cfg.stt_model, encoding='pcm_mulaw',
                                                           sample_rate=8000)
        self.conn = await self._cm.__aenter__()
        self.tasks = [asyncio.create_task(self._send()), asyncio.create_task(self._recv())]
        log.info('Ink websocket open (model %s)', self.cfg.stt_model)

    def feed(self, mulaw):
        if self.record:
            self.raw += mulaw
        if self.gate is not None:
            mulaw = b''.join(self.gate.push(mulaw[i:i + FRAME]) for i in range(0, len(mulaw), FRAME))
        self.buf += mulaw
        while len(self.buf) >= CHUNK_BYTES:
            chunk, self.buf = bytes(self.buf[:CHUNK_BYTES]), self.buf[CHUNK_BYTES:]
            try:
                self.q.put_nowait(chunk)
            except asyncio.QueueFull:
                if time.perf_counter() - self._behind_logged > 5:      # once per 5 s, not 10 times a second
                    self._behind_logged = time.perf_counter()
                    log.warning('Ink is behind or down: dropping call audio (captions paused)')

    async def _send(self):
        try:
            while True:
                await self.conn.send_raw(await self.q.get())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  the call goes on; only the captions stop
            self.alive = False
            log.exception('sending call audio to Ink failed: no more captions on this call')

    async def _recv(self):
        try:
            async for ev in self.conn:
                kind = getattr(ev, 'type', None)
                if kind == 'turn.update':
                    await self.on_text(ev.transcript, False, time.perf_counter())
                elif kind == 'turn.end':
                    await self.on_text(ev.transcript, True, time.perf_counter())
                elif kind == 'error':
                    log.error('Ink error: %s', ev)
        except Exception:  # noqa: BLE001
            log.exception('Ink receiver stopped')
        finally:
            self.alive = False

    async def close(self):
        for t in self.tasks:
            t.cancel()
        if self.gate is not None and self.gate.total:
            log.info('noise gate passed %.0f%% of call audio (background %.0f dBFS)',
                     100 * self.gate.passed / self.gate.total, self.gate.floor)
        if self.record and self.raw:
            with open(self.record, 'wb') as f:
                f.write(bytes(self.raw))
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
