"""Audio helpers for the voice: trim Sonic's padding off a phrase, and encode it to Twilio's mu-law.

Sonic answers each flush (one phrase) with its speech padded by ~0.5 s of silence (measured 2026-10-25, sonic-3.6).
Played as is, that padding becomes a gap Austin never made, so each phrase is cut to its speech plus a natural pause.
"""
import numpy as np

RATE = 8000                 # Twilio's mu-law 8 kHz; Sonic is asked for linear 16-bit at the same rate
TRIM_FRAME = 80             # 10 ms frames for finding speech
TRIM_REL = 0.06             # a frame is speech above this share of the phrase's loudest frame
TRIM_FLOOR = 0.004          # ... and above this absolute RMS (int16 / 32768)
START_FLOOR = 0.01          # StreamTrim: speech starts above this (it can't know the phrase's loudest frame yet)
PRE_S = 0.02                # kept before the speech: the attack
FADE_S = 0.005


def trim(pcm, post_s=0.04):
    """int16 samples -> the speech in them, PRE_S before and post_s after (the release and a pause), faded."""
    x = np.asarray(pcm, np.int16)
    n = len(x) // TRIM_FRAME
    if n == 0:
        return x[:0]
    rms = np.sqrt(((x[:n * TRIM_FRAME].astype(np.float64).reshape(n, TRIM_FRAME) / 32768) ** 2).mean(1))
    loud = np.flatnonzero(rms > max(TRIM_FLOOR, TRIM_REL * rms.max()))
    if len(loud) == 0:
        return x[:0]
    a = max(0, loud[0] * TRIM_FRAME - int(PRE_S * RATE))
    b = min(len(x), (loud[-1] + 1) * TRIM_FRAME + int(post_s * RATE))
    return fade(x[a:b])


class StreamTrim:
    """trim(), as the audio streams in: leading silence is dropped as soon as speech shows, then the speech is passed on
    up to its last loud frame so far; whatever follows (a pause inside the phrase, or Sonic's ~0.5 s of padding) is held
    until more speech comes or end() trims it to post_s. Loud means above TRIM_FLOOR and TRIM_REL of the loudest frame
    so far, as in trim(). So a phrase starts playing with its first sound, not after Sonic has made all of it."""

    def __init__(self, post_s=0.04):
        self.post_s = post_s
        self.buf = np.zeros(0, np.int16)    # the phrase from its first sound (before: leading silence, being looked at)
        self.started, self.sent, self.peak, self.last = False, 0, 0.0, 0    # last: end of the last loud frame

    def _frames(self, x):
        n = len(x) // TRIM_FRAME
        return np.sqrt(((x[:n * TRIM_FRAME].astype(np.float64).reshape(n, TRIM_FRAME) / 32768) ** 2).mean(1))

    def push(self, pcm):
        """More samples -> the samples to play now."""
        old = len(self.buf)
        self.buf = np.concatenate([self.buf, np.asarray(pcm, np.int16)])
        if not self.started:
            loud = np.flatnonzero(self._frames(self.buf) > START_FLOOR)
            if not len(loud):
                return self.buf[:0]
            a = max(0, loud[0] * TRIM_FRAME - int(PRE_S * RATE))
            k = int(FADE_S * RATE)
            self.buf = np.concatenate([fade_in(self.buf[a:a + k]), self.buf[a + k:]])
            self.started, old = True, 0
        first = old // TRIM_FRAME             # frames not looked at yet (from the one the last push cut short)
        rms = self._frames(self.buf[first * TRIM_FRAME:])
        if len(rms):
            self.peak = max(self.peak, float(rms.max()))
            loud = np.flatnonzero(rms > max(TRIM_FLOOR, TRIM_REL * self.peak))
            if len(loud):
                self.last = max(self.last, (first + loud[-1] + 1) * TRIM_FRAME)
        out, self.sent = self.buf[self.sent:self.last], max(self.sent, self.last)
        return out

    def end(self):
        """The phrase is complete -> its held tail: up to post_s after the last loud frame, faded out."""
        if not self.started:
            return self.buf[:0]
        x = self.buf[self.sent:min(len(self.buf), self.last + int(self.post_s * RATE))]
        self.sent = len(self.buf)
        return fade_out(x)


def fade_in(x):
    x = np.asarray(x, np.float64)
    x *= np.linspace(0.0, 1.0, len(x), endpoint=False)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def fade_out(x):
    x = np.asarray(x, np.float64)
    k = min(int(FADE_S * RATE), len(x))
    if k:
        x[len(x) - k:] *= np.linspace(1.0, 0.0, k, endpoint=False)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def fade(x):
    x = np.asarray(x, np.float64)
    k = min(int(FADE_S * RATE), len(x) // 2)
    if k:
        ramp = np.linspace(0.0, 1.0, k, endpoint=False)
        x[:k] *= ramp
        x[len(x) - k:] *= ramp[::-1]
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def _mulaw_encode_table():
    """G.711 mu-law encoder for every int16 value, as a lookup table: the Sun reference (CPython's audioop), 14-bit
    magnitude + bias 33, clipped at 8159, segment from the highest set bit."""
    v = np.arange(-32768, 32768, dtype=np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    mag = np.minimum(np.where(v < 0, -v, v), 8159) + 33
    seg = np.searchsorted(np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]), mag)
    code = np.where(seg >= 8, 0x7F, (seg << 4) | ((mag >> (seg + 1)) & 0x0F))     # past the top segment: loudest
    return (code ^ mask).astype(np.uint8)


_ENC = _mulaw_encode_table()


def to_mulaw(pcm):
    """int16 samples -> G.711 mu-law bytes, for Twilio."""
    return _ENC[np.asarray(pcm, np.int16).astype(np.int32) + 32768].tobytes()
