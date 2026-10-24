"""Fake camera reader on the team bus: schema-valid cam.tick / cam.final on a timer, no camera, no model, no GPU.

Loops demo sentences. Each segment opens with 300 ms of pre-roll, grows every 100 ms (one cam.tick = the whole
segment re-read; tick k of an utterance covers ceil(k * 2.5) frames, so 3, 2, 3, 2 new frames), ends 0.5 s after the last word, and gets a cam.final 5-best ~1.2 s later (the measured beam time).
Token ids are the model's real ids (unigram5000_units.txt, blank 0), CTC frames are peaky, the newest word is
sometimes misread as a look-alike for a few frames and then corrected, the long sentence runs past the 8 s cap
(new seg_id, 2 s carried over), the face is lost for a moment in one sentence, and the emotion changes per sentence.
timing_ms is made up but plausible: MediaPipe and front end scale with the new frames, Conformer and CTC with T.

    venv-live/bin/python -m bus.stub [--pub ipc:///tmp/reclaim/cam.tick]
    venv-live/bin/python -m bus.listen             # in another terminal
"""
import argparse
import logging
import math
import os
import time

import numpy as np

import lipread
from bus import TICK_S, now
from bus import BLANK, CamFinal, CamTick, EMOTIONS, Emotion, TOPK, VOCAB
from lipread.types import FPS

UNITS = os.path.join(lipread.LIPFLOW, 'lipflow', 'unigram5000_units.txt')
SPM = os.path.join(lipread.WORK, 'third_party', 'auto_avsr', 'spm', 'unigram', 'unigram5000.model')

SENTENCES = [
    ('WHAT DO YOU NEED', 'neutral'),
    ('I WOULD LIKE SOME WATER PLEASE', 'happy'),
    ('CAN YOU CALL MY DAUGHTER', 'neutral'),
    ('I AM FEELING A LITTLE TIRED', 'sad'),
    ('THANK YOU FOR YOUR HELP', 'happy'),
    ('I WANTED TO TELL YOU THAT THE DOCTOR SAID I CAN GO HOME ON MONDAY IF MY BLOOD TESTS LOOK GOOD AGAIN', 'surprised'),
]
LOOKALIKES = {'WHAT': 'WHY', 'NEED': 'MEET', 'WATER': 'WAITER', 'CALL': 'CALM', 'TIRED': 'TIRE', 'HELP': 'HELD',
              'HOME': 'HOPE', 'GOOD': 'COULD', 'MY': 'BY', 'TELL': 'SELL'}
FACE_LOST = (2, 1.0, 1.4)


class Tokenizer:
    """Words -> model token ids. Model ids are line numbers of unigram5000_units.txt (blank 0), NOT the
    SentencePiece model's own ids."""

    def __init__(self, units=UNITS, spm=SPM):
        self.ids, self.pieces = {}, {}
        for line in open(units, encoding='utf-8'):
            p, i = line.split()
            self.ids[p] = int(i)
            self.pieces[int(i)] = p
        self.pieces[BLANK] = '<blank>'
        self.sp = None
        try:
            import sentencepiece
            self.sp = sentencepiece.SentencePieceProcessor(model_file=spm)
        except Exception:  # noqa: BLE001  (no sentencepiece or no model file: greedy longest match below)
            pass

    def encode(self, text):
        if self.sp is not None:
            return [self.ids.get(p, self.ids['<unk>']) for p in self.sp.encode(text.upper(), out_type=str)]
        out = []
        for w in text.upper().split():
            s = '▁' + w
            while s:
                n = next(n for n in range(len(s), 0, -1) if s[:n] in self.ids or n == 1)
                out.append(self.ids.get(s[:n], self.ids['<unk>']))
                s = s[n:]
        return out

    def text(self, ids):
        """CTC best path ids -> text (collapse repeats, drop blanks)."""
        out, prev = [], None
        for i in ids:
            if i != prev and i != BLANK:
                out.append(self.pieces[i])
            prev = i
        return ' '.join(''.join(out).replace('▁', ' ').split())


def labels(tok, text, word_s=0.4, preroll_s=0.3, still_s=0.5):
    """Per model frame: (label id or blank, word index or -1). Each word gets word_s; its pieces sit 2 frames each
    in the middle of the word, the rest is blank."""
    n_pre, n_word = int(preroll_s * FPS), int(word_s * FPS)
    lab, widx = [BLANK] * n_pre, [-1] * n_pre
    for k, w in enumerate(text.split()):
        ids = tok.encode(w)
        frames = [BLANK] * n_word
        start = max(0, (n_word - 2 * len(ids)) // 2)
        for j, i in enumerate(ids):
            for f in (start + 2 * j, start + 2 * j + 1):
                if f < n_word:
                    frames[f] = i
        lab += frames
        widx += [k] * n_word
    n_still = int(still_s * FPS)
    return lab + [BLANK] * n_still, widx + [-1] * n_still


class Stub:
    def __init__(self, sentences=SENTENCES, tick_s=TICK_S, beam_s=1.2, gap_s=1.5, cap_s=8.0, carry_s=2.0, seed=0):
        self.tok = Tokenizer()
        self.sentences = sentences
        self.period = tick_s
        self.beam_s, self.gap_s = beam_s, gap_s
        self.cap, self.carry = int(cap_s * FPS), int(carry_s * FPS)
        self.seed = seed
        self.k = -1
        self.seg_id = 0
        self.phase, self.next_at = 'gap', None
        self.finals = []

    def _start(self, now):
        self.k = (self.k + 1) % len(self.sentences)
        text, self.label = self.sentences[self.k]
        self.lab, self.widx = labels(self.tok, text)
        self.alt = [self.tok.encode(LOOKALIKES.get(w, w)) for w in text.split()]
        self.t0 = now
        self.lo = 0
        self.ticks = 0
        self.shown = 0
        self.seg_id += 1
        self.phase = 'speak'

    def _row(self, f, n_frames):
        """(top-k ids, top-k log-probs, blank log-prob) of utterance frame f when n_frames have been seen."""
        rng = np.random.default_rng((self.seed, self.seg_id, f, n_frames - f < 4))
        lab, w = self.lab[f], self.widx[f]
        p = {}
        if lab == BLANK:
            p[BLANK] = 0.9
        else:
            p[lab], p[BLANK] = 0.8, 0.08
            alt = self.alt[w][0] if w >= 0 and self.alt[w] else lab
            if alt != lab:
                p[alt] = 0.85 if n_frames - f < 4 and w % 2 == 1 else 0.06
                if p[alt] > p[lab]:
                    p[lab] = 0.1
        while len(p) < TOPK:
            p.setdefault(int(rng.integers(1, VOCAB - 1)), float(rng.uniform(1e-5, 4e-3)))
        ids = sorted(p, key=p.get, reverse=True)
        total = sum(p.values())
        logp = [math.log(p[i] / max(total, 1.0)) for i in ids]
        return ids, logp, math.log(p.get(BLANK, 1e-3) / max(total, 1.0))

    def _tick(self, now, n_frames):
        hi = min(n_frames, len(self.lab), self.lo + self.cap)
        rows = [self._row(f, hi) for f in range(self.lo, hi)]
        ids = np.array([r[0] for r in rows], dtype=np.int32)
        frame_t = self.t0 + np.arange(self.lo, hi) / FPS
        k, a, b = FACE_LOST
        face_ok = not (self.k == k and a <= now - self.t0 <= b)
        speaking = np.array([self.widx[f] >= 0 for f in range(self.lo, hi)])
        activity = np.where(speaking, 0.7 + 0.2 * np.sin(np.arange(self.lo, hi) * 0.9), 0.05).astype(np.float32)
        scores = {e: (0.7 if e == self.label else 0.3 / (len(EMOTIONS) - 1)) for e in EMOTIONS}
        T, new = hi - self.lo, max(0, hi - self.shown)
        self.shown = hi
        j = np.random.default_rng((self.seed, self.seg_id, hi, 1)).uniform(0.9, 1.15, 4)
        tm = {'T': int(T), 'mediapipe': round(5.5 * new * j[0], 3), 'frontend': round(1.2 * new * j[1], 3),
              'conformer': round((6.0 + 0.11 * T) * j[2], 3), 'ctc': round((0.4 + 0.012 * T) * j[3], 3)}
        tm['total'] = round(sum(tm[k] for k in ('mediapipe', 'frontend', 'conformer', 'ctc')) + 0.8, 3)
        return CamTick(self.seg_id, float(frame_t[0]), frame_t, ids,
                       np.array([r[1] for r in rows], dtype=np.float32),
                       np.array([r[2] for r in rows], dtype=np.float32),
                       self.tok.text(ids[:, 0].tolist()), activity, face_ok,
                       Emotion(self.label, scores[self.label], scores), timing_ms=tm)

    def _final(self, now, hi):
        """cam.final for the current segment (frames lo..hi), due beam_s later."""
        words = sorted({self.widx[f] for f in range(self.lo, hi) if self.widx[f] >= 0})
        truth = [self.sentences[self.k][0].split()[w] for w in words]
        nbest, seen = [], set()
        for j in range(len(truth) + 1):
            cand = ' '.join(LOOKALIKES.get(w, w) if i == j - 1 else w for i, w in enumerate(truth))
            if cand not in seen:
                seen.add(cand)
                nbest.append((cand, round(-3.2 - 0.4 * len(truth) - 1.3 * len(nbest), 3)))
            if len(nbest) == 5:
                break
        self.finals.append((now + self.beam_s, CamFinal(self.seg_id, tuple(nbest))))

    def tick(self, now):
        """Messages due at `now` (monotonic seconds)."""
        out = []
        if self.next_at is None:
            self.next_at = now + 0.5
        if self.phase == 'gap' and now >= self.next_at:
            self._start(now)
        if self.phase == 'speak':
            n = math.ceil(self.ticks * self.period * FPS - 1e-9)
            self.ticks += 1
            if n - self.lo > self.cap:
                if self.shown < self.lo + self.cap:
                    out.append(self._tick(now, self.lo + self.cap))
                self._final(now, self.lo + self.cap)
                self.lo += self.cap - self.carry
                self.seg_id += 1
            if n > self.lo:
                out.append(self._tick(now, n))
            if n >= len(self.lab):
                self._final(now, len(self.lab))
                self.phase, self.next_at = 'gap', now + self.gap_s
        due = [m for t, m in self.finals if t <= now]
        self.finals = [(t, m) for t, m in self.finals if t > now]
        return out + due


def run_loop(stub, send, clock=now, sleep=time.sleep, duration=None):
    """Tick the stub at its rate and send() each message; until `duration` clock seconds have passed (None: forever)."""
    start = clock()
    next_tick = start
    while duration is None or clock() - start < duration:
        for msg in stub.tick(clock()):
            send(msg)
        next_tick += stub.period
        sleep(max(0.0, next_tick - clock()))


def main():
    from bus import CAM_PUB, Publisher
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pub', default=CAM_PUB, help='address to bind (env RECLAIM_CAM_PUB)')
    ap.add_argument('--duration', type=float, default=None, help='seconds to run (default: forever)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

    pub = Publisher(args.pub)
    stub = Stub()
    print(f'stub cam.tick / cam.final on {pub.addr}  (Ctrl+C to stop)', flush=True)
    try:
        run_loop(stub, pub.send, duration=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()


if __name__ == '__main__':
    main()
