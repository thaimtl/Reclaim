"""cam.tick -> words: the CTC greedy path over the top-k ids, pieces joined into words with their frame spans.

Each tick re-reads the whole segment, so this runs on the whole tick every time. Per frame the best id is the
top-k's first id, or blank when blank_logp beats it. Repeats collapse and blanks drop (standard CTC); every piece
keeps the run of frames that emitted it. A piece starting with '▁' starts a new word.

A word's camera confidence is its least sure piece's peak probability: CTC emits each piece as a short spike,
and the spike's height is how sure the model was. Its alternatives are cheap ones only: for a one-piece word,
the other whole-word pieces in the top-k at the spike frame (with their probabilities, most likely first), then
the word's lip look-alikes from bench/lookalikes.py (same viseme sequence, so the camera cannot tell them apart),
most frequent first by the camera's own SentencePiece unigram scores. MAX_ALTS keeps the first few.
"""
import os
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, third_party: outside the repo
UNITS = os.path.join(WORK, 'third_party', 'lipflow', 'lipflow', 'unigram5000_units.txt')
SPM = os.path.join(WORK, 'third_party', 'auto_avsr', 'spm', 'unigram', 'unigram5000.model')

BLANK = 0
UNK = 1
WORD_START = '▁'  # '▁'
ALT_MIN_P = 0.05       # a competing whole-word piece at the spike frame becomes an alt above this probability
MAX_ALTS = 6           # alts per word, competitors first, then look-alikes


@dataclass
class Word:
    """One word of a pass. The decoder fills the fields after `alts` as the word moves through locking."""
    w: str                  # lowercase text (Claude's look-alike choice or a repair may change it)
    f0: int                 # first and last frame (inclusive) of its pieces in the tick
    f1: int
    t0: float               # capture times of those frames
    t1: float
    cam: float              # camera confidence 0..1
    alts: list = field(default_factory=list)  # [(word, camera prob)]: CTC competitors and look-alikes
    seg_id: int = 0
    passes: int = 0         # passes survived unchanged
    frozen: bool = False    # from a capped segment: no more passes will re-read it
    paused: bool = False    # the lips were still for a while after it
    face_ok: bool = True    # the face was found on the newest frame of its pass
    source: str = 'camera'  # camera / emg / claude: who chose this text
    emg: float = None       # EMG support 0..1, None until EMG exists
    claude: float = 0.0     # Claude's predicted weight for the word, 0..1
    nbest: float = None     # cam.final n-best support 0..1 once settled
    score: float = 0.0      # newest lock score
    lock_t: float = None    # decoder time it locked
    raw: str = None         # the text it locked with (before any Claude repair)
    spoken: str = None      # the text sent on voice.say, None until it leaves the repair window

    @property
    def mid(self):
        return (self.t0 + self.t1) / 2


def load_units(path=UNITS):
    """id -> piece. Line n of the units file is id n; blank (0) and <eos> (5048) are not in it."""
    with open(path, encoding='utf-8') as f:
        return {n: line.split()[0] for n, line in enumerate(f, 1) if line.strip()}


def load_word_logfreq(path=SPM):
    """word -> log-prob of its whole-word piece in the camera's SentencePiece unigram model.

    The model was fitted on Auto-AVSR's training text (LRS3 and friends), so the score is a word frequency from
    the same kind of speech the camera reads. Read straight from the .model protobuf (ModelProto field 1 =
    pieces, each with field 1 = piece, field 2 = float score), so the live venv needs no sentencepiece.
    """
    with open(path, 'rb') as f:
        buf = f.read()
    out = {}
    for field_no, _, v in _proto_fields(buf):
        if field_no != 1:
            continue
        piece, score = None, None
        for f2, _, v2 in _proto_fields(v):
            if f2 == 1:
                piece = v2.decode('utf-8')
            elif f2 == 2:
                score = struct.unpack('<f', v2)[0]
        if piece and score is not None and piece.startswith(WORD_START) and len(piece) > 1:
            out[piece[1:].lower()] = score
    return out


def _proto_fields(buf):
    """(field number, wire type, value) of one protobuf message; value is bytes except for varints."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        wt = key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f'protobuf wire type {wt} not supported')
        yield key >> 3, wt, v


def _varint(buf, i):
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


class LookAlikes:
    """word -> its lip look-alikes, most frequent first (unknown frequency last, then alphabetical)."""

    def __init__(self, lexicon, logfreq):
        self.lexicon, self.logfreq, self.cache = lexicon, logfreq, {}

    def __call__(self, word):
        if word not in self.cache:
            self.cache[word] = tuple(sorted(self.lexicon.lookalikes(word),
                                            key=lambda a: (-self.logfreq.get(a, -np.inf), a)))
        return self.cache[word]


def load_lookalikes(units, spm=SPM):
    """Look-alikes over the camera's whole-word pieces, ranked by frequency; None when cmudict or the model is missing."""
    try:
        from bench.lookalikes import Lexicon, load_cmudict
        pron, logfreq = load_cmudict(), load_word_logfreq(spm)
    except OSError as e:
        print(f'fusion: no look-alikes ({e})', file=sys.stderr)
        return None
    vocab = {p[1:].lower() for p in units.values() if p.startswith(WORD_START) and p[1:].isalpha()}
    return LookAlikes(Lexicon(vocab, pron), logfreq)


def greedy_ids(topk_ids, topk_logp, blank_logp):
    """Best id per frame: the top-k's first, or blank when blank beats it."""
    best = np.array(topk_ids[:, 0], dtype=np.int64)
    best[blank_logp > topk_logp[:, 0]] = BLANK
    return best


def piece_runs(best):
    """Collapse the greedy path: [(id, first frame, last frame)], blanks dropped, repeats merged."""
    runs, prev = [], BLANK
    for t, i in enumerate(best.tolist()):
        if i != BLANK:
            if i == prev:
                runs[-1][2] = t
            else:
                runs.append([i, t, t])
        prev = i
    return runs


def extract_words(tick, units, lookalikes=None):
    """A cam.tick -> [Word], in reading order. lookalikes(word) -> [word] adds lip look-alikes to the alts."""
    ids, logp, frame_t = tick['topk_ids'], tick['topk_logp'], tick['frame_t']
    words, cur = [], None
    for i, f0, f1 in piece_runs(greedy_ids(ids, logp, tick['blank_logp'])):
        piece = units.get(i)
        if piece is None or i == UNK:  # <eos> or <unk>: no text
            continue
        peak = f0 + int(np.argmax(logp[f0:f1 + 1, 0]))
        p = float(np.exp(logp[peak, 0]))
        if piece.startswith(WORD_START) or cur is None:
            cur = {'text': piece.lstrip(WORD_START), 'f0': f0, 'f1': f1, 'probs': [p], 'peaks': [peak]}
            words.append(cur)
        else:
            cur['text'] += piece
            cur['f1'] = f1
            cur['probs'].append(p)
            cur['peaks'].append(peak)
    out = []
    for c in words:
        if not c['text']:  # a bare '▁' with nothing after it
            continue
        w = Word(w=c['text'].lower(), f0=c['f0'], f1=c['f1'], t0=float(frame_t[c['f0']]),
                 t1=float(frame_t[c['f1']]), cam=min(c['probs']))
        if len(c['peaks']) == 1:
            w.alts = competitors(ids[c['peaks'][0]], logp[c['peaks'][0]], units, w.w)
        if lookalikes is not None:
            have = {a for a, _ in w.alts}
            w.alts += [(a, w.cam) for a in lookalikes(w.w) if a not in have and a != w.w]
        w.alts = w.alts[:MAX_ALTS]
        out.append(w)
    return out


def competitors(ids, logp, units, word):
    """Other whole-word pieces in one frame's top-k above ALT_MIN_P: [(word, prob)], most likely first."""
    out = []
    for i, lp in zip(ids[1:].tolist(), logp[1:].tolist()):
        p = float(np.exp(lp))
        if p < ALT_MIN_P:
            break
        piece = units.get(i, '')
        text = piece[1:].lower()
        if piece.startswith(WORD_START) and text and text != word:
            out.append((text, p))
    return out
