"""EMG support for the decoder, from svr/serve.py's emg.read: how strongly SVR-1's letters back each word against its
lip look-alikes, and the voice's pace and effort. numpy only: torch and the GPU stay in svr.serve.

support(word, now) takes the newest fresh emg.read, the EMG frames inside the word's span [t0 - PAD_S, t1 + PAD_S]
(camera frame times and EMG frame times are both bus.now(); offline both are the recorder's clock), and scores the
word and each look-alike in word.alts by the CTC log-likelihood of its letters over those frames. The support is the
word's share, softmax(loglik / TEMP). It is None (the camera carries the word alone) when there is no read, it is
older than STALE_S, the rig's signal is bad (ok False), the read does not cover the word yet, the word has no
alternatives, none of the candidates fits the span, or EMG is undecided: its best share is less than DECISIVE above
chance (1/k). Without that last rule an undecided EMG (share ~1/k) would pull a word the camera is sure of below
the lock threshold, since the gate gives EMG half the weight.

choose(word, now): the look-alike EMG clearly prefers, when the camera can't tell them apart (within CAM_CLOSE of
the word's camera prob) and EMG's support for it beats the word's by SWAP_MARGIN; None otherwise.

Letters are svr/text.py's: a-z, 0-9, space, CTC blank last (38); punctuation and apostrophes are dropped as its
clean_text does (tests check they match; svr.text needs unidecode, which the decoder's venv doesn't have).
"""
import string

import numpy as np

CHARS = string.ascii_lowercase + string.digits + ' '
BLANK = len(CHARS)                 # 37
_ID = {c: i for i, c in enumerate(CHARS)}
STALE_S = 0.3
PAD_S = 0.1                        # EMG leads the lips a little, and the camera's word span is its pieces only
TEMP = 1.0
SWAP_MARGIN = 0.3
DECISIVE = 0.2                     # best share at least this far above 1/k, or EMG has no say
CAM_CLOSE = 0.15
MAX_CANDS = 8
VOICE_RANGE = (0.25, 4.0)          # pace and effort are clamped to this before voice.say
INFEASIBLE = -1e4


def letter_ids(word):
    return [_ID[c] for c in word.lower() if c in _ID and c != ' ']


def ctc_loglik(logp, labels, blank=BLANK):
    """log P(labels | frames) under CTC, logp (T, C) per-frame log-probs. INFEASIBLE when no path fits."""
    T, L = len(logp), len(labels)
    if L == 0:
        return float(logp[:, blank].sum()) if T else 0.0
    if T < L + sum(a == b for a, b in zip(labels, labels[1:])):
        return INFEASIBLE
    ext = np.full(2 * L + 1, blank)
    ext[1::2] = labels
    S = len(ext)
    skip = np.zeros(S, bool)                       # s may come from s - 2: a label not repeating the one before
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    lp = logp[:, ext].astype(np.float64)           # (T, S)
    a = np.full(S, -np.inf)
    a[0], a[1] = lp[0, 0], lp[0, 1]
    for t in range(1, T):
        prev1 = np.concatenate(([-np.inf], a[:-1]))
        prev2 = np.where(skip, np.concatenate(([-np.inf, -np.inf], a[:-2])), -np.inf)
        a = np.logaddexp(np.logaddexp(a, prev1), prev2) + lp[t]
    out = np.logaddexp(a[-1], a[-2])
    return float(out) if np.isfinite(out) else INFEASIBLE


class EmgSupport:
    def __init__(self):
        self.read, self.read_t = None, -np.inf

    def on_read(self, msg, now):
        """One emg.read. A not-ok one drops the last read at once (the Uno was pulled, or a channel went bad)."""
        if msg.get('ok'):
            self.read, self.read_t = msg, now
        else:
            self.read = None

    def _fresh(self, now):
        return self.read is not None and now - self.read_t <= STALE_S

    def scores(self, word, now):
        """{candidate: support} for the word and its alternatives, or None (see the module docstring)."""
        if not self._fresh(now) or not word.alts:
            return None
        ft = np.asarray(self.read['frame_t'])
        if not len(ft) or ft[0] > word.t0 or ft[-1] < word.t1:
            return None
        span = (ft >= word.t0 - PAD_S) & (ft <= word.t1 + PAD_S)
        logp = np.asarray(self.read['letter_logp'], np.float32)[span]
        cands = list(dict.fromkeys([word.w] + [a for a, _ in word.alts]))[:MAX_CANDS]
        ll = np.array([ctc_loglik(logp, letter_ids(c)) for c in cands])
        if np.all(ll <= INFEASIBLE):
            return None
        z = (ll - ll.max()) / TEMP
        p = np.exp(z)
        p /= p.sum()
        return dict(zip(cands, p.tolist()))

    def support(self, word, now):
        s = self.scores(word, now)
        if s is None or max(s.values()) - 1 / len(s) < DECISIVE:
            return None
        return s[word.w]

    def choose(self, word, now):
        """(look-alike, its support) EMG settles the word on, or None."""
        s = self.scores(word, now)
        if s is None:
            return None
        best = None
        for a, p_cam in word.alts:
            if a in s and p_cam >= word.cam - CAM_CLOSE and s[a] >= s[word.w] + SWAP_MARGIN \
                    and (best is None or s[a] > best[1]):
                best = (a, s[a])
        return best

    def voice(self, now):
        """(pace, effort) for voice.say: the newest fresh read's, 1.0 = Austin's normal without one."""
        if not self._fresh(now):
            return 1.0, 1.0
        lo, hi = VOICE_RANGE
        clamp = lambda x: 1.0 if x is None or not np.isfinite(x) else float(min(max(x, lo), hi))
        return clamp(self.read.get('pace')), clamp(self.read.get('effort'))
