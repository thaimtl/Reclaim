"""Faint -> solid words for the live stream (D9).

A word turns solid when its CTC end is >= m frames behind the newest frame (hyp.n_frames - 1 - end_frame >= m) AND the
word, with everything before it (solid prefix included), is the same text at the same position over the last `agree`
updates. Words turn solid in reading order, so an Update is always solid words first, then faint ones.

Solid words stay solid until the utterance ends. Each new hyp is aligned to the solid prefix like this:
    - if the hyp's first n words read exactly like the n solid words, its words after position n are the faint tail;
    - otherwise (the hyp re-segmented or rewrote the prefix) the faint tail is every hyp word whose CTC end frame is
      later than the last solid word's end frame.
Solid words keep the end frame they had when they turned solid.

    c = Committer(margin_frames=8)
    update = c.update(hyp)      # LiveHyp -> Update; a new utt_id resets
"""
from collections import deque

from live.types import FAINT, SOLID, Update, Word, frame_time


class Committer:
    def __init__(self, margin_frames=8, agree=2):
        if agree < 1:
            raise ValueError('agree must be >= 1')
        self.margin_frames = margin_frames
        self.agree = agree
        self.revisions = 0          # faint words changed or dropped since start (all utterances)
        self.last_revisions = 0     # the same, for the last update only
        self.updates = 0            # updates seen since start
        self.reset()

    def reset(self):
        """Start a new utterance."""
        self.utt_id = None
        self.t0 = 0.0
        self.solid = []             # [WordEnd, ...] committed prefix
        self._history = deque(maxlen=self.agree)
        self._prev = ()             # texts of the previous update
        self._prev_solid = 0

    def _tail(self, hyp):
        n = len(self.solid)
        if n == 0:
            return list(hyp.words)
        if tuple(w.w for w in hyp.words[:n]) == tuple(w.w for w in self.solid):
            return list(hyp.words[n:])
        last = self.solid[-1].end_frame
        return [w for w in hyp.words if w.end_frame > last]

    def update(self, hyp):
        if hyp.utt_id != self.utt_id:
            self.reset()
            self.utt_id = hyp.utt_id
        self.t0 = hyp.t0
        seq = self.solid + self._tail(hyp)
        texts = tuple(w.w for w in seq)

        changed = sum(1 for j in range(self._prev_solid, len(self._prev))
                      if j >= len(texts) or texts[j] != self._prev[j])
        self.last_revisions = changed
        self.revisions += changed
        self.updates += 1

        self._history.append(texts)
        newest = hyp.n_frames - 1
        if len(self._history) >= self.agree:
            i = len(self.solid)
            while i < len(seq):
                if newest - seq[i].end_frame < self.margin_frames:
                    break
                if any(h[:i + 1] != texts[:i + 1] for h in self._history):
                    break
                self.solid.append(seq[i])
                i += 1

        self._prev, self._prev_solid = texts, len(self.solid)
        n_solid = len(self.solid)
        words = tuple(Word(w.w, SOLID if k < n_solid else FAINT, frame_time(hyp.t0, w.end_frame))
                      for k, w in enumerate(seq))
        return Update(utt_id=hyp.utt_id, t=hyp.t, words=words)

    @property
    def text(self):
        """Live text of the current utterance (solid + faint) as of the last update."""
        return ' '.join(self._prev)
