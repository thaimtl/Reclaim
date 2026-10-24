"""The joint search's language model (docs/SVR1_SPEC.md section 6): the conversational n-gram mixed with a cache of
this call's own words.

    lm = ConversationLM()                                  # data/lm/conversational.binary (python -m fusion.build_lm)
    lm.add_text('hi, this is Sarah from admissions')       # every final partner.text line (the judge)
    lm.add_text('i wanted to ask about my application')    # Austin's words as they lock
    s = lm.start()                                         # state at the start of a sentence
    lp, s = lm.score(s, 'sarah')                           # ln P(word | the words before it), next state
    lp_end = lm.end(s)                                     # ln P(end of sentence | ...)

P(w | h) = (1 - lam) * P_ngram(w | h) + lam * P_cache(w | previous word), where
    P_ngram   the pruned 4-gram over everyday dialogue (KenLM), the general sense of which words follow which
    P_cache   bigram counts of the call so far, backed off to its unigram counts: names and topics just mentioned
              become likely at once, even words the n-gram never saw. Counts fade with a half-life of
              `half_life` words, so the current topic counts most.
    lam       grows with the cache from 0 (empty: the n-gram alone) towards `cache_weight`.
Words are normalised like SVR-1's text (svr.text.clean_text). A query costs microseconds, so the joint search can
score every hypothesis on every 100 ms pass. Claude's predictions are separate: a bonus in the joint search (hints).
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
DEFAULT_LM = os.path.join(WORK, 'data', 'lm', 'conversational.binary')
LN10 = math.log(10)


def _normalise(text):
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from svr.text import clean_text
    return clean_text(text).split()


class ConversationLM:
    def __init__(self, path=DEFAULT_LM, cache_weight=0.15, half_life=400, cache_ramp=30, backoff=2.0):
        """cache_weight: the cache's share once it is full; half_life: words until a mention counts half;
        cache_ramp: cache size (words) at which it gets half of cache_weight; backoff: weight of the cache's unigram
        inside its bigram estimate."""
        import kenlm
        self._kenlm = kenlm
        self.model = kenlm.Model(path)
        self.cache_weight, self.cache_ramp, self.backoff = cache_weight, cache_ramp, backoff
        self.decay = 0.5 ** (1 / half_life)
        self.uni, self.bi, self.ctx, self.total = {}, {}, {}, 0.0

    # ------------------------------------------------------------------ the call so far
    def add_text(self, text, weight=1.0):
        """Add words to the cache (older ones fade); returns the normalised words added."""
        words = _normalise(text)
        if not words:
            return words
        fade = self.decay ** len(words)
        for table in (self.uni, self.bi, self.ctx):
            for k in table:
                table[k] *= fade
        self.total *= fade
        prev = '<s>'
        for w in words:
            self.uni[w] = self.uni.get(w, 0.0) + weight
            self.bi[prev, w] = self.bi.get((prev, w), 0.0) + weight
            self.ctx[prev] = self.ctx.get(prev, 0.0) + weight
            self.total += weight
            prev = w
        self.bi[prev, '</s>'] = self.bi.get((prev, '</s>'), 0.0) + weight  # where sentences end
        self.ctx[prev] = self.ctx.get(prev, 0.0) + weight
        return words

    def reset(self):
        """A new call: forget the cache."""
        self.uni, self.bi, self.ctx, self.total = {}, {}, {}, 0.0

    def _lam(self):
        return self.cache_weight * self.total / (self.total + self.cache_ramp)

    def _cache_prob(self, prev, word):
        p_uni = self.uni.get(word, 0.0) / self.total if self.total else 0.0
        return (self.bi.get((prev, word), 0.0) + self.backoff * p_uni) / (self.ctx.get(prev, 0.0) + self.backoff)

    def _mix(self, log10_ngram, prev, word):
        lam = self._lam()
        p = (1 - lam) * 10 ** log10_ngram
        if lam > 0:
            p += lam * self._cache_prob(prev, word)
        return math.log(p)

    # ------------------------------------------------------------------ scoring
    def start(self):
        s = self._kenlm.State()
        self.model.BeginSentenceWrite(s)
        return s, '<s>'

    def score(self, state, word):
        """(ln P(word | history), next state). `word` is one word; it is normalised like SVR-1's text."""
        words = _normalise(word)
        if len(words) != 1:
            raise ValueError(f'score() takes one word, got {word!r}')
        w = words[0]
        s, prev = state
        out = self._kenlm.State()
        log10 = self.model.BaseScore(s, w, out)
        return self._mix(log10, prev, w), (out, w)

    def end(self, state):
        s, prev = state
        out = self._kenlm.State()
        return self._mix(self.model.BaseScore(s, '</s>', out), prev, '</s>')

    def sentence(self, text, eos=True):
        """ln P of a whole sentence (start, every word, end)."""
        s, total = self.start(), 0.0
        for w in _normalise(text):
            lp, s = self.score(s, w)
            total += lp
        return total + (self.end(s) if eos else 0.0)
