"""Turning per-frame character log-probabilities into text or a phrase choice.

greedy_decode      best class per frame, merge repeats, drop blanks
BeamLMDecoder      CTC prefix beam search with a KenLM word language model. A Python port of
                   ctcdecode 1.0.3 (parlance/ctcdecode, derived from PaddlePaddle DeepSpeech),
                   which Gaddy's evaluation uses, including its quirks:
                     * every completed word must be in the LM vocabulary (ctcdecode builds a
                       dictionary FST from it), and an unfinished last word must be a prefix of one;
                     * LM score = alpha * ln P(word | previous words) + beta, added when a space ends a word;
                     * an n-gram containing an out-of-vocabulary word scores -1000 (not converted to ln);
                     * the final ranking ignores the LM score of the unfinished last word.
score_phrases      CTC log-likelihood of each candidate phrase (the 10-phrase classifier)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from svr.text import BLANK, CHARS, NUM_CLASSES, SPACE, clean_text, ids_to_text

NEG_INF = -math.inf
OOV_SCORE = -1000.0
LOG10_E = 0.4342944819


def greedy_decode(log_probs):
    """log_probs: (time, classes) array -> text."""
    best = np.asarray(log_probs).argmax(-1)
    ids = [int(c) for i, c in enumerate(best) if c != BLANK and (i == 0 or c != best[i - 1])]
    return ids_to_text(ids)


def _log_sum_exp(x, y):
    if x == NEG_INF:
        return y
    if y == NEG_INF:
        return x
    m = max(x, y)
    return math.log(math.exp(x - m) + math.exp(y - m)) + m


def read_kenlm_vocabulary(lm_path):
    """Words stored at the end of a KenLM binary, in the order KenLM enumerates them."""
    with open(lm_path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 64 * 1024 * 1024)
        f.seek(size - chunk)
        tail = f.read()
    start = tail.rfind(b'<unk>\0<s>\0</s>\0')
    if start < 0:
        start = tail.rfind(b'<unk>\0')
    assert start >= 0, 'could not find the vocabulary in the KenLM binary'
    return [w.decode('utf-8', errors='replace') for w in tail[start:].split(b'\0') if w]


class _Node:
    __slots__ = ('char', 'parent', 'children', 'b_prev', 'nb_prev', 'b_cur', 'nb_cur',
                 'log_prob_c', 'score', 'exists', 'word')

    def __init__(self, char=-1, parent=None, word=''):
        self.char = char
        self.parent = parent
        self.children = {}
        self.b_prev = self.nb_prev = self.b_cur = self.nb_cur = NEG_INF
        self.log_prob_c = NEG_INF
        self.score = NEG_INF
        self.exists = True
        self.word = word  # characters of the current (unfinished) word

    def remove(self):
        self.exists = False
        if not self.children and self.parent is not None:
            del self.parent.children[self.char]
            if not self.parent.children and not self.parent.exists:
                self.parent.remove()

    def path(self):
        out, node = [], self
        while node.char != -1:
            out.append(node.char)
            node = node.parent
        return out[::-1]


class BeamLMDecoder:
    def __init__(self, lm_path, alpha=1.5, beta=1.85, beam_width=100, cutoff_top_n=40):
        import kenlm
        self.kenlm = kenlm
        self.lm = kenlm.Model(lm_path)
        self.order = self.lm.order
        self.alpha, self.beta = alpha, beta
        self.beam_width, self.cutoff_top_n = beam_width, cutoff_top_n
        char_set = set(CHARS) - {' '}
        self.words = {w for w in read_kenlm_vocabulary(lm_path)
                      if w not in ('<unk>', '<s>', '</s>') and set(w) <= char_set}
        self.word_prefixes = {w[:i] for w in self.words for i in range(1, len(w) + 1)}
        self._lm_cache = {}

    # ----- language model -----
    def _cond_log_prob(self, ngram):
        """ln P(last word | the words before it), or OOV_SCORE."""
        cached = self._lm_cache.get(ngram)
        if cached is not None:
            return cached
        state, out = self.kenlm.State(), self.kenlm.State()
        self.lm.NullContextWrite(state)
        result = 0.0
        for word in ngram:
            if word not in self.lm:
                result = OOV_SCORE
                break
            result = self.lm.BaseScore(state, word, out)
            state, out = out, state
        else:
            result = result / LOG10_E
        self._lm_cache[ngram] = result
        return result

    def _make_ngram(self, node):
        """The last `order` words ending at node, padded with <s> at the sentence start."""
        words = []
        current = node
        for i in range(self.order):
            chars = []
            while current.char != SPACE and current.char != -1:
                chars.append(CHARS[current.char])
                current = current.parent
            words.append(''.join(reversed(chars)))
            if current.char == -1:
                words.extend(['<s>'] * (self.order - i - 1))
                break
            current = current.parent  # skip the space
        return tuple(reversed(words))

    def _lm_bonus(self, node):
        return self._cond_log_prob(self._make_ngram(node)) * self.alpha + self.beta

    # ----- search -----
    def _child(self, node, c, log_prob_c):
        child = node.children.get(c)
        if child is not None:
            if child.log_prob_c < log_prob_c:
                child.log_prob_c = log_prob_c
            if not child.exists:
                child.exists = True
                child.b_prev = child.nb_prev = child.b_cur = child.nb_cur = NEG_INF
            return child
        if c == SPACE:
            if node.word not in self.words:
                return None
            word = ''
        else:
            word = node.word + CHARS[c]
            if word not in self.word_prefixes:
                return None
        child = _Node(c, node, word)
        child.log_prob_c = log_prob_c
        node.children[c] = child
        return child

    @staticmethod
    def _collect(root):
        out, stack = [], [root]
        while stack:
            node = stack.pop()
            if node.exists:
                node.b_prev, node.nb_prev = node.b_cur, node.nb_cur
                node.b_cur = node.nb_cur = NEG_INF
                node.score = _log_sum_exp(node.b_prev, node.nb_prev)
                out.append(node)
            stack.extend(reversed(list(node.children.values())))
        return out

    def decode(self, log_probs):
        """log_probs: (time, 38) natural-log probabilities -> best text."""
        log_probs = np.asarray(log_probs, dtype=np.float64)
        assert log_probs.shape[1] == NUM_CLASSES
        root = _Node()
        root.score = root.b_prev = 0.0
        prefixes = [root]
        sort_key = lambda n: (-n.score, n.char)
        n_cand = min(self.cutoff_top_n, NUM_CLASSES)
        for frame in log_probs:
            num = min(len(prefixes), self.beam_width)
            prefixes[:num] = sorted(prefixes[:num], key=sort_key)
            min_cutoff = prefixes[num - 1].score + frame[BLANK] - max(0.0, self.beta)
            full_beam = num == self.beam_width
            if n_cand < NUM_CLASSES:
                cands = sorted(range(NUM_CLASSES), key=lambda i: -frame[i])[:n_cand]
            else:
                cands = range(NUM_CLASSES)
            for c in cands:
                lp = float(frame[c])
                for prefix in prefixes[:self.beam_width]:
                    if full_beam and lp + prefix.score < min_cutoff:
                        break
                    if c == BLANK:
                        prefix.b_cur = _log_sum_exp(prefix.b_cur, lp + prefix.score)
                        continue
                    if c == prefix.char:
                        prefix.nb_cur = _log_sum_exp(prefix.nb_cur, lp + prefix.nb_prev)
                    new = self._child(prefix, c, lp)
                    if new is None:
                        continue
                    log_p = NEG_INF
                    if c == prefix.char and prefix.b_prev > NEG_INF:
                        log_p = lp + prefix.b_prev
                    elif c != prefix.char:
                        log_p = lp + prefix.score
                    if c == SPACE:
                        log_p += self._lm_bonus(prefix)
                    new.nb_cur = _log_sum_exp(new.nb_cur, log_p)
            prefixes = self._collect(root)
            if len(prefixes) >= self.beam_width:
                prefixes.sort(key=sort_key)
                for node in prefixes[self.beam_width:]:
                    node.remove()
                prefixes = prefixes[:self.beam_width]
        best = min(prefixes[:self.beam_width], key=sort_key)
        return ids_to_text(best.path())


def score_phrases(log_probs, phrases):
    """CTC log-likelihood (natural log) of each phrase given (time, 38) log-probs.

    Higher is better. A phrase that cannot fit in the available frames scores -inf.
    """
    lp = torch.as_tensor(np.asarray(log_probs), dtype=torch.float64)
    targets = [torch.tensor([CHARS.index(ch) for ch in clean_text(p)], dtype=torch.long) for p in phrases]
    n = len(phrases)
    lp = lp[:, None, :].expand(-1, n, -1)
    lengths = torch.full((n,), lp.shape[0], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    nll = F.ctc_loss(lp, torch.cat(targets), lengths, target_lengths, blank=BLANK,
                     reduction='none', zero_infinity=False)
    return (-nll).tolist()
