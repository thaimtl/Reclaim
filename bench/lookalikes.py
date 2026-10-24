"""Lip look-alike (homophene) machinery: pronunciations, visemes, and candidate-sentence generators.

Two words are lip look-alikes when their pronunciations map to the same viseme sequence: they look the same
on the lips (bat / mat / pat). A camera cannot tell them apart; EMG should.

Viseme classes (consonants after Jeffers & Barley; vowels coarsely by lip shape):
    p b m | f v | th dh | t d n l | s z | sh zh ch jh | k g ng hh | r | w | y
    spread/front vowels | open vowels | rounded vowels
"""
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
CMUDICT = os.path.join(WORK, 'data', 'lexicon', 'cmudict.dict')

VISEME = {}
for group, phones in {
    'P': 'P B M', 'F': 'F V', 'TH': 'TH DH', 'T': 'T D N L', 'S': 'S Z', 'SH': 'SH ZH CH JH',
    'K': 'K G NG HH', 'R': 'R', 'W': 'W', 'Y': 'Y',
    'E': 'IY IH EY EH AE AY', 'A': 'AH AA AW', 'O': 'AO OW OY UH UW ER',
}.items():
    for ph in phones.split():
        VISEME[ph] = group


def load_cmudict(path=CMUDICT):
    """word -> first pronunciation as a list of ARPAbet phonemes without stress digits."""
    pron = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split('#')[0].split()
            if not parts:
                continue
            word = parts[0].lower()
            if '(' in word:  # alternate pronunciation
                continue
            pron[word] = [''.join(c for c in p if not c.isdigit()) for p in parts[1:]]
    return pron


def viseme_key(phones):
    return tuple(VISEME.get(p, p) for p in phones)


class Lexicon:
    """Pronunciations restricted to a vocabulary, indexed by viseme sequence."""

    def __init__(self, vocabulary, pron=None):
        pron = pron or load_cmudict()
        self.pron = {w: pron[w] for w in vocabulary if w in pron}
        self.by_viseme = {}
        for w, p in self.pron.items():
            self.by_viseme.setdefault(viseme_key(p), []).append(w)
        for k in self.by_viseme:
            self.by_viseme[k].sort()

    def lookalikes(self, word):
        """Other vocabulary words with exactly the same viseme sequence."""
        if word not in self.pron:
            return []
        return [w for w in self.by_viseme[viseme_key(self.pron[word])] if w != word]

    def near_lookalikes(self, word):
        """Words whose viseme sequence differs from this word's by one substituted viseme class."""
        if word not in self.pron:
            return []
        key = viseme_key(self.pron[word])
        out = []
        for other_key, words in self.by_viseme.items():
            if len(other_key) == len(key) and sum(a != b for a, b in zip(key, other_key)) == 1:
                out.extend(words)
        return out


def corpus_vocabulary(texts, clean):
    vocab = {}
    for t in texts:
        for w in clean(t).split():
            vocab[w] = vocab.get(w, 0) + 1
    return vocab


def homophene_candidates(sentence, lexicon, n, rng, max_swaps=3):
    """n-1 sentences that look like `sentence` on the lips, plus the sentence itself.

    Tier 1 swaps exact look-alikes (same viseme sequence) at 1..max_swaps positions; if a sentence has too few
    swappable words, tier 2 adds near look-alikes (one viseme class different). Returns (candidates, tiers).
    """
    words = sentence.split()
    exact = {i: lexicon.lookalikes(w) for i, w in enumerate(words)}
    near = {i: lexicon.near_lookalikes(w) for i, w in enumerate(words)}
    seen, out, tiers = {sentence}, [], []
    for tier, options in ((1, exact), (2, {i: exact[i] + near[i] for i in exact})):
        slots = [i for i, alts in options.items() if alts]
        attempts = 0
        while slots and len(out) < n - 1 and attempts < 400:
            attempts += 1
            k = rng.randint(1, min(max_swaps, len(slots)))
            new = list(words)
            for i in rng.sample(slots, k):
                new[i] = rng.choice(options[i])
            cand = ' '.join(new)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
                tiers.append(tier)
    return out, tiers


def near_duplicate_candidates(sentence, vocab_words, n, rng, lexicon=None):
    """n-1 distractors that differ from `sentence` by 1-2 random (non-look-alike) word substitutions."""
    words = sentence.split()
    seen, out = {sentence}, []
    attempts = 0
    while len(out) < n - 1 and attempts < 400:
        attempts += 1
        new = list(words)
        for i in rng.sample(range(len(words)), min(len(words), rng.randint(1, 2))):
            choice = rng.choice(vocab_words)
            if lexicon is not None and choice in lexicon.lookalikes(words[i]):
                continue
            new[i] = choice
        cand = ' '.join(new)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def trials_from(items, make, n, seed, description):
    rng = random.Random(seed)
    trials = []
    for it in items:
        made = make(it['reference'], rng)
        negatives, extra = (made if isinstance(made, tuple) else (made, None))
        candidates = negatives + [it['reference']]
        order = list(range(len(candidates)))
        rng.shuffle(order)
        candidates = [candidates[i] for i in order]
        trial = {'id': it['id'], 'candidates': candidates, 'answer_index': candidates.index(it['reference'])}
        if extra is not None:
            tiers = extra + [0]
            trial['tiers'] = [tiers[i] for i in order]  # 0 = true sentence, 1 = exact look-alike, 2 = near
        trials.append(trial)
    return {'description': description, 'n_candidates': n, 'seed': seed, 'trials': trials}


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
