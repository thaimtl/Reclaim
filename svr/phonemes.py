"""Phoneme targets and disputed-word marking for SVR-1.

Phonemes: the 39 ARPAbet phonemes (stress removed) from CMUdict. A sentence with any out-of-dictionary word gets
no phoneme target (its phoneme loss is masked); letter CTC still trains on it.
Verifier tokens: phonemes plus a word-boundary token, so the verifier knows where words start and end.
Disputed words: for a candidate set (e.g. the camera's top-N), a word is disputed when the candidates do not all
agree on it at that position (aligned with difflib), which is exactly where EMG has to decide.
"""
import difflib
import functools
import os

PHONEMES = ('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z '
            'ZH').split()
PHONE_BLANK = len(PHONEMES)          # 39: CTC blank for the phoneme head
N_PHONE_CLASSES = len(PHONEMES) + 1  # 40
WORD_BOUNDARY = len(PHONEMES)        # 39: verifier token between words
TOKEN_PAD = len(PHONEMES) + 1        # 40
N_TOKENS = len(PHONEMES) + 2         # verifier vocabulary
_ID = {p: i for i, p in enumerate(PHONEMES)}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
CMUDICT = os.path.join(WORK, 'data', 'lexicon', 'cmudict.dict')


G2P_SUPPLEMENT = os.path.join(WORK, 'data', 'lexicon', 'g2p_supplement.dict')


@functools.lru_cache(maxsize=1)
def cmudict():
    """word -> phonemes. clean_text strips apostrophes, so contractions are also indexed without them ("dont");
    words CMUdict lacks (Woking, Horsell, ...) come from g2p_supplement.dict (g2p_en, made by svr.make_g2p)."""
    pron = {}
    for path in (CMUDICT, G2P_SUPPLEMENT):
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.split('#')[0].split()
                if not parts or '(' in parts[0]:
                    continue
                word = parts[0].lower()
                phones = [''.join(c for c in p if not c.isdigit()) for p in parts[1:]]
                pron.setdefault(word, phones)
                pron.setdefault(word.replace("'", ''), phones)
    return pron


def word_phonemes(word):
    p = cmudict().get(word)
    return None if p is None else [_ID[x] for x in p if x in _ID]


def sentence_phonemes(clean_sentence):
    """Phoneme ids for a cleaned (lowercase, unpunctuated) sentence, or None if any word is unknown."""
    out = []
    for w in clean_sentence.split():
        p = word_phonemes(w)
        if p is None:
            return None
        out.extend(p)
    return out


def verifier_tokens(clean_sentence, disputed_words=None):
    """(token ids, disputed flags): phonemes per word with a boundary token between words.
    Unknown words fall back to one boundary token so the sentence still has a shape."""
    words = clean_sentence.split()
    disputed_words = disputed_words or [False] * len(words)
    toks, flags = [], []
    for i, (w, d) in enumerate(zip(words, disputed_words)):
        if i:
            toks.append(WORD_BOUNDARY)
            flags.append(0)
        p = word_phonemes(w) or [WORD_BOUNDARY]
        toks.extend(p)
        flags.extend([int(d)] * len(p))
    return toks, flags


def disputed_words(candidates):
    """For each candidate, a bool per word: True where the other candidates disagree at that position."""
    split = [c.split() for c in candidates]
    out = []
    for i, words in enumerate(split):
        flags = [False] * len(words)
        for j, other in enumerate(split):
            if i == j:
                continue
            for tag, a0, a1, _, _ in difflib.SequenceMatcher(a=words, b=other, autojunk=False).get_opcodes():
                if tag != 'equal':
                    for k in range(a0, a1):
                        flags[k] = True
        out.append(flags)
    return out
