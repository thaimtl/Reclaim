"""Freeze the candidate sets for the look-alike and near-duplicate benchmarks (Berkeley test set).

    venv-svr/bin/python -m bench.make_candidate_splits

Writes bench/splits/{homophene_10,homophene_50,nearduplicate_10}.json, in the same trial format as the
phrase-ID splits (candidates + answer_index), so bench.score handles them unchanged.
Substitute words come from the Berkeley corpus vocabulary, so every candidate uses words from the same books.
"""
import glob
import json
import os

from bench.common import DATA_ROOT, SPLITS_DIR, load_split
from bench.lookalikes import (Lexicon, corpus_vocabulary, homophene_candidates, near_duplicate_candidates,
                              trials_from, write_json)
from svr.text import clean_text

SEED = 0


def main():
    texts = []
    for path in glob.glob(os.path.join(DATA_ROOT, '*', '*', '*_info.json')):
        with open(path) as f:
            t = json.load(f)['text']
        if t:
            texts.append(t)
    vocab = corpus_vocabulary(texts, clean_text)
    vocab = {w: c for w, c in vocab.items() if (len(w) > 1 or w in ('a', 'i')) and not w.isdigit()}
    lexicon = Lexicon(vocab)
    common = sorted(w for w, c in vocab.items() if c >= 3 and w in lexicon.pron)
    print(f'corpus vocabulary {len(vocab)} words, {len(lexicon.pron)} with pronunciations, '
          f'{sum(len(v) > 1 for v in lexicon.by_viseme.values())} look-alike groups')

    items = load_split('open_vocab_test')['items']
    for n in (10, 50):
        trials = trials_from(items, lambda s, rng: homophene_candidates(s, lexicon, n, rng), n, SEED,
                             f'Each open-vocab test recording with {n - 1} lip look-alike sentences: words swapped for '
                             'words with the same viseme sequence (tier 1), topped up with one-viseme-different words '
                             '(tier 2) when a sentence has too few. Simulates a camera proposing, EMG judging.')
        short = sum(len(t['candidates']) < n for t in trials['trials'])
        tier1 = sum(x == 1 for t in trials['trials'] for x in t['tiers'])
        tier2 = sum(x == 2 for t in trials['trials'] for x in t['tiers'])
        print(f'homophene_{n}: {len(trials["trials"])} trials, {short} with fewer than {n} candidates, '
              f'negatives tier1 {tier1} / tier2 {tier2}')
        write_json(os.path.join(SPLITS_DIR, f'homophene_{n}.json'), trials)

    trials = trials_from(items, lambda s, rng: near_duplicate_candidates(s, common, 10, rng, lexicon), 10, SEED,
                         'Each open-vocab test recording with 9 near-duplicates: 1-2 words replaced by random '
                         'non-look-alike corpus words. Simulates a fixed set of similar rehearsed sentences.')
    write_json(os.path.join(SPLITS_DIR, 'nearduplicate_10.json'), trials)
    print('nearduplicate_10:', len(trials['trials']), 'trials')
    ex = trials_from(items[:1], lambda s, rng: homophene_candidates(s, lexicon, 6, rng), 6, SEED, '')
    print('example:', json.dumps(ex['trials'][0]['candidates'], indent=1))


if __name__ == '__main__':
    main()
