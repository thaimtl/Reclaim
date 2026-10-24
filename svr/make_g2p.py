"""Pronunciations for words CMUdict lacks, with g2p_en (a neural grapheme-to-phoneme model), written once to
data/lexicon/g2p_supplement.dict in CMUdict format so training needs no g2p dependency.

    <env with g2p_en>/bin/python -m svr.make_g2p [extra text files, one sentence per line]

Covers every Berkeley sentence and, if present, our wearer's prompts (recordings/*/manifest.jsonl).
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
sys.path.insert(0, ROOT)

from svr.phonemes import G2P_SUPPLEMENT, PHONEMES, cmudict  # noqa: E402
from svr.text import clean_text  # noqa: E402


def main():
    texts = []
    for p in glob.glob(os.path.join(WORK, 'silent_speech', 'emg_data', '*', '*', '*_info.json')):
        with open(p) as f:
            texts.append(json.load(f)['text'] or '')
    for p in glob.glob(os.path.join(WORK, 'recordings', '*', 'manifest.jsonl')):
        texts += [json.loads(line).get('prompt') or '' for line in open(p) if line.strip()]
    for p in sys.argv[1:]:
        texts += open(p).read().splitlines()
    known = cmudict()
    oov = sorted({w for t in texts for w in clean_text(t).split() if w not in known and not w.isdigit()})
    print(len(oov), 'words without a pronunciation', flush=True)
    from g2p_en import G2p
    g2p = G2p()
    valid = set(PHONEMES)
    existing = open(G2P_SUPPLEMENT).read().splitlines() if os.path.exists(G2P_SUPPLEMENT) else []
    with open(G2P_SUPPLEMENT, 'w') as f:
        for line in existing:
            f.write(line + '\n')
        for w in oov:
            ph = [p for p in g2p(w) if ''.join(c for c in p if not c.isdigit()) in valid]
            if ph:
                f.write(f"{w} {' '.join(ph)}\n")
    print('wrote', G2P_SUPPLEMENT)


if __name__ == '__main__':
    main()
