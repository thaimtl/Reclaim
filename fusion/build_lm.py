"""Build the joint search's conversational language model (docs/SVR1_SPEC.md section 6): a pruned word 4-gram over
everyday dialogue, in KenLM's binary format. Stages, in order (all output under $RECLAIM_WORKDIR/data/lm/):

    python -m fusion.build_lm download   # raw corpora into data/lm/raw/<source>/ (about 2 GB streamed, 15 min)
    python -m fusion.build_lm corpus     # normalised text, one utterance per line -> data/lm/corpus.txt
    python -m fusion.build_lm build      # 4-gram (KenLM lmplz, singleton 3/4-grams pruned) -> data/lm/conversational.binary
    python -m fusion.build_lm compare    # perplexity of the recording prompts: Gaddy & Klein's book LM vs this one

Sources (free for research; licences in each dataset):
    OpenSubtitles 2018, English (OPUS)   film and TV dialogue; every 3rd line of the first 36M, about 12M lines
    Cornell Movie-Dialogs                film conversations, about 300k lines
    DailyDialog (train)                  everyday two-person chats, about 90k utterances (CC BY-NC-SA 4.0)
    Topical-Chat (train)                 casual chats between people, about 190k messages (CDLA-Sharing 1.0)
Text is normalised exactly like SVR-1's targets (svr.text.clean_text: ASCII, no punctuation, lowercase), so the words
the joint search builds and the LM's words match. Every recording prompt (docs/recording/, sessions 1-3) is removed
on an exact match, since the joint search's weights are fitted on sessions 1-2; and any line that contains a session 3
sentence as whole words is removed too, since session 3 is the test set.
`build` needs KenLM's lmplz and build_binary: KENLM_BIN=<folder with both> (built from github.com/kpu/kenlm); it sorts in
LMPLZ_MEMORY (default 8G) of RAM, with its sort files in LMPLZ_TMP (default: the system temp dir; must be a local disk).
"""
import collections
import glob
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
OUT = os.path.join(WORK, 'data', 'lm')
RAW = os.path.join(OUT, 'raw')
CORPUS = os.path.join(OUT, 'corpus.txt')
LM = os.path.join(OUT, 'conversational.binary')
BOOK_LM = os.path.join(WORK, 'baseline_gaddy', 'checkpoints', 'lm.binary')
PROMPTS = os.path.join(ROOT, 'docs', 'recording')
SOURCES = {
    'opensubtitles': 'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz',
    'cornell': 'https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    'dailydialog': 'https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet',
    'topicalchat': 'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/train.json',
}
SUBTITLE_LINES, SUBTITLE_STRIDE = 36_000_000, 3
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from svr.text import clean_text  # noqa: E402
MAX_WORDS = 40


def download():
    for name, url in SOURCES.items():
        d = os.path.join(RAW, name)
        dest = os.path.join(d, {'opensubtitles': 'sample.txt', 'dailydialog': 'train.parquet'}.get(name, os.path.basename(url)))
        if os.path.exists(dest):
            print(f'{name}: already downloaded')
            continue
        os.makedirs(d, exist_ok=True)
        print(f'{name}: {url}', flush=True)
        part = dest + '.part'  # renamed only when complete, so an interrupted download is retried
        if name == 'opensubtitles':  # stream: keep every 3rd line of the first 36M (spreads over more films)
            with urllib.request.urlopen(url) as r, gzip.open(r, 'rt', encoding='utf-8', errors='replace') as f, \
                    open(part, 'w') as out:
                for i, line in enumerate(f):
                    if i >= SUBTITLE_LINES:
                        break
                    if i % SUBTITLE_STRIDE == 0:
                        out.write(line)
        else:
            urllib.request.urlretrieve(url, part)
        os.replace(part, dest)


def raw_lines():
    """(source, utterance) from every downloaded corpus, unnormalised."""
    with open(os.path.join(RAW, 'opensubtitles', 'sample.txt'), encoding='utf-8', errors='replace') as f:
        for line in f:
            yield 'opensubtitles', line.lstrip('- ')
    with zipfile.ZipFile(glob.glob(os.path.join(RAW, 'cornell', '*.zip'))[0]) as z:
        name = next(n for n in z.namelist() if n.endswith('movie_lines.txt'))
        for line in io.TextIOWrapper(z.open(name), encoding='iso-8859-1'):
            yield 'cornell', line.split(' +++$+++ ')[-1]
    import pyarrow.parquet as pq  # DailyDialog's own site is gone: Hugging Face copy
    for dialogue in pq.read_table(os.path.join(RAW, 'dailydialog', 'train.parquet'), columns=['utterances']).to_pylist():
        for utt in dialogue['utterances']:
            yield 'dailydialog', utt
    with open(glob.glob(os.path.join(RAW, 'topicalchat', '*.json'))[0]) as f:
        for conv in json.load(f).values():
            for turn in conv['content']:
                yield 'topicalchat', turn['message']


def normalise(text):
    return ' '.join(clean_text(text).split())


TEST_PROMPTS = 'session3_test.txt'


def recording_prompts(pattern='*.txt'):
    out = set()
    for path in glob.glob(os.path.join(PROMPTS, pattern)):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    out.add(normalise(line.split('\t', 1)[-1]))
    return out


def corpus():
    excluded = recording_prompts()
    test = sorted(recording_prompts(TEST_PROMPTS), key=len, reverse=True)
    inside_test = re.compile(r'(?:^| )(?:' + '|'.join(re.escape(p) for p in test) + r')(?: |$)')
    stats = collections.Counter()
    removed = contains_test = 0
    with open(CORPUS + '.tmp', 'w') as out:
        for source, text in raw_lines():
            t = normalise(text)
            if not t or len(t.split()) > MAX_WORDS:
                continue
            if t in excluded:
                removed += 1
                continue
            if inside_test.search(t):  # a longer line that holds a whole session 3 sentence
                contains_test += 1
                continue
            out.write(t + '\n')
            stats[source] += len(t.split())
    os.replace(CORPUS + '.tmp', CORPUS)
    print(f'{len(excluded)} recording prompts excluded ({removed} exact matches removed); '
          f'{contains_test} more lines removed for containing one of the {len(test)} session 3 sentences')
    print('words per source:', dict(stats), 'total', sum(stats.values()))


def build():
    kb = os.environ.get('KENLM_BIN')
    if not kb:
        sys.exit('set KENLM_BIN to the folder with lmplz and build_binary (github.com/kpu/kenlm)')
    arpa = os.path.join(OUT, 'conversational.arpa')
    # lmplz memory-maps its sort files: a network share (e.g. a 9p mount) segfaults it
    tmp = os.environ.get('LMPLZ_TMP', tempfile.gettempdir())
    os.makedirs(tmp, exist_ok=True)
    with open(CORPUS) as f_in, open(arpa, 'w') as f_out:
        subprocess.run([os.path.join(kb, 'lmplz'), '-o', '4', '--prune', '0', '0', '1', '1', '-S',
                        os.environ.get('LMPLZ_MEMORY', '8G'), '-T', tmp], stdin=f_in, stdout=f_out, check=True)
    local = os.path.join(tmp, 'conversational.binary')  # build_binary also needs a local disk
    subprocess.run([os.path.join(kb, 'build_binary'), '-q', '8', '-b', '8', 'trie', arpa, local], check=True)
    shutil.copyfile(local, LM + '.tmp')
    os.replace(LM + '.tmp', LM)
    os.remove(local)
    os.remove(arpa)
    print(f'{LM}: {os.path.getsize(LM) / 1e6:.0f} MB')


def compare():
    """Perplexity of the recording prompts (conversational, never in the LM's training text) under each LM."""
    import kenlm
    prompts = sorted(recording_prompts())
    n_words = sum(len(p.split()) + 1 for p in prompts)  # + end of sentence
    for name, path in (('book (Gaddy & Klein)', BOOK_LM), ('conversational', LM)):
        if not os.path.exists(path):
            print(f'{name}: {path} missing')
            continue
        m = kenlm.Model(path)
        log10 = sum(m.score(p) for p in prompts)
        oov = sum(w not in m for p in prompts for w in p.split())
        print(f'{name:22s} perplexity {10 ** (-log10 / n_words):7.1f}   out-of-vocabulary words {oov}   '
              f'size {os.path.getsize(path) / 1e6:.0f} MB   ({len(prompts)} prompts, {n_words} words)')


if __name__ == '__main__':
    stages = {'download': download, 'corpus': corpus, 'build': build, 'compare': compare}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        sys.exit('usage: python -m fusion.build_lm ' + '|'.join(stages))
    os.makedirs(OUT, exist_ok=True)
    stages[sys.argv[1]]()
