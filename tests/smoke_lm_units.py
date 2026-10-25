"""Unit checks for the joint search's language model (fusion.lm), on a tiny hand-written n-gram (CPU, no data).

    venv-svr/bin/python -m tests.smoke_lm_units
"""
import math
import os
import tempfile

from fusion.lm import ConversationLM

ARPA = r"""
\data\
ngram 1=6
ngram 2=4

\1-grams:
-6.0	<unk>	0
-99	<s>	-0.5
-0.8	</s>	0
-0.6	hello	-0.3
-0.7	world	-0.3
-0.9	there	-0.3

\2-grams:
-0.2	<s> hello
-0.3	hello world
-0.4	hello there
-0.2	world </s>

\end\
"""


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tiny.arpa')
        with open(path, 'w') as f:
            f.write(ARPA.lstrip())
        lm = ConversationLM(path)
        ref = lm.model.score('hello world', bos=True, eos=True) * math.log(10)
        check('empty cache = the n-gram alone (chained word scores + end = KenLM sentence score)',
              abs(lm.sentence('hello world') - ref) < 1e-6, f'{lm.sentence("hello world"):.4f} vs {ref:.4f}')
        check('text is normalised like SVR-1 (case, punctuation)',
              abs(lm.sentence('Hello, WORLD!') - lm.sentence('hello world')) < 1e-12)
        s = lm.start()
        before, _ = lm.score(s, 'sarah')
        lm.add_text('Hi, this is Sarah from admissions.')
        after, _ = lm.score(lm.start(), 'sarah')
        check('a name the n-gram never saw becomes likely once it is said', after > before + 1.0,
              f'ln P {before:.2f} -> {after:.2f}')
        _, s1 = lm.score(lm.start(), 'from')
        follow, _ = lm.score(s1, 'admissions')
        check('the cache knows which word followed which', follow > after, f'{follow:.2f} vs {after:.2f}')
        fresh = lm.score(lm.start(), 'admissions')[0]
        lm.add_text(' '.join(['hello world'] * 400))
        faded = lm.score(lm.start(), 'admissions')[0]
        check('older mentions fade', faded < fresh, f'{fresh:.2f} -> {faded:.2f}')
        lm.reset()
        check('reset() forgets the call', abs(lm.sentence('hello world') - ref) < 1e-6)
        try:
            lm.score(lm.start(), 'two words')
            check('score() rejects more than one word', False)
        except ValueError:
            check('score() rejects more than one word', True)
    print('LM UNITS PASS')


if __name__ == '__main__':
    main()
