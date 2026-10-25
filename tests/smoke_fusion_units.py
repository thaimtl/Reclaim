"""Unit checks for the fusion process (fusion.search, fusion.lock, fusion.core) on synthetic camera readings and a
tiny hand-written n-gram (CPU; needs the camera's units file and SentencePiece model under RECLAIM_WORKDIR).

    venv-svr/bin/python -m tests.smoke_fusion_units
"""
import itertools
import os
import tempfile

import numpy as np

from fusion.core import Fusion
from fusion.lm import ConversationLM
from fusion.lock import FEATURES, LockModel
from fusion.offline import align
from fusion.search import BLANK, CamReading, Gate, JointSearch, Pieces, ctc_loglik, viterbi_word_ends

ARPA = r"""
\data\
ngram 1=8
ngram 2=5

\1-grams:
-6.0	<unk>	0
-99	<s>	-0.5
-0.8	</s>	0
-0.6	i	-0.3
-0.9	want	-0.3
-0.9	water	-0.3
-1.5	waiter	-0.3
-1.2	what	-0.3

\2-grams:
-0.2	<s> i
-0.3	i want
-0.4	want water
-0.2	water </s>
-1.4	want waiter

\end\
"""


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


class StubLexicon:
    def __init__(self, table):
        self.table = table

    def lookalikes(self, word):
        return self.table.get(word, [])


def reading(pieces, words, seg_id=1, t0=100.0, gap=3, tail=4, noise=None, seed=0):
    """A peaky cam.tick: each piece on one frame (p 0.9), blanks between; noise {word: (alt, p_alt)} puts some of a
    word's mass on a look-alike's first piece."""
    rng = np.random.default_rng(seed)
    seq = []
    for w in words:
        ids = pieces.word_ids(w)
        alt = (noise or {}).get(w)
        for k, i in enumerate(ids):
            seq.append((i, pieces.word_ids(alt[0])[0] if alt and k == 0 else None, alt[1] if alt and k == 0 else 0))
    T = gap + len(seq) * (gap + 1) + tail
    ids = np.zeros((T, 32), np.int32)
    logp = np.zeros((T, 32), np.float32)
    blank = np.zeros(T, np.float32)
    filler = [i for i in range(2, 600) if i not in {s[0] for s in seq}][:40]
    hot = {gap + k * (gap + 1): s for k, s in enumerate(seq)}
    for t in range(T):
        top = {}
        if t in hot:
            tok, alt, pa = hot[t]
            top[tok] = 0.9 - pa
            if alt:
                top[alt] = pa
            pb = 0.05
        else:
            pb = 0.97
        rest = rng.permutation(filler)
        mass = 1.0 - pb - sum(top.values())
        for i in rest[:32 - len(top) - 1]:
            top[int(i)] = mass / 200
        top[BLANK] = pb
        items = sorted(top.items(), key=lambda kv: -kv[1])[:32]
        ids[t] = [i for i, _ in items]
        logp[t] = np.log([p for _, p in items])
        blank[t] = np.log(pb)
    frame_t = t0 + np.arange(T) * 0.04
    return {'seg_id': seg_id, 'seg_start': float(frame_t[0]), 'frame_t': frame_t, 'topk_ids': ids,
            'topk_logp': logp, 'blank_logp': blank, 'activity': np.full(T, 0.6, np.float32), 'face_ok': True}


def brute_ctc(cam, seq):
    """CTC forward over the extended label sequence, straight from the definition."""
    em = cam.emissions(sorted(set(seq)))
    col = {t: j for j, t in enumerate(sorted(set(seq)))}
    lab = [BLANK] + [x for t in seq for x in (t, BLANK)]
    S = len(lab)
    e = lambda t, s: cam.blank[t] if lab[s] == BLANK else em[t, col[lab[s]]]  # noqa: E731
    a = np.full(S, -np.inf)
    a[0], a[1] = e(0, 0), e(0, 1)
    for t in range(1, cam.T):
        b = np.full(S, -np.inf)
        for s in range(S):
            terms = [a[s]] + ([a[s - 1]] if s >= 1 else [])
            if s >= 2 and lab[s] != BLANK and lab[s] != lab[s - 2]:
                terms.append(a[s - 2])
            b[s] = np.logaddexp.reduce(terms) + e(t, s)
        a = b
    return np.logaddexp(a[-1], a[-2])


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tiny.arpa')
        with open(path, 'w') as f:
            f.write(ARPA)
        lm = ConversationLM(path)
        pieces = Pieces()
        search = JointSearch(lm, pieces, StubLexicon({'water': ['waiter'], 'waiter': ['water']}))
        w = {'cam': 1.0, 'emg': 0.0, 'lm': 0.3, 'beta': 1.0, 'gamma': 0.5}

        cam = CamReading(reading(pieces, ['i', 'want', 'water']))
        seqs = [list(itertools.chain(*[pieces.word_ids(x) for x in s.split()])) for s in
                ('i want water', 'i want waiter', 'i want')]
        fast = ctc_loglik(cam, seqs)
        slow = np.array([brute_ctc(cam, s) for s in seqs])
        check('camera CTC log-likelihood matches the definition', np.allclose(fast, slow, atol=1e-3),
              f'{fast.round(2)} vs {slow.round(2)}')
        check('the true reading scores highest under the camera', fast[0] > fast[1] and fast[0] > fast[2])

        beam = search.camera_beam(cam, w, {})
        words, _ = pieces.split_words(beam[0][0])
        check('camera beam reads the sentence', words == ['i', 'want', 'water'], ' '.join(words))

        cands = search.candidates(cam, w, {}, final=True)
        texts = [' '.join(c.words) for c in cands]
        check('look-alike swaps join the candidates', 'i want waiter' in texts, f'{len(cands)} candidates')
        sc = search.judge_scores(cands, cam, None, {}, True)
        check('judge scores: one row per candidate, seven columns', sc.shape == (len(cands), 7))

        # the camera dropped a short word: 'i water' -> inserting 'want' isn't short, but 'i want water' -> 'i want the water'
        drop = CamReading(reading(pieces, ['want', 'water']))
        cands_d = search.candidates(drop, w, {}, final=True)
        ins = [c for c in cands_d if c.source == 'insert']
        check('short words the camera dropped are tried, unseen by the camera',
              ins and all(c.unseen() == 1 for c in ins) and any(c.words == ('i', 'want', 'water') for c in ins),
              str([' '.join(c.words) for c in ins][:5]))
        ph = search.candidates(drop, w, {'i': 0.3}, final=True, phrases=[['i', 'want', 'water']])
        check("Claude's phrase fills in the dropped word", any(c.source == 'phrase' and c.words == ('i', 'want', 'water')
                                                              for c in ph))
        sc_d = search.judge_scores(cands_d, drop, None, {}, True)
        k = next(j for j, c in enumerate(cands_d) if c.words == ('i', 'want', 'water'))
        k0 = next(j for j, c in enumerate(cands_d) if c.words == ('want', 'water'))
        check('an unseen word costs the camera nothing', abs(sc_d[k, 0] - sc_d[k0, 0]) < 1e-6 and sc_d[k, 6] == 1)
        lk2 = LockModel(repeat_floor=0.5)
        check('the final pass speaks a sentence unless it is unreadable; short words never count',
              lk2.settle(['i', 'want', 'water'], [0.1, 0.9, 0.8]) and not lk2.settle(['i', 'want', 'water'],
                                                                                      [0.9, 0.2, 0.3]))

        # EMG settles a look-alike: the camera is unsure between water and waiter, EMG strongly favours waiter
        cam2 = CamReading(reading(pieces, ['i', 'want', 'water'], noise={'water': ('waiter', 0.42)}))
        cands = search.candidates(cam2, w, {}, final=True)
        emg = lambda texts: np.array([0.0 if t.endswith('waiter') else -12.0 for t in texts])  # noqa: E731
        sc = search.judge_scores(cands, cam2, emg, {}, True)
        S0 = search.combine(sc, w)
        S1 = search.combine(sc, dict(w, emg=1.0))
        b0, b1 = cands[int(np.argmax(S0))].words, cands[int(np.argmax(S1))].words
        check('EMG can overturn the camera on a look-alike', b0[-1] == 'water' and b1[-1] == 'waiter', f'{b0} -> {b1}')
        hint = search.combine(search.judge_scores(cands, cam2, None, {'waiter': 30.0}, True), w)
        check("Claude's hints add to a word's score", cands[int(np.argmax(hint))].words[-1] == 'waiter')

        locked = search.candidates(cam, w, {}, locked=['i', 'went'])
        check('every candidate keeps the locked words', all(c.words[:2] == ('i', 'went') for c in locked))

        ends = viterbi_word_ends(cam, [pieces.word_ids(x) for x in ('i', 'want', 'water')])
        check('word end times rise along the sentence', ends is not None and ends[0] < ends[1] < ends[2],
              str(np.round(ends, 2) if ends else None))

        check('align: exact', align(['i', 'want', 'water'], ['i', 'want', 'water']) == [0, 1, 2])
        check('align: a partial reading matches the start', align(['i', 'want'], ['i', 'want', 'water']) == [0, 1])
        check('align: a wrong word matches nothing', align(['i', 'went', 'water'], ['i', 'want', 'water']) == [0, None, 2])
        check('align: a later segment matches the end', align(['water'], ['i', 'want', 'water']) == [2])

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (400, len(FEATURES)))
        y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0.5).astype(int)
        lk = LockModel().fit(X, y)
        check('lock model learns from the joint posterior', lk.theta[0] > 1.0, f'theta_joint {lk.theta[0]:.2f}')
        lk.tau0, lk.kappa, lk.tau_min, lk.tau_max = 0.9, 0.2, 0.8, 0.95
        check('a predictable moment lowers the threshold, within its range',
              lk.threshold(0.0) == 0.9 and abs(lk.threshold(1.0) - 0.8) < 1e-9 and lk.threshold(0.25) < 0.9)
        f = np.zeros(len(FEATURES))
        check('no camera-EMG agreement, no lock', not lk.may_lock(f, 0.99, 0.9))

        # the core, camera only: ticks grow, words lock left to right, the voice gets them after the repair window
        lock = LockModel(tau0=0.5, tau_min=0.5, tau_max=0.5, agree_floor=0.0)
        core = Fusion(search, Gate(), lock, svr=None, use_hints=True, repair_s=0.3)
        full = reading(pieces, ['i', 'want', 'water'])
        out, t = [], 0.0
        T = len(full['frame_t'])
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        topics = [tp for tp, _ in out]
        check('dec.hyp on every pass', topics.count('dec.hyp') == len(range(8, T + 1, 3)))
        out2 = []
        for k in range(1, 40):
            out2 += core.step(t + 0.05 * k)
            if k == 12:
                out2 += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]},
                                  t + 0.05 * k)
        out += out2
        locks = [x['w'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        finals = [m for tp, m in out if tp == 'voice.say' and m['final']]
        check('the sentence locks', locks == ['i', 'want', 'water'], ' '.join(locks))
        check('the voice says the locked words once, in order', said == locks, ' '.join(said))
        check('the sentence ends with one final voice.say', len(finals) == 1)
        from bus.team import decode, encode
        try:
            back = [decode(encode(tp, m)) for tp, m in out]
            check('every message goes through the bus encoder', len(back) == len(out))
        except Exception as e:  # noqa: BLE001
            check('every message goes through the bus encoder', False, f'{type(e).__name__}: {e}')
        lock_t = [x['t'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        check('every lock carries its time, cam and alts', len(lock_t) == 3 and all(
            set(x) >= {'w', 'conf', 'alts', 'cam', 'emg', 't'} for tp, m in out if tp == 'dec.lock' for x in m['words']))

        # a repair inside the repair window changes what the voice says
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        lk_msgs = [m for tp, m in out if tp == 'dec.lock']
        idx = sum(len(m['words']) for m in lk_msgs) - 1
        last = [x for m in lk_msgs for x in m['words']][-1]['w']
        out += core.feed('claude.repair', {'utt_id': 1, 'edits': [{'index': idx, 'from': last, 'to': 'what'}],
                                           'please_repeat': False}, t)
        for k in range(1, 60):
            out += core.step(t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        check('a repair before the voice speaks changes the spoken word', said[idx] == 'what', ' '.join(said))

        # a word below the threshold at the end: Please repeat, and the voice doesn't say it
        lock = LockModel(tau0=0.999, tau_min=0.999, tau_max=0.999, agree_floor=0.0, repeat_floor=0.9999)
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        for k in range(1, 80):
            out += core.step(t + 0.05 * k)
        rep = [m for tp, m in out if tp == 'dec.repeat']
        check('an unclear sentence ends in dec.repeat', len(rep) == 1 and rep[0]['reason']['words'],
              str(rep[0]['reason'] if rep else None))
        check('the final voice.say still closes the sentence',
              any(tp == 'voice.say' and m['final'] for tp, m in out))
    from fusion.run import drain

    class Q:
        def __init__(self, items):
            self.items = list(items)

        def recv(self, timeout_ms=None):
            return self.items.pop(0) if self.items else (None, None)

    tick = lambda seg, k: ('cam.tick', {'seg_id': seg, 'k': k})  # noqa: E731
    got = drain(Q([tick(1, 2), ('emg.raw', {}), tick(1, 3), tick(2, 1)]), tick(1, 1))
    check('drain skips stale ticks of a segment, keeps the rest in order',
          [(t, m.get('k')) for t, m in got] == [('emg.raw', None), ('cam.tick', 3), ('cam.tick', 1)],
          str([(t, m.get('seg_id'), m.get('k')) for t, m in got]))
    print('FUSION UNITS PASS')


if __name__ == '__main__':
    main()
