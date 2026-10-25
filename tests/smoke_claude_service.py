"""Smoke-test the Claude service with a fake Claude: message shapes, the edit guard rails, the append-only prefix,
and the bus loop end to end over inproc ZeroMQ. --live also makes two real calls (needs ANTHROPIC_API_KEY) and
checks that the second one reads the cache.

    python -m tests.smoke_claude_service [--live]
"""
import argparse
import copy
import json
import os
import threading
import time
from types import SimpleNamespace

import zmq

from bus.team import Pub, Sub
from claude_service import brain as brain_mod
from claude_service.brain import SCHEMA, SYSTEM, Brain
from claude_service.service import TOPICS, run

failures = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


class FakeClient:
    """Stands in for anthropic.Anthropic(); replies with the queued JSON (or, given a function, with what it returns
    for the request; a SimpleNamespace is returned as the response itself) and records every request."""

    def __init__(self, replies):
        self.replies, self.calls = replies if callable(replies) else list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        r = self.replies(kw) if callable(self.replies) else self.replies.pop(0)
        if isinstance(r, SimpleNamespace):
            return r
        text = json.dumps(r)
        usage = SimpleNamespace(input_tokens=10, cache_read_input_tokens=0, output_tokens=5)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=text)], stop_reason='end_turn',
                               model=kw['model'], usage=usage, _request_id='req_fake')


def reply(edits=(), please_repeat=False, words=(('yes', 0.9),), predictability=0.8):
    return {'edits': list(edits), 'please_repeat': please_repeat, 'predict_words': [{'w': w, 'weight': x} for w, x in words],
            'predict_phrases': ['a little'], 'predictability': predictability}


def lock(utt, *words):
    return {'utt_id': utt, 'words': [{'w': w, 'conf': 0.6, 'alts': list(alts), 'cam': 0.5, 'emg': 0.7, 't': 1.0}
                                     for w, alts in words]}


def unit():
    fake = FakeClient([
        reply(words=(('Yes', 1.7), ('', 0.5)), predictability=-1),
        reply(edits=[{'index': 2, 'from': 'nerve', 'to': 'nervous'},   # good
                     {'index': 1, 'from': 'little', 'to': 'lot'},      # "lot" is not an alt
                     {'index': 0, 'from': 'b', 'to': 'a'},             # wrong "from"
                     {'index': 9, 'from': 'x', 'to': 'y'}]),           # not under review
        reply(edits=[{'index': 0, 'from': 'a', 'to': 'the'}]),         # nothing under review
    ])
    b = Brain(fake)
    check('not due when idle', not b.due(0.0, mouthing=False))
    b.partner({'text': 'are you', 'is_final': False, 't': 0.0})
    check('partial judge text ignored', not b.due(0.0, mouthing=False))
    b.partner({'text': 'Are you feeling nervous about the demo?', 'is_final': True, 't': 0.1})
    check('due after a final judge line', b.due(0.1, mouthing=False))

    target = b.next_request(0.1)
    r, log = b.ask()
    out, _ = b.apply(r, target, 0.2)
    check('predict only when nothing is under review', [t for t, _ in out] == ['claude.predict'])
    p = out[0][1]
    check('predict words cleaned and clipped', p['words'] == [{'w': 'yes', 'weight': 1.0}], p['words'])
    check('predictability clipped', p['predictability'] == 0.0)
    check('predict fields', set(p) == {'words', 'phrases', 'predictability', 'ttl_s', 't'})
    check('log has request, reply and delay', {'request', 'reply', 'ms', 'cache_read'} <= set(log))

    b.locked(lock(7, ('a', ['the']), ('little', ['lid']), ('nerve', ['nervous', 'nerf'])))
    check('due after a lock', b.due(0.3, mouthing=False))
    target = b.next_request(0.3)
    check('request lists the review', 'Review:' in b.messages[-1]['content'] and 'nervous' in b.messages[-1]['content'])
    b.locked(lock(8, ('hi', [])))  # Austin moves on while Claude thinks
    r, _ = b.ask()
    out, rejected = b.apply(r, target, 0.5)
    rep = dict(out)['claude.repair']
    check('only the valid edit survives', rep['edits'] == [{'index': 2, 'from': 'nerve', 'to': 'nervous'}], rep['edits'])
    check('three edits rejected', len(rejected) == 3)
    check('repair keeps its utterance', rep['utt_id'] == 7)
    check('repaired word can swap back', target['words'][2]['w'] == 'nervous' and 'nerve' in target['words'][2]['alts'])
    check('new utterance untouched', b.utts[8][0]['w'] == 'hi')

    b.next_request(0.6)
    target = {'utt_id': 8, 'words': b.utts[8], 'review': []}
    r, _ = b.ask()
    out, rejected = b.apply(r, target, 0.7)
    check('no repair without review', [t for t, _ in out] == ['claude.predict'] and len(rejected) == 1)

    calls = fake.calls
    check('append-only prefix', all(c['messages'] == n['messages'][:len(c['messages'])] for c, n in zip(calls, calls[1:])))
    check('frozen system, schema, settings', all(c['system'] == SYSTEM and c['output_config']['format']['schema'] == SCHEMA
                                                 and c['output_config']['effort'] == 'low' for c in calls))
    check('model', all(c['model'] == 'claude-opus-5-5' for c in calls))
    check('roles alternate', [m['role'] for m in calls[-1]['messages']] == ['user', 'assistant'] * 2 + ['user'])


def late_lock():
    """A late dec.lock for the previous utterance lands at its own index; reviews of it still apply."""
    fake = FakeClient([reply(edits=[{'index': 2, 'from': 'vine', 'to': 'fine'}]), reply()])
    b = Brain(fake)
    b.locked(lock(1, ('i', []), ('feel', [])))
    b.locked(lock(2, ('yes', ['jess'])))
    b.locked(lock(1, ('vine', ['fine'])))  # late, after utterance 2 started
    b.locked(lock(2, ('sure', [])))
    check('late lock: indices per utterance', [w['w'] for w in b.utts[1]] == ['i', 'feel', 'vine']
          and [w['w'] for w in b.utts[2]] == ['yes', 'sure'])
    t1 = b.next_request(0.0)
    check('late lock: oldest utterance reviewed first', t1['utt_id'] == 1 and t1['review'] == [0, 1, 2], t1['review'])
    r, _ = b.ask()
    out, rejected = b.apply(r, t1, 0.1)
    rep = dict(out)['claude.repair']
    check('late lock: repair hits the right word', rep['utt_id'] == 1 and not rejected
          and rep['edits'] == [{'index': 2, 'from': 'vine', 'to': 'fine'}], rep)
    check('late lock: other utterance still due', b.due(0.1, mouthing=False))
    t2 = b.next_request(0.1)
    check('late lock: then the newer one', t2['utt_id'] == 2 and t2['review'] == [0, 1], t2)
    b = Brain(FakeClient([]))
    for u in range(brain_mod.MAX_UTTS + 1):
        b.locked(lock(u, ('a', [])))
    check('late lock: bounded', len(b.utts) == brain_mod.MAX_UTTS and 0 not in b.utts)
    b.locked(lock(0, ('late', [])))
    check('late lock: dropped utterance ignored', 0 not in b.utts and max(b.utts) == brain_mod.MAX_UTTS)


def empty_reply():
    """A refusal or an empty reply never leaves an empty assistant turn behind."""
    empty = SimpleNamespace(content=[], stop_reason='refusal', model='claude-opus-5-5', usage=None)
    blank = SimpleNamespace(content=[SimpleNamespace(type='text', text='')], stop_reason='end_turn',
                            model='claude-opus-5-5', usage=None)
    fake = FakeClient([empty, blank, reply()])
    b = Brain(fake)
    for k in range(3):
        b.partner({'text': f'line {k}', 'is_final': True})
        target = b.next_request(10.0 * k)
        r, log = b.ask()
        if r is None:
            b.failed(target, 10.0 * k, refused=log.get('refused', False))
    ok = all(m['content'] for c in fake.calls for m in c['messages'])
    check('empty reply: no empty turn sent', ok and r is not None, [len(c['messages']) for c in fake.calls])
    roles = [m['role'] for m in b.messages]
    check('empty reply: starts on user, no two assistants in a row', roles[0] == 'user'
          and all(not (x == y == 'assistant') for x, y in zip(roles, roles[1:])), roles)
    check('empty reply: refused and empty turns dropped, their judge lines carried into the next request',
          len(b.messages) == 2 and all(f'line {k}' in b.messages[0]['content'] for k in range(3)),
          b.messages[0]['content'])


def bad_words():
    """Malformed dec.lock words keep their slot (indices stay the decoder's) and are never reviewed."""
    b = Brain(FakeClient([reply(edits=[{'index': 3, 'from': 'vine', 'to': 'fine'}])]))
    bad = b.locked({'utt_id': 1, 'words': ['hi', {'conf': 0.5}, {'w': 'a', 'alts': None, 'conf': 'x'},
                                           {'w': 'vine', 'alts': ['fine', 3]}]})
    check('bad words: counted, slots kept', bad == 1 and [w['w'] for w in b.utts[1]] == ['hi', '?', 'a', 'vine'])
    try:
        t = b.next_request(0.0)
        r, _ = b.ask()
        out, rejected = b.apply(r, t, 0.1)
        ok = t['review'] == [0, 2, 3] and dict(out)['claude.repair']['edits'] == [{'index': 3, 'from': 'vine', 'to': 'fine'}]
    except Exception as e:
        ok = f'{type(e).__name__}: {e}'
    check('bad words: request and repair still work', ok is True, ok)

    class Once(Brain):  # next_request breaks once; the loop reports it and goes on
        n = 0

        def next_request(self, now):
            self.n += 1
            if self.n == 1:
                raise RuntimeError('boom')
            return super().next_request(now)

    msgs = [('partner.text', {'text': 'hi', 'is_final': True}), ('dec.lock', {'utt_id': 1, 'words': 'x'})]
    sent, b = [], Once(FakeClient(lambda kw: reply()))
    sub = SimpleNamespace(recv=lambda timeout_ms: msgs.pop(0) if msgs else (None, None))
    pub = SimpleNamespace(send=lambda t, m: sent.append((t, m)))
    end = time.perf_counter() + 1.0
    b.partner({'text': 'again', 'is_final': True})
    run(b, sub, pub, stop=lambda: time.perf_counter() > end or any(t == 'claude.predict' for t, _ in sent))
    errs = [m.get('error', '') for t, m in sent if t == 'timing.claude']
    check('bad words: request error reported, loop goes on', any('request failed' in e for e in errs)
          and any('bad message' in e for e in errs), errs)
    b.partner({'text': 'later', 'is_final': True})
    msgs.append(('dec.lock', {'utt_id': 2, 'words': [{'w': 'ok'}]}))
    end = time.perf_counter() + 1.0
    n0 = len(sent)
    run(b, sub, pub, stop=lambda: time.perf_counter() > end or any(t == 'claude.predict' for t, _ in sent[n0:]))
    check('bad words: later calls still made', any(t == 'claude.predict' for t, _ in sent[n0:]))


def cadence():
    """While mouthing, a call only when the unlocked words moved, and at most every PREDICT_EVERY_S."""
    b = Brain(FakeClient([]))
    b.last_request = 0.0
    check('cadence: no call when nothing changed', not b.due(10.0, mouthing=True))
    b.hyp({'unlocked': ['i', 'am']})
    check('cadence: not before the interval', not b.due(1.0, mouthing=True))
    check('cadence: call after new unlocked words', b.due(2.5, mouthing=True))
    b.next_request(2.5)
    check('cadence: request carries them', 'i am' in b.messages[-1]['content'])
    b.hyp({'unlocked': ['i', 'am']})
    check('cadence: same words again, no call', not b.due(10.0, mouthing=True))


def trim():
    """History stays bounded, starts on a user turn, and is append-only between trims."""
    fake = FakeClient(lambda kw: reply())
    b = Brain(fake)
    for k in range(brain_mod.MAX_MESSAGES):
        b.partner({'text': f'line {k}', 'is_final': True})
        b.next_request(float(k))
        b.ask()
    lens = [len(c['messages']) for c in fake.calls]
    check('trim: bounded', max(lens) <= brain_mod.MAX_MESSAGES and len(b.messages) <= brain_mod.MAX_MESSAGES, max(lens))
    check('trim: starts on a user turn', all(c['messages'][0]['role'] == 'user' for c in fake.calls))
    pairs = list(zip(fake.calls, fake.calls[1:]))
    breaks = [k for k, (c, n) in enumerate(pairs) if c['messages'] != n['messages'][:len(c['messages'])]]
    trims = [k for k, (c, n) in enumerate(pairs) if len(n['messages']) < len(c['messages'])]
    check('trim: append-only between trims', breaks == trims and len(trims) >= 1, f'breaks {breaks}, trims {trims}')


def requirements():
    path = os.path.join(os.path.dirname(__file__), '..', 'claude_service', 'requirements.txt')
    names = {l.split('>')[0].split('=')[0].strip() for l in open(path) if l.strip()}
    check('requirements cover the bus', {'numpy', 'pyzmq', 'msgpack', 'anthropic'} <= names, names)


def bus():
    """The loop end to end over inproc sockets: predict, repair, bad messages survived, and the wire format."""
    ctx = zmq.Context.instance()
    pub_in, out = Pub('inproc://claude-in', ctx), Pub('inproc://claude-out', ctx)
    sub, listen = Sub(['inproc://claude-in'], TOPICS, ctx), Sub(['inproc://claude-out'], [''], ctx)

    def answer(req):  # repair when something is under review, so extra calls from resends do no harm
        if 'Review:' in req['messages'][-1]['content']:
            return reply(edits=[{'index': 0, 'from': 'vine', 'to': 'fine'}])
        return reply()

    fake = FakeClient(answer)
    done = threading.Event()
    th = threading.Thread(target=run, args=(Brain(fake), sub, out, done.is_set))
    th.start()
    got = []  # (topic, msg) as received

    def wait_for(pred, resend=None, deadline_s=5.0):
        end, last = time.perf_counter() + deadline_s, 0.0
        while time.perf_counter() < end:
            if resend and time.perf_counter() - last > 0.2:  # PUB/SUB drops until the subscription arrives
                resend()
                last = time.perf_counter()
            topic, msg = listen.recv(timeout_ms=50)
            if topic is not None:
                got.append((topic, msg))
                if pred(topic, msg):
                    return True
        return False

    try:
        ok1 = wait_for(lambda t, m: t == 'claude.predict', resend=lambda: pub_in.send(
            'partner.text', {'text': 'How are you?', 'is_final': True, 't': 0.0}))
        check('bus: predict after a judge line', ok1, [t for t, _ in got])
        pub_in.sock.send_multipart([b'dec.lock', b'not msgpack'])
        pub_in.send('dec.lock', {'words': []})  # no utt_id
        pub_in.send('dec.lock', {'utt_id': 5, 'words': 'x'})  # words not a list
        pub_in.send('dec.lock', {'utt_id': 6, 'words': [{'alts': None}]})  # word without 'w'
        pub_in.send('dec.lock', lock(1, ('vine', ['fine'])))
        ok2 = wait_for(lambda t, m: t == 'claude.repair' and m['edits'])
        check('bus: repair applied', ok2 and got[-1][1]['edits'] == [{'index': 0, 'from': 'vine', 'to': 'fine'}],
              got[-1])
        time.sleep(0.2)  # let any call from the resends finish
        wait_for(lambda t, m: False, deadline_s=0.3)
    finally:
        done.set()
        th.join()
        for s in (sub, listen, pub_in, out):
            s.close()
    errors = [m['error'] for t, m in got if t == 'timing.claude' and 'error' in m]
    check('bus: bad messages reported, loop survives', len(errors) == 4 and all('bad message' in e for e in errors),
          errors)
    timings = [m for t, m in got if t == 'timing.claude' and 'error' not in m]
    check('bus: one timing per call', len(timings) == len(fake.calls), f'{len(timings)} vs {len(fake.calls)}')
    check('bus: wire v == 2, float t_pub', all(m['v'] == 2 and isinstance(m['t_pub'], float) for _, m in got))
    seqs = {}
    for t, m in got:
        seqs.setdefault(t, []).append(m['seq'])
    check('bus: seq counts up per topic', all(v == list(range(1, len(v) + 1)) for v in seqs.values()), seqs)


def live():
    import anthropic
    b = Brain(anthropic.Anthropic(timeout=30.0))
    b.partner({'text': 'Hi Austin, thanks for calling. Are you feeling nervous about the demo?', 'is_final': True})
    target = b.next_request(0.0)
    r1, log1 = b.ask()
    check('live: first reply parses', r1 is not None, log1.get('error') or log1.get('stop_reason'))
    if r1 is None:
        return
    b.apply(r1, target, 0.0)
    print('  predict:', [w['w'] for w in r1['predict_words']][:10], r1['predictability'], f"{log1['ms']:.0f} ms")
    b.locked(lock(1, ('yes', ['jess']), ('a', []), ('little', []), ('nerve', ['nervous', 'nerf'])))
    target = b.next_request(1.0)
    r2, log2 = b.ask()
    check('live: second reply parses', r2 is not None, log2.get('error'))
    if r2 is None:
        return
    out, rejected = b.apply(r2, target, 1.0)
    print('  repair:', dict(out)['claude.repair'], 'rejected:', rejected, f"{log2['ms']:.0f} ms")
    check('live: second call reads the cache', log2['cache_read'] > 0, f"cache_read={log2['cache_read']}")


def failed_call():
    """Every failed call (error, timeout, cut short, bad JSON, refusal) goes through Brain.failed: no half turn stays in
    the history; after a backoff counted from the failure, the judge lines and the request's own words under review go
    into the next request; after MAX_FAILURES in a row what the failing request carried is dropped (nothing loops or
    grows); a success resets it; a refused request's words are not retried. One edit per word."""
    def boom(kw):
        raise TimeoutError('read timed out')
    cut = SimpleNamespace(content=[SimpleNamespace(type='thinking', thinking='hm', signature='x')],
                          stop_reason='max_tokens', model='claude-opus-5-5', usage=None)
    bad = SimpleNamespace(content=[SimpleNamespace(type='text', text='{not json')], stop_reason='end_turn',
                          model='claude-opus-5-5', usage=None)
    for name, fail in (('error', boom), ('max_tokens', lambda kw: cut), ('bad JSON', lambda kw: bad)):
        answers = [fail, fail, fail, lambda kw: reply()]
        b = Brain(FakeClient(lambda kw: answers.pop(0)(kw)))
        b.partner({'text': 'what did you eat', 'is_final': True})
        b.locked(lock(3, ('a', ()), ('vine', ('fine',))))
        b.locked(lock(4, ('so', ())))                                  # another utterance, not in this request
        target = b.next_request(0.0)
        r, log = b.ask()
        kept = b.failed(target, 20.0)                                  # a timeout: handled 20 s after it was sent
        check(f'{name}: no half turn, content kept', r is None and kept and b.messages == []
              and b.review == {3: [0, 1], 4: [0]} and b.events == ['what did you eat'], (b.messages, b.review))
        check(f'{name}: backoff counts from the failure', not b.due(20.5, False) and b.due(21.0, False))
        b.partner({'text': 'and to drink', 'is_final': True})          # arrives after the failure
        target = b.next_request(21.0)
        check(f'{name}: the retry asks the same, and the new line', 'vine' in str(b.messages[-1])
              and 'what did you eat' in str(b.messages[-1]) and 'and to drink' in str(b.messages[-1]))
        r, log = b.ask()
        check(f'{name}: second failure: longer backoff', r is None and b.failed(target, 22.0)
              and not b.due(23.5, False) and b.due(24.0, False))
        target = b.next_request(24.0)
        r, log = b.ask()
        check(f'{name}: third failure in a row drops what that request carried, not the other utterance',
              not b.failed(target, 25.0) and 3 not in b.review and b.review.get(4) == [0] and b.events == []
              and b.messages == [], (b.review, b.events))
        target = b.next_request(40.0)
        r, log = b.ask()
        b.succeeded()
        check(f'{name}: a success resets the backoff', r is not None and b.failures == 0 and len(b.messages) == 2
              and b.due(40.0, False) is False)
    refuse = SimpleNamespace(content=[], stop_reason='refusal', model='claude-opus-5-5', usage=None)
    b = Brain(FakeClient([refuse]))
    b.partner({'text': 'how are you', 'is_final': True})
    b.locked(lock(6, ('fine', ('vine',))))
    target = b.next_request(0.0)
    r, log = b.ask()
    b.failed(target, 0.0, refused=log.get('refused', False))
    check('refusal: judge line kept, its words not retried', log.get('refused') and b.events == ['how are you']
          and 6 not in b.review and b.messages == [], (b.events, b.review))
    b = Brain(FakeClient([]))
    for k in range(30):
        b.partner({'text': f'line {k}', 'is_final': True})
    target = b.next_request(0.0)
    b.failed(target, 0.0)
    check('pending judge lines are bounded', len(b.events) == brain_mod.MAX_PENDING_LINES and b.events[-1] == 'line 29')
    b = Brain(FakeClient([reply(edits=[{'index': 1, 'from': 'vine', 'to': 'fine'},
                                       {'index': 1, 'from': 'fine', 'to': 'vine'}])]))
    b.locked(lock(5, ('a', ()), ('vine', ('fine',))))
    target = b.next_request(0.0)
    r, log = b.ask()
    out, rejected = b.apply(r, target, 0.0)
    rep = [m for t, m in out if t == 'claude.repair'][0]
    check('one edit per word: the swap back is rejected', rep['edits'] == [{'index': 1, 'from': 'vine', 'to': 'fine'}]
          and len(rejected) == 1, (rep, rejected))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true')
    args = ap.parse_args()
    unit()
    late_lock()
    empty_reply()
    failed_call()
    bad_words()
    cadence()
    trim()
    requirements()
    bus()
    if args.live:
        live()
    raise SystemExit(1 if failures else 0)
