"""Smoke-test the Claude service with a fake Claude: message shapes, the edit guard rails, the append-only prefix,
and the bus loop end to end over inproc ZeroMQ. --live also makes two real calls (needs ANTHROPIC_API_KEY) and
checks that the second one reads the cache.

    python -m tests.smoke_claude_service [--live]
"""
import argparse
import copy
import json
import threading
import time
from types import SimpleNamespace

import zmq

from bus.team import Pub, Sub
from claude_service.brain import SCHEMA, SYSTEM, Brain
from claude_service.service import TOPICS, run

failures = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


class FakeClient:
    """Stands in for anthropic.Anthropic(); replies with the queued JSON (or, given a function, with what it returns
    for the request) and records every request."""

    def __init__(self, replies):
        self.replies, self.calls = replies if callable(replies) else list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        r = self.replies(kw) if callable(self.replies) else self.replies.pop(0)
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
    check('new utterance untouched', b.words[0]['w'] == 'hi')

    b.next_request(0.6)
    target = {'utt_id': 8, 'words': b.words, 'review': []}
    r, _ = b.ask()
    out, rejected = b.apply(r, target, 0.7)
    check('no repair without review', [t for t, _ in out] == ['claude.predict'] and len(rejected) == 1)

    calls = fake.calls
    check('append-only prefix', all(c['messages'] == n['messages'][:len(c['messages'])] for c, n in zip(calls, calls[1:])))
    check('frozen system, schema, settings', all(c['system'] == SYSTEM and c['output_config']['format']['schema'] == SCHEMA
                                                 and c['output_config']['effort'] == 'low' for c in calls))
    check('model', all(c['model'] == 'claude-opus-5-5' for c in calls))
    check('roles alternate', [m['role'] for m in calls[-1]['messages']] == ['user', 'assistant'] * 2 + ['user'])


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
    check('bus: bad messages reported, loop survives', len(errors) == 2 and all('bad message' in e for e in errors),
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true')
    args = ap.parse_args()
    unit()
    bus()
    if args.live:
        live()
    raise SystemExit(1 if failures else 0)
